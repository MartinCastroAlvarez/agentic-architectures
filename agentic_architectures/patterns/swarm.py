"""Swarm / handoffs: peer agents pass the conversation to each other; no central manager.

One agent is active at a time. Each agent has instructions, its own tools and a list of
agents it may hand off to. It either answers, calls a tool, or hands off with
``{"handoff": "<agent>", "note": "..."}``; the shared conversation moves to the new agent
with the note. Code enforces the handoff graph (an agent can only hand off to the
agents listed for it) and ``max_handoffs``.

When to use
    Customer-service style flows: a triage agent hands off to billing, refunds, tech;
    each specialist can hand back or onwards. Decentralised and cheap per turn.

Trade-offs
    No supervisor call per turn (cheaper than ``supervisor``), and each agent's prompt
    stays small. But control is distributed: nobody owns the whole task, cycles are
    possible, and debugging needs the trace of who held the conversation when.

Cost profile
    1 call per agent turn (+ tool rounds). Triage + one specialist: 2-3 calls.

Failure modes
    Handoff loops (A → B → A: capped by ``max_handoffs``), handoffs to agents that do
    not exist or are not allowed (refused, the agent is told), a specialist answering
    out of its scope, and context lost in the note (the full transcript is passed).

Sources
    OpenAI, "Swarm" (2024) and the OpenAI Agents SDK handoffs. LangGraph swarm
    (``langgraph-swarm``). Anthropic, "Building effective agents" (2024) on routing
    as the deterministic cousin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import assistant
from agentic_architectures.core.model import system
from agentic_architectures.core.model import user
from agentic_architectures.core.parsing import DATA_RULE
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import ACT_PROTOCOL
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import parse_action
from agentic_architectures.core.runtime import run_pattern

NAME = "swarm"
HANDOFF = '- to hand the conversation to another agent: {"handoff": "<agent>", "note": "..."}'


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    tools: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()


DEFAULT_AGENTS = (
    Agent(
        "triage",
        "You are triage. Do not answer yourself: hand off maths to 'math', questions about the "
        "documents to 'research', and writing or explanations to 'writer'.",
        handoffs=("math", "research", "writer"),
    ),
    Agent(
        "math", "You solve quantitative questions with the calculator.", ("calculator",), ("writer", "triage")
    ),
    Agent(
        "research",
        "You answer from the local documents and cite [doc-id].",
        ("search", "read_file"),
        ("writer", "triage"),
    ),
    Agent("writer", "You write short, clear explanations.", (), ("triage",)),
)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    agents: tuple[Agent, ...] = DEFAULT_AGENTS,
    start: str = "triage",
    max_handoffs: int = 4,
    max_turns: int = 10,
) -> tuple[str, str, dict[str, Any]]:
    by_name = {a.name: a for a in agents}
    active = by_name[start]
    path = [active.name]
    conversation: list[Message] = [user(f"Task: {task}")]
    handoffs = 0
    for _ in range(max_turns):
        rt.budget.step()
        tools = rt.tools.subset(active.tools)
        prompt = (
            f"You are agent '{active.name}'. {active.instructions}\n\nTools:\n{tools.catalog()}\n"
            f"You may hand off to: {', '.join(active.handoffs) or 'nobody'}\n\n"
            f"{ACT_PROTOCOL}\n{HANDOFF}\n{DATA_RULE}"
        )
        text = rt.chat(f"swarm:{active.name}", [system(prompt), *conversation], json_mode=True)
        action = parse_action(text)
        target = as_str(action.extra.get("handoff"), limit=64).strip()
        if target:
            conversation.append(assistant(text))
            if target not in active.handoffs or target not in by_name:
                rt.note("guard", "handoff", f"{active.name} -> {target!r} refused")
                conversation.append(
                    user(f"Handoff to {target!r} is not allowed. Allowed: {list(active.handoffs)}")
                )
                continue
            handoffs += 1
            if handoffs > max_handoffs:
                rt.note("guard", "handoff", f"max_handoffs={max_handoffs} reached")
                return partial.answer, "max_handoffs", {"path": path, "handoffs": handoffs - 1}
            note = as_str(action.extra.get("note"), limit=500)
            rt.note("handoff", f"{active.name}->{target}", note)
            active = by_name[target]
            path.append(active.name)
            conversation.append(user(f"Handoff note from the previous agent:\n{untrusted('handoff', note)}"))
            continue
        if action.is_tool:
            conversation.append(assistant(text))
            observation = rt.call_tool(action.tool, action.args, tools=tools)
            conversation.append(user(f"Observation:\n{observation}"))
            continue
        partial.offer(action.answer)
        return action.answer, "answered", {"path": path, "handoffs": handoffs}
    return partial.answer, "max_turns", {"path": path, "handoffs": handoffs}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
