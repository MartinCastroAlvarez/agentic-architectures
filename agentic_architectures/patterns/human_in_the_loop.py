"""Human-in-the-loop: pause before a side effect, ask a person, resume from a checkpoint.

The agent is a ReAct loop, except that a call to a tool marked ``requires_approval``
is **not run**: the loop stops and returns status ``interrupted`` with a JSON-safe
``checkpoint`` (the conversation and the pending call). Later, possibly in another
process, ``resume(rt, checkpoint, Decision(...))`` continues: *approve* runs the call,
*edit* runs it with arguments the person changed (only declared parameters), *reject*
tells the model it was refused (with the person's note) and lets it continue. An
``approve`` callback can stand in for the person (a policy, a CLI prompt).

When to use
    Irreversible or costly actions (sending, paying, deleting, writing), low-confidence
    decisions, regulated workflows, and any agent that acts for someone else.

Trade-offs
    Safety and accountability for latency (a person is slow) and engineering: state
    must be serialisable and resumable, and approvals need an interface. Ask for
    approval rarely and with a clear summary, or people approve without reading.

Cost profile
    The same as ReAct; the pause costs no LLM calls. Resuming re-sends the stored
    conversation (no work is repeated).

Failure modes
    Approval fatigue; a checkpoint tampered with between pause and resume (resume
    re-validates the tool, its approval flag and the arguments; sign checkpoints if they
    cross a trust boundary); a model that rephrases a rejected action and tries again
    (every attempt pauses again).

Sources
    Anthropic, "Building effective agents" (2024) on human checkpoints. LangGraph
    ``interrupt()`` / ``Command(resume=...)`` (2024). CrewAI ``@human_feedback`` (see
    crewai-hierarchical-planning).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Literal

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import assistant
from agentic_architectures.core.model import system
from agentic_architectures.core.model import user
from agentic_architectures.core.parsing import DATA_RULE
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import ACT_PROTOCOL
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import parse_action
from agentic_architectures.core.runtime import run_pattern

NAME = "human_in_the_loop"
INSTRUCTIONS = (
    "Solve the task with the tools. When asked to save or send something, use the tool for "
    "it: a person approves side effects before they happen."
)
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class Decision:
    action: Literal["approve", "reject", "edit"]
    args: dict[str, str] | None = None
    note: str = ""


Approver = Callable[[str, dict[str, Any]], Decision]


class CheckpointError(ValueError):
    pass


def _loop(
    rt: Runtime,
    messages: list[Message],
    steps: int,
    max_steps: int,
    approve: Approver | None,
    partial: Partial,
    log: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    while steps < max_steps:
        steps += 1
        rt.budget.step()
        text = rt.chat("hitl", messages, json_mode=True)
        action = parse_action(text)
        if not action.is_tool:
            partial.offer(action.answer)
            return action.answer, "answered", {"approvals": log, "steps": steps}
        messages.append(assistant(text))
        tool = rt.tools.get(action.tool)
        if tool is not None and tool.requires_approval:
            pending = {"tool": action.tool, "args": action.args}
            if approve is None:
                rt.note("interrupt", action.tool, json.dumps(action.args)[:120])
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "messages": [m.to_dict() for m in messages],
                    "pending": pending,
                    "steps": steps,
                    "max_steps": max_steps,
                    "approvals": log,
                }
                return partial.answer, "interrupted", {"checkpoint": checkpoint, "pending": pending}
            decision = approve(action.tool, dict(action.args))
            observation = _apply(rt, pending, decision, log)
        else:
            observation = rt.call_tool(action.tool, action.args)
        messages.append(user(f"Observation:\n{observation}"))
    return partial.answer, "max_steps", {"approvals": log, "steps": steps}


def _apply(rt: Runtime, pending: dict[str, Any], decision: Decision, log: list[dict[str, Any]]) -> str:
    name = str(pending["tool"])
    log.append({"tool": name, "decision": decision.action, "note": decision.note[:300]})
    rt.note("approval", name, f"{decision.action} {decision.note}".strip())
    if decision.action == "reject":
        return untrusted(f"tool:{name}", f"rejected by a human: {decision.note or 'no reason given'}")
    args = pending["args"]
    if decision.action == "edit":
        tool = rt.tools.get(name)
        allowed = set(tool.parameters) if tool else set()
        edits = decision.args or {}
        if set(edits) - allowed:
            raise CheckpointError(f"edit may only change {sorted(allowed)}")
        args = {**args, **edits}
    return rt.call_tool(name, args)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    approve: Approver | None = None,
    max_steps: int = 6,
) -> tuple[str, str, dict[str, Any]]:
    messages = [
        system(f"{INSTRUCTIONS}\n\nTools:\n{rt.tools.catalog()}\n\n{ACT_PROTOCOL}\n{DATA_RULE}"),
        user(f"Task: {task}"),
    ]
    return _loop(rt, messages, 0, max_steps, approve, partial, [])


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)


def _resume(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    checkpoint: dict[str, Any],
    decision: Decision,
    approve: Approver | None = None,
) -> tuple[str, str, dict[str, Any]]:
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError("unknown checkpoint version")
    pending = checkpoint.get("pending")
    if not isinstance(pending, dict) or not isinstance(pending.get("args"), dict):
        raise CheckpointError("checkpoint has no pending call")
    tool = rt.tools.get(str(pending.get("tool")))
    if tool is None or not tool.requires_approval:
        raise CheckpointError("the pending call is not an approval-gated tool")
    raw = checkpoint.get("messages")
    if not isinstance(raw, list) or not raw:
        raise CheckpointError("checkpoint has no messages")
    messages: list[Message] = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}:
            raise CheckpointError("bad message in checkpoint")
        messages.append(Message(m["role"], str(m.get("content", ""))))
    if messages[0].role != "system":
        raise CheckpointError("checkpoint must start with the system prompt")
    log = list(checkpoint.get("approvals") or [])
    observation = _apply(rt, pending, decision, log)
    messages.append(user(f"Observation:\n{observation}"))
    steps = int(checkpoint.get("steps", 0))
    max_steps = min(int(checkpoint.get("max_steps", 6)), 20)
    return _loop(rt, messages, steps, max_steps, approve, partial, log)


def resume(rt: Runtime, checkpoint: dict[str, Any], decision: Decision, **options: Any) -> Result:
    """Continue an interrupted run with a person's decision."""
    return run_pattern(NAME, _resume, rt, "", checkpoint=checkpoint, decision=decision, **options)
