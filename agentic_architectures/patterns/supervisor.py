"""Supervisor / hierarchical multi-agent: a manager picks which specialist acts next.

A supervisor call reads the task and the transcript of what the workers have reported
and picks the next worker (or ``FINISH``). Each worker is a small agent with its own
instructions and only its own tools. A worker can itself be a **team** with its own
supervisor, which makes the system hierarchical: the top supervisor delegates
"research" to the research team, whose supervisor delegates to a searcher and an
analyst. Code validates every choice (unknown names are refused) and ``max_turns``
bounds each level.

When to use
    Several distinct skills or tool sets, where the order of work depends on results:
    research + analysis + writing, triage + specialists, a manager over teams.

Trade-offs
    Clear separation of roles and least-privilege tools per worker, but every turn costs
    a supervisor call on top of the worker's, and the supervisor is a bottleneck that
    sees everything (context grows). Hierarchy contains that growth per team at the
    price of more calls.

Cost profile
    Per turn: 1 supervisor + the worker's calls. Flat, 2 workers: about 5 calls;
    hierarchical with one team: about 8.

Failure modes
    Ping-pong between two workers (bounded by ``max_turns``), the supervisor finishing
    too early or never, a worker's output instructing the supervisor (delimited as
    untrusted), and lost context between levels (the team returns one summary).

Sources
    Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent
    Conversation" (2023). LangGraph multi-agent supervisor and hierarchical teams
    tutorials (2024). CrewAI hierarchical process (see crewai-hierarchical-planning).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern

NAME = "supervisor"
FINISH = "FINISH"
SUPERVISE = (
    "You are the supervisor of: {menu}\n"
    "Pick who acts next and what they should do, or finish when the task is solved. Reply "
    'with JSON: {{"next": "<worker>|FINISH", "instruction": "...", "answer": "<final answer if FINISH>"}}'
)


@dataclass(frozen=True)
class Worker:
    name: str
    role: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Team:
    """A worker that is itself a supervisor over workers."""

    name: str
    role: str
    members: tuple[Worker | Team, ...] = field(default_factory=tuple)


Member = Worker | Team

RESEARCHER = Worker("researcher", "searches the local documents and reads files", ("search", "read_file"))
ANALYST = Worker("analyst", "computes numbers with the calculator", ("calculator",))
WRITER = Worker("writer", "writes the final answer from the findings")
FLAT = (RESEARCHER, ANALYST, WRITER)
HIERARCHICAL = (Team("research_team", "finds facts and computes figures", (RESEARCHER, ANALYST)), WRITER)


def _supervise(
    rt: Runtime, task: str, members: tuple[Member, ...], label: str, max_turns: int, partial: Partial
) -> tuple[str, str, list[str]]:
    by_name = {m.name: m for m in members}
    menu = "; ".join(f"{m.name} ({m.role})" for m in members)
    reports: list[tuple[str, str]] = []
    turns: list[str] = []
    last = ""
    for _ in range(max_turns):
        rt.budget.step()
        transcript = "\n".join(untrusted(f"worker:{n}", r) for n, r in reports) or "(no reports yet)"
        decision = rt.llm_json(label, SUPERVISE.format(menu=menu), f"Task: {task}\n\nReports:\n{transcript}")
        choice = as_str(decision.get("next"), limit=64).strip()
        if choice.upper() == FINISH:
            answer = as_str(decision.get("answer")).strip() or last
            rt.note("route", label, f"FINISH after {len(turns)} turns")
            partial.offer(answer)
            return answer, "finished", turns
        member = by_name.get(choice)
        if member is None:
            rt.note("guard", label, f"unknown worker {choice!r}")
            reports.append(
                ("supervisor", f"error: {choice!r} is not a worker; choose from {sorted(by_name)}")
            )
            continue
        instruction = as_str(decision.get("instruction"), task, limit=1000)
        rt.note("route", label, f"-> {member.name}: {instruction}")
        turns.append(member.name)
        if isinstance(member, Team):
            output, _, sub = _supervise(
                rt,
                f"{task}\nTeam assignment: {instruction}",
                member.members,
                f"supervisor:{member.name}",
                max_turns,
                partial,
            )
            turns.extend(f"{member.name}/{s}" for s in sub)
        else:
            output, _, _ = act(
                rt,
                f"worker:{member.name}",
                f"You are the {member.name}: you {member.role}. Do only your part.",
                f"Task: {task}\nYour instruction: {instruction}\nReports so far:\n{transcript}",
                tools=rt.tools.subset(member.tools),
                max_steps=3,
            )
        last = output or last
        reports.append((member.name, output or "(no output)"))
    return last, "max_turns", turns


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    members: tuple[Member, ...] = FLAT,
    max_turns: int = 5,
) -> tuple[str, str, dict[str, Any]]:
    answer, status, turns = _supervise(rt, task, members, "supervisor", max_turns, partial)
    return answer, status, {"turns": turns}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
