"""ReWOO: plan every tool call up front with placeholders, run them without the model, solve once.

The planner writes the whole tool plan in one call, with later steps referring to
earlier evidence as ``#E1``, ``#E2``... The worker runs the tools **without calling the
model**, substituting each placeholder with that step's output. Steps whose inputs are
ready run in parallel, level by level (the LLMCompiler idea). The solver reads the plan
and the evidence and answers. Exactly two LLM calls, whatever the number of tools.

When to use
    Tool-heavy tasks whose steps can be foreseen: multi-hop lookups, "fetch A and B,
    then compute", report assembly. Big token savings over ReAct, which re-sends the
    growing history at every step.

Trade-offs
    Cheap and fast, but blind: the plan cannot react to a surprising observation. If a
    tool fails, the solver gets the error as evidence; there is no replan (add one, or
    fall back to ReAct). Placeholders insert raw evidence into later tool inputs, so
    tools must validate their input (they do: the calculator refuses non-arithmetic).

Cost profile
    2 LLM calls (planner + solver), n tool calls. Critical path 2.

Failure modes
    References to steps that do not exist or come later (rejected before anything
    runs); a plan that needs information only known after a tool runs; evidence
    too long for the solver (capped).

Sources
    Xu et al., "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented
    Language Models" (2023). Kim et al., "An LLM Compiler for Parallel Function Calling"
    (LLMCompiler, ICML 2024). See langgraph-plan-and-execute (ReWOO and
    LLMCompiler-lite in LangGraph).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "rewoo"
REF = re.compile(r"#E(\d+)")
MAX_EVIDENCE_IN_ARG = 500
PLAN = (
    "Plan every tool call needed to solve the task, before seeing any result. Tools:\n{tools}\n"
    "Refer to the output of an earlier step as #E1, #E2, ... inside later arguments.\n"
    'Reply with JSON: {{"plan": [{{"id": "E1", "why": "...", "tool": "<name>", "args": {{...}}}}, ...]}}'
)
SOLVE = "Answer the task using the plan and the evidence. Be brief; say so if evidence is missing."


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class PlanStep:
    number: int
    why: str
    tool: str
    args: dict[str, str]

    @property
    def deps(self) -> set[int]:
        return {int(n) for v in self.args.values() for n in REF.findall(v)}


def parse_plan(data: dict[str, Any], max_steps: int) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for i, item in enumerate(as_list(data.get("plan"), limit=max_steps), start=1):
        if not isinstance(item, dict):
            raise PlanError(f"step {i} is not an object")
        args = item.get("args")
        step = PlanStep(
            i,
            as_str(item.get("why"), limit=300),
            as_str(item.get("tool"), limit=64),
            {str(k): as_str(v, limit=1000) for k, v in (args.items() if isinstance(args, dict) else [])},
        )
        bad = {d for d in step.deps if not 1 <= d < i}
        if bad:
            raise PlanError(f"step E{i} refers to {sorted(bad)}: only earlier steps may be referenced")
        steps.append(step)
    return steps


def levels(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Group steps so each level depends only on earlier levels (run a level in parallel)."""
    depth: dict[int, int] = {}
    for s in steps:
        depth[s.number] = 1 + max((depth[d] for d in s.deps), default=-1)
    grouped: dict[int, list[PlanStep]] = {}
    for s in steps:
        grouped.setdefault(depth[s.number], []).append(s)
    return [grouped[k] for k in sorted(grouped)]


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_steps: int = 6,
    parallel: bool = True,
) -> tuple[str, str, dict[str, Any]]:
    data = rt.llm_json("rewoo:planner", PLAN.format(tools=rt.tools.catalog()), f"Task: {task}")
    try:
        plan = parse_plan(data, max_steps)
    except PlanError as exc:
        rt.note("guard", "plan", str(exc))
        return "", "invalid_plan", {"error": str(exc)}
    evidence: dict[int, str] = {}

    def execute(step: PlanStep) -> tuple[int, str]:
        def sub(match: re.Match[str]) -> str:
            return evidence.get(int(match.group(1)), "")[:MAX_EVIDENCE_IN_ARG]

        args = {k: REF.sub(sub, v) for k, v in step.args.items()}
        rt.budget.charge_tool()
        _, output = rt.tools.run(step.tool, args)
        rt.note("tool", step.tool or "?", f"E{step.number} {args} -> {output}")
        return step.number, output

    batches = levels(plan) if parallel else [[s] for s in plan]
    for batch in batches:
        for number, output in rt.parallel(execute, batch) if len(batch) > 1 else map(execute, batch):
            evidence[number] = output
    rt.note("plan", "worker", f"{len(plan)} tool calls in {len(batches)} level(s), no LLM calls")
    lines = [
        f"#E{s.number} = {s.tool}({s.args}) because {s.why}\n"
        + untrusted(f"evidence:E{s.number}", evidence.get(s.number, ""))
        for s in plan
    ]
    answer = rt.llm(
        "rewoo:solver",
        SOLVE,
        f"Task: {task}\n\nPlan and evidence:\n" + ("\n".join(lines) or "(no tools used)"),
    ).strip()
    partial.offer(answer)
    return answer, "answered", {"plan": [f"E{s.number}:{s.tool}" for s in plan], "levels": len(batches)}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
