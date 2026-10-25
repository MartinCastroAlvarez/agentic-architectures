"""Plan-and-execute: a planner writes the steps, an executor runs them one at a time, a replanner adjusts.

The planner returns an explicit list of steps (validated and capped). The executor is
a small ReAct agent that runs **one** step with the results so far. After each step the
replanner either finishes with an answer or returns the remaining steps (edited,
reordered, or with recovery steps inserted). ``max_steps`` executed steps and
``max_replans`` replans bound the run.

When to use
    Multi-step tasks where an explicit plan helps: long research, data pipelines,
    anything a person should be able to read (and approve) before it runs.

Trade-offs
    The plan makes the agent legible and keeps the executor's prompt small (it sees one
    step, not the whole history), and a cheaper model can execute. Replanning after
    every step costs an extra call per step; skip it for plans that rarely change.

Cost profile
    1 planner + per step (executor 1-3 + replanner 1). For 2 steps: about 5-7 calls.

Failure modes
    Plans that are too coarse ("research the topic") or too long; the replanner
    rubber-stamping a failing plan; the executor drifting beyond its step. All are
    bounded by the caps and visible in the trace.

Sources
    Wang et al., "Plan-and-Solve Prompting" (ACL 2023). LangChain, "Plan-and-Execute
    Agents" (2023). BabyAGI (Nakajima, 2023). See langgraph-plan-and-execute and
    crewai-hierarchical-planning (CrewAI ``planning=True``).
"""

from __future__ import annotations

from typing import Any

from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern

NAME = "plan_execute"
PLAN = (
    "Write a short plan (1-{n} steps) to solve the task with the available tools:\n{tools}\n"
    'Reply with JSON: {{"steps": ["...", ...]}}'
)
EXECUTE = "Carry out ONLY the current step of the plan. Use a tool if the step needs one."
REPLAN = (
    "Given the task, the plan and the results so far, either finish or update the plan. "
    'Reply with JSON: {"answer": "<final answer>"} when the task is solved, otherwise '
    '{"steps": ["<remaining steps>", ...]}.'
)


def _steps(value: Any, cap: int) -> list[str]:
    return [s for s in (as_str(x, limit=500).strip() for x in as_list(value, limit=cap)) if s]


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_steps: int = 6,
    max_replans: int = 3,
    executor_steps: int = 3,
) -> tuple[str, str, dict[str, Any]]:
    plan = _steps(
        rt.llm_json("planner", PLAN.format(n=max_steps, tools=rt.tools.catalog()), f"Task: {task}").get(
            "steps"
        ),
        max_steps,
    ) or [task]
    rt.note("plan", "planner", " | ".join(plan))
    done: list[tuple[str, str]] = []
    replans = 0
    while plan and len(done) < max_steps:
        step = plan.pop(0)
        history = "\n".join(untrusted(f"step:{i + 1}", f"{s}\n-> {r}") for i, (s, r) in enumerate(done))
        result, status, _ = act(
            rt,
            "executor",
            EXECUTE,
            f"Task: {task}\nCompleted steps:\n{history or '(none)'}\nCurrent step: {step}",
            max_steps=executor_steps,
        )
        done.append((step, result or f"(step ended: {status})"))
        partial.offer(result)
        history = "\n".join(untrusted(f"step:{i + 1}", f"{s}\n-> {r}") for i, (s, r) in enumerate(done))
        decision = rt.llm_json(
            "replanner",
            REPLAN,
            f"Task: {task}\nRemaining plan: {plan}\nResults so far:\n{history}",
        )
        answer = as_str(decision.get("answer")).strip()
        if answer:
            partial.offer(answer)
            return answer, "answered", {"plan": [s for s, _ in done], "replans": replans}
        new_plan = _steps(decision.get("steps"), max_steps)
        if new_plan and new_plan != plan:
            if replans >= max_replans:
                rt.note("guard", "replanner", "max_replans reached: keeping the current plan")
            else:
                replans += 1
                plan = new_plan
                rt.note("plan", "replanner", " | ".join(plan))
    status = "plan_exhausted" if not plan else "max_steps"
    return partial.answer, status, {"plan": [s for s, _ in done], "replans": replans}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
