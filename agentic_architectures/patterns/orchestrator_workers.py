"""Orchestrator-workers: an LLM decides the subtasks at run time, workers do them, a synthesiser merges.

Unlike sectioning, the subtasks are not known in advance: the orchestrator reads the
task and writes a list of subtasks (capped at ``max_subtasks``; code drops empty or
duplicate ones). Each worker is a small tool-using agent with its own step limit; the
workers run in parallel; a final call synthesises their results.

When to use
    Complex tasks whose decomposition depends on the input: changes across several
    files, research questions with an unknown number of angles, multi-source reports.

Trade-offs
    Flexible and parallel, but the plan is only as good as the orchestrator's first
    guess (there is no replanning; see ``plan_execute``), and costs scale with the
    number of subtasks the model chooses, so cap it.

Cost profile
    1 orchestrator + Σ worker calls (1-3 each with tools) + 1 synthesiser.
    Critical path ≈ 2 + the slowest worker.

Failure modes
    Over-decomposition (ten subtasks for a one-line question); overlapping workers
    duplicating tool calls; a worker failing silently (its status is kept and shown to
    the synthesiser); the synthesiser trusting worker text (delimited as untrusted).

Sources
    Anthropic, "Building effective agents" (2024), "Workflow: orchestrator-workers".
    See langgraph-parallel-agents (dynamic ``Send`` fan-out to researcher subgraphs).
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

NAME = "orchestrator_workers"
ORCHESTRATE = (
    "Break the task into the fewest independent subtasks needed (1-{n}). Reply with JSON: "
    '{{"subtasks": [{{"id": "short-id", "instruction": "..."}}, ...]}}'
)
WORKER = "You are a worker. Do only your subtask, using tools when needed, and report the result."
SYNTHESIZE = "Combine the workers' results into one short, complete answer to the task."


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_subtasks: int = 4,
    worker_steps: int = 3,
) -> tuple[str, str, dict[str, Any]]:
    plan = rt.llm_json("orchestrator", ORCHESTRATE.format(n=max_subtasks), f"Task: {task}")
    subtasks: list[tuple[str, str]] = []
    for item in as_list(plan.get("subtasks"), limit=max_subtasks):
        if not isinstance(item, dict):
            continue
        sid = as_str(item.get("id"), f"s{len(subtasks) + 1}", limit=40)
        instruction = as_str(item.get("instruction"), limit=1000).strip()
        if instruction and instruction not in (s[1] for s in subtasks):
            subtasks.append((sid, instruction))
    if not subtasks:
        subtasks = [("whole", task)]
        rt.note("guard", "orchestrator", "no usable subtasks: one worker gets the whole task")
    rt.note("fan-out", "workers", ", ".join(s for s, _ in subtasks))

    def work(sub: tuple[str, str]) -> tuple[str, str, str]:
        sid, instruction = sub
        answer, status, _ = act(
            rt, "worker", WORKER, f"Overall task: {task}\nYour subtask: {instruction}", max_steps=worker_steps
        )
        return sid, status, answer

    results = rt.parallel(work, subtasks)
    rt.note("fan-in", "workers", ", ".join(f"{s}={st}" for s, st, _ in results))
    blocks = "\n".join(
        untrusted(f"worker:{sid}:{status}", text or "(no result)") for sid, status, text in results
    )
    answer = rt.llm("synthesize", SYNTHESIZE, f"Task: {task}\n\nResults:\n{blocks}").strip()
    partial.offer(answer)
    return (
        answer,
        "completed",
        {"subtasks": [s for s, _ in subtasks], "workers": {s: st for s, st, _ in results}},
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
