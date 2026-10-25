"""Prompt chaining with gates: a fixed sequence of LLM steps, checked in code between steps.

Each step's output is the next step's input. Between steps a *gate* (a plain Python
check, not a model) decides whether the output is good enough to continue; a failing
gate retries the step once with the reason, then stops the chain with
``gate_failed:<step>`` instead of passing bad work downstream.

When to use
    The task decomposes cleanly into fixed subtasks known in advance (outline, draft,
    polish; extract, validate, format; translate, check). You trade latency for
    accuracy by making each call easier.

Trade-offs
    Simple, testable, and each step can have its own prompt and model. But the path is
    rigid: a task that needs a different decomposition gets the same steps, and latency
    is the sum of every step.

Cost profile
    One LLM call per step (3 by default), + 1 per gate retry. Critical path = steps.

Failure modes
    Errors compound down the chain if gates are weak; gates that are too strict stop
    good work; later steps can "fix" earlier content into something wrong, so keep the
    final gate strict.

Sources
    Anthropic, "Building effective agents" (2024), "Workflow: prompt chaining". Google
    ADK's ``SequentialAgent`` (see adk-sequential-pipeline).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "chaining"
Gate = Callable[[str], str | None]
"""A gate returns ``None`` when the output passes, or the reason it fails."""


def at_least_bullets(n: int) -> Gate:
    def gate(text: str) -> str | None:
        bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "*", "1", "2", "3"))]
        return None if len(bullets) >= n else f"needs at least {n} bullet points"

    return gate


def max_chars(n: int) -> Gate:
    def gate(text: str) -> str | None:
        if not text.strip():
            return "empty output"
        return None if len(text) <= n else f"longer than {n} characters"

    return gate


def no_placeholders(text: str) -> str | None:
    lowered = text.lower()
    for marker in ("todo", "tbd", "lorem", "[insert"):
        if marker in lowered:
            return f"contains a placeholder ({marker})"
    return None


def all_of(*gates: Gate) -> Gate:
    def gate(text: str) -> str | None:
        for g in gates:
            reason = g(text)
            if reason:
                return reason
        return None

    return gate


@dataclass(frozen=True)
class Step:
    name: str
    instructions: str
    gate: Gate | None = None


DEFAULT_STEPS = (
    Step(
        "outline",
        "Write an outline of the answer as 2-4 short bullet points ('- ...'). No prose.",
        at_least_bullets(2),
    ),
    Step("draft", "Write the answer from this outline, in at most five sentences.", max_chars(1500)),
    Step(
        "polish",
        "Tighten this draft: keep every fact, cut filler, at most three sentences.",
        all_of(max_chars(700), no_placeholders),
    ),
)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    steps: tuple[Step, ...] = DEFAULT_STEPS,
    max_retries: int = 1,
) -> tuple[str, str, dict[str, Any]]:
    current = ""
    outputs: dict[str, str] = {}
    for step in steps:
        prompt = f"Task: {task}"
        if current:
            prompt += "\n\nPrevious step output:\n" + untrusted(f"step:{len(outputs)}", current)
        reason: str | None = None
        for attempt in range(max_retries + 1):
            rt.budget.step()
            retry = f"\n\nYour last output failed a check: {reason}. Fix that." if reason else ""
            output = rt.llm(f"chain:{step.name}", step.instructions, prompt + retry).strip()
            reason = step.gate(output) if step.gate else None
            rt.note("gate", step.name, f"attempt {attempt + 1}: {reason or 'pass'}")
            if not reason:
                break
        if reason:
            return current, f"gate_failed:{step.name}", {"outputs": outputs, "reason": reason}
        current = outputs[step.name] = output
        partial.offer(current)
    return current, "completed", {"outputs": outputs}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
