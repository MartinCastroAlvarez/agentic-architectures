"""Reflection / Reflexion: try, get judged, write a lesson, try again with the lessons.

An actor (a small tool-using agent) attempts the task. An **evaluator** judges the
attempt: preferably external code (unit tests, a schema, an expected value), otherwise
a self-evaluation call. On failure, a reflection call turns the feedback into a short
verbal lesson, stored in an episodic memory that every later attempt (and, if you keep
the memory object, every later run of the same task) sees. Basic *reflection* is the
same loop with self-critique as the evaluator.

When to use
    Tasks with a reliable success signal where a failed attempt teaches something
    general: coding against tests, puzzles, tool-use tasks with checkable results.

Trade-offs
    Learns within a run (and across runs) without fine-tuning, but each failed attempt
    costs about two extra calls, and with only self-evaluation the agent can convince
    itself a wrong answer is right. The evaluator matters more than the reflection.

Cost profile
    Per attempt: actor (1-3) + evaluator (0 if code, else 1) + reflection (1 on
    failure). Two attempts with a code evaluator: about 4-6 calls.

Failure modes
    Vague lessons ("be more careful"); lessons that overfit one attempt; an evaluator
    that can be satisfied without solving the task (reward hacking); poisoned memory
    (lessons are model output: capped, de-duplicated, delimited as untrusted).

Sources
    Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"
    (NeurIPS 2023). Madaan et al., "Self-Refine" (2023). See langgraph-reflection-agent
    (basic reflection, Reflexion with sandboxed unit tests, LATS-lite).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern

NAME = "reflexion"
Evaluator = Callable[[str], tuple[bool, str]]
"""``evaluate(answer) -> (passed, feedback)``: external, trusted code."""

ACTOR = "Solve the task. Use tools for facts and arithmetic. Apply every lesson from earlier attempts."
SELF_EVAL = (
    "Judge whether the answer fully and correctly solves the task. Reply with JSON: "
    '{"pass": true|false, "feedback": "..."}'
)
REFLECT = (
    "The attempt below failed. Write ONE short, general lesson (one sentence) that would "
    "make the next attempt succeed. Do not repeat the answer."
)
MAX_LESSONS = 8
MAX_LESSON_CHARS = 300


def contains(expected: str) -> Evaluator:
    """An external evaluator: passes when ``expected`` appears in the answer."""

    def evaluate(answer: str) -> tuple[bool, str]:
        ok = expected.lower() in answer.lower()
        return ok, "correct" if ok else "the answer is wrong or missing the key figure"

    return evaluate


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    evaluator: Evaluator | None = None,
    lessons: list[str] | None = None,
    max_trials: int = 3,
    actor_steps: int = 3,
) -> tuple[str, str, dict[str, Any]]:
    memory = lessons if lessons is not None else []
    recalled = len(memory)
    trials: list[dict[str, Any]] = []
    for trial in range(1, max_trials + 1):
        rt.budget.step()
        prompt = f"Task: {task}"
        if memory:
            prompt += "\n\nLessons from earlier attempts:\n" + untrusted("lessons", "\n".join(memory))
        answer, _, _ = act(rt, "actor", ACTOR, prompt, max_steps=actor_steps)
        if evaluator is not None:
            passed, feedback = evaluator(answer)
            rt.note("evaluate", "external", f"trial {trial}: {'pass' if passed else 'fail'}")
        else:
            verdict = rt.llm_json(
                "reflexion:evaluate", SELF_EVAL, f"Task: {task}\nAnswer:\n{untrusted('answer', answer)}"
            )
            passed = verdict.get("pass") is True
            feedback = as_str(verdict.get("feedback"), limit=1000)
        trials.append({"trial": trial, "answer": answer, "passed": passed})
        partial.offer(answer)
        if passed:
            return answer, "passed", {"trials": trials, "lessons": list(memory), "recalled": recalled}
        if trial == max_trials:
            break
        lesson = rt.llm(
            "reflect",
            REFLECT,
            f"Task: {task}\nAttempt:\n{untrusted('attempt', answer)}\n"
            f"Feedback:\n{untrusted('feedback', feedback)}",
        ).strip()[:MAX_LESSON_CHARS]
        if lesson and lesson not in memory:
            memory.append(lesson)
            del memory[:-MAX_LESSONS]
        rt.note("memory", "lesson", lesson)
    return (
        trials[-1]["answer"] if trials else "",
        "max_trials",
        {"trials": trials, "lessons": list(memory), "recalled": recalled},
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
