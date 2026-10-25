"""Evaluator-optimizer (generator-critic): one model drafts, another scores and gives feedback, repeat.

The generator writes a draft; the evaluator returns a 0-10 score and concrete
feedback; the generator revises with the feedback. The loop ends when the score
reaches ``threshold`` or after ``max_rounds``. **Code decides acceptance**: the score is
clamped and compared in Python, a ``"pass": true`` the model writes is ignored, and the
best draft (not the last) is returned.

When to use
    There are clear evaluation criteria and iterating measurably helps: translation,
    writing to a spec, code against a review checklist, search that needs another pass.

Trade-offs
    Better output for 2x+ the calls. An LLM critic shares the generator's blind spots;
    prefer an external evaluator (tests, a schema, a linter) when one exists, which is
    the step from this pattern to Reflexion.

Cost profile
    2 calls per round (generate + evaluate), 1-3 rounds. Critical path 2 x rounds.

Failure modes
    A lenient critic (passes everything) or a harsh one (never converges: capped by
    ``max_rounds``); oscillation between two drafts; the generator gaming the critic's
    wording rather than improving.

Sources
    Anthropic, "Building effective agents" (2024), "Workflow: evaluator-optimizer".
    Madaan et al., "Self-Refine" (2023). See adk-sequential-pipeline (``LoopAgent``
    writer↔critic) and langsmith-agent-evals (LLM-as-judge).
"""

from __future__ import annotations

from typing import Any

from agentic_architectures.core.parsing import as_int
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "evaluator_optimizer"
GENERATE = "Write a short, accurate answer to the task. If feedback is given, address every point."
EVALUATE = (
    "You are a strict reviewer. Score the draft 0-10 for correctness, completeness and "
    'brevity. Reply with JSON: {"score": <0-10>, "feedback": "<what to fix>"}'
)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_rounds: int = 3,
    threshold: int = 8,
) -> tuple[str, str, dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    best = ("", -1)
    feedback = ""
    for i in range(max_rounds):
        rt.budget.step()
        prompt = f"Task: {task}"
        if rounds:
            prompt += "\n\nPrevious draft:\n" + untrusted("draft", rounds[-1]["draft"])
            prompt += "\nFeedback:\n" + untrusted("critic", feedback)
        draft = rt.llm("generate", GENERATE, prompt).strip()
        verdict = rt.llm_json("evaluate", EVALUATE, f"Task: {task}\n\nDraft:\n{untrusted('draft', draft)}")
        score = as_int(verdict.get("score"), 0, low=0, high=10)
        feedback = as_str(verdict.get("feedback"), "no feedback", limit=1500)
        rounds.append({"round": i + 1, "draft": draft, "score": score})
        if score > best[1]:
            best = (draft, score)
            partial.offer(draft)
        rt.note("gate", "critic", f"round {i + 1}: score {score} (threshold {threshold})")
        if score >= threshold:
            return draft, "accepted", {"rounds": rounds}
    return best[0], "max_rounds", {"rounds": rounds}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
