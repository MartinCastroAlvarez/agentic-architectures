"""Parallelization: sectioning (split, run at once, aggregate) and voting (ask n times, majority).

*Sectioning* runs independent aspects of a task (facts, risks, an example...) as
parallel LLM calls on a bounded thread pool, then one aggregator call merges them.
*Voting* asks the same question ``voters`` times in parallel and keeps the majority
answer (self-consistency); code counts the votes, and a run without a quorum says so.

When to use
    Sectioning: independent subtasks, or guardrails run beside the main call.
    Voting: answers that can be compared (labels, numbers, yes/no) where a single
    sample is unreliable, e.g. flagging content or picking a classification.

Trade-offs
    More calls (and tokens) for lower latency (sectioning) or higher reliability
    (voting). Wall time is the slowest branch, not the sum. The aggregator is a
    bottleneck and a single point of failure; votes need a normaliser to compare.

Cost profile
    Sectioning: n + 1 calls, critical path 2. Voting: n calls, critical path 1.

Failure modes
    Branches that are not really independent (duplicated or contradictory sections);
    correlated votes (same model, same prompt, temperature 0: raise temperature or vary
    prompts); rate limits when n grows (bounded by ``max_workers``).

Sources
    Anthropic, "Building effective agents" (2024), "Workflow: parallelization". Wang
    et al., "Self-Consistency Improves Chain of Thought Reasoning" (2022). See
    langgraph-parallel-agents (``Send`` fan-out / fan-in).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import extract_json
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "parallelization"
DEFAULT_SECTIONS = {
    "facts": "State the key facts needed to answer.",
    "caveats": "List the main caveats or risks, briefly.",
    "example": "Give one short concrete example.",
}


def normalise(answer: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", answer.lower()).strip()


def _sectioning(
    rt: Runtime, task: str, partial: Partial, sections: dict[str, str]
) -> tuple[str, str, dict[str, Any]]:
    def one(item: tuple[str, str]) -> tuple[str, str]:
        name, instructions = item
        return name, rt.llm(f"section:{name}", instructions, f"Task: {task}").strip()

    parts = rt.parallel(one, sections.items())
    rt.note("fan-in", "sections", f"{len(parts)} sections")
    merged = "\n".join(untrusted(f"section:{name}", text) for name, text in parts)
    answer = rt.llm(
        "aggregate",
        "Merge these sections into one short answer (at most four sentences).",
        f"Task: {task}\n\nSections:\n{merged}",
    ).strip()
    partial.offer(answer)
    return answer, "completed", {"sections": dict(parts)}


def _voting(
    rt: Runtime, task: str, partial: Partial, voters: int, quorum: float
) -> tuple[str, str, dict[str, Any]]:
    def vote(i: int) -> str:
        text = rt.llm(
            "vote",
            'Answer with JSON: {"answer": "<short answer>"} (a word, number or short phrase).',
            f"Task: {task}",
            json_mode=True,
        )
        data = extract_json(text)
        return as_str(data.get("answer") if data else text, limit=200).strip()

    votes = rt.parallel(vote, range(voters))
    counts = Counter(normalise(v) for v in votes if v)
    if not counts:
        return "", "no_votes", {"votes": votes}
    top, n = counts.most_common(1)[0]
    winner = next(v for v in votes if normalise(v) == top)
    rt.note("vote", "tally", f"{dict(counts)} -> {winner!r} ({n}/{voters})")
    partial.offer(winner)
    status = "majority" if n / voters > quorum else "no_majority"
    return winner, status, {"votes": votes, "counts": dict(counts), "agreement": round(n / voters, 3)}


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    mode: str = "sectioning",
    sections: dict[str, str] | None = None,
    voters: int = 3,
    quorum: float = 0.5,
) -> tuple[str, str, dict[str, Any]]:
    if mode == "sectioning":
        return _sectioning(rt, task, partial, sections or DEFAULT_SECTIONS)
    if mode == "voting":
        if not 1 <= voters <= 15:
            raise ValueError("voters must be between 1 and 15")
        return _voting(rt, task, partial, voters, quorum)
    raise ValueError("mode must be 'sectioning' or 'voting'")


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
