"""Multi-agent debate with a judge: several agents answer, read each other, revise; a judge decides.

Round 1: ``debaters`` agents answer independently (in parallel). Each later round,
every debater sees the other debaters' previous answers and arguments and may revise
or defend. If all debaters agree after a round, the debate stops early. A judge call
then picks the winning answer; code validates the judge's pick and falls back to the
majority answer if the pick is invalid.

When to use
    Questions where independent reasoning paths catch each other's mistakes: maths and
    logic, factual claims, evaluations where one model's bias should be checked.

Trade-offs
    Often more accurate than one sample or plain voting, because agents can be
    persuaded by a correct argument, but costs ``debaters x rounds + 1`` calls and adds
    rounds of latency. Debaters from the same model can converge on the same wrong
    answer (use different models or personas).

Cost profile
    n x r + 1 calls (3 x 2 + 1 = 7), fewer with early consensus. Critical path r + 1.

Failure modes
    Conformity (the majority bullies a correct minority), a judge swayed by length or
    position, and debates that never converge (bounded by ``rounds``).

Sources
    Du et al., "Improving Factuality and Reasoning in Language Models through
    Multiagent Debate" (2023). Liang et al., "Encouraging Divergent Thinking in Large
    Language Models through Multi-Agent Debate" (2023). AutoGen multi-agent debate
    examples (Wu et al., 2023).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_architectures.core.parsing import as_int
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern
from agentic_architectures.patterns.parallelization import normalise

NAME = "debate"
DEBATE = (
    "You are debater {i} of {n}. Answer the task. If other debaters' answers are shown, "
    "check their reasoning: change your answer only if they convinced you. Reply with JSON: "
    '{{"answer": "<short answer>", "argument": "<one or two sentences>"}}'
)
JUDGE = (
    "You are the judge. Pick the best-supported answer. Reply with JSON: "
    '{"winner": <debater number>, "answer": "<final answer>", "reason": "..."}'
)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    debaters: int = 3,
    rounds: int = 2,
) -> tuple[str, str, dict[str, Any]]:
    if not 2 <= debaters <= 7 or not 1 <= rounds <= 5:
        raise ValueError("debaters must be 2-7 and rounds 1-5")
    positions: list[tuple[str, str]] = []
    history: list[list[str]] = []
    for r in range(1, rounds + 1):
        rt.budget.step()
        previous = list(positions)

        def speak(i: int, previous: list[tuple[str, str]] = previous) -> tuple[str, str]:
            prompt = f"Task: {task}"
            others = [f"Debater {j + 1}: {a}\nArgument: {g}" for j, (a, g) in enumerate(previous) if j != i]
            if others:
                prompt += "\n\nOther debaters (previous round):\n" + untrusted("peers", "\n".join(others))
            data = rt.llm_json(f"debate:{i + 1}", DEBATE.format(i=i + 1, n=debaters), prompt)
            return as_str(data.get("answer"), limit=500).strip(), as_str(data.get("argument"), limit=800)

        positions = rt.parallel(speak, range(debaters))
        history.append([a for a, _ in positions])
        distinct = {normalise(a) for a, _ in positions}
        rt.note("debate", f"round {r}", " | ".join(a for a, _ in positions))
        if len(distinct) == 1 and r < rounds:
            rt.note("debate", "consensus", f"after round {r}")
            break
    counts = Counter(normalise(a) for a, _ in positions if a)
    majority = next((a for a, _ in positions if counts and normalise(a) == counts.most_common(1)[0][0]), "")
    partial.offer(majority)
    transcript = "\n".join(f"Debater {i + 1}: {a}\nArgument: {g}" for i, (a, g) in enumerate(positions))
    verdict = rt.llm_json(
        "judge", JUDGE, f"Task: {task}\n\nFinal positions:\n{untrusted('debate', transcript)}"
    )
    winner = as_int(verdict.get("winner"), 0, low=-1, high=1000)
    if 1 <= winner <= debaters:
        answer = as_str(verdict.get("answer")).strip() or positions[winner - 1][0]
        status = "judged"
    else:
        rt.note("guard", "judge", "invalid pick: using the majority answer")
        answer, status = majority, "majority_fallback"
    partial.offer(answer)
    return answer, status, {"rounds": history, "winner": winner, "consensus": len(counts) == 1}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
