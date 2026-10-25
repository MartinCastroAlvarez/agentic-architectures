"""Evaluator-optimizer: generate, score, revise with feedback until the score clears the bar."""

from agentic_architectures.patterns import evaluator_optimizer
from examples._common import IDEMPOTENCY
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(evaluator_optimizer.run(runtime(), IDEMPOTENCY, threshold=8))
    print("scores by round:", [r["score"] for r in result.data["rounds"]])


if __name__ == "__main__":
    main()
