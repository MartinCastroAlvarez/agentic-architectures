"""Debate: three debaters, two rounds, a judge."""

from agentic_architectures.patterns import debate
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(debate.run(runtime(), SPEED, debaters=3, rounds=2))
    for i, answers in enumerate(result.data["rounds"], start=1):
        print(f"round {i}: {answers}")


if __name__ == "__main__":
    main()
