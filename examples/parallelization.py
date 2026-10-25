"""Parallelization: sectioning (fan-out + aggregate) and voting (majority of n samples)."""

from agentic_architectures.patterns import parallelization
from examples._common import IDEMPOTENCY
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    show(parallelization.run(runtime(), IDEMPOTENCY, mode="sectioning"))
    voted = show(parallelization.run(runtime(), SPEED, mode="voting", voters=5))
    print("votes:", voted.data["counts"])


if __name__ == "__main__":
    main()
