"""Routing: one classification call, then a specialist with only the tools it needs."""

from agentic_architectures.patterns import routing
from examples._common import IDEMPOTENCY
from examples._common import PORT
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    for task in (SPEED, PORT, IDEMPOTENCY):
        result = show(routing.run(runtime(), task))
        print("route:", result.data["route"])


if __name__ == "__main__":
    main()
