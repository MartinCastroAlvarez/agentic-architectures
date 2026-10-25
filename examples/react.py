"""ReAct: thought -> tool -> observation, until an answer or the step limit."""

from agentic_architectures.patterns import react
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    show(react.run(runtime(), SPEED, max_steps=5))


if __name__ == "__main__":
    main()
