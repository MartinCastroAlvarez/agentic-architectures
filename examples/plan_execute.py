"""Plan-and-execute: plan, run one step at a time, replan or finish after each step."""

from agentic_architectures.patterns import plan_execute
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(plan_execute.run(runtime(), SPEED))
    print("executed steps:", result.data["plan"])


if __name__ == "__main__":
    main()
