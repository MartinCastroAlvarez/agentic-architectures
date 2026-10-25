"""Autonomous loop: a task list, tools, and hard budgets that end the run, not the model."""

from agentic_architectures.core.budget import Budget
from agentic_architectures.patterns import autonomous
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(
        autonomous.run(runtime(budget=Budget(max_llm_calls=8, max_tool_calls=4, max_seconds=60)), PORT)
    )
    print("done:", result.data["done"])
    tight = show(autonomous.run(runtime(budget=Budget(max_llm_calls=2)), PORT))
    print("with 2 calls the budget stops it:", tight.status)


if __name__ == "__main__":
    main()
