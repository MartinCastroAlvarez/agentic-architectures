"""Prompt chaining: outline -> draft -> polish, with a code gate between each step."""

from agentic_architectures.patterns import chaining
from examples._common import IDEMPOTENCY
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(chaining.run(runtime(), IDEMPOTENCY))
    for step, text in result.data["outputs"].items():
        print(f"--- {step}\n{text}")


if __name__ == "__main__":
    main()
