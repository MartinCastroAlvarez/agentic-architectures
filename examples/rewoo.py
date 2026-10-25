"""ReWOO: the whole tool plan in one call, tools without the model, one solver call."""

from agentic_architectures.patterns import rewoo
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(rewoo.run(runtime(), PORT))
    print("plan:", result.data["plan"], "levels:", result.data["levels"])


if __name__ == "__main__":
    main()
