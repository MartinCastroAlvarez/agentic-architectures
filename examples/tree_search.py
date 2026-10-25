"""Tree search (LATS-lite): expand candidates, score them, follow the best by UCT."""

from agentic_architectures.patterns import tree_search
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(tree_search.run(runtime(), PORT, branching=3, max_expansions=3))
    for node in result.data["tree"]:
        print(node)


if __name__ == "__main__":
    main()
