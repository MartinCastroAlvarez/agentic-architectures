"""Swarm / handoffs: triage hands the conversation to a peer specialist."""

from agentic_architectures.patterns import swarm
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(swarm.run(runtime(), PORT))
    print("path:", " -> ".join(result.data["path"]))


if __name__ == "__main__":
    main()
