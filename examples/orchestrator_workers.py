"""Orchestrator-workers: the model picks the subtasks, workers run them in parallel."""

from agentic_architectures.patterns import orchestrator_workers
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    result = show(orchestrator_workers.run(runtime(), PORT, max_subtasks=3))
    print("workers:", result.data["workers"])


if __name__ == "__main__":
    main()
