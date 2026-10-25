"""Supervisor and hierarchical teams: a manager routes work to specialists (or to a team)."""

from agentic_architectures.patterns import supervisor
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    flat = show(supervisor.run(runtime(), SPEED))
    print("turns:", flat.data["turns"])
    teams = show(supervisor.run(runtime(), SPEED, members=supervisor.HIERARCHICAL))
    print("turns:", teams.data["turns"])


if __name__ == "__main__":
    main()
