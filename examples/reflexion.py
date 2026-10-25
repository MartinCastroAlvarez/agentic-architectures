"""Reflexion: fail, write a lesson, try again with it; lessons carry over to the next run."""

from agentic_architectures.patterns import reflexion
from agentic_architectures.toy import TASKS
from examples._common import runtime
from examples._common import show


def main() -> None:
    task = TASKS[0]
    lessons: list[str] = []
    first = show(
        reflexion.run(runtime(), task.question, evaluator=reflexion.contains(task.expect), lessons=lessons)
    )
    print("trials:", len(first.data["trials"]), "lessons:", lessons)
    second = show(
        reflexion.run(runtime(), task.question, evaluator=reflexion.contains(task.expect), lessons=lessons)
    )
    print("second run, trials:", len(second.data["trials"]))


if __name__ == "__main__":
    main()
