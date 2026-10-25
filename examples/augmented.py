"""Augmented LLM: retrieval + memory + a tool, in one call (plus a tool round)."""

from agentic_architectures.patterns import augmented
from agentic_architectures.patterns.memory import LongTermMemory
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    memory = LongTermMemory()
    show(augmented.run(runtime(), PORT, memory=memory, user_id="demo"))
    print("memory now holds:", memory.all("demo"))


if __name__ == "__main__":
    main()
