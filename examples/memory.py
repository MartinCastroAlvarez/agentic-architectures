"""Memory: facts learned in one session are recalled in the next; old turns fold into a summary."""

from agentic_architectures.patterns import memory
from examples._common import runtime
from examples._common import show


def main() -> None:
    store = memory.LongTermMemory()  # pass a path to keep it in a JSON file
    session1 = memory.ShortTermMemory()
    show(
        memory.run(
            runtime(),
            "My name is Ada. I prefer metric units.",
            long_term=store,
            short_term=session1,
            user_id="ada",
        )
    )
    print("long-term facts:", store.all("ada"))
    session2 = memory.ShortTermMemory()  # a new thread: no history, same user
    show(
        memory.run(runtime(), "Which units do I prefer?", long_term=store, short_term=session2, user_id="ada")
    )


if __name__ == "__main__":
    main()
