"""Human-in-the-loop: the run pauses before a side effect; a decision resumes it."""

import json
import tempfile
from pathlib import Path

from agentic_architectures.patterns import human_in_the_loop as hitl
from examples._common import SPEED
from examples._common import runtime
from examples._common import show


def main() -> None:
    with tempfile.TemporaryDirectory() as space:
        rt = runtime(workspace=space)
        paused = show(hitl.run(rt, SPEED))
        checkpoint = json.dumps(paused.data["checkpoint"])  # store it anywhere: it is plain JSON
        print("pending:", paused.data["pending"])
        done = show(hitl.resume(rt, json.loads(checkpoint), hitl.Decision("approve", note="looks right")))
        print("approvals:", done.data["approvals"])
        print("written:", [p.name for p in (Path(space) / "notes").iterdir()])


if __name__ == "__main__":
    main()
