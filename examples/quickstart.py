"""Run every pattern on the same task with the offline toy model and compare the cost."""

from agentic_architectures import build_runtime
from agentic_architectures.patterns import PATTERNS
from agentic_architectures.toy import TASKS
from agentic_architectures.toy import ToyModel


def main() -> None:
    task = TASKS[0].question
    print(f"Task: {task}\n")
    for key, pattern in PATTERNS.items():
        result = pattern(build_runtime(ToyModel()), task)
        s = result.stats
        counts = f"llm={s['llm_calls']:<2} tools={s['tool_calls']:<2} path={s['critical_path']}"
        print(f"{key:<18} {result.status:<12} {counts}")


if __name__ == "__main__":
    main()
