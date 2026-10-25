"""Run every pattern on the same task set with the toy fake model and print calls and latency.

The fake model sleeps ``latency + per_token_latency x completion_tokens`` per call, so
wall-clock times are real: sequential calls add up, parallel ones overlap. Because the
toy model is an oracle, the numbers compare **cost and latency shapes**, not answer
quality. ``critical path`` is the longest chain of LLM calls that had to wait for one
another, the floor on latency however fast the hardware.

    python -m agentic_architectures.bench --json docs/benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from agentic_architectures import __version__
from agentic_architectures import build_runtime
from agentic_architectures.patterns import PATTERNS
from agentic_architectures.patterns.reflexion import contains
from agentic_architectures.toy import TASKS
from agentic_architectures.toy import ToyModel

DETERMINISTIC = ("llm_calls", "tool_calls", "tokens", "critical_path")


def run_benchmark(latency: float = 0.05, per_token_latency: float = 0.001) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, pattern in PATTERNS.items():
        runs: list[dict[str, Any]] = []
        for task in TASKS:
            with tempfile.TemporaryDirectory(prefix="agentic-bench-") as space:
                model = ToyModel(latency=latency, per_token_latency=per_token_latency)
                rt = build_runtime(model, workspace=space)
                options: dict[str, Any] = {"evaluator": contains(task.expect)} if key == "reflexion" else {}
                result = pattern(rt, task.question, **options)
            runs.append(
                {
                    "task": task.id,
                    "status": result.status,
                    "solved": task.expect.lower() in result.answer.lower(),
                    "seconds": round(result.seconds, 3),
                    **{k: result.stats[k] for k in DETERMINISTIC},
                }
            )
        rows.append(
            {
                "pattern": key,
                "title": pattern.title,
                **{f"mean_{k}": round(mean(r[k] for r in runs), 2) for k in DETERMINISTIC},
                "mean_seconds": round(mean(r["seconds"] for r in runs), 3),
                "solved": sum(r["solved"] for r in runs),
                "runs": runs,
            }
        )
    return {
        "version": __version__,
        "python": platform.python_version(),
        "tasks": [t.id for t in TASKS],
        "latency": latency,
        "per_token_latency": per_token_latency,
        "rows": rows,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "| Pattern | LLM calls | Tool calls | Tokens | Critical path | Wall time (s) | Solved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    n = len(results["tasks"])
    for r in results["rows"]:
        lines.append(
            f"| {r['title']} | {r['mean_llm_calls']:g} | {r['mean_tool_calls']:g} | {r['mean_tokens']:g} | "
            f"{r['mean_critical_path']:g} | {r['mean_seconds']:.2f} | {r['solved']}/{n} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-bench", description=__doc__.splitlines()[0])
    parser.add_argument("--latency", type=float, default=0.05, help="seconds per call")
    parser.add_argument("--per-token-latency", type=float, default=0.001, help="seconds per output token")
    parser.add_argument("--json", type=Path, default=None, help="write the full results here")
    args = parser.parse_args(argv)
    results = run_benchmark(args.latency, args.per_token_latency)
    print(markdown(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
