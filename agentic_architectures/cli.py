"""``python -m agentic_architectures <pattern|all|list> "<task>"``.

The default provider is a local ollama (``AGENTIC_PROVIDER``, default ``ollama``);
``--provider fake`` runs offline with the toy model. Every run prints its trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from agentic_architectures import build_runtime
from agentic_architectures.core.budget import Budget
from agentic_architectures.core.providers import ProviderError
from agentic_architectures.core.providers import model_from_env
from agentic_architectures.core.runtime import Result
from agentic_architectures.patterns import PATTERNS
from agentic_architectures.patterns import human_in_the_loop
from agentic_architectures.toy import TASKS

MAX_TASK_CHARS = 4000


def _ask(tool: str, args: dict[str, Any]) -> human_in_the_loop.Decision:
    print(f"\n[approval] the agent wants to call {tool} with {json.dumps(args)[:500]}", file=sys.stderr)
    if not sys.stdin.isatty():
        print("[approval] stdin is not a terminal: rejected", file=sys.stderr)
        return human_in_the_loop.Decision("reject", note="no human available")
    print("approve? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = input().strip().lower()
    if answer in {"y", "yes"}:
        return human_in_the_loop.Decision("approve")
    return human_in_the_loop.Decision("reject", note="rejected at the prompt")


def _policy(name: str) -> human_in_the_loop.Approver:
    if name == "yes":
        return lambda tool, args: human_in_the_loop.Decision("approve", note="--approve yes")
    if name == "no":
        return lambda tool, args: human_in_the_loop.Decision("reject", note="--approve no")
    return _ask


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentic-architectures", description="Run an agentic architecture and print its trace."
    )
    parser.add_argument("pattern", help="a pattern key, 'all', or 'list'")
    parser.add_argument("task", nargs="?", default=TASKS[0].question, help="the task (default: a demo task)")
    parser.add_argument("--provider", choices=["ollama", "openai", "fake"], default=None)
    parser.add_argument("--model", default=None, help="model id (default: qwen3:4b on ollama)")
    parser.add_argument("--docs", default=None, help="folder of .md/.txt documents to search (read-only)")
    parser.add_argument("--workspace", default=None, help="the only folder tools may write to")
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument(
        "--approve", choices=["ask", "yes", "no"], default="ask", help="human-in-the-loop policy"
    )
    parser.add_argument("--latency", type=float, default=0.0, help="fake provider: seconds per call")
    parser.add_argument("--quiet", action="store_true", help="no live trace")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    return parser.parse_intermixed_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pattern == "list":
        for key, pattern in PATTERNS.items():
            print(f"{key:<18} {pattern.title}")
        return 0
    keys = list(PATTERNS) if args.pattern == "all" else [args.pattern]
    unknown = [k for k in keys if k not in PATTERNS]
    if unknown:
        print(f"unknown pattern {unknown[0]!r}; try 'list'", file=sys.stderr)
        return 2
    if len(args.task) > MAX_TASK_CHARS:
        print(f"task longer than {MAX_TASK_CHARS} characters", file=sys.stderr)
        return 2
    env = dict(os.environ)
    if args.provider:
        env["AGENTIC_PROVIDER"] = args.provider
    if args.model:
        env["AGENTIC_MODEL"] = args.model
    env["AGENTIC_FAKE_LATENCY"] = str(args.latency)
    try:
        model = model_from_env(env)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    results: list[Result] = []
    for key in keys:
        rt = build_runtime(
            model,
            **({"docs_dir": args.docs} if args.docs else {}),
            workspace=args.workspace,
            budget=Budget(max_llm_calls=args.max_llm_calls, max_seconds=args.max_seconds),
            echo=not (args.quiet or args.json),
        )
        options: dict[str, Any] = {}
        if key == "human_in_the_loop":
            options["approve"] = _policy(args.approve)
        if not (args.quiet or args.json):
            print(f"\n=== {key}: {PATTERNS[key].title}")
        try:
            result = PATTERNS[key](rt, args.task, **options)
        except ProviderError as exc:
            print(f"error: {exc} (is ollama running? try --provider fake)", file=sys.stderr)
            return 1
        results.append(result)
        if not args.json:
            print(f"\n{result.answer}\n{result.summary()}")
    if args.json:
        payload = [
            {
                "pattern": r.pattern,
                "status": r.status,
                "answer": r.answer,
                "stats": r.stats,
                "data": _jsonable(r.data),
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    return 0


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
