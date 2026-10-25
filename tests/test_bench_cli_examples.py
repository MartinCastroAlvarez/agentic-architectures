"""The benchmark, the CLI and every example run offline; the committed numbers are real."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agentic_architectures import bench
from agentic_architectures import cli
from agentic_architectures.patterns import PATTERNS
from agentic_architectures.toy import TASKS
from agentic_architectures.toy import ToyModel
from agentic_architectures.toy import find_task

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "benchmark" / "results.json"
EXAMPLES = sorted(p.stem for p in (ROOT / "examples").glob("[a-z]*.py"))


def test_every_pattern_solves_every_benchmark_task_with_the_toy_model() -> None:
    results = bench.run_benchmark(latency=0.0, per_token_latency=0.0)
    assert [r["pattern"] for r in results["rows"]] == list(PATTERNS)
    for row in results["rows"]:
        assert row["solved"] == len(TASKS), row["pattern"]
    table = bench.markdown(results)
    assert table.count("\n") == len(PATTERNS) + 1


def test_committed_benchmark_matches_a_fresh_run() -> None:
    committed = json.loads(RESULTS.read_text())
    fresh = bench.run_benchmark(latency=0.0, per_token_latency=0.0)
    for old, new in zip(committed["rows"], fresh["rows"], strict=True):
        for key in bench.DETERMINISTIC:
            assert old[f"mean_{key}"] == new[f"mean_{key}"], (old["pattern"], key)
    assert committed["latency"] > 0  # the README numbers were measured with simulated latency


def test_bench_main_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    assert bench.main(["--latency", "0", "--per-token-latency", "0", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["rows"]
    assert "| ReAct |" in capsys.readouterr().out


def test_cli_runs_patterns_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0
    assert "tree_search" in capsys.readouterr().out
    assert cli.main(["react", "--provider", "fake", TASKS[0].question]) == 0
    out = capsys.readouterr().out
    assert "llm       react" in out and "status=answered" in out
    assert cli.main(["all", "--provider", "fake", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [p["pattern"] for p in payload][:2] == ["augmented", "chaining"]
    assert cli.main(["human_in_the_loop", "--provider", "fake", "--approve", "yes", "--quiet"]) == 0
    assert "status=answered" in capsys.readouterr().out
    assert cli.main(["human_in_the_loop", "--provider", "fake", "--approve", "no", "--quiet"]) == 0


def test_cli_refuses_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["nope", "--provider", "fake"]) == 2
    assert cli.main(["react", "--provider", "fake", "x" * 5000]) == 2
    assert cli.main(["react", "--provider", "ollama", "x"]) in {0, 1}  # 1 when no ollama is running
    capsys.readouterr()


def test_the_toy_model_handles_unknown_tasks() -> None:
    task = find_task("Task: write a haiku")
    assert task.id == "custom" and "write a haiku" in task.answer
    assert ToyModel().name == "toy"


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_runs(
    name: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENTIC_PROVIDER", raising=False)
    importlib.import_module(f"examples.{name}").main()
    assert "status=" in capsys.readouterr().out or name == "quickstart"


def test_one_example_per_pattern_module() -> None:
    modules = {p.stem for p in (ROOT / "agentic_architectures" / "patterns").glob("[a-z]*.py")}
    assert modules <= set(EXAMPLES)
