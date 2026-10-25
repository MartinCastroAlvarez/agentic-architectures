# Contributing to agentic-architectures

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install
poetry run pytest
poetry run ruff check agentic_architectures tests examples && poetry run ruff format --check agentic_architectures tests examples
poetry run mypy agentic_architectures
poetry run bandit -c pyproject.toml -q -r agentic_architectures
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## Adding a pattern

1. One module in `agentic_architectures/patterns/` with a `_run(rt, task, *, partial,
   **options) -> (answer, status, data)` body and a `run()` wrapper through
   `run_pattern`. Talk to the world only through `rt.llm`, `rt.chat`, `rt.call_tool`,
   `rt.retrieve` and `rt.parallel`.
2. A docstring with: when to use, trade-offs, cost profile (LLM calls per task),
   failure modes, sources (papers or canonical posts).
3. Every loop bounded by its own option and by `rt.budget.step()`; every model choice
   (a route, a worker, a handoff, a score) validated in code.
4. Register it in `patterns/__init__.py`, teach `toy.py` its roles, add
   `examples/<module>.py`, tests that prove its control flow, a Mermaid diagram and a
   row in the README tables, then rerun the benchmark.

## What this project will not accept

- Runtime dependencies. The package is standard library only (a test enforces it).
- Tests that call a real model or touch the network (the provider tests use a stub
  server on 127.0.0.1).
- `eval`, `exec`, `pickle`, `subprocess`, `shell=`, or a tool that can reach outside its
  directory.
- A loop that the limits don't bound, or a model deciding its own acceptance.
- Hand-edited benchmark numbers.

## Updating the benchmark

```bash
poetry run python -m agentic_architectures.bench --json docs/benchmark/results.json
```

Paste the printed table into the README with the Python version and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
