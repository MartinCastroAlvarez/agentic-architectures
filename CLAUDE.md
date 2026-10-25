# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`agentic-architectures` is a framework-free catalog of agentic architectures, one
readable module per pattern on one shared runtime, and the hub that links the sibling
repos implementing each pattern in a real framework.

## The rules

1. **Standard library only** in `agentic_architectures/`. No runtime dependencies.
2. **Local by default.** ollama on loopback is the default; the OpenAI-compatible
   adapter stays optional. No telemetry, no hosted calls in tests, examples or CI.
3. **Patterns go through the runtime.** `rt.llm` / `rt.chat` / `rt.call_tool` /
   `rt.parallel` only, so tracing, budgets and untrusted delimiting apply to everything.
4. **Model output is untrusted.** Parse with `extract_json` / `parse_action`, validate
   every choice in code, put tool output and other agents' text in `untrusted()` blocks.
5. **Every loop is bounded** by its own option and `rt.budget`; a run ends with a status
   and the best answer so far, never an exception from a budget.
6. **Offline tests** with `ScriptedModel`, asserting control flow, not prose.
7. **README numbers come from real runs** of `python -m agentic_architectures.bench`.
8. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config.

## Done means

`poetry run pytest` (coverage >= 90%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` runs;
README (tables, diagrams, series matrix) and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
