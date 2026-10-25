# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- A framework-free runtime with zero runtime dependencies: the `ChatModel` protocol, an
  ollama adapter over plain HTTP (loopback only by default), an optional
  OpenAI-compatible adapter, a thread-safe scripted fake (`ScriptedModel`) and a toy
  "brain" for examples and the benchmark (`ToyModel`); the `Tool` protocol and
  `ToolRegistry` (argument validation, capped output, errors returned as text); a
  `Tracer` that records every step and the critical path; a `Budget` on LLM calls, tool
  calls, steps, tokens and seconds; `untrusted()` delimiting for tool output, documents
  and other agents' messages.
- Sandboxed tools: an AST-allowlist calculator, BM25 search over bundled documents,
  read-only file tools and an approval-gated `write_note`, all confined to one directory.
- 18 pattern modules (20 runnable variants): augmented LLM, prompt chaining with gates,
  routing, parallelization (sectioning, voting), orchestrator-workers,
  evaluator-optimizer, ReAct, plan-and-execute, ReWOO (with LLMCompiler-style parallel
  levels), Reflexion, LATS-lite tree search, supervisor and hierarchical teams, swarm
  handoffs, debate with a judge, human-in-the-loop (interrupt, JSON checkpoint, resume),
  short- and long-term memory, agentic RAG, and an autonomous loop with budgets and stall
  detection. Each docstring covers when to use it, trade-offs, cost, failure modes and
  sources.
- One runnable example per pattern, `python -m agentic_architectures` (every pattern,
  traces, JSON output) and `python -m agentic_architectures.bench` (calls, tokens,
  critical path and wall time with simulated latency), with committed results.
