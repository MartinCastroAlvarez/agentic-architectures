# Architecture

```
agentic_architectures/
  core/
    model.py        Message, Reply, the ChatModel protocol (chat(messages, *, label, json_mode))
    providers.py    OllamaModel, OpenAICompatibleModel (http.client, loopback guard), model_from_env
    fake.py         ScriptedModel: replies by label (exact, prefix, "*"), records calls, simulated latency
    runtime.py      Runtime (llm, chat, call_tool, retrieve, parallel, note), Result, run_pattern, act
    budget.py       Budget: llm calls, tool calls, steps, tokens, seconds -> BudgetExceeded
    trace.py        Tracer: thread-safe steps, render(), stats() with the critical path
    tools.py        Tool protocol, ToolRegistry, Calculator (AST), SearchDocs, ReadFile, ListFiles, WriteNote
    retrieval.py    BM25 Retriever, confined document loader
    parsing.py      extract_json, untrusted() delimiting, DATA_RULE, coercion helpers
  patterns/         one module per architecture (see the README); __init__ is the catalog
  toy.py            ToyBrain / ToyModel: role-aware deterministic replies for the benchmark tasks
  bench.py          every pattern x every task -> calls, tokens, critical path, wall time
  cli.py            python -m agentic_architectures <pattern|all|list> "<task>"
  data/docs/        small Markdown documents for search and RAG
examples/           one runnable script per pattern + quickstart
tests/              control-flow tests per pattern, core, providers (stub server), hygiene
```

## One runtime, many patterns

```
pattern._run(rt, task, partial=…, **options)
   │  rt.llm(label, instructions, prompt)   ──► Budget.charge_llm ─► ChatModel.chat ─► Tracer.record("llm")
   │  rt.call_tool(name, args)               ──► Budget.charge_tool ─► ToolRegistry.run ─► untrusted(output)
   │  rt.parallel(fn, items)                 ──► ThreadPoolExecutor(max_workers) + critical-path join
   │  rt.budget.step()                       ──► one per loop iteration
   ▼
run_pattern ─► Result(answer, status, data, stats, seconds, trace)
               BudgetExceeded ─► status "budget:<limit>", answer = partial.answer
```

- **Labels** are the glue between a pattern, the fake model and the trace. A label is a
  role (`router`, `executor`, `worker:analyst`, `supervisor:research_team`, `swarm:triage`).
  `ScriptedModel` resolves an exact label, then its prefix before `:`, then `*`.
- **`act`** is the shared tool loop: a JSON `{"tool", "args"}` or `{"answer"}`
  protocol, bounded by `max_steps`, refusing a repeated identical call. Unparseable text
  is taken as the answer (small local models often drop the JSON at the end).
- **Partial answers.** Patterns `partial.offer()` their best answer as they go, so a run
  cut by the budget still returns something useful with an honest status.
- **Critical path.** Each thread carries a depth; an LLM call is one deeper than the
  last call on its branch. `parallel` starts every branch at the current depth and
  continues from the deepest. `stats()["critical_path"]` is the longest chain.

## Statuses

| Pattern | Statuses |
| --- | --- |
| all | `budget:llm_calls`, `budget:tool_calls`, `budget:steps`, `budget:tokens`, `budget:seconds` |
| act-based (augmented, react, routing) | `answered`, `max_steps` |
| chaining | `completed`, `gate_failed:<step>` |
| parallelization | `completed`, `majority`, `no_majority`, `no_votes` |
| orchestrator_workers | `completed` |
| evaluator_optimizer | `accepted`, `max_rounds` |
| plan_execute | `answered`, `plan_exhausted`, `max_steps` |
| rewoo | `answered`, `invalid_plan` |
| reflexion | `passed`, `max_trials` |
| tree_search | `solved`, `max_expansions`, `exhausted` |
| supervisor | `finished`, `max_turns` |
| swarm | `answered`, `max_handoffs`, `max_turns` |
| debate | `judged`, `majority_fallback` |
| human_in_the_loop | `answered`, `interrupted`, `max_steps` |
| memory | `answered` |
| agentic_rag | `answered`, `no_evidence` |
| autonomous | `goal_met`, `stalled`, `max_iterations` |

## Design choices

- **No framework, no dependencies.** The point is to see each pattern's control flow in
  plain code; the sibling repos show the same patterns in LangGraph, ADK, CrewAI and
  LangChain.
- **Threads, not asyncio.** Model calls are I/O bound and the fan-out is small; a bounded
  `ThreadPoolExecutor` keeps every pattern synchronous and readable.
- **JSON checkpoints.** Human-in-the-loop state is plain JSON (no pickle), so it can be
  stored anywhere and is re-validated on resume.
- **The toy model is an oracle.** It knows the benchmark answers so the benchmark
  measures cost and latency shapes; quality needs a real model and an eval harness.
