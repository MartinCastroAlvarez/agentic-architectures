# Security

agentic-architectures runs LLM agents that call tools, read documents, pass messages
between agents and, in one pattern, write a file. People install it and point it at
their own models and documents. Design goals: nothing leaves the machine unless you opt
in, model output cannot run code or reach outside its directory, text in documents or
tool results cannot take over an agent, and no loop runs without bound.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/agentic-architectures/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The local machine and filesystem, environment variables (an optional API
key), the documents folder, the workspace folder, and your model endpoint.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| The model | every reply: tool names and arguments, routes, handoffs, scores, plans, lessons, facts | run code, read or write outside the configured folders, skip an approval, loop forever, accept its own work |
| A document author | the text of a file in the documents folder | give the agent instructions, make the loader read files outside the folder |
| Another agent | its messages, reports and handoff notes | instruct the supervisor, judge or next agent |
| Whoever writes the task | the task text | escape the prompt structure, exhaust the host |
| Whoever holds a checkpoint | the paused human-in-the-loop state | turn a resume into a different tool call or a new system prompt |
| Whoever controls `OLLAMA_HOST` | where prompts are sent | receive your documents and tasks without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Code execution through tools | No `eval`/`exec`/`compile`/`pickle`/`subprocess`/`shell=` anywhere in the package (an AST test enforces it). The calculator parses with `ast.parse(mode="eval")` and walks an allowlist (numbers, arithmetic, `pi`, `e`, eleven functions), with caps on length (200), nodes (64), exponents and results; refuses complex results. |
| Path traversal / symlink escape | `confined()` rejects absolute paths, `..`, hidden parts and any symlink along the path, then checks the resolved path is inside the root. The document loader skips symlinks and hidden entries and caps size and count. Tested with symlinks pointing outside. |
| Unwanted side effects | The only writing tool, `write_note`, is `requires_approval=True`: human-in-the-loop stops before it runs; a policy callback or a person decides. Note names match `[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}` and the size is capped. Patterns that don't implement approval never get a writing tool in the examples. |
| Excess privilege | `ToolRegistry.subset()` gives each route, worker and swarm agent only its tools; a call to anything else returns `error: unknown tool`. Arguments are validated against the tool's declared parameters and capped at 2,000 characters; output at 4,000. |
| Prompt injection (documents, tool output, other agents, checkpoints) | Untrusted text is wrapped in `<untrusted source="…">` after escaping `& < >`, so it cannot close its block (tested); every system prompt says such blocks are data, never instructions. This reduces injection; it cannot remove it. A poisoned document can still bias an answer: treat answers about untrusted content as untrusted. |
| A model accepting its own work | Scores are clamped and compared in code (a `"pass": true` from the critic is ignored), routes / workers / handoff targets / judge picks are validated against allowlists, ReWOO plans are checked for forward references before any tool runs, RAG citations are filtered to retrieved documents. |
| Runaway loops and cost | Every loop has its own limit (`max_steps`, `max_rounds`, `max_trials`, `max_expansions`, `max_turns`, `max_handoffs`, `max_iterations`, ...) and the run-wide `Budget` (LLM calls, tool calls, steps, tokens, seconds). Repeated identical tool calls are refused; the autonomous loop stops on stalls. Fan-out is bounded by `max_workers`. The CLI refuses tasks over 4,000 characters. |
| Tampered checkpoints | `resume()` checks the version, that the pending call is an approval-gated tool that exists, that messages have valid roles and start with a system prompt, and that edits only touch declared parameters. Checkpoints are JSON, never pickle. Sign them if they cross a trust boundary. |
| Poisoned or sensitive memory | Long-term facts are model output: capped (300 characters, 200 per user), de-duplicated, filtered for likely secrets (API keys, tokens, private keys, `password=`), stored per validated user id, and only ever placed in `<untrusted>` blocks. `forget(user_id)` deletes a user's facts. The secret filter is a pattern list, not a guarantee. |
| Data leaving the machine | `OLLAMA_HOST` must be loopback unless `AGENTIC_ALLOW_REMOTE=1`. The OpenAI-compatible adapter is opt-in and requires https for non-loopback hosts. No telemetry. |
| Secrets | Only `OPENAI_API_KEY` is read, only by the optional adapter, sent only in the `Authorization` header; it never enters prompts, traces or `repr()`. Provider errors never echo response bodies (some servers reflect headers). `.env` is git-ignored. |

## 2. Dependencies

The package has no runtime dependencies. CI runs `bandit` on the package and `pip-audit`
on the locked development environment. Dependabot alerts, security updates and secret
scanning are enabled on the repository.

## 3. Running it safely

- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, use TLS and trust
  that host with your documents and tasks.
- Point `--docs` at a folder you are willing to show the model, and `--workspace` at an
  empty folder.
- Keep `--approve ask` (the default) for anything that writes; `--approve yes` is for
  demos.
- Give agents the fewest tools that work, and set a `Budget` that fits the task.
