"""The shared runtime: one object that carries the model, tools, tracer and budget.

Every pattern is a plain function ``run(rt, task, **options) -> Result``. Patterns
talk to the world only through ``Runtime.llm`` (traced, budgeted) and
``Runtime.call_tool`` (traced, budgeted, output delimited as untrusted), so the same
guarantees hold for all of them. ``act`` is the shared "tool call or final answer"
step that ReAct, workers, executors and swarm agents build on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeVar

from agentic_architectures.core.budget import Budget
from agentic_architectures.core.budget import BudgetExceeded
from agentic_architectures.core.model import ChatModel
from agentic_architectures.core.model import Message
from agentic_architectures.core.model import assistant
from agentic_architectures.core.model import system
from agentic_architectures.core.model import user
from agentic_architectures.core.parsing import DATA_RULE
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import extract_json
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.retrieval import Hit
from agentic_architectures.core.retrieval import Retriever
from agentic_architectures.core.tools import ToolRegistry
from agentic_architectures.core.trace import Tracer

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Runtime:
    model: ChatModel
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    retriever: Retriever | None = None
    budget: Budget = field(default_factory=Budget)
    tracer: Tracer = field(default_factory=Tracer)
    max_workers: int = 4

    # ------------------------------------------------------------------ model
    def chat(self, label: str, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        self.budget.charge_llm()
        started = self.tracer.now()
        reply = self.model.chat(list(messages), label=label, json_mode=json_mode)
        self.budget.charge_tokens(reply.tokens)
        self.tracer.record("llm", label, reply.text, started=started, tokens=reply.tokens)
        return reply.text

    def llm(self, label: str, instructions: str, prompt: str, *, json_mode: bool = False) -> str:
        """One system + one user message. The data rule is always appended."""
        return self.chat(label, [system(f"{instructions}\n{DATA_RULE}"), user(prompt)], json_mode=json_mode)

    def llm_json(self, label: str, instructions: str, prompt: str) -> dict[str, Any]:
        """Like ``llm`` but parsed; an unparseable reply becomes ``{}`` (callers default)."""
        return extract_json(self.llm(label, instructions, prompt, json_mode=True)) or {}

    # ------------------------------------------------------------------ tools
    def call_tool(self, name: str, args: Any, *, tools: ToolRegistry | None = None) -> str:
        """Run a tool and return its output wrapped as untrusted data."""
        registry = self.tools if tools is None else tools
        self.budget.charge_tool()
        started = self.tracer.now()
        _, output = registry.run(name, args)
        shown = json.dumps(args, sort_keys=True)[:80] if isinstance(args, dict) else "?"
        self.tracer.record("tool", name, f"{shown} -> {output}", started=started)
        return untrusted(f"tool:{name}", output)

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        if self.retriever is None:
            return []
        hits = self.retriever.search(query, k)
        self.tracer.record("retrieve", "bm25", f"{query!r} -> {[h.doc.id for h in hits]}")
        return hits

    def note(self, kind: str, label: str, detail: str = "") -> None:
        """Record a non-LLM step (gate, route, vote, handoff, interrupt...)."""
        self.tracer.record(kind, label, detail)

    # ------------------------------------------------------------------ fan-out
    def parallel(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Map ``fn`` over ``items`` on a bounded thread pool, keeping order.

        The tracer's critical path forks here: every branch starts at the current
        depth, and the caller continues from the deepest branch.
        """
        work = list(items)
        if not work:
            return []
        base = self.tracer.fork_depth()
        depths: list[int] = [base] * len(work)

        def branch(pair: tuple[int, T]) -> R:
            index, item = pair
            self.tracer.join_branch(base)
            try:
                return fn(item)
            finally:
                depths[index] = self.tracer.fork_depth()

        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(work)))) as pool:
            results = list(pool.map(branch, enumerate(work)))
        self.tracer.merge_depths(depths)
        return results


# ---------------------------------------------------------------------- results
@dataclass
class Result:
    pattern: str
    answer: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    trace: Tracer | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return not self.status.startswith(("budget:", "error", "gate_failed", "interrupted"))

    def summary(self) -> str:
        s = self.stats
        return (
            f"[{self.pattern}] status={self.status} llm_calls={s.get('llm_calls', 0)} "
            f"tool_calls={s.get('tool_calls', 0)} tokens={s.get('tokens', 0)} "
            f"critical_path={s.get('critical_path', 0)} seconds={self.seconds:.3f}"
        )


class Partial:
    """The best answer so far, so a run cut short by the budget still returns something."""

    def __init__(self) -> None:
        self.answer = ""
        self.data: dict[str, Any] = {}

    def offer(self, answer: str) -> None:
        if answer:
            self.answer = answer


PatternFn = Callable[..., tuple[str, str, dict[str, Any]]]


def run_pattern(name: str, fn: PatternFn, rt: Runtime, task: str, **options: Any) -> Result:
    """Run one pattern body and package its outcome; budgets end runs, not processes."""
    partial = Partial()
    start = time.perf_counter()
    try:
        answer, status, data = fn(rt, task, partial=partial, **options)
    except BudgetExceeded as exc:
        rt.note("budget", exc.limit, str(exc))
        answer, status, data = partial.answer, f"budget:{exc.limit}", partial.data
    return Result(name, answer, status, data, rt.tracer.stats(), time.perf_counter() - start, rt.tracer)


# ---------------------------------------------------------------------- act
ACT_PROTOCOL = """Reply with ONE JSON object and nothing else:
- to use a tool: {"thought": "...", "tool": "<name>", "args": {...}}
- to finish:     {"thought": "...", "answer": "<final answer>"}"""


@dataclass(frozen=True, slots=True)
class Action:
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tool(self) -> bool:
        return bool(self.tool)

    def key(self) -> str:
        return json.dumps([self.tool, self.args], sort_keys=True, default=str)


def parse_action(text: str) -> Action:
    """Read a tool call or an answer. Unparseable text is taken as the answer."""
    data = extract_json(text)
    if data is None:
        return Action(answer=text.strip()[:4000])
    args = data.get("args")
    return Action(
        thought=as_str(data.get("thought"), limit=1000),
        tool=as_str(data.get("tool"), limit=64),
        args=args if isinstance(args, dict) else {},
        answer=as_str(data.get("answer")),
        extra={k: v for k, v in data.items() if k not in {"thought", "tool", "args", "answer"}},
    )


def act(
    rt: Runtime,
    label: str,
    instructions: str,
    prompt: str,
    *,
    tools: ToolRegistry | None = None,
    max_steps: int = 4,
    partial: Partial | None = None,
) -> tuple[str, str, list[Action]]:
    """A bounded tool loop. Returns ``(answer, status, actions)``.

    Status is ``answered`` or ``max_steps``. A repeated identical tool call is refused
    (the model is told) instead of being run again: the commonest way loops stall.
    """
    registry = rt.tools if tools is None else tools
    messages: list[Message] = [
        system(f"{instructions}\n\nTools:\n{registry.catalog()}\n\n{ACT_PROTOCOL}\n{DATA_RULE}"),
        user(prompt),
    ]
    actions: list[Action] = []
    seen: set[str] = set()
    for _ in range(max_steps):
        rt.budget.step()
        text = rt.chat(label, messages, json_mode=True)
        action = parse_action(text)
        actions.append(action)
        if not action.is_tool:
            if partial is not None:
                partial.offer(action.answer)
            return action.answer, "answered", actions
        messages.append(assistant(text))
        if action.key() in seen:
            rt.note("guard", "repeat", f"{action.tool} called again with the same args")
            observation = untrusted(
                f"tool:{action.tool}", "error: same call already made; use the result above"
            )
        else:
            seen.add(action.key())
            observation = rt.call_tool(action.tool, action.args, tools=registry)
        messages.append(user(f"Observation:\n{observation}"))
    return "", "max_steps", actions
