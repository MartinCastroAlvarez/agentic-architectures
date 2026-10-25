"""Augmented LLM: one model call, augmented with retrieval, memory and tools.

The building block of every other pattern (Anthropic, "Building effective agents",
2024): before the model is called, relevant documents are retrieved and relevant
memories recalled; the model may call a tool or two; the exchange is written back to
memory. No planning, no loop beyond a couple of tool rounds.

When to use
    Most tasks. Try this before any multi-step architecture: a well-augmented single
    call often beats an elaborate agent on cost, latency and debuggability.

Trade-offs
    Cheap and predictable, but the model gets one shot at choosing what to do; there is
    no critique or replanning. Retrieval quality caps answer quality.

Cost profile
    1 LLM call, + 1 per tool round (``max_tool_rounds``, default 2). Retrieval and
    memory lookups are local and free.

Failure modes
    Irrelevant retrieved passages distract the model; stale or wrong memories get
    repeated with confidence; a poisoned document can bias the answer (it is delimited
    as untrusted data, which reduces but does not remove prompt injection).

Sources
    Anthropic, "Building effective agents" (2024), section "Building block: the
    augmented LLM". Lewis et al., "Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks" (2020).
"""

from __future__ import annotations

from typing import Any

from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern
from agentic_architectures.patterns.memory import LongTermMemory

NAME = "augmented"
INSTRUCTIONS = (
    "You are a helpful assistant. Use the retrieved passages and memories when they are "
    "relevant, call a tool only when you need a fact or a calculation, and answer briefly."
)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    memory: LongTermMemory | None = None,
    user_id: str = "default",
    k: int = 3,
    max_tool_rounds: int = 2,
) -> tuple[str, str, dict[str, Any]]:
    hits = rt.retrieve(task, k)
    recalled = memory.search(user_id, task, k=3) if memory else []
    if memory is not None:
        rt.note("memory", "recall", f"{len(recalled)} memories")
    context = [untrusted(f"doc:{h.doc.id}", h.doc.text) for h in hits]
    if recalled:
        context.append(untrusted("memory", "\n".join(recalled)))
    prompt = f"Task: {task}\n\nContext:\n" + ("\n".join(context) or "(none)")
    answer, status, actions = act(
        rt, NAME, INSTRUCTIONS, prompt, max_steps=max_tool_rounds + 1, partial=partial
    )
    if memory is not None and answer:
        memory.add(user_id, f"Asked: {task[:200]} | Answered: {answer[:300]}")
        rt.note("memory", "write", "1 memory")
    return (
        answer,
        status,
        {
            "retrieved": [h.doc.id for h in hits],
            "recalled": len(recalled),
            "tools": [a.tool for a in actions if a.is_tool],
        },
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
