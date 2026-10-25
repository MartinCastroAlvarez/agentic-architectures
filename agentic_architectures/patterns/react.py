"""ReAct: interleave reasoning and acting (thought → tool call → observation → ...).

The model writes a thought and either a tool call or a final answer; the runtime runs
the tool and feeds the (delimited) observation back; repeat until an answer or
``max_steps``. A repeated identical call is refused rather than re-run, and the step
limit and the run budget bound the loop whatever the model does.

When to use
    Open-ended tasks where the next step depends on what the last tool returned and the
    number of steps is not known: question answering over tools, debugging, browsing.

Trade-offs
    Flexible and grounded, but sequential (one call per step), and the whole history is
    re-sent every step so tokens grow quadratically. Small models drift or loop. If
    the steps are predictable, ReWOO or plan-and-execute are cheaper.

Cost profile
    1 call per step; typically 2-4 for a single-tool question. Critical path = steps.

Failure modes
    Hallucinated tool names or arguments (rejected by the registry with a readable
    error), looping on the same call (guarded), stopping early with an unsupported
    answer, and indirect prompt injection through tool output (delimited, and tools
    are least-privilege).

Sources
    Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR
    2023). See langchain-tag-agent and langsmith-agent-evals (ReAct agents in
    LangChain / LangGraph).
"""

from __future__ import annotations

from typing import Any

from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern

NAME = "react"
INSTRUCTIONS = (
    "Solve the task step by step. Think briefly, then either call one tool or give the "
    "final answer. Use tools for facts and arithmetic instead of guessing."
)


def _run(rt: Runtime, task: str, *, partial: Partial, max_steps: int = 6) -> tuple[str, str, dict[str, Any]]:
    answer, status, actions = act(
        rt, NAME, INSTRUCTIONS, f"Task: {task}", max_steps=max_steps, partial=partial
    )
    return (
        answer,
        status,
        {
            "steps": len(actions),
            "thoughts": [a.thought for a in actions if a.thought],
            "tools": [a.tool for a in actions if a.is_tool],
        },
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
