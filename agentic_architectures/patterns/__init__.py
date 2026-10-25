"""The catalog: every pattern, its runnable variants, and the sibling repos that implement it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.patterns import agentic_rag
from agentic_architectures.patterns import augmented
from agentic_architectures.patterns import autonomous
from agentic_architectures.patterns import chaining
from agentic_architectures.patterns import debate
from agentic_architectures.patterns import evaluator_optimizer
from agentic_architectures.patterns import human_in_the_loop
from agentic_architectures.patterns import memory
from agentic_architectures.patterns import orchestrator_workers
from agentic_architectures.patterns import parallelization
from agentic_architectures.patterns import plan_execute
from agentic_architectures.patterns import react
from agentic_architectures.patterns import reflexion
from agentic_architectures.patterns import rewoo
from agentic_architectures.patterns import routing
from agentic_architectures.patterns import supervisor
from agentic_architectures.patterns import swarm
from agentic_architectures.patterns import tree_search


@dataclass(frozen=True)
class Pattern:
    key: str
    title: str
    run: Callable[..., Result]
    options: dict[str, Any] = field(default_factory=dict)

    def __call__(self, rt: Runtime, task: str, **overrides: Any) -> Result:
        return self.run(rt, task, **{**self.options, **overrides})


def _auto_approve(tool: str, args: dict[str, Any]) -> human_in_the_loop.Decision:
    return human_in_the_loop.Decision("approve", note="auto-approved (benchmark policy)")


PATTERNS: dict[str, Pattern] = {
    p.key: p
    for p in (
        Pattern("augmented", "Augmented LLM", augmented.run),
        Pattern("chaining", "Prompt chaining + gates", chaining.run),
        Pattern("routing", "Routing", routing.run),
        Pattern("sectioning", "Parallelization: sectioning", parallelization.run, {"mode": "sectioning"}),
        Pattern("voting", "Parallelization: voting", parallelization.run, {"mode": "voting"}),
        Pattern("orchestrator", "Orchestrator-workers", orchestrator_workers.run),
        Pattern("evaluator", "Evaluator-optimizer", evaluator_optimizer.run),
        Pattern("react", "ReAct", react.run),
        Pattern("plan_execute", "Plan-and-execute", plan_execute.run),
        Pattern("rewoo", "ReWOO", rewoo.run),
        Pattern("reflexion", "Reflexion", reflexion.run),
        Pattern("tree_search", "Tree search (LATS-lite)", tree_search.run),
        Pattern("supervisor", "Supervisor", supervisor.run),
        Pattern("hierarchical", "Hierarchical teams", supervisor.run, {"members": supervisor.HIERARCHICAL}),
        Pattern("swarm", "Swarm / handoffs", swarm.run),
        Pattern("debate", "Debate + judge", debate.run),
        Pattern("human_in_the_loop", "Human-in-the-loop", human_in_the_loop.run, {"approve": _auto_approve}),
        Pattern("memory", "Short + long-term memory", memory.run),
        Pattern("agentic_rag", "Agentic RAG", agentic_rag.run),
        Pattern("autonomous", "Autonomous loop + budgets", autonomous.run),
    )
}

__all__ = ["PATTERNS", "Pattern"]
