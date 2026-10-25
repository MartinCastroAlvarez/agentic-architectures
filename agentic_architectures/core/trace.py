"""A tracer that records every step a pattern takes.

Each LLM call, tool call, gate, route, handoff, vote, interrupt... becomes a ``Step``.
``Tracer.render()`` prints the run as a timeline; ``Tracer.stats()`` counts calls and
tokens and computes the *critical path* (the longest chain of LLM calls that had to
wait for each other), which is what latency really depends on.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

MAX_DETAIL = 160


@dataclass(frozen=True, slots=True)
class Step:
    index: int
    kind: str
    label: str
    detail: str
    started: float
    duration: float = 0.0
    tokens: int = 0
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _short(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= MAX_DETAIL else flat[: MAX_DETAIL - 1] + "…"


@dataclass
class Tracer:
    """Thread-safe step log. ``echo=True`` prints steps as they happen."""

    echo: bool = False
    steps: list[Step] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def now(self) -> float:
        return time.perf_counter() - self._t0

    # The critical-path depth of the current thread: an LLM call depends on every call
    # that finished before it on the same logical branch.
    def _depth(self) -> int:
        return int(getattr(self._local, "depth", 0))

    def fork_depth(self) -> int:
        """Capture the depth before fanning out; pass it to ``join_branch`` in each branch."""
        return self._depth()

    def join_branch(self, depth: int) -> None:
        self._local.depth = depth

    def merge_depths(self, depths: list[int]) -> None:
        """After a fan-in, continue from the deepest branch."""
        self._local.depth = max([self._depth(), *depths])

    def record(
        self,
        kind: str,
        label: str,
        detail: str = "",
        *,
        started: float | None = None,
        tokens: int = 0,
    ) -> Step:
        end = self.now()
        start = end if started is None else started
        depth = self._depth()
        if kind == "llm":
            depth += 1
            self._local.depth = depth
        with self._lock:
            step = Step(len(self.steps), kind, label, _short(detail), start, end - start, tokens, depth)
            self.steps.append(step)
        if self.echo:
            print(self.format(step), flush=True)
        return step

    @staticmethod
    def format(step: Step) -> str:
        return f"[{step.started:7.3f}s] {step.kind:<9} {step.label:<22} {step.detail}"

    def render(self) -> str:
        with self._lock:
            return "\n".join(self.format(s) for s in self.steps)

    def stats(self) -> dict[str, int]:
        with self._lock:
            steps = list(self.steps)
        llm = [s for s in steps if s.kind == "llm"]
        return {
            "llm_calls": len(llm),
            "tool_calls": sum(1 for s in steps if s.kind == "tool"),
            "tokens": sum(s.tokens for s in llm),
            "critical_path": max((s.depth for s in llm), default=0),
            "steps": len(steps),
        }

    def to_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self.steps]
