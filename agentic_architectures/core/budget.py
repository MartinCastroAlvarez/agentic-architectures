"""Hard limits every run is held to, whatever the pattern and whatever the model says.

A ``Budget`` is charged by the runtime before each LLM call (calls), after it
(tokens), before each tool call, and by loops on each iteration (steps). Exceeding any
limit raises ``BudgetExceeded``; ``run_pattern`` turns that into a result with status
``budget:<limit>`` and the best answer so far, never a crash and never another call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from dataclasses import field


class BudgetExceeded(Exception):
    def __init__(self, limit: str, used: float, cap: float) -> None:
        super().__init__(f"budget exceeded: {limit} ({used:g} > {cap:g})")
        self.limit = limit
        self.used = used
        self.cap = cap


@dataclass
class Budget:
    max_llm_calls: int = 40
    max_tool_calls: int = 40
    max_steps: int = 40
    max_tokens: int = 200_000
    max_seconds: float = 300.0
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    tokens: int = 0
    _start: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_llm_calls", "max_tool_calls", "max_steps", "max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def _check_time(self) -> None:
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded("seconds", round(self.elapsed, 3), self.max_seconds)

    def charge_llm(self) -> None:
        with self._lock:
            self._check_time()
            if self.tokens >= self.max_tokens:
                raise BudgetExceeded("tokens", self.tokens, self.max_tokens)
            if self.llm_calls + 1 > self.max_llm_calls:
                raise BudgetExceeded("llm_calls", self.llm_calls + 1, self.max_llm_calls)
            self.llm_calls += 1

    def charge_tokens(self, tokens: int) -> None:
        with self._lock:
            self.tokens += tokens

    def charge_tool(self) -> None:
        with self._lock:
            self._check_time()
            if self.tool_calls + 1 > self.max_tool_calls:
                raise BudgetExceeded("tool_calls", self.tool_calls + 1, self.max_tool_calls)
            self.tool_calls += 1

    def step(self) -> None:
        with self._lock:
            self._check_time()
            if self.steps + 1 > self.max_steps:
                raise BudgetExceeded("steps", self.steps + 1, self.max_steps)
            self.steps += 1
