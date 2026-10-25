"""A scripted fake chat model for tests, examples and the benchmark.

``ScriptedModel`` answers by *label* (the role a call plays). A script maps a label to
a queue of replies; a reply is a string or a callable ``(messages) -> str``. Lookup
tries the exact label, then the part before the first ``:`` (so ``"worker"`` serves
``"worker:researcher"``), then ``"*"``. A handler function can replace the script
entirely. Every call is recorded in ``calls`` so tests can assert on control flow.

``latency`` and ``per_token_latency`` make each call sleep, so wall-clock measurements
of parallel patterns are real (parallel calls overlap, sequential ones add up).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import Reply
from agentic_architectures.core.model import estimate_tokens

ReplyFn = Callable[[Sequence[Message]], str]
Scripted = str | ReplyFn | Mapping[str, Any]
Handler = Callable[[str, Sequence[Message]], str]


class ScriptExhausted(AssertionError):
    """The pattern made a call the script did not expect."""


@dataclass(frozen=True, slots=True)
class Call:
    label: str
    messages: tuple[Message, ...]
    reply: str

    @property
    def prompt(self) -> str:
        return "\n".join(m.content for m in self.messages)


class ScriptedModel:
    """Deterministic replies by label; thread-safe."""

    def __init__(
        self,
        script: Mapping[str, Sequence[Scripted] | Scripted] | None = None,
        *,
        handler: Handler | None = None,
        latency: float = 0.0,
        per_token_latency: float = 0.0,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.latency = latency
        self.per_token_latency = per_token_latency
        self._handler = handler
        self._queues: dict[str, deque[Scripted]] = {}
        for label, replies in (script or {}).items():
            if isinstance(replies, str | Mapping) or callable(replies):
                self._queues[label] = deque([replies])
            else:
                self._queues[label] = deque(replies)
        self._lock = threading.Lock()
        self.calls: list[Call] = []

    def labels(self) -> list[str]:
        with self._lock:
            return [c.label for c in self.calls]

    def count(self, prefix: str) -> int:
        return sum(1 for label in self.labels() if label == prefix or label.startswith(prefix + ":"))

    def _next(self, label: str, messages: Sequence[Message]) -> str:
        if self._handler is not None:
            return self._handler(label, messages)
        with self._lock:
            for key in (label, label.split(":", 1)[0], "*"):
                queue = self._queues.get(key)
                if queue:
                    item = queue.popleft() if len(queue) > 1 else queue[0]
                    break
            else:
                raise ScriptExhausted(f"no scripted reply for label {label!r}")
        if isinstance(item, Mapping):
            return json.dumps(item)
        if callable(item):
            return item(messages)
        return item

    def chat(self, messages: Sequence[Message], *, label: str = "", json_mode: bool = False) -> Reply:
        text = self._next(label, messages)
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        completion_tokens = estimate_tokens(text)
        delay = self.latency + self.per_token_latency * completion_tokens
        if delay > 0:
            time.sleep(delay)
        with self._lock:
            self.calls.append(Call(label, tuple(messages), text))
        return Reply(text, prompt_tokens, completion_tokens)
