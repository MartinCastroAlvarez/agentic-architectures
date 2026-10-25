"""The one model interface every pattern uses.

A pattern never imports a provider. It calls ``ChatModel.chat`` with a list of messages
and a *label* naming the role the call plays in the architecture (``"router"``,
``"planner"``, ``"worker:researcher"``...). Real providers ignore the label; the
scripted fake uses it to pick a reply, and the tracer prints it, so a trace reads like
the architecture diagram.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message. ``content`` is plain text."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Reply:
    """What a model returned, with token counts (estimated when the provider has none)."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class ChatModel(Protocol):
    """Anything that turns messages into a reply.

    ``json_mode`` asks the provider for a JSON object when it supports that (ollama
    ``format=json``, OpenAI ``response_format``). Patterns still parse defensively.
    Implementations must be safe to call from several threads at once.
    """

    name: str

    def chat(self, messages: Sequence[Message], *, label: str = "", json_mode: bool = False) -> Reply: ...


def system(text: str) -> Message:
    return Message("system", text)


def user(text: str) -> Message:
    return Message("user", text)


def assistant(text: str) -> Message:
    return Message("assistant", text)


def estimate_tokens(text: str) -> int:
    """A provider-free estimate: about four characters per token."""
    return len(text) // 4 + 1
