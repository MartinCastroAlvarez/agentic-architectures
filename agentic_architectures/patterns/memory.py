"""Memory: short-term (this conversation) vs long-term (across conversations).

Short-term memory is the message history of one thread, kept inside a window: when it
grows past ``max_messages`` the oldest turns are folded into a running summary by one
LLM call. Long-term memory is a store of facts about a user that outlives the thread:
after each turn an extraction call proposes facts, code filters them (length, likely
secrets, duplicates) and stores them under the user's namespace; before each turn the
relevant facts are recalled by keyword overlap.

When to use
    Short-term: any multi-turn chat. Long-term: assistants that should remember
    preferences or facts across sessions (and users who have agreed to that).

Trade-offs
    Summaries lose detail; a bigger window costs tokens on every call. Long-term memory
    personalises but can go stale, be wrong, or be poisoned by what a user (or a
    document) says, and it is personal data: namespace it per user and let people
    delete it.

Cost profile
    1 LLM call per turn, + 1 extraction call per turn (long-term), + 1 summary call
    each time the window overflows.

Failure modes
    Facts extracted from a joke or a hypothetical; contradictory facts (both kept);
    recall by keywords missing paraphrases (swap in embeddings); a secret pasted in chat
    being remembered (filtered here by pattern, which is not exhaustive).

Sources
    Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023). Park et al.,
    "Generative Agents" (2023). LangGraph's checkpointer vs store split, shown in
    langgraph-short-long-term-memory.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import assistant
from agentic_architectures.core.model import system
from agentic_architectures.core.model import user
from agentic_architectures.core.parsing import DATA_RULE
from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.retrieval import tokenize
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "memory"
USER_ID = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
SECRETISH = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|-----BEGIN"
    r"|password\s*[:=]|api[_ -]?key\s*[:=])",
    re.IGNORECASE,
)
MAX_FACT = 300
MAX_FACTS_PER_USER = 200

CHAT = "You are a helpful assistant with memory. Use remembered facts when relevant. Be brief."
EXTRACT = (
    "Extract durable facts about the user from their message (preferences, name, goals). "
    'Reply with JSON: {"facts": ["...", ...]} (an empty list if there are none).'
)
SUMMARIZE = "Summarise this conversation in at most three sentences, keeping names, numbers and decisions."


@dataclass
class ShortTermMemory:
    """A message window plus a running summary of what fell out of it."""

    max_messages: int = 8
    keep_recent: int = 4
    summary: str = ""
    messages: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def overflowing(self) -> bool:
        return len(self.messages) > self.max_messages

    def fold(self, rt: Runtime) -> None:
        """Summarise everything except the newest ``keep_recent`` messages."""
        old, recent = self.messages[: -self.keep_recent], self.messages[-self.keep_recent :]
        transcript = "\n".join(f"{m.role}: {m.content}" for m in old)
        prior = f"Earlier summary: {self.summary}\n" if self.summary else ""
        self.summary = rt.llm(
            "memory:summarize", SUMMARIZE, prior + untrusted("transcript", transcript)
        ).strip()[:1500]
        self.messages = recent
        rt.note("memory", "fold", f"{len(old)} messages -> summary")

    def context(self) -> list[Message]:
        head = (
            [user(f"Summary of earlier turns:\n{untrusted('summary', self.summary)}")] if self.summary else []
        )
        return head + list(self.messages)


class LongTermMemory:
    """Facts per user, optionally persisted to one JSON file (no pickle, no database)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._facts: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text("utf-8"))
            if isinstance(raw, dict):
                self._facts = {
                    str(k): [str(f)[:MAX_FACT] for f in v][:MAX_FACTS_PER_USER]
                    for k, v in raw.items()
                    if USER_ID.match(str(k)) and isinstance(v, list)
                }

    @staticmethod
    def _check(user_id: str) -> str:
        if not USER_ID.match(user_id):
            raise ValueError("invalid user id")
        return user_id

    def add(self, user_id: str, fact: str) -> bool:
        """Store a fact; returns False when it was filtered out or already known."""
        fact = " ".join(str(fact).split())[:MAX_FACT]
        if not fact or SECRETISH.search(fact):
            return False
        with self._lock:
            facts = self._facts.setdefault(self._check(user_id), [])
            if fact.lower() in (f.lower() for f in facts):
                return False
            facts.append(fact)
            del facts[:-MAX_FACTS_PER_USER]
            self._save()
        return True

    def all(self, user_id: str) -> list[str]:
        with self._lock:
            return list(self._facts.get(self._check(user_id), []))

    def search(self, user_id: str, query: str, k: int = 3) -> list[str]:
        terms = set(tokenize(query))
        scored = [(len(terms & set(tokenize(f))), i, f) for i, f in enumerate(self.all(user_id))]
        return [f for score, _, f in sorted(scored, key=lambda t: (-t[0], -t[1])) if score][:k]

    def forget(self, user_id: str) -> None:
        with self._lock:
            self._facts.pop(self._check(user_id), None)
            self._save()

    def _save(self) -> None:
        if self.path:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._facts, indent=1, sort_keys=True), "utf-8")
            tmp.replace(self.path)


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    long_term: LongTermMemory | None = None,
    short_term: ShortTermMemory | None = None,
    user_id: str = "default",
) -> tuple[str, str, dict[str, Any]]:
    long_term = long_term if long_term is not None else LongTermMemory()
    short_term = short_term if short_term is not None else ShortTermMemory()
    recalled = long_term.search(user_id, task, k=5)
    rt.note("memory", "recall", f"{len(recalled)} facts for {user_id}")
    messages = [system(f"{CHAT}\n{DATA_RULE}")]
    if recalled:
        messages.append(user("Remembered facts:\n" + untrusted("memory", "\n".join(recalled))))
    messages += short_term.context()
    messages.append(user(task))
    answer = rt.chat("memory:chat", messages).strip()
    partial.offer(answer)
    short_term.add(user(task))
    short_term.add(assistant(answer))
    if short_term.overflowing():
        short_term.fold(rt)
    extracted = rt.llm_json("memory:extract", EXTRACT, untrusted("user_message", task))
    stored = [f for f in as_list(extracted.get("facts")) if long_term.add(user_id, str(f))]
    rt.note("memory", "write", f"{len(stored)} new facts")
    return answer, "answered", {"recalled": recalled, "stored": stored, "summary": short_term.summary}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
