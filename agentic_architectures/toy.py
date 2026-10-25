"""A toy "brain" for the scripted fake model: deterministic replies for every pattern's roles.

``ToyModel`` is a :class:`~agentic_architectures.core.fake.ScriptedModel` whose handler
reads the call's label and prompt and answers the way that role should: the router
returns a route, a ReAct step calls the calculator until it has seen the calculator's
output, the critic scores a draft by whether it contains the expected figure, and so
on. It knows the answers to the three benchmark tasks in ``TASKS`` (it is an oracle),
so the benchmark measures **control flow, calls, tokens and latency**, not model
quality. A few roles are scripted to be wrong first (a Reflexion first attempt, one
voter, one debater, the first tree-search candidates) so the loops visibly do work.

Unknown tasks (e.g. from the CLI) get a generic writing answer.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from agentic_architectures.core.fake import ScriptedModel
from agentic_architectures.core.model import Message


@dataclass(frozen=True)
class ToyTask:
    id: str
    kind: str  # math | lookup | write
    question: str
    answer: str
    short: str
    expect: str
    wrong: str
    tool: str | None = None
    args: dict[str, str] = field(default_factory=dict)


TASKS = (
    ToyTask(
        "speed",
        "math",
        "A train covers 120 km in 1.5 hours and then 80 km in 0.5 hours. What is its average speed in km/h?",
        "The average speed is 100 km/h: 200 km in 2 hours.",
        "100 km/h",
        "100",
        "About 90 km/h.",
        "calculator",
        {"expression": "(120 + 80) / (1.5 + 0.5)"},
    ),
    ToyTask(
        "port",
        "lookup",
        "Which TCP port does a local ollama server listen on by default?",
        "ollama listens on port 11434 by default [ollama].",
        "11434",
        "11434",
        "Port 8080.",
        "search",
        {"query": "ollama default port"},
    ),
    ToyTask(
        "idempotency",
        "write",
        "In two sentences, why do payment APIs use idempotency keys?",
        "An idempotency key lets a client retry a payment request after a timeout without "
        "charging twice. The server stores the key with the first result and replays that "
        "result for every retry.",
        "safe retry",
        "retry",
        "They make requests faster.",
    ),
)

UNTRUSTED = re.compile(r'<untrusted source="([^"]+)">\n(.*?)\n</untrusted>', re.DOTALL)


def blocks(text: str, prefix: str) -> list[tuple[str, str]]:
    return [(s, b) for s, b in UNTRUSTED.findall(text) if s.startswith(prefix)]


def find_task(text: str) -> ToyTask:
    for task in TASKS:
        if task.question in text:
            return task
    match = re.search(r"(?:Task|Goal|Question): (.+)", text)
    question = (match.group(1) if match else text).strip()[:200]
    return ToyTask(
        "custom",
        "write",
        question,
        f"(fake model) A short answer to: {question}",
        "(fake)",
        "(fake model)",
        "(fake model) I am not sure.",
    )


def _j(value: Any) -> str:
    return json.dumps(value)


class ToyBrain:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._votes = 0

    # A tool-using step: call the task's tool once if this role has it, then answer.
    @staticmethod
    def act(t: ToyTask, text: str) -> str:
        has_tool = t.tool is not None and f"- {t.tool}(" in text
        seen = t.tool is not None and f'source="tool:{t.tool}"' in text
        if has_tool and not seen:
            return _j({"thought": f"I need the {t.tool} tool.", "tool": t.tool, "args": t.args})
        return _j({"thought": "I have what I need.", "answer": t.answer})

    def __call__(self, label: str, messages: Sequence[Message]) -> str:
        text = "\n".join(m.content for m in messages)
        t = find_task(text)
        role, _, sub = label.partition(":")
        handler = getattr(self, f"_{role}", None) if role.isidentifier() else None
        if not callable(handler):
            return self.act(t, text)
        reply: str = handler(t, text, sub)
        return reply

    # ------------------------------------------------------------ chaining
    def _chain(self, t: ToyTask, text: str, step: str) -> str:
        if step == "outline":
            return f"- {t.answer}\n- Why it matters for the reader."
        if step == "draft":
            return f"{t.answer} That is the whole point."
        return t.answer

    # ------------------------------------------------------------ routing
    def _router(self, t: ToyTask, text: str, _: str) -> str:
        return _j({"route": t.kind, "reason": f"looks like a {t.kind} question"})

    # ------------------------------------------------------------ parallelization
    def _section(self, t: ToyTask, text: str, name: str) -> str:
        return f"{name}: {t.answer}"

    def _aggregate(self, t: ToyTask, text: str, _: str) -> str:
        return t.answer

    def _vote(self, t: ToyTask, text: str, _: str) -> str:
        with self._lock:
            self._votes += 1
            wrong = self._votes % 3 == 0
        return _j({"answer": t.wrong if wrong else t.short})

    # ------------------------------------------------------------ orchestrator-workers
    def _orchestrator(self, t: ToyTask, text: str, _: str) -> str:
        first = f"Use the {t.tool} tool to get the fact the task needs" if t.tool else "Draft the key point"
        return _j(
            {
                "subtasks": [
                    {"id": "gather", "instruction": first},
                    {"id": "explain", "instruction": "Explain the result in one sentence"},
                ]
            }
        )

    def _synthesize(self, t: ToyTask, text: str, _: str) -> str:
        return t.answer

    # ------------------------------------------------------------ evaluator-optimizer
    def _generate(self, t: ToyTask, text: str, _: str) -> str:
        return t.answer if 'source="critic"' in text else t.wrong

    def _evaluate(self, t: ToyTask, text: str, _: str) -> str:
        draft = " ".join(b for _, b in blocks(text, "draft"))
        if t.expect.lower() in draft.lower():
            return _j({"score": 9, "feedback": "Correct and concise."})
        return _j({"score": 4, "feedback": "Wrong or missing the key figure; check the facts."})

    # ------------------------------------------------------------ planning
    def _planner(self, t: ToyTask, text: str, _: str) -> str:
        steps = (
            [f"Use the {t.tool} tool to get the fact", "Write the final answer"]
            if t.tool
            else ["Write the answer"]
        )
        return _j({"steps": steps})

    def _replanner(self, t: ToyTask, text: str, _: str) -> str:
        results = " ".join(b for _, b in blocks(text, "step:"))
        if t.expect.lower() in results.lower():
            return _j({"answer": t.answer})
        return _j({"steps": ["Write the final answer"]})

    def _rewoo(self, t: ToyTask, text: str, role: str) -> str:
        if role == "planner":
            plan = [{"id": "E1", "why": "get the fact", "tool": t.tool, "args": t.args}] if t.tool else []
            return _j({"plan": plan})
        return t.answer

    # ------------------------------------------------------------ reflexion
    def _actor(self, t: ToyTask, text: str, _: str) -> str:
        if 'source="lessons"' not in text:
            return _j({"thought": "I can answer from memory.", "answer": t.wrong})
        return self.act(t, text)

    def _reflect(self, t: ToyTask, text: str, _: str) -> str:
        how = (
            f"use the {t.tool} tool instead of guessing" if t.tool else "state the core mechanism explicitly"
        )
        return f"Lesson: {how}, and give the exact answer."

    def _reflexion(self, t: ToyTask, text: str, _: str) -> str:
        answer = " ".join(b for _, b in blocks(text, "answer"))
        ok = t.expect.lower() in answer.lower()
        return _j({"pass": ok, "feedback": "correct" if ok else "wrong answer"})

    # ------------------------------------------------------------ tree search
    def _expand(self, t: ToyTask, text: str, _: str) -> str:
        if blocks(text, "parent"):
            return _j({"candidates": [t.answer, f"Partial: {t.short}, roughly."]})
        return _j(
            {"candidates": [t.wrong, f"Partial: {t.short}, roughly.", "I would need more information."]}
        )

    def _value(self, t: ToyTask, text: str, _: str) -> str:
        candidate = " ".join(b for _, b in blocks(text, "candidate"))
        if candidate.startswith("Partial"):
            return _j({"score": 6, "reflection": "On the right track; make it complete."})
        if t.expect.lower() in candidate.lower():
            return _j({"score": 10, "reflection": "Complete and correct."})
        return _j({"score": 2, "reflection": "Wrong or vague."})

    # ------------------------------------------------------------ supervisor
    def _supervisor(self, t: ToyTask, text: str, team: str) -> str:
        needed = {"calculator": "analyst", "search": "researcher"}.get(t.tool or "", "")
        reported = {s.removeprefix("worker:") for s, _ in blocks(text, "worker:")}
        if team:  # a team's own supervisor
            if needed and needed not in reported:
                return _j({"next": needed, "instruction": "Get the fact the task needs."})
            return _j({"next": "FINISH", "answer": f"Findings: {t.answer}"})
        if "research_team" in text.split("Task:")[0]:
            needed = "research_team" if needed else ""
        if needed and needed not in reported:
            return _j({"next": needed, "instruction": "Get the fact the task needs."})
        if "writer" not in reported:
            return _j({"next": "writer", "instruction": "Write the final answer."})
        return _j({"next": "FINISH", "answer": t.answer})

    # ------------------------------------------------------------ swarm
    def _swarm(self, t: ToyTask, text: str, agent: str) -> str:
        if agent == "triage":
            target = {"math": "math", "lookup": "research"}.get(t.kind, "writer")
            return _j({"handoff": target, "note": f"This is a {t.kind} question."})
        return self.act(t, text)

    # ------------------------------------------------------------ debate
    def _debate(self, t: ToyTask, text: str, who: str) -> str:
        if who == "3" and not blocks(text, "peers"):
            return _j({"answer": t.wrong, "argument": "A quick estimate."})
        return _j({"answer": t.short, "argument": "Checked step by step."})

    def _judge(self, t: ToyTask, text: str, _: str) -> str:
        return _j({"winner": 1, "answer": t.answer, "reason": "best supported"})

    # ------------------------------------------------------------ human-in-the-loop
    def _hitl(self, t: ToyTask, text: str, _: str) -> str:
        if t.tool and f'source="tool:{t.tool}"' not in text:
            return self.act(t, text)
        if 'source="tool:write_note"' not in text:
            return _j(
                {
                    "thought": "Save the answer for the user.",
                    "tool": "write_note",
                    "args": {"name": t.id, "text": t.answer},
                }
            )
        return _j({"answer": t.answer})

    # ------------------------------------------------------------ memory
    def _memory(self, t: ToyTask, text: str, role: str) -> str:
        if role == "extract":
            message = " ".join(b for _, b in blocks(text, "user_message"))
            facts = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", message) if re.match(r"(My|I) ", s.strip())
            ]
            return _j({"facts": facts})
        if role == "summarize":
            return "Summary: " + " ".join(re.findall(r"user: ([^\n]{0,60})", text))[:300]
        if t.id != "custom":
            return t.answer
        remembered = " ".join(b for _, b in blocks(text, "memory"))
        last = messages_last_user(text)
        if remembered and "?" in last:
            return f"From what I remember: {' '.join(remembered.split())}"
        return "Noted."

    # ------------------------------------------------------------ agentic RAG
    def _rag(self, t: ToyTask, text: str, role: str) -> str:
        if role == "decide":
            if t.kind == "lookup":
                return _j({"retrieve": True, "query": t.args.get("query", t.question)})
            return _j({"retrieve": False, "query": ""})
        docs = blocks(text, "doc:")
        if role == "grade":
            relevant = [s.removeprefix("doc:") for s, b in docs if t.expect.lower() in b.lower()]
            return _j({"relevant": relevant, "rewrite": "" if relevant else t.question})
        return _j({"answer": t.answer, "citations": [s.removeprefix("doc:") for s, _ in docs]})

    # ------------------------------------------------------------ autonomous
    def _autonomous(self, t: ToyTask, text: str, _: str) -> str:
        if "Open tasks: (none)" in text and "Finished:\n(none)" in text:
            first = f"Use the {t.tool} tool" if t.tool else "Draft the answer"
            return _j({"add_tasks": [first, "Write the final answer"]})
        if t.tool and f'source="tool:{t.tool}"' not in text:
            return _j({"tool": t.tool, "args": t.args})
        if "Finished:\n(none)" in text:
            return _j({"done": t.answer})
        return _j({"answer": t.answer})


def messages_last_user(text: str) -> str:
    return text.strip().splitlines()[-1] if text.strip() else ""


class ToyModel(ScriptedModel):
    """``ScriptedModel`` driven by ``ToyBrain``; ``latency`` seconds per call (+ per token)."""

    def __init__(self, *, latency: float = 0.0, per_token_latency: float = 0.0) -> None:
        super().__init__(handler=ToyBrain(), latency=latency, per_token_latency=per_token_latency, name="toy")
