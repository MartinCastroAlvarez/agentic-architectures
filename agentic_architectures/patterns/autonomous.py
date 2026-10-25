"""Autonomous loop agent: pursue a goal with a task list until done, stuck or out of budget.

The agent keeps a small task list and a notebook. Each iteration it either adds tasks
(``{"add_tasks": [...]}``), works on the current task with a tool, marks it done with a
note (``{"done": "<note>"}``), or declares the goal met (``{"answer": ...}``). The loop
does not trust the model to stop: it ends on the answer, on a **budget** (LLM calls,
tool calls, tokens, wall-clock seconds, iterations; all enforced by ``Budget``), or on a
**stall** (the same action ``stall_after`` times in a row, or nothing added, run or
finished for that many iterations).

When to use
    Open-ended goals in a trusted sandbox where the number of steps is unknown and a
    person is not watching every step: research sweeps, maintenance chores, coding
    agents with tests. Only with hard budgets and a narrow tool set.

Trade-offs
    The most autonomy and the least predictability: cost and duration vary run to run,
    errors compound over long horizons, and evaluation is hard. Most production
    "agents" are better served by a workflow with one agentic step.

Cost profile
    1 call per iteration, bounded by the budget (default here: 12 calls, 8 tool calls,
    60 s). Typically 4-8 calls for a small goal.

Failure modes
    Loops and busywork (stall detection), task-list explosion (capped at
    ``max_tasks``), goal drift, runaway cost (budgets), and acting on injected
    instructions found in tool output (delimited; narrow tools).

Sources
    Nakajima, "BabyAGI" (2023). Significant Gravitas, "AutoGPT" (2023). Anthropic,
    "Building effective agents" (2024), "Agents" (stopping conditions, sandboxing).
    Wang et al., "A Survey on Large Language Model based Autonomous Agents" (2023).
"""

from __future__ import annotations

from typing import Any

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import assistant
from agentic_architectures.core.model import system
from agentic_architectures.core.model import user
from agentic_architectures.core.parsing import DATA_RULE
from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import parse_action
from agentic_architectures.core.runtime import run_pattern

NAME = "autonomous"
INSTRUCTIONS = """You are an autonomous agent working towards a goal with a task list.
Reply with ONE JSON object per turn:
- plan work:        {{"add_tasks": ["...", ...]}}
- use a tool:       {{"tool": "<name>", "args": {{...}}}}
- finish a task:    {{"done": "<what you found or did>"}}
- goal achieved:    {{"answer": "<final answer>"}}
Tools:
{tools}"""


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_iterations: int = 12,
    max_tasks: int = 6,
    stall_after: int = 3,
) -> tuple[str, str, dict[str, Any]]:
    todo: list[str] = []
    done: list[tuple[str, str]] = []
    messages: list[Message] = [system(INSTRUCTIONS.format(tools=rt.tools.catalog()) + "\n" + DATA_RULE)]
    last_key, repeats, idle = "", 0, 0
    for _ in range(max_iterations):
        rt.budget.step()
        notebook = untrusted("notebook", "\n".join(f"{t}: {n}" for t, n in done)) if done else "(none)"
        state = f"Goal: {task}\nOpen tasks: {todo or '(none)'}\nFinished:\n{notebook}"
        text = rt.chat(NAME, [*messages, user(state)], json_mode=True)
        action = parse_action(text)
        key = action.key() + str(action.extra)
        repeats = repeats + 1 if key == last_key else 1
        last_key = key
        if repeats >= stall_after:
            rt.note("guard", "stall", f"same action {repeats} times")
            return partial.answer, "stalled", {"done": done, "todo": todo}
        if action.answer:
            partial.offer(action.answer)
            return action.answer, "goal_met", {"done": done, "todo": todo}
        progress = False
        new = [as_str(t, limit=200).strip() for t in as_list(action.extra.get("add_tasks"))]
        room = max_tasks - len(todo) - len(done)
        added = [t for t in new if t and t not in todo][: max(0, room)]
        if added:
            todo.extend(added)
            progress = True
            rt.note("plan", "tasks", " | ".join(added))
        note = as_str(action.extra.get("done"), limit=800).strip()
        if note:
            finished = todo.pop(0) if todo else "(unplanned)"
            done.append((finished, note))
            partial.offer(note)
            progress = True
            rt.note("plan", "done", finished)
        if action.is_tool:
            observation = rt.call_tool(action.tool, action.args)
            messages += [assistant(text), user(f"Observation:\n{observation}")]
            progress = True
        idle = 0 if progress else idle + 1
        if idle >= stall_after:
            rt.note("guard", "stall", f"no progress for {idle} iterations")
            return partial.answer, "stalled", {"done": done, "todo": todo}
    return partial.answer, "max_iterations", {"done": done, "todo": todo}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
