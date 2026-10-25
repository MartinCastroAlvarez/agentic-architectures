"""Routing: classify the input once, then hand it to a specialised handler.

A router call picks one route by name; code validates the choice (an unknown or
missing route falls back to ``general``) and dispatches to that route's handler, which
has its own instructions and **only the tools that route needs** (least privilege: the
writing route cannot call the calculator, the maths route cannot search).

When to use
    Distinct categories of input that are better handled separately: support intents,
    easy vs hard questions (send easy ones to a cheaper model), languages, domains.

Trade-offs
    Separation of concerns and cheaper handling of easy cases, at the cost of one extra
    call and a single point of failure: a misrouted request gets a confident wrong
    specialist. Keep a ``general`` fallback and measure routing accuracy.

Cost profile
    1 router call + the handler's calls (1, or more with tools). Critical path 2+.
    A rules-based or embedding classifier can replace the router call entirely.

Failure modes
    Misclassification on ambiguous input; the router inventing a route (handled by the
    fallback); prompt injection steering the route (routes only narrow tools, never
    widen them).

Sources
    Anthropic, "Building effective agents" (2024), "Workflow: routing". Ong et al.,
    "RouteLLM: Learning to Route LLMs with Preference Data" (2024). See also
    agent-model-router (model + effort per request) and crewai-hierarchical-planning
    (a Flow router in front of a crew).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import act
from agentic_architectures.core.runtime import run_pattern

NAME = "routing"


@dataclass(frozen=True)
class Route:
    name: str
    description: str
    instructions: str
    tools: tuple[str, ...] = ()


DEFAULT_ROUTES = (
    Route(
        "math",
        "arithmetic, units, numbers to compute",
        "You solve quantitative questions. Always compute with the calculator.",
        ("calculator",),
    ),
    Route(
        "lookup",
        "facts that are in the local documents",
        "You answer from the documents. Search first, cite the [doc-id].",
        ("search", "read_file"),
    ),
    Route("write", "explanations, summaries, prose", "You write clear, short explanations."),
    Route("general", "anything else", "You are a helpful assistant. Answer briefly."),
)
FALLBACK = "general"


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    routes: tuple[Route, ...] = DEFAULT_ROUTES,
) -> tuple[str, str, dict[str, Any]]:
    by_name = {r.name: r for r in routes}
    menu = "\n".join(f"- {r.name}: {r.description}" for r in routes)
    choice = rt.llm_json(
        "router",
        f"Classify the request into exactly one route.\nRoutes:\n{menu}\n"
        'Reply with JSON: {"route": "<name>", "reason": "..."}',
        untrusted("request", task),
    )
    picked = as_str(choice.get("route")).strip().lower()
    route = by_name.get(picked) or by_name.get(FALLBACK) or routes[-1]
    rt.note("route", route.name, f"router said {picked!r}" + ("" if picked == route.name else " (fallback)"))
    answer, status, actions = act(
        rt,
        f"route:{route.name}",
        route.instructions,
        f"Task: {task}",
        tools=rt.tools.subset(route.tools),
        partial=partial,
    )
    return (
        answer,
        status,
        {"route": route.name, "router_said": picked, "tools": [a.tool for a in actions if a.tool]},
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
