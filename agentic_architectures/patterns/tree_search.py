"""Tree search (LATS-lite): expand candidate answers, score them, search the tree by UCT.

The root is the task. Each expansion selects the most promising node by UCT
(exploitation: mean value; exploration: rarely visited nodes), asks the model for
``branching`` candidate continuations of that node in one call, scores each child with
a value call (0-10 plus a one-line reflection) in parallel, and backpropagates the score
to every ancestor. The search stops when a child reaches ``threshold`` or after
``max_expansions``; the best-scoring node wins, wherever it is in the tree.

When to use
    Hard problems with a usable value signal where the first idea is often wrong:
    puzzles, planning, code with tests, multi-step reasoning. Use sparingly.

Trade-offs
    Explores alternatives instead of committing to one path, and can recover from a
    bad first step. But it is the most expensive pattern here, and an LLM value
    function is noisy: with no reliable score, search adds cost, not quality.

Cost profile
    Per expansion: 1 expand call + ``branching`` value calls. With branching 3 and 2
    expansions: 8 calls; critical path 2 per expansion.

Failure modes
    A flat value function (every child scores 5: search degenerates to random); the
    tree exceeding the budget (bounded by ``max_expansions`` and ``max_depth``);
    duplicate children (de-duplicated).

Sources
    Zhou et al., "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in
    Language Models" (LATS, ICML 2024). Yao et al., "Tree of Thoughts" (NeurIPS 2023).
    Kocsis & Szepesvári, "Bandit based Monte-Carlo Planning" (UCT, 2006). See
    langgraph-reflection-agent (LATS-lite in LangGraph).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from agentic_architectures.core.parsing import as_int
from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "tree_search"
EXPAND = (
    "Propose {n} different candidate answers to the task, each improving on the parent "
    'attempt and its reflection if there is one. Reply with JSON: {{"candidates": ["...", ...]}}'
)
VALUE = (
    "Score how well the candidate solves the task, 0-10, and reflect in one line on what is "
    'missing. Reply with JSON: {"score": <0-10>, "reflection": "..."}'
)


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int
    text: str
    score: int = 0
    reflection: str = ""
    visits: int = 0
    value: float = 0.0
    children: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0


def uct(node: Node, parent_visits: int, c: float) -> float:
    if node.visits == 0:
        return math.inf
    return node.mean + c * math.sqrt(math.log(max(parent_visits, 1)) / node.visits)


def select(tree: dict[int, Node], max_depth: int, c: float) -> Node | None:
    """Walk from the root by UCT to an expandable node (one with no children yet)."""
    node = tree[0]
    while node.children:
        options = [
            tree[i]
            for i in node.children
            if tree[i].depth < max_depth and _expandable(tree, tree[i], max_depth)
        ]
        if not options:
            return None
        node = max(options, key=lambda n: (uct(n, node.visits, c), -n.id))
    return node if node.depth < max_depth else None


def _expandable(tree: dict[int, Node], node: Node, max_depth: int) -> bool:
    if node.depth >= max_depth:
        return False
    if not node.children:
        return True
    return any(_expandable(tree, tree[i], max_depth) for i in node.children)


def backpropagate(tree: dict[int, Node], node: Node, reward: float) -> None:
    current: Node | None = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = tree[current.parent] if current.parent is not None else None


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    branching: int = 3,
    max_depth: int = 3,
    max_expansions: int = 3,
    threshold: int = 9,
    exploration: float = 1.0,
) -> tuple[str, str, dict[str, Any]]:
    tree: dict[int, Node] = {0: Node(0, None, 0, "", visits=1)}
    best: Node | None = None
    status = "max_expansions"
    for expansion in range(1, max_expansions + 1):
        rt.budget.step()
        parent = select(tree, max_depth, exploration)
        if parent is None:
            status = "exhausted"
            break
        context = f"Task: {task}"
        if parent.id:
            context += "\n\nParent attempt:\n" + untrusted("parent", parent.text)
            context += "\nReflection:\n" + untrusted("reflection", parent.reflection)
        data = rt.llm_json("expand", EXPAND.format(n=branching), context)
        texts: list[str] = []
        for c in as_list(data.get("candidates"), limit=branching):
            text = as_str(c, limit=2000).strip()
            if text and text not in texts:
                texts.append(text)
        children = []
        for text in texts:
            node = Node(len(tree), parent.id, parent.depth + 1, text)
            tree[node.id] = node
            parent.children.append(node.id)
            children.append(node)

        def score(node: Node) -> tuple[int, str]:
            verdict = rt.llm_json(
                "value", VALUE, f"Task: {task}\n\nCandidate:\n{untrusted('candidate', node.text)}"
            )
            return as_int(verdict.get("score"), 0, low=0, high=10), as_str(
                verdict.get("reflection"), limit=500
            )

        for node, (value, reflection) in zip(children, rt.parallel(score, children), strict=True):
            node.score, node.reflection = value, reflection
            backpropagate(tree, node, value / 10)
            if best is None or node.score > best.score:
                best = node
                partial.offer(node.text)
        rt.note(
            "search",
            f"expansion {expansion}",
            f"node {parent.id} -> " + ", ".join(f"{n.id}:{n.score}" for n in children),
        )
        if best is not None and best.score >= threshold:
            status = "solved"
            break
    answer = best.text if best else ""
    return (
        answer,
        status,
        {
            "nodes": len(tree) - 1,
            "best": {"id": best.id, "depth": best.depth, "score": best.score} if best else None,
            "tree": [
                {"id": n.id, "parent": n.parent, "score": n.score, "visits": n.visits}
                for n in tree.values()
                if n.id
            ],
        },
    )


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
