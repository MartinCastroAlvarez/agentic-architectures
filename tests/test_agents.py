"""Single agents: ReAct, plan-and-execute, ReWOO, Reflexion, tree search."""

from __future__ import annotations

import math
from collections.abc import Sequence

from agentic_architectures.core.model import Message
from agentic_architectures.patterns import plan_execute
from agentic_architectures.patterns import react
from agentic_architectures.patterns import reflexion
from agentic_architectures.patterns import rewoo
from agentic_architectures.patterns import tree_search
from tests.conftest import MakeRt
from tests.conftest import j

CALC = j(thought="compute", tool="calculator", args={"expression": "6*7"})


# ----------------------------------------------------------------- ReAct
def test_react_thinks_acts_observes_and_answers(make_rt: MakeRt) -> None:
    rt, model = make_rt({"react": [CALC, j(thought="done", answer="42")]})
    result = react.run(rt, "6 x 7?")
    assert (result.answer, result.status) == ("42", "answered")
    assert result.data == {"steps": 2, "thoughts": ["compute", "done"], "tools": ["calculator"]}
    assert '<untrusted source="tool:calculator">\n42\n</untrusted>' in model.calls[1].prompt


def test_react_is_bounded_and_reports_bad_tools_to_the_model(make_rt: MakeRt) -> None:
    rt, model = make_rt({"react": j(tool="rm_rf", args={})})
    result = react.run(rt, "t", max_steps=3)
    assert result.status == "max_steps" and len(model.calls) == 3
    assert "error: unknown tool 'rm_rf'" in model.calls[1].prompt
    rt2, _ = make_rt({"react": "The answer is plain text."})
    assert react.run(rt2, "t").answer == "The answer is plain text."


# ----------------------------------------------------------------- plan-and-execute
def test_plan_execute_finishes_early_when_the_replanner_has_the_answer(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "planner": j(steps=["compute 6*7", "write it up", "polish"]),
            "executor": [CALC, j(answer="42")],
            "replanner": j(answer="It is 42."),
        }
    )
    result = plan_execute.run(rt, "t")
    assert (result.answer, result.status) == ("It is 42.", "answered")
    assert result.data["plan"] == ["compute 6*7"]
    assert model.labels() == ["planner", "executor", "executor", "replanner"]
    assert "Current step: compute 6*7" in model.calls[1].prompt


def test_plan_execute_replans_and_is_bounded(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "planner": j(steps=["a"]),
            "executor": j(answer="r"),
            "replanner": [j(steps=["b", "c"]), j(steps=["c"]), j(steps=["x"]), j(steps=["y"])],
        }
    )
    result = plan_execute.run(rt, "t", max_replans=1, max_steps=4)
    assert result.data["replans"] == 1
    assert result.data["plan"] == ["a", "b", "c"]
    assert result.status == "plan_exhausted"
    assert "step:2" in model.calls[-1].prompt


def test_plan_execute_without_a_plan_runs_the_task_as_one_step(make_rt: MakeRt) -> None:
    rt, _ = make_rt({"planner": "?", "executor": j(answer="r"), "replanner": j(answer="done")})
    result = plan_execute.run(rt, "the task")
    assert result.data["plan"] == ["the task"]


# ----------------------------------------------------------------- ReWOO
def test_rewoo_uses_two_llm_calls_and_substitutes_evidence(make_rt: MakeRt) -> None:
    plan = [
        {"id": "E1", "tool": "calculator", "args": {"expression": "6*7"}},
        {"id": "E2", "tool": "calculator", "args": {"expression": "2*3"}},
        {"id": "E3", "tool": "calculator", "args": {"expression": "#E1 + #E2"}},
    ]
    rt, model = make_rt({"rewoo:planner": j(plan=plan), "rewoo:solver": "48"})
    result = rewoo.run(rt, "t")
    assert result.answer == "48" and len(model.calls) == 2
    assert result.data["levels"] == 2  # E1 and E2 in parallel, then E3
    assert '<untrusted source="evidence:E3">\n48\n</untrusted>' in model.calls[1].prompt
    assert result.stats["tool_calls"] == 3


def test_rewoo_refuses_forward_references_before_running_anything(make_rt: MakeRt) -> None:
    plan = [{"id": "E1", "tool": "calculator", "args": {"expression": "#E2"}}]
    rt, model = make_rt({"rewoo:planner": j(plan=plan)})
    result = rewoo.run(rt, "t")
    assert result.status == "invalid_plan" and len(model.calls) == 1 and result.stats["tool_calls"] == 0
    rt2, _ = make_rt({"rewoo:planner": j(plan=["nope"])})
    assert rewoo.run(rt2, "t").status == "invalid_plan"


def test_rewoo_sequential_mode_and_tool_errors_become_evidence(make_rt: MakeRt) -> None:
    plan = [
        {"id": "E1", "tool": "missing", "args": {}},
        {"id": "E2", "tool": "calculator", "args": {"expression": "1"}},
    ]
    rt, model = make_rt({"rewoo:planner": j(plan=plan), "rewoo:solver": "s"})
    result = rewoo.run(rt, "t", parallel=False)
    assert result.data["levels"] == 2
    assert "error: unknown tool" in model.calls[1].prompt


# ----------------------------------------------------------------- Reflexion
def test_reflexion_learns_a_lesson_and_keeps_it_for_the_next_run(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {"actor": [j(answer="41"), j(answer="42")], "reflect": "Lesson: compute, don't guess."}
    )
    lessons: list[str] = []
    result = reflexion.run(rt, "6 x 7?", evaluator=reflexion.contains("42"), lessons=lessons)
    assert result.status == "passed" and result.answer == "42"
    assert [t["passed"] for t in result.data["trials"]] == [False, True]
    assert "Lesson: compute" in model.calls[2].prompt
    assert lessons == ["Lesson: compute, don't guess."]
    rt2, model2 = make_rt({"actor": j(answer="42")})
    again = reflexion.run(rt2, "6 x 7?", evaluator=reflexion.contains("42"), lessons=lessons)
    assert again.data["recalled"] == 1 and "Lesson: compute" in model2.calls[0].prompt


def test_reflexion_stops_after_max_trials_and_can_self_evaluate(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {"actor": j(answer="nope"), "reflect": "L", "reflexion:evaluate": j(feedback="wrong")}
    )
    result = reflexion.run(rt, "t", max_trials=2)
    assert result.status == "max_trials"
    assert model.labels() == ["actor", "reflexion:evaluate", "reflect", "actor", "reflexion:evaluate"]
    rt2, _ = make_rt({"actor": j(answer="ok"), "reflexion:evaluate": j(**{"pass": True})})
    assert reflexion.run(rt2, "t").status == "passed"


# ----------------------------------------------------------------- tree search
def test_uct_backpropagation_and_selection() -> None:
    tree = {0: tree_search.Node(0, None, 0, "", visits=1)}
    a = tree_search.Node(1, 0, 1, "a")
    b = tree_search.Node(2, 0, 1, "b")
    tree.update({1: a, 2: b})
    tree[0].children = [1, 2]
    assert tree_search.uct(a, 1, 1.0) == math.inf
    tree_search.backpropagate(tree, a, 0.9)
    tree_search.backpropagate(tree, b, 0.1)
    assert tree[0].visits == 3 and a.mean == 0.9
    assert tree_search.select(tree, max_depth=3, c=0.1) is a
    assert tree_search.select(tree, max_depth=1, c=0.1) is None


def test_tree_search_expands_the_best_node_and_stops_when_solved(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "expand": [j(candidates=["bad", "partial", "bad"]), j(candidates=["best", "partial2"])],
            "value": lambda m: j(score={"bad": 1, "partial": 6, "best": 10, "partial2": 5}[_cand(m)]),
        }
    )
    result = tree_search.run(rt, "t", branching=3)
    assert (result.answer, result.status) == ("best", "solved")
    assert result.data["best"] == {"id": 3, "depth": 2, "score": 10}
    assert 'source="parent">\npartial' in model.calls[3].prompt  # the second expansion grew the 6-point node
    assert result.data["nodes"] == 4  # duplicates ("bad" twice) dropped
    assert result.stats["critical_path"] == 4


def test_tree_search_is_bounded_and_keeps_the_best_node(make_rt: MakeRt) -> None:
    rt, model = make_rt({"expand": lambda m: j(candidates=[f"c{len(m[-1].content)}"]), "value": j(score=3)})
    result = tree_search.run(rt, "t", max_expansions=3, branching=1)
    assert result.status == "max_expansions" and model.count("expand") == 3
    rt2, _ = make_rt({"expand": j(candidates=["only"]), "value": j(score=1)})
    assert tree_search.run(rt2, "t", max_depth=1, max_expansions=5).status == "exhausted"


def _cand(messages: Sequence[Message]) -> str:
    text = messages[-1].content
    return text.split('source="candidate">\n')[1].split("\n</untrusted>")[0]
