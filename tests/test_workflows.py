"""Workflows: augmented LLM, chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer."""

from __future__ import annotations

import time

import pytest

from agentic_architectures.patterns import augmented
from agentic_architectures.patterns import chaining
from agentic_architectures.patterns import evaluator_optimizer
from agentic_architectures.patterns import orchestrator_workers
from agentic_architectures.patterns import parallelization
from agentic_architectures.patterns import routing
from agentic_architectures.patterns.memory import LongTermMemory
from tests.conftest import MakeRt
from tests.conftest import j

CALC = j(tool="calculator", args={"expression": "6*7"})


# ----------------------------------------------------------------- augmented
def test_augmented_injects_retrieval_and_memory_then_writes_memory(make_rt: MakeRt) -> None:
    rt, model = make_rt({"augmented": [CALC, j(answer="42")]})
    memory = LongTermMemory()
    memory.add("u1", "the user likes ollama default port trivia")
    result = augmented.run(rt, "What is the ollama default port?", memory=memory, user_id="u1")
    assert (result.answer, result.status) == ("42", "answered")
    first = model.calls[0].prompt
    assert 'source="doc:ollama"' in first and 'source="memory"' in first
    assert 'source="tool:calculator"' in model.calls[1].prompt
    assert result.data["tools"] == ["calculator"] and "ollama" in result.data["retrieved"]
    assert any(f.startswith("Asked: What is the ollama") for f in memory.all("u1"))


def test_augmented_tool_rounds_are_capped(make_rt: MakeRt) -> None:
    rt, model = make_rt({"augmented": [CALC, j(tool="calculator", args={"expression": "1"})]})
    result = augmented.run(rt, "loop", max_tool_rounds=1)
    assert result.status == "max_steps" and len(model.calls) == 2


# ----------------------------------------------------------------- chaining
def test_chaining_passes_outputs_down_and_retries_a_failed_gate(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {"chain:outline": ["no bullets", "- a\n- b"], "chain:draft": "Draft text.", "chain:polish": "Final."}
    )
    result = chaining.run(rt, "task")
    assert (result.answer, result.status) == ("Final.", "completed")
    assert model.labels() == ["chain:outline", "chain:outline", "chain:draft", "chain:polish"]
    assert "failed a check: needs at least 2 bullet points" in model.calls[1].prompt
    assert "- a\n- b" in model.calls[2].prompt and "Draft text." in model.calls[3].prompt


def test_chaining_stops_at_a_gate_that_keeps_failing(make_rt: MakeRt) -> None:
    rt, model = make_rt({"chain:outline": "- a\n- b", "chain:draft": "x", "chain:polish": "TODO later"})
    result = chaining.run(rt, "task")
    assert result.status == "gate_failed:polish" and result.answer == "x"
    assert model.count("chain:polish") == 2
    assert not result.ok
    assert chaining.max_chars(5)("") == "empty output"


# ----------------------------------------------------------------- routing
def test_routing_dispatches_with_least_privilege_tools(make_rt: MakeRt) -> None:
    rt, model = make_rt({"router": j(route="math"), "route:math": [CALC, j(answer="42")]})
    result = routing.run(rt, "6 times 7?")
    assert result.data["route"] == "math" and result.answer == "42"
    system = model.calls[1].messages[0].content
    assert "- calculator(" in system and "- search(" not in system and "- write_note(" not in system


@pytest.mark.parametrize("reply", [j(route="hack_the_planet"), "not json", j(other=1)])
def test_routing_falls_back_to_general_on_a_bad_choice(make_rt: MakeRt, reply: str) -> None:
    rt, model = make_rt({"router": reply, "route": j(answer="ok")})
    result = routing.run(rt, "x")
    assert result.data["route"] == "general"
    assert model.labels()[-1] == "route:general"


def test_a_route_cannot_use_tools_it_was_not_given(make_rt: MakeRt) -> None:
    rt, model = make_rt({"router": j(route="write"), "route:write": [CALC, j(answer="done")]})
    routing.run(rt, "x")
    assert "error: unknown tool 'calculator'" in model.calls[-1].prompt


# ----------------------------------------------------------------- parallelization
def test_sectioning_fans_out_in_parallel_and_aggregates(make_rt: MakeRt) -> None:
    rt, model = make_rt({"section": "part", "aggregate": "merged"}, latency=0.1)
    start = time.perf_counter()
    result = parallelization.run(rt, "t", mode="sectioning")
    elapsed = time.perf_counter() - start
    assert result.answer == "merged" and model.count("section") == 3
    assert elapsed < 0.35  # 3 parallel + 1 sequential, not 4 sequential
    assert result.stats["critical_path"] == 2
    assert model.calls[-1].prompt.count('<untrusted source="section:') == 3


def test_voting_takes_the_majority_and_reports_no_majority(make_rt: MakeRt) -> None:
    rt, _ = make_rt({"vote": [j(answer="42"), j(answer="41"), j(answer=" 42 ")]})
    result = parallelization.run(rt, "t", mode="voting", voters=3)
    assert result.status == "majority" and result.answer.strip() == "42"
    assert result.data["agreement"] == pytest.approx(0.667, abs=1e-3)
    rt2, _ = make_rt({"vote": [j(answer="a"), j(answer="b")]})
    assert parallelization.run(rt2, "t", mode="voting", voters=2).status == "no_majority"
    rt3, _ = make_rt({"vote": j(answer="")})
    assert parallelization.run(rt3, "t", mode="voting").status == "no_votes"


def test_parallelization_rejects_bad_options(make_rt: MakeRt) -> None:
    rt, _ = make_rt({})
    with pytest.raises(ValueError):
        parallelization.run(rt, "t", mode="nope")
    with pytest.raises(ValueError):
        parallelization.run(rt, "t", mode="voting", voters=99)


# ----------------------------------------------------------------- orchestrator-workers
def test_orchestrator_caps_and_dedupes_subtasks_and_workers_run_in_parallel(make_rt: MakeRt) -> None:
    subtasks = [{"id": f"s{i}", "instruction": f"do {i % 3}"} for i in range(9)]
    rt, model = make_rt(
        {"orchestrator": j(subtasks=subtasks), "worker": j(answer="done"), "synthesize": "final"},
        latency=0.05,
    )
    result = orchestrator_workers.run(rt, "t", max_subtasks=5)
    assert result.answer == "final"
    assert result.data["subtasks"] == ["s0", "s1", "s2"]  # 5 kept by the cap, 2 duplicates dropped
    assert model.count("worker") == 3
    assert result.stats["critical_path"] == 3


def test_orchestrator_without_a_usable_plan_gives_one_worker_the_task(make_rt: MakeRt) -> None:
    rt, model = make_rt({"orchestrator": "rubbish", "worker": j(answer="w"), "synthesize": "s"})
    result = orchestrator_workers.run(rt, "t")
    assert result.data["subtasks"] == ["whole"] and model.count("worker") == 1


# ----------------------------------------------------------------- evaluator-optimizer
def test_evaluator_loop_stops_at_threshold_and_ignores_a_model_pass_flag(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "generate": ["d1", "d2", "d3"],
            "evaluate": [j(score=3, feedback="fix A", **{"pass": True}), j(score=15, feedback="")],
        }
    )
    result = evaluator_optimizer.run(rt, "t", threshold=8)
    assert (result.answer, result.status) == ("d2", "accepted")
    assert [r["score"] for r in result.data["rounds"]] == [3, 10]
    assert "fix A" in model.calls[2].prompt


def test_evaluator_returns_the_best_draft_when_it_never_converges(make_rt: MakeRt) -> None:
    rt, _ = make_rt({"generate": ["d1", "d2", "d3"], "evaluate": [j(score=5), j(score=7), j(score=2)]})
    result = evaluator_optimizer.run(rt, "t", max_rounds=3)
    assert (result.answer, result.status) == ("d2", "max_rounds")
