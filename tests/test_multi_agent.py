"""Multi-agent: supervisor (flat and hierarchical), swarm handoffs, debate with a judge."""

from __future__ import annotations

import pytest

from agentic_architectures.patterns import debate
from agentic_architectures.patterns import supervisor
from agentic_architectures.patterns import swarm
from tests.conftest import MakeRt
from tests.conftest import j


# ----------------------------------------------------------------- supervisor
def test_supervisor_delegates_in_order_and_finishes(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "supervisor": [
                j(next="analyst", instruction="compute 6*7"),
                j(next="writer", instruction="write it"),
                j(next="FINISH", answer="It is 42."),
            ],
            "worker:analyst": [j(tool="calculator", args={"expression": "6*7"}), j(answer="42")],
            "worker:writer": j(answer="Forty-two."),
        }
    )
    result = supervisor.run(rt, "t")
    assert (result.answer, result.status, result.data["turns"]) == (
        "It is 42.",
        "finished",
        ["analyst", "writer"],
    )
    analyst_system = model.calls[1].messages[0].content
    assert "- calculator(" in analyst_system and "- search(" not in analyst_system
    assert 'source="worker:analyst">\n42' in model.calls[4].prompt


def test_supervisor_refuses_unknown_workers_and_is_bounded(make_rt: MakeRt) -> None:
    rt, model = make_rt({"supervisor": j(next="root_shell"), "worker": j(answer="x")})
    result = supervisor.run(rt, "t", max_turns=3)
    assert result.status == "max_turns" and model.count("supervisor") == 3
    assert "'root_shell' is not a worker" in model.calls[-1].prompt


def test_hierarchical_teams_have_their_own_supervisor(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "supervisor": [
                j(next="research_team", instruction="find it"),
                j(next="writer"),
                j(next="FINISH"),
            ],
            "supervisor:research_team": [j(next="researcher"), j(next="FINISH", answer="found: 11434")],
            "worker:researcher": j(answer="11434"),
            "worker:writer": j(answer="Port 11434."),
        }
    )
    result = supervisor.run(rt, "t", members=supervisor.HIERARCHICAL)
    assert result.answer == "Port 11434."  # FINISH without an answer keeps the last report
    assert result.data["turns"] == ["research_team", "research_team/researcher", "writer"]
    assert model.labels()[:4] == [
        "supervisor",
        "supervisor:research_team",
        "worker:researcher",
        "supervisor:research_team",
    ]


# ----------------------------------------------------------------- swarm
def test_swarm_hands_off_along_the_allowed_graph(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "swarm:triage": j(handoff="math", note="numbers"),
            "swarm:math": [j(tool="calculator", args={"expression": "6*7"}), j(answer="42")],
        }
    )
    result = swarm.run(rt, "t")
    assert (result.answer, result.data["path"], result.data["handoffs"]) == ("42", ["triage", "math"], 1)
    assert 'source="handoff">\nnumbers' in model.calls[1].prompt
    assert "- calculator(" in model.calls[1].messages[0].content


def test_swarm_refuses_disallowed_handoffs_and_caps_loops(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {"swarm:triage": [j(handoff="admin"), j(handoff="writer")], "swarm:writer": j(handoff="triage")}
    )
    result = swarm.run(rt, "t", max_handoffs=3)
    assert "Handoff to 'admin' is not allowed" in model.calls[1].prompt
    assert result.status == "max_handoffs"
    assert result.data["path"] == ["triage", "writer", "triage", "writer"]
    rt2, _ = make_rt({"swarm:triage": j(tool="search", args={"query": "x"})})
    assert swarm.run(rt2, "t", max_turns=2).status == "max_turns"  # triage has no tools: errors, then the cap


# ----------------------------------------------------------------- debate
def test_debate_rounds_show_peers_and_the_judge_decides(make_rt: MakeRt) -> None:
    rt, model = make_rt(
        {
            "debate:1": j(answer="42", argument="6*7"),
            "debate:2": j(answer="42", argument="six sevens"),
            "debate:3": [j(answer="41", argument="guess"), j(answer="42", argument="convinced")],
            "judge": j(winner=3, answer="42"),
        }
    )
    result = debate.run(rt, "t", debaters=3, rounds=2)
    assert (result.answer, result.status) == ("42", "judged")
    assert result.data["rounds"] == [["42", "42", "41"], ["42", "42", "42"]]
    assert len(model.calls) == 7 and result.stats["critical_path"] == 3
    round2 = [c for c in model.calls if c.label == "debate:3"][1].prompt
    assert 'source="peers"' in round2 and "Debater 1: 42" in round2 and "Debater 3" not in round2


def test_debate_stops_on_consensus_and_falls_back_on_a_bad_judge(make_rt: MakeRt) -> None:
    rt, model = make_rt({"debate": j(answer="42"), "judge": j(winner=9)})
    result = debate.run(rt, "t", debaters=3, rounds=4)
    assert model.count("debate") == 3  # consensus after round 1
    assert (result.answer, result.status) == ("42", "majority_fallback")
    with pytest.raises(ValueError):
        debate.run(rt, "t", debaters=1)
