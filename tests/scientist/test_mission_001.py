"""Mission 001 headless verification: discover gravity, cross-verify, persist.

The full scientific method, no UI: plan (3 independent drop heights) ->
execute (physics runs the world) -> observe (measurement-only records) ->
hypothesize (fit + derive g) -> cross-verify (agreement across independent
experiments) -> publish (knowledge base persists; future missions do not
re-discover).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from pymo.scientist import (
    ExperimentSpec,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
)
from pymo.universes import load_universe


@pytest.fixture(scope="module")
def lab():
    return Laboratory(load_universe("universe_001"))


def test_universe_truth_loaded_physics_side(lab):
    # Human/benchmark-facing check: the physics side holds the real values.
    summary = lab.truth_summary()
    assert summary.startswith("universe=Unknown Planet")
    assert "gravity=9.81" in summary


def test_observation_record_is_measurement_only(lab):
    record = lab.run_experiment(ExperimentSpec(kind="drop", drop_height=10.0))
    assert record.field_names == ("t", "z")
    assert len(record.t) > 30
    assert record.z[0] > 9.5            # starts near the drop height
    assert record.z[-1] > 0.5           # clean free-fall segment (pre-contact)
    assert record.z[-1] < record.z[0]   # actually fell


def test_mission_discovers_gravity(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "knowledge.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    report = agent.run_mission(Mission(
        id="001", title="Discover Gravity",
        objective="Determine the gravitational acceleration of Unknown Planet",
        target="gravity", unit="m/s^2",
    ))
    assert report.status == "DISCOVERED"
    assert report.value == pytest.approx(9.81, rel=1e-3)
    assert report.confidence >= 0.999
    assert report.knowledge_saved

    data = json.loads((tmp_path / "knowledge.json").read_text(encoding="utf-8"))
    assert data["universe"] == "universe_001"
    assert data["laws"][0]["name"] == "gravity"
    assert data["laws"][0]["confidence"] >= 0.999
    assert len(data["laws"][0]["experiments"]) == 3


def test_knowledge_persists_and_prevents_rediscovery(lab, tmp_path):
    path = tmp_path / "knowledge.json"
    mission = Mission(id="001", title="Discover Gravity", objective="",
                      target="gravity", unit="m/s^2")

    agent = ScientistAgent(lab, KnowledgeBase(path, universe="universe_001"))
    first = agent.run_mission(mission)
    assert first.status == "DISCOVERED"

    # A fresh agent (new session) reads the saved knowledge...
    kb2 = KnowledgeBase(path, universe="universe_001")
    assert kb2.knows("gravity")
    agent2 = ScientistAgent(lab, kb2)
    second = agent2.run_mission(mission)

    # ...and does NOT re-run experiments — civilization accumulates.
    assert second.status == "CONCLUDED_FROM_KNOWLEDGE"
    assert second.experiments_run == 0
    assert second.value == pytest.approx(9.81, rel=1e-3)


def test_cross_experiment_consistency(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    report = agent.run_mission(Mission(
        id="001", title="Discover Gravity", objective="",
        target="gravity", unit="m/s^2",
    ))
    assert len(report.estimates) == 3
    spread = (max(report.estimates) - min(report.estimates)) / abs(report.value)
    assert spread < 0.001    # same g from 5m, 10m, 20m — that is a law
