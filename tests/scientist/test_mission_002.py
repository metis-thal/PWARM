"""Mission 002 verification: autonomous scientific experimentation.

The AI must decide WHAT to test (information gain), respect scientific
dependencies (friction needs gravity), discover material properties through
its own experiment designs, and honestly report what it CANNOT know
(density is unidentifiable in a gravity+contact world — equivalence
principle).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from pymo.scientist import (
    REGISTRY,
    ExperimentDesigner,
    ExperimentSpec,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
    ScientistState,
)
from pymo.universes import load_universe

MISSION = Mission(id="002", title="Material Discovery",
                  objective="Characterize the materials of Material World",
                  target="material_A", unit="", min_confidence=0.95)


@pytest.fixture(scope="module")
def lab():
    return Laboratory(load_universe("universe_002"))


def test_designer_prefers_the_most_informative_design(lab):
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_002")
    state = ScientistState(load_universe("universe_002").manifest, knowledge)
    proposal = ExperimentDesigner().choose(state)
    assert proposal is not None
    # A taller drop resolves restitution better (eps ~ 1/sqrt(h)).
    assert proposal.design["height"] == 50.0
    assert "reduce" in proposal.reason and "gain" in proposal.reason


def test_slide_design_requires_gravity_first(lab):
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_002")
    state = ScientistState(load_universe("universe_002").manifest, knowledge)
    # Gravity unknown: friction claims are unreachable (mu = -slope/g).
    scored_claims = set()
    designer = ExperimentDesigner()
    for _ in range(12):
        proposal = designer.choose(state)
        if proposal is None:
            break
        scored_claims.add(proposal.claim)
        break  # only the first choice matters for this assertion
    assert "material_A.friction" not in scored_claims


def test_drop_test_measures_restitution(lab):
    record = lab.run_experiment(
        ExperimentSpec(kind="drop_test", drop_height=10.0, material="material_A"))
    hypothesis, fitted_g = REGISTRY["drop_test"].derive(record, None, "material_A")
    assert hypothesis.value == pytest.approx(0.72, rel=0.05)
    assert fitted_g == pytest.approx(9.81, rel=1e-3)
    assert hypothesis.r2 > 0.99


def test_slide_test_measures_friction(lab):
    record = lab.run_experiment(
        ExperimentSpec(kind="slide_test", material="material_A", v0=5.0))
    hypothesis, _ = REGISTRY["slide_test"].derive(record, 9.81, "material_A")
    assert hypothesis.value == pytest.approx(0.40, rel=0.10)
    assert hypothesis.r2 > 0.95


def test_adaptive_mission_discovers_materials(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "knowledge.json", universe="universe_002")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(load_universe("universe_002").manifest, knowledge)

    report = agent.run_adaptive_mission(MISSION, state)

    assert report.status == "DISCOVERED"
    assert report.knowledge_saved
    mat_a = knowledge.material("material_A")
    assert mat_a is not None
    assert mat_a["restitution"] == pytest.approx(0.72, rel=0.05)
    assert mat_a["friction"] == pytest.approx(0.40, rel=0.10)
    mat_b = knowledge.material("material_B")
    assert mat_b["restitution"] == pytest.approx(0.45, rel=0.05)
    assert mat_b["friction"] == pytest.approx(0.25, rel=0.10)
    # Density is NOT in the published properties: the scientist never
    # claims what it cannot measure.
    assert "density" not in mat_a and "density" not in mat_b


def test_density_reported_unidentifiable(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "knowledge.json", universe="universe_002")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(load_universe("universe_002").manifest, knowledge)
    agent.run_adaptive_mission(MISSION, state)
    belief = state.belief("material_A.density")
    assert belief.status == "unidentifiable"
    assert "density" in state.report() and "unidentifiable" in state.report()


def test_gravity_discovered_as_experiment_byproduct(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "knowledge.json", universe="universe_002")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(load_universe("universe_002").manifest, knowledge)
    assert state.belief("gravity").status == "unknown"
    agent.run_adaptive_mission(MISSION, state)
    # The first drop's free-fall fit established gravity for free.
    assert state.belief("gravity").status == "known"
    assert state.belief("gravity").midpoint == pytest.approx(9.81, rel=1e-3)


def test_second_mission_concludes_from_knowledge(lab, tmp_path):
    path = tmp_path / "knowledge.json"
    knowledge = KnowledgeBase(path, universe="universe_002")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(load_universe("universe_002").manifest, knowledge)
    agent.run_adaptive_mission(MISSION, state)

    # Fresh state, same knowledge: every interval already collapsed.
    state2 = ScientistState(load_universe("universe_002").manifest, knowledge)
    report2 = agent.run_adaptive_mission(MISSION, state2)
    assert report2.experiments_run == 0
    assert knowledge.material("material_A") is not None


def tmp_knowledge() -> str:
    import tempfile
    return str(Path(tempfile.gettempdir()) / "m002_designer.json")
