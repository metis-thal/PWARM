"""Mission 003 verification: science under constraints.

The AI must spend a LIMITED budget wisely (value = information / cost, not
raw information), never re-measure what it already knows, and — the payoff —
when density proves UNIDENTIFIABLE with drop/slide apparatus, analyze the
gap, FILE an instrument request, receive the granted Fluid Tank, and identify
density with the new buoyancy experiment under the EXTENDED budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from pymo.scientist import (
    REGISTRY,
    ExperimentBudget,
    ExperimentDesigner,
    ExperimentSpec,
    InstrumentCatalog,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
    ScientistState,
    analyze_gap,
)
from pymo.universes import load_universe

MISSION = Mission(id="003", title="Science Under Constraints",
                  objective="Characterize Constrained World within budget",
                  target="material_A", unit="", min_confidence=0.95)


def _universe():
    return load_universe("universe_003")


def _budget_from_universe() -> ExperimentBudget:
    cfg = _universe().budget
    return ExperimentBudget(
        experiments=int(cfg["experiments"]),
        simulation_steps=int(cfg["simulation_steps"]),
        compute_cost=float(cfg["compute_cost"]))


@pytest.fixture(scope="module")
def lab():
    return Laboratory(_universe())


def test_budget_ranks_by_value_not_raw_information(lab):
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_003")
    state = ScientistState(_universe().manifest, knowledge)
    proposal = ExperimentDesigner().choose(state)
    assert proposal is not None
    # A 50 m drop has the best resolution, but a 10 m drop already crosses
    # the knowledge threshold at ~1/4 the cost: value, not gain, decides.
    assert proposal.design["height"] == 10.0
    assert proposal.cost == pytest.approx(1.5)
    assert "value" in proposal.reason and "cost" in proposal.reason


def test_unaffordable_candidates_are_refused_not_chosen(lab):
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_003")
    state = ScientistState(_universe().manifest, knowledge)
    designer = ExperimentDesigner()
    budget = ExperimentBudget(experiments=1, simulation_steps=1000,
                              compute_cost=0.9)   # cheapest design costs 1.0
    assert designer.available_experiments(state)   # candidates exist ...
    assert designer.choose(state, budget) is None  # ... but none affordable


def test_buoyancy_test_measures_density(lab):
    record = lab.run_experiment(
        ExperimentSpec(kind="buoyancy_test", drop_height=2.0,
                       material="material_A"))
    hypothesis, _ = REGISTRY["buoyancy_test"].derive(record, 9.79, "material_A")
    assert hypothesis.value == pytest.approx(7800.0, rel=0.05)
    assert hypothesis.r2 > 0.99


def test_buoyancy_requires_known_gravity(lab):
    record = lab.run_experiment(
        ExperimentSpec(kind="buoyancy_test", drop_height=2.0,
                       material="material_A"))
    with pytest.raises(ValueError):
        REGISTRY["buoyancy_test"].derive(record, None, "material_A")


def test_gap_analysis_names_the_missing_instrument():
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_003")
    state = ScientistState(_universe().manifest, knowledge)
    state.mark_unidentifiable(
        "material_A.density",
        "no available experiment informs this parameter")
    request = analyze_gap(state)
    assert request is not None
    assert request.instrument == "fluid_tank"
    assert request.capability == "fluid_immersion"
    assert "material_A.density" in request.target_claims
    # A known parameter never triggers a request.
    state2 = ScientistState(_universe().manifest, knowledge)
    assert analyze_gap(state2) is None


def test_catalog_grants_once_and_refuses_mismatches():
    catalog = InstrumentCatalog(_universe().instruments)
    knowledge = KnowledgeBase(Path(tmp_knowledge()), universe="universe_003")
    state = ScientistState(_universe().manifest, knowledge)
    state.mark_unidentifiable("material_A.density", "gap")
    request = analyze_gap(state)
    grant = catalog.grant(request)
    assert grant is not None
    assert grant.enables == "buoyancy_test"
    assert grant.budget["experiments"] == 2
    assert catalog.granted == ("fluid_tank",)
    assert catalog.grant(request) is None          # one grant only
    from pymo.scientist import InstrumentRequest
    assert catalog.grant(InstrumentRequest(
        capability="telepathy", instrument="fluid_tank",
        target_claims=("x",), observable="", reason="")) is None


def test_constrained_mission_full_arc(lab, tmp_path):
    """The Mission 003 story, end to end:

    budget forces value-ranked choices -> restitution + friction + gravity
    established -> density UNIDENTIFIABLE -> instrument request -> Fluid
    Tank granted + budget extended -> buoyancy identifies density.
    """
    knowledge = KnowledgeBase(tmp_path / "knowledge.json",
                              universe="universe_003")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(_universe().manifest, knowledge)
    budget = _budget_from_universe()
    catalog = InstrumentCatalog(_universe().instruments)

    report = agent.run_constrained_mission(MISSION, state, budget,
                                           catalog=catalog)

    assert report.status == "DISCOVERED"
    assert report.knowledge_saved

    # The instrument arc actually happened, exactly once.
    assert catalog.granted == ("fluid_tank",)
    assert "fluid_tank" in report.summary

    # Gravity established as the free-fall byproduct (9.79, NOT 9.81).
    assert state.belief("gravity").status == "known"
    assert state.belief("gravity").midpoint == pytest.approx(9.79, rel=2e-3)

    # All six material properties published — density included, which was
    # impossible in Mission 002's drop/slide world.
    for material, truth in (("material_A", {"density": 7800.0,
                                            "restitution": 0.72,
                                            "friction": 0.40}),
                            ("material_B", {"density": 2700.0,
                                            "restitution": 0.45,
                                            "friction": 0.25})):
        props = knowledge.material(material)
        assert props is not None, material
        for prop, value in truth.items():
            assert props[prop] == pytest.approx(value, rel=0.10), (
                material, prop)

    # Budget discipline: the base envelope plus the grant, never more.
    assert budget.used_experiments <= 4 + 2
    assert budget.used_cost <= budget.compute_cost + 1e-9
    assert budget.used_steps <= budget.simulation_steps


def test_constrained_mission_without_catalog_stays_honest(lab, tmp_path):
    """No instrumentation director: density stays UNIDENTIFIABLE and the
    scientist never guesses — the Mission 002 outcome, now under budget."""
    knowledge = KnowledgeBase(tmp_path / "knowledge.json",
                              universe="universe_003")
    agent = ScientistAgent(lab, knowledge)
    state = ScientistState(_universe().manifest, knowledge)
    budget = _budget_from_universe()

    report = agent.run_constrained_mission(MISSION, state, budget)

    assert report.status == "DISCOVERED"          # contact props established
    assert state.belief("material_A.density").status == "unidentifiable"
    assert state.belief("material_B.density").status == "unidentifiable"
    assert "density" not in (knowledge.material("material_A") or {})
    # The locked instrument was never granted.
    assert budget.used_experiments <= 4


def tmp_knowledge() -> str:
    import tempfile
    return str(Path(tempfile.gettempdir()) / "m003_designer.json")
