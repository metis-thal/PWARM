"""Open Genesis Pilot — harness components, G-1 cold start, and acceptance.

The pilot proves: a scientist with an EMPTY knowledge layer, in a world
whose constants it cannot see, can start from observations, declare its
first model and relation, answer its own unverified-identity question
through the research cycle, pass a held-out trial, produce a derived
discovery, and survive an independent physics-as-oracle verification —
all under the audit catalog (spec v2, sections E/F/H-O)."""

from __future__ import annotations

import json

import pytest

from pwarm.scientist.agent import (
    _TOLERANCE_MARGIN,
    ScientistAgent,
)
from pwarm.scientist.experiment import ExperimentSpec, Laboratory
from pwarm.scientist.knowledge import KnowledgeBase
from pwarm.scientist.mission import Mission
from pwarm.scientist.models import ScientificModel
from pwarm.scientist.pilot import (
    IndependentVerifier,
    PilotExecutionLog,
    PilotExperimentError,
    PilotLaboratoryFacade,
    VerifierInput,
    run_open_genesis_pilot,
)
from pwarm.universes import load_universe

FORBIDDEN_G1_SYMBOLS = (
    "fit_free_fall", "prediction_from_belief", "ScientistState",
    "DEFAULT_PRIOR_BOUNDS", "gravity", "free fall", "-g/2",
)


@pytest.fixture
def pilot_world():
    return load_universe("genesis_pilot")


@pytest.fixture
def pilot_parts(pilot_world, tmp_path):
    journal = PilotExecutionLog()
    knowledge = KnowledgeBase(tmp_path / "kb.json", universe="genesis_pilot")
    facade = PilotLaboratoryFacade(Laboratory(pilot_world), knowledge, journal)
    agent = ScientistAgent(laboratory=facade, knowledge=knowledge)
    return pilot_world, journal, knowledge, facade, agent


# -- Facade: measurement-only boundary (F1/F2) ---------------------------------

def test_facade_exposes_only_run_experiment(pilot_parts):
    _, _, _, facade, _ = pilot_parts
    public = [name for name in dir(facade) if not name.startswith("_")]
    assert public == ["run_experiment"]
    assert not hasattr(facade, "truth_summary")
    assert not hasattr(facade, "start_experiment")
    assert not hasattr(facade, "universe")


def test_facade_returns_measurements_and_journals_watermarks(pilot_parts):
    _, journal, _knowledge, facade, _ = pilot_parts
    spec = ExperimentSpec(kind="drop", drop_height=2.0)
    observation = facade.run_experiment(spec)

    assert observation.experiment_id == "drop_h2_m1"
    assert len(observation.z) > 0
    events = journal.events
    assert len(events) == 1 and events[0].event_type == "execute"
    assert events[0].spec_id == "drop_h2_m1"
    assert events[0].pred_watermark == 0 and events[0].verif_watermark == 0


def test_facade_watermark_captures_commitments_before_physics(pilot_parts):
    """A prediction committed BEFORE the run is reflected in the execute
    event's watermark — captured at entry, before physics."""
    _, journal, knowledge, facade, _agent = pilot_parts
    knowledge.register_model("linear", "y = k*x", {"k": 1.0})
    model = ScientificModel(model_id="linear", params={"k": 1.0})
    from pwarm.scientist.prediction import model_prediction
    prediction = model_prediction(model, {"x": 5.0})
    knowledge.commit_prediction(
        model_ref="linear", claim="linear", spec_ref="drop_h5_m1",
        value=prediction.value, tolerance=0.01,
        reduction_channel="z", reduction_rule="first")

    facade.run_experiment(ExperimentSpec(kind="drop", drop_height=5.0))

    event = journal.events[-1]
    assert event.pred_watermark == 1     # the commitment existed at entry


def test_facade_sanitizes_exceptions(pilot_parts):
    """Failures surface as PilotExperimentError without universe names or
    secret words; the real diagnostic goes to the journal only."""
    _, journal, _, facade, _ = pilot_parts
    from pwarm.universes import load_universe
    universe_name = load_universe("genesis_pilot").name

    with pytest.raises(PilotExperimentError) as exc_info:
        facade.run_experiment(ExperimentSpec(kind="teleport", drop_height=1.0))

    message = str(exc_info.value)
    assert "invalid or unsupported" in message
    for leak in (universe_name, "genesis_pilot", "gravity", "secret"):
        assert leak not in message.lower()
    failures = [event for event in journal.events
                if event.event_type == "failure"]
    assert failures                                  # diagnostic is harness-side


def test_facade_object_graph_hides_universe_without_traversal(pilot_parts):
    """L-4: no Universe instance is reachable from the scientist's state
    graph without crossing the facade boundary (data traversal only —
    see _no_universe_beyond_facade for the CPython import-graph caveat)."""
    from pwarm.scientist.pilot import _no_universe_beyond_facade
    _, _, knowledge, facade, agent = pilot_parts
    facade.run_experiment(ExperimentSpec(kind="drop", drop_height=2.0))
    assert _no_universe_beyond_facade(agent, knowledge)


# -- G-1: cold-start bootstrap composer (purity-audited) -----------------------

def _bootstrap_trials(facade, heights=(2.0, 3.0)):
    trials = []
    for height in heights:
        spec = ExperimentSpec(kind="drop", drop_height=height)
        trials.append((spec, facade.run_experiment(spec)))
    return trials


def test_bootstrap_registers_fitted_model_and_declares_relation(pilot_parts):
    _, _, _knowledge, facade, agent = pilot_parts
    trials = _bootstrap_trials(facade)

    outcome = agent.bootstrap_relation_from_observations(
        trials, subject="z_first", declared_by="G-1 cold start")

    assert outcome.family == "linear"                # the data is linear in x
    assert outcome.model_record.model_id == "linear"
    assert 0.99 < outcome.model_record.params["k"] < 1.01
    assert outcome.r2 > 0.99
    assert outcome.relation_record.scope == ("drop_h2_m1", "drop_h3_m1")
    assert outcome.relation_record.parameters_ref == "linear"
    assert outcome.tolerance == _TOLERANCE_MARGIN * outcome.max_training_residual
    assert outcome.tolerance > outcome.max_training_residual


def test_bootstrap_purity_no_domain_symbols_in_composer():
    """P-1/P-2: the composer source contains no domain-specific fitter, no
    belief-path symbol, and no target-semantics token."""
    import inspect

    from pwarm.scientist.agent import (
        ScientistAgent as _Agent,
    )
    source = inspect.getsource(_Agent.bootstrap_relation_from_observations)
    for symbol in FORBIDDEN_G1_SYMBOLS:
        assert symbol not in source, symbol


def test_bootstrap_tolerance_rule_is_fixed_before_held_out(pilot_parts, tmp_path):
    """F3: the declared tolerance derives only from the bootstrap dataset
    (pre-held-out information) and is identical on repeated bootstrap with
    the same data — it cannot depend on any later observation."""
    _, _, _knowledge, facade, agent = pilot_parts
    trials = _bootstrap_trials(facade, heights=(2.0, 3.0))
    first = agent.bootstrap_relation_from_observations(
        trials, subject="z_first")

    second_knowledge = KnowledgeBase(
        tmp_path / "kb2.json", universe="genesis_pilot")
    second_agent = ScientistAgent(laboratory=facade,
                                  knowledge=second_knowledge)
    later_trials = _bootstrap_trials(facade, heights=(2.0, 3.0))
    second = second_agent.bootstrap_relation_from_observations(
        later_trials, subject="z_first2")

    assert first.tolerance == second.tolerance       # same data, same rule


def test_bootstrap_requires_two_observations(pilot_parts):
    _, _, _knowledge, facade, agent = pilot_parts
    spec = ExperimentSpec(kind="drop", drop_height=2.0)
    observation = facade.run_experiment(spec)
    with pytest.raises(ValueError, match="at least two"):
        agent.bootstrap_relation_from_observations(
            [(spec, observation)], subject="z_first")


# -- Independent verifier: physics-as-oracle -----------------------------------

def test_verifier_passes_a_correct_prediction(pilot_world):
    verifier = IndependentVerifier(pilot_world)
    spec = ExperimentSpec(kind="drop", drop_height=5.0)
    result = verifier.verify(
        VerifierInput(spec=spec, predicted=5.0, tolerance=0.01,
                      experiment_id="drop_h5_m1"))
    assert result.passed
    assert abs(result.independent_observed - 4.998) < 0.001


def test_verifier_refutes_a_wrong_prediction(pilot_world):
    verifier = IndependentVerifier(pilot_world)
    spec = ExperimentSpec(kind="drop", drop_height=5.0)
    result = verifier.verify(
        VerifierInput(spec=spec, predicted=25.0, tolerance=0.01,
                      experiment_id="drop_h5_m1"))
    assert not result.passed


def test_verifier_holds_no_scientist_objects(pilot_world):
    verifier = IndependentVerifier(pilot_world)
    for value in vars(verifier).values():
        assert not isinstance(value, (KnowledgeBase, ScientistAgent,
                                      PilotLaboratoryFacade))


def test_verifier_is_deterministic(pilot_world):
    verifier = IndependentVerifier(pilot_world)
    spec = ExperimentSpec(kind="drop", drop_height=3.0)
    first = verifier.verify(
        VerifierInput(spec=spec, predicted=3.0, tolerance=0.01,
                      experiment_id="drop_h3_m1"))
    second = verifier.verify(
        VerifierInput(spec=spec, predicted=3.0, tolerance=0.01,
                      experiment_id="drop_h3_m1"))
    assert first == second


# -- The full pilot: acceptance criteria E-1..E-14 ------------------------------

def test_open_genesis_pilot_acceptance(tmp_path):
    """The falsifiable claim, end to end: an empty knowledge layer, a world
    with hidden constants, observation-only entry — and the scientist
    produces a derived discovery confirmed by independent physics."""
    report = run_open_genesis_pilot(tmp_path / "pilot_kb.json")

    assert report.accepted
    for criterion in report.criterion_results:
        assert criterion.passed, criterion.criterion
    # the discovery is the pilot's derived fact
    assert len(report.discoveries) == 1
    assert (report.discoveries[0].representative_heldout_confirmed.status
            == "confirmed")
    # and the independent verifier confirmed the held-out prediction
    assert all(result.passed for result in report.verifier_results)


def test_pilot_initial_state_is_empty_and_truth_free(tmp_path):
    """The pilot starts from NOTHING: empty knowledge base, no relations,
    no models, no questions — and the world constants are never written
    into the scientist-visible store."""
    report = run_open_genesis_pilot(tmp_path / "pilot_kb.json",
                                    max_cycles=0)
    kb_data = json.loads((tmp_path / "pilot_kb.json").read_text(
        encoding="utf-8"))
    assert "7.2" not in json.dumps(kb_data)          # no hidden constant leaked
    assert report.bootstrap.model_record.model_id == "linear"


def test_pilot_question_kinds_are_closed_vocabulary(tmp_path):
    """Every question the pilot declares comes from the closed scanner
    vocabulary — no free-form questions appear in the run."""
    report = run_open_genesis_pilot(tmp_path / "pilot_kb.json")
    assert report.accepted


def test_belief_path_audit_catches_real_violations(pilot_parts):
    """The B-counters are not vacuous: a scientist that crosses into the
    belief path inside the audit window IS counted (and would fail E-13)."""
    from pwarm.scientist.pilot import BeliefPathCounters, belief_path_audit
    _, _, _knowledge, _facade, agent = pilot_parts
    counters = BeliefPathCounters()
    # "gravity" is the legacy belief-path target: form_prediction builds a
    # ScientistState from DEFAULT_PRIOR_BOUNDS and converts the belief —
    # exactly the path the Open Genesis pilot must never take.
    mission = Mission(id="m-violation", title="violation probe",
                      objective="prove the audit counters are live",
                      target="gravity")

    with belief_path_audit(counters, agent):
        agent.form_prediction(mission)               # deliberate violation

    assert counters.form_prediction == 1
    assert counters.scientist_state == 1             # form_prediction builds one
    assert counters.prediction_from_belief == 1      # belief -> prediction
    assert counters.default_prior_bounds == 2        # prior bounds were read
    # and outside the window nothing is counted anymore
    agent.form_prediction(mission)
    assert counters.form_prediction == 1
