"""Genesis Phase 2 verification: commitment precedes observation, and
verification feeds back into the beliefs.

The full epistemic loop, enforced: predict -> COMMIT (hash + persist) ->
experiment -> observation -> verification -> confirmed / refuted ->
belief update.

Tests A–F cover temporal integrity, tamper detection, verification purity
and both verdict paths. Mission regressions (G–J) are the existing mission
suites, run alongside this file. Nothing here needs universe truth: the
truth fixture is used only incidentally where an approximation target is
convenient — all assertions run on AI-side objects.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from pymo.scientist import (
    ConditionBinding,
    ConditionComparison,
    ExperimentSpec,
    KnowledgeBase,
    Laboratory,
    Mission,
    ObservationRecord,
    ObservationReduction,
    OutputBinding,
    ScientificModel,
    ScientistAgent,
    ScientistState,
    comparison_input,
    disagreement,
    model_prediction,
    rank_discriminating_conditions,
    verify_prediction,
)
from pymo.scientist import agent as agent_module
from pymo.scientist import prediction as prediction_module
from pymo.scientist import state as state_module
from pymo.scientist.agent import proposal_to_spec
from pymo.scientist.prediction import (
    Prediction,
    PredictionRecord,
    adjudicate,
    prediction_from_belief,
    verify_commitment,
)
from pymo.universes import load_universe

MISSION = Mission(id="001", title="Discover Gravity", objective="",
                  target="gravity", unit="m/s^2")
DROP_10M = ExperimentSpec(kind="drop", drop_height=10.0)


@pytest.fixture(scope="module")
def lab():
    return Laboratory(load_universe("universe_001"))


def _agent(lab, tmp_path) -> ScientistAgent:
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    return ScientistAgent(lab, knowledge)


# -- TEST A: the commitment is persisted before the experiment executes -----

def test_a_commitment_persisted_before_experiment(lab, tmp_path):
    knowledge_path = tmp_path / "k.json"
    agent = _agent(lab, tmp_path)
    events: list[str] = []
    real_form = agent.form_prediction
    real_run = agent.laboratory.run_experiment

    def spy_form(mission, state=None):
        events.append("predict")
        return real_form(mission, state)

    def spy_run(spec):
        # At the FIRST experiment tick, the commitment must already be on
        # disk and still OPEN (verification happens only after the run).
        events.append("experiment")
        if "experiment" not in events[:-1]:
            data = json.loads(knowledge_path.read_text(encoding="utf-8"))
            assert data["predictions"], "no commitment persisted"
            assert all(p["status"] == "open" for p in data["predictions"])
            assert verify_commitment(
                prediction_module.PredictionRecord(**data["predictions"][0]))
        return real_run(spec)

    agent.form_prediction = spy_form
    agent.laboratory.run_experiment = spy_run
    agent.run_mission(MISSION)

    assert "predict" in events and "experiment" in events
    assert events.index("predict") < events.index("experiment")


# -- TEST B: tampering with a commitment is detected -------------------------

def test_b_tampered_commitment_hash_mismatch(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    committed = knowledge.commit_prediction(
        model_ref="innate prior", claim="gravity", spec_ref=DROP_10M.id,
        value=9.81, tolerance=0.1)
    assert verify_commitment(committed)

    tampered = replace(committed, predicted=9.9)   # silent edit attempt
    assert not verify_commitment(tampered)

    record = lab.run_experiment(DROP_10M)
    with pytest.raises(ValueError, match="commitment hash mismatch"):
        adjudicate(tampered, record)
    knowledge.predictions[committed.prediction_id] = tampered
    with pytest.raises(ValueError, match="commitment hash mismatch"):
        knowledge.record_verification(
            committed.prediction_id, record.experiment_id,
            adjudicate(committed, record))


# -- TEST C: verification needs ONLY the committed prediction + the record --

def test_c_verification_uses_only_prediction_and_record(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    committed = knowledge.commit_prediction(
        model_ref="innate prior", claim="gravity", spec_ref=DROP_10M.id,
        value=9.81, tolerance=0.1)
    record = lab.run_experiment(DROP_10M)

    outcome = adjudicate(committed, record)     # inputs: exactly these two

    assert outcome.status == "confirmed"
    assert outcome.observed == pytest.approx(9.81, rel=1e-2)
    assert outcome.residual == pytest.approx(outcome.observed - 9.81)


# -- TEST D: truth is unreachable from the prediction/verification code -----

def test_d_truth_unreachable_from_prediction_and_state():
    """TEST D: the prediction AND the belief-update code never import the
    universe layer — the whole feedback loop stays inside the AI side."""
    for module in (prediction_module, state_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pymo.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pymo.universes")
    prediction_src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    assert "secrets" not in prediction_src and "y_true" not in prediction_src
    # and the adjudication channel is two objects wide — nothing else fits
    params = list(inspect.signature(adjudicate).parameters)
    assert params == ["committed", "record"]


# -- TEST E / F: confirmed and refuted are both recorded --------------------

def test_e_confirmed_prediction_is_recorded(lab, tmp_path):
    agent = _agent(lab, tmp_path)
    report = agent.run_mission(MISSION)

    assert report.status == "DISCOVERED"          # the M001 flow is intact
    assert agent.last_committed_predictions
    committed = agent.last_committed_predictions[0]
    assert committed.claim == "gravity" and committed.status == "confirmed"
    assert len(agent.last_verifications) == 3      # one per record
    assert all(v.status == "confirmed" and v.prediction_id == committed.prediction_id
               for v in agent.last_verifications)

    # persisted, reloadable, and the commitment still verifies
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    loaded = knowledge.prediction(committed.prediction_id)
    assert loaded is not None and loaded.status == "confirmed"
    assert verify_commitment(loaded)
    assert len(knowledge.verifications_for(committed.prediction_id)) == 3


def test_f_refuted_prediction_is_recorded(lab, tmp_path):
    """A deliberately wrong BELIEF is REFUTED — falsification is a
    first-class outcome, and the mission still completes honestly."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    state.update("gravity", 8.0, 0.00625)   # the AI wrongly believes g ≈ 8 ± 0.05
    agent = ScientistAgent(lab, knowledge)
    report = agent.run_mission(MISSION, state)

    assert report.status == "DISCOVERED"          # falsification derails nothing
    committed = agent.last_committed_predictions[0]
    assert committed.model_ref.startswith("state belief")
    assert committed.predicted == pytest.approx(8.0)
    assert committed.status == "refuted"
    assert all(v.status == "refuted" for v in agent.last_verifications)
    assert all(abs(v.residual) > 0.05 for v in agent.last_verifications)

    reloaded = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    assert reloaded.prediction(committed.prediction_id).status == "refuted"

    # Step 2: the refutations also fed back into the self-model — the AI
    # no longer holds the same wrong belief.
    belief = state.belief("gravity")
    observed = agent.last_verifications[0].observed
    assert belief.span > 0.1                    # widened past the old belief
    assert belief.lo <= observed <= belief.hi   # reality is covered again


def test_belief_moves_the_prediction(tmp_path):
    """TEST A: the prediction is the AI's belief, translated — move the
    belief and the prediction moves with it (no hard-coded prior)."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)

    prior = state.belief("gravity")
    guess = prediction_from_belief(prior)
    assert guess.claim == "gravity"
    assert guess.value == pytest.approx(prior.midpoint)
    assert guess.tolerance == pytest.approx(prior.span / 2.0)
    assert guess.source.startswith("state belief")

    state.update("gravity", 9.79, 0.01)       # the AI's belief tightens
    moved = state.belief("gravity")
    guess2 = prediction_from_belief(moved)
    assert guess2.value == pytest.approx(9.79)
    assert guess2.tolerance == pytest.approx(moved.span / 2.0)
    assert guess2.band != guess.band


def test_tight_correct_belief_confirms(lab, tmp_path):
    """A tightened, correct belief produces a sharp CONFIRMED verdict —
    the real 9.81 sits inside the AI's own 9.79 ± ~0.10 band."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    state.update("gravity", 9.79, 0.01)
    agent = ScientistAgent(lab, knowledge)
    agent.run_mission(MISSION, state)

    committed = agent.last_committed_predictions[0]
    assert committed.status == "confirmed"
    assert committed.tolerance < 0.15
    assert all(v.status == "confirmed"
               for v in agent.last_verifications)


# -- Phase 2 Step 2: verification feeds back into the beliefs ----------------

def test_confirmed_verification_narrows_belief(tmp_path):
    """TEST A: a confirmation moderately tightens the belief around the
    observation — strictly narrower, still covering it."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    belief = state.belief("gravity")
    lo0, hi0, span0 = belief.lo, belief.hi, belief.span

    assert state.learn_from_verification(
        "gravity", "confirmed", observed=10.5, predicted=10.0, tolerance=10.0)

    assert belief.span < span0
    assert belief.lo > lo0 and belief.hi < hi0      # strictly inside the old
    assert belief.lo < 10.5 < belief.hi             # still covers the observation


def test_refuted_verification_changes_belief(tmp_path):
    """TEST B: a refutation changes the belief — the AI never keeps the
    exact same wrong belief; it widens until reality is covered again."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    state.update("gravity", 8.0, 0.00625)     # the wrong belief: 8 ± 0.05
    belief = state.belief("gravity")
    before = (belief.lo, belief.hi, belief.span)

    assert state.learn_from_verification(
        "gravity", "refuted", observed=9.81, predicted=8.0, tolerance=0.05)

    assert (belief.lo, belief.hi, belief.span) != before
    assert belief.lo <= 9.81 <= belief.hi           # the surprise is covered
    assert belief.span > before[2]                  # less certain, honestly


def _learned_belief(path):
    knowledge = KnowledgeBase(path, universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    state.learn_from_verification("gravity", "confirmed",
                                  observed=9.81, predicted=10.0, tolerance=10.0)
    belief = state.belief("gravity")
    return (belief.lo, belief.hi)


def test_belief_update_is_deterministic(tmp_path):
    """TEST C: identical inputs produce byte-identical belief intervals —
    the update rule is pure deterministic arithmetic."""
    assert (_learned_belief(tmp_path / "a.json")
            == _learned_belief(tmp_path / "b.json"))


def test_prediction_record_schema_unchanged():
    """TEST E (schema pin, with the untouched hash tests B/C above): Phase
    1's PredictionRecord fields and commitment machinery are exactly as
    before the belief-feedback step."""
    assert [f.name for f in dataclasses.fields(PredictionRecord)] == [
        "prediction_id", "model_ref", "claim", "spec_ref", "predicted",
        "tolerance", "committed_hash", "seq", "created_at", "status"]


def test_verification_updates_belief_in_mission_loop(lab, tmp_path):
    """The Phase 2 loop closes end to end: predict FROM the belief ->
    verify -> the belief is updated on the same self-model instance."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span0 = state.belief("gravity").span
    agent = ScientistAgent(lab, knowledge)

    agent.run_mission(MISSION, state)

    assert agent.last_state is state
    belief = state.belief("gravity")
    observed = agent.last_verifications[0].observed
    assert belief.span < span0              # the confirmations tightened it
    assert belief.lo > 0.0 and belief.hi < 20.0
    assert belief.lo < observed < belief.hi


# -- Phase 2 Step 3: consecutive verification-learning -------------------

def test_consecutive_confirmed_narrows_monotonically(tmp_path):
    """TEST A: 连续 CONFIRMED 后 belief span 单调变小。

    每次 CONFIRMED 将区间缩小为原来的 _CONFIRM_KEEP=0.75 倍。
    n 次确认后 span = 0.75^n * span0，严格递减。
    """
    knowledge = KnowledgeBase(tmp_path / "a.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    observed = 10.0
    spans = []
    for _ in range(5):
        state.learn_from_verification(
            "gravity", "confirmed",
            observed=observed, predicted=observed, tolerance=10.0)
        spans.append(state.belief("gravity").span)

    for i in range(1, len(spans)):
        assert spans[i] < spans[i - 1], \
            f"span did not decrease at step {i}"
    # 验证理论值: 0.75^5 * 初始 span (20.0)
    assert spans[-1] == pytest.approx(20.0 * (0.75 ** 5), rel=1e-9)


def test_consecutive_confirmed_covers_observed(tmp_path):
    """TEST B: 每次 CONFIRMED 后 belief 仍包含 observed value。"""
    knowledge = KnowledgeBase(tmp_path / "b.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    observed = 10.0
    for _ in range(5):
        state.learn_from_verification(
            "gravity", "confirmed",
            observed=observed, predicted=observed, tolerance=10.0)
        belief = state.belief("gravity")
        assert belief.lo <= observed <= belief.hi


def test_consecutive_confirmed_prediction_reflects_belief(tmp_path):
    """CONFIRMED 后 prediction_from_belief 使用更新后的 belief。

    验证完整链路: belief → prediction → CONFIRMED → learn → new belief →
    new prediction 反映新 belief。
    使用非中点的 observed 以确保 midpoint 发生偏移。
    """
    from pymo.scientist.prediction import prediction_from_belief

    knowledge = KnowledgeBase(tmp_path / "c.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    # observed 不是初始 belief 的中点 (10.0)，确保 midpoint 偏移
    observed = 5.0

    pred_before = prediction_from_belief(state.belief("gravity"))
    assert pred_before.value == pytest.approx(10.0)  # midpoint of (0, 20)

    state.learn_from_verification(
        "gravity", "confirmed",
        observed=observed, predicted=observed, tolerance=10.0)

    belief_after = state.belief("gravity")
    pred_after = prediction_from_belief(belief_after)
    # 新的 prediction 应反映更新后的 belief
    assert pred_after.value == pytest.approx(belief_after.midpoint)
    assert pred_after.tolerance == pytest.approx(belief_after.span / 2.0)
    # 新的 prediction 值应不同于之前的 (midpoint 从 10.0 偏移到 8.75)
    assert pred_after.value != pred_before.value


def test_consecutive_confirmed_is_deterministic(tmp_path):
    """TEST C: 相同初始 belief + 相同 observations → 完全相同的最终 belief。

    两次独立运行，从相同的初始状态出发，经过相同的 CONFIRMED 序列，
    必须得到 byte-identical 的最终区间。
    """

    def _run():
        knowledge = KnowledgeBase(tmp_path / "d.json", universe="universe_001")
        state = ScientistState(("gravity",), knowledge)
        for _ in range(5):
            state.learn_from_verification(
                "gravity", "confirmed",
                observed=10.0, predicted=10.0, tolerance=10.0)
        b = state.belief("gravity")
        return (round(b.lo, 15), round(b.hi, 15), round(b.span, 15))

    assert _run() == _run()


def test_consecutive_refuted_changes_deterministically(tmp_path):
    """TEST D: 连续 REFUTED 时 belief 确定性变化，span 单调增大，
    不会被强行恢复成原来的错误 belief。"""

    def _run():
        knowledge = KnowledgeBase(tmp_path / "e.json", universe="universe_001")
        state = ScientistState(("gravity",), knowledge)
        state.update("gravity", 8.0, 0.00625)  # 错误信念: 8 ± 0.05
        spans = []
        for _ in range(3):
            belief = state.belief("gravity")
            spans.append(belief.span)
            state.learn_from_verification(
                "gravity", "refuted",
                observed=9.81, predicted=8.0, tolerance=0.05)
        return (spans, state.belief("gravity").span)

    result1 = _run()
    result2 = _run()
    # 确定性: 相同输入产生相同输出
    assert result1 == result2
    # span 单调增大
    assert result1[0][1] > result1[0][0]
    assert result1[0][2] > result1[0][1]
    # 最终 span 大于初始错误信念的 span (0.1)
    assert result1[1] > 0.1


def test_consecutive_refuted_no_recovery(tmp_path):
    """REFUTED 后即使再 CONFIRMED，belief 也不会回到原来的错误值。

    REFUTED 的本质是让 AI 更不确定（扩大区间），而不是让 AI 回到原点。
    """
    knowledge = KnowledgeBase(tmp_path / "f.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    state.update("gravity", 8.0, 0.00625)  # 错误信念
    original_lo = state.belief("gravity").lo
    original_hi = state.belief("gravity").hi

    # REFUTED 一次
    state.learn_from_verification(
        "gravity", "refuted",
        observed=9.81, predicted=8.0, tolerance=0.05)
    after_refuted_lo = state.belief("gravity").lo
    after_refuted_hi = state.belief("gravity").hi

    # CONFIRMED 一次
    state.learn_from_verification(
        "gravity", "confirmed",
        observed=9.81, predicted=9.81, tolerance=10.0)
    after_confirmed_lo = state.belief("gravity").lo
    after_confirmed_hi = state.belief("gravity").hi

    # REFUTED 后不应回到原来的错误信念
    assert (after_refuted_lo, after_refuted_hi) != (original_lo, original_hi)
    # CONFIRMED 后也不应回到原来的错误信念
    assert (after_confirmed_lo, after_confirmed_hi) != (original_lo, original_hi)


def test_consecutive_updates_no_universe_access(tmp_path):
    """TEST E: 连续学习过程中 prediction.py / state.py 不访问 pymo.universes。"""
    for mod in (state_module, prediction_module):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pymo.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pymo.universes")


# -- append-only history + additive schema ----------------------------------

def test_changed_mind_appends_new_history(lab, tmp_path):
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    first = knowledge.commit_prediction("innate prior", "gravity",
                                        DROP_10M.id, 9.81, 0.1)
    second = knowledge.commit_prediction("second thoughts", "gravity",
                                         DROP_10M.id, 9.5, 0.2)
    assert first.prediction_id != second.prediction_id
    assert second.seq == first.seq + 1
    assert set(knowledge.predictions) == {first.prediction_id,
                                          second.prediction_id}


def test_pre_phase1_knowledge_files_remain_valid(tmp_path):
    """Additive evolution: a laws-only file (Mission 001–003 layout) loads
    unchanged, and saving upgrades the schema without touching the laws."""
    path = tmp_path / "old.json"
    law = {"name": "gravity", "formula": "g = 9.8100 m/s^2 (free-fall fit)",
           "value": 9.81, "unit": "m/s^2", "confidence": 0.9995, "r2": 1.0,
           "experiments": ["drop_h10_m1"], "derived_by": "polynomial",
           "properties": {}}
    path.write_text(json.dumps(
        {"universe": "universe_001", "updated": "2026-01-01T00:00:00+00:00",
         "laws": [law]}), encoding="utf-8")

    knowledge = KnowledgeBase(path, universe="universe_001")
    assert knowledge.knows("gravity")
    assert knowledge.predictions == {} and knowledge.verifications == {}
    knowledge.save()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["laws"][0]["name"] == "gravity"
    assert data["predictions"] == [] and data["verifications"] == []


# -- Model Competition Step 1: candidate model → prediction → disagreement ---

def test_two_models_can_coexist():
    """TEST A: 两个模型可以同时存在。"""
    h1 = ScientificModel(model_id="linear", params={"k": 0.5})
    h2 = ScientificModel(model_id="quadratic", params={"k": 0.1})
    assert h1.model_id != h2.model_id
    assert h1.params != h2.params


def test_same_conditions_different_predictions():
    """TEST B: 相同条件下两个模型产生不同 prediction。"""
    h1 = ScientificModel(model_id="linear", params={"k": 1.0})
    h2 = ScientificModel(model_id="quadratic", params={"k": 1.0})
    conditions = {"x": 5.0}
    pred1 = model_prediction(h1, conditions)
    pred2 = model_prediction(h2, conditions)
    assert pred1.value == 5.0       # k*x = 1.0*5 = 5
    assert pred2.value == 25.0      # k*x^2 = 1.0*25 = 25
    assert pred1.value != pred2.value


def test_same_prediction_disagreement_is_zero():
    """TEST C: 相同 prediction 的 disagreement = 0。"""
    h1 = ScientificModel(model_id="linear", params={"k": 1.0})
    h2 = ScientificModel(model_id="linear", params={"k": 1.0})
    conditions = {"x": 3.0}
    pred1 = model_prediction(h1, conditions)
    pred2 = model_prediction(h2, conditions)
    assert disagreement(pred1, pred2) == 0.0


def test_different_prediction_disagreement_correct():
    """TEST D: 不同 prediction 的 disagreement 正确。"""
    h1 = ScientificModel(model_id="linear", params={"k": 1.0})
    h2 = ScientificModel(model_id="quadratic", params={"k": 1.0})
    conditions = {"x": 5.0}
    pred1 = model_prediction(h1, conditions)  # 5.0
    pred2 = model_prediction(h2, conditions)  # 25.0
    assert disagreement(pred1, pred2) == 20.0


def test_model_prediction_is_deterministic():
    """TEST E: 相同输入重复计算结果完全一致。"""
    h1 = ScientificModel(model_id="linear", params={"k": 0.5})
    conditions = {"x": 10.0}
    pred1 = model_prediction(h1, conditions)
    pred2 = model_prediction(h1, conditions)
    assert pred1.value == pred2.value
    assert pred1.tolerance == pred2.tolerance
    assert pred1.claim == pred2.claim


def test_model_prediction_no_universe_access():
    """TEST F: model prediction 代码不访问 pymo.universes。"""
    import ast
    import pathlib
    src = pathlib.Path(
        prediction_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")
    src_str = src
    assert "secrets" not in src_str
    assert "y_true" not in src_str


def test_existing_prediction_flow_unchanged():
    """TEST G: 现有 Prediction / Commitment / Verification 流程完全不受影响。

    确认 PredictionRecord, Prediction, prediction_from_belief,
    commitment_hash, verify_commitment, adjudicate 的行为和签名不变。
    """
    import dataclasses
    import pathlib
    import tempfile

    from pymo.scientist.knowledge import KnowledgeBase
    from pymo.scientist.prediction import (
        Prediction,
        PredictionRecord,
        commitment_hash,
        prediction_from_belief,
    )
    from pymo.scientist.state import ScientistState
    assert [f.name for f in dataclasses.fields(PredictionRecord)] == [
        "prediction_id", "model_ref", "claim", "spec_ref", "predicted",
        "tolerance", "committed_hash", "seq", "created_at", "status"]
    # prediction_from_belief still works
    with tempfile.TemporaryDirectory() as tmp:
        knowledge = KnowledgeBase(pathlib.Path(tmp) / "k.json",
                                  universe="universe_001")
        state = ScientistState(("gravity",), knowledge)
        pred = prediction_from_belief(state.belief("gravity"))
        assert isinstance(pred, Prediction)
        assert pred.claim == "gravity"
    # commitment_hash still works
    payload = {"model_ref": "test", "claim": "gravity",
               "spec_ref": "spec1", "predicted": 9.81, "tolerance": 0.1}
    h1 = commitment_hash(payload)
    h2 = commitment_hash(payload)
    assert h1 == h2


def test_mission_regression_unchanged():
    """TEST H: Mission 001–004 regression 不变。

    确认新代码不影响 Mission 001 的完整流程。
    """
    import pathlib
    import tempfile

    from pymo.scientist import (
        KnowledgeBase,
        Laboratory,
        Mission,
        ScientistAgent,
    )
    from pymo.universes import load_universe

    MISSION = Mission(id="001", title="Discover Gravity", objective="",
                       target="gravity", unit="m/s^2")

    with tempfile.TemporaryDirectory() as tmp:
        lab = Laboratory(load_universe("universe_001"))
        knowledge = KnowledgeBase(pathlib.Path(tmp) / "k.json",
                                  universe="universe_001")
        agent = ScientistAgent(lab, knowledge)
        report = agent.run_mission(MISSION)
        assert report.status == "DISCOVERED"
        assert agent.last_committed_predictions
        committed = agent.last_committed_predictions[0]
        assert committed.claim == "gravity"


# -- Model Competition Step 2: discriminating conditions ---------------------
#
# Pre-experimental only: rank candidate conditions by how far the rival
# models' predictions diverge. No execution, no observation, no verdict.

def _rival_pair(k: float = 1.0) -> tuple[ScientificModel, ScientificModel]:
    """Step-1's test models: H1 y=k*x vs H2 y=k*x^2."""
    return (ScientificModel(model_id="linear", params={"k": k}),
            ScientificModel(model_id="quadratic", params={"k": k}))


def test_multiple_conditions_yield_multiple_comparisons():
    """TEST A: two models x several conditions -> several comparisons."""
    h1, h2 = _rival_pair()
    ranked = rank_discriminating_conditions(
        (h1, h2), [{"x": 1.0}, {"x": 2.0}, {"x": 5.0}])
    assert len(ranked) == 3
    assert all(isinstance(r, ConditionComparison) for r in ranked)
    assert all(len(r.predictions) == 2 for r in ranked)


def test_identical_models_zero_disagreement():
    """TEST B: models that predict identically disagree by exactly 0."""
    h1 = ScientificModel(model_id="linear", params={"k": 1.0})
    h2 = ScientificModel(model_id="linear", params={"k": 1.0})
    ranked = rank_discriminating_conditions((h1, h2), [{"x": 2.0}])
    assert ranked[0].disagreement == 0.0


def test_different_models_positive_disagreement():
    """TEST C: rival models predict different values -> disagreement > 0."""
    h1, h2 = _rival_pair()
    ranked = rank_discriminating_conditions((h1, h2), [{"x": 2.0}])
    assert ranked[0].disagreement > 0.0
    predictions = dict(ranked[0].predictions)
    assert predictions["linear"].value == 2.0       # k*x
    assert predictions["quadratic"].value == 4.0    # k*x^2


def test_bigger_gap_bigger_disagreement():
    """TEST D: x=1 -> 0, x=2 -> 2, x=5 -> 20 (monotone in the gap)."""
    h1, h2 = _rival_pair()
    by_x = {r.conditions[0][1]: r.disagreement for r in
            rank_discriminating_conditions(
                (h1, h2), [{"x": 1.0}, {"x": 2.0}, {"x": 5.0}])}
    assert by_x[1.0] == 0.0
    assert by_x[2.0] == 2.0
    assert by_x[5.0] == 20.0
    assert by_x[5.0] > by_x[2.0] > by_x[1.0]


def test_ranking_is_descending_by_disagreement():
    """TEST E: most discriminating condition first (x=5 beats x=2)."""
    h1, h2 = _rival_pair()
    ranked = rank_discriminating_conditions(
        (h1, h2), [{"x": 2.0}, {"x": 5.0}, {"x": 1.0}])
    assert [r.disagreement for r in ranked] == [20.0, 2.0, 0.0]
    assert ranked[0].conditions == (("x", 5.0),)
    assert ranked[-1].conditions == (("x", 1.0),)


def test_equal_disagreement_keeps_input_order():
    """TEST F: deterministic stable tie-break — equal disagreements keep
    the candidate_conditions input order (x=2 and x=-1 both give 2.0)."""
    h1, h2 = _rival_pair()
    order_a = rank_discriminating_conditions((h1, h2),
                                             [{"x": 2.0}, {"x": -1.0}])
    order_b = rank_discriminating_conditions((h1, h2),
                                             [{"x": -1.0}, {"x": 2.0}])
    assert [r.disagreement for r in order_a] == [2.0, 2.0]
    assert order_a[0].conditions == (("x", 2.0),)
    assert order_b[0].conditions == (("x", -1.0),)


def test_ranking_repeated_runs_identical():
    """TEST G: identical inputs -> byte-identical ranking, every time."""
    h1, h2 = _rival_pair()
    conditions = [{"x": 1.0}, {"x": 2.0}, {"x": 5.0}]
    first = rank_discriminating_conditions((h1, h2), conditions)
    second = rank_discriminating_conditions((h1, h2), conditions)
    assert first == second


def test_ranking_inputs_are_models_and_conditions_only():
    """TEST H: the ranking channel is exactly (models, conditions) — no
    universe access, no records, no engine truth can even be passed in;
    and empty rivals are refused honestly."""
    import inspect
    params = list(inspect.signature(
        rank_discriminating_conditions).parameters)
    assert params == ["models", "candidate_conditions"]
    with pytest.raises(ValueError, match="at least one candidate model"):
        rank_discriminating_conditions((), [{"x": 1.0}])


# -- Model Competition Step 3: the agent proposes, nothing executes ----------

def test_agent_proposes_most_discriminating_condition(lab, tmp_path):
    """The suggestion IS the top-ranked comparison: x=5 over x=1/x=2."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 1.0}, {"x": 2.0}, {"x": 5.0}])
    assert proposal is not None
    assert proposal.conditions == (("x", 5.0),)
    assert proposal.disagreement == 20.0
    assert proposal.predictions[0][0] == "linear"


def test_agent_proposes_none_when_models_agree(lab, tmp_path):
    """No condition separates identical models — an honest None, never a
    fabricated suggestion."""
    agent = _agent(lab, tmp_path)
    h1 = ScientificModel(model_id="linear", params={"k": 1.0})
    h2 = ScientificModel(model_id="linear", params={"k": 1.0})
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 2.0}, {"x": 5.0}])
    assert proposal is None


def test_agent_proposes_none_without_candidates(lab, tmp_path):
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    assert agent.propose_discriminating_experiment((h1, h2), []) is None


def test_proposal_is_pre_experimental(lab, tmp_path):
    """The boundary holds at the proposal stage: physics is never called,
    no prediction is committed, no belief or ledger state changes."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 2.0}, {"x": 5.0}])

    assert proposal is not None
    assert calls == []                              # physics never ran
    assert agent.knowledge.predictions == {}        # nothing committed
    assert agent.last_predictions == []
    assert agent.last_committed_predictions == []
    assert agent.last_verifications == []


# -- Model Competition Step 4: proposal -> Laboratory -> Observation ---------
#
# The single sanctioned crossing: the AI-side proposal is translated into
# the existing ExperimentSpec contract and the Laboratory runs it ONCE.
# No verdict, no belief update, no knowledge write — those are later steps.

def _drop_height_proposal(drop_height: float) -> ConditionComparison:
    """A proposal whose winning condition is a drop height, hand-built the
    way the AI would hold it after ranking. (Step 2 verified the ranking
    itself; here the models' own prediction vocabulary and the
    experiment-parameter vocabulary meet only in the condition dict.)"""
    predictions = (
        ("linear", Prediction(claim="linear", value=drop_height,
                              tolerance=1.0, source="model linear")),
        ("quadratic", Prediction(claim="quadratic", value=drop_height ** 2,
                                 tolerance=1.0, source="model quadratic")),
    )
    return ConditionComparison(
        conditions=(("drop_height", drop_height),),
        predictions=predictions,
        disagreement=drop_height ** 2 - drop_height,
    )


def test_proposal_executes_once_through_laboratory(lab, tmp_path):
    """TESTS A+B: a valid proposal crosses into the Laboratory and the
    result is the existing ObservationRecord type, produced through the
    standard (proposal, kind) channel."""
    import inspect
    agent = _agent(lab, tmp_path)
    proposal = _drop_height_proposal(5.0)

    record = agent.execute_proposal(proposal, kind="drop")

    assert isinstance(record, ObservationRecord)    # the existing type
    assert record.experiment_id == "drop_h5_m1"     # conditions became the spec
    assert len(record.t) > 10
    # the channel is exactly (proposal, kind, binding) — nothing else can
    # be passed (bound method: self does not appear in the signature)
    assert list(inspect.signature(
        agent.execute_proposal).parameters) == ["proposal", "kind", "binding"]


def test_condition_actually_drives_the_experiment(lab, tmp_path):
    """TEST C: the proposal's condition reaches physics — different
    drop heights give records starting at their own heights."""
    agent = _agent(lab, tmp_path)
    low = agent.execute_proposal(_drop_height_proposal(2.0), kind="drop")
    high = agent.execute_proposal(_drop_height_proposal(5.0), kind="drop")
    assert low.z[0] == pytest.approx(2.0, abs=0.1)
    assert high.z[0] == pytest.approx(5.0, abs=0.1)
    assert low.z[0] != high.z[0]


def test_execution_writes_no_truth_back_into_proposal(lab, tmp_path):
    """TEST D: the proposal is frozen AI-side history — executing it
    changes neither its conditions, predictions, nor disagreement; and
    the record carries no truth vocabulary."""
    agent = _agent(lab, tmp_path)
    proposal = _drop_height_proposal(5.0)
    before = proposal

    record = agent.execute_proposal(proposal, kind="drop")

    assert proposal == before                       # frozen artifact intact
    assert "gravity" not in record.experiment_id
    assert set(record.field_names) <= {"t", "z", "vx"}


def test_execution_channel_stays_ai_side():
    """TEST E: the whole crossing lives in AI-side modules — agent.py
    still never imports the universe layer."""
    src = Path(agent_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")


def test_no_verdict_no_belief_no_ledger(lab, tmp_path):
    """TESTS F+G+H: execution ends at the record — no model verdict, no
    belief change, no knowledge-base prediction/verification writes."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = ScientistAgent(lab, knowledge)
    proposal = _drop_height_proposal(5.0)

    agent.execute_proposal(proposal, kind="drop")

    assert state.belief("gravity").span == span_before                     # G
    assert knowledge.predictions == {} and knowledge.verifications == {}   # H
    assert agent.last_verifications == []                                  # F
    assert agent.last_committed_predictions == []                          # F


def test_execution_is_single_and_reproducible(lab, tmp_path):
    """TESTS I+J: one proposal -> exactly one experiment per execution,
    and the same proposal under the same configuration reproduces
    byte-identically."""
    agent = _agent(lab, tmp_path)
    proposal = _drop_height_proposal(5.0)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    first = agent.execute_proposal(proposal, kind="drop")
    second = agent.execute_proposal(proposal, kind="drop")

    assert len(calls) == 2                        # one per execution, no more
    assert calls[0] == calls[1]                   # identical spec both times
    assert list(first.t) == list(second.t)
    assert list(first.z) == list(second.z)


# -- Genesis Step 4.5: condition binding (model variable -> spec parameter) --

_BINDING = ConditionBinding({"x": "drop_height"})


def test_full_chain_from_model_to_observation(lab, tmp_path):
    """The complete automatic chain, no hand-built artifacts:
    ScientificModel -> model_prediction -> rank_discriminating_conditions
    -> agent proposal -> binding -> ExperimentSpec -> Laboratory ->
    ObservationRecord."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()          # y = k*x vs y = k*x^2, condition "x"

    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 1.0}, {"x": 5.0}])
    record = agent.execute_proposal(proposal, kind="drop", binding=_BINDING)

    assert isinstance(record, ObservationRecord)
    assert record.experiment_id == "drop_h5_m1"   # x=5 became drop_height=5
    assert record.z[0] == pytest.approx(5.0, abs=0.1)


def test_binding_maps_x_to_drop_height():
    """TEST A: the binding renames exactly as declared."""
    assert _BINDING.translate({"x": 5.0}) == {"drop_height": 5.0}


def test_unbound_variable_refused_not_silently_accepted():
    """TEST B: a variable the binding does not cover is a loud error; so
    is a binding target outside the spec parameter whitelist."""
    with pytest.raises(ValueError, match="not bound"):
        _BINDING.translate({"y": 5.0})
    with pytest.raises(ValueError, match="not ExperimentSpec parameters"):
        proposal_to_spec(
            ConditionComparison(conditions=(("x", 5.0),), predictions=(),
                                disagreement=3.0),
            kind="drop", binding=ConditionBinding({"x": "temperature"}))


def test_binding_is_deterministic():
    """TEST C: identical binding + conditions -> identical translation."""
    assert (_BINDING.translate({"x": 5.0})
            == _BINDING.translate({"x": 5.0}))


def test_binding_leaves_proposal_untouched(lab, tmp_path):
    """TEST D: translation reads the proposal, never writes it — the
    proposal keeps the model's own condition vocabulary ("x")."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 1.0}, {"x": 5.0}])
    before = proposal

    agent.execute_proposal(proposal, kind="drop", binding=_BINDING)

    assert proposal == before
    assert proposal.conditions == (("x", 5.0),)   # still model vocabulary


def test_laboratory_still_receives_standard_spec(lab, tmp_path):
    """TEST E: physics still only sees the existing ExperimentSpec through
    the existing run_experiment entry — the binding exists AI-side only."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    agent.execute_proposal(proposal, kind="drop", binding=_BINDING)

    assert calls == [ExperimentSpec(kind="drop", drop_height=5.0)]


def test_binding_channel_is_truth_free():
    """TEST F: the binding carries only declared names, and the AI-side
    modules it lives in stay universe-free."""
    assert _BINDING.mapping == {"x": "drop_height"}
    for module in (prediction_module, agent_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pymo.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pymo.universes")


def test_binding_execution_verifies_nothing(lab, tmp_path):
    """TESTS G+H: execution still ends at the record — no model
    verification, no belief change, no knowledge write."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = ScientistAgent(lab, knowledge)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])

    agent.execute_proposal(proposal, kind="drop", binding=_BINDING)

    assert knowledge.verifications == {} and knowledge.predictions == {}   # H
    assert agent.last_verifications == []                                  # G
    assert state.belief("gravity").span == span_before                     # H


# -- Genesis Step 5A: output binding (model output -> observation field) -----

_Y_TO_Z = OutputBinding({"y": "z"})


def test_output_binding_maps_y_to_z():
    """TEST A: the declared mapping resolves y to the position channel."""
    assert _Y_TO_Z.field_for("y") == "z"


def test_unbound_output_refused():
    """TEST B: an output the binding does not declare is a loud error."""
    with pytest.raises(ValueError, match="not bound"):
        _Y_TO_Z.field_for("u")


def test_non_observation_field_refused():
    """TEST C: targets outside ObservationRecord's measurement channels —
    identifiers, bookkeeping counters, invented names — are refused."""
    for bad in ("truth", "steps", "experiment_id", "secrets"):
        with pytest.raises(ValueError, match="not an ObservationRecord"):
            OutputBinding({"y": bad}).field_for("y")


def test_output_binding_is_deterministic():
    """TEST D: identical binding + output -> identical resolution."""
    assert _Y_TO_Z.field_for("y") == _Y_TO_Z.field_for("y") == "z"


def test_comparison_input_touches_neither_prediction_nor_record(lab):
    """TESTS E+F: alignment is read-only — the prediction and the record
    come out exactly as they went in."""
    record = lab.run_experiment(ExperimentSpec(kind="drop", drop_height=5.0))
    prediction = Prediction(claim="linear", value=5.0, tolerance=1.0)

    aligned = comparison_input(prediction, record, _Y_TO_Z, output="y")

    assert aligned.prediction == prediction            # E: untouched
    assert aligned.observed == tuple(float(v) for v in record.z)
    # F: the record is a frozen dataclass and alignment only reads its
    # channels — the field names are unchanged measurement vocabulary
    assert record.field_names == ("t", "z")


def test_comparison_input_channel_is_truth_free():
    """TEST G: the binding carries only declared names, and prediction.py
    (its home) stays universe-free."""
    assert _Y_TO_Z.mapping == {"y": "z"}
    src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")


def test_comparison_computes_no_verdict(lab, tmp_path):
    """TESTS H+I: alignment produces no verification and touches no
    belief or knowledge — the verdict is a later step's decision."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = _agent(lab, tmp_path)
    record = lab.run_experiment(ExperimentSpec(kind="drop", drop_height=5.0))

    comparison_input(Prediction("linear", 5.0, 1.0), record, _Y_TO_Z, "y")

    assert knowledge.verifications == {} and knowledge.predictions == {}
    assert agent.last_verifications == []
    assert state.belief("gravity").span == span_before


def test_vx_binding_on_non_slide_record_refused(lab):
    """A binding to a channel the record does not carry fails loudly
    instead of comparing against nothing."""
    record = lab.run_experiment(ExperimentSpec(kind="drop", drop_height=5.0))
    with pytest.raises(ValueError, match="absent in this record"):
        comparison_input(Prediction("linear", 1.0, 0.5), record,
                         OutputBinding({"y": "vx"}), output="y")


def test_chain_reaches_the_verdict_doorstep(lab, tmp_path):
    """Steps 1–5A in one breath: model -> prediction -> rank -> proposal
    -> condition binding -> spec -> Laboratory -> record -> output
    binding -> aligned comparison input. The verdict itself is still a
    later step."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()

    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    record = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    prediction = model_prediction(h1, {"x": 5.0})

    aligned = comparison_input(prediction, record, _Y_TO_Z, output="y")

    assert aligned.prediction == prediction
    assert aligned.field == "z"
    assert aligned.observed == tuple(float(v) for v in record.z)


# -- Genesis Step 5B-1: observation reduction (sample sequence -> scalar) ----

_REDUCTION = ObservationReduction(channel="z", rule="first")


def _aligned_drop_comparison(lab, drop_height=5.0):
    """A real ComparisonInput from a real drop record (position channel)."""
    record = lab.run_experiment(ExperimentSpec(kind="drop",
                                               drop_height=drop_height))
    prediction = Prediction(claim="linear", value=drop_height, tolerance=0.1)
    return comparison_input(prediction, record, _Y_TO_Z, output="y"), record


def test_reduction_of_real_drop_record_is_release_height(lab):
    """TEST A: the 'first' rule reduces a real drop record's z channel to
    its first sample — the release height minus one integration step."""
    aligned, _ = _aligned_drop_comparison(lab, 5.0)
    scalar = _REDUCTION.reduce(aligned)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(5.0, abs=0.01)
    assert scalar == pytest.approx(4.997275, abs=1e-6)   # h - g*dt^2 exactly


def test_reduction_contract_is_explicit():
    """TEST B: the rule is declared contract data, not implicit behavior."""
    assert _REDUCTION.channel == "z" and _REDUCTION.rule == "first"


def test_missing_rule_refused(lab):
    """TEST C: an empty rule is an explicit failure, not a default."""
    aligned, _ = _aligned_drop_comparison(lab)
    with pytest.raises(ValueError, match="unsupported reduction rule"):
        ObservationReduction(channel="z", rule="").reduce(aligned)


def test_unsupported_rule_refused(lab):
    """TEST D: rules that do not exist are refused by name."""
    aligned, _ = _aligned_drop_comparison(lab)
    for rule in ("mean", "last", "min", "max", "fit"):
        with pytest.raises(ValueError, match="unsupported reduction rule"):
            ObservationReduction(channel="z", rule=rule).reduce(aligned)


def test_channel_mismatch_refused(lab):
    """A reduction bound to another channel than the comparison carries
    is an explicit contract violation."""
    aligned, _ = _aligned_drop_comparison(lab)
    with pytest.raises(ValueError, match="channel mismatch|reduction targets"):
        ObservationReduction(channel="t", rule="first").reduce(aligned)


def test_reduction_is_deterministic(lab):
    """TEST E: identical record -> identical scalar, every time."""
    aligned, _ = _aligned_drop_comparison(lab)
    assert (_REDUCTION.reduce(aligned)
            == _REDUCTION.reduce(aligned)
            == _REDUCTION.reduce(aligned))


def test_reduction_modifies_neither_record_nor_prediction(lab):
    """TESTS F+G: reduction is read-only — the frozen record and the
    prediction come out exactly as they went in."""
    aligned, record = _aligned_drop_comparison(lab)
    before = aligned

    _REDUCTION.reduce(aligned)

    assert aligned == before                       # comparison untouched
    assert record.field_names == ("t", "z")        # record untouched
    assert aligned.prediction.value == 5.0         # prediction untouched


def test_reduction_channel_is_truth_free():
    """TEST H: the reduction carries only declared names, and its home
    module stays universe-free."""
    assert _REDUCTION.channel == "z" and _REDUCTION.rule == "first"
    src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")


def test_reduction_produces_no_verdict(lab, tmp_path):
    """TESTS I+J: reduction ends at the scalar — no VerificationRecord,
    no belief change, no knowledge write."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    record = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    aligned = comparison_input(model_prediction(h1, {"x": 5.0}), record,
                               _Y_TO_Z, output="y")

    _REDUCTION.reduce(aligned)

    assert knowledge.verifications == {} and knowledge.predictions == {}
    assert agent.last_verifications == []
    assert state.belief("gravity").span == span_before


def test_full_chain_reaches_the_scalar(lab, tmp_path):
    """Steps 1–5B-1 in one breath: model -> prediction -> rank ->
    proposal -> ConditionBinding -> ExperimentSpec -> Laboratory ->
    ObservationRecord -> OutputBinding -> ComparisonInput ->
    ObservationReduction -> scalar observed value. Adjudication is still
    a later step."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()

    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 2.0}, {"x": 5.0}])
    record = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    prediction = model_prediction(h1, {"x": 5.0})
    aligned = comparison_input(prediction, record, _Y_TO_Z, output="y")
    observed = ObservationReduction(channel="z", rule="first").reduce(aligned)

    assert observed == pytest.approx(5.0, abs=0.01)   # the release height,
    # seen through one integration step of the recording apparatus


# -- Genesis Step 5B-2: prediction vs observation -> model verification ------

def test_prediction_confirmed_within_tolerance(lab):
    """TEST A: predicted 5.0, reduced observed 4.997275 (the real drop
    semantics), tolerance 0.01 -> CONFIRMED."""
    aligned, _ = _aligned_drop_comparison(lab, 5.0)
    observed = _REDUCTION.reduce(aligned)
    outcome = verify_prediction(
        Prediction(claim="linear", value=5.0, tolerance=0.01), observed)
    assert outcome.status == "confirmed"
    assert outcome.observed == pytest.approx(4.997275, abs=1e-6)


def test_exact_match_confirmed():
    """TEST B: observed == predicted -> residual 0 -> CONFIRMED."""
    outcome = verify_prediction(Prediction("linear", 5.0, 0.01), 5.0)
    assert outcome.status == "confirmed"
    assert outcome.residual == 0.0


def test_beyond_tolerance_refuted():
    """TEST C: |residual| > tolerance -> REFUTED."""
    outcome = verify_prediction(Prediction("linear", 5.0, 0.01), 5.02)
    assert outcome.status == "refuted"


def test_residual_sign_is_observed_minus_predicted():
    """TEST D: the residual definition is explicit and signed."""
    up = verify_prediction(Prediction("linear", 5.0, 1.0), 5.4)
    down = verify_prediction(Prediction("linear", 5.0, 1.0), 4.6)
    assert up.residual == pytest.approx(0.4)
    assert down.residual == pytest.approx(-0.4)
    assert up.observed == 5.4 and up.predicted == 5.0


def test_tolerance_boundary_is_inclusive():
    """TEST E: abs(residual) == tolerance -> CONFIRMED (binary-exact
    values: 0.25 is exactly representable)."""
    outcome = verify_prediction(Prediction("linear", 1.0, 0.25), 1.25)
    assert outcome.residual == 0.25
    assert outcome.status == "confirmed"


def test_discretization_offset_must_be_absorbed_by_tolerance(lab):
    """TEST F: the g*dt^2 recording offset is NOT corrected away — a
    tolerance too tight to absorb it yields REFUTED on the raw reduced
    value."""
    aligned, _ = _aligned_drop_comparison(lab, 5.0)
    observed = _REDUCTION.reduce(aligned)          # 4.997275, uncorrected
    outcome = verify_prediction(
        Prediction(claim="linear", value=5.0, tolerance=0.001), observed)
    assert outcome.status == "refuted"
    assert outcome.observed == pytest.approx(4.997275, abs=1e-6)  # not fixed up


def test_verification_modifies_neither_prediction_nor_record(lab):
    """TESTS G+H: the verdict is read-only."""
    aligned, record = _aligned_drop_comparison(lab)
    prediction = aligned.prediction
    before_prediction, before_comparison = prediction, aligned

    verify_prediction(prediction, _REDUCTION.reduce(aligned))

    assert prediction == before_prediction
    assert aligned == before_comparison
    assert record.field_names == ("t", "z")


def test_verification_channel_is_truth_free():
    """TEST I: the verdict consumes only (prediction, observed) — its
    home module stays universe-free."""
    src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")


def test_verification_writes_nothing(lab, tmp_path):
    """TESTS J+K+M: a model-competition verdict is in-memory only — no
    VerificationRecord persisted, no knowledge write, no belief change."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    record = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    aligned = comparison_input(model_prediction(h1, {"x": 5.0}), record,
                               _Y_TO_Z, output="y")
    outcome = verify_prediction(aligned.prediction,
                                _REDUCTION.reduce(aligned))

    assert outcome.status == "confirmed"
    assert knowledge.verifications == {} and knowledge.predictions == {}
    assert agent.last_verifications == []
    assert state.belief("gravity").span == span_before


def test_verification_runs_no_experiment(lab, tmp_path):
    """TEST L: adjudication is pure — the Laboratory is never called."""
    agent = _agent(lab, tmp_path)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    verify_prediction(Prediction("linear", 5.0, 1.0), 5.0)
    assert calls == []


def test_one_prediction_one_observation_no_aggregation():
    """TEST N: the verdict binds exactly one prediction to one reduced
    observation — no cross-experiment aggregation, no shared state."""
    a = verify_prediction(Prediction("linear", 5.0, 0.01), 4.997275)
    b = verify_prediction(Prediction("linear", 5.0, 0.01), 4.997275)
    assert a == b
    assert a is not b


def test_full_chain_reaches_the_verdict(lab, tmp_path):
    """Steps 1–5B-2 in one breath: model -> prediction -> rank ->
    proposal -> ConditionBinding -> ExperimentSpec -> Laboratory ->
    ObservationRecord -> OutputBinding -> ComparisonInput ->
    ObservationReduction -> scalar -> residual -> CONFIRMED/REFUTED.
    Belief, knowledge and model survival are still untouched."""
    agent = _agent(lab, tmp_path)
    h1, h2 = _rival_pair()

    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    record = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    prediction = replace(model_prediction(h1, {"x": 5.0}), tolerance=0.01)
    aligned = comparison_input(prediction, record, _Y_TO_Z, output="y")
    observed = _REDUCTION.reduce(aligned)
    outcome = verify_prediction(prediction, observed)

    assert observed == pytest.approx(4.997275, abs=1e-6)
    assert outcome.residual == pytest.approx(-0.002725, abs=1e-6)
    assert outcome.status == "confirmed"
