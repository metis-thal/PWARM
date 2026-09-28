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

from pwarm.scientist import (
    CompetitionState,
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
from pwarm.scientist import agent as agent_module
from pwarm.scientist import prediction as prediction_module
from pwarm.scientist import state as state_module
from pwarm.scientist.agent import proposal_to_spec
from pwarm.scientist.prediction import (
    Prediction,
    PredictionOutcome,
    PredictionRecord,
    VerificationRecord,
    adjudicate,
    prediction_from_belief,
    verify_commitment,
)
from pwarm.universes import load_universe

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
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")
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
    from pwarm.scientist.prediction import prediction_from_belief

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
    """TEST E: 连续学习过程中 prediction.py / state.py 不访问 pwarm.universes。"""
    for mod in (state_module, prediction_module):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")


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
    """TEST F: model prediction 代码不访问 pwarm.universes。"""
    import ast
    import pathlib
    src = pathlib.Path(
        prediction_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pwarm.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pwarm.universes")
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

    from pwarm.scientist.knowledge import KnowledgeBase
    from pwarm.scientist.prediction import (
        Prediction,
        PredictionRecord,
        commitment_hash,
        prediction_from_belief,
    )
    from pwarm.scientist.state import ScientistState
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

    from pwarm.scientist import (
        KnowledgeBase,
        Laboratory,
        Mission,
        ScientistAgent,
    )
    from pwarm.universes import load_universe

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
                assert not alias.name.startswith("pwarm.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pwarm.universes")


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
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")


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
                assert not alias.name.startswith("pwarm.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pwarm.universes")


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
                assert not alias.name.startswith("pwarm.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pwarm.universes")


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
                assert not alias.name.startswith("pwarm.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pwarm.universes")


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


# -- Genesis Step 6: competition prediction commitment ------------------------

def _committed_rivals(agent, tolerances=None):
    """rank -> propose -> commit BOTH rivals under the winning condition."""
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    records = agent.commit_discriminating_predictions(
        (h1, h2), proposal, kind="drop",
        binding=ConditionBinding({"x": "drop_height"}),
        tolerances=tolerances)
    return records, proposal


def test_both_rivals_commit_independently(lab, tmp_path):
    """TESTS A+B+C: two rivals -> two predictions -> two commitments that
    coexist without overwriting each other."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    records, _ = _committed_rivals(agent)

    assert len(records) == 2
    assert {r.claim for r in records} == {"linear", "quadratic"}
    assert {r.predicted for r in records} == {5.0, 25.0}
    assert all(verify_commitment(r) for r in records)
    assert len(knowledge.predictions) == 2          # both coexist
    assert records[0].prediction_id != records[1].prediction_id


def test_shared_spec_ref_independent_ids_and_hashes(lab, tmp_path):
    """TESTS D+E: both records point at the SAME experiment spec, yet
    carry independent ids, sequence indices and hashes."""
    agent = _agent(lab, tmp_path)
    records, _ = _committed_rivals(agent)
    assert records[0].spec_ref == records[1].spec_ref == "drop_h5_m1"
    assert records[0].prediction_id != records[1].prediction_id
    assert records[0].committed_hash != records[1].committed_hash
    assert records[1].seq == records[0].seq + 1


def test_commitment_phase_never_calls_the_laboratory(lab, tmp_path):
    """TEST F: the commitment phase is pure AI-side bookkeeping — physics
    is never invoked."""
    agent = _agent(lab, tmp_path)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    _committed_rivals(agent)
    assert calls == []


def test_later_prediction_changes_cannot_touch_commitments(lab, tmp_path):
    """TEST G: commitments are frozen history — a different prediction
    object afterwards changes nothing about the saved record, and
    tampering with a STORED record is detected by its hash."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    guess_a = model_prediction(h1, {"x": 5.0})
    records = agent.commit_discriminating_predictions(
        (h1, h2), proposal, kind="drop",
        binding=ConditionBinding({"x": "drop_height"}))

    replace(guess_a, value=9.9, tolerance=0.5)      # a changed mind, in memory
    assert records[0].predicted == 5.0              # commitment unchanged
    assert verify_commitment(records[0])

    tampered = replace(records[0], predicted=9.9)   # edit the STORED record
    assert not verify_commitment(tampered)
    knowledge.predictions[records[0].prediction_id] = tampered
    outcome = PredictionOutcome(claim="linear", predicted=9.9, observed=5.0,
                                residual=-4.9, status="refuted")
    with pytest.raises(ValueError, match="hash mismatch"):
        knowledge.record_verification(records[0].prediction_id,
                                      "drop_h5_m1", outcome)


def test_rivals_can_have_different_tolerances(lab, tmp_path):
    """TEST H: per-model tolerances flow into each commitment untouched."""
    agent = _agent(lab, tmp_path)
    records, _ = _committed_rivals(
        agent, tolerances={"linear": 0.01, "quadratic": 0.5})
    assert {r.claim: r.tolerance for r in records} == {
        "linear": 0.01, "quadratic": 0.5}


def test_commitment_phase_writes_only_commitments(lab, tmp_path):
    """TESTS I+J+K: commitment ends at PredictionRecords — no
    VerificationRecord, no belief change, nothing else in the store."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    agent = ScientistAgent(lab, knowledge)

    _committed_rivals(agent)

    assert knowledge.verifications == {}                        # I
    assert len(knowledge.predictions) == 2                      # K: commitments only
    assert agent.last_verifications == []
    assert state.belief("gravity").span == span_before          # J


def test_commitment_channel_is_truth_free():
    """TESTS L+M: the commitment path's home modules stay universe-free."""
    knowledge_module = sys.modules["pwarm.scientist.knowledge"]
    for module in (prediction_module, agent_module, knowledge_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")


def test_commitments_precede_execution(lab, tmp_path):
    """TEST N: commit A, commit B — only then may execute_proposal run.
    The spy checks, at the moment physics is invoked, that BOTH
    commitments are already persisted."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    events: list[tuple[str, int]] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        events.append(("experiment", len(knowledge.predictions)))
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run

    records = agent.commit_discriminating_predictions(
        (h1, h2), proposal, kind="drop",
        binding=ConditionBinding({"x": "drop_height"}))
    events.append(("commit", len(knowledge.predictions)))
    agent.execute_proposal(proposal, kind="drop",
                           binding=ConditionBinding({"x": "drop_height"}))

    assert events == [("commit", 2), ("experiment", 2)]
    assert len(records) == 2


def test_gravity_commitment_flow_unchanged(lab, tmp_path):
    """TEST O: the ordinary belief-driven gravity commitment flow is
    untouched by the competition path."""
    agent = _agent(lab, tmp_path)
    report = agent.run_mission(MISSION)
    assert report.status == "DISCOVERED"
    assert agent.last_committed_predictions
    assert agent.last_committed_predictions[0].claim == "gravity"


def test_competition_chain_stops_at_commitment(lab, tmp_path):
    """The Step 6 chain in one breath: model A -> prediction A, model B ->
    prediction B -> rank -> proposal -> commit A -> commit B -> STOP.
    No experiment runs; both commitments sit in the ledger, hash-verified."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    h1, h2 = _rival_pair()

    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    records = agent.commit_discriminating_predictions(
        (h1, h2), proposal, kind="drop",
        binding=ConditionBinding({"x": "drop_height"}))

    assert len(records) == 2
    assert len(knowledge.predictions) == 2
    assert all(verify_commitment(r) for r in records)
    assert calls == []                              # stopped before physics


# -- Genesis Step 7: one observation, many competing verdicts -----------------

def _competed_and_executed(lab, tmp_path, tolerances=None):
    """Steps 2–6 + one execution: rivals committed, experiment run ONCE."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    h1, h2 = _rival_pair()
    proposal = agent.propose_discriminating_experiment(
        (h1, h2), [{"x": 5.0}])
    records = agent.commit_discriminating_predictions(
        (h1, h2), proposal, kind="drop",
        binding=ConditionBinding({"x": "drop_height"}), tolerances=tolerances)
    observation = agent.execute_proposal(
        proposal, kind="drop", binding=ConditionBinding({"x": "drop_height"}))
    return knowledge, agent, records, observation


def _verify_rivals(agent, records, observation):
    return agent.verify_competing_predictions(
        records, observation, OutputBinding({"y": "z"}), "y",
        ObservationReduction(channel="z", rule="first"))


def test_one_observation_yields_two_verifications(lab, tmp_path):
    """TEST A: two committed predictions + one record -> two
    VerificationRecords."""
    knowledge, agent, records, observation = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.01, "quadratic": 0.5})
    verifications = _verify_rivals(agent, records, observation)
    assert len(verifications) == 2
    assert len(knowledge.verifications) == 2


def test_verifications_link_to_own_predictions(lab, tmp_path):
    """TESTS B+C: each verification carries its own prediction_id, and
    both share the single experiment's id."""
    _, agent, records, observation = _competed_and_executed(lab, tmp_path)
    verifications = _verify_rivals(agent, records, observation)
    assert {v.prediction_id for v in verifications} == {
        r.prediction_id for r in records}
    assert all(v.experiment_id == observation.experiment_id
               for v in verifications)


def test_typical_competition_one_confirmed_one_refuted(lab, tmp_path):
    """TEST D: the canonical case — linear (predicted 5.0, observed
    4.997275) confirms; quadratic (predicted 25.0) refutes."""
    _, agent, records, observation = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.01, "quadratic": 0.5})
    verifications = _verify_rivals(agent, records, observation)
    statuses = {v.prediction_id: v.status for v in verifications}
    linear = next(r for r in records if r.claim == "linear")
    quadratic = next(r for r in records if r.claim == "quadratic")
    assert statuses[linear.prediction_id] == "confirmed"
    assert statuses[quadratic.prediction_id] == "refuted"


def test_both_confirm_and_both_refute_are_representable(lab, tmp_path):
    """TEST E: the verdicts are independent — both can confirm (a wide
    tolerance) or both refute (tight ones); no winner is implied."""
    _, agent, records, observation = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.01, "quadratic": 30.0})
    both_confirmed = _verify_rivals(agent, records, observation)
    assert all(v.status == "confirmed" for v in both_confirmed)

    _, agent2, records2, observation2 = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.001, "quadratic": 0.5})
    both_refuted = _verify_rivals(agent2, records2, observation2)
    assert all(v.status == "refuted" for v in both_refuted)


def test_residuals_are_per_prediction(lab, tmp_path):
    """TESTS F+G: residual = observed - each prediction's own value, and
    each verdict used its own commitment's tolerance."""
    observed_expected = 4.997275
    _, agent, records, observation = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.01, "quadratic": 0.5})
    verifications = _verify_rivals(agent, records, observation)
    by_pid = {v.prediction_id: v for v in verifications}
    for record in records:
        v = by_pid[record.prediction_id]
        assert v.residual == pytest.approx(observed_expected - record.predicted,
                                           abs=1e-6)
        assert f"tolerance {record.tolerance:g}" in v.evidence


def test_memory_prediction_changes_cannot_affect_verdicts(lab, tmp_path):
    """TEST H: verdicts are computed from the COMMITTED record — later
    edits to in-memory prediction objects change nothing."""
    _knowledge, agent, records, observation = _competed_and_executed(lab, tmp_path)
    h1, h2 = _rival_pair()
    for model in (h1, h2):
        replace(model_prediction(model, {"x": 5.0}), value=99.0, tolerance=99.0)
    verifications = _verify_rivals(agent, records, observation)
    assert {v.status for v in verifications} == {"confirmed", "refuted"}


def test_tampered_commitment_is_refused_not_quietly_verified(lab, tmp_path):
    """TEST I: editing a stored PredictionRecord breaks its hash — the
    verification refuses loudly and records nothing for it."""
    knowledge, agent, records, observation = _competed_and_executed(lab, tmp_path)
    tampered = replace(records[0], predicted=4.9)   # a real content edit
    assert not verify_commitment(tampered)
    knowledge.predictions[records[0].prediction_id] = tampered
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_rivals(agent, records, observation)
    assert len(knowledge.verifications) == 0        # nothing slipped through


def test_verification_phase_never_calls_the_laboratory(lab, tmp_path):
    """TEST J: the same observation verifies everyone — physics is not
    invoked again during verification."""
    _, agent, records, observation = _competed_and_executed(lab, tmp_path)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    _verify_rivals(agent, records, observation)
    assert calls == []


def test_verification_leaves_no_competition_state(lab, tmp_path):
    """TEST K: after verification — no belief update, no new predictions,
    no elimination/winner/survivor state anywhere in the store."""
    knowledge, agent, records, observation = _competed_and_executed(lab, tmp_path)
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span

    _verify_rivals(agent, records, observation)

    assert state.belief("gravity").span == span_before
    assert len(knowledge.predictions) == 2          # no new predictions
    assert all(r.status in ("confirmed", "refuted")
               for r in knowledge.predictions.values())
    assert not any("winner" in v.evidence or "survivor" in v.evidence
                   for v in knowledge.verifications.values())


def test_verifications_persist_independently(lab, tmp_path):
    """TEST L: both VerificationRecords survive a save/reload cycle with
    their own ids and links intact."""
    _knowledge, agent, records, observation = _competed_and_executed(lab, tmp_path)
    verifications = _verify_rivals(agent, records, observation)
    reloaded = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    assert len(reloaded.verifications) == 2
    assert {v.prediction_id for v in reloaded.verifications.values()} == {
        v.prediction_id for v in verifications}


def test_reverification_appends_no_aggregation(lab, tmp_path):
    """TEST M: verifying the same commitments against the same experiment
    again appends fresh records — no accumulated score, no win rate."""
    knowledge, agent, records, observation = _competed_and_executed(lab, tmp_path)
    first = _verify_rivals(agent, records, observation)
    second = _verify_rivals(agent, records, observation)
    assert len(knowledge.verifications) == 4
    assert [v.status for v in second] == [v.status for v in first]
    assert second[0].verification_id != first[0].verification_id


def test_competition_verification_channel_is_truth_free():
    """TEST N: the verification path's home modules stay universe-free."""
    knowledge_module = sys.modules["pwarm.scientist.knowledge"]
    for module in (prediction_module, agent_module, knowledge_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")


def test_gravity_verification_flow_unchanged(lab, tmp_path):
    """TEST O: the ordinary belief-driven gravity verification flow is
    untouched by the competition path."""
    agent = _agent(lab, tmp_path)
    report = agent.run_mission(MISSION)
    assert report.status == "DISCOVERED"
    assert agent.last_verifications
    assert all(v.status == "confirmed" for v in agent.last_verifications)


def test_full_chain_stops_at_two_verifications(lab, tmp_path):
    """TEST P: Steps 1–7 in one breath — model A -> prediction, model B ->
    prediction -> rank -> proposal -> commit A -> commit B -> execute ONCE
    -> ObservationRecord -> OutputBinding -> ComparisonInput ->
    ObservationReduction -> verify A -> verify B. Final state: two
    VerificationRecords; belief and knowledge learning untouched."""
    knowledge, agent, records, observation = _competed_and_executed(
        lab, tmp_path, tolerances={"linear": 0.01, "quadratic": 0.5})
    verifications = _verify_rivals(agent, records, observation)

    assert len(verifications) == 2
    assert len(knowledge.verifications) == 2
    assert len(knowledge.predictions) == 2
    assert {v.status for v in verifications} == {"confirmed", "refuted"}


# -- Genesis Step 8: per-model evidence summary (aggregate, no verdict) -------
#
# Same model, many independent VerificationRecords -> one deterministic,
# AI-side EvidenceSummary. The summary is a faithful accounting and NOTHING
# more: no winner, no weight, no elimination, no belief, no probability.
# It reads only persisted PredictionRecords + VerificationRecords, tracing
# verification -> prediction_id -> model_ref, and writes nothing.

def _commit_and_verify(knowledge, model_ref, cases):
    """Build a model's evidence history directly in the store.

    ``cases``: list of (claim, predicted, tolerance, experiment_id, observed).
    Returns (predictions, verifications) in commit order.
    """
    predictions, verifications = [], []
    for claim, predicted, tolerance, experiment_id, observed in cases:
        pred = knowledge.commit_prediction(
            model_ref=model_ref, claim=claim, spec_ref=experiment_id,
            value=predicted, tolerance=tolerance)
        predictions.append(pred)
        residual = observed - predicted
        outcome = PredictionOutcome(
            claim=claim, predicted=predicted, observed=observed,
            residual=residual,
            status="confirmed" if abs(residual) <= tolerance else "refuted")
        verifications.append(knowledge.record_verification(
            pred.prediction_id, experiment_id, outcome))
    return predictions, verifications


def _step8_scenario(tmp_path):
    """The canonical Step 8 scenario: two rival models with shared and
    independent experiments, persisted and ready to be summarized."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    a_predictions, a_verifications = _commit_and_verify(knowledge, "model A", [
        # Prediction A1 -> Experiment E1 -> CONFIRMED
        ("linear", 5.0, 0.5, "E1", 5.0),
        # Prediction A2 -> Experiment E2 -> CONFIRMED
        ("linear", 6.0, 0.5, "E2", 6.0),
        # Prediction A3 -> Experiment E3 -> REFUTED
        ("quadratic", 9.0, 0.5, "E3", 8.0),
    ])
    b_predictions, b_verifications = _commit_and_verify(knowledge, "model B", [
        # Prediction B1 -> Experiment E1 -> REFUTED
        ("quadratic", 8.0, 0.5, "E1", 9.0),
        # Prediction B2 -> Experiment E2 -> CONFIRMED
        ("linear", 6.0, 0.5, "E2", 6.0),
    ])
    return (knowledge, a_predictions, a_verifications,
            b_predictions, b_verifications)


def test_a_three_independent_experiment_count_counted(tmp_path):
    """TEST A: one model, 3 independent experiments (confirmed, confirmed,
    refuted) -> the summary counts exactly 3."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    summary = knowledge.evidence_for_model("model A")

    assert summary.verification_count == 3
    assert summary.confirmed_count == 2
    assert summary.refuted_count == 1
    assert summary.independent_experiment_count == 3
    assert summary.experiment_ids == ("E1", "E2", "E3")


def test_b_two_rivals_summarized_independently(tmp_path):
    """TEST B: two competing models -> each summary is its own history."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    summary_a = knowledge.evidence_for_model("model A")
    summary_b = knowledge.evidence_for_model("model B")

    assert summary_a.model_ref == "model A"
    assert summary_b.model_ref == "model B"
    assert summary_a.verification_count == 3
    assert summary_b.verification_count == 2
    assert summary_a.confirmed_count == 2 and summary_a.refuted_count == 1
    assert summary_b.confirmed_count == 1 and summary_b.refuted_count == 1
    assert summary_b.experiment_ids == ("E1", "E2")
    assert set(summary_a.prediction_ids).isdisjoint(set(summary_b.prediction_ids))


def test_c_same_experiment_counts_once_per_model(tmp_path):
    """TEST C: a model verified twice against the SAME experiment -> that
    experiment_id appears once; the verifications still count individually."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model C", [
        ("linear", 5.0, 0.5, "E1", 5.0),    # C1 -> E1 -> confirmed
        ("linear", 5.5, 0.5, "E1", 5.5),    # C2 -> E1 -> confirmed (same exp)
    ])
    summary = knowledge.evidence_for_model("model C")

    assert summary.verification_count == 2
    assert summary.independent_experiment_count == 1
    assert summary.experiment_ids == ("E1",)


def test_d_same_prediction_not_double_counted(tmp_path):
    """TEST D: one prediction verified against two experiments -> it appears
    once in prediction_ids while both verifications are still counted."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    preds, _ = _commit_and_verify(knowledge, "model D", [
        ("linear", 5.0, 0.5, "E1", 5.0),    # D1 -> E1 -> confirmed
    ])
    # verify the SAME prediction against a second experiment
    pred = preds[0]
    outcome = PredictionOutcome(
        claim=pred.claim, predicted=pred.predicted, observed=5.0,
        residual=0.0, status="confirmed")
    knowledge.record_verification(pred.prediction_id, "E2", outcome)
    summary = knowledge.evidence_for_model("model D")

    assert summary.verification_count == 2
    assert summary.independent_experiment_count == 2
    assert summary.prediction_ids == (pred.prediction_id,)


def test_e_prediction_to_model_association_is_correct(tmp_path):
    """TEST E: each summary's evidence traces back to the right model via
    prediction_id -> model_ref; every verification lands in its owner's
    summary with its experiment_id and prediction_id preserved."""
    knowledge, a_preds, a_verifs, _, b_verifs = _step8_scenario(tmp_path)
    summary_a = knowledge.evidence_for_model("model A")

    assert summary_a.prediction_ids == tuple(
        sorted(p.prediction_id for p in a_preds))
    assert summary_a.evidence == tuple(a_verifs)
    # every A verification is present, carrying its own prediction+experiment
    for verif, pred in zip(a_verifs, a_preds):
        assert verif.prediction_id == pred.prediction_id
    # and B's evidence is exactly B's, disjoint from A's
    summary_b = knowledge.evidence_for_model("model B")
    assert summary_b.evidence == tuple(b_verifs)


def test_f_nonexistent_model_and_orphan_are_not_guessed(tmp_path):
    """TEST F: a model_ref with no history is an honest empty summary, and an
    orphaned VerificationRecord (unknown prediction_id) is skipped — its model
    is never guessed into anyone's summary."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)

    empty = knowledge.evidence_for_model("nobody")
    assert empty.model_ref == "nobody"
    assert empty.verification_count == 0
    assert empty.prediction_ids == () and empty.experiment_ids == ()
    assert empty.residuals == () and empty.statuses == ()
    assert empty.evidence == ()

    # an orphan: a verification whose prediction_id has no matching prediction
    orphan = VerificationRecord(
        verification_id="verif-9999", prediction_id="pred-nope",
        experiment_id="E9", observed=1.0, residual=0.0,
        status="confirmed", evidence="orphaned record")
    knowledge.verifications["verif-9999"] = orphan

    summary_a = knowledge.evidence_for_model("model A")
    summary_b = knowledge.evidence_for_model("model B")
    assert all(v.prediction_id != "pred-nope" for v in summary_a.evidence)
    assert all(v.prediction_id != "pred-nope" for v in summary_b.evidence)
    assert knowledge.evidence_for_model("pred-nope").verification_count == 0


def test_g_residuals_preserved_in_deterministic_order(tmp_path):
    """TEST G: residuals keep their recorded values and a deterministic
    (verification-id) order — no re-sorting, no re-derivation."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    summary_a = knowledge.evidence_for_model("model A")
    summary_b = knowledge.evidence_for_model("model B")

    # A: 5.0-5.0, 6.0-6.0, 8.0-9.0 in verification-id order
    assert summary_a.residuals == (0.0, 0.0, -1.0)
    assert summary_a.statuses == ("confirmed", "confirmed", "refuted")
    # B: 9.0-8.0, 6.0-6.0
    assert summary_b.residuals == (1.0, 0.0)
    assert summary_b.statuses == ("refuted", "confirmed")
    # residuals match the recorded VerificationRecords exactly
    assert all(summary_a.residuals[i] == v.residual
               for i, v in enumerate(summary_a.evidence))


def test_h_repeated_queries_are_identical(tmp_path):
    """TEST H: identical store -> byte-identical summary, every time."""
    def _run():
        knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
        return knowledge.evidence_for_model("model A")

    assert _run() == _run()


def test_i_summary_does_not_change_knowledge(tmp_path):
    """TEST I: summarizing is a pure read — predictions, verifications and
    the on-disk store are untouched."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    path = tmp_path / "k.json"
    before_predictions = dict(knowledge.predictions)
    before_verifications = dict(knowledge.verifications)
    before_disk = path.read_text(encoding="utf-8")

    knowledge.evidence_for_model("model A")
    knowledge.evidence_for_model("model B")

    assert knowledge.predictions == before_predictions
    assert knowledge.verifications == before_verifications
    assert path.read_text(encoding="utf-8") == before_disk


def test_j_summary_does_not_change_belief(tmp_path):
    """TEST J: summarizing never touches the AI's self-model (beliefs)."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span

    knowledge.evidence_for_model("model A")
    knowledge.evidence_for_model("model B")

    assert state.belief("gravity").span == span_before


def test_k_summary_produces_no_verification(tmp_path):
    """TEST K: summarizing creates no VerificationRecord."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    count_before = len(knowledge.verifications)

    knowledge.evidence_for_model("model A")
    knowledge.evidence_for_model("model B")

    assert len(knowledge.verifications) == count_before


def test_l_summary_never_calls_the_laboratory(lab, tmp_path):
    """TEST L: summarizing is a pure store query — physics is never invoked."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    agent = ScientistAgent(lab, knowledge)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    knowledge.evidence_for_model("model A")
    knowledge.evidence_for_model("model B")

    assert calls == []


def test_m_evidence_summary_channel_is_truth_free():
    """TEST M: the summary's home modules never import the universe layer —
    no leakage into the summary. (The "secrets"/"y_true" vocabulary check is
    scoped to prediction.py, where EvidenceSummary lives; knowledge.py's
    docstring legitimately mentions "secrets" as a boundary concept.)"""
    knowledge_module = sys.modules["pwarm.scientist.knowledge"]
    for module in (prediction_module, knowledge_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")
    prediction_src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    assert "secrets" not in prediction_src and "y_true" not in prediction_src


def test_step8_full_scenario_no_verdict_fields(tmp_path):
    """The complete Step 8 deliverable: EvidenceSummary(A) and
    EvidenceSummary(B) are correct, E1/E2/E3 link correctly, and there is no
    winner, weight, elimination or belief anywhere in the summary."""
    knowledge, _, _, _, _ = _step8_scenario(tmp_path)
    summary_a = knowledge.evidence_for_model("model A")
    summary_b = knowledge.evidence_for_model("model B")

    # Model A: 3 independent experiments, 2 confirmed 1 refuted
    assert summary_a.verification_count == 3
    assert summary_a.confirmed_count == 2 and summary_a.refuted_count == 1
    assert summary_a.independent_experiment_count == 3
    assert summary_a.experiment_ids == ("E1", "E2", "E3")

    # Model B: 2 independent experiments, 1 confirmed 1 refuted
    assert summary_b.verification_count == 2
    assert summary_b.confirmed_count == 1 and summary_b.refuted_count == 1
    assert summary_b.independent_experiment_count == 2
    assert summary_b.experiment_ids == ("E1", "E2")

    # the summaries expose only evidence-accounting fields — never a verdict
    allowed = {"model_ref", "prediction_ids", "experiment_ids",
               "independent_prediction_count", "independent_experiment_count",
               "independent_evidence_count", "independent_evidence",
               "verification_count",
               "confirmed_count", "refuted_count", "residuals", "statuses",
               "evidence"}
    for summary in (summary_a, summary_b):
        assert set(dataclasses.asdict(summary)) == allowed
        assert not any("winner" in str(v) or "survivor" in str(v)
                       for v in summary.evidence)


# -- Genesis Step 9: independent evidence semantics ---------------------------
#
# verification_count (all persisted records) is NOT the independent evidence
# count. Independent evidence is defined as a distinct (prediction_id,
# experiment_id) pair: re-verifying the SAME prediction against the SAME
# experiment adds no independent evidence, while a distinct prediction OR a
# distinct experiment does. These tests pin that definition and the three
# independent counts it implies.

def _duplicate_verify(knowledge, model_ref, prediction, experiment_id,
                      observed, status):
    """Re-verify an EXISTING prediction against an experiment (append-only)."""
    residual = observed - prediction.predicted
    outcome = PredictionOutcome(
        claim=prediction.claim, predicted=prediction.predicted,
        observed=observed, residual=residual, status=status)
    return knowledge.record_verification(
        prediction.prediction_id, experiment_id, outcome)


def test_step9_a_duplicate_verification_not_independent(tmp_path):
    """TEST A: the same prediction re-verified against the SAME experiment
    grows verification_count but NEVER the independent counts."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    preds, _ = _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),     # P1 -> E1 -> confirmed
    ])
    _duplicate_verify(knowledge, "model X", preds[0], "E1", 5.0, "confirmed")
    summary = knowledge.evidence_for_model("model X")

    assert summary.verification_count == 2            # both persisted
    assert summary.independent_prediction_count == 1  # one prediction
    assert summary.independent_experiment_count == 1  # one experiment
    assert summary.independent_evidence_count == 1    # ONE pair, not repeated
    assert len(summary.independent_evidence) == 1


def test_step9_b_same_prediction_different_experiments(tmp_path):
    """TEST B: one prediction verified against several DISTINCT experiments IS
    multiple independent evidence — each experiment is an independent
    observation testing the same claim."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    preds, _ = _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),     # P1 -> E1 -> confirmed
    ])
    _duplicate_verify(knowledge, "model X", preds[0], "E2", 5.0, "confirmed")
    _duplicate_verify(knowledge, "model X", preds[0], "E3", 5.0, "confirmed")
    summary = knowledge.evidence_for_model("model X")

    assert summary.verification_count == 3
    assert summary.independent_prediction_count == 1    # one prediction
    assert summary.independent_experiment_count == 3    # three experiments
    assert summary.independent_evidence_count == 3      # three distinct pairs
    assert {v.experiment_id for v in summary.independent_evidence} == \
        {"E1", "E2", "E3"}


def test_step9_c_different_predictions_same_experiment(tmp_path):
    """TEST C: several DISTINCT predictions verified against ONE experiment is
    multiple independent evidence at the pair level (one per prediction), BUT
    they all share a single experimental observation — a caveat the future
    weighting step must respect."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),     # P1 -> E1 -> confirmed
        ("quadratic", 9.0, 0.5, "E1", 8.0),  # P2 -> E1 -> refuted
    ])
    summary = knowledge.evidence_for_model("model X")

    assert summary.verification_count == 2
    assert summary.independent_prediction_count == 2   # two predictions
    assert summary.independent_experiment_count == 1   # one experiment
    assert summary.independent_evidence_count == 2     # two distinct pairs
    assert summary.experiment_ids == ("E1",)
    assert summary.independent_experiment_count < summary.independent_evidence_count


def test_step9_d_different_prediction_and_experiment(tmp_path):
    """TEST D: distinct prediction AND distinct experiment is the clearest
    independent evidence — all independent counts coincide."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),     # P1 -> E1 -> confirmed
        ("quadratic", 9.0, 0.5, "E2", 8.0),  # P2 -> E2 -> refuted
    ])
    summary = knowledge.evidence_for_model("model X")

    assert summary.verification_count == 2
    assert summary.independent_prediction_count == 2
    assert summary.independent_experiment_count == 2
    assert summary.independent_evidence_count == 2
    assert summary.experiment_ids == ("E1", "E2")


def test_step9_e_two_models_counted_independently(tmp_path):
    """TEST E: two models' independent evidence are computed independently,
    even when they share experiments and predictions never overlap."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    a_preds, _ = _commit_and_verify(knowledge, "model A", [
        ("linear", 5.0, 0.5, "E1", 5.0),     # A1 -> E1 -> confirmed
        ("linear", 6.0, 0.5, "E2", 6.0),     # A2 -> E2 -> confirmed
    ])
    _duplicate_verify(knowledge, "model A", a_preds[0], "E1", 5.0, "confirmed")
    _commit_and_verify(knowledge, "model B", [
        ("quadratic", 8.0, 0.5, "E1", 9.0),  # B1 -> E1 -> refuted
        ("quadratic", 9.0, 0.5, "E3", 8.0),  # B2 -> E3 -> refuted
    ])
    summary_a = knowledge.evidence_for_model("model A")
    summary_b = knowledge.evidence_for_model("model B")

    assert summary_a.verification_count == 3
    assert summary_a.independent_evidence_count == 2   # (A1,E1) + (A2,E2)
    assert summary_b.verification_count == 2
    assert summary_b.independent_evidence_count == 2
    assert set(summary_a.prediction_ids).isdisjoint(set(summary_b.prediction_ids))
    # the shared E1 counts once for each model, but never leaks across models
    assert summary_a.independent_experiment_count == 2
    assert summary_b.independent_experiment_count == 2


def test_step9_f_deterministic(tmp_path):
    """TEST F: identical store -> byte-identical independent evidence."""
    def _run(path):
        knowledge = KnowledgeBase(path, universe="universe_001")
        preds, _ = _commit_and_verify(knowledge, "model X", [
            ("linear", 5.0, 0.5, "E1", 5.0),
        ])
        _duplicate_verify(knowledge, "model X", preds[0], "E2", 5.0, "confirmed")
        return knowledge.evidence_for_model("model X")

    assert _run(tmp_path / "a.json") == _run(tmp_path / "b.json")


def test_step9_g_no_record_modification(tmp_path):
    """TEST G: summarizing changes no persisted record — predictions and
    verifications (including their order) are untouched."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),
        ("quadratic", 9.0, 0.5, "E1", 8.0),
    ])
    before_predictions = dict(knowledge.predictions)
    before_verifications = dict(knowledge.verifications)

    knowledge.evidence_for_model("model X")

    assert knowledge.predictions == before_predictions
    assert knowledge.verifications == before_verifications


def test_step9_h_no_new_verification(tmp_path):
    """TEST H: summarizing creates no VerificationRecord."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),
    ])
    count_before = len(knowledge.verifications)

    knowledge.evidence_for_model("model X")

    assert len(knowledge.verifications) == count_before


def test_step9_i_no_physics_call(lab, tmp_path):
    """TEST I: summarizing is a pure store query — the Laboratory is never
    called."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),
    ])
    agent = ScientistAgent(lab, knowledge)
    calls: list[ExperimentSpec] = []
    real_run = agent.laboratory.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    agent.laboratory.run_experiment = spy_run
    knowledge.evidence_for_model("model X")

    assert calls == []


def test_step9_j_no_belief_change(tmp_path):
    """TEST J: summarizing never touches the AI's self-model (beliefs)."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),
    ])
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span

    knowledge.evidence_for_model("model X")

    assert state.belief("gravity").span == span_before


def test_step9_k_no_truth_leakage():
    """TEST K: the independent-evidence logic lives in universe-free modules —
    no truth vocabulary, no universe import."""
    knowledge_module = sys.modules["pwarm.scientist.knowledge"]
    for module in (prediction_module, knowledge_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")
    prediction_src = Path(prediction_module.__file__).read_text(encoding="utf-8")
    assert "secrets" not in prediction_src and "y_true" not in prediction_src


def test_step9_combined_history_counts_everything(tmp_path):
    """The full distinction in one history: duplicate pairs, same-prediction
    multi-experiment, multi-prediction same-experiment, and clean pairs all
    coexist, and each independent count reflects only the non-redundant set."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    # P1 -> E1 (confirmed) verified TWICE (one duplicate)
    p1, _ = _commit_and_verify(knowledge, "model X", [
        ("linear", 5.0, 0.5, "E1", 5.0),
    ])
    _duplicate_verify(knowledge, "model X", p1[0], "E1", 5.0, "confirmed")
    # P1 -> E2 (confirmed): same prediction, new experiment
    _duplicate_verify(knowledge, "model X", p1[0], "E2", 5.0, "confirmed")
    # P2 -> E1 (refuted): new prediction, same experiment as P1
    p2, _ = _commit_and_verify(knowledge, "model X", [
        ("quadratic", 9.0, 0.5, "E1", 8.0),
    ])
    # P3 -> E3 (confirmed): clean pair
    p3, _ = _commit_and_verify(knowledge, "model X", [
        ("linear", 7.0, 0.5, "E3", 7.0),
    ])

    summary = knowledge.evidence_for_model("model X")

    assert summary.verification_count == 5              # 5 persisted records
    assert summary.independent_prediction_count == 3    # P1, P2, P3
    assert summary.independent_experiment_count == 3    # E1, E2, E3
    # distinct pairs: (P1,E1) (P1,E2) (P2,E1) (P3,E3) = 4
    assert summary.independent_evidence_count == 4
    assert len(summary.independent_evidence) == 4
    pairs = {(v.prediction_id, v.experiment_id)
             for v in summary.independent_evidence}
    assert pairs == {(p1[0].prediction_id, "E1"),
                     (p1[0].prediction_id, "E2"),
                     (p2[0].prediction_id, "E1"),
                     (p3[0].prediction_id, "E3")}
    # the duplicate (P1,E1) collapses to a single representative
    assert len([v for v in summary.independent_evidence
                if v.prediction_id == p1[0].prediction_id
                and v.experiment_id == "E1"]) == 1


# -- Genesis Step 10: competition state (facts per model + experiment) --------
#
# The statistical unit is (model_ref, experiment_id): one experiment is one
# physics observation. Confirmed/refuted are FACTS, not scores — no winner,
# no weight, no probability, no elimination, ever, at this layer.

def _step10_scenario(tmp_path):
    """The Step 10 case: Model A (E1/C, E2/C, E3/R) and Model B (E1/R,
    E2/C, E3/C) — shared experiments E1/E2, independent E3 verdicts."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model A", [
        ("linear", 5.0, 0.5, "E1", 5.0),        # E1 -> CONFIRMED
        ("linear", 6.0, 0.5, "E2", 6.0),        # E2 -> CONFIRMED
        ("quadratic", 9.0, 0.5, "E3", 8.0),     # E3 -> REFUTED
    ])
    _commit_and_verify(knowledge, "model B", [
        ("quadratic", 8.0, 0.5, "E1", 9.0),     # E1 -> REFUTED
        ("linear", 6.0, 0.5, "E2", 6.0),        # E2 -> CONFIRMED
        ("linear", 7.0, 0.5, "E3", 7.0),        # E3 -> CONFIRMED
    ])
    return knowledge


def test_full_scenario_yields_correct_states(tmp_path):
    """The complete case: both models' states count 3 experiments with
    the right confirmed/refuted splits, conflicts empty."""
    knowledge = _step10_scenario(tmp_path)
    state_a = knowledge.competition_state("model A")
    state_b = knowledge.competition_state("model B")

    assert state_a.model_ref == "model A"
    assert state_a.experiment_ids == ("E1", "E2", "E3")
    assert state_a.independent_experiment_count == 3
    assert state_a.confirmed_count == 2 and state_a.refuted_count == 1
    assert state_a.conflicts == ()

    assert state_b.experiment_ids == ("E1", "E2", "E3")
    assert state_b.independent_experiment_count == 3
    assert state_b.confirmed_count == 2 and state_b.refuted_count == 1
    assert state_b.conflicts == ()


def test_three_independent_experiments_counted_once_each(tmp_path):
    """TEST A: one model, E1 confirmed + E2 confirmed + E3 refuted ->
    exactly 3 independent experiments."""
    knowledge = _step10_scenario(tmp_path)
    state = knowledge.competition_state("model A")
    assert state.independent_experiment_count == 3
    assert state.experiment_ids == ("E1", "E2", "E3")


def test_reverification_does_not_add_experiments(tmp_path):
    """TEST B: re-verifying the same prediction against the same
    experiment adds a VerificationRecord but NO experiment."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model A", [("linear", 5.0, 0.5, "E1", 5.0)])
    # re-verify the SAME prediction against the SAME experiment
    prediction = next(iter(knowledge.predictions.values()))
    outcome = PredictionOutcome(claim="linear", predicted=5.0, observed=5.01,
                                residual=0.01, status="confirmed")
    knowledge.record_verification(prediction.prediction_id, "E1", outcome)

    state = knowledge.competition_state("model A")
    assert state.independent_experiment_count == 1
    assert state.confirmed_count == 1
    assert len(knowledge.verifications) == 2        # both records persisted


def test_same_model_different_experiments_counted_separately(tmp_path):
    """TEST C: one model across E1/E2/E3 -> three distinct experiments."""
    knowledge = _step10_scenario(tmp_path)
    state = knowledge.competition_state("model A")
    assert state.experiment_ids == ("E1", "E2", "E3")


def test_shared_experiment_counts_once_per_model(tmp_path):
    """TEST D: E1 is shared by both models — each state lists it once,
    with its OWN verdict (A confirmed, B refuted); the shared observation
    is never mistaken for two physics experiments within one model."""
    knowledge = _step10_scenario(tmp_path)
    state_a = knowledge.competition_state("model A")
    state_b = knowledge.competition_state("model B")
    assert "E1" in state_a.experiment_ids and "E1" in state_b.experiment_ids
    assert state_a.experiment_ids.count("E1") == 1
    assert state_b.experiment_ids.count("E1") == 1
    # each model's own verdict on E1 comes from its own prediction
    assert state_a.confirmed_count == 2 and state_a.refuted_count == 1
    assert state_b.refuted_count == 1


def test_consistent_predictions_collapse_to_one_experiment(tmp_path):
    """TEST E (agreement): the same model, two predictions, same
    experiment, both confirmed -> ONE experiment, no conflict."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model A", [
        ("linear", 5.0, 0.5, "E1", 5.0),
        ("linear", 5.02, 0.5, "E1", 5.0),       # second prediction, same E1
    ])
    state = knowledge.competition_state("model A")
    assert state.independent_experiment_count == 1
    assert state.confirmed_count == 1 and state.refuted_count == 0
    assert state.conflicts == ()


def test_conflicting_predictions_reported_not_guessed(tmp_path):
    """TEST E (conflict): the same model, two predictions, same
    experiment, disagreeing verdicts -> the experiment is reported as a
    conflict and counted NOWHERE else; no tie-break is invented."""
    knowledge = KnowledgeBase(tmp_path / "k.json", universe="universe_001")
    _commit_and_verify(knowledge, "model A", [
        ("linear", 5.0, 0.5, "E1", 5.0),        # confirmed
        ("quadratic", 25.0, 0.5, "E1", 5.0),    # refuted, same E1
    ])
    state = knowledge.competition_state("model A")
    assert state.independent_experiment_count == 1
    assert state.conflicts == ("E1",)
    assert state.confirmed_count == 0 and state.refuted_count == 0


def test_state_fields_are_facts_not_scores(tmp_path):
    """TEST F: the state carries exactly the fact fields — no winner, no
    weight, no probability, no elimination, no score of any kind."""
    _step10_scenario(tmp_path)
    assert [f.name for f in dataclasses.fields(CompetitionState)] == [
        "model_ref", "experiment_ids", "independent_experiment_count",
        "confirmed_count", "refuted_count", "conflicts",
        "provenance", "verifications"]


def test_state_is_deterministic(tmp_path):
    """TEST G: repeated queries return byte-identical states."""
    knowledge = _step10_scenario(tmp_path)
    assert (knowledge.competition_state("model A")
            == knowledge.competition_state("model A"))


def test_state_query_writes_nothing(tmp_path):
    """TEST H: querying a state does not modify the knowledge base — the
    persisted records are identical before and after."""
    knowledge = _step10_scenario(tmp_path)
    predictions_before = dict(knowledge.predictions)
    verifications_before = dict(knowledge.verifications)

    knowledge.competition_state("model A")

    assert knowledge.predictions == predictions_before
    assert knowledge.verifications == verifications_before


def test_state_query_never_calls_the_laboratory(tmp_path):
    """TEST I: the state is a pure read over persisted records."""
    knowledge = _step10_scenario(tmp_path)
    lab = Laboratory(load_universe("universe_001"))
    calls: list[ExperimentSpec] = []
    real_run = lab.run_experiment

    def spy_run(spec):
        calls.append(spec)
        return real_run(spec)

    lab.run_experiment = spy_run
    knowledge.competition_state("model A")
    assert calls == []


def test_state_query_leaves_belief_unchanged(tmp_path):
    """TEST J: querying a state does not touch the scientist's beliefs."""
    knowledge = _step10_scenario(tmp_path)
    state = ScientistState(("gravity",), knowledge)
    span_before = state.belief("gravity").span
    knowledge.competition_state("model A")
    assert state.belief("gravity").span == span_before


def test_state_channel_is_truth_free():
    """TEST K: the state's home modules stay universe-free."""
    knowledge_module = sys.modules["pwarm.scientist.knowledge"]
    for module in (prediction_module, knowledge_module):
        src = Path(module.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pwarm.universes")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pwarm.universes")


def test_provenance_traces_model_to_verification(tmp_path):
    """Full provenance: every (prediction_id, experiment_id,
    verification_id) triple in a state resolves to real persisted
    records owned by that model."""
    knowledge = _step10_scenario(tmp_path)
    state = knowledge.competition_state("model A")
    assert len(state.provenance) == 3
    for prediction_id, experiment_id, verification_id in state.provenance:
        prediction = knowledge.predictions[prediction_id]
        assert prediction.model_ref == "model A"
        verification = knowledge.verifications[verification_id]
        assert verification.prediction_id == prediction_id
        assert verification.experiment_id == experiment_id
