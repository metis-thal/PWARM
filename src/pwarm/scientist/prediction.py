"""Prediction — the AI's commitment, hashed and persisted BEFORE observing.

Genesis Phase 1 adds the missing epistemic primitive to the loop:

    AI world model -> Prediction -> COMMIT (hash + persist)
        -> Laboratory experiment -> ObservationRecord
        -> verification -> confirmed / refuted -> world model update

A :class:`PredictionRecord` is append-only scientific history: the
committed content is covered by ``committed_hash`` (sha256 over a
deterministic serialization), and the only sanctioned transition is the
status flip ``open -> confirmed | refuted`` performed by the verification
path. A changed mind commits a NEW prediction; historical predictions are
never edited or deleted.

The verification stage (:func:`adjudicate`) uses ONLY the committed
prediction and the new :class:`ObservationRecord` — no engine truth, no
ground-truth answer of any kind, and this module never imports the
universe layer.

Genesis Phase 2, step 1: the guess itself now comes from the scientist's
self-model — :func:`prediction_from_belief` translates one AI-side
:class:`~pwarm.scientist.state.Belief` into a prediction. There is no
hard-coded prior: move the belief and the prediction moves with it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .experiment import ObservationRecord
from .hypothesis import fit_free_fall

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime import would
    from .state import Belief  # cycle (state -> knowledge -> prediction)


# -- Candidate Model Representation --------------------------------------

@dataclass(frozen=True)
class ScientificModel:
    """A candidate scientific model — pure AI-side object.

    Contains only an identifier and parameters. The model does NOT
    store a prediction; predictions are computed by :func:`model_prediction`
    given the model and experimental conditions.

    This object never accesses pwarm.universes, UniverseSecrets,
    ObservationRecord, or any experimental result.
    """

    model_id: str
    params: dict[str, float] = field(default_factory=dict)


def _linear_formula(params: dict[str, float],
                     conditions: dict[str, float]) -> float:
    """y = k * x — linear model."""
    k = params["k"]
    x = conditions["x"]
    return k * x


def _quadratic_formula(params: dict[str, float],
                        conditions: dict[str, float]) -> float:
    """y = k * x^2 — quadratic model."""
    k = params["k"]
    x = conditions["x"]
    return k * x * x


_FORMULAS: dict[str, Callable[[dict[str, float], dict[str, float]], float]] = {
    "linear": _linear_formula,
    "quadratic": _quadratic_formula,
}


def model_prediction(model: ScientificModel,
                      conditions: dict[str, float],
                      tolerance: float = 1.0) -> Prediction:
    """Compute a model's prediction from its parameters and conditions.

    Pure function: given the same model and conditions, always returns
    the same Prediction. Never accesses pwarm.universes, UniverseSecrets,
    ObservationRecord, or any experimental result.

    The formula is determined by ``model_id`` (e.g. "linear",
    "quadratic"). Parameters are read from ``model.params``.

    The ``tolerance`` default serves PRE-commitment ranking only (disagreement
    ranking never reads it). A tolerance that bears on a VERDICT must be
    declared explicitly at commitment time — the competition commit path
    refuses to guess one.
    """
    formula = _FORMULAS.get(model.model_id)
    if formula is None:
        raise ValueError(
            f"unknown model_id {model.model_id!r}; "
            f"known: {list(_FORMULAS.keys())}")
    value = formula(model.params, conditions)
    return Prediction(
        claim=model.model_id,
        value=float(value),
        tolerance=tolerance,
        source=f"model {model.model_id}",
    )


def disagreement(prediction_a: Prediction,
                 prediction_b: Prediction) -> float:
    """Deterministic disagreement between two predictions.

    Minimum version: absolute difference of predicted values.
    Does NOT use uncertainty weighting, information gain,
    Bayesian evidence, or any complex metric.
    """
    return abs(prediction_a.value - prediction_b.value)


# -- Model Competition Step 2: discriminating conditions -------------------

@dataclass(frozen=True)
class ConditionComparison:
    """One candidate condition's discriminating power across the models.

    Step-2 deliverable: information from BEFORE the experiment — how far
    the candidate models would diverge under this condition. It contains
    no verdict: no winner, no survival, no refutation. The experiment,
    its observation and the model verdicts are later steps.
    """

    conditions: tuple[tuple[str, float], ...]
    predictions: tuple[tuple[str, Prediction], ...]   # (model_id, prediction),
    # in the input order of `models` — deterministic
    disagreement: float                                # spread max - min


def rank_discriminating_conditions(
        models: Sequence[ScientificModel],
        candidate_conditions: Sequence[dict[str, float]],
) -> list[ConditionComparison]:
    """Rank candidate conditions by how much they would separate models.

    For each candidate condition, EVERY model predicts under it (pure
    :func:`model_prediction` calls); the condition's disagreement is the
    spread of the predicted values (max - min — for two models exactly
    ``disagreement``). Results are sorted by disagreement DESCENDING with
    a deterministic stable tie-break: equal disagreements keep the input
    order of ``candidate_conditions``.

    Pure and pre-experimental: the only inputs are the models and the
    conditions; no universe access, no observation records, no engine
    truth, no execution, no winner selection. Identical inputs always
    produce an identical ranking.
    """
    if not models:
        raise ValueError("at least one candidate model is required")
    comparisons = []
    for conditions in candidate_conditions:
        predictions = tuple(
            (model.model_id, model_prediction(model, conditions))
            for model in models)
        values = [prediction.value for _, prediction in predictions]
        comparisons.append(ConditionComparison(
            conditions=tuple(sorted(conditions.items())),
            predictions=predictions,
            disagreement=max(values) - min(values),
        ))
    return sorted(comparisons, key=lambda c: c.disagreement, reverse=True)


# -- Genesis Step 4.5: condition binding (model variable -> spec parameter) -

@dataclass(frozen=True)
class ConditionBinding:
    """Explicit AI-side contract between the two condition vocabularies:
    which ExperimentSpec parameter each model condition variable means.

    Binding is DECLARED data, never guessed — an unmapped model variable
    is refused loudly, and nothing here knows about hidden truths or
    experimental results: it only renames condition keys.
    """

    mapping: dict[str, str]     # model variable -> ExperimentSpec parameter

    def translate(self, conditions: dict[str, float]) -> dict[str, float]:
        """Rename model-vocabulary conditions into spec parameters.

        Refuses, loudly, any variable the binding does not cover —
        silence would be a guessed conversion.
        """
        unmapped = [key for key in conditions if key not in self.mapping]
        if unmapped:
            raise ValueError(
                f"condition variable(s) {sorted(unmapped)} are not bound "
                f"to experiment parameters; binding covers "
                f"{sorted(self.mapping)}")
        return {self.mapping[key]: value for key, value in conditions.items()}


# -- Genesis Step 5A: output binding (model output -> observation field) ----

# ObservationRecord's measurement channels (its ``field_names``): the
# identifier (experiment_id) and the bookkeeping counter (steps) are not
# observable channels a model output could map to.
_OBSERVATION_FIELDS = frozenset({"t", "z", "vx"})


@dataclass(frozen=True)
class OutputBinding:
    """Explicit AI-side contract between the model's output vocabulary and
    the observation channel: which ObservationRecord measurement field
    each model output variable means.

    Like ConditionBinding, this is DECLARED data, never guessed — an
    unbound output or a non-observation target is refused loudly.
    """

    mapping: dict[str, str]     # model output variable -> record field

    def field_for(self, output: str) -> str:
        """The observation field this model output variable maps to."""
        if output not in self.mapping:
            raise ValueError(
                f"model output {output!r} is not bound to an observation "
                f"field; binding covers {sorted(self.mapping)}")
        field = self.mapping[output]
        if field not in _OBSERVATION_FIELDS:
            raise ValueError(
                f"{field!r} is not an ObservationRecord measurement field; "
                f"valid: {sorted(_OBSERVATION_FIELDS)}")
        return field


@dataclass(frozen=True)
class ComparisonInput:
    """The aligned pair an adjudicator would consume, and nothing more:
    the model's claimed value plus the observed channel it maps to. No
    verdict exists at this step."""

    prediction: Prediction
    field: str                    # ObservationRecord measurement channel
    observed: tuple[float, ...]   # the channel's measured samples


def comparison_input(prediction: Prediction,
                     record: ObservationRecord,
                     binding: OutputBinding,
                     output: str) -> ComparisonInput:
    """Align a model prediction with its bound observation channel.

    Pure and read-only: extracts the bound field's measured samples from
    the record; computes no confirmed/refuted verdict and updates nothing
    — adjudication is a later step.
    """
    field = binding.field_for(output)
    channels = {"t": record.t, "z": record.z, "vx": record.vx}
    series = channels[field]
    if series is None:
        raise ValueError(
            f"observation field {field!r} is absent in this record "
            f"(experiment {record.experiment_id!r})")
    return ComparisonInput(prediction=prediction, field=field,
                           observed=tuple(float(v) for v in series))


# -- Genesis Step 5B-1: observation reduction (sample sequence -> scalar) ----

# Reduction rules over a measurement channel's sample sequence. First
# version carries exactly one rule, chosen to match the drop experiment's
# recorded semantics (see ObservationReduction).
_REDUCTION_RULES = {
    "first": lambda series: series[0],
}


@dataclass(frozen=True)
class ObservationReduction:
    """Explicit, reproducible contract for reducing a measurement
    channel's sample sequence to the scalar a verdict would compare.

    First version, one rule: ``"first"`` — the channel's first recorded
    sample. This matches the drop experiment's recorded semantics: the
    record starts one integration step after release, so the first z
    sample is ``release_height - g*dt^2`` (semi-implicit Euler) — the
    observation closest to the release height that a drop-height
    prediction claims, with a small constant discretization offset that
    is identical for every release height.

    Declared, never guessed: an unsupported rule, an empty rule, a
    channel mismatch, or an empty sample sequence is refused loudly.
    """

    channel: str      # ObservationRecord measurement field (e.g. "z")
    rule: str         # reduction rule name; only "first" exists

    def reduce(self, comparison: ComparisonInput) -> float:
        """Reduce the comparison's observed samples to one scalar."""
        if comparison.field != self.channel:
            raise ValueError(
                f"reduction targets channel {self.channel!r} but the "
                f"comparison carries {comparison.field!r}")
        rule = _REDUCTION_RULES.get(self.rule)
        if rule is None:
            raise ValueError(
                f"unsupported reduction rule {self.rule!r}; "
                f"supported: {sorted(_REDUCTION_RULES)}")
        if not comparison.observed:
            raise ValueError(
                f"no samples to reduce on channel {self.channel!r}")
        return float(rule(comparison.observed))


# -- Prediction, Commitment, Verification (Phase 1 + Phase 2) ----------

@dataclass(frozen=True)
class Prediction:
    """The AI's pre-experimental guess (in-memory form, before commitment)."""

    claim: str                    # e.g. "gravity"
    value: float                  # predicted value
    tolerance: float              # half-width of the acceptance band
    source: str = ""              # model reference — never the universe

    @property
    def band(self) -> tuple[float, float]:
        return (self.value - self.tolerance, self.value + self.tolerance)


@dataclass(frozen=True)
class PredictionRecord:
    """A committed prediction — append-only scientific history.

    ``committed_hash`` covers the scientific fields: model_ref, claim,
    spec_ref, predicted, tolerance AND the declared verification contract
    (condition_binding, output_binding, output, reduction_channel,
    reduction_rule) — what the AI promised and HOW it may be adjudicated,
    all frozen before the experiment runs. seq / created_at / status are
    bookkeeping and stay outside the hash.

    Status machine (the only sanctioned writes, performed by the
    verification path in ``KnowledgeBase.record_verification``):
    ``open -> confirmed | refuted``; a later verification may overturn a
    confirmation (``confirmed -> refuted``); ``refuted`` is ABSORBING —
    later verifications still append records, but the status never leaves
    refuted. A changed mind commits a NEW prediction; historical
    predictions are never edited or deleted.
    """

    prediction_id: str
    model_ref: str
    claim: str
    spec_ref: str                 # conditions: the planned experiment specs
    predicted: float
    tolerance: float
    # -- the declared verification contract (hashed) ------------------------
    condition_binding: dict[str, str] = field(default_factory=dict)
    # model condition variable -> ExperimentSpec parameter ({} when the
    # proposal already spoke spec vocabulary)
    output_binding: dict[str, str] = field(default_factory=dict)
    # model output variable -> ObservationRecord field ({} only for the
    # legacy belief path, whose contract is the free-fall fit)
    output: str = ""              # the model output variable compared
    reduction_channel: str = ""   # ObservationReduction channel
    reduction_rule: str = ""      # ObservationReduction rule
    # -- bookkeeping (outside the hash) --------------------------------------
    committed_hash: str = ""
    seq: int = 0                  # monotonic commitment index
    created_at: str = ""          # ISO timestamp (informational)
    status: str = "open"          # open | confirmed | refuted


# -- Experiment identity: the spec_ref contract ------------------------------
#
# A committed prediction declares, in ``spec_ref``, WHICH experiments it may
# be verified against: the planned battery's experiment ids joined with
# ", ". A verification persisted into the ledger must reference an
# experiment from this declared set — a prediction committed for E1 is never
# verifiable against E2. The two helpers below are the single definition of
# that format; the commit side serializes through ``format_spec_refs`` and
# the persistence side parses through ``parse_spec_refs``, so the identity
# relationship has exactly one source.

def format_spec_refs(spec_ids: Sequence[str]) -> str:
    """Serialize the planned experiment battery into a PredictionRecord's
    ``spec_ref`` (", "-joined experiment ids)."""
    return ", ".join(spec_ids)


def parse_spec_refs(spec_ref: str) -> tuple[str, ...]:
    """The experiment ids a committed prediction's ``spec_ref`` covers."""
    return tuple(s.strip() for s in spec_ref.split(",") if s.strip())


@dataclass(frozen=True)
class VerificationRecord:
    """Verdict for one committed prediction against ONE new observation.

    ``experiment_id`` must be covered by the owning prediction's committed
    ``spec_ref`` (see ``parse_spec_refs``) — the persistence gate refuses
    any other experiment, so verification ↔ commitment identity can never
    mismatch in the ledger.

    ``created_at`` is informational bookkeeping (like PredictionRecord's):
    temporal ordering of the ledger is carried by the verification ids.
    """

    verification_id: str
    prediction_id: str
    experiment_id: str            # observation reference
    observed: float               # derived from the record (AI-side fit only)
    residual: float               # observed - predicted
    status: str                   # confirmed | refuted
    evidence: str                 # human-readable comparison result
    created_at: str = ""          # ISO timestamp (informational)


@dataclass(frozen=True)
class EvidenceSummary:
    """Deterministic, AI-side aggregation of ONE model's verification history.

    Answers "what experimental evidence does this model currently have?" by
    reading ONLY persisted :class:`PredictionRecord` s and
    :class:`VerificationRecord` s and tracing each verification back through
    ``prediction_id -> model_ref`` — the model is never guessed, and a
    verification whose prediction_id has no matching prediction is skipped
    (its model cannot be known).

    This is a faithful accounting and NOTHING more: no winner, no weight, no
    elimination, no belief, no probability. It is a pure read — it writes no
    verification, changes no record, and never consults the Laboratory. All
    collections are deterministically ordered (verification id order for the
    per-evidence lists; sorted for the distinct id sets).

    The three independent counts are deliberately distinguished from the raw
    count:

    * ``verification_count`` — every persisted VerificationRecord, including
      re-verifications of the same (prediction, experiment) pair.
    * ``independent_prediction_count`` — the number of DISTINCT prediction_ids.
    * ``independent_experiment_count`` — the number of DISTINCT experiment_ids.
    * ``independent_evidence_count`` — the number of DISTINCT
      (prediction_id, experiment_id) pairs, i.e. non-redundant verification
      events. Re-verifying the SAME prediction against the SAME experiment
      does NOT add independent evidence; a distinct prediction OR a distinct
      experiment does.
    * ``independent_evidence`` — one representative :class:`VerificationRecord`
      per distinct (prediction_id, experiment_id) pair (the first, in
      verification-id order), so each independent evidence keeps its
      experiment_id / prediction_id / status / residual.
    """

    model_ref: str
    prediction_ids: tuple[str, ...]           # distinct, sorted
    experiment_ids: tuple[str, ...]           # distinct, sorted
    independent_prediction_count: int         # len(prediction_ids)
    independent_experiment_count: int         # len(experiment_ids)
    independent_evidence_count: int           # distinct (prediction, experiment) pairs
    verification_count: int                   # every persisted VerificationRecord
    confirmed_count: int
    refuted_count: int
    residuals: tuple[float, ...]              # one per verification, deterministic order
    statuses: tuple[str, ...]                 # aligned with residuals
    evidence: tuple[VerificationRecord, ...]  # per-evidence linkage, same order
    independent_evidence: tuple[VerificationRecord, ...]  # one per distinct pair


# -- Genesis Step 10: competition state (facts per model + experiment) -------

@dataclass(frozen=True)
class CompetitionState:
    """One competition model's experimental standing, stated as FACTS.

    The statistical unit is ``(model_ref, experiment_id)``: one experiment
    is one physics observation, so an experiment counts ONCE for a model
    no matter how many of its predictions were verified against it or how
    often. When the same model's predictions disagree about the same
    experiment (one confirmed, one refuted), the state says so explicitly
    via ``conflicts`` instead of guessing which is right.

    Facts, not scores: no winner, no weight, no probability, no
    elimination — those are later semantic decisions. Full provenance is
    preserved: model -> prediction_id -> experiment_id ->
    verification_id -> VerificationRecord.
    """

    model_ref: str
    experiment_ids: tuple[str, ...]           # distinct, sorted
    independent_experiment_count: int         # len(experiment_ids)
    confirmed_count: int                      # experiments with unanimous confirmed
    refuted_count: int                        # experiments with unanimous refuted
    conflicts: tuple[str, ...]                # experiments with disagreeing verdicts
    provenance: tuple[tuple[str, str, str], ...]  # (prediction_id, experiment_id,
    # verification_id) — every verification behind this state, deterministic order
    verifications: tuple[VerificationRecord, ...]  # full provenance records


def competition_state(model_ref: str,
                      evidence: EvidenceSummary) -> CompetitionState:
    """Build one competition model's standing from its evidence summary.

    The statistical unit is ``(model_ref, experiment_id)``: every distinct
    experiment the model has been tested in counts ONCE, no matter how
    many of its predictions were verified against it or how often. An
    experiment's verdict is the agreement of those verifications:
    unanimous confirmed -> confirmed, unanimous refuted -> refuted, and
    any disagreement is reported as a conflict — never silently resolved,
    never tie-broken. Pure function over the summary; facts, not scores.
    """
    by_experiment: dict[str, list[VerificationRecord]] = {}
    for verification in evidence.independent_evidence:   # deterministic order
        by_experiment.setdefault(verification.experiment_id,
                                 []).append(verification)
    experiment_ids = tuple(sorted(by_experiment))
    confirmed: list[str] = []
    refuted: list[str] = []
    conflicts: list[str] = []
    for experiment_id in experiment_ids:
        statuses = {v.status for v in by_experiment[experiment_id]}
        if statuses == {"confirmed"}:
            confirmed.append(experiment_id)
        elif statuses == {"refuted"}:
            refuted.append(experiment_id)
        else:
            conflicts.append(experiment_id)
    return CompetitionState(
        model_ref=model_ref,
        experiment_ids=experiment_ids,
        independent_experiment_count=len(experiment_ids),
        confirmed_count=len(confirmed),
        refuted_count=len(refuted),
        conflicts=tuple(conflicts),
        provenance=tuple((v.prediction_id, v.experiment_id, v.verification_id)
                         for v in evidence.evidence),
        verifications=evidence.evidence,
    )


@dataclass(frozen=True)
class PredictionOutcome:
    """Transient adjudication result (before persistence)."""

    claim: str
    predicted: float
    observed: float
    residual: float
    status: str                   # confirmed | refuted


# A collapsed belief ("known") predicts its own midpoint; without a floor
# the acceptance band would be a knife edge the apparatus could never hit.
_MIN_TOLERANCE = 1e-3


def prediction_from_belief(belief: Belief) -> Prediction:
    """Translate one AI-side belief into a prediction commitment.

    The prediction IS the belief: its midpoint as the value, its own
    interval as the acceptance band (floored for collapsed beliefs). The
    belief comes from the scientist's self-model — formed BEFORE any
    experiment runs; no observation data participates.
    """
    return Prediction(
        claim=belief.name,
        value=belief.midpoint,
        tolerance=max(belief.span / 2.0, _MIN_TOLERANCE),
        source=f"state belief ({belief.status})",
    )


def commitment_payload(record: PredictionRecord) -> dict:
    """The tamper-evident content of a commitment: the scientific fields
    AND the declared verification contract (bookkeeping fields stay
    outside the hash)."""
    return {
        "model_ref": record.model_ref,
        "claim": record.claim,
        "spec_ref": record.spec_ref,
        "predicted": round(float(record.predicted), 12),
        "tolerance": round(float(record.tolerance), 12),
        "condition_binding": dict(record.condition_binding),
        "output_binding": dict(record.output_binding),
        "output": record.output,
        "reduction_channel": record.reduction_channel,
        "reduction_rule": record.reduction_rule,
    }


def commitment_hash(payload: dict) -> str:
    """sha256 over a deterministic serialization (sorted keys, no spaces)."""
    blob = json.dumps(payload, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_commitment(record: PredictionRecord) -> bool:
    """Recompute the commitment hash — False means the committed content
    was edited after the fact (tampering or corruption)."""
    return record.committed_hash == commitment_hash(commitment_payload(record))


def verify_prediction(prediction: Prediction,
                      observed: float) -> PredictionOutcome:
    """Adjudicate ONE prediction against ONE reduced observation scalar.

    This is the single source of the adjudication rule (shared with
    evaluate/adjudicate): ``residual = observed - predicted`` and
    ``confirmed iff abs(residual) <= tolerance``. The observed scalar
    must come from an ObservationReduction — it is accepted as given and
    never re-derived here; the tolerance comes from the prediction and
    is never adjusted to fit the observation.
    """
    residual = float(observed) - prediction.value
    confirmed = abs(residual) <= prediction.tolerance
    return PredictionOutcome(
        claim=prediction.claim,
        predicted=prediction.value,
        observed=float(observed),
        residual=residual,
        status="confirmed" if confirmed else "refuted",
    )


def evaluate(prediction: Prediction,
             record: ObservationRecord) -> PredictionOutcome:
    """Compare a guess with an observation record (pure, AI-side).

    Phase-1 scope: gravity claims from free-fall records, using the same
    AI-side fit as everywhere else. The record is the only input — no
    engine state, no hidden truth. The verdict itself is
    :func:`verify_prediction`'s rule, applied to the fitted value.
    """
    if prediction.claim != "gravity":
        raise ValueError(
            f"Phase-1 predictions cover 'gravity' only, not {prediction.claim!r}")
    if len(record.t) < 5:
        raise ValueError("record too short to evaluate a prediction")
    _, hypothesis = fit_free_fall(record)
    return verify_prediction(prediction, hypothesis.value)


def adjudicate(committed: PredictionRecord,
               record: ObservationRecord) -> PredictionOutcome:
    """Verify one COMMITTED prediction against one NEW observation record.

    The only inputs are the committed prediction (hash-checked first — a
    tampered commitment is refused, never quietly verified) and the record.
    """
    if not verify_commitment(committed):
        raise ValueError(
            f"commitment hash mismatch for {committed.prediction_id}: a "
            "committed prediction may never be edited — supersede it with "
            "a new prediction instead")
    guess = Prediction(claim=committed.claim, value=committed.predicted,
                       tolerance=committed.tolerance,
                       source=committed.model_ref)
    return evaluate(guess, record)
