"""Epistemic ledger records — the dataclasses of the hypothesis-testing
account.

A :class:`Prediction` is the AI's in-memory guess; a
:class:`PredictionRecord` is its append-only commitment (sha256-anchored);
a :class:`VerificationRecord` is one verdict; :class:`EvidenceSummary` and
:class:`CompetitionState` are deterministic, provenance-complete
aggregations. Facts, never scores: no winner, no weight, no probability,
no elimination lives anywhere in these structures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


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
    verification path in ``KnowledgeBase.record_verification`` and by
    ``KnowledgeBase.supersede``):
    ``open -> confirmed | refuted``; a later verification may overturn a
    confirmation (``confirmed -> refuted``); ``refuted`` is ABSORBING —
    later verifications still append records, but the status never leaves
    refuted; ``open -> superseded`` withdraws a never-verified commitment
    in favor of a NEW prediction that carries ``supersedes`` pointing back
    (a superseded commitment can never be verified). Verdicts are
    immutable history: confirmed/refuted predictions can never be
    superseded, edited or deleted.
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
    status: str = "open"          # open | confirmed | refuted | superseded
    supersedes: str = ""          # prediction_id this commitment replaces


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
    last_verification_at: str = ""            # newest created_at among this
    # model's verifications (informational; "" when none is timestamped)


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

    ``last_verification_at`` is the newest ``created_at`` among the
    model's verifications (ISO strings compare correctly within one
    format); "" when none carries a timestamp.
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
    timestamps = [v.created_at for v in evidence.evidence if v.created_at]
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
        last_verification_at=max(timestamps) if timestamps else "",
    )


@dataclass(frozen=True)
class PredictionOutcome:
    """Transient adjudication result (before persistence)."""

    claim: str
    predicted: float
    observed: float
    residual: float
    status: str                   # confirmed | refuted


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
