"""Adjudication — the single verdict rule, applied through declared
reductions on both verification paths.

``residual = observed - predicted`` and ``confirmed iff
abs(residual) <= tolerance`` — :func:`verify_prediction` is the ONE source
of that rule. The observed scalar always comes from a declared
:class:`~pwarm.scientist.contracts.ObservationReduction` (accepted as
given, never re-derived here); the tolerance comes from the prediction
and is never adjusted to fit the observation.

The legacy path (:func:`adjudicate`) honors the commitment's DECLARED
reduction — an empty contract is refused, a commitment is never
re-interpreted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import ComparisonInput, ObservationReduction
from .records import (
    Prediction,
    PredictionOutcome,
    PredictionRecord,
    verify_commitment,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; attribute access only
    from .experiment import ObservationRecord


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


# The legacy belief path's declared verification contract: the record's
# vertical channel, reduced by the free-fall fit's gravity scalar.
LEGACY_REDUCTION = ("z", "free_fall_g")


def evaluate(prediction: Prediction,
             record: ObservationRecord) -> PredictionOutcome:
    """Compare a guess with an observation record (pure, AI-side).

    The legacy reduction, applied through the SAME machinery as the
    competition path: the record's channels are reduced by the declared
    ``free_fall_g`` rule (the free-fall fit's gravity scalar) and the
    verdict is :func:`verify_prediction`'s single rule. The coherence
    guard lives with the rule — a non-gravity claim or a too-short record
    is refused by the reduction itself, on every path that uses it.
    """
    channels = {"t": tuple(float(v) for v in record.t),
                "z": tuple(float(v) for v in record.z)}
    comparison = ComparisonInput(prediction=prediction, field="z",
                                 observed=channels["z"], channels=channels)
    observed = ObservationReduction(channel="z",
                                    rule="free_fall_g").reduce(comparison)
    return verify_prediction(prediction, observed)


def adjudicate(committed: PredictionRecord,
               record: ObservationRecord) -> PredictionOutcome:
    """Verify one COMMITTED prediction against one NEW observation record.

    The only inputs are the committed prediction (hash-checked first — a
    tampered commitment is refused, never quietly verified) and the
    record. The commitment's DECLARED reduction is honored: an empty
    contract is refused (a commitment is never re-interpreted), and only
    the legacy ``(z, free_fall_g)`` contract is implemented here —
    competition contracts go through verify_competing_predictions.
    """
    if not verify_commitment(committed):
        raise ValueError(
            f"commitment hash mismatch for {committed.prediction_id}: a "
            "committed prediction may never be edited — supersede it with "
            "a new prediction instead")
    if not committed.reduction_rule:
        raise ValueError(
            f"prediction {committed.prediction_id} was committed without a "
            "verification contract — declare one at commit time (the "
            "legacy path declares ('z', 'free_fall_g')); never re-interpret")
    if (committed.reduction_channel,
            committed.reduction_rule) != LEGACY_REDUCTION:
        raise ValueError(
            f"prediction {committed.prediction_id} declares the reduction "
            f"({committed.reduction_channel!r}, "
            f"{committed.reduction_rule!r}); legacy adjudication implements "
            f"only {LEGACY_REDUCTION} — competition contracts go through "
            "verify_competing_predictions")
    guess = Prediction(claim=committed.claim, value=committed.predicted,
                       tolerance=committed.tolerance,
                       source=committed.model_ref)
    return evaluate(guess, record)
