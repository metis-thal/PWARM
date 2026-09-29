"""Genesis core — public facade over the split epistemic modules.

The AI's commitment, hashed and persisted BEFORE observing:

    AI world model -> Prediction -> COMMIT (hash + persist)
        -> Laboratory experiment -> ObservationRecord
        -> verification -> confirmed / refuted -> world model update

A :class:`PredictionRecord` is append-only scientific history anchored by
``committed_hash``; the verification stage uses ONLY the committed
prediction and the new :class:`ObservationRecord` — no engine truth, no
ground-truth answer of any kind, and no module here imports the universe
layer.

The implementation lives in four cohesive modules (imported and
re-exported here, so every historical import path keeps working):

* :mod:`pwarm.scientist.models`       — candidate models, rival
  predictions, discriminating-condition ranking, belief translation.
* :mod:`pwarm.scientist.contracts`    — ConditionBinding, OutputBinding,
  the spec_ref identity format, and the declared reduction registry.
* :mod:`pwarm.scientist.records`      — the ledger dataclasses and the
  commitment hash.
* :mod:`pwarm.scientist.adjudication` — the single verdict rule and both
  commitment-honoring adjudication entries.
"""

from .adjudication import (
    LEGACY_REDUCTION,
    adjudicate,
    evaluate,
    verify_prediction,
)
from .contracts import (
    ComparisonInput,
    ConditionBinding,
    ObservationReduction,
    OutputBinding,
    comparison_input,
    format_spec_refs,
    parse_spec_refs,
)
from .models import (
    ConditionComparison,
    ScientificModel,
    disagreement,
    model_prediction,
    prediction_from_belief,
    rank_discriminating_conditions,
)
from .records import (
    CompetitionState,
    EvidenceSummary,
    Prediction,
    PredictionOutcome,
    PredictionRecord,
    VerificationRecord,
    commitment_hash,
    commitment_payload,
    competition_state,
    verify_commitment,
)

__all__ = [
    "LEGACY_REDUCTION",
    "ComparisonInput",
    "CompetitionState",
    "ConditionBinding",
    "ConditionComparison",
    "EvidenceSummary",
    "ObservationReduction",
    "OutputBinding",
    "Prediction",
    "PredictionOutcome",
    "PredictionRecord",
    "ScientificModel",
    "VerificationRecord",
    "adjudicate",
    "commitment_hash",
    "commitment_payload",
    "comparison_input",
    "competition_state",
    "disagreement",
    "evaluate",
    "format_spec_refs",
    "model_prediction",
    "parse_spec_refs",
    "prediction_from_belief",
    "rank_discriminating_conditions",
    "verify_commitment",
    "verify_prediction",
]
