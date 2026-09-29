"""Candidate models — the AI's rival hypotheses and their predictions.

Pure AI-side objects: a :class:`ScientificModel` is an identifier plus
parameters, its predictions are computed by :func:`model_prediction` given
the model and experimental conditions, and :func:`rank_discriminating_conditions`
ranks conditions by how far the rivals would diverge — all BEFORE any
experiment runs. No universe access, no observation records, no engine
truth of any kind.

Genesis Phase 2, step 1: :func:`prediction_from_belief` translates one
AI-side :class:`~pwarm.scientist.state.Belief` into a prediction. There is
no hard-coded prior: move the belief and the prediction moves with it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .records import Prediction

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
        # KL-1: the prediction's model reference IS the registered identity
        # id (KnowledgeBase.register_model) — never a decorated string.
        source=model.model_id,
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
