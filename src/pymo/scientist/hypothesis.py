"""Hypotheses — the AI scientist's claims, derived from observations.

A :class:`Hypothesis` turns a fitted law into a physical claim (e.g. from
``z(t) = c + b*t + a*t^2`` derive ``gravity = -2a``). :func:`verify` then
cross-checks hypotheses from INDEPENDENT experiments: the same constant
emerging from different experimental conditions is what makes a law
scientific, not a single good fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymo.ai import LawDiscovery
from pymo.ai.law_discovery import PolynomialBackend

from .experiment import ObservationRecord

_FREE_FALL_MIN_SAMPLES = 5


@dataclass(frozen=True)
class Hypothesis:
    """One experiment's answer to one claim."""

    claim: str                 # "gravity"
    formula: str               # human-readable fitted law
    value: float               # derived constant (e.g. g in m/s^2)
    unit: str
    r2: float
    source: str                # discovery backend
    experiment_id: str


@dataclass(frozen=True)
class Verification:
    """Cross-experiment verdict for a claim."""

    claim: str
    estimates: list[float]
    mean: float
    rel_spread: float
    mean_r2: float
    confidence: float
    stable: bool


def fit_free_fall(record: ObservationRecord) -> tuple[object, Hypothesis]:
    """Fit ``z(t) = c + b*t + a*t^2`` to one drop record and derive gravity.

    Derivation path: with polynomial backends the native coefficients give
    ``g = -2 * coeffs[2]`` exactly; any other backend is interrogated as a
    black box through ``DiscoveredLaw.predict`` (central second difference,
    exact for quadratics).
    """
    law = LawDiscovery(PolynomialBackend(degree=2)).discover_from_observation(
        record.t, record.z)

    if law.coeffs is not None and len(law.coeffs) >= 3:
        g = -2.0 * float(law.coeffs[2])
    else:
        t0 = 0.5 * (float(record.t[0]) + float(record.t[-1]))
        h = max(1e-3, 0.01 * (float(record.t[-1]) - float(record.t[0])))

        def _f(tt: float) -> float:
            return float(np.asarray(law.predict(np.array([tt]))).ravel()[0])

        g = -((_f(t0 + h) - 2.0 * _f(t0) + _f(t0 - h)) / (h * h))

    hypothesis = Hypothesis(
        claim="gravity",
        formula=f"z(t) = {law.expression}",
        value=g,
        unit="m/s^2",
        r2=float(law.r2),
        source=law.backend,
        experiment_id=record.experiment_id,
    )
    return law, hypothesis


def verify(hypotheses: list[Hypothesis]) -> Verification:
    """Cross-experiment verification: agreement across independent runs.

    ``confidence = mean(R^2) * agreement`` where agreement decays with the
    relative spread of the estimates; ``stable`` requires the spread below
    0.1% and fits essentially perfect — the same constant from different
    drop heights is what makes gravity a law rather than a fit.
    """
    if not hypotheses:
        raise ValueError("no hypotheses to verify")
    estimates = [h.value for h in hypotheses]
    mean = float(np.mean(estimates))
    rel_spread = float(np.std(estimates) / (abs(mean) + 1e-12))
    mean_r2 = float(np.mean([h.r2 for h in hypotheses]))
    confidence = mean_r2 * max(0.0, 1.0 - min(1.0, rel_spread / 0.05))
    return Verification(
        claim=hypotheses[0].claim,
        estimates=estimates,
        mean=mean,
        rel_spread=rel_spread,
        mean_r2=mean_r2,
        confidence=confidence,
        stable=bool(rel_spread < 0.001 and mean_r2 > 0.99),
    )
