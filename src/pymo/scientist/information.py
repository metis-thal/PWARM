"""Information gain — why scientists don't run random experiments.

An experiment is worth running only if it is expected to shrink the
uncertainty of some claim. :func:`expected_information_gain` scores a design
as the fraction of the prior interval it is expected to remove, given the
apparatus' measurement resolution:

    gain = 1 - posterior_span / prior_span      (0 = learns nothing)

The resolution model is honest about the apparatus: state sampling has a
fixed time resolution, so a taller drop (faster impact) pins restitution
down tighter (eps ~ 1/sqrt(h)) and a faster slide gives a longer
deceleration record for friction (eps ~ 1/sqrt(v0)). Parameters no
available experiment can identify (density, in a gravity+contact world)
score exactly 0 — the designer must never spend budget on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .state import ScientistState

# Apparatus resolution constants (relative, at reference conditions).
_EPS_DROP_AT_10M = 0.02      # restitution from a 10 m drop
_EPS_SLIDE_AT_5MS = 0.03     # friction from a 5 m/s slide
_H_REF = 10.0
_V_REF = 5.0


@dataclass(frozen=True)
class MeasurementModel:
    """How tightly one experiment design is expected to pin one claim."""

    kind: str
    claim: str                  # full parameter name, e.g. "material_A.restitution"
    design: dict
    rel_resolution: float       # expected relative eps; inf if unidentifiable

    @property
    def identifiable(self) -> bool:
        return math.isfinite(self.rel_resolution)


def relative_resolution(kind: str, design: dict) -> float:
    """Expected relative measurement resolution of a design (inf = cannot
    measure). Taller drops and faster slides resolve better."""
    if kind == "drop_test":
        height = float(design.get("height", _H_REF))
        return _EPS_DROP_AT_10M * math.sqrt(_H_REF / max(height, 1e-6))
    if kind == "slide_test":
        v0 = float(design.get("v0", _V_REF))
        return _EPS_SLIDE_AT_5MS * math.sqrt(_V_REF / max(v0, 1e-6))
    return float("inf")


def expected_information_gain(state: ScientistState, model: MeasurementModel) -> float:
    """Fraction of the prior interval the design is expected to remove."""
    if not model.identifiable:
        return 0.0
    belief = state.belief(model.claim)
    if belief is None or belief.status == "known":
        return 0.0
    prior_span = belief.hi - belief.lo
    if prior_span <= 0.0:
        return 0.0
    center = max(abs(belief.midpoint), 1e-9)
    posterior_span = 2.0 * model.rel_resolution * center
    gain = 1.0 - min(1.0, posterior_span / prior_span)
    return max(0.0, gain)
