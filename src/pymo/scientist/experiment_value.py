"""Experiment value — information per unit cost, the currency of real science.

Mission 002 ranked designs by expected information gain alone; with free
experiments that was fine. Mission 003 makes experiments COSTLY, so the
designer must rank by VALUE:

    value = expected_utility / cost

Two ideas make the utility honest:

1. **Cost model** — a design's cost tracks the apparatus effort it needs
   (setup + instrument time): taller drops and faster slides cost more,
   and every new instrument brings its own price tag.

2. **Threshold-aware utility** — the mission's goal is KNOWLEDGE (an
   interval below :data:`~pymo.scientist.uncertainty.KNOWN_REL_WIDTH`), not
   merely a narrower one. Because a measurement replaces the belief interval
   with ``estimate ± eps``, an experiment whose resolution cannot cross the
   knowledge threshold can never turn its claim into knowledge — no matter
   how often it is repeated. Such designs keep only a fraction of their
   utility (they are still useful for triage, never sufficient). The
   rational scientist therefore buys the CHEAPEST design that crosses the
   threshold, not the most precise one: a 10 m drop and a 50 m drop both
   deliver "known" restitution, at 1.5 vs 5.5 cost units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .information import relative_resolution
from .uncertainty import KNOWN_REL_WIDTH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .designer import ExperimentProposal

# Utility multiplier for designs whose resolution cannot cross the
# knowledge threshold (triage value only).
COARSE_UTILITY = 0.25

_DROP_SETUP = 0.5      # cost units: rig the sample above the anvil
_SLIDE_SETUP = 0.5     # cost units: fixture + launch mechanism
_BUOYANCY_BASE = 1.0   # cost units: fill the tank, handle the sample


def experiment_cost(kind: str, design: dict) -> float:
    """Apparatus effort of one design, in cost units."""
    if kind == "drop_test":
        return _DROP_SETUP + float(design.get("height", 10.0)) / 10.0
    if kind == "slide_test":
        return _SLIDE_SETUP + float(design.get("v0", 5.0)) / 10.0
    if kind == "buoyancy_test":
        return _BUOYANCY_BASE + float(design.get("depth", 2.0)) / 4.0
    return 1.0


def crosses_threshold(kind: str, design: dict) -> bool:
    """Whether this design's resolution is fine enough to reach "known"."""
    return 2.0 * relative_resolution(kind, design) <= KNOWN_REL_WIDTH + 1e-12


def expected_utility(gain: float, kind: str, design: dict) -> float:
    """Information gain, discounted for designs that cannot yield knowledge."""
    factor = 1.0 if crosses_threshold(kind, design) else COARSE_UTILITY
    return gain * factor


def rank(proposals: list[ExperimentProposal]) -> list[ExperimentProposal]:
    """Sort by expected value (information per cost unit), best first."""
    return sorted(proposals, key=lambda p: p.expected_value, reverse=True)
