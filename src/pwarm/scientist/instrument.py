"""Instruments — how the scientist escapes "I cannot know this".

Mission 003's deepest behavior: when the scientist concludes a claim is
UNIDENTIFIABLE, it does not shrug — it analyzes WHY (no available observable
depends on the parameter), names the capability its apparatus is missing,
and files an :class:`InstrumentRequest`. The instrumentation director (the
system side) checks the universe's instrument catalog and, if the instrument
exists, GRANTS it: the experiment kind is unlocked and the budget is
extended (new instruments are expensive).

    unidentifiable -> gap analysis -> instrument request -> grant
    -> budget extension -> new experiments -> the claim becomes known

The gap-analysis table is the scientist's DOMAIN KNOWLEDGE (physics
textbooks: "to measure density you need buoyancy, or mass and volume") —
prior knowledge about MEASUREMENT, never about the universe's secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .state import ScientistState


@dataclass(frozen=True)
class InstrumentRequest:
    """The scientist's formal statement of what its apparatus is missing."""

    capability: str              # e.g. "fluid_immersion"
    instrument: str              # e.g. "fluid_tank"
    target_claims: tuple[str, ...]
    observable: str              # the physical quantity that would identify them
    reason: str


@dataclass(frozen=True)
class InstrumentGrant:
    """The director's answer: instrument unlocked + budget extension."""

    instrument: str
    capability: str
    enables: str                 # experiment kind unlocked by the instrument
    budget: dict                 # {experiments, simulation_steps, compute_cost} additions


# Domain knowledge: which measurement capability identifies which parameter
# suffix. (Textbook measurement theory — NOT universe secrets.)
_GAP_ANALYSIS: dict[str, dict] = {
    "density": {
        "capability": "fluid_immersion",
        "instrument": "fluid_tank",
        "observable": ("buoyant force F = rho_fluid * g * V on an immersed "
                       "sample — the only macroscopic observable in this "
                       "world that depends on density"),
        "reason": ("drop and slide dynamics are density-invariant (equivalence "
                   "principle: a = g regardless of mass); immersing the sample "
                   "in a fluid produces a buoyant force proportional to the "
                   "displaced volume, making density identifiable"),
    },
}


def analyze_gap(state: ScientistState) -> InstrumentRequest | None:
    """Turn unidentifiable claims into a concrete instrument request.

    Groups unidentifiable beliefs by parameter suffix, looks each suffix up
    in the domain-knowledge table, and returns the first actionable request
    (None when the scientist has no proposal for closing the gap — then the
    honest answer remains "I cannot know this")."""
    unidentifiable = [b for b in state.beliefs.values()
                      if b.status == "unidentifiable"]
    by_suffix: dict[str, list[str]] = {}
    for belief in unidentifiable:
        suffix = belief.name.rsplit(".", 1)[-1]
        by_suffix.setdefault(suffix, []).append(belief.name)

    for suffix in sorted(by_suffix):
        analysis = _GAP_ANALYSIS.get(suffix)
        if analysis is None:
            continue
        return InstrumentRequest(
            capability=analysis["capability"],
            instrument=analysis["instrument"],
            target_claims=tuple(sorted(by_suffix[suffix])),
            observable=analysis["observable"],
            reason=analysis["reason"],
        )
    return None


class InstrumentCatalog:
    """The universe's instrument inventory (system side).

    Locked instruments can be granted ONCE; the grant unlocks the
    experiment kind and extends the budget per the catalog's
    ``grant_budget``. Requests for unknown instruments, mismatched
    capabilities, or already-granted instruments are refused (None)."""

    def __init__(self, specs: dict[str, dict] | None = None):
        self._specs: dict[str, dict] = dict(specs or {})
        self._granted: set[str] = set()

    def grant(self, request: InstrumentRequest) -> InstrumentGrant | None:
        spec = self._specs.get(request.instrument)
        if spec is None or request.instrument in self._granted:
            return None
        if str(spec.get("capability", "")) != request.capability:
            return None
        self._granted.add(request.instrument)
        return InstrumentGrant(
            instrument=request.instrument,
            capability=request.capability,
            enables=str(spec.get("enables", "")),
            budget=dict(spec.get("grant_budget", {})),
        )

    @property
    def granted(self) -> tuple[str, ...]:
        """Instruments granted so far (audit trail)."""
        return tuple(sorted(self._granted))
