"""Experiment Designer — the AI decides WHAT to test next, and WHY.

Replaces the fixed planner of Mission 001 with the Mission 002 loop:

    Unknowns -> Uncertainty Model -> Experiment Designer -> ...

:func:`ExperimentDesigner.choose` scores every candidate design (experiment
type x condition grid) by :func:`~pymo.scientist.information.expected_information_gain`
against the current :class:`~pymo.scientist.state.ScientistState` and returns
the single most informative proposal — with an explicit ``reason`` a human
can audit. When no design clears the minimum gain, the designer returns
None: the mission is concluded (or the remaining unknowns are
unidentifiable — see ``unreachable_claims``).

Scientific dependencies are respected: a slide test derives mu from the
deceleration slope DIVIDED by g, so slide designs are only proposed once
gravity is established.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .information import MeasurementModel, expected_information_gain
from .state import ScientistState, suffix_name


@dataclass(frozen=True)
class ExperimentProposal:
    """One chosen experiment, with the reasoning that justifies it."""

    kind: str                      # "drop_test" | "slide_test"
    design: dict                   # {"material": ..., "height": ...} / {"v0": ...}
    claim: str                     # primary parameter this design informs
    expected_gain: float           # fraction of prior interval expected to go
    rel_resolution: float          # expected relative measurement resolution
    reason: str


class ExperimentDesigner:
    """Scores candidate designs by expected information gain."""

    def __init__(self,
                 drop_heights: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0),
                 slide_speeds: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
                 min_gain: float = 0.05):
        self.drop_heights = tuple(drop_heights)
        self.slide_speeds = tuple(slide_speeds)
        self.min_gain = min_gain

    # -- candidate generation ------------------------------------------------

    def _designs_for(self, claim: str) -> list[dict]:
        suffix = suffix_name(claim)
        material = claim.split(".", 1)[0] if "." in claim else None
        if suffix == "restitution":
            return [{"material": material, "height": float(h)}
                    for h in self.drop_heights]
        if suffix == "friction":
            return [{"material": material, "v0": float(v)}
                    for v in self.slide_speeds]
        return []

    def _scored(self, state: ScientistState) -> list[ExperimentProposal]:
        """All informative candidate designs, sorted by expected gain (desc).

        A drop test informs TWO claims at once — its primary target
        (restitution) and gravity as a byproduct of the free-fall fit — so
        it appears under both; duplicates (same kind + design) keep the
        max-gain claim. Slide designs require gravity to be established
        first (mu = -slope/g is undefined otherwise).
        """
        proposals: list[ExperimentProposal] = []
        gravity_known = (state.belief("gravity") is not None
                         and state.belief("gravity").status == "known")
        materials = sorted({n.split(".", 1)[0] for n in state.unknown_parameters()
                            if "." in n})

        for claim in state.unknown_parameters():
            suffix = suffix_name(claim)
            if suffix == "restitution":
                material = claim.split(".", 1)[0]
                for height in self.drop_heights:
                    design = {"material": material, "height": height}
                    eps = _resolution("drop_test", design)
                    gain = expected_information_gain(
                        state, MeasurementModel("drop_test", claim, design, eps))
                    proposals.append(ExperimentProposal(
                        "drop_test", design, claim, gain, eps, ""))
            elif suffix == "friction":
                if not gravity_known:
                    continue
                material = claim.split(".", 1)[0]
                for v0 in self.slide_speeds:
                    design = {"material": material, "v0": v0}
                    eps = _resolution("slide_test", design)
                    gain = expected_information_gain(
                        state, MeasurementModel("slide_test", claim, design, eps))
                    proposals.append(ExperimentProposal(
                        "slide_test", design, claim, gain, eps, ""))
            elif suffix == "gravity":
                # Any drop re-derives g from its free-fall segment.
                if not materials:
                    continue
                for height in self.drop_heights:
                    design = {"material": materials[0], "height": height}
                    eps = _resolution("drop_test", design)
                    gain = expected_information_gain(
                        state, MeasurementModel("drop_test", "gravity", design, eps))
                    proposals.append(ExperimentProposal(
                        "drop_test", design, "gravity", gain, eps, ""))

        # Deduplicate identical (kind, design) pairs across claims.
        best: dict[tuple, ExperimentProposal] = {}
        for p in proposals:
            key = (p.kind, tuple(sorted(p.design.items())))
            if key not in best or p.expected_gain > best[key].expected_gain:
                best[key] = p
        return sorted(best.values(), key=lambda p: p.expected_gain, reverse=True)

    # -- the designer's decision ---------------------------------------------

    def choose(self, state: ScientistState) -> ExperimentProposal | None:
        """The single most informative experiment, or None when nothing
        clears the minimum-gain bar (mission concluded or stuck)."""
        scored = self._scored(state)
        if not scored:
            return None
        best = scored[0]
        if best.expected_gain < self.min_gain:
            return None
        reason = (f"reduce {best.claim} uncertainty: expected gain "
                  f"{best.expected_gain:.2f} of remaining interval "
                  f"({self._describe(best.design)} — best of "
                  f"{len(scored)} candidate designs)")
        return replace(best, reason=reason)

    def unreachable_claims(self, state: ScientistState) -> list[str]:
        """Unknowns NO candidate design informs (density in a gravity+contact
        world; Young's modulus without a deformable solver). These are the
        honest "I cannot know this with my apparatus" parameters."""
        covered = {p.claim for p in self._scored(state)}
        return [n for n in state.unknown_parameters() if n not in covered]

    @staticmethod
    def _describe(design: dict) -> str:
        parts = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in sorted(design.items())]
        return ", ".join(parts)


def _resolution(kind: str, design: dict) -> float:
    from .information import relative_resolution
    return relative_resolution(kind, design)
