"""Experiment designer — the AI scientist's decision-making.

Mission 002 doctrine: rank candidate designs by expected information gain.
Mission 003 doctrine: experiments cost resources, so the designer ranks by
VALUE (expected utility per cost unit) under a live budget, never re-measures
certain claims, only proposes experiment kinds its apparatus actually
supports, and unlocks new kinds when an instrument grant arrives.

The decision pipeline:

    candidate designs -> predicted gain + resolution -> utility (threshold-
    aware) -> value = utility / cost -> affordability filter -> choice
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from . import experiment_value
from .information import MeasurementModel, expected_information_gain, relative_resolution
from .state import ScientistState, suffix_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .budget import ExperimentBudget


@dataclass(frozen=True)
class ExperimentProposal:
    """One candidate experiment, with the economics that justify it."""

    kind: str                      # "drop_test" | "slide_test" | "buoyancy_test"
    design: dict                   # {"material": ..., "height": ...} / {"v0": ...} / {"depth": ...}
    claim: str                     # primary parameter this design informs
    expected_gain: float           # fraction of prior interval expected to go
    rel_resolution: float          # expected relative measurement resolution
    reason: str
    cost: float = 1.0              # Mission 003: apparatus cost units
    expected_value: float = 0.0    # Mission 003: expected utility / cost


class ExperimentDesigner:
    """Scores candidate designs by expected VALUE under the budget."""

    def __init__(self,
                 drop_heights: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0),
                 slide_speeds: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
                 buoyancy_depths: tuple[float, ...] = (1.0, 2.0, 4.0),
                 min_gain: float = 0.05,
                 enabled_kinds: tuple[str, ...] = ("drop_test", "slide_test")):
        self.drop_heights = tuple(drop_heights)
        self.slide_speeds = tuple(slide_speeds)
        self.buoyancy_depths = tuple(buoyancy_depths)
        self.min_gain = min_gain
        self.enabled_kinds = tuple(enabled_kinds)

    def enable_kind(self, kind: str) -> None:
        """Unlock an experiment kind (after an instrument grant)."""
        if kind not in self.enabled_kinds:
            self.enabled_kinds = self.enabled_kinds + (kind,)

    # -- candidate generation ------------------------------------------------

    def _designs_for(self, claim: str) -> list[tuple[str, dict]]:
        """(kind, design) candidates for one claim's parameter suffix."""
        suffix = suffix_name(claim)
        material = claim.split(".", 1)[0] if "." in claim else None
        if suffix == "restitution":
            return [("drop_test", {"material": material, "height": float(h)})
                    for h in self.drop_heights]
        if suffix == "friction":
            return [("slide_test", {"material": material, "v0": float(v)})
                    for v in self.slide_speeds]
        if suffix == "density":
            return [("buoyancy_test", {"material": material, "depth": float(d)})
                    for d in self.buoyancy_depths]
        return []

    def _scored(self, state: ScientistState) -> list[ExperimentProposal]:
        """All informative candidate designs, ranked by expected value.

        A drop test informs TWO claims at once — its primary target
        (restitution) and gravity as a free-fall byproduct — so it appears
        under both; duplicates (same kind + design) keep the highest-value
        claim. Slide and buoyancy designs require gravity to be established
        first (mu = -slope/g and rho = rho_f*g/(g-a) are undefined
        otherwise). Kinds not yet enabled (no instrument grant) never
        appear.
        """
        proposals: list[ExperimentProposal] = []
        gravity_known = (state.belief("gravity") is not None
                         and state.belief("gravity").status == "known")
        materials = sorted({n.split(".", 1)[0] for n in state.unknown_parameters()
                            if "." in n})

        for claim in state.unknown_parameters():
            suffix = suffix_name(claim)
            if suffix == "gravity":
                # Any drop re-derives g from its free-fall segment.
                if not materials:
                    continue
                candidates = [("drop_test",
                               {"material": materials[0], "height": float(h)})
                              for h in self.drop_heights]
            else:
                candidates = self._designs_for(claim)

            for kind, design in candidates:
                if kind not in self.enabled_kinds:
                    continue
                if kind in ("slide_test", "buoyancy_test") and not gravity_known:
                    continue
                eps = relative_resolution(kind, design)
                gain = expected_information_gain(
                    state, MeasurementModel(kind, claim, design, eps))
                cost = experiment_value.experiment_cost(kind, design)
                utility = experiment_value.expected_utility(gain, kind, design)
                proposals.append(ExperimentProposal(
                    kind, design, claim, gain, eps, "",
                    cost=cost,
                    expected_value=utility / cost if cost > 0 else 0.0))

        # Deduplicate identical (kind, design) pairs across claims.
        best: dict[tuple, ExperimentProposal] = {}
        for p in proposals:
            key = (p.kind, tuple(sorted(p.design.items())))
            if key not in best or p.expected_value > best[key].expected_value:
                best[key] = p
        return experiment_value.rank(list(best.values()))

    # -- the designer's decision ---------------------------------------------

    def available_experiments(self, state: ScientistState,
                              budget: ExperimentBudget | None = None
                              ) -> list[ExperimentProposal]:
        """The ranked candidate menu (for dashboards and audits). Callers
        check ``budget.can_afford(p.cost)`` to mark proposals unaffordable."""
        return self._scored(state)

    def choose(self, state: ScientistState,
               budget: ExperimentBudget | None = None
               ) -> ExperimentProposal | None:
        """The best-value affordable experiment, or None when nothing
        clears the minimum-gain bar, nothing is affordable, or the mission
        has concluded."""
        scored = self._scored(state)
        affordable = [p for p in scored
                      if budget is None or budget.can_afford(p.cost)]
        if not affordable:
            return None
        best = affordable[0]
        if best.expected_gain < self.min_gain:
            return None
        if budget is None:
            budget_note = f"best of {len(affordable)} candidate designs"
        else:
            budget_note = (f"best of {len(affordable)} affordable designs; "
                           f"budget {budget.used_experiments}/"
                           f"{budget.experiments} experiments, "
                           f"{budget.used_cost:.1f}/{budget.compute_cost:.1f} cost")
        reason = (f"reduce {best.claim} uncertainty: value "
                  f"{best.expected_value:.2f} = gain {best.expected_gain:.2f} "
                  f"/ cost {best.cost:.1f} ({self._describe(best.design)} — "
                  f"{budget_note})")
        return replace(best, reason=reason)

    def unreachable_claims(self, state: ScientistState) -> list[str]:
        """Unknowns NO enabled candidate design informs (density before the
        fluid tank arrives; Young's modulus without a deformable solver).
        These are the honest "I cannot know this with my apparatus"
        parameters."""
        covered = {p.claim for p in self._scored(state)}
        return [n for n in state.unknown_parameters() if n not in covered]

    @staticmethod
    def _describe(design: dict) -> str:
        parts = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in sorted(design.items())]
        return ", ".join(parts)
