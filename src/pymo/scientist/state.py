"""ScientistState — what the AI knows, what it doesn't, and how tightly.

The state is the uncertainty model of Mission 002's scientific method:

    Unknowns -> Uncertainty Model -> Experiment Designer -> ...

It tracks one :class:`Belief` per manifest parameter (interval + status) and
syncs with the knowledge base: established laws collapse their intervals;
everything else stays honestly unknown. Density in a gravity+contact world is
the canonical example of *unidentifiable* — no apparatus experiment can pin
it down (equivalence principle), and the state says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge import KnowledgeBase
from .uncertainty import KNOWN_REL_WIDTH as _KNOWN_REL_WIDTH

# Prior intervals by parameter-name suffix (the AI's starting ignorance).
DEFAULT_PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "gravity": (0.0, 20.0),
    "air.density": (0.0, 10.0),
    "density": (100.0, 10000.0),
    "restitution": (0.0, 1.0),
    "friction": (0.0, 1.0),
    "young_modulus": (0.0, 1e12),
}

_ABS_FLOOR = 1e-9

# Genesis Phase 2 Step 2 — verification-learning rates (deterministic
# metacognition, not Bayesian inference): a confirmation keeps this share
# of the distance to each boundary; a refutation pads the surprise by this
# share of the old span.
_CONFIRM_KEEP = 0.75
_REFUTED_PAD = 0.25


@dataclass
class Belief:
    """The AI's current knowledge about one parameter."""

    name: str
    lo: float
    hi: float
    status: str = "unknown"    # unknown | constrained | known | unidentifiable
    reason: str = ""

    @property
    def span(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.hi + self.lo)


def _prior_bounds_for(name: str, bounds: dict[str, tuple[float, float]] | None) -> tuple[float, float]:
    if bounds and name in bounds:
        return bounds[name]
    if bounds:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in bounds:
            return bounds[suffix]
    return DEFAULT_PRIOR_BOUNDS.get(name, DEFAULT_PRIOR_BOUNDS.get(suffix_name(name), (0.0, 1.0)))


def suffix_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


class ScientistState:
    """The AI scientist's self-model: known / unknown / how uncertain."""

    def __init__(self, manifest: tuple[str, ...], knowledge: KnowledgeBase,
                 prior_bounds: dict[str, tuple[float, float]] | None = None):
        self.knowledge = knowledge
        self.beliefs: dict[str, Belief] = {}
        for name in manifest:
            lo, hi = _prior_bounds_for(name, prior_bounds)
            self.beliefs[name] = Belief(name=name, lo=lo, hi=hi)
        self.sync_from_knowledge()

    # -- queries -------------------------------------------------------------

    def belief(self, name: str) -> Belief | None:
        return self.beliefs.get(name)

    def unknown_parameters(self) -> list[str]:
        """Parameters not yet established (the designer works on these)."""
        return [n for n, b in self.beliefs.items()
                if b.status in ("unknown", "constrained")]

    def known_parameters(self) -> list[str]:
        return [n for n, b in self.beliefs.items() if b.status == "known"]

    def interval(self, name: str) -> tuple[float, float]:
        b = self.beliefs.get(name)
        return (b.lo, b.hi) if b is not None else (0.0, 0.0)

    # -- updates ---------------------------------------------------------------

    def update(self, name: str, estimate: float, rel_resolution: float) -> None:
        """Collapse a parameter's interval around a measurement.

        The posterior width reflects the experiment's resolution (a noisy
        apparatus leaves a wider interval than a precise one) — science
        progresses by shrinking intervals, not by magic certainty.
        """
        belief = self.beliefs.get(name)
        if belief is None:
            return
        width = max(abs(estimate) * rel_resolution, _ABS_FLOOR)
        belief.lo = estimate - width
        belief.hi = estimate + width
        rel_width = (belief.hi - belief.lo) / max(abs(estimate), _ABS_FLOOR)
        belief.status = "known" if rel_width <= _KNOWN_REL_WIDTH else "constrained"
        belief.reason = ""

    def learn_from_verification(self, claim: str, status: str,
                                observed: float, predicted: float,
                                tolerance: float) -> bool:
        """Genesis Phase 2 Step 2: fold one verification verdict back into
        the self-model. Cognition updating, NOT law publication — the
        knowledge base is untouched, and status transitions stay with
        update / mark_unidentifiable.

        CONFIRMED — the observation landed inside the committed band: pull
        each boundary toward the observation, keeping ``_CONFIRM_KEEP`` of
        the old distance. The interval stays inside the old one, narrows,
        and still covers the observation.

        REFUTED — the observation fell outside the band: widen the belief
        until it covers the surprise (padded by ``_REFUTED_PAD`` of the old
        span). The AI ends up LESS certain — the honest response to being
        wrong, and it can never keep holding the exact same wrong belief.

        Inputs are the existing belief, the verification's AI-side measured
        value, and the committed prediction's reference points — never
        universe truth. Deterministic: identical inputs, identical floats.
        ``predicted``/``tolerance`` ride along as the event's provenance
        (this step's rule reads only status + observed + current belief).
        Returns True when the belief moved.
        """
        belief = self.beliefs.get(claim)
        if belief is None:
            return False
        if status == "confirmed":
            new_lo = observed - _CONFIRM_KEEP * (observed - belief.lo)
            new_hi = observed + _CONFIRM_KEEP * (belief.hi - observed)
        elif status == "refuted":
            pad = _REFUTED_PAD * max(belief.span, _ABS_FLOOR)
            new_lo = min(belief.lo, observed - pad)
            new_hi = max(belief.hi, observed + pad)
        else:
            return False
        belief.lo, belief.hi = new_lo, new_hi
        return True

    def mark_unidentifiable(self, name: str, reason: str) -> None:
        belief = self.beliefs.get(name)
        if belief is None:
            return
        belief.status = "unidentifiable"
        belief.reason = reason

    def revive(self, name: str) -> None:
        """Restore a claim to "unknown" — a NEW instrument may make it
        identifiable where the old apparatus could not (Mission 003's
        instrument arc: unidentifiable -> request -> grant -> measurable)."""
        belief = self.beliefs.get(name)
        if belief is None:
            return
        belief.status = "unknown"
        belief.reason = ""

    def sync_from_knowledge(self) -> None:
        """Collapse intervals for everything the knowledge base establishes."""
        g = self.knowledge.get("gravity")
        if g is not None and "gravity" in self.beliefs:
            b = self.beliefs["gravity"]
            b.lo, b.hi, b.status = g.value, g.value, "known"

        for name, belief in self.beliefs.items():
            if "." not in name:
                continue
            material, prop = name.split(".", 1)
            props = self.knowledge.material(material)
            if props is not None and prop in props:
                value = props[prop]
                belief.lo, belief.hi = value, value
                belief.status = "known"

    # -- reporting -------------------------------------------------------------

    def report(self) -> str:
        """The metacognition summary: I know this / I don't know that / why."""
        lines = ["ScientistState — self model"]
        known = self.known_parameters()
        if known:
            lines.append(f"  known ({len(known)}):")
            for name in known:
                b = self.beliefs[name]
                lines.append(f"    {name} = {b.midpoint:.6g}")
        unknowns = [b for b in self.beliefs.values() if b.status in ("unknown", "constrained")]
        if unknowns:
            lines.append(f"  unknown ({len(unknowns)}):")
            for b in unknowns:
                lines.append(f"    {name_interval(b)}")
        unidentifiable = [b for b in self.beliefs.values() if b.status == "unidentifiable"]
        if unidentifiable:
            lines.append(f"  unidentifiable with current apparatus ({len(unidentifiable)}):")
            for b in unidentifiable:
                lines.append(f"    {b.name} — {b.reason}")
        return "\n".join(lines)


def name_interval(b: Belief) -> str:
    return f"{b.name} in [{b.lo:.4g}, {b.hi:.4g}]"
