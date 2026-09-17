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
