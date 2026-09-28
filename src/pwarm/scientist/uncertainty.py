"""Uncertainty — the scientist's honest bookkeeping of what it does not know.

Mission 002 gave every parameter an interval; Mission 003 makes that
uncertainty OPERATIONAL: a belief's width no longer merely describes
knowledge, it drives decisions.

    known           → never re-measured (do not waste budget confirming
                      what you already know)
    measurable      → uncertain enough that an experiment can shrink it
    unidentifiable  → no available apparatus observable depends on it —
                      reported as "I cannot know this (yet)", never guessed

The module is deliberately dependency-light (``state`` is imported only for
type checking) so both :mod:`state` and :mod:`experiment_value` can build on
it without import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .state import Belief, ScientistState

# A belief whose interval is narrower than this fraction of its estimate is
# "known". 5% accommodates the apparatus' best single-shot resolutions
# (slide at v0=20 → 3%, drop at h≥10 → 4%, buoyancy at depth≥2 → ~5%).
KNOWN_REL_WIDTH = 0.05


def relative_width(belief: Belief) -> float:
    """Interval width as a fraction of the estimate (1.0 for a fresh prior)."""
    span = belief.hi - belief.lo
    return abs(span) / max(abs(belief.midpoint), 1e-9)


def certainty_class(belief: Belief) -> str:
    """One of ``known`` | ``measurable`` | ``unidentifiable``."""
    if belief.status == "known":
        return "known"
    if belief.status == "unidentifiable":
        return "unidentifiable"
    return "measurable"


def total_uncertainty(state: ScientistState) -> float:
    """How much the scientist still doesn't know, as one number.

    Sum of clamped relative widths over every non-known belief — the
    dashboard's "remaining uncertainty" gauge and the mission's stop signal
    (0.0 means every parameter is either known or honestly unidentifiable).
    """
    return sum(min(relative_width(b), 1.0)
               for b in state.beliefs.values() if b.status != "known")


def knowledge_lines(state: ScientistState) -> list[tuple[str, str]]:
    """Dashboard KNOWLEDGE panel entries: ``(text, color)`` per belief."""
    lines: list[tuple[str, str]] = []
    for name in sorted(state.beliefs):
        belief = state.beliefs[name]
        cls = certainty_class(belief)
        if cls == "known":
            lines.append((f"  {name} = {belief.midpoint:.4g}  ✓", "#7dffa8"))
        elif cls == "unidentifiable":
            lines.append((f"  {name} = ?  UNIDENTIFIABLE", "#ff9d9d"))
        else:
            lines.append((f"  {name} in [{belief.lo:.3g}, {belief.hi:.3g}",
                          "#ffd479"))
    return lines
