"""Slide test — measure a material's kinetic friction coefficient.

Physical experiment: launch a box of the chosen material across the ground
at speed v0. Coulomb friction decelerates it linearly:

    v_x(t) = v0 - mu * g * t        (until it stops)

Fitting the deceleration line and dividing its slope by the (already known)
gravitational acceleration yields mu. The experiment REQUIRES gravity to be
established first (Mission 001 provides it; the designer refuses to propose
slide tests otherwise) — a real scientific dependency between experiments.

Apparatus note: the engine's contact solver acts on linear velocity only
(no angular impulses), so the box slides without tipping and the
deceleration is exactly mu * g.
"""

from __future__ import annotations

import numpy as np

from ..hypothesis import Hypothesis

KIND = "slide_test"
CLAIM_SUFFIX = "friction"
DESIGN_SPEEDS = (2.0, 5.0, 10.0, 20.0)
_VX_STOP = 0.05        # m/s — below this the body counts as stopped


def design_grid(material: str) -> list[dict]:
    """Candidate designs: the same measurement at different launch speeds."""
    return [{"material": material, "v0": float(v)} for v in DESIGN_SPEEDS]


def claim_of(design: dict) -> str:
    return f"{design['material']}.friction"


def relative_resolution(design: dict) -> float:
    from ..information import relative_resolution as _resolution
    return _resolution(KIND, design)


def derive(record, g_known: float | None = None,
           material: str = "material") -> tuple[Hypothesis, float | None]:
    """Derive the friction coefficient from the sliding deceleration record."""
    if g_known is None or g_known <= 0.0:
        raise ValueError(
            "slide_test requires known gravity (establish it with a drop first)")
    if record.vx is None or len(record.vx) < 6:
        raise ValueError("slide_test record too short to derive friction")

    sliding = record.vx > _VX_STOP
    t_slide = record.t[sliding]
    v_slide = record.vx[sliding]
    if len(t_slide) < 5:
        return Hypothesis(
            claim=f"{material}.friction",
            formula="stopped too quickly to fit a deceleration line",
            value=0.0, unit="", r2=0.0,
            source="polynomial", experiment_id=record.experiment_id,
        ), None

    slope, intercept = np.polyfit(t_slide, v_slide, 1)   # v = v0 - mu g t
    mu = -float(slope) / float(g_known)

    pred = intercept + slope * t_slide
    ss_res = float(np.sum((v_slide - pred) ** 2))
    ss_tot = float(np.sum((v_slide - v_slide.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    hypothesis = Hypothesis(
        claim=f"{material}.friction",
        formula=(f"v_x(t) = {float(intercept):.4f} + {float(slope):.4f}*t "
                 f"-> mu = {mu:.4f} (g = {g_known:.4f})"),
        value=mu,
        unit="",
        r2=r2,
        source="polynomial",
        experiment_id=record.experiment_id,
    )
    return hypothesis, None
