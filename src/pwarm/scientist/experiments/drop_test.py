"""Drop test — measure a material's restitution from a bouncing sphere.

Physical experiment: drop a sphere of the chosen material from height h onto
a hard anvil (ground with restitution 1.0 so the pair minimum passes the
material's value through). Measure the impact velocity from the free-fall
fit and the rebound velocity from the bounce apex:

    e = v_after / v_before = sqrt(2 g (z_apex - z_contact)) / |v(t_contact)|

e is a velocity RATIO, so it is independent of how well g is known — the
free-fall fit supplies g as a byproduct (which is how the scientist
re-establishes gravity in a new universe).
"""

from __future__ import annotations

import math

import numpy as np

from ..hypothesis import Hypothesis

KIND = "drop_test"
CLAIM_SUFFIX = "restitution"
INFORMS_GRAVITY = True          # every drop re-derives g from its free-fall fit
DESIGN_HEIGHTS = (5.0, 10.0, 20.0, 50.0)


def design_grid(material: str) -> list[dict]:
    """Candidate designs: the same measurement at different drop heights."""
    return [{"material": material, "height": float(h)} for h in DESIGN_HEIGHTS]


def claim_of(design: dict) -> str:
    return f"{design['material']}.restitution"


def relative_resolution(design: dict) -> float:
    from ..information import relative_resolution as _resolution
    return _resolution(KIND, design)


def derive(record, g_known: float | None = None,
           material: str = "material") -> tuple[Hypothesis, float | None]:
    """Derive restitution (and the fitted g) from a bounce record.

    The record spans free fall -> contact -> first bounce apex. The contact
    frame is the lowest sample; the pre-contact parabola fit yields both the
    local g and the impact velocity.
    """
    t, z = record.t, record.z
    if len(t) < 10:
        raise ValueError("drop_test record too short to derive restitution")

    contact = int(np.argmin(z))          # lowest point = first ground contact
    pre_t, pre_z = t[:contact], z[:contact]
    if len(pre_t) < 8:
        raise ValueError("no free-fall segment before contact")

    a_deg2, b_deg1, _c0 = np.polyfit(pre_t, pre_z, 2)   # z = c + b t + a t^2
    g_fit = -2.0 * float(a_deg2)
    v_before = abs(float(b_deg1) + 2.0 * float(a_deg2) * float(t[contact]))

    z_apex = float(np.max(z[contact:]))
    v_after = math.sqrt(max(0.0, 2.0 * g_fit * (z_apex - float(z[contact]))))
    restitution = v_after / v_before if v_before > 1e-9 else 0.0

    pred = np.polynomial.polynomial.polyval(pre_t, [_c0, b_deg1, a_deg2])
    ss_res = float(np.sum((pre_z - pred) ** 2))
    ss_tot = float(np.sum((pre_z - pre_z.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    hypothesis = Hypothesis(
        claim=f"{material}.restitution",
        formula=(f"e = v_after/v_before = {v_after:.4f}/{v_before:.4f} "
                 f"(drop fit: z(t) = {_c0:.3f} + {b_deg1:.4f}*t + {a_deg2:.5f}*t^2)"),
        value=restitution,
        unit="",
        r2=r2,
        source="polynomial",
        experiment_id=record.experiment_id,
    )
    return hypothesis, g_fit
