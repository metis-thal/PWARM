"""Buoyancy test — the Fluid Tank instrument measures density.

This is Mission 003's payoff experiment: the parameter that was honestly
UNIDENTIFIABLE under the drop/slide apparatus (equivalence principle — no
observable depends on density) becomes identifiable once the scientist can
immerse the sample in a fluid.

Physical experiment: a sphere built at the material's TRUE density
(mass = rho * V) is released at rest below the water surface. The tank
applies the buoyant force F = rho_fluid * g * V, so the descent
acceleration is

    a = g (1 - rho_fluid / rho)      =>      rho = rho_fluid * g / (g - a)

The AI designs the release depth; the fluid density is an INSTRUMENT
specification (a controlled variable of the tank, visible to the AI like
the drop rig's height), and g must already be known (free-fall byproduct).

The observable degrades for very dense samples: rho = rho_f*g/(g-a) is
hyper-sensitive to acceleration error as a -> g (denom -> 0), so the derive
refuses fits in that regime (roughly rho > 50*rho_fluid). Samples near the
fluid density are fine — they barely accelerate, but rho stays resolvable.
"""

from __future__ import annotations

import numpy as np

from ..hypothesis import Hypothesis

KIND = "buoyancy_test"
CLAIM_SUFFIX = "density"
INFORMS_GRAVITY = False
DESIGN_DEPTHS = (1.0, 2.0, 4.0)   # release depth below the surface (m)

# Instrument specification: the tank holds water. This is a controlled
# variable of the apparatus (like the drop rig's height grid), not a
# universe secret — the AI is told what its instruments are.
FLUID_DENSITY = 1000.0            # kg/m^3


def design_grid(material: str) -> list[dict]:
    """Candidate designs: the same immersion at different release depths."""
    return [{"material": material, "depth": float(d)} for d in DESIGN_DEPTHS]


def claim_of(design: dict) -> str:
    return f"{design['material']}.density"


def relative_resolution(design: dict) -> float:
    from ..information import relative_resolution as _resolution
    return _resolution(KIND, design)


def derive(record, g_known: float | None = None,
           material: str = "material") -> tuple[Hypothesis, float | None]:
    """Derive density from the submerged-descent parabola."""
    if g_known is None or g_known <= 0.0:
        raise ValueError(
            "buoyancy_test requires known gravity (establish it with a drop first)")
    t, z = record.t, record.z
    if len(t) < 10:
        raise ValueError("buoyancy_test record too short to derive density")

    a2, b1, c0 = np.polyfit(t, z, 2)      # z = c0 + b1 t + a2 t^2
    a_fit = -2.0 * float(a2)              # downward acceleration magnitude
    denom = g_known - a_fit
    if denom <= 0.02 * g_known:
        raise ValueError(
            "submerged acceleration too close to g to resolve density "
            "(sample too dense for this tank — rho ~ rho_f*g/(g-a) "
            "error amplification saturated)")

    rho = FLUID_DENSITY * g_known / denom

    pred = np.polynomial.polynomial.polyval(t, [c0, b1, a2])
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    hypothesis = Hypothesis(
        claim=f"{material}.density",
        formula=(f"a_submerged = {a_fit:.4f} m/s^2 -> "
                 f"rho = rho_f*g/(g-a) = {rho:.1f} kg/m^3 "
                 f"(rho_f = {FLUID_DENSITY:.0f}, g = {g_known:.4f})"),
        value=rho,
        unit="kg/m^3",
        r2=r2,
        source="polynomial",
        experiment_id=record.experiment_id,
    )
    return hypothesis, None
