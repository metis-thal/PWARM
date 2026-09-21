"""Immersion test — Mission 004 Day 1: measure the unknown liquid.

The dual of the Fluid Tank (Mission 003): there the FLUID was the known
instrument specification and the sample carried the hidden density; here
the AMBIENT LIQUID is the universe's hidden truth and the sample is a
CERTIFIED REFERENCE SPHERE — a standard of known density, part of the
apparatus like the drop rig's height scale.

Physical experiment: the rig releases the reference sphere at rest below
the unknown liquid's surface. The same hydrostatic buoyancy mechanism as
Mission 003 applies F = rho_fluid * g * V, so the descent acceleration is

    a = g (1 - rho_fluid / rho_sample)

with rho_sample KNOWN (the certification) — the ambient liquid's density
is therefore inferable in principle from the descent parabola.

Day 1 scope is the measurement infrastructure only:

    immersion_test -> ExperimentSession -> existing buoyancy physics
                   -> ObservationRecord (t, z)

The inference (derive, claim routing for the ambient liquid) is Day 2
work and deliberately absent: the scientist does not yet have a claim
vocabulary for the ambient liquid, and inventing one here would put
information in the AI's hands that it has not earned.
"""

from __future__ import annotations

KIND = "immersion_test"

# Instrument specification: the rig's certified reference sphere, a
# standard of KNOWN density — apparatus metadata, symmetric to
# buoyancy_test.FLUID_DENSITY (the 003 tank's water). It is NOT the
# ambient liquid's density and is never read from the universe: the rig
# builds the sample at exactly this value (physics-side), and it must
# stay independent of every universe's hidden truths.
SAMPLE_DENSITY = 2700.0            # kg/m^3 (certified reference sphere)

DESIGN_DEPTHS = (1.0, 2.0, 4.0)    # release depth below the surface (m)


def design_grid() -> list[dict]:
    """Candidate designs: the certified sphere at different release depths.

    The sphere is fixed apparatus — unlike buoyancy_test there is no
    material choice; the design space is the release depth only.
    """
    return [{"depth": float(d)} for d in DESIGN_DEPTHS]
