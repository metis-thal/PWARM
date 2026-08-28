"""Impulse-based contact solver for the pymo 2D physics kernel.

Applies sequential impulses to resolve contacts while conserving momentum
(exchange impulses between bodies), then performs a Baumgarte-style positional
correction to eliminate residual penetration.
"""

from __future__ import annotations

import numpy as np

from .bodies import cross
from .collision import Contact


def solve_contacts(contacts: list[Contact], iterations: int = 10, baumgarte: float = 0.2) -> None:
    """Resolve all contacts via sequential impulses.

    Mutates body velocities in place. Momentum is conserved because impulses are
    applied as equal-and-opposite pairs (a receives +j*n, b receives -j*n).
    """
    if not contacts:
        return

    # Velocity-level resolution (impulse solver)
    for _ in range(iterations):
        for c in contacts:
            _apply_impulse(c)

    # Positional correction to remove penetration (Baumgarte)
    for c in contacts:
        _positional_correction(c, baumgarte)


def _apply_impulse(c: Contact) -> None:
    a, b = c.a, c.b
    n = c.normal
    if a.inv_mass == 0 and b.inv_mass == 0:
        return

    ra = c.point - a.pos
    rb = c.point - b.pos

    # Relative velocity at contact point
    va = a.vel + cross_scalar(a.ang_vel, ra)
    vb = b.vel + cross_scalar(b.ang_vel, rb)
    rv = vb - va

    # Normal component
    vn = float(np.dot(rv, n))
    if vn > 0.0:
        return  # separating

    # Effective mass along normal
    raxn = cross(ra, n)
    rbxn = cross(rb, n)
    inv_mass_sum = (
        a.inv_mass + b.inv_mass + a.inv_inertia * raxn**2 + b.inv_inertia * rbxn**2
    )
    if inv_mass_sum < 1e-12:
        return

    e = c.restitution
    j = -(1.0 + e) * vn / inv_mass_sum
    impulse = j * n

    # Apply equal-and-opposite impulses (conserves momentum)
    a.vel = a.vel - a.inv_mass * impulse
    a.ang_vel -= a.inv_inertia * cross(ra, impulse)
    b.vel = b.vel + b.inv_mass * impulse
    b.ang_vel += b.inv_inertia * cross(rb, impulse)

    # Friction (tangential)
    tangent = rv - vn * n
    tnorm = np.linalg.norm(tangent)
    if tnorm > 1e-12:
        tangent = tangent / tnorm
        vt = float(np.dot(rv, tangent))
        jt = -vt / inv_mass_sum
        max_friction = c.friction * abs(j)
        jt = np.clip(jt, -max_friction, max_friction)
        friction_impulse = jt * tangent
        a.vel = a.vel - a.inv_mass * friction_impulse
        a.ang_vel -= a.inv_inertia * cross(ra, friction_impulse)
        b.vel = b.vel + b.inv_mass * friction_impulse
        b.ang_vel += b.inv_inertia * cross(rb, friction_impulse)


def _positional_correction(c: Contact, baumgarte: float) -> None:
    """Push overlapping bodies apart along the contact normal."""
    a, b = c.a, c.b
    total_inv = a.inv_mass + b.inv_mass
    if total_inv < 1e-12 or c.penetration <= 0.0:
        return
    correction = max(c.penetration - 0.001, 0.0) / total_inv * baumgarte
    corr = correction * c.normal
    a.pos = a.pos - a.inv_mass * corr
    b.pos = b.pos + b.inv_mass * corr


def cross_scalar(w: float, r: np.ndarray) -> np.ndarray:
    """2D angular cross product: w x r = (-w*r_y, w*r_x)."""
    return np.array([-w * r[1], w * r[0]])
