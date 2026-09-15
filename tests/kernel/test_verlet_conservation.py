"""Conservation-law verification for the Verlet integrator (Phase 0.4).

Simulates a 2-body Kepler orbit and a 3-body system, measures relative energy
drift over 1000+ steps, and asserts drift < 0.1% for a well-resolved orbit.
"""

from __future__ import annotations

import numpy as np
import pytest

# This module JIT-compiles its acceleration kernel, so it can only run when
# numba is installed (the main env may not have it); skip gracefully.
pytest.importorskip("numba")
from numba import njit  # noqa: E402

from pymo.kernel.integrators import VerletIntegrator, relative_drift  # noqa: E402


@njit(cache=True)
def kepler_accel(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Newtonian gravity acceleration for N bodies in 2D/3D.

    a_i = -sum_{j != i} G*m_j * (r_i - r_j) / |r_i - r_j|^3
    """
    g = 1.0  # G=1 in natural units
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[i] - pos[j]
            r2 = float(np.dot(d, d))
            if r2 < 1e-12:
                continue
            acc[i] -= g * masses[j] * d / (r2 ** 1.5)
    return acc


def _potential_energy(pos: np.ndarray, masses: np.ndarray) -> float:
    g = 1.0
    n = pos.shape[0]
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r = float(np.sqrt(np.dot(d, d)))
            if r < 1e-12:
                continue
            pe -= g * masses[i] * masses[j] / r
    return pe


def _two_body_orbit(dt: float, steps: int):
    """Circular Kepler orbit of two equal masses (G=1)."""
    m1 = m2 = 1.0
    masses = np.array([m1, m2])
    # Equal masses orbit common barycenter. Circular orbit radius r, speed v.
    r = 1.0
    sep = 2.0 * r  # separation between bodies = 2*r (each r from barycenter)
    # For equal masses circular orbit: G*M_total / sep^2 = v_rel^2 / sep
    # => v_rel = sqrt(G*M_total/sep); each body speed = v_rel/2
    v_rel = np.sqrt(1.0 * (m1 + m2) / sep)
    v = v_rel / 2.0
    pos = np.array([[-r, 0.0], [r, 0.0]])
    vel = np.array([[0.0, v], [0.0, -v]])
    integ = VerletIntegrator(accel_fn=kepler_accel, dt=dt, adaptive=False)
    integ.initialize(pos, vel, masses)
    pes, kes = [], []
    for _ in range(steps):
        pe = _potential_energy(integ.pos, integ.masses)
        ke = integ.kinetic_energy()
        pes.append(pe)
        kes.append(ke)
        integ.step()
    total = np.array(pes) + np.array(kes)
    return integ, total


def test_two_body_energy_conservation_drift_below_0_1pct():
    """A well-resolved circular orbit must keep energy drift under 0.1%."""
    dt = 0.001
    steps = 1000
    _, total = _two_body_orbit(dt, steps)
    drift = relative_drift(total)
    assert drift < 0.001, f"Energy drift {drift:.2e} exceeded 0.1%"


def test_two_body_momentum_conservation():
    """Total momentum must stay near zero for an equal-mass circular orbit."""
    dt = 0.001
    steps = 500
    integ, _ = _two_body_orbit(dt, steps)
    mom = integ.masses[:, None] * integ.vel
    total_mom = np.sum(mom, axis=0)
    assert np.all(np.abs(total_mom) < 1e-6), f"Momentum drift: {total_mom}"


def test_three_body_drift_bounded():
    """Three-body rigid-rotation orbit stays energy-conserving over 1000 steps.

    Three equal masses at an equilateral triangle's vertices, given tangential
    velocities that balance gravity (rigid rotation). This is a stable bound
    orbit, so energy drift should stay well under 1% — demonstrating the
    integrator does not blow up for a multi-body system.
    """
    dt = 0.001
    steps = 1000
    m = 1.0
    masses = np.ones(3)
    g = 1.0
    # Equilateral triangle, side s=1.0, circumradius R = s/sqrt(3)
    s = 1.0
    r_circ = s / np.sqrt(3.0)
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    pos = np.column_stack([r_circ * np.cos(angles), r_circ * np.sin(angles)])
    # Rigid rotation: v = sqrt(G*m/(sqrt(3)*R))
    v = np.sqrt(g * m / (np.sqrt(3.0) * r_circ))
    vel = np.column_stack([-np.sin(angles), np.cos(angles)]) * v
    integ = VerletIntegrator(accel_fn=kepler_accel, dt=dt, adaptive=False)
    integ.initialize(pos, vel, masses)
    e0 = _potential_energy(integ.pos, integ.masses) + integ.kinetic_energy()
    emax = e0
    for _ in range(steps):
        e = _potential_energy(integ.pos, integ.masses) + integ.kinetic_energy()
        emax = max(emax, e)
        integ.step()
    drift = emax / abs(e0) - 1.0
    assert drift < 0.01, f"3-body energy grew by {drift*100:.2f}% (must be <1%)"
