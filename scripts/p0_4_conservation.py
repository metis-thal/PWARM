"""Phase 0.4 benchmark: measure energy drift for the Verlet integrator.

Self-contained (does not import test internals).

Usage:
    .\\.venv\\Scripts\\python.exe scripts/p0_4_conservation.py
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymo.kernel.integrators import VerletIntegrator, relative_drift


@njit(cache=True)
def kepler_accel(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Newtonian gravity acceleration for N bodies in 2D/3D (G=1)."""
    g = 1.0
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
            r = float(np.sqrt(np.dot(pos[i] - pos[j], pos[i] - pos[j])))
            if r < 1e-12:
                continue
            pe -= g * masses[i] * masses[j] / r
    return pe


def _two_body_orbit(dt: float, steps: int):
    """Circular Kepler orbit of two equal masses (G=1)."""
    m1 = m2 = 1.0
    masses = np.array([m1, m2])
    r = 1.0
    sep = 2.0 * r
    v_rel = np.sqrt(1.0 * (m1 + m2) / sep)
    v = v_rel / 2.0
    pos = np.array([[-r, 0.0], [r, 0.0]])
    vel = np.array([[0.0, v], [0.0, -v]])
    integ = VerletIntegrator(accel_fn=kepler_accel, dt=dt, adaptive=False)
    integ.initialize(pos, vel, masses)
    pes, kes = [], []
    for _ in range(steps):
        pes.append(_potential_energy(integ.pos, integ.masses))
        kes.append(integ.kinetic_energy())
        integ.step()
    total = np.array(pes) + np.array(kes)
    return integ, total


def main() -> None:
    print("=" * 62)
    print("Phase 0.4 — Verlet energy conservation benchmark")
    print("=" * 62)
    print(f"{'system':<28}{'steps':>6}{'dt':>8}{'drift':>12}")
    print("-" * 62)

    cases = [
        ("2-body (well-resolved)", 0.001, 1000),
        ("2-body (coarse)", 0.01, 1000),
        ("2-body (very coarse)", 0.05, 1000),
    ]
    for name, dt, steps in cases:
        _, total = _two_body_orbit(dt, steps)
        drift = relative_drift(total)
        status = "PASS (<0.1%)" if drift < 0.001 else "FAIL"
        print(f"{name:<28}{steps:>6}{dt:>8.3f}{drift:>12.2e}  {status}")

    # 3-body stability: rigid-rotation equilateral triangle
    dt, steps = 0.001, 1000
    m, g, s = 1.0, 1.0, 1.0
    r_circ = s / np.sqrt(3.0)
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    pos = np.column_stack([r_circ * np.cos(angles), r_circ * np.sin(angles)])
    v = np.sqrt(g * m / (np.sqrt(3.0) * r_circ))
    vel = np.column_stack([-np.sin(angles), np.cos(angles)]) * v
    integ = VerletIntegrator(accel_fn=kepler_accel, dt=dt, adaptive=False)
    integ.initialize(pos, vel, np.ones(3))
    e0 = _potential_energy(integ.pos, integ.masses) + integ.kinetic_energy()
    emax = e0
    for _ in range(steps):
        e = _potential_energy(integ.pos, integ.masses) + integ.kinetic_energy()
        emax = max(emax, e)
        integ.step()
    print(
        f"{'3-body (stable)':<28}{steps:>6}{dt:>8.3f}{abs(emax-e0)/abs(e0):>12.2e}  peak |change|"
    )

    print("-" * 62)
    print("Target: 2-body drift < 0.1% for well-resolved orbit.")


if __name__ == "__main__":
    main()
