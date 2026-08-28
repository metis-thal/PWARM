"""Numerical integrators for the pymo physics kernel.

The physics kernel is the GROUND TRUTH layer of the project. These integrators
advance particle positions/velocities under a user-supplied acceleration
function. All higher layers (rules, AI, viz) treat this as truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

# Type alias: acceleration function a(t, pos) -> accel. pos shape (N, D).
AccelFn = callable


@njit(cache=True)
def _velocity_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
    masses: np.ndarray,
    accel_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """One velocity-Verlet step with fixed dt.

    Parameters
    ----------
    pos : (N, D) current positions.
    vel : (N, D) current velocities.
    acc : (N, D) current accelerations (computed at time t).
    dt : step size.
    masses : (N,) masses.
    accel_fn : callable (pos, masses) -> (N, D) acceleration.

    Returns
    -------
    (new_pos, new_vel) at time t+dt, with acc updated to time t+dt.
    """
    # Half-kick: update velocities by a*dt/2
    vel_half = vel + acc * (0.5 * dt)
    # Drift: advance positions by vel_half*dt
    pos_new = pos + vel_half * dt
    # Recompute acceleration at new positions
    acc_new = accel_fn(pos_new, masses)
    # Second half-kick
    vel_new = vel_half + acc_new * (0.5 * dt)
    return pos_new, vel_new, acc_new


@dataclass
class VerletIntegrator:
    """Velocity-Verlet integrator with optional adaptive timestep.

    Parameters
    ----------
    accel_fn : callable(pos, masses) -> (N, D) acceleration. The physics rule
        that computes forces/accelerations from positions.
    dt : float, default 0.01. Base (or initial) timestep.
    adaptive : bool, default True. Use a simple stability-based adaptive step.
    """

    accel_fn: AccelFn
    dt: float = 0.01
    adaptive: bool = True

    # Internal state
    pos: np.ndarray | None = field(default=None, init=False, repr=False)
    vel: np.ndarray | None = field(default=None, init=False, repr=False)
    acc: np.ndarray | None = field(default=None, init=False, repr=False)
    masses: np.ndarray | None = field(default=None, init=False, repr=False)
    t: float = field(default=0.0, init=False)
    step_count: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False, repr=False)

    def initialize(self, pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> None:
        """Set initial conditions and compute initial accelerations."""
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        masses = np.asarray(masses, dtype=float)
        if pos.shape[0] != vel.shape[0] or pos.shape[0] != masses.shape[0]:
            raise ValueError("pos, vel, masses must have matching leading dims")
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.masses = masses.copy()
        self.acc = np.asarray(self.accel_fn(self.pos, self.masses), dtype=float)
        self.t = 0.0
        self.step_count = 0
        self.history = []

    def step(self, n: int = 1) -> None:
        """Advance the system by `n` integration steps."""
        if self.pos is None:
            raise RuntimeError("VerletIntegrator not initialized")
        for _ in range(n):
            dt = self._effective_dt()
            self.pos, self.vel, self.acc = _velocity_verlet_step(
                self.pos, self.vel, self.acc, dt, self.masses, self.accel_fn
            )
            self.t += dt
            self.step_count += 1

    def _effective_dt(self) -> float:
        """Adaptive step based on max acceleration magnitude."""
        if not self.adaptive or self.acc is None:
            return self.dt
        amax = float(np.max(np.linalg.norm(self.acc, axis=1))) if self.acc.size else 0.0
        if amax <= 1e-12:
            return self.dt
        # Keep displacement per step below a fraction of the characteristic scale
        dt_adapt = np.sqrt(self.dt / max(amax, 1e-12))
        return float(np.clip(dt_adapt, self.dt * 0.5, self.dt))

    def kinetic_energy(self) -> float:
        """Total kinetic energy: sum 0.5*m*|v|^2."""
        if self.vel is None or self.masses is None:
            return 0.0
        return float(0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1)))

    def record(self, extra: dict | None = None) -> None:
        """Append the current state to `history` for observation/replay."""
        if self.pos is None:
            return
        entry = {
            "t": self.t,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "ke": self.kinetic_energy(),
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)


def total_energy(ke: float, pe: float) -> float:
    """Convenience: total mechanical energy."""
    return ke + pe


def relative_drift(series: np.ndarray) -> float:
    """Relative deviation of a scalar series from its initial value.

    Returns ``max(|E(t)-E(0)|) / |E(0)|`` over the series.
    """
    e0 = series[0]
    if abs(e0) < 1e-15:
        return float(np.max(np.abs(series[1:] - e0)))
    return float(np.max(np.abs(series - e0)) / abs(e0))
