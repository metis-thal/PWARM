"""Thermodynamics rules: heat conduction and temperature fields.

Implements Fourier heat conduction between contacting bodies and a 2D
temperature-field diffusion model (for field visualization and AI observation).

Energy is conserved: heat leaving one body is added to the other, and field
diffusion conserves total field energy.

This lives in the rules layer — it extends the physics kernel's ground truth
with thermal behavior, all derived from equations (no hardcoded phenomena).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymo.kernel.bodies import Body


def heat_capacity(body: Body) -> float:
    """Thermal capacity C = m * c (J/K)."""
    return body.mass * body.material.specific_heat


def internal_energy(body: Body) -> float:
    """Internal thermal energy relative to 0 K: E = C * T."""
    return heat_capacity(body) * body.temperature


class BodyThermalSystem:
    """Handles heat conduction between bodies in contact.

    Uses Fourier's law: dQ/dt = k_eff * A * (T_hi - T_lo) / d
    where k_eff is the contact conductance, A the contact area proxy and d the
    contact gap proxy. Heat flows only between bodies with non-zero
    conductivity (conductors), and total energy is conserved exactly.
    """

    def __init__(self, contact_area: float = 1.0, contact_distance: float = 0.1):
        self.contact_area = contact_area
        self.contact_distance = contact_distance

    def conduct_between(self, a: Body, b: Body, dt: float) -> float:
        """Transfer heat between two bodies in contact over dt.

        Returns the net heat transferred (positive = a loses, b gains).
        Conservation: heat removed from a is exactly added to b.
        """
        ka = a.material.thermal_conductivity
        kb = b.material.thermal_conductivity
        if ka <= 0.0 or kb <= 0.0:
            return 0.0  # insulator blocks conduction
        # Effective conductance (series combination)
        k_eff = 2.0 * ka * kb / (ka + kb)
        ca = heat_capacity(a)
        cb = heat_capacity(b)
        if ca <= 0 or cb <= 0:
            return 0.0
        dT = a.temperature - b.temperature
        if abs(dT) < 1e-12:
            return 0.0
        # Fourier flux
        q = k_eff * self.contact_area * dT / self.contact_distance * dt
        # Limit so neither body overshoots the other's temperature (stability)
        max_q = 0.5 * ca * abs(dT)
        q = np.clip(q, -max_q, max_q)
        # Transfer: heat flows hot -> cold. q > 0 means a (hotter) loses heat.
        a.heat -= q
        b.heat += q
        # Update temperatures from heat (a loses q/ca, b gains q/cb)
        a.temperature -= q / ca
        b.temperature += q / cb
        return q

    def step_contacts(
        self, bodies: list[Body], contacts: list, dt: float
    ) -> float:
        """Conduct heat across a list of (a, b) contact pairs.

        `contacts` is any iterable of objects with `.a` and `.b` attributes
        (e.g. pymo.kernel.collision.Contact) or (Body, Body) tuples.

        Returns total heat transferred.
        """
        total = 0.0
        for c in contacts:
            if hasattr(c, "a") and hasattr(c, "b"):
                total += self.conduct_between(c.a, c.b, dt)
            else:
                a, b = c
                total += self.conduct_between(a, b, dt)
        return total

    def step_pairwise(self, bodies: list[Body], dt: float, contact_fn=None) -> float:
        """Conduct heat between all pairs that are in contact.

        If `contact_fn` is given it should return True if two bodies touch;
        otherwise all pairs conduct (for small scenes / tests).
        """
        total = 0.0
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                if contact_fn is not None and not contact_fn(a, b):
                    continue
                total += self.conduct_between(a, b, dt)
        return total


@dataclass
class TemperatureField:
    """A 2D temperature field on a uniform grid (for diffusion + viz)."""

    nx: int
    ny: int
    dx: float
    dy: float
    temperatures: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.temperatures = np.zeros((self.ny, self.nx), dtype=float)

    def index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(x / self.dx, 0, self.nx - 1))
        iy = int(np.clip(y / self.dy, 0, self.ny - 1))
        return ix, iy

    def set(self, x: float, y: float, value: float) -> None:
        ix, iy = self.index(x, y)
        self.temperatures[iy, ix] = value

    def total_energy(self) -> float:
        """Sum of field temperatures (proxy for energy; conserved by diffusion)."""
        return float(np.sum(self.temperatures))


def diffuse_field(field: TemperatureField, alpha: float, dt: float) -> float:
    """One explicit diffusion step on a field: dT/dt = alpha * laplacian(T).

    Uses a 5-point stencil with Neumann (no-flux) boundaries applied to ALL
    cells (including edges/corners via ghost cells), so total field energy
    (sum of T) is conserved exactly up to floating-point error.

    Stability requires dt < dx^2 / (4*alpha). Returns max |dT|.
    """
    T = field.temperatures
    ny, nx = T.shape
    # Padded array with ghost cells; Neumann BC => ghost = adjacent boundary value
    Tp = np.zeros((ny + 2, nx + 2), dtype=float)
    Tp[1:-1, 1:-1] = T
    # No-flux: set ghost cells equal to the nearest boundary cell
    Tp[0, 1:-1] = T[0, :]      # top
    Tp[-1, 1:-1] = T[-1, :]    # bottom
    Tp[1:-1, 0] = T[:, 0]      # left
    Tp[1:-1, -1] = T[:, -1]    # right
    Tp[0, 0] = T[0, 0]
    Tp[0, -1] = T[0, -1]
    Tp[-1, 0] = T[-1, 0]
    Tp[-1, -1] = T[-1, -1]

    lap = (
        Tp[:-2, 1:-1] + Tp[2:, 1:-1] + Tp[1:-1, :-2] + Tp[1:-1, 2:] - 4.0 * T
    ) / (field.dx * field.dy)
    dT = alpha * lap * dt
    T += dT
    return float(np.max(np.abs(dT)))
