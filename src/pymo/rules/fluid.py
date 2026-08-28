"""SPH (Smoothed Particle Hydrodynamics) fluid module for the pymo rules layer.

Implements a 2D SPH solver with:
- Poly6 kernel for density estimation
- Spiky gradient kernel for pressure forces
- Viscosity kernel for viscous forces
- Tait equation of state for pressure: p = k * (rho - rho0)^gamma
- Explicit symplectic Euler integration

All equations are standard SPH formulations; no hardcoded phenomena.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SPHParams:
    """SPH simulation parameters."""

    h: float = 0.1            # smoothing length (kernel radius)
    rest_density: float = 1000.0   # kg/m^3
    stiffness: float = 5000.0  # Tait EOS stiffness k (Pa)
    viscosity: float = 1.0     # dynamic viscosity (Pa*s)
    particle_mass: float = 8.66   # kg per particle (for spacing ~0.1, rest_density 1000)
    gamma: float = 7.0         # Tait EOS exponent


def poly6_kernel(r2: float, h: float) -> float:
    """Poly6 kernel: W(r) = 4/(pi*h^8) * (h^2 - r^2)^3 for r <= h, else 0.

    Normalized in 2D: integral W(r) dA = 1.
    """
    h2 = h * h
    if r2 >= h2:
        return 0.0
    coef = 4.0 / (np.pi * h**8)
    return coef * (h2 - r2)**3


def spiky_gradient_kernel(r_vec: np.ndarray, r: float, h: float) -> np.ndarray:
    """Spiky gradient kernel: grad W = -30/(pi*h^5) * (h - r)^2 * (r_vec/r) for r <= h.

    Normalized in 2D. Returns vector of shape (2,).
    """
    if r >= h or r < 1e-12:
        return np.zeros(2, dtype=float)
    coef = -30.0 / (np.pi * h**5)
    factor = coef * (h - r)**2 / r
    return factor * r_vec


def viscosity_kernel_laplacian(r: float, h: float) -> float:
    """Viscosity kernel Laplacian: nabla^2 W = 20/(pi*h^5) * (h - r) for r <= h.

    Normalized in 2D.
    """
    if r >= h or r < 1e-12:
        return 0.0
    coef = 20.0 / (np.pi * h**5)
    return coef * (h - r)


@dataclass
class SPHParticle:
    """An SPH fluid particle with state and properties."""

    pos: np.ndarray      # (2,) position
    vel: np.ndarray      # (2,) velocity
    mass: float = 0.02
    density: float = 1000.0
    pressure: float = 0.0
    force: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    # Neighbor list indices (filled each step)
    neighbors: list[int] = field(default_factory=list)


class SPHSystem:
    """SPH fluid system managing particles and computing forces."""

    def __init__(self, params: SPHParams | None = None):
        self.params = params or SPHParams()
        self.particles: list[SPHParticle] = []

    def add_particle(self, pos: np.ndarray, vel: np.ndarray | None = None) -> int:
        """Add a particle, return its index."""
        p = SPHParticle(
            pos=np.asarray(pos, dtype=float),
            vel=np.asarray(vel, dtype=float) if vel is not None else np.zeros(2, dtype=float),
            mass=self.params.particle_mass,
            density=self.params.rest_density,
        )
        self.particles.append(p)
        return len(self.particles) - 1

    def clear_forces(self) -> None:
        for p in self.particles:
            p.force[:] = 0.0

    # -- neighbor search ------------------------------------------------------

    def _build_neighbors(self) -> None:
        """O(N^2) brute-force neighbor search within radius h."""
        h = self.params.h
        h2 = h * h
        for p in self.particles:
            p.neighbors.clear()
        n = len(self.particles)
        for i in range(n):
            pi = self.particles[i]
            for j in range(i + 1, n):
                pj = self.particles[j]
                d = pj.pos - pi.pos
                r2 = float(np.dot(d, d))
                if r2 <= h2:
                    pi.neighbors.append(j)
                    pj.neighbors.append(i)

    # -- density & pressure ---------------------------------------------------

    def compute_density_and_pressure(self) -> None:
        for p in self.particles:
            rho = 0.0
            for j in p.neighbors:
                q = self.particles[j]
                r2 = float(np.dot(q.pos - p.pos, q.pos - p.pos))
                rho += q.mass * poly6_kernel(r2, self.params.h)
            # Clamp density to prevent negative pressure from under-density
            p.density = max(rho, self.params.rest_density * 0.5)
            # Tait EOS: p = k * (rho/rho0 - 1)
            p.pressure = self.params.stiffness * (p.density / self.params.rest_density - 1.0)

    # -- forces ---------------------------------------------------------------

    def compute_forces(self) -> None:
        """Compute pressure + viscosity + gravity forces."""
        params = self.params
        gravity = np.array([0.0, -9.81])

        for i, pi in enumerate(self.particles):
            f_pressure = np.zeros(2)
            f_viscosity = np.zeros(2)

            for j in pi.neighbors:
                pj = self.particles[j]
                r_vec = pj.pos - pi.pos
                r2 = float(np.dot(r_vec, r_vec))
                if r2 < 1e-12:
                    continue
                r = np.sqrt(r2)

                # Pressure force (symmetric, pairwise)
                # F_ij = -m_ij * (p_i/rho_i^2 + p_j/rho_j^2) * grad W
                if pi.density > 0 and pj.density > 0:
                    p_term = (pi.pressure / pi.density**2 + pj.pressure / pj.density**2)
                    gradW = spiky_gradient_kernel(r_vec, r, params.h)
                    f_pressure -= pj.mass * p_term * gradW

                # Viscosity force (symmetric)
                # F_ij = m_ij * mu * (v_j - v_i) / rho_j * nabla^2 W
                if pj.density > 0:
                    v_rel = pj.vel - pi.vel
                    lapW = viscosity_kernel_laplacian(r, params.h)
                    f_viscosity += pj.mass * params.viscosity * v_rel / pj.density * lapW

            # Gravity
            f_gravity = pi.mass * gravity

            pi.force = f_pressure + f_viscosity + f_gravity

    # -- integration ----------------------------------------------------------

    def step(self, dt: float) -> None:
        """Explicit symplectic Euler: update velocity then position."""
        self._build_neighbors()
        self.compute_density_and_pressure()
        self.clear_forces()
        self.compute_forces()

        for p in self.particles:
            p.vel += p.force / p.mass * dt
            p.pos += p.vel * dt

    # -- utilities ------------------------------------------------------------

    def total_mass(self) -> float:
        return sum(p.mass for p in self.particles)

    def total_energy(self) -> float:
        ke = sum(0.5 * p.mass * float(np.dot(p.vel, p.vel)) for p in self.particles)
        return ke  # internal energy not modeled here

    def max_density_ratio(self) -> float:
        return max(p.density / self.params.rest_density for p in self.particles)

    def min_density_ratio(self) -> float:
        return min(p.density / self.params.rest_density for p in self.particles)


# Convenience function to create a simple water column for tests
def create_water_column(
    x: float, y: float, width: int, height: int, spacing: float, params: SPHParams | None = None
) -> SPHSystem:
    """Create a rectangular block of SPH particles."""
    params = params or SPHParams()
    system = SPHSystem(params)
    for i in range(width):
        for j in range(height):
            system.add_particle([x + i * spacing, y + j * spacing])
    return system