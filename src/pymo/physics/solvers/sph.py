"""
SPH Solver — Smoothed Particle Hydrodynamics for fluid simulation.

WCSPH (Weakly Compressible SPH) with boundary particles for fluid-rigid coupling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from ..core.entity import ComponentMask
from . import CouplingData
from .sph_kernels import neighbors_to_csr, sph_density, sph_forces

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.state import State


@dataclass
class SPHOptions:
    particle_radius: float = 0.025
    kernel: str = "cubic"  # "cubic", "quintic", "wendland"
    viscosity: float = 0.1
    # TODO: replace the central-gradient term with proper Akinci cohesion;
    # the current formulation is unphysical, so surface tension defaults off.
    surface_tension: float = 0.0
    rest_density: float = 1000.0
    # Tait EOS bulk-modulus scale B (gamma=7). Sound speed
    # c_s = sqrt(7 * B / rho0) ~= 10 m/s, the textbook artificial sound
    # speed for stable explicit WCSPH.
    stiffness: float = 14286.0


class SPHSolver:
    """WCSPH fluid solver with neighbor search and boundary handling."""

    name = "sph"
    required_components = [ComponentMask.SPH_PARTICLE]

    # Safety cap: bounds the worst-case CFL demand so a frame cannot hang.
    _MAX_SUBSTEPS = 400

    def __init__(self, options: dict | None = None):
        self.options = SPHOptions(**(options or {}))
        self.scene = None
        self._kdtree = None
        self._neighbors = None

    def initialize(self, scene) -> None:
        self.scene = scene

    def step(self, state: State, dt: float, contacts: list) -> State:
        """WCSPH step with CFL-based internal substepping.

        Explicit WCSPH is only stable for dt <= ~0.25 h / (c_s + v_max),
        where c_s = sqrt(7 B / rho0) is the Tait (gamma=7) sound speed. The
        discrete nearest-neighbor pair stiffness is stiffer than the
        continuum acoustic limit: omega ~ 660 * c_s for the cubic spline at
        spacing h/2, and symplectic Euler needs omega * dt < 2. Since
        omega * dt = 660 * factor * h * c/(c+v) is independent of the EOS
        stiffness, the factor must stay below ~0.06; 0.04 leaves margin.
        The engine frame dt is split into as many internal sub-steps as the
        CFL condition demands, so callers keep a render-friendly frame rate.
        """
        new_state = state.copy()

        if state.sph_pos is None or len(state.sph_pos) == 0:
            return new_state

        remaining = float(dt)
        for _ in range(self._MAX_SUBSTEPS):
            if remaining <= 1e-12:
                break
            h = self.options.particle_radius * 2.0  # Smoothing length
            c_s = float(np.sqrt(7.0 * self.options.stiffness
                                / self.options.rest_density))
            v_max = float(np.max(np.linalg.norm(new_state.sph_vel, axis=1)))
            dt_sub = min(0.04 * h / (c_s + v_max), remaining)
            new_state = self._substep(new_state, dt_sub)
            remaining -= dt_sub
        return new_state

    def _substep(self, state: State, dt: float) -> State:
        """One CFL-sized WCSPH sub-step: neighbor search -> density ->
        pressure -> forces -> integrate -> boundary."""
        new_state = state.copy()

        h = self.options.particle_radius * 2.0  # Smoothing length
        h2 = h * h

        # 1. Neighbor search (KD-tree), converted to CSR for the kernels
        neighbors = self._find_neighbors(state.sph_pos, h)
        new_state.sph_neighbor_indices = neighbors
        nbr_start, nbr_flat = neighbors_to_csr(neighbors)

        # 2. Compute density (numba kernel with pure-Python fallback).
        # Self-contribution included for every particle (the previous loop
        # only applied it to the last index due to an indentation bug).
        pos64 = state.sph_pos.astype(np.float64)
        mass64 = state.sph_mass.astype(np.float64)
        density = sph_density(pos64, mass64, nbr_start, nbr_flat, float(h))
        new_state.sph_density = density.astype(np.float32)

        # 3. Compute pressure (Tait EOS, gamma=7 — the standard WCSPH water
        # equation of state). B = stiffness; c_s = sqrt(7 B / rho0).
        ratio = density / self.options.rest_density
        pressure = self.options.stiffness * (ratio ** 7.0 - 1.0)
        pressure = np.maximum(pressure, 0.0)
        new_state.sph_pressure = pressure.astype(np.float32)

        # 4. Compute forces (pressure + viscosity + near-contact repulsion +
        # surface tension in a single neighbor pass; numba kernel with
        # pure-Python fallback). Contact parameters: repulsion below the
        # EOS's p=0 clamp so sparse neighborhoods (free surface, wall piles)
        # cannot interpenetrate. r_c stays below all legitimate compression
        # distances; the stiffness matches the EOS compressive scale.
        d_lat = self.options.particle_radius  # equilibrium lattice spacing
        c_s2 = 7.0 * self.options.stiffness / self.options.rest_density
        contact_rc = 0.35 * h
        contact_acc = c_s2 / d_lat
        forces = sph_forces(
            pos64,
            state.sph_vel.astype(np.float64),
            mass64,
            density,
            pressure,
            nbr_start,
            nbr_flat,
            float(h),
            float(self.options.viscosity),
            float(self.options.surface_tension),
            float(contact_rc),
            float(contact_acc),
        )
        forces = forces.astype(np.float32)

        # 5. External forces (gravity)
        if self.scene:
            forces += state.sph_mass[:, None] * self.scene.gravity

        # 6. Coupling forces from rigid bodies
        if state.sph_rigid_contact_forces is not None:
            forces += state.sph_rigid_contact_forces

        # 7. Time integration (Symplectic Euler) with a wall-aware drift:
        # zero the normal velocity BEFORE the position update when the drift
        # would cross the floor, so a particle stops at the wall instead of
        # tunneling through a pinned particle and clamping onto the exact
        # same position (an exact coincidence double-counts the kernel
        # self-weight and blasts the neighborhood through the Tait EOS).
        pr = self.options.particle_radius
        accel = forces / state.sph_mass[:, None]
        new_vel = state.sph_vel + accel * dt
        drift = new_vel[:, 2] * dt
        cross = (state.sph_pos[:, 2] >= pr) & (state.sph_pos[:, 2] + drift < pr)
        new_vel[cross, 2] = 0.0
        new_state.sph_vel = new_vel
        new_state.sph_pos = state.sph_pos + new_vel * dt

        # 8. Boundary handling (safety net for residual penetration)
        self._handle_boundaries(new_state)

        return new_state

    def _find_neighbors(self, positions: np.ndarray, h: float) -> list[list[int]]:
        """Find neighbors within smoothing radius using KD-tree."""
        tree = cKDTree(positions)
        neighbors = tree.query_ball_tree(tree, h)
        return neighbors

    def _kernel(self, r2: float, h: float) -> float:
        """Cubic spline kernel."""
        h2 = h * h
        q2 = r2 / h2
        if q2 >= 1.0:
            return 0.0
        norm = 8.0 / (np.pi * h**3)  # 3D normalization
        # Pieces split at q = 0.5 (q2 = 0.25), matching sph_kernels.
        if q2 < 0.25:
            return norm * (1.0 - 6.0 * q2 + 6.0 * q2 * np.sqrt(q2))
        else:
            return norm * 2.0 * (1.0 - np.sqrt(q2))**3

    def _grad_kernel(self, r: np.ndarray, r_norm: float, h: float) -> np.ndarray:
        """Gradient of cubic spline kernel (correct derivative, vanishes at q=0)."""
        q = r_norm / h
        if q >= 1.0:
            return np.zeros(3, dtype=np.float32)

        norm = 48.0 / (np.pi * h**3)
        if q < 0.5:
            factor = norm * (3.0 * q * q - 2.0 * q) / h
        else:
            t = 1.0 - q
            factor = -norm * t * t / h

        return r * factor / r_norm if r_norm > 1e-6 else np.zeros(3, dtype=np.float32)

    def _handle_boundaries(self, state: State) -> None:
        """Floor boundary: clamp position, inelastic normal response.

        Zeroing the normal velocity models a solid wall that absorbs the
        normal momentum (deceleration happens through pressure, as in a
        real fluid against a container). The previous dampened reflection
        bounced the bottom layer back UP into the falling column, doubling
        the impact relative velocity and feeding an instability cascade.
        """
        # Floor at z=0
        if state.sph_pos is not None:
            mask = state.sph_pos[:, 2] < self.options.particle_radius
            state.sph_pos[mask, 2] = self.options.particle_radius
            state.sph_vel[mask, 2] = 0.0

    def get_coupling_data(self, state: State) -> CouplingData:
        data = CouplingData()
        if state.sph_pos is not None:
            data.velocities["rigid"] = state.sph_vel
            data.velocities["fem"] = state.sph_vel
            data.velocities["mpm"] = state.sph_vel
            data.temperatures["thermal"] = np.ones(len(state.sph_pos)) * 293.15
        return data

    def apply_coupling(self, state: State, coupling: CouplingData) -> State:
        new_state = state.copy()

        # Rigid body contact forces on SPH particles
        if "rigid" in coupling.forces:
            new_state.sph_rigid_contact_forces = coupling.forces["rigid"]

        # Thermal coupling
        if "thermal" in coupling.temperatures:
            pass  # Temperature affects viscosity, etc.

        return new_state


class SPHFluidBuilder:
    """Helper to create SPH fluid entities."""

    @staticmethod
    def create_box_fluid(
        position: np.ndarray,
        size: np.ndarray,
        spacing: float,
        mass_per_particle: float = 0.001
    ) -> dict:
        """Create SPH fluid particles in a box."""
        from ..core.component import SPHParticleComponent
        from ..core.entity import ComponentMask

        nx = int(size[0] / spacing) + 1
        ny = int(size[1] / spacing) + 1
        nz = int(size[2] / spacing) + 1

        positions = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = position + np.array([i, j, k], dtype=np.float32) * spacing
                    positions.append(pos)

        positions = np.array(positions, dtype=np.float32)
        n = len(positions)

        return {
            "mask": ComponentMask.FLUID,
            "sph_particles": SPHParticleComponent(
                positions=positions,
                velocities=np.zeros((n, 3), dtype=np.float32),
                densities=np.full(n, 1000.0, dtype=np.float32),
                pressures=np.zeros(n, dtype=np.float32),
                masses=np.full(n, mass_per_particle, dtype=np.float32),
            ),
            "particle_count": n,
        }
