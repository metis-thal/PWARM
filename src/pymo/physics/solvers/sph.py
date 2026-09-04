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

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.state import State


@dataclass
class SPHOptions:
    particle_radius: float = 0.025
    kernel: str = "cubic"  # "cubic", "quintic", "wendland"
    viscosity: float = 0.1
    surface_tension: float = 0.072
    rest_density: float = 1000.0
    stiffness: float = 1000.0  # Pressure stiffness


class SPHSolver:
    """WCSPH fluid solver with neighbor search and boundary handling."""
    
    name = "sph"
    required_components = [ComponentMask.SPH_PARTICLE]
    
    def __init__(self, options: dict | None = None):
        self.options = SPHOptions(**(options or {}))
        self.scene = None
        self._kdtree = None
        self._neighbors = None
    
    def initialize(self, scene) -> None:
        self.scene = scene
    
    def step(self, state: State, dt: float, contacts: list) -> State:
        """WCSPH step: neighbor search -> density -> pressure -> forces -> integrate."""
        new_state = state.copy()
        
        if state.sph_pos is None or len(state.sph_pos) == 0:
            return new_state
        
        n = len(state.sph_pos)
        h = self.options.particle_radius * 2.0  # Smoothing length
        h2 = h * h
        
        # 1. Neighbor search
        neighbors = self._find_neighbors(state.sph_pos, h)
        new_state.sph_neighbor_indices = neighbors
        
        # 2. Compute density
        density = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in neighbors[i]:
                if i == j:
                    continue
                r = state.sph_pos[j] - state.sph_pos[i]
                r2 = np.dot(r, r)
                if r2 < h2:
                    density[i] += state.sph_mass[j] * self._kernel(r2, h)
        density[i] += state.sph_mass[i] * self._kernel(0, h)
        new_state.sph_density = density
        
        # 3. Compute pressure (Tait equation)
        pressure = self.options.stiffness * (density / self.options.rest_density - 1.0)
        pressure = np.maximum(pressure, 0)
        new_state.sph_pressure = pressure
        
        # 4. Compute forces
        forces = np.zeros((n, 3), dtype=np.float32)
        
        # Pressure forces
        for i in range(n):
            for j in neighbors[i]:
                if i == j:
                    continue
                r = state.sph_pos[j] - state.sph_pos[i]
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-6:
                    continue
                
                # Pressure force: -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad W
                grad_w = self._grad_kernel(r, r_norm, h)
                p_term = (pressure[i] / (density[i]**2 + 1e-6) + 
                         pressure[j] / (density[j]**2 + 1e-6))
                forces[i] -= state.sph_mass[j] * p_term * grad_w
        
        # Viscosity forces
        for i in range(n):
            for j in neighbors[i]:
                if i == j:
                    continue
                r = state.sph_pos[j] - state.sph_pos[i]
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-6:
                    continue
                
                # Artificial viscosity
                v_ij = state.sph_vel[i] - state.sph_vel[j]
                r_hat = r / r_norm
                v_dot_r = np.dot(v_ij, r_hat)
                
                if v_dot_r < 0:
                    mu = h * v_dot_r / (r_norm**2 + 0.01 * h2)
                    pi_ij = (-self.options.viscosity * mu + 
                            0.1 * mu**2) / ((density[i] + density[j]) * 0.5)
                    grad_w = self._grad_kernel(r, r_norm, h)
                    forces[i] -= state.sph_mass[j] * pi_ij * grad_w
        
        # Surface tension (simplified)
        if self.options.surface_tension > 0:
            for i in range(n):
                for j in neighbors[i]:
                    if i == j:
                        continue
                    r = state.sph_pos[j] - state.sph_pos[i]
                    r_norm = np.linalg.norm(r)
                    if r_norm < 1e-6:
                        continue
                    grad_w = self._grad_kernel(r, r_norm, h)
                    forces[i] += self.options.surface_tension * state.sph_mass[j] * grad_w
        
        # 5. External forces (gravity)
        if self.scene:
            forces += state.sph_mass[:, None] * self.scene.gravity
        
        # 6. Coupling forces from rigid bodies
        if state.sph_rigid_contact_forces is not None:
            forces += state.sph_rigid_contact_forces
        
        # 7. Time integration (Symplectic Euler)
        accel = forces / state.sph_mass[:, None]
        new_state.sph_vel = state.sph_vel + accel * dt
        new_state.sph_pos = state.sph_pos + new_state.sph_vel * dt
        
        # 8. Boundary handling (keep particles in domain)
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
        if q2 < 0.5:
            return norm * (1.0 - 6.0 * q2 + 6.0 * q2 * np.sqrt(q2))
        else:
            return norm * 2.0 * (1.0 - np.sqrt(q2))**3
    
    def _grad_kernel(self, r: np.ndarray, r_norm: float, h: float) -> np.ndarray:
        """Gradient of cubic spline kernel."""
        h2 = h * h
        q = r_norm / h
        if q >= 1.0:
            return np.zeros(3, dtype=np.float32)
        
        norm = 48.0 / (np.pi * h**3)
        if q < 0.5:
            factor = norm * (6.0 * q - 6.0 * np.sqrt(q)) / h
        else:
            factor = norm * (-2.0 * (1.0 - np.sqrt(q)) / (np.sqrt(q) * h))
        
        return r * factor / r_norm if r_norm > 1e-6 else np.zeros(3, dtype=np.float32)
    
    def _handle_boundaries(self, state: State) -> None:
        """Simple boundary reflection."""
        # Floor at z=0
        if state.sph_pos is not None:
            mask = state.sph_pos[:, 2] < self.options.particle_radius
            state.sph_pos[mask, 2] = self.options.particle_radius
            state.sph_vel[mask, 2] = np.abs(state.sph_vel[mask, 2]) * 0.5  # Dampened bounce
    
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