"""
Time Stepper — Unified time integration with sub-steps and coupling iterations.

Velocity-Verlet for non-stiff, implicit for stiff (thermal, FEM).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.state import State
    from ..collision.system import ContactList, CollisionSystem
    from ..coupling.coupler import Coupler
    from ..solvers.base import Solver


@dataclass
class TimeStepperOptions:
    dt: float = 1.0 / 60.0
    substeps: int = 1
    coupling_iterations: int = 3
    # Solver-specific substeps
    rigid_substeps: int = 1
    sph_substeps: int = 1
    fem_substeps: int = 1
    thermal_substeps: int = 1
    # Stability
    max_velocity: float = 100.0
    cfl_condition: bool = True  # Auto-adjust dt based on CFL


class TimeStepper:
    """
    Unified time stepping for multi-physics simulation.
    
    Algorithm per frame:
    for sub_step in range(substeps):
        dt_sub = dt / substeps
        1. Collision detection (once per sub-step)
        2. Each solver steps (can be parallelized)
        3. Merge partial states
        4. Coupling iterations
        5. Conservation check
    """
    
    def __init__(self, scene: Scene, options: TimeStepperOptions | None = None):
        self.scene = scene
        self.options = options or TimeStepperOptions()
        self.current_substep = 0
        self._conservation_history: list[dict] = []
    
    def step(self, state: State) -> State:
        """Advance simulation by one frame (dt). Returns new State."""
        dt = self.options.dt
        substeps = self.options.substeps
        dt_sub = dt / substeps
        
        new_state = state
        
        for sub in range(substeps):
            self.current_substep = sub
            
            # 1. Collision detection (once per sub-step)
            contacts = self.scene.collision_system.detect(new_state)
            
            # 2. Solver steps
            solver_states = {}
            for name, solver in self.scene.solvers.items():
                # Determine sub-steps for this solver
                solver_substeps = getattr(self.options, f"{name}_substeps", 1)
                solver_dt = dt_sub / solver_substeps
                
                solver_state = new_state
                for _ in range(solver_substeps):
                    solver_state = solver.step(solver_state, solver_dt, contacts)
                
                solver_states[name] = solver_state
            
            # 3. Merge solver states (additive for disjoint DOFs)
            new_state = self._merge_states(new_state, solver_states)
            
            # 4. Coupling iterations (Gauss-Seidel)
            for _ in range(self.options.coupling_iterations):
                new_state = self.scene.coupler.couple(new_state, dt_sub)
            
            # 5. Conservation check
            self._check_conservation(new_state, state)
        
        return new_state
    
    def _merge_states(self, base: State, solver_states: dict[str, State]) -> State:
        """
        Merge per-solver state updates.
        
        Each solver owns disjoint DOFs. Merge by copying arrays.
        """
        # Start with base
        merged = base.copy()
        
        # Copy updated arrays from each solver
        for name, s_state in solver_states.items():
            if name == "rigid":
                merged.rigid_pos = s_state.rigid_pos
                merged.rigid_quat = s_state.rigid_quat
                merged.rigid_linvel = s_state.rigid_linvel
                merged.rigid_angvel = s_state.rigid_angvel
                merged.rigid_force_accum = s_state.rigid_force_accum
                merged.rigid_torque_accum = s_state.rigid_torque_accum
            elif name == "sph":
                merged.sph_pos = s_state.sph_pos
                merged.sph_vel = s_state.sph_vel
                merged.sph_density = s_state.sph_density
                merged.sph_pressure = s_state.sph_pressure
                merged.sph_neighbor_indices = s_state.sph_neighbor_indices
                merged.sph_rigid_contact_forces = s_state.sph_rigid_contact_forces
            elif name == "fem":
                merged.fem_pos = s_state.fem_pos
                merged.fem_vel = s_state.fem_vel
                merged.fem_F = s_state.fem_F
                merged.fem_contact_forces = s_state.fem_contact_forces
            elif name == "mpm":
                merged.mpm_pos = s_state.mpm_pos
                merged.mpm_vel = s_state.mpm_vel
                merged.mpm_F = s_state.mpm_F
            elif name == "pbd":
                merged.pbd_pos = s_state.pbd_pos
                merged.pbd_pred_pos = s_state.pbd_pred_pos
                merged.pbd_vel = s_state.pbd_vel
                merged.pbd_external_forces = s_state.pbd_external_forces
            elif name == "thermal":
                merged.thermal_temp = s_state.thermal_temp
                merged.thermal_flux_from_chemistry = s_state.thermal_flux_from_chemistry
                merged.thermal_flux_from_rigid = s_state.thermal_flux_from_rigid
            elif name == "chemistry":
                merged.chem_conc = s_state.chem_conc
                merged.chem_advection_vel = s_state.chem_advection_vel
            elif name == "geology":
                merged.geo_rock_type = s_state.geo_rock_type
                merged.geo_porosity = s_state.geo_porosity
                merged.geo_elevation = s_state.geo_elevation
                merged.geo_sediment_flux = s_state.geo_sediment_flux
                merged.geo_uplift_rate = s_state.geo_uplift_rate
        
        # Update time
        merged.t = base.t + self.options.dt
        
        return merged
    
    def _check_conservation(self, new_state: State, old_state: State) -> None:
        """Check mass, energy, momentum conservation."""
        new_state.global_quantities.compute_from_state(new_state)
        old_state.global_quantities.compute_from_state(old_state)
        
        changes = new_state.global_quantities.relative_change(old_state.global_quantities)
        
        # Log significant changes
        for key, change in changes.items():
            if change > 1e-3:  # 0.1% threshold
                self._conservation_history.append({
                    "frame": self.scene.frame if self.scene else 0,
                    "quantity": key,
                    "relative_change": change,
                    "old": getattr(old_state.global_quantities, key),
                    "new": getattr(new_state.global_quantities, key),
                })
                print(f"[CONSERVATION WARNING] {key}: {change*100:.3f}% change")
        
        # Keep last 1000 entries
        if len(self._conservation_history) > 1000:
            self._conservation_history = self._conservation_history[-1000:]
    
    def get_conservation_report(self) -> dict:
        """Get conservation statistics."""
        if not self._conservation_history:
            return {"status": "no_data"}
        
        by_quantity = {}
        for entry in self._conservation_history:
            q = entry["quantity"]
            if q not in by_quantity:
                by_quantity[q] = []
            by_quantity[q].append(entry["relative_change"])
        
        report = {}
        for q, changes in by_quantity.items():
            report[q] = {
                "max": max(changes),
                "mean": np.mean(changes),
                "count": len(changes),
            }
        
        return {
            "total_violations": len(self._conservation_history),
            "by_quantity": report,
        }


class VelocityVerletIntegrator:
    """Velocity-Verlet for non-stiff systems (rigid bodies, SPH)."""
    
    @staticmethod
    def step_position(pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        return pos + vel * dt
    
    @staticmethod
    def step_velocity(vel: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        return vel + accel * dt
    
    @staticmethod
    def step(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, dt: float) -> tuple:
        """Full Velocity-Verlet step."""
        # v(t + dt/2) = v(t) + a(t) * dt/2
        vel_half = vel + accel * (dt * 0.5)
        # x(t + dt) = x(t) + v(t + dt/2) * dt
        pos_new = pos + vel_half * dt
        # a(t + dt) = f(x(t+dt)) / m
        # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        # (accel_new computed by solver after position update)
        return pos_new, vel_half


class SymplecticEulerIntegrator:
    """Symplectic Euler (semi-implicit) for better energy conservation."""
    
    @staticmethod
    def step(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, dt: float) -> tuple:
        vel_new = vel + accel * dt
        pos_new = pos + vel_new * dt
        return pos_new, vel_new


class ImplicitEulerIntegrator:
    """Implicit Euler for stiff systems (thermal, FEM implicit)."""
    
    @staticmethod
    def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve A x = b. Use scipy.sparse.linalg.spsolve for sparse."""
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix
        if hasattr(A, 'tocsc'):
            return spsolve(A.tocsc(), b)
        return np.linalg.solve(A, b)


def create_time_stepper(scene: Scene, **kwargs) -> TimeStepper:
    """Factory for creating time stepper."""
    options = TimeStepperOptions(**kwargs)
    return TimeStepper(scene, options)