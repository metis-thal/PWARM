"""
Explicit Coupler — Manages multi-physics interaction between solvers.

Genesis-style pairwise coupling with Gauss-Seidel iterations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.state import State
    from ..solvers.base import Solver, CouplingData


@dataclass
class CouplerOptions:
    """Which solver pairs are coupled."""
    rigid_sph: bool = True
    rigid_fem: bool = True
    rigid_mpm: bool = True
    rigid_pbd: bool = True
    sph_fem: bool = True
    sph_mpm: bool = True
    sph_pbd: bool = True
    fem_mpm: bool = True
    thermal_all: bool = True
    chemistry_sph: bool = True
    chemistry_fem: bool = True
    geology_thermal: bool = True
    geology_rigid: bool = True
    
    # Coupling parameters
    coupling_iterations: int = 3
    rigid_fluid_buoyancy: bool = True
    rigid_fluid_drag: bool = True
    thermal_contact_conductance: float = 1000.0  # W/(m²·K)
    
    def enabled_pairs(self) -> dict[tuple[str, str], bool]:
        return {
            ("rigid", "sph"): self.rigid_sph,
            ("rigid", "fem"): self.rigid_fem,
            ("rigid", "mpm"): self.rigid_mpm,
            ("rigid", "pbd"): self.rigid_pbd,
            ("sph", "fem"): self.sph_fem,
            ("sph", "mpm"): self.sph_mpm,
            ("sph", "pbd"): self.sph_pbd,
            ("fem", "mpm"): self.fem_mpm,
            ("thermal", "rigid"): self.thermal_all,
            ("thermal", "sph"): self.thermal_all,
            ("thermal", "fem"): self.thermal_all,
            ("thermal", "mpm"): self.thermal_all,
            ("thermal", "pbd"): self.thermal_all,
            ("thermal", "chemistry"): self.thermal_all,
            ("chemistry", "sph"): self.chemistry_sph,
            ("chemistry", "fem"): self.chemistry_fem,
            ("geology", "thermal"): self.geology_thermal,
            ("geology", "rigid"): self.geology_rigid,
        }


class Coupler:
    """
    Explicit multi-physics coupler.
    
    Algorithm (Gauss-Seidel):
    1. Each solver exports coupling data (forces, velocities, temps, etc.)
    2. For each enabled pair, compute interaction forces
    3. Each solver applies received coupling
    4. Repeat for N iterations
    """
    
    def __init__(self, scene: Scene, options: CouplerOptions | None = None):
        self.scene = scene
        self.options = options or CouplerOptions()
        self.enabled = self.options.enabled_pairs()
        self.iteration = 0
    
    def couple(self, state: State, dt: float) -> State:
        """Execute coupling iterations. Returns new State."""
        new_state = state
        
        for iteration in range(self.options.coupling_iterations):
            self.iteration = iteration
            
            # 1. Collect coupling data from all solvers
            coupling_data = {}
            for name, solver in self.scene.solvers.items():
                coupling_data[name] = solver.get_coupling_data(new_state)
            
            # 2. Compute pairwise interactions
            interaction_forces = self._compute_interactions(coupling_data, dt)
            
            # 3. Apply interactions to coupling data
            for solver_name, forces in interaction_forces.items():
                if solver_name in coupling_data:
                    for target_name, force in forces.items():
                        coupling_data[solver_name].add_force(target_name, force)
            
            # 4. Each solver applies coupling
            for name, solver in self.scene.solvers.items():
                if name in coupling_data:
                    new_state = solver.apply_coupling(new_state, coupling_data[name])
        
        return new_state
    
    def _compute_interactions(self, coupling_data: dict[str, CouplingData], dt: float) -> dict[str, dict[str, np.ndarray]]:
        """Compute pairwise interaction forces. Returns {solver_name: {target_name: force_array}}."""
        interactions = {name: {} for name in coupling_data.keys()}
        
        # Rigid ↔ SPH (fluid-structure interaction)
        if self.enabled.get(("rigid", "sph"), False):
            self._rigid_sph_interaction(coupling_data, interactions, dt)
        
        # Rigid ↔ FEM (contact)
        if self.enabled.get(("rigid", "fem"), False):
            self._rigid_fem_interaction(coupling_data, interactions, dt)
        
        # Rigid ↔ MPM (contact)
        if self.enabled.get(("rigid", "mpm"), False):
            self._rigid_mpm_interaction(coupling_data, interactions, dt)
        
        # Rigid ↔ PBD (contact)
        if self.enabled.get(("rigid", "pbd"), False):
            self._rigid_pbd_interaction(coupling_data, interactions, dt)
        
        # SPH ↔ FEM (fluid-pressure on deformable)
        if self.enabled.get(("sph", "fem"), False):
            self._sph_fem_interaction(coupling_data, interactions, dt)
        
        # Thermal ↔ All (heat transfer)
        if self.enabled.get(("thermal", "rigid"), False):
            self._thermal_interactions(coupling_data, interactions, dt)
        
        # Chemistry ↔ SPH (species transport)
        if self.enabled.get(("chemistry", "sph"), False):
            self._chemistry_sph_interaction(coupling_data, interactions, dt)
        
        # Geology ↔ Thermal (crustal heat flow)
        if self.enabled.get(("geology", "thermal"), False):
            self._geology_thermal_interaction(coupling_data, interactions, dt)
        
        return interactions
    
    def _rigid_sph_interaction(self, coupling_data, interactions, dt):
        """Fluid-structure: rigid bodies displace fluid, buoyancy, drag."""
        rigid_data = coupling_data.get("rigid")
        sph_data = coupling_data.get("sph")
        
        if rigid_data is None or sph_data is None:
            return
        
        # Rigid body boundary particles exert forces on SPH particles
        # Buoyancy: Archimedes principle
        # Drag: Stokes or quadratic drag
        pass
    
    def _rigid_fem_interaction(self, coupling_data, interactions, dt):
        """Rigid-deformable contact via penalty or Lagrange multipliers."""
        pass
    
    def _rigid_mpm_interaction(self, coupling_data, interactions, dt):
        """Rigid-MPM contact. MPM particles collide with rigid boundary."""
        pass
    
    def _rigid_pbd_interaction(self, coupling_data, interactions, dt):
        """Rigid-PBD contact. PBD particles constrained by rigid surfaces."""
        pass
    
    def _sph_fem_interaction(self, coupling_data, interactions, dt):
        """SPH fluid pressure loads on FEM surface."""
        pass
    
    def _thermal_interactions(self, coupling_data, interactions, dt):
        """Heat conduction across material interfaces."""
        thermal_data = coupling_data.get("thermal")
        if thermal_data is None:
            return
        
        # Thermal contact conductance between all solid solvers
        for solver_name in ["rigid", "fem", "mpm", "pbd"]:
            if solver_name in coupling_data:
                # Heat flux = h * (T_solid - T_thermal) at interface
                pass
    
    def _chemistry_sph_interaction(self, coupling_data, interactions, dt):
        """Species advection-diffusion in SPH flow."""
        pass
    
    def _geology_thermal_interaction(self, coupling_data, interactions, dt):
        """Crustal heat flow, radiogenic heating, surface cooling."""
        pass


def create_coupler(scene: Scene, **kwargs) -> Coupler:
    """Factory for creating coupler with options."""
    options = CouplerOptions(**kwargs)
    return Coupler(scene, options)