"""
Base Solver Class — All physics solvers inherit from this.

Each solver owns disjoint DOFs in State. Solvers communicate ONLY via Coupler.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.state import State
    from ..core.entity import Entity, ComponentMask


@dataclass
class CouplingData:
    """Data exported by a solver for coupling with other solvers."""
    # Forces/velocities to apply to other solvers
    forces: dict[str, np.ndarray] = None  # target_solver -> (N, 3) forces
    velocities: dict[str, np.ndarray] = None  # target_solver -> (N, 3) velocities
    temperatures: dict[str, np.ndarray] = None  # target_solver -> (N,) temps
    concentrations: dict[str, np.ndarray] = None  # target_solver -> (N, num_species)
    # Contact information
    contacts: list = None  # List of Contact objects
    # Geometry for collision
    aabbs: np.ndarray | None = None  # (N, 6) min/max
    
    def __post_init__(self):
        if self.forces is None:
            self.forces = {}
        if self.velocities is None:
            self.velocities = {}
        if self.temperatures is None:
            self.temperatures = {}
        if self.concentrations is None:
            self.concentrations = {}
        if self.contacts is None:
            self.contacts = []
    
    def add_force(self, target_solver: str, force: np.ndarray) -> None:
        """Add force to apply to target solver's entities."""
        if target_solver in self.forces:
            self.forces[target_solver] += force
        else:
            self.forces[target_solver] = force.copy()
    
    def add_velocity(self, target_solver: str, velocity: np.ndarray) -> None:
        if target_solver in self.velocities:
            self.velocities[target_solver] = velocity.copy()
        else:
            self.velocities[target_solver] = velocity.copy()


class Solver(ABC):
    """
    Base class for all physics solvers.
    
    Each solver:
    - Owns disjoint DOFs in State (positions, velocities, etc.)
    - Steps its own physics given dt and contacts
    - Exports coupling data for other solvers
    - Applies coupling data from other solvers
    """
    
    name: str = "base"
    required_components: list[ComponentMask] = []
    
    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.scene: Scene | None = None
        self._initialized = False
    
    @abstractmethod
    def step(self, state: State, dt: float, contacts: list) -> State:
        """
        Advance solver by dt. Returns NEW State (immutable).
        
        Args:
            state: Current immutable state
            dt: Time step
            contacts: Contact list from CollisionSystem
            
        Returns:
            New State with updated DOFs for this solver
        """
        pass
    
    @abstractmethod
    def get_coupling_data(self, state: State) -> CouplingData:
        """
        Export data needed by other solvers.
        
        Returns:
            CouplingData with forces, velocities, temperatures, etc.
        """
        pass
    
    @abstractmethod
    def apply_coupling(self, state: State, coupling: CouplingData) -> State:
        """
        Apply forces/constraints from other solvers.
        
        Args:
            state: Current state
            coupling: CouplingData from other solvers (via Coupler)
            
        Returns:
            New State with coupling applied
        """
        pass
    
    def initialize(self, scene: Scene) -> None:
        """Called once after all solvers registered."""
        self.scene = scene
        self._initialized = True
    
    def reset(self) -> None:
        """Reset solver state (for checkpoint restore)."""
        pass
    
    def get_entities(self, state: State) -> list[Entity]:
        """Get entities this solver processes."""
        if self.scene is None:
            return []
        mask = ComponentMask.NONE
        for comp in self.required_components:
            mask |= comp
        return self.scene.query_entities(mask)
    
    def __repr__(self) -> str:
        return f"{self.name}Solver(initialized={self._initialized})"


# Import stub solvers
from .stubs import (
    FEMSolver, FEMOptions,
    MPMSolver, MPMOptions,
    PBDSolver, PBDOptions,
    ChemistrySolver, ChemistryOptions,
    ThermalSolver, ThermalOptions,
    GeologySolver, GeologyOptions,
)

# Import core solvers
from .rigid import RigidSolver, RigidOptions, RigidBodyBuilder
from .sph import SPHSolver, SPHOptions, SPHFluidBuilder

__all__ = [
    "Solver", "CouplingData",
    "RigidSolver", "RigidOptions", "RigidBodyBuilder",
    "SPHSolver", "SPHOptions", "SPHFluidBuilder",
    "FEMSolver", "FEMOptions",
    "MPMSolver", "MPMOptions", 
    "PBDSolver", "PBDOptions",
    "ChemistrySolver", "ChemistryOptions",
    "ThermalSolver", "ThermalOptions",
    "GeologySolver", "GeologyOptions",
]