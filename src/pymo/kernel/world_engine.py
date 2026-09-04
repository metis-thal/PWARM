"""WorldEngine (Legacy) — Deprecated.

This is the OLD WorldEngine implementation. 
The NEW unified physics world engine is at: pymo.physics.WorldEngine

Migration guide:
- Old: from pymo.kernel.world_engine import WorldEngine
- New: from pymo.physics import WorldEngine, WorldEngineConfig

The new engine uses:
- Single Scene + single immutable State (double-buffered)
- Explicit multi-physics Coupler (rigid↔SPH, rigid↔FEM, thermal↔all, etc.)
- Unified CollisionSystem (SAP + GJK/EPA + CCD)
- Velocity-Verlet + implicit integrators with sub-steps
- AI layer: Observer → LawDiscovery → ClosedLoopAI
- Asset parsers: URDF, MJCF, GLTF
- Parallel environments: Ray / multiprocessing / vectorized
"""

import warnings

warnings.warn(
    "pymo.kernel.world_engine.WorldEngine is deprecated. "
    "Use pymo.physics.WorldEngine instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy classes for existing code
from .bodies3d import Body, Material, sphere_body, box_body
from .world3d import World3D
from pymo.rules.chemistry import ChemicalSystem, step_chemistry
from pymo.rules.fluid import SPHSystem, SPHParams
from pymo.rules.thermal import BodyThermalSystem
from pymo.geology import GeologySolver, GeologySolverConfig

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Legacy dataclasses for backwards compatibility
@dataclass
class Observation:
    name: str
    t: np.ndarray
    values: np.ndarray


@dataclass
class TimeSeriesDataset:
    observations: list[Observation] = field(default_factory=list)

    def get(self, name: str) -> np.ndarray | None:
        for obs in self.observations:
            if obs.name == name:
                return obs.values
        return None

    def time(self) -> np.ndarray:
        if not self.observations:
            return np.array([])
        return self.observations[0].t

    def names(self) -> list[str]:
        return [o.name for o in self.observations]


# Legacy WorldEngine class (deprecated, kept for API compatibility)
@dataclass
class LegacyWorldEngine:
    """Deprecated: Use pymo.physics.WorldEngine instead."""
    
    world: World3D = field(default_factory=World3D)
    thermal: BodyThermalSystem = field(default_factory=BodyThermalSystem)
    sph: SPHSystem | None = None
    chemistry: ChemicalSystem | None = None
    solar_flux: float = 0.0
    environment_temp: float = 293.15
    body_albedos: dict[int, float] = field(default_factory=dict)
    enable_geology: bool = False
    geology_solver: GeologySolver | None = None
    geology_config: GeologySolverConfig | None = None
    _initial_total_mass: float = 0.0
    _initial_total_energy: float = 0.0
    t: float = 0.0
    step_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    observe: bool = False
    sample_every: int = 1
    _obs_times: list[float] = field(default_factory=list)
    _obs_series: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enable_geology and self.geology_solver is None:
            self.geology_config = self.geology_config or GeologySolverConfig()
            self.geology_solver = GeologySolver(self.geology_config)
            self.geology_solver.initialize()

    def add_body(self, body: Body, albedo: float = 0.3) -> int:
        """Add a 3D rigid body. Returns its index."""
        self.world.add(body)
        idx = len(self.world.bodies) - 1
        self.body_albedos[idx] = albedo
        return idx

    def tick(self, dt: float | None = None) -> dict[str, Any]:
        # Legacy implementation - see git history for full code
        raise NotImplementedError("Use pymo.physics.WorldEngine instead")


# Alias for backwards compatibility
WorldEngine = LegacyWorldEngine

__all__ = [
    "WorldEngine",
    "WorldEngineConfig", 
    "create_world_engine",
    "LegacyWorldEngine",
    "Observation",
    "TimeSeriesDataset",
]