"""
Component Definitions — Data structures for each physics domain.

Components are pure data (no behavior). Solvers read/write component arrays in State.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .entity import EntityID


class Component:
    """Base class for all components. Marker interface."""
    pass


@dataclass(slots=True)
class TransformComponent(Component):
    """World transform: position + rotation. All entities have this."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # quaternion w,x,y,z
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))


@dataclass(slots=True)
class RigidBodyComponent(Component):
    """Rigid body dynamics state. One per dynamic rigid entity."""
    mass: float = 1.0
    inv_mass: float = 1.0
    inertia_local: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))  # Local inertia tensor
    inv_inertia_local: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    gravity_scale: float = 1.0
    is_static: bool = False
    is_kinematic: bool = False
    # Forces/torques accumulated this frame
    force_accum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    torque_accum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Contact cache
    contact_normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    contact_depth: float = 0.0


@dataclass(slots=True)
class SPHParticleComponent(Component):
    """SPH fluid particle arrays. One component per fluid entity (holds N particles)."""
    # Particle state (N, 3)
    positions: np.ndarray | None = None   # (N, 3)
    velocities: np.ndarray | None = None  # (N, 3)
    densities: np.ndarray | None = None   # (N,)
    pressures: np.ndarray | None = None   # (N,)
    masses: np.ndarray | None = None      # (N,)
    smoothing_h: float = 0.025
    rest_density: float = 1000.0
    viscosity: float = 0.1
    surface_tension: float = 0.072
    # Boundary handling
    boundary_particles: np.ndarray | None = None  # (M, 3) static boundary particles
    boundary_normals: np.ndarray | None = None    # (M, 3)
    # Neighborhood (computed each step)
    neighbor_indices: list[list[int]] | None = None
    neighbor_distances: list[list[float]] | None = None
    # Coupling
    rigid_contact_forces: np.ndarray | None = None  # (N, 3) forces from rigid bodies


@dataclass(slots=True)
class FEMNodeComponent(Component):
    """FEM deformable body nodes. One component per deformable entity."""
    # Reference configuration
    rest_positions: np.ndarray | None = None   # (N, 3)
    # Current state
    positions: np.ndarray | None = None        # (N, 3)
    velocities: np.ndarray | None = None       # (N, 3)
    # Material properties (per element or per node)
    youngs_modulus: float = 1e5
    poissons_ratio: float = 0.3
    density: float = 1000.0
    # Deformation gradient (per element)
    deformation_grad: np.ndarray | None = None  # (E, 3, 3)
    # Tetrahedral mesh connectivity
    elements: np.ndarray | None = None          # (E, 4) node indices
    # Boundary conditions
    fixed_nodes: np.ndarray | None = None       # (K,) fixed node indices
    # Coupling
    contact_forces: np.ndarray | None = None    # (N, 3) external forces


@dataclass(slots=True)
class MPMParticleComponent(Component):
    """Material Point Method particles. Sand, snow, clay, etc."""
    positions: np.ndarray | None = None       # (N, 3)
    velocities: np.ndarray | None = None      # (N, 3)
    masses: np.ndarray | None = None          # (N,)
    volumes: np.ndarray | None = None         # (N,)
    # Deformation gradient
    F: np.ndarray | None = None               # (N, 3, 3)
    # Material
    material_type: np.ndarray | None = None   # (N,) int: 0=sand, 1=clay, 2=snow, 3=water
    youngs_modulus: float = 1e5
    poissons_ratio: float = 0.3
    # Plasticity (for sand/clay)
    hardening: float = 10.0
    # Grid (for MPM)
    grid_size: tuple[int, int, int] = (64, 64, 64)
    grid_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    cell_size: float = 0.05


@dataclass(slots=True)
class PBDParticleComponent(Component):
    """Position-Based Dynamics particles. Cloth, hair, ropes."""
    positions: np.ndarray | None = None       # (N, 3)
    predicted_positions: np.ndarray | None = None  # (N, 3)
    velocities: np.ndarray | None = None      # (N, 3)
    inv_masses: np.ndarray | None = None      # (N,) 0 for fixed
    # Constraints
    distance_constraints: np.ndarray | None = None    # (C, 2) particle indices
    distance_rest_lengths: np.ndarray | None = None   # (C,)
    bending_constraints: np.ndarray | None = None     # (C, 3) or (C, 4)
    bending_rest_angles: np.ndarray | None = None     # (C,)
    # Collision
    collision_radius: float = 0.01
    # Coupling
    external_forces: np.ndarray | None = None         # (N, 3)


@dataclass(slots=True)
class ThermalComponent(Component):
    """Temperature field. Can be per-node (FEM) or per-particle (SPH/MPM) or per-voxel (Geology)."""
    temperatures: np.ndarray | None = None      # (N,) or (Nx, Ny, Nz)
    # Material thermal properties
    conductivity: float = 1.0      # W/(m·K)
    specific_heat: float = 1000.0  # J/(kg·K)
    density: float = 1000.0        # kg/m³
    # Phase change
    melting_temp: float = 273.15
    latent_heat: float = 3.34e5
    # Boundary conditions
    fixed_temp_nodes: np.ndarray | None = None
    fixed_temperatures: np.ndarray | None = None
    # Heat sources
    heat_source: float = 0.0       # W/m³
    # Coupling
    heat_flux_from_chemistry: float = 0.0
    heat_flux_from_rigid: float = 0.0


class ChemicalSpecies(IntEnum):
    """Predefined chemical species for common simulations."""
    WATER = 0
    OXYGEN = 1
    CARBON_DIOXIDE = 2
    NUTRIENT = 3
    BIOMASS = 4
    POLLUTANT = 5
    CUSTOM_START = 100


@dataclass(slots=True)
class ChemistryComponent(Component):
    """Species concentration fields. Reaction-diffusion-advection."""
    # Concentrations: (N, num_species) or (Nx, Ny, Nz, num_species)
    concentrations: np.ndarray | None = None
    num_species: int = 4
    # Diffusion coefficients (per species)
    diffusion_coeffs: np.ndarray | None = None  # (num_species,)
    # Reaction rates (user-defined)
    reaction_rates: dict[str, float] = field(default_factory=dict)
    # Source/sink terms
    sources: np.ndarray | None = None  # (N, num_species) or (grid, num_species)
    # Coupling
    advection_velocity: np.ndarray | None = None  # (N, 3) from SPH/fluid velocity


class RockType(IntEnum):
    """Geological rock types."""
    AIR = 0
    SEDIMENT = 1
    SANDSTONE = 2
    SHALE = 3
    LIMESTONE = 4
    GRANITE = 5
    BASALT = 6
    METAMORPHIC = 7
    MAGMA = 8


@dataclass(slots=True)
class GeologyComponent(Component):
    """Geological state: rock type, porosity, stratigraphy, erosion."""
    # Voxel grid (Nx, Ny, Nz)
    rock_types: np.ndarray | None = None        # (Nx, Ny, Nz) int
    porosity: np.ndarray | None = None          # (Nx, Ny, Nz) [0,1]
    permeability: np.ndarray | None = None      # (Nx, Ny, Nz) m²
    # Stratigraphy
    layer_ids: np.ndarray | None = None         # (Nx, Ny, Nz) int
    layer_ages: np.ndarray | None = None        # (num_layers,) years
    # Surface processes
    elevation: np.ndarray | None = None         # (Nx, Ny) surface height
    sediment_flux: np.ndarray | None = None     # (Nx, Ny) m³/yr
    # Tectonics
    uplift_rate: np.ndarray | None = None       # (Nx, Ny) m/yr
    stress_tensor: np.ndarray | None = None     # (Nx, Ny, Nz, 3, 3)
    # Coupling
    erosion_rate: float = 0.0
    thermal_conductivity_map: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class CollisionShapeComponent(Component):
    """Collision geometry for broad/narrow phase."""
    class ShapeType(IntEnum):
        SPHERE = 0
        BOX = 1
        CAPSULE = 2
        CYLINDER = 3
        CONVEX_HULL = 4
        TRIANGLE_MESH = 5
        SDF = 6  # Signed Distance Field
    
    shape_type: ShapeType = ShapeType.SPHERE
    # Owning entity (set at creation; used by collision system for Contact/AABB identity)
    entity_id: EntityID | None = None
    # Sphere
    radius: float = 0.5
    # Box
    half_extents: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    # Capsule/Cylinder
    half_height: float = 1.0
    # Convex hull / mesh
    vertices: np.ndarray | None = None      # (V, 3)
    indices: np.ndarray | None = None       # (I, 3) triangle indices
    # SDF
    sdf_grid: np.ndarray | None = None      # (Nx, Ny, Nz)
    sdf_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    sdf_cell_size: float = 0.1
    # Collision filtering
    collision_group: int = 1
    collision_mask: int = 0xFFFFFFFF
    # Material
    friction: float = 0.5
    restitution: float = 0.0
    # CCD
    use_ccd: bool = False