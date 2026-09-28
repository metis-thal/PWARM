"""
PWARM Physics - Unified Multi-Physics World Engine.

Architecture (Genesis-inspired):
- Single Scene, single immutable State (double-buffered)
- Multiple Solvers: Rigid, SPH, FEM, MPM, PBD, Thermal, Chemistry, Geology
- Explicit Coupler: pairwise interactions (rigid<->SPH, rigid<->FEM, thermal<->all, etc.)
- Unified CollisionSystem: SAP broad phase + GJK/EPA narrow phase + CCD
- TimeStepper: sub-steps + Gauss-Seidel coupling iterations
- AI Layer: Observer -> LawDiscovery -> ClosedLoopAI
- Interface: URDF/MJCF/GLTF parsers, GUI, Sensors, Parallel environments
"""

# WorldEngineConfig defined locally
from dataclasses import dataclass, field

import numpy as np

from ..interface import (
    GUI,
    CameraConfig,
    CameraSensor,
    EnvConfig,
    ForceSensor,
    ParallelEnv,
    SensorData,
    SensorType,
    VectorizedEnv,
    load_gltf,
    load_mjcf,
    load_urdf,
    parse_gltf,
    parse_mjcf,
    parse_urdf,
)

# AI Layer imports
from .ai import (
    Observation,
    TimeSeriesDataset,
    WorldObserver,
    create_collision_experiment,
    create_free_fall_experiment,
)
from .collision import (
    AABB,
    CollisionSystem,
    ConservativeCCD,
    Contact,
    ContactList,
    GJKNarrowPhase,
    SAPBroadPhase,
)
from .core import (
    ChemistryComponent,
    CollisionShapeComponent,
    ComponentMask,
    Entity,
    EntityID,
    EntityManager,
    FEMNodeComponent,
    GeologyComponent,
    GlobalQuantities,
    MPMParticleComponent,
    PBDParticleComponent,
    RigidBodyComponent,
    Scene,
    SPHParticleComponent,
    State,
    ThermalComponent,
    TransformComponent,
)
from .coupling import Coupler, CouplerOptions, create_coupler
from .integrator import (
    ImplicitEulerIntegrator,
    SymplecticEulerIntegrator,
    TimeStepper,
    TimeStepperOptions,
    VelocityVerletIntegrator,
    create_time_stepper,
)
from .solvers import CouplingData, Solver


@dataclass
class WorldEngineConfig:
    """Configuration for WorldEngine."""
    # Simulation
    dt: float = 1.0 / 60.0
    substeps: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    coupling_iterations: int = 3
    
    # Solvers
    rigid: dict = field(default_factory=dict)
    sph: dict = field(default_factory=dict)
    fem: dict = field(default_factory=dict)
    mpm: dict = field(default_factory=dict)
    pbd: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)
    chemistry: dict = field(default_factory=dict)
    geology: dict = field(default_factory=dict)
    
    # Features
    enable_ai: bool = False
    checkpoint_interval: int = 0
    checkpoint_path: str = "checkpoints/"

def create_world_engine(config: WorldEngineConfig | None = None) -> "WorldEngine":
    """Factory function for creating WorldEngine."""
    return WorldEngine(config or WorldEngineConfig())


# New WorldEngine implementation using the physics architecture
class WorldEngine:
    """
    Unified physics world engine (NEW architecture).
    
    Uses:
    - Single Scene + single immutable State (double-buffered)
    - Multiple Solvers: Rigid, SPH, FEM, MPM, PBD, Thermal, Chemistry, Geology
    - Explicit Coupler for multi-physics interactions
    - Unified CollisionSystem (SAP + GJK/EPA + CCD)
    - TimeStepper with sub-steps and coupling iterations
    - Double buffer for sim/render separation
    """
    
    def __init__(self, config: WorldEngineConfig):
        self.config = config
        self.frame = 0
        self.total_time = 0.0
        self._needs_state_rebuild = False
        
        # Import here to avoid circular imports
        from .collision import CollisionSystem
        from .core.scene import Scene
        from .coupling import Coupler, CouplerOptions
        from .integrator import TimeStepper, TimeStepperOptions
        
        # Build scene
        self.scene = Scene()
        self.scene.gravity = np.array(config.gravity, dtype=np.float32)
        self.scene.dt = config.dt
        self.scene.substeps = config.substeps
        self.scene.enable_ai = config.enable_ai
        self.scene.checkpoint_interval = config.checkpoint_interval
        self.scene.checkpoint_path = config.checkpoint_path
        
        # Register enabled solvers
        from .solvers import (
            ChemistrySolver,
            FEMSolver,
            GeologySolver,
            MPMSolver,
            PBDSolver,
            RigidSolver,
            SPHSolver,
            ThermalSolver,
        )
        
        solver_map = {
            "rigid": (config.rigid.get("enabled", True), RigidSolver),
            "sph": (config.sph.get("enabled", False), SPHSolver),
            "fem": (config.fem.get("enabled", False), FEMSolver),
            "mpm": (config.mpm.get("enabled", False), MPMSolver),
            "pbd": (config.pbd.get("enabled", False), PBDSolver),
            "thermal": (config.thermal.get("enabled", False), ThermalSolver),
            "chemistry": (config.chemistry.get("enabled", False), ChemistrySolver),
            "geology": (config.geology.get("enabled", False), GeologySolver),
        }
        
        for name, (enabled, solver_class) in solver_map.items():
            if enabled:
                options = {k: v for k, v in config.__dict__.get(name, {}).items() if k != "enabled"}
                solver = solver_class(options)
                self.scene.register_solver(name, solver)
        
        # Shared systems
        self.scene.coupler = Coupler(self.scene, CouplerOptions(**{
            k: v for k, v in config.__dict__.items() 
            if k in ['rigid_sph', 'rigid_fem', 'rigid_mpm', 'rigid_pbd', 
                     'sph_fem', 'sph_mpm', 'sph_pbd', 'fem_mpm',
                     'thermal_all', 'chemistry_sph', 'chemistry_fem',
                     'geology_thermal', 'geology_rigid']
        }))
        self.scene.collision_system = CollisionSystem()
        self.scene.collision_system.scene = self.scene
        self.scene.time_stepper = TimeStepper(self.scene, TimeStepperOptions(
            dt=config.dt,
            substeps=config.substeps,
            coupling_iterations=config.coupling_iterations,
        ))
        
        # Initialize solvers
        for solver in self.scene.solvers.values():
            solver.initialize(self.scene)
        
        # Create initial double buffer state
        self.scene.double_buffer_write = self.scene._create_initial_state()
        self.scene.double_buffer_read = self.scene.double_buffer_write.copy()
        
        # AI layer
        if config.enable_ai:
            from .ai import ClosedLoopAI, LawDiscovery, WorldObserver
            self.observer = WorldObserver(self.scene)
            self.law_discovery = LawDiscovery()
            self.closed_loop_ai = ClosedLoopAI()
        else:
            self.observer = None
            self.law_discovery = None
            self.closed_loop_ai = None
    
    def create_rigid_body(self, 
                          position: tuple[float, float, float],
                          mass: float = 1.0,
                          shape: str = "sphere",
                          shape_params: dict | None = None):
        """Convenience: create rigid body entity."""
        from .core.component import CollisionShapeComponent, RigidBodyComponent, TransformComponent
        from .core.entity import ComponentMask, Entity, EntityID
        
        entity = Entity(id=EntityID(), name=f"rigid_{len(self.scene.entities)}")
        entity.mask = ComponentMask.RIGID_DYNAMIC if mass > 0 else ComponentMask.RIGID_STATIC
        
        # Transform
        transform = TransformComponent()
        transform.position = np.array(position, dtype=np.float32)
        entity.add(ComponentMask.TRANSFORM)
        entity.user_data['transform'] = transform
        
        # Rigid body
        rb = RigidBodyComponent()
        rb.mass = mass
        rb.inv_mass = 1.0 / mass if mass > 0 else 0.0
        rb.is_static = mass <= 0
        entity.add(ComponentMask.RIGID_BODY)
        entity.user_data['rigid_body'] = rb
        
        # Collision shape
        shape_comp = CollisionShapeComponent()
        shape_comp.friction = shape_params.get("friction", 0.5) if shape_params else 0.5
        shape_comp.restitution = shape_params.get("restitution", 0.0) if shape_params else 0.0
        shape_comp.use_ccd = bool(shape_params.get("use_ccd", False)) if shape_params else False
        if shape == "sphere":
            shape_comp.shape_type = CollisionShapeComponent.ShapeType.SPHERE
            shape_comp.radius = shape_params.get("radius", 0.5) if shape_params else 0.5
        elif shape == "box":
            shape_comp.shape_type = CollisionShapeComponent.ShapeType.BOX
            shape_comp.half_extents = np.array(shape_params.get("half_extents", [1,1,1]), dtype=np.float32)
        elif shape == "capsule":
            shape_comp.shape_type = CollisionShapeComponent.ShapeType.CAPSULE
            shape_comp.radius = shape_params.get("radius", 0.5)
            shape_comp.half_height = shape_params.get("half_height", 1.0)
        entity.add(ComponentMask.COLLISION_SHAPE)
        entity.user_data['collision_shape'] = shape_comp
        shape_comp.entity_id = entity.id
        
        added_entity = self.scene.add_entity(entity)
        self._needs_state_rebuild = True
        
        # Store shape info for rendering
        if not hasattr(self, '_shape_info'):
            self._shape_info = {}
        self._shape_info[added_entity.rigid_index] = {
            'shape': shape,
            'params': shape_params or {},
        }
        
        return added_entity
    
    def finalize_setup(self) -> None:
        """Call after adding all initial entities to rebuild state arrays."""
        if getattr(self, '_needs_state_rebuild', False):
            self.scene.double_buffer_write = self.scene._create_initial_state()
            self.scene.double_buffer_read = self.scene.double_buffer_write.copy()
            self._needs_state_rebuild = False

    def create_sph_fluid(self,
                         positions,
                         velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                         mass: float | None = None):
        """Create an SPH fluid entity from an (N, 3) array of particle positions.

        Default mass assumes the standard WCSPH lattice spacing
        d = particle_radius = h/2 (0.025 with defaults), where the
        cubic-spline per-particle kernel sum ~= 1/d^3, so
        mass = rest_density * d^3 starts the lattice at equilibrium
        (rest density, zero pressure). For a different lattice spacing,
        pass mass = rest_density / (per-particle kernel sum) explicitly.
        """
        from .core.component import SPHParticleComponent, TransformComponent
        from .core.entity import ComponentMask, Entity, EntityID

        positions = np.asarray(positions, dtype=np.float32)
        n = len(positions)

        entity = Entity(id=EntityID(), name=f"sph_{len(self.scene.entities)}")
        entity.mask = ComponentMask.SPH_PARTICLE

        transform = TransformComponent()
        transform.position = positions.mean(axis=0)
        entity.add(ComponentMask.TRANSFORM)
        entity.user_data['transform'] = transform

        sph = SPHParticleComponent()
        sph.positions = positions
        sph.velocities = np.tile(np.array(velocity, dtype=np.float32), (n, 1))
        if mass is None:
            # Standard WCSPH lattice: spacing d = particle_radius = h/2, so
            # ~26 neighbors sit inside the kernel support. For the cubic
            # spline at h = 2d the per-particle kernel sum ~= 1/d^3, hence
            # mass = rest_density * d^3 starts the lattice at equilibrium.
            mass = 1000.0 * 0.025 ** 3  # 0.015625 kg
        sph.masses = np.full(n, mass, dtype=np.float32)
        entity.add(ComponentMask.SPH_PARTICLE)
        entity.user_data['sph'] = sph

        added_entity = self.scene.add_entity(entity)
        self.scene.entities.set_particle_counts(added_entity, sph=n)
        self._needs_state_rebuild = True
        return added_entity
    
    def tick(self) -> State:
        """Single simulation step."""
        new_state = self.scene.step()
        self.frame += 1
        self.total_time = new_state.t
        return new_state
    
    def run(self, steps: int) -> list[State]:
        """Run multiple steps."""
        return [self.tick() for _ in range(steps)]
    
    def get_render_snapshot(self) -> State | None:
        return self.scene.get_render_snapshot()
    
    def checkpoint(self, path: str | None = None) -> str:
        return self.scene.checkpoint(path)
    
    def restore(self, path: str) -> None:
        self.scene.restore_checkpoint(path)
    
    def __repr__(self) -> str:
        return (f"WorldEngine(frame={self.frame}, time={self.total_time:.3f}, "
                f"solvers={list(self.scene.solvers.keys())}, "
                f"entities={len(self.scene.entities)})")


__all__ = [
    "AABB",
    "GUI",
    "CameraConfig",
    "CameraSensor",
    "ChemistryComponent",
    "CollisionShapeComponent",
    "CollisionSystem",
    "ComponentMask",
    "ConservativeCCD",
    # Collision
    "Contact",
    "ContactList",
    # Coupling
    "Coupler",
    "CouplerOptions",
    "CouplingData",
    "Entity",
    "EntityID",
    "EntityManager",
    "EnvConfig",
    "FEMNodeComponent",
    "ForceSensor",
    "GJKNarrowPhase",
    "GeologyComponent",
    "GlobalQuantities",
    "ImplicitEulerIntegrator",
    "MPMParticleComponent",
    "Observation",
    "PBDParticleComponent",
    "ParallelEnv",
    "RigidBodyComponent",
    "SAPBroadPhase",
    "SPHParticleComponent",
    # Core
    "Scene",
    "SensorData",
    "SensorType",
    # Solvers
    "Solver",
    "State",
    "SymplecticEulerIntegrator",
    "ThermalComponent",
    "TimeSeriesDataset",
    # Integrator
    "TimeStepper",
    "TimeStepperOptions",
    "TransformComponent",
    "VectorizedEnv",
    "VelocityVerletIntegrator",
    # WorldEngine
    "WorldEngine",
    "WorldEngineConfig",
    # AI Layer
    "WorldObserver",
    "create_collision_experiment",
    "create_coupler",
    "create_free_fall_experiment",
    "create_time_stepper",
    "create_world_engine",
    "load_gltf",
    "load_mjcf",
    # Interface
    "load_urdf",
    "parse_gltf",
    "parse_mjcf",
    "parse_urdf",
]

# Version
__version__ = "0.5.0"