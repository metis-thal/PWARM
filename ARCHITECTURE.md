# PWARM — Classic Physics World Simulation Architecture

**Reference:** Genesis World architecture (multi-solver, unified scene/state, explicit coupler)
**Core Principles:**
1. **Zero hardcoded macro phenomena** — combustion, fracture, flow MUST emerge from equations
2. **Ground truth / AI separation** — physics engine is absolute truth; AI is learned approximation
3. **Visual acceptance every phase** — no headless development; every phase demoable
4. **Reproducibility** — given identical initial conditions, identical evolution trajectory; checkpoint restore
5. **Conservation monitoring** — track mass, energy, momentum; prevent malignant injection/loss
6. **Sim/render separation** — render reads immutable snapshots; never drives simulation

---

## Four-Layer Architecture (Genesis-inspired)

```
pymo/
├── compiler/          # [Future] Python → CUDA/Metal/Vulkan (Quadrants-style)
├── render/            # Visualization: GPU instancing, PBR, ray-tracing, camera sensors
├── physics/           # Unified multi-physics engine (THIS LAYER)
│   ├── core/          # Scene, State, Time, Entity, Component
│   ├── solvers/       # Rigid, SPH, FEM/MPM, PBD, Thermal, Chemistry, Geology
│   ├── coupling/      # Explicit coupler: rigid↔SPH, rigid↔FEM, SPH↔FEM, thermal↔all
│   ├── collision/     # Broad phase (SAP) + Narrow phase (GJK/EPA, CCD)
│   └── integrator/    # Velocity-Verlet, symplectic Euler, implicit for stiff systems
├── ai/                # Observer, Law Discovery, Closed-Loop Experimentation
└── interface/         # Asset parsing (URDF/MJCF/GLB), GUI, sensors, parallel envs
```

---

## Physics Layer: Unified Scene & State

**Single source of truth** — one `Scene`, one `State` shared by ALL solvers.

```python
# physics/core/scene.py
class Scene:
    """Container for all entities and global simulation parameters."""
    entities: list[Entity]           # All entities in the world
    gravity: Vector3                 # Global gravity
    dt: float                        # Base time step
    substeps: int                    # Sub-steps per frame
    coupler: Coupler                 # Multi-physics coupler
    collision_system: CollisionSystem # Shared collision detection
    
# physics/core/state.py
class State:
    """Immutable snapshot at time t. Double-buffered for render thread."""
    t: float                         # Simulation time
    entity_states: dict[EntityID, EntityState]  # Per-entity data
    global_quantities: GlobalQuantities         # Mass, energy, momentum totals
    
class EntityState:
    """Per-entity physics state (varies by solver type)."""
    # Rigid body
    pos: Vector3
    quat: Quaternion
    linvel: Vector3
    angvel: Vector3
    # SPH/Particle
    particle_pos: array[N, 3]
    particle_vel: array[N, 3]
    particle_mass: array[N]
    # FEM/MPM
    node_pos: array[N, 3]
    node_vel: array[N, 3]
    deformation_grad: array[N, 3, 3]
    # Thermal
    temperature: array[N]
    # Chemistry
    species_concentration: array[N, num_species]
    # Geology
    rock_type: array[N]
    porosity: array[N]
```

---

## Solver Registry (Extensible)

```python
# physics/solvers/__init__.py
class Solver(ABC):
    """Base class for all physics solvers."""
    name: str
    required_components: list[ComponentType]
    
    @abstractmethod
    def step(self, state: State, dt: float) -> State:
        """Advance solver by dt. Returns new State (immutable)."""
    
    @abstractmethod
    def get_coupling_data(self, state: State) -> CouplingData:
        """Export data needed by other solvers (forces, velocities, temps)."""
    
    @abstractmethod
    def apply_coupling(self, state: State, coupling: CouplingData) -> State:
        """Apply forces/constraints from other solvers."""

# Registry
SOLVERS = {
    "rigid": RigidSolver,      # GJK/EPA, impulse-based contacts
    "sph": SPHSolver,          # Navier-Stokes via SPH
    "fem": FEMSolver,          # Deformable solids, implicit integration
    "mpm": MPMSolver,          # Material Point Method (sand, snow, clay)
    "pbd": PBDSolver,          # Position-Based Dynamics (cloth, hair)
    "thermal": ThermalSolver,  # Heat equation (implicit Euler)
    "chemistry": ChemistrySolver, # Reaction-diffusion
    "geology": GeologySolver,  # Stratigraphy, erosion, tectonics
}
```

---

## Explicit Coupler (Genesis-style)

```python
# physics/coupling/coupler.py
class Coupler:
    """Manages multi-physics interaction. Explicit coupling strategy."""
    
    def __init__(self, scene: Scene):
        self.scene = scene
        self.enabled_pairs = {
            ("rigid", "sph"): True,
            ("rigid", "fem"): True,
            ("rigid", "mpm"): True,
            ("sph", "fem"): True,
            ("thermal", "rigid"): True,
            ("thermal", "sph"): True,
            ("thermal", "fem"): True,
            ("thermal", "chemistry"): True,
            ("chemistry", "sph"): True,
            ("geology", "thermal"): True,
            ("geology", "rigid"): True,
        }
    
    def couple(self, state: State, dt: float) -> State:
        """Execute one coupling iteration."""
        # 1. Each solver exports coupling data
        coupling_data = {name: s.get_coupling_data(state) 
                        for name, s in self.scene.solvers.items()}
        
        # 2. Compute interaction forces/constraints per enabled pair
        for (a, b), enabled in self.enabled_pairs.items():
            if not enabled: continue
            force_a_on_b, force_b_on_a = self._compute_interaction(
                coupling_data[a], coupling_data[b], dt)
            coupling_data[a].add_force(force_b_on_a)
            coupling_data[b].add_force(force_a_on_b)
        
        # 3. Each solver applies received coupling
        new_state = state
        for name, solver in self.scene.solvers.items():
            new_state = solver.apply_coupling(new_state, coupling_data[name])
        
        return new_state
    
    def _compute_interaction(self, data_a, data_b, dt):
        """Pairwise interaction: rigid↔fluid, rigid↔deformable, fluid↔deformable, thermal↔all."""
        # Rigid-Fluid: boundary particles on rigid surface exert pressure/viscosity on fluid
        # Rigid-Deformable: contact constraints via penalty or Lagrange multipliers
        # Thermal: heat flux = -k * grad(T) across material interfaces
        # Chemistry: species diffusion across phase boundaries
        pass
```

---

## Collision System (Shared by All Solvers)

```python
# physics/collision/system.py
class CollisionSystem:
    """Unified collision detection for all solvers."""
    
    def __init__(self):
        self.broad_phase = SAPBroadPhase()      # Sweep and Prune
        self.narrow_phase = GJKNarrowPhase()    # GJK/EPA for convex
        self.ccd = ConservativeCCD()            # Continuous collision detection
    
    def detect(self, state: State) -> ContactList:
        """Return all contacts for current state."""
        # 1. Broad phase: AABB overlap pairs
        pairs = self.broad_phase.query(state)
        
        # 2. Narrow phase: exact contact (point, normal, depth)
        contacts = []
        for a, b in pairs:
            contact = self.narrow_phase.collide(a, b, state)
            if contact:
                contacts.append(contact)
        
        # 3. CCD for fast-moving objects
        ccd_contacts = self.ccd.sweep(state, self.narrow_phase)
        contacts.extend(ccd_contacts)
        
        return ContactList(contacts)

class Contact:
    entity_a: EntityID
    entity_b: EntityID
    point: Vector3
    normal: Vector3
    depth: float
    friction: float
    restitution: float
    solver_types: tuple[SolverType, SolverType]  # e.g., ("rigid", "sph")
```

---

## Time Integration (Unified Loop)

```python
# physics/integrator/time_stepper.py
class TimeStepper:
    """Unified time stepping with sub-steps and coupling iterations."""
    
    def __init__(self, scene: Scene):
        self.scene = scene
        self.dt = scene.dt
        self.substeps = scene.substeps
        self.coupling_iterations = 3  # Gauss-Seidel style
    
    def step(self, state: State) -> State:
        """Advance simulation by one frame (dt)."""
        sub_dt = self.dt / self.substeps
        
        for _ in range(self.substeps):
            # 1. Collision detection (once per sub-step)
            contacts = self.scene.collision_system.detect(state)
            
            # 2. Solver steps (can be parallelized per solver)
            solver_states = {}
            for name, solver in self.scene.solvers.items():
                solver_states[name] = solver.step(state, sub_dt, contacts)
            
            # 3. Merge solver states (state is additive for disjoint DOFs)
            state = self._merge_states(state, solver_states)
            
            # 4. Coupling iterations
            for _ in range(self.coupling_iterations):
                state = self.scene.coupler.couple(state, sub_dt)
            
            # 5. Conservation check
            self._check_conservation(state)
        
        return state
    
    def _merge_states(self, base: State, solver_states: dict) -> State:
        """Merge per-solver updates. Each solver owns disjoint DOFs."""
        # Rigid: pos, quat, linvel, angvel
        # SPH: particle_pos, particle_vel
        # FEM/MPM: node_pos, node_vel, deformation_grad
        # Thermal: temperature
        # Chemistry: species_concentration
        # Geology: rock_type, porosity
        pass
```

---

## AI Layer (Observation → Hypothesis → Verification → Experiment)

```python
# ai/observer.py
class WorldObserver:
    """Samples ground-truth state at intervals. Produces dataset for AI."""
    def __init__(self, scene: Scene, sample_every: int = 1):
        self.scene = scene
        self.sample_every = sample_every
        self.trajectory = []
    
    def observe(self, steps: int) -> Dataset:
        for i in range(steps):
            state = self.scene.time_stepper.step(self.scene.current_state)
            self.scene.current_state = state
            if i % self.sample_every == 0:
                self.trajectory.append(self._extract_observables(state))
        return Dataset(self.trajectory)

# ai/law_discovery.py
class LawDiscovery:
    """Symbolic regression on observed quantities. Discovers governing equations."""
    def discover(self, dataset: Dataset, quantities: list[str]) -> list[Law]:
        # Uses gplearn / PySR for symbolic regression
        # Returns laws like: y(t) = 10 - 4.905*t^2
        pass

# ai/closed_loop.py
class ClosedLoopAI:
    """Full cycle: observe → hypothesize → verify → experiment → refine."""
    def run(self, world_engine: WorldEngine, target_quantities: list[str]):
        # 1. Observe
        dataset = WorldObserver(world_engine.scene).observe(1000)
        # 2. Hypothesize
        laws = LawDiscovery().discover(dataset, target_quantities)
        # 3. Verify (compare predictions vs ground truth)
        errors = self._verify(laws, dataset)
        # 4. Experiment (design interventions to reduce uncertainty)
        experiments = self._design_experiments(laws, errors)
        # 5. Execute experiments in world engine
        for exp in experiments:
            world_engine.execute_experiment(exp)
        # 6. Refine
        return self.run(world_engine, target_quantities)  # Recursive refinement
```

---

## Interface Layer

```python
# interface/asset_parser.py
def load_urdf(path: str) -> list[Entity]:
    """Parse URDF → rigid bodies + joints + collision shapes."""

def load_mjcf(path: str) -> list[Entity]:
    """Parse MuJoCo XML → full scene."""

def load_gltf(path: str) -> list[Entity]:
    """Parse GLB/GLTF → mesh + materials + physics properties."""

# interface/gui.py
class GUI:
    """Built-in viewer with camera sensors, entity inspector, parameter tuning."""
    def __init__(self, scene: Scene):
        self.renderer = NyxRenderer(scene)  # or Luisa/Pyrender
    
    def run(self):
        while self.running:
            # Render reads from double-buffered snapshot
            snapshot = self.scene.double_buffer.read()
            self.renderer.render(snapshot)
            self._handle_input()

# interface/sensors.py
class CameraSensor:
    """RGB, depth, segmentation, optical flow."""
    def render(self, snapshot: State) -> SensorData:
        pass

class ForceSensor:
    """Contact forces, joint torques."""
    pass

# interface/parallel.py
class ParallelEnv:
    """Heterogeneous parallel environments (Ray/MPS)."""
    def __init__(self, num_envs: int, scene_fn: Callable):
        self.envs = [scene_fn() for _ in range(num_envs)]
    
    def step(self, actions: list) -> list[State]:
        # Vectorized step across all envs
        pass
```

---

## Configuration (Pydantic, Genesis-style)

```python
# config/sim_options.py
class SimOptions(BaseModel):
    dt: float = 1e-2
    substeps: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    floor_height: float = 0.0
    requires_grad: bool = False  # For differentiable physics

# config/coupler_options.py
class CouplerOptions(BaseModel):
    rigid_sph: bool = True
    rigid_fem: bool = True
    rigid_mpm: bool = True
    sph_fem: bool = True
    thermal_all: bool = True
    chemistry_sph: bool = True
    geology_thermal: bool = True

# config/solver_options.py
class RigidOptions(BaseModel):
    solver_type: Literal["impulse", "lcps"] = "impulse"
    contact_tolerance: float = 1e-4
    max_iterations: int = 20

class SPHOptions(BaseModel):
    particle_radius: float = 0.025
    kernel: Literal["cubic", "quintic", "wendland"] = "cubic"
    viscosity: float = 0.1
    surface_tension: float = 0.072

class ThermalOptions(BaseModel):
    implicit: bool = True
    conductivity: dict[MaterialID, float] = {}
    specific_heat: dict[MaterialID, float] = {}
```

---

## WorldEngine (Unified Entry Point)

```python
# kernel/world_engine.py
class WorldEngine:
    """Single entry point for all simulation. Couples everything."""
    
    def __init__(self, config: WorldEngineConfig):
        # Build scene
        self.scene = Scene(config.sim_options)
        self.scene.gravity = config.sim_options.gravity
        self.scene.dt = config.sim_options.dt
        self.scene.substeps = config.sim_options.substeps
        
        # Instantiate enabled solvers
        if config.enable_rigid:
            self.scene.solvers["rigid"] = RigidSolver(config.rigid_options)
        if config.enable_sph:
            self.scene.solvers["sph"] = SPHSolver(config.sph_options)
        if config.enable_fem:
            self.scene.solvers["fem"] = FEMSolver(config.fem_options)
        if config.enable_mpm:
            self.scene.solvers["mpm"] = MPMSolver(config.mpm_options)
        if config.enable_pbd:
            self.scene.solvers["pbd"] = PBDSolver(config.pbd_options)
        if config.enable_thermal:
            self.scene.solvers["thermal"] = ThermalSolver(config.thermal_options)
        if config.enable_chemistry:
            self.scene.solvers["chemistry"] = ChemistrySolver(config.chem_options)
        if config.enable_geology:
            self.scene.solvers["geology"] = GeologySolver(config.geo_options)
        
        # Shared systems
        self.scene.collision_system = CollisionSystem()
        self.scene.coupler = Coupler(self.scene, config.coupler_options)
        self.time_stepper = TimeStepper(self.scene)
        
        # Double buffer for rendering
        self.double_buffer = DoubleBuffer()
        
        # AI layer
        if config.enable_ai:
            self.ai = ClosedLoopAI()
    
    def tick(self) -> State:
        """Single simulation step. Returns new immutable state."""
        new_state = self.time_stepper.step(self.current_state)
        self.current_state = new_state
        self.double_buffer.swap(new_state)
        return new_state
    
    def run(self, steps: int):
        for _ in range(steps):
            self.tick()
    
    def checkpoint(self, path: str):
        """Save full state for reproducibility."""
        torch.save({
            "state": self.current_state,
            "scene_config": self.scene.config,
            "rng_state": torch.get_rng_state(),
        }, path)
    
    def restore(self, path: str):
        """Restore from checkpoint."""
        data = torch.load(path)
        self.current_state = data["state"]
        torch.set_rng_state(data["rng_state"])
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        WORLD ENGINE                              │
├─────────────────────────────────────────────────────────────────┤
│  SCENE (single)                                                 │
│  ├── Entities (rigid, fluid, deformable, thermal, chemical,     │
│  │          geological) — each with Component mask              │
│  ├── Global params (gravity, dt, substeps)                      │
│  └── Coupler config                                             │
├─────────────────────────────────────────────────────────────────┤
│  STATE (immutable, double-buffered)                             │
│  ├── t, entity_states[entity_id] → EntityState                  │
│  └── global_quantities (mass, energy, momentum)                 │
├─────────────────────────────────────────────────────────────────┤
│  TIME STEPPER                                                   │
│  ├── For each sub-step:                                         │
│  │   1. CollisionSystem.detect(State) → ContactList             │
│  │   2. Solver.step(State, dt, Contacts) → partial State        │
│  │   3. Merge partial States                                    │
│  │   4. Coupler.couple(State, dt) × N iterations               │
│  │   5. Conservation check                                       │
│  └── Swap double buffer                                         │
├─────────────────────────────────────────────────────────────────┤
│  SOLVERS (all read/write shared State)                          │
│  ├── RigidSolver     → pos, quat, linvel, angvel               │
│  ├── SPHSolver       → particle_pos, particle_vel, density     │
│  ├── FEMSolver       → node_pos, node_vel, F                   │
│  ├── MPMSolver       → particle_pos, particle_vel, F           │
│  ├── PBDSolver       → particle_pos, constraints               │
│  ├── ThermalSolver   → temperature                             │
│  ├── ChemistrySolver → species_concentration                   │
│  └── GeologySolver   → rock_type, porosity, stratigraphy       │
├─────────────────────────────────────────────────────────────────┤
│  COUPLER (explicit, pairwise)                                   │
│  ├── rigid↔sph: boundary particles, buoyancy, drag             │
│  ├── rigid↔fem/mpm: contact constraints, friction              │
│  ├── sph↔fem/mpm: fluid pressure on deformable, porosity       │
│  ├── thermal↔all: heat flux, latent heat, reaction heat        │
│  ├── chemistry↔sph: species diffusion, reaction sources        │
│  └── geology↔thermal: crustal heat flow, radiogenic heating    │
├─────────────────────────────────────────────────────────────────┤
│  RENDER (reads ONLY from double buffer)                         │
│  ├── GPU instancing for rigid/particle entities                │
│  ├── PBR shaders with temperature emission                     │
│  ├── Nyx/Luisa/Pyrender backends                               │
│  └── Camera sensors: RGB, depth, seg, flow                     │
├─────────────────────────────────────────────────────────────────┤
│  AI LAYER (observes ground truth, discovers laws)               │
│  ├── WorldObserver → Dataset                                    │
│  ├── LawDiscovery → Symbolic expressions                       │
│  ├── Verification → Relative error vs ground truth             │
│  └── ClosedLoopAI → Experiment design → execute → refine       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Scene, State, Entity, Component system | 🔄 In progress |
| 2 | RigidSolver (GJK/EPA, impulse contacts) | ✅ Done (kernel/collision3d, world3d) |
| 3 | SPHSolver (WCSPH, boundary particles) | ✅ Done (rules/fluid) |
| 4 | ThermalSolver (implicit heat eq) | ✅ Done (geology/processes/thermal) |
| 5 | GeologySolver (stratigraphy, erosion) | ✅ Done (geology/) |
| 6 | FEMSolver / MPMSolver | ⏳ TODO |
| 7 | PBDSolver | ⏳ TODO |
| 8 | ChemistrySolver | ⏳ TODO |
| 9 | Explicit Coupler (all pairs) | ⏳ TODO |
| 10 | SAP Broad Phase + CCD | ⏳ TODO |
| 11 | Double Buffer + Nyx Renderer | ✅ Done (viz/gl_renderer) |
| 12 | AI Layer (Observer, LawDiscovery, ClosedLoop) | ✅ Done (ai/) |
| 13 | Asset Parsers (URDF, MJCF, GLTF) | ⏳ TODO |
| 14 | GUI + Parallel Envs | ⏳ TODO |

---

## Key Differences from Current Codebase

| Current | Target (Classic Physics World) |
|---------|-------------------------------|
| Separate World / World3D / WorldEngine | **Single Scene + Single State** |
| Solvers loosely coupled via WorldEngine | **Explicit Coupler with pairwise interaction** |
| Collision per-solver | **Unified CollisionSystem (SAP + GJK/EPA + CCD)** |
| Time stepping per-module | **Unified TimeStepper with sub-steps & coupling iterations** |
| State mutable, passed by reference | **Immutable State, double-buffered for render** |
| AI separate from physics | **AI observes ground truth, never drives physics** |
| No asset parsing | **URDF/MJCF/GLTF loaders** |
| No parallel envs | **Ray/MPS heterogeneous environments** |
| No compiler layer | **Future: Quadrants (Python → GPU kernels)** |

---

## Next Steps

1. **Create `physics/core/`** — Scene, State, Entity, Component, TimeStepper
2. **Refactor existing solvers** to inherit `Solver` base class with `step()`, `get_coupling_data()`, `apply_coupling()`
3. **Implement `Coupler`** with all enabled pairs
4. **Implement `CollisionSystem`** (SAP + GJK/EPA + CCD) shared by all
5. **Add FEMSolver, MPMSolver, PBDSolver, ChemistrySolver**
6. **Wire double buffer** between TimeStepper and GLRenderer
7. **Add asset parsers** (URDF, MJCF, GLTF)
8. **Build GUI** with Nyx renderer
9. **Add parallel environment support**
10. **Documentation & examples** for each solver + coupling demos