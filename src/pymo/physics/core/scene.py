"""
Scene — Container for all entities and global simulation parameters.
Single source of truth for the physics world.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .entity import EntityManager, ComponentMask

if TYPE_CHECKING:
    from .entity import Entity, EntityID
    from .state import State
    from ..solvers.base import Solver
    from ..coupling.coupler import Coupler
    from ..collision.system import CollisionSystem
    from ..integrator.time_stepper import TimeStepper


@dataclass(slots=True)
class Scene:
    """
    Physics scene — holds all entities, global parameters, and solver references.
    One scene per simulation world.
    """
    # Entity management
    entities: EntityManager = field(default_factory=lambda: EntityManager())
    
    # Global simulation parameters
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float32))
    dt: float = 1.0 / 60.0
    substeps: int = 1
    time: float = 0.0
    frame: int = 0
    
    # Solver registry (name -> solver instance)
    solvers: dict[str, Solver] = field(default_factory=dict)
    
    # Shared systems
    coupler: Coupler | None = None
    collision_system: CollisionSystem | None = None
    time_stepper: TimeStepper | None = None
    
    # Rendering double buffer
    double_buffer_read: State | None = None
    double_buffer_write: State | None = None
    
    # Configuration
    enable_ai: bool = False
    checkpoint_interval: int = 0  # 0 = disabled
    checkpoint_path: str = "checkpoints/"
    
    # Statistics
    stats: dict = field(default_factory=dict)
    
    def add_entity(self, entity: Entity) -> Entity:
        """Add entity to scene. Stores the entity directly (preserving user_data, indices, etc.)."""
        self.entities._entities[entity.id] = entity
        self.entities._add_to_archetype(entity)
        self.entities._assign_indices(entity)
        return entity
    
    def remove_entity(self, entity_id: EntityID) -> None:
        """Remove entity from scene."""
        self.entities.destroy(entity_id)
    
    def get_entity(self, entity_id: EntityID) -> Entity | None:
        return self.entities.get(entity_id)
    
    def query_entities(self, mask: ComponentMask) -> list[Entity]:
        return self.entities.query(mask)
    
    def register_solver(self, name: str, solver: Solver) -> None:
        """Register a physics solver."""
        self.solvers[name] = solver
        solver.scene = self
    
    def get_solver(self, name: str) -> Solver | None:
        return self.solvers.get(name)
    
    def step(self) -> State:
        """Advance simulation by one frame (dt)."""
        if self.time_stepper is None:
            raise RuntimeError("TimeStepper not initialized")
        
        # Get current write state
        current_state = self.double_buffer_write
        if current_state is None:
            current_state = self._create_initial_state()
            self.double_buffer_write = current_state
        
        # Step simulation
        new_state = self.time_stepper.step(current_state)
        
        # Update scene time
        self.time += self.dt
        self.frame += 1
        
        # Swap double buffer (render reads from read buffer)
        self.double_buffer_read = new_state
        self.double_buffer_write = new_state
        
        # Checkpoint if enabled
        if self.checkpoint_interval > 0 and self.frame % self.checkpoint_interval == 0:
            self._checkpoint()
        
        return new_state
    
    def _create_initial_state(self) -> State:
        """Create initial state from entities."""
        from .state import State, GlobalQuantities
        from .entity import ComponentMask
        
        counts = self.entities.total_counts()
        
        rigid_quat = np.zeros((counts["rigid"], 4), dtype=np.float32)
        rigid_quat[:, 0] = 1.0
        
        state = State(
            t=self.time,
            dt=self.dt,
            # Rigid bodies
            rigid_pos=np.zeros((counts["rigid"], 3), dtype=np.float32),
            rigid_quat=rigid_quat,
            rigid_linvel=np.zeros((counts["rigid"], 3), dtype=np.float32),
            rigid_angvel=np.zeros((counts["rigid"], 3), dtype=np.float32),
            rigid_mass=np.ones(counts["rigid"], dtype=np.float32),
            rigid_inv_mass=np.ones(counts["rigid"], dtype=np.float32),
            rigid_inertia_local=np.tile(np.eye(3, dtype=np.float32), (counts["rigid"], 1, 1)),
            rigid_inv_inertia_local=np.tile(np.eye(3, dtype=np.float32), (counts["rigid"], 1, 1)),
            rigid_force_accum=np.zeros((counts["rigid"], 3), dtype=np.float32),
            rigid_torque_accum=np.zeros((counts["rigid"], 3), dtype=np.float32),
            # SPH
            sph_pos=np.zeros((counts["sph"], 3), dtype=np.float32),
            sph_vel=np.zeros((counts["sph"], 3), dtype=np.float32),
            sph_density=np.zeros(counts["sph"], dtype=np.float32),
            sph_pressure=np.zeros(counts["sph"], dtype=np.float32),
            sph_mass=np.zeros(counts["sph"], dtype=np.float32),
            # FEM
            fem_pos=np.zeros((counts["fem"], 3), dtype=np.float32),
            fem_vel=np.zeros((counts["fem"], 3), dtype=np.float32),
            fem_rest_pos=np.zeros((counts["fem"], 3), dtype=np.float32),
            # MPM
            mpm_pos=np.zeros((counts["mpm"], 3), dtype=np.float32),
            mpm_vel=np.zeros((counts["mpm"], 3), dtype=np.float32),
            mpm_mass=np.zeros(counts["mpm"], dtype=np.float32),
            mpm_volume=np.zeros(counts["mpm"], dtype=np.float32),
            mpm_F=np.tile(np.eye(3, dtype=np.float32), (counts["mpm"], 1, 1)),
            # PBD
            pbd_pos=np.zeros((counts["pbd"], 3), dtype=np.float32),
            pbd_pred_pos=np.zeros((counts["pbd"], 3), dtype=np.float32),
            pbd_vel=np.zeros((counts["pbd"], 3), dtype=np.float32),
            pbd_inv_mass=np.zeros(counts["pbd"], dtype=np.float32),
            # Thermal
            thermal_temp=np.zeros(counts["thermal"], dtype=np.float32),
            # Chemistry
            chem_conc=np.zeros((counts["chemistry"], 4), dtype=np.float32),
            # Geology
            geo_rock_type=np.zeros((1, 1, 1), dtype=np.int32),  # Placeholder
            geo_porosity=np.zeros((1, 1, 1), dtype=np.float32),
            # Global
            global_quantities=GlobalQuantities(),
        )
        
        # Fill initial values from entities
        self._populate_state_from_entities(state)
        
        return state
    
    def _populate_state_from_entities(self, state: State) -> None:
        """Copy entity component data into state arrays."""
        from .component import TransformComponent, RigidBodyComponent, CollisionShapeComponent

        # Build EntityID -> array-index mappings for solvers/collision lookups
        state.entity_to_rigid.clear()
        state.entity_to_sph.clear()
        for entity in self.entities:
            if entity.rigid_index is not None:
                state.entity_to_rigid[entity.id] = entity.rigid_index
            if entity.sph_start is not None:
                state.entity_to_sph[entity.id] = entity.sph_start

        for entity in self.entities:
            if entity.has(ComponentMask.RIGID_BODY) and entity.rigid_index is not None:
                i = entity.rigid_index
                
                # Get components from user_data (stored during create_rigid_body)
                transform = entity.user_data.get('transform')
                rb = entity.user_data.get('rigid_body')
                shape = entity.user_data.get('collision_shape')
                
                if transform is not None:
                    state.rigid_pos[i] = transform.position
                    state.rigid_quat[i] = transform.rotation
                
                if rb is not None:
                    state.rigid_mass[i] = rb.mass
                    state.rigid_inv_mass[i] = rb.inv_mass
                    state.rigid_inertia_local[i] = rb.inertia_local
                    state.rigid_inv_inertia_local[i] = rb.inv_inertia_local
    
    def _checkpoint(self) -> None:
        """Save checkpoint for reproducibility."""
        import torch
        import os
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, f"checkpoint_frame_{self.frame:08d}.pt")
        
        torch.save({
            "frame": self.frame,
            "time": self.time,
            "state": self.double_buffer_write,
            "rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }, path)
    
    def restore_checkpoint(self, path: str) -> None:
        """Restore from checkpoint."""
        import torch
        
        data = torch.load(path, map_location="cpu")
        self.frame = data["frame"]
        self.time = data["time"]
        self.double_buffer_write = data["state"]
        self.double_buffer_read = data["state"]
        np.random.set_state(data["rng_state"])
        torch.set_rng_state(data["torch_rng_state"])
    
    def get_render_snapshot(self) -> State | None:
        """Get immutable snapshot for rendering thread."""
        return self.double_buffer_read
    
    def __repr__(self) -> str:
        return (f"Scene(entities={len(self.entities)}, time={self.time:.3f}, "
                f"dt={self.dt}, solvers={list(self.solvers.keys())})")