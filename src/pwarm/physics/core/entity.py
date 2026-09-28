"""
Entity and Component System — Archetype-based ECS for physics solvers.

Each entity has a ComponentMask indicating which solvers own its data.
Solvers only process entities with their required components.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntFlag


class ComponentMask(IntFlag):
    """Bitmask of components an entity possesses. Determines which solvers process it."""
    NONE = 0
    TRANSFORM = 1 << 0           # Position, rotation (all entities)
    RIGID_BODY = 1 << 1          # Mass, inertia, velocity (RigidSolver)
    SPH_PARTICLE = 1 << 2        # Particle arrays (SPHSolver)
    FEM_NODE = 1 << 3            # Node arrays (FEMSolver)
    MPM_PARTICLE = 1 << 4        # Particle arrays (MPMSolver)
    PBD_PARTICLE = 1 << 5        # Particle arrays (PBDSolver)
    THERMAL = 1 << 6             # Temperature field (ThermalSolver)
    CHEMISTRY = 1 << 7           # Species concentrations (ChemistrySolver)
    GEOLOGY = 1 << 8             # Rock type, porosity (GeologySolver)
    COLLISION_SHAPE = 1 << 9     # Collision geometry (CollisionSystem)
    # Composite masks for common combinations
    RIGID_DYNAMIC = TRANSFORM | RIGID_BODY | COLLISION_SHAPE
    RIGID_STATIC = TRANSFORM | COLLISION_SHAPE
    FLUID = TRANSFORM | SPH_PARTICLE | COLLISION_SHAPE
    DEFORMABLE_FEM = TRANSFORM | FEM_NODE | COLLISION_SHAPE
    DEFORMABLE_MPM = TRANSFORM | MPM_PARTICLE | COLLISION_SHAPE
    CLOTH = TRANSFORM | PBD_PARTICLE | COLLISION_SHAPE
    THERMAL_OBJECT = TRANSFORM | THERMAL | COLLISION_SHAPE
    REACTING_FLUID = TRANSFORM | SPH_PARTICLE | CHEMISTRY | COLLISION_SHAPE
    GEOLOGICAL = TRANSFORM | GEOLOGY | THERMAL | COLLISION_SHAPE


@dataclass(slots=True)
class EntityID:
    """Unique entity identifier."""
    value: uuid.UUID = field(default_factory=uuid.uuid4)
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityID):
            return NotImplemented
        return self.value == other.value
    
    def __str__(self) -> str:
        return str(self.value)[:8]


@dataclass(slots=True)
class Entity:
    """
    Physics entity — container for components.
    All data lives in State; Entity is just an ID + component mask.
    """
    id: EntityID
    mask: ComponentMask = ComponentMask.NONE
    name: str = ""
    # Solver-specific data indices (into State arrays)
    rigid_index: int | None = None
    sph_start: int | None = None
    sph_count: int | None = None
    fem_start: int | None = None
    fem_count: int | None = None
    mpm_start: int | None = None
    mpm_count: int | None = None
    pbd_start: int | None = None
    pbd_count: int | None = None
    thermal_index: int | None = None
    chemistry_index: int | None = None
    geology_index: int | None = None
    # Metadata
    user_data: dict = field(default_factory=dict)
    
    def has(self, component: ComponentMask) -> bool:
        return (self.mask & component) != ComponentMask.NONE
    
    def add(self, component: ComponentMask) -> Entity:
        self.mask |= component
        return self
    
    def remove(self, component: ComponentMask) -> Entity:
        self.mask &= ~component
        return self
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


class EntityManager:
    """Manages entity creation, deletion, and archetype queries."""
    
    def __init__(self):
        self._entities: dict[EntityID, Entity] = {}
        self._archetypes: dict[ComponentMask, set[EntityID]] = {}
        self._next_rigid_index = 0
        self._next_sph_index = 0
        self._next_fem_index = 0
        self._next_mpm_index = 0
        self._next_pbd_index = 0
        self._next_thermal_index = 0
        self._next_chemistry_index = 0
        self._next_geology_index = 0
    
    def create(self, name: str = "", mask: ComponentMask = ComponentMask.NONE) -> Entity:
        entity = Entity(id=EntityID(), name=name, mask=mask)
        self._entities[entity.id] = entity
        self._add_to_archetype(entity)
        self._assign_indices(entity)
        return entity
    
    def destroy(self, entity_id: EntityID) -> None:
        if entity_id in self._entities:
            entity = self._entities.pop(entity_id)
            self._remove_from_archetype(entity)
    
    def get(self, entity_id: EntityID) -> Entity | None:
        return self._entities.get(entity_id)
    
    def query(self, mask: ComponentMask) -> list[Entity]:
        """Return all entities with ALL bits in mask set."""
        result = []
        for archetype_mask, entity_ids in self._archetypes.items():
            if (archetype_mask & mask) == mask:
                result.extend(self._entities[eid] for eid in entity_ids)
        return result
    
    def query_exact(self, mask: ComponentMask) -> list[Entity]:
        """Return entities with EXACTLY this mask."""
        entity_ids = self._archetypes.get(mask, set())
        return [self._entities[eid] for eid in entity_ids]
    
    def _add_to_archetype(self, entity: Entity) -> None:
        if entity.mask not in self._archetypes:
            self._archetypes[entity.mask] = set()
        self._archetypes[entity.mask].add(entity.id)
    
    def _remove_from_archetype(self, entity: Entity) -> None:
        if entity.mask in self._archetypes:
            self._archetypes[entity.mask].discard(entity.id)
            if not self._archetypes[entity.mask]:
                del self._archetypes[entity.mask]
    
    def _assign_indices(self, entity: Entity) -> None:
        """Assign solver array indices for new components."""
        if entity.has(ComponentMask.RIGID_BODY):
            entity.rigid_index = self._next_rigid_index
            self._next_rigid_index += 1
        if entity.has(ComponentMask.SPH_PARTICLE):
            entity.sph_start = self._next_sph_index
            # Count assigned later when particles are created
        if entity.has(ComponentMask.FEM_NODE):
            entity.fem_start = self._next_fem_index
        if entity.has(ComponentMask.MPM_PARTICLE):
            entity.mpm_start = self._next_mpm_index
        if entity.has(ComponentMask.PBD_PARTICLE):
            entity.pbd_start = self._next_pbd_index
        if entity.has(ComponentMask.THERMAL):
            entity.thermal_index = self._next_thermal_index
            self._next_thermal_index += 1
        if entity.has(ComponentMask.CHEMISTRY):
            entity.chemistry_index = self._next_chemistry_index
            self._next_chemistry_index += 1
        if entity.has(ComponentMask.GEOLOGY):
            entity.geology_index = self._next_geology_index
            self._next_geology_index += 1
    
    def set_particle_counts(self, entity: Entity, sph: int = 0, fem: int = 0, 
                            mpm: int = 0, pbd: int = 0) -> None:
        """Update particle/node counts and advance global indices."""
        if entity.has(ComponentMask.SPH_PARTICLE) and sph > 0:
            entity.sph_count = sph
            self._next_sph_index += sph
        if entity.has(ComponentMask.FEM_NODE) and fem > 0:
            entity.fem_count = fem
            self._next_fem_index += fem
        if entity.has(ComponentMask.MPM_PARTICLE) and mpm > 0:
            entity.mpm_count = mpm
            self._next_mpm_index += mpm
        if entity.has(ComponentMask.PBD_PARTICLE) and pbd > 0:
            entity.pbd_count = pbd
            self._next_pbd_index += pbd
    
    def total_counts(self) -> dict[str, int]:
        return {
            "rigid": self._next_rigid_index,
            "sph": self._next_sph_index,
            "fem": self._next_fem_index,
            "mpm": self._next_mpm_index,
            "pbd": self._next_pbd_index,
            "thermal": self._next_thermal_index,
            "chemistry": self._next_chemistry_index,
            "geology": self._next_geology_index,
        }
    
    def __iter__(self):
        return iter(self._entities.values())
    
    def __len__(self) -> int:
        return len(self._entities)