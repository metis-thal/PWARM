"""
PWARM Physics Core — Unified Scene, State, Entity, Component System.

Single source of truth for all physics solvers. Immutable state, double-buffered for render.
"""

from .scene import Scene
from .state import State, GlobalQuantities
from .entity import Entity, EntityID, ComponentMask, EntityManager
from .component import (
    Component,
    TransformComponent,
    RigidBodyComponent,
    SPHParticleComponent,
    FEMNodeComponent,
    MPMParticleComponent,
    PBDParticleComponent,
    ThermalComponent,
    ChemistryComponent,
    GeologyComponent,
    CollisionShapeComponent,
)

__all__ = [
    "Scene",
    "State",
    "GlobalQuantities",
    "Entity",
    "EntityID",
    "ComponentMask",
    "EntityManager",
    "Component",
    "TransformComponent",
    "RigidBodyComponent",
    "SPHParticleComponent",
    "FEMNodeComponent",
    "MPMParticleComponent",
    "PBDParticleComponent",
    "ThermalComponent",
    "ChemistryComponent",
    "GeologyComponent",
    "CollisionShapeComponent",
]