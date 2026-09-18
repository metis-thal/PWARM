"""
PWARM Physics Core — Unified Scene, State, Entity, Component System.

Single source of truth for all physics solvers. Immutable state, double-buffered for render.
"""

from .component import (
    ChemistryComponent,
    CollisionShapeComponent,
    Component,
    FEMNodeComponent,
    GeologyComponent,
    MPMParticleComponent,
    PBDParticleComponent,
    RigidBodyComponent,
    SPHParticleComponent,
    ThermalComponent,
    TransformComponent,
)
from .entity import ComponentMask, Entity, EntityID, EntityManager
from .scene import Scene
from .state import GlobalQuantities, State

__all__ = [
    "ChemistryComponent",
    "CollisionShapeComponent",
    "Component",
    "ComponentMask",
    "Entity",
    "EntityID",
    "EntityManager",
    "FEMNodeComponent",
    "GeologyComponent",
    "GlobalQuantities",
    "MPMParticleComponent",
    "PBDParticleComponent",
    "RigidBodyComponent",
    "SPHParticleComponent",
    "Scene",
    "State",
    "ThermalComponent",
    "TransformComponent",
]