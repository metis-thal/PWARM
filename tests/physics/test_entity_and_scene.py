"""Tests for the ECS core and Scene facade (physics.core).

Complements the solver tests: EntityManager archetypes and index assignment,
Scene state construction from entities, double-buffer stepping, and the
solver registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.physics.core.component import (
    RigidBodyComponent,
    TransformComponent,
)
from pwarm.physics.core.entity import ComponentMask, Entity, EntityID, EntityManager
from pwarm.physics.core.scene import Scene
from pwarm.physics.core.state import State


# --------------------------------------------------------------------- EntityID
def test_entity_id_identity_semantics() -> None:
    """EntityIDs hash/compare by UUID and render as an 8-char prefix."""
    a = EntityID()
    b = EntityID()
    assert a != b
    assert a == EntityID(a.value)
    assert hash(a) == hash(EntityID(a.value))
    assert len(str(a)) == 8
    assert (a == 42) is False  # non-EntityID comparison is NotImplemented


# ------------------------------------------------------------------ EntityManager
def test_entity_manager_create_and_query() -> None:
    """Creation registers archetypes; queries filter by mask membership."""
    manager = EntityManager()
    rigid = manager.create("ball", ComponentMask.RIGID_DYNAMIC)
    manager.create("water", ComponentMask.FLUID)
    assert len(manager) == 2
    assert {e.name for e in manager.query(ComponentMask.RIGID_BODY)} == {"ball"}
    assert {e.name for e in manager.query(ComponentMask.TRANSFORM)} == {"ball", "water"}
    exact = manager.query_exact(ComponentMask.FLUID)
    assert [e.name for e in exact] == ["water"]
    assert manager.get(rigid.id) is rigid
    assert manager.get(EntityID()) is None


def test_entity_manager_index_assignment() -> None:
    """Rigid/thermal/chemistry/geology indices increment per creation."""
    manager = EntityManager()
    a = manager.create("a", ComponentMask.RIGID_DYNAMIC)
    b = manager.create("b", ComponentMask.RIGID_DYNAMIC)
    t = manager.create("hot", ComponentMask.THERMAL_OBJECT)
    c = manager.create("react", ComponentMask.REACTING_FLUID)
    g = manager.create("rock", ComponentMask.GEOLOGICAL)
    assert (a.rigid_index, b.rigid_index) == (0, 1)
    assert t.thermal_index == 0
    # REACTING_FLUID carries SPH+CHEMISTRY but no THERMAL bit
    assert (c.thermal_index, c.chemistry_index, c.sph_start) == (None, 0, 0)
    assert (g.geology_index, g.thermal_index) == (0, 1)
    counts = manager.total_counts()
    assert counts["rigid"] == 2 and counts["thermal"] == 2
    assert counts["chemistry"] == 1 and counts["geology"] == 1


def test_entity_manager_destroy_removes_archetype() -> None:
    """destroy() removes the entity and prunes the empty archetype bucket."""
    manager = EntityManager()
    entity = manager.create("solo", ComponentMask.RIGID_DYNAMIC)
    mask = entity.mask
    assert mask in manager._archetypes
    manager.destroy(entity.id)
    assert len(manager) == 0
    assert mask not in manager._archetypes
    manager.destroy(entity.id)  # unknown id: silent no-op


def test_entity_manager_particle_counts() -> None:
    """set_particle_counts records counts and advances the global indices."""
    manager = EntityManager()
    fluid = manager.create("fluid", ComponentMask.FLUID)
    cloth = manager.create("cloth", ComponentMask.CLOTH)
    manager.set_particle_counts(fluid, sph=100)
    manager.set_particle_counts(cloth, pbd=50, fem=0)  # fem=0 ignored
    assert fluid.sph_count == 100
    assert cloth.pbd_count == 50
    assert manager.total_counts()["sph"] == 100
    assert manager.total_counts()["pbd"] == 50
    assert manager.total_counts()["fem"] == 0


# ------------------------------------------------------------------------- Scene
def _rigid_entity(name: str, position, mass: float = 1.0) -> Entity:
    entity = Entity(id=EntityID(), name=name, mask=ComponentMask.RIGID_DYNAMIC)
    transform = TransformComponent()
    transform.position = np.asarray(position, dtype=np.float32)
    rb = RigidBodyComponent()
    rb.mass = mass
    rb.inv_mass = 1.0 / mass if mass > 0 else 0.0
    entity.user_data["transform"] = transform
    entity.user_data["rigid_body"] = rb
    return entity


def test_scene_add_remove_and_query() -> None:
    """add_entity stores user data; remove/get/query round-trip."""
    scene = Scene()
    entity = _rigid_entity("ball", (1.0, 2.0, 3.0))
    scene.add_entity(entity)
    assert scene.get_entity(entity.id) is entity
    assert len(scene.query_entities(ComponentMask.RIGID_BODY)) == 1
    scene.remove_entity(entity.id)
    assert scene.get_entity(entity.id) is None


def test_scene_register_solver() -> None:
    """register_solver stores the solver and back-links the scene."""

    class Dummy:
        pass

    scene = Scene()
    dummy = Dummy()
    scene.register_solver("dummy", dummy)  # type: ignore[arg-type]
    assert scene.get_solver("dummy") is dummy
    assert dummy.scene is scene  # type: ignore[attr-defined]
    assert scene.get_solver("missing") is None


def test_scene_step_without_stepper_raises() -> None:
    """step() before a TimeStepper is wired raises RuntimeError."""
    scene = Scene()
    with pytest.raises(RuntimeError, match="TimeStepper not initialized"):
        scene.step()


class FakeStepper:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, state: State) -> State:
        self.calls += 1
        state = state.copy()
        state.t += state.dt
        return state


def test_scene_step_swaps_double_buffers() -> None:
    """step() advances time/frame and publishes to both buffers."""
    scene = Scene()
    scene.time_stepper = FakeStepper()  # type: ignore[assignment]
    first = scene.step()
    assert scene.frame == 1
    assert scene.time == pytest.approx(scene.dt)
    assert scene.double_buffer_read is first
    assert scene.double_buffer_write is first
    assert scene.get_render_snapshot() is first
    second = scene.step()
    assert second is not first
    assert scene.frame == 2


def test_scene_initial_state_from_entities() -> None:
    """_create_initial_state sizes arrays from entity counts and fills them."""
    scene = Scene()
    scene.add_entity(_rigid_entity("a", (1.0, 0.0, 0.0), mass=2.0))
    scene.add_entity(_rigid_entity("b", (0.0, 5.0, 0.0), mass=0.0))  # static
    state = scene._create_initial_state()
    assert state.rigid_pos.shape == (2, 3)
    assert np.allclose(state.rigid_pos[0], [1.0, 0.0, 0.0])
    assert state.rigid_mass[0] == pytest.approx(2.0)
    assert state.rigid_inv_mass[1] == 0.0  # zero-mass body is static
    assert state.rigid_quat[0][0] == 1.0  # identity quaternion
    # entity -> rigid index mapping is published for sensors/solvers
    assert len(state.entity_to_rigid) == 2
    assert state.t == 0.0


def test_scene_repr() -> None:
    """repr reports entity count, time, dt and solver names."""
    scene = Scene()
    text = repr(scene)
    assert "entities=0" in text and "solvers=[]" in text
