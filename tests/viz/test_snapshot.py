"""Tests for the render snapshot layer (viz.snapshot).

Scope: camera state math (position/view/projection), model-matrix and AABB
builders, the physics-engine snapshot builder against a duck-typed fake
engine, the legacy world builder (driven through the pwarm.kernel stubs from
tests/conftest.py), and the DoubleBuffer publish/read contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pwarm.kernel.bodies3d import BoxShape, CylinderShape, SphereShape

from pwarm.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    InstanceData,
    MeshType,
    SceneSnapshot,
    _quat_to_matrix,
    build_model_matrix,
    build_snapshot_from_physics_engine,
    build_snapshot_from_world,
    compute_aabb_box,
    compute_aabb_sphere,
)


# ----------------------------------------------------------------- camera math
def test_camera_position_spherical_offset() -> None:
    """position() offsets the target by the spherical (elevation, azimuth).

    Pinned as implemented: elevation is the polar angle measured from +z
    (elevation 0 looks straight down, pi/2 lies in the horizontal plane).
    """
    cam = CameraState(target=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                      distance=10.0, elevation=np.pi / 2, azimuth=0.0)
    pos = cam.position()
    assert np.allclose(pos, [11.0, 2.0, 3.0])  # horizontal +x at pi/2
    cam_top = CameraState(distance=5.0, elevation=0.0, azimuth=0.0)
    assert np.allclose(cam_top.position(), [0.0, 0.0, 5.0], atol=1e-6)  # +z


def test_view_matrix_is_orthonormal_and_maps_eye() -> None:
    """The view matrix rotates into camera space and translates eye to origin."""
    cam = CameraState(distance=10.0, elevation=np.pi / 4, azimuth=0.5)
    view = cam.view_matrix()
    rot = view[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)
    eye = cam.position()
    hom = np.append(eye, 1.0)
    assert np.allclose(view @ hom, [0.0, 0.0, 0.0, 1.0], atol=1e-4)


def test_projection_matrix_entries() -> None:
    """Standard OpenGL perspective entries follow the fov/aspect/near/far."""
    cam = CameraState(fov=90.0, near=0.5, far=100.0)
    proj = cam.projection_matrix(aspect=2.0)
    f = 1.0 / np.tan(np.radians(45.0))
    assert proj[0, 0] == pytest.approx(f / 2.0)
    assert proj[1, 1] == pytest.approx(f)
    assert proj[2, 2] == pytest.approx((100.0 + 0.5) / (0.5 - 100.0))
    assert proj[2, 3] == pytest.approx(2 * 100.0 * 0.5 / (0.5 - 100.0))
    assert proj[3, 2] == pytest.approx(-1.0)


# ------------------------------------------------------- model matrices / AABBs
def test_quat_to_matrix_identity_and_zero_norm_fallback() -> None:
    """Identity quaternion gives I; a zero-norm quaternion falls back to I."""
    identity = _quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(identity, np.eye(3), atol=1e-6)
    degenerate = _quat_to_matrix(np.zeros(4))
    assert np.allclose(degenerate, np.eye(3), atol=1e-6)


def test_quat_to_matrix_z_rotation() -> None:
    """A 90-degree rotation about z maps +x to +y."""
    half_sqrt2 = np.sqrt(2.0) / 2.0
    rot = _quat_to_matrix(np.array([half_sqrt2, 0.0, 0.0, half_sqrt2]))
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0],
                       atol=1e-6)


def test_build_model_matrix_translation_and_scale() -> None:
    """The model matrix carries rotation, translation and optional scale."""
    pos = np.array([1.0, 2.0, 3.0])
    orn = np.array([1.0, 0.0, 0.0, 0.0])
    model = build_model_matrix(pos, orn)
    assert np.allclose(model[:3, 3], pos)
    assert np.allclose(model[:3, :3], np.eye(3), atol=1e-6)
    scaled = build_model_matrix(pos, orn, scale=np.array([2.0, 3.0, 4.0]))
    assert np.allclose(np.diag(scaled)[:3], [2.0, 3.0, 4.0])


def test_compute_aabb_sphere() -> None:
    """A sphere's AABB is pos +- radius."""
    lo, hi = compute_aabb_sphere(2.0, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(lo, [-1.0, -1.0, -1.0])
    assert np.allclose(hi, [3.0, 3.0, 3.0])


def test_compute_aabb_box_rotation_swaps_extents() -> None:
    """A 90-degree z rotation swaps the x/y extents of the box AABB."""
    pos = np.array([5.0, 5.0, 0.0])
    half = np.array([1.0, 2.0, 0.5])
    lo, hi = compute_aabb_box(half, pos, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(hi - lo, [2.0, 4.0, 1.0])
    quarter = np.sqrt(2.0) / 2.0
    lo_r, hi_r = compute_aabb_box(half, pos,
                                  np.array([quarter, 0.0, 0.0, quarter]))
    assert np.allclose(hi_r - lo_r, [4.0, 2.0, 1.0])


# ------------------------------------------------------- engine snapshot build
class FakeGlobalQuantities:
    def __init__(self) -> None:
        self.total_kinetic_energy = 12.0
        self.total_thermal_energy = 3.0
        self.total_energy = 15.0
        self.total_mass = 2.0


def make_fake_engine(with_read: bool = True) -> SimpleNamespace:
    state = SimpleNamespace(
        t=1.5,
        rigid_pos=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        rigid_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        rigid_linvel=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        sph_pos=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        global_quantities=FakeGlobalQuantities(),
    )
    scene = SimpleNamespace(
        double_buffer_read=state if with_read else None,
        double_buffer_write=state,
    )
    return SimpleNamespace(scene=scene, frame=7,
                           _shape_info={
                               0: {"shape": "sphere", "params": {"radius": 0.5}},
                               1: {"shape": "box",
                                   "params": {"half_extents": [1.0, 1.0, 1.0]}},
                           })


def test_build_snapshot_from_physics_engine() -> None:
    """Rigid bodies map to instances; speeds drive colour; fluid is copied."""
    snapshot = build_snapshot_from_physics_engine(make_fake_engine())
    assert snapshot.step == 7
    assert len(snapshot.instances) == 2
    sphere, box = snapshot.instances
    assert sphere.mesh_type is MeshType.SPHERE
    assert box.mesh_type is MeshType.BOX
    assert np.allclose(sphere.model_matrix[:3, 3], 0.0)
    # stationary body keeps the base colour; the fast body reddens
    assert np.allclose(sphere.base_color, [0.3, 0.4, 0.8], atol=1e-6)
    assert box.base_color[0] > 0.3
    assert snapshot.fluid_positions is not None
    assert len(snapshot.fluid_positions) == 2
    assert np.allclose(snapshot.fluid_color, [0.2, 0.5, 1.0])
    assert snapshot.kinetic_energy == 12.0
    assert snapshot.total_mass == 2.0
    assert snapshot.camera is not None


def test_build_snapshot_falls_back_to_write_buffer() -> None:
    """A None read buffer falls back to the write buffer."""
    snapshot = build_snapshot_from_physics_engine(make_fake_engine(with_read=False))
    assert len(snapshot.instances) == 2


def test_build_snapshot_capsule_maps_to_cylinder() -> None:
    """Capsule shapes render as cylinders with (r, r, hh) scale."""
    engine = make_fake_engine()
    engine._shape_info = {0: {"shape": "capsule",
                              "params": {"radius": 0.4, "half_height": 1.2}}}
    engine.scene.double_buffer_read.rigid_pos = np.zeros((1, 3))
    engine.scene.double_buffer_read.rigid_quat = np.array([[1.0, 0, 0, 0]])
    engine.scene.double_buffer_read.rigid_linvel = np.zeros((1, 3))
    snapshot = build_snapshot_from_physics_engine(engine)
    assert snapshot.instances[0].mesh_type is MeshType.CYLINDER
    assert np.allclose(np.diag(snapshot.instances[0].model_matrix)[:3],
                       [0.4, 0.4, 1.2])


def test_build_snapshot_without_rigid_bodies() -> None:
    """An empty state yields no instances but keeps the fluid/energy fields."""
    engine = make_fake_engine()
    state = engine.scene.double_buffer_read
    state.rigid_pos = None
    state.rigid_quat = None
    state.rigid_linvel = None
    snapshot = build_snapshot_from_physics_engine(engine)
    assert snapshot.instances == []
    assert snapshot.fluid_positions is not None


def test_build_snapshot_without_fluid() -> None:
    """A state without SPH particles leaves the fluid fields empty."""
    engine = make_fake_engine()
    engine.scene.double_buffer_read.sph_pos = None
    snapshot = build_snapshot_from_physics_engine(engine)
    assert snapshot.fluid_positions is None
    assert snapshot.fluid_color is None


# -------------------------------------------------------- legacy world builder
def test_build_snapshot_from_world_with_stubs() -> None:
    """The legacy builder maps stub kernel shapes through the same pipeline."""
    from pwarm.kernel.bodies3d import Body as Body3D

    bodies = [
        Body3D(pos=np.array([0.0, 0.0, 0.5]), mass=2.0, temperature=300.0,
               shape=SphereShape(0.5)),
        Body3D(pos=np.array([1.0, 0.0, 0.25]), mass=1.0, temperature=400.0,
               static=True, shape=BoxShape(np.array([1.0, 1.0, 0.25]))),
        Body3D(pos=np.array([2.0, 0.0, 0.5]), mass=1.0, temperature=350.0,
               shape=CylinderShape(0.3, 0.5)),
    ]
    world = SimpleNamespace(
        bodies=bodies,
        t=2.0,
        step_count=9,
        total_kinetic_energy=lambda: 4.0,
    )
    engine = SimpleNamespace(
        world=world,
        sph=SimpleNamespace(particles=[
            SimpleNamespace(pos=np.array([0.0, 1.0])),
            SimpleNamespace(pos=np.array([1.0, 2.0])),
        ]),
        chemistry=SimpleNamespace(masses={"water": 1.5}),
        enable_geology=False,
    )
    snapshot = build_snapshot_from_world(engine)
    assert snapshot.t == 2.0 and snapshot.step == 9
    assert [i.mesh_type for i in snapshot.instances] == [
        MeshType.SPHERE, MeshType.BOX, MeshType.CYLINDER]
    assert snapshot.instances[0].temperature == 300.0
    assert np.allclose(snapshot.instances[1].base_color, [0.6, 0.7, 0.8])
    # 2-D fluid positions are padded to 3-D
    assert snapshot.fluid_positions.shape == (2, 3)
    assert np.allclose(snapshot.fluid_positions[:, 2], 0.0)
    assert snapshot.chemistry_masses == {"water": 1.5}
    assert snapshot.total_mass == pytest.approx(4.0)
    assert snapshot.kinetic_energy == 4.0
    # thermal energy: sum(m * c * T) with c = 1000
    assert snapshot.thermal_energy == pytest.approx(
        2.0 * 1000.0 * 300.0 + 1.0 * 1000.0 * 400.0 + 1.0 * 1000.0 * 350.0)


def test_build_snapshot_from_world_minimal_engine() -> None:
    """Missing optional engine features fall back to documented defaults."""
    world = SimpleNamespace(bodies=[], t=0.0, step_count=0)
    engine = SimpleNamespace(world=world)  # no sph/chemistry/geology
    snapshot = build_snapshot_from_world(engine)
    assert snapshot.instances == []
    assert snapshot.chemistry_masses == {}
    assert snapshot.fluid_positions is None
    assert snapshot.total_mass == 0.0
    assert snapshot.environment_temp == 293.15
    assert snapshot.solar_flux == 0.0


# ------------------------------------------------------------------ DoubleBuffer
def test_double_buffer_lifecycle() -> None:
    """read-before-write is None; writes publish, flag and replace."""
    buffer = DoubleBuffer()
    assert buffer.read() is None
    assert buffer.has_snapshot is False
    assert buffer.is_dirty() is False
    first = SceneSnapshot(t=1.0)
    buffer.write(first)
    assert buffer.read() is first
    assert buffer.has_snapshot is True
    assert buffer.is_dirty() is True
    buffer.mark_clean()
    assert buffer.is_dirty() is False
    second = SceneSnapshot(t=2.0)
    buffer.write(second)
    assert buffer.read() is second  # last write wins
    assert buffer.is_dirty() is True


def test_instance_data_defaults() -> None:
    """InstanceData carries documented defaults."""
    instance = InstanceData(model_matrix=np.eye(4), mesh_type=MeshType.SPHERE)
    assert instance.temperature == 293.15
    assert np.allclose(instance.base_color, [0.6, 0.7, 0.8])
    assert MeshType.SPHERE == "sphere"  # str enum for shader lookups
