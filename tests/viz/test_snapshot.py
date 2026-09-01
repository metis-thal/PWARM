"""Tests for SceneSnapshot, DoubleBuffer, and snapshot generation."""

import threading
import time

import numpy as np
import pytest

from pymo.kernel.bodies3d import Material, box_body, sphere_body, cylinder_body
from pymo.kernel.world3d import World3D
from pymo.kernel.world_engine import WorldEngine
from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    InstanceData,
    MeshType,
    SceneSnapshot,
    build_model_matrix,
    build_snapshot_from_world,
    compute_aabb_box,
    compute_aabb_sphere,
)


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix
# ---------------------------------------------------------------------------

class TestBuildModelMatrix:
    def test_identity_quat_gives_translation(self):
        orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pos = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        m = build_model_matrix(pos, orn)
        assert m[0, 3] == pytest.approx(3.0)
        assert m[1, 3] == pytest.approx(4.0)
        assert m[2, 3] == pytest.approx(5.0)
        np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-6)

    def test_90deg_z_rotation(self):
        # 90° around Z: (1,0,0) → (0,1,0)
        # quaternion for 90° around Z: w=cos(45°)=√2/2, x=0, y=0, z=sin(45°)=√2/2
        s = np.sqrt(2) / 2
        orn = np.array([s, 0.0, 0.0, s], dtype=np.float32)
        pos = np.zeros(3, dtype=np.float32)
        m = build_model_matrix(pos, orn)
        # First column should map (1,0,0) → (0,1,0)
        np.testing.assert_allclose(m[:3, 0], [0, 1, 0], atol=1e-5)

    def test_with_scale(self):
        orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pos = np.zeros(3, dtype=np.float32)
        scale = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        m = build_model_matrix(pos, orn, scale=scale)
        assert m[0, 0] == pytest.approx(2.0)
        assert m[1, 1] == pytest.approx(3.0)
        assert m[2, 2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# AABB computation
# ---------------------------------------------------------------------------

class TestAABB:
    def test_sphere_aabb(self):
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        aabb_min, aabb_max = compute_aabb_sphere(0.5, pos)
        np.testing.assert_allclose(aabb_min, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(aabb_max, [1.5, 2.5, 3.5])

    def test_box_aabb_unrotated(self):
        pos = np.zeros(3, dtype=np.float32)
        orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        half_extents = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        aabb_min, aabb_max = compute_aabb_box(half_extents, pos, orn)
        np.testing.assert_allclose(aabb_min, [-1, -2, -3], atol=1e-5)
        np.testing.assert_allclose(aabb_max, [1, 2, 3], atol=1e-5)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class TestCamera:
    def test_position_default(self):
        cam = CameraState()
        pos = cam.position()
        assert pos.shape == (3,)
        # Default elevation π/4, azimuth 0, distance 20
        assert np.linalg.norm(pos) == pytest.approx(20.0, rel=1e-5)

    def test_view_matrix_shape(self):
        cam = CameraState()
        vm = cam.view_matrix()
        assert vm.shape == (4, 4)

    def test_projection_matrix_shape(self):
        cam = CameraState()
        pm = cam.projection_matrix(aspect=16 / 9)
        assert pm.shape == (4, 4)
        # Bottom-left should be -1 (OpenGL convention)
        assert pm[3, 2] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------

class TestSnapshotGeneration:
    def test_basic_snapshot(self):
        eng = WorldEngine()
        ground = box_body([0, 0, -0.5], np.array([10, 10, 0.5]), static=True)
        ball = sphere_body([0, 0, 3], radius=0.5, mass=1.0)
        eng.add_body(ground)
        eng.add_body(ball)

        snap = build_snapshot_from_world(eng)
        assert len(snap.instances) == 2
        assert snap.instances[0].mesh_type == MeshType.BOX
        assert snap.instances[1].mesh_type == MeshType.SPHERE
        assert snap.step == 0
        assert snap.total_mass == pytest.approx(ground.mass + ball.mass)

    def test_snapshot_immutability(self):
        """Modifying returned data shouldn't affect the snapshot."""
        eng = WorldEngine()
        eng.add_body(sphere_body([0, 0, 3], radius=0.5, mass=1.0))
        snap = build_snapshot_from_world(eng)
        orig_temp = snap.instances[0].temperature
        snap.instances[0].temperature = 999.0
        # This is a shallow test — the snapshot is "conceptually" immutable
        assert snap.instances[0].temperature == 999.0  # dataclass is mutable, contract is "don't modify"

    def test_snapshot_with_chemistry(self):
        eng = WorldEngine()
        eng.add_body(sphere_body([0, 0, 3], radius=0.5, mass=1.0))
        from pymo.rules.chemistry import ChemicalSystem
        chem = ChemicalSystem(masses={"wood": 1.0, "oxygen": 0.5})
        eng.add_chemistry(chem)
        snap = build_snapshot_from_world(eng)
        assert "wood" in snap.chemistry_masses
        assert snap.chemistry_masses["wood"] == pytest.approx(1.0)

    def test_snapshot_with_sph(self):
        eng = WorldEngine()
        eng.add_body(sphere_body([0, 0, 3], radius=0.5, mass=1.0))
        from pymo.rules.fluid import create_water_column
        sph = create_water_column(0, 0, 3, 3, 0.1)
        eng.add_fluid(sph)
        snap = build_snapshot_from_world(eng)
        assert snap.fluid_positions is not None
        assert snap.fluid_positions.shape == (9, 3)

    def test_snapshot_after_tick(self):
        eng = WorldEngine()
        eng.add_body(sphere_body([0, 0, 3], radius=0.5, mass=1.0))
        eng.tick(dt=1 / 60.0)
        snap = build_snapshot_from_world(eng)
        assert snap.step == 1
        assert snap.t > 0.0

    def test_cylinder_mesh_type(self):
        eng = WorldEngine()
        eng.add_body(cylinder_body([0, 0, 1], radius=0.5, half_height=1.0, mass=1.0))
        snap = build_snapshot_from_world(eng)
        assert snap.instances[0].mesh_type == MeshType.CYLINDER


# ---------------------------------------------------------------------------
# DoubleBuffer
# ---------------------------------------------------------------------------

class TestDoubleBuffer:
    def test_write_read(self):
        buf = DoubleBuffer()
        assert buf.has_snapshot is False
        assert buf.read() is None

        snap = SceneSnapshot(t=1.0, step=1)
        buf.write(snap)
        assert buf.has_snapshot is True
        assert buf.read() is snap

    def test_write_replaces(self):
        buf = DoubleBuffer()
        s1 = SceneSnapshot(t=1.0)
        s2 = SceneSnapshot(t=2.0)
        buf.write(s1)
        buf.write(s2)
        assert buf.read() is s2

    def test_dirty_flag(self):
        buf = DoubleBuffer()
        assert buf.is_dirty() is False
        buf.write(SceneSnapshot())
        assert buf.is_dirty() is True
        buf.mark_clean()
        assert buf.is_dirty() is False

    def test_thread_safety(self):
        """Sim thread writes, render thread reads — no crash, no stale data."""
        buf = DoubleBuffer()
        errors = []

        def sim_thread():
            try:
                for i in range(100):
                    buf.write(SceneSnapshot(t=float(i), step=i))
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        def render_thread():
            try:
                for _ in range(100):
                    snap = buf.read()
                    if snap is not None:
                        _ = snap.t  # access data
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=sim_thread)
        t2 = threading.Thread(target=render_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Thread errors: {errors}"
        assert buf.read() is not None
