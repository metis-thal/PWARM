"""Tests for GLRenderer mesh generation and shader compilation.

All tests here are pure CPU (mesh geometry, config, frustum math) — none
creates a GL context, so they run on headless CI. The optional viz deps
(glfw/moderngl) are guarded at import: without them the module is skipped
entirely instead of erroring.
"""

import pytest

pytest.importorskip("glfw", reason="viz extra not installed")
pytest.importorskip("moderngl", reason="viz extra not installed")

import numpy as np

from pymo.viz.gl_renderer import (
    FrustumPlanes,
    GLRenderer,
    RendererConfig,
    _make_box_mesh,
    _make_cylinder_mesh,
    _make_sphere_mesh,
)
from pymo.viz.snapshot import CameraState, DoubleBuffer

# ---------------------------------------------------------------------------
# Mesh generation tests (pure CPU, no GPU)
# ---------------------------------------------------------------------------

class TestSphereMesh:
    def test_vertex_count(self):
        verts, norms, _idx = _make_sphere_mesh(1.0, 16, 12)
        assert verts.shape[1] == 3
        assert norms.shape[1] == 3
        assert len(verts) == len(norms)

    def test_radius(self):
        verts, _, _ = _make_sphere_mesh(2.0, 8, 6)
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 2.0, atol=1e-5)

    def test_normals_unit_length(self):
        _, norms, _ = _make_sphere_mesh(1.0, 16, 12)
        lengths = np.linalg.norm(norms, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_index_count(self):
        _verts, _, idx = _make_sphere_mesh(1.0, 16, 12)
        # Triangles: stacks * sectors * 2 * 3
        expected = 12 * 16 * 2 * 3
        assert len(idx) == expected


class TestBoxMesh:
    def test_vertex_count(self):
        verts, norms, _idx = _make_box_mesh(1.0, 1.0, 1.0)
        assert len(verts) == 24  # 6 faces * 4 vertices
        assert len(norms) == 24

    def test_face_count(self):
        _verts, _, idx = _make_box_mesh(1.0, 1.0, 1.0)
        # 6 faces * 2 triangles * 3 indices
        assert len(idx) == 36

    def test_normals_are_axis_aligned(self):
        _, norms, _ = _make_box_mesh(1.0, 1.0, 1.0)
        for n in norms:
            # Each normal should have exactly one non-zero component
            assert np.sum(np.abs(n) > 0.5) == 1


class TestCylinderMesh:
    def test_vertex_count(self):
        verts, _norms, _idx = _make_cylinder_mesh(1.0, 1.0, 16)
        # Side: (segments+1)*2 + 2 cap centers
        expected_side = (16 + 1) * 2
        assert len(verts) == expected_side + 2

    def test_normals_unit_length(self):
        _, norms, _ = _make_cylinder_mesh(1.0, 1.0, 16)
        lengths = np.linalg.norm(norms, axis=1)
        # Cap normals are axis-aligned, side normals are radial
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_cap_indices(self):
        _verts, _, idx = _make_cylinder_mesh(1.0, 1.0, 16)
        # Should have side + cap triangles
        assert len(idx) > 0


# ---------------------------------------------------------------------------
# RendererConfig
# ---------------------------------------------------------------------------

class TestRendererConfig:
    def test_defaults(self):
        cfg = RendererConfig()
        assert cfg.window_size == (1280, 720)
        assert cfg.target_fps == 60
        assert cfg.temp_min == 200.0
        assert cfg.temp_max == 800.0

    def test_custom(self):
        cfg = RendererConfig(window_size=(800, 600), target_fps=30)
        assert cfg.window_size == (800, 600)
        assert cfg.target_fps == 30


# ---------------------------------------------------------------------------
# GLRenderer construction (no GPU needed)
# ---------------------------------------------------------------------------

class TestGLRendererInit:
    def test_creation(self):
        buf = DoubleBuffer()
        renderer = GLRenderer(buf)
        assert renderer.buffer is buf
        assert renderer.running is False
        assert renderer._last_snapshot is None

    def test_with_config(self):
        buf = DoubleBuffer()
        cfg = RendererConfig(window_size=(640, 480))
        renderer = GLRenderer(buf, cfg)
        assert renderer.config.window_size == (640, 480)

    def test_frustum_cull_default_on(self):
        buf = DoubleBuffer()
        renderer = GLRenderer(buf)
        assert renderer.config.frustum_cull is True


# ---------------------------------------------------------------------------
# FrustumPlanes tests (pure CPU math)
# ---------------------------------------------------------------------------

class TestFrustumPlanes:
    def test_identity_matrix_full_frustum(self):
        """Identity VP → aabb at origin should be inside."""
        vp = np.eye(4, dtype=np.float32)
        frustum = FrustumPlanes.from_view_projection(vp)
        inside = frustum.is_aabb_inside(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        assert inside is True

    def test_far_away_aabb_culled(self):
        """An AABB far behind the camera should be culled."""
        cam = CameraState()
        cam.distance = 10.0
        cam.elevation = np.pi / 4
        cam.azimuth = 0.0
        view = cam.view_matrix()
        proj = cam.projection_matrix(1.0)
        vp = proj @ view

        frustum = FrustumPlanes.from_view_projection(vp)

        # Camera is at ~(7,0,7) looking at (0,0,0) with azimuth=0, elev=π/4
        # An AABB far behind the camera (positive x,z, behind it)
        behind = frustum.is_aabb_inside(
            np.array([50, -1, 50], dtype=np.float32),
            np.array([52, 1, 52], dtype=np.float32),
        )
        assert behind is False

    def test_nearby_aabb_visible(self):
        """An AABB in front of the camera should be visible."""
        cam = CameraState()
        cam.distance = 10.0
        cam.elevation = np.pi / 4
        cam.azimuth = 0.0
        view = cam.view_matrix()
        proj = cam.projection_matrix(1.0)
        vp = proj @ view

        frustum = FrustumPlanes.from_view_projection(vp)

        # A box at the origin (in front of the camera looking at 0,0,0)
        visible = frustum.is_aabb_inside(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        assert visible is True

    def test_plane_count(self):
        vp = np.eye(4, dtype=np.float32)
        frustum = FrustumPlanes.from_view_projection(vp)
        assert frustum.planes.shape == (6, 4)
