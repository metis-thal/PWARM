"""Windowless tests for the GL renderer state machines (viz.gl_renderer).

The existing tests/viz/test_gl_renderer.py covers the CPU mesh generators and
frustum culling. This module covers the renderer's input handling, camera
manipulation, and windowless fallbacks — no GLFW window or OpenGL context is
created (glfw constants are plain integers).
"""

from __future__ import annotations

import numpy as np
import pytest

glfw = pytest.importorskip("glfw")

from pwarm.viz.gl_renderer import GLRenderer, RendererConfig
from pwarm.viz.snapshot import CameraState, DoubleBuffer, SceneSnapshot


@pytest.fixture
def renderer() -> GLRenderer:
    return GLRenderer(DoubleBuffer(), RendererConfig(window_size=(320, 240)))


def _snapshot_with_camera(**kwargs) -> SceneSnapshot:
    return SceneSnapshot(camera=CameraState(**kwargs))


# ------------------------------------------------------------------ properties
def test_windowless_properties(renderer: GLRenderer) -> None:
    """Before init(): no context, no framebuffer, default camera, not running."""
    assert renderer.ctx is None
    assert renderer.framebuffer_size() == (0, 0)
    assert renderer.running is False
    assert isinstance(renderer.camera, CameraState)


def test_camera_property_prefers_last_snapshot(renderer: GLRenderer) -> None:
    """Once a snapshot arrived, the camera property exposes its camera."""
    renderer._last_snapshot = _snapshot_with_camera(distance=42.0)
    assert renderer.camera.distance == 42.0
    assert renderer.camera is renderer._last_snapshot.camera


def test_render_frame_without_window_is_noop(renderer: GLRenderer) -> None:
    """render_frame/set_title/poll-safe calls do nothing without a window."""
    renderer.buffer.write(_snapshot_with_camera())
    renderer.render_frame()  # must not raise
    renderer.set_title("ignored")


def test_poll_key_without_window_returns_sentinel(
        renderer: GLRenderer, monkeypatch: pytest.MonkeyPatch) -> None:
    """poll_key pumps events, then reports -1 when no window exists."""
    pumped: list[bool] = []
    monkeypatch.setattr(glfw, "poll_events", lambda: pumped.append(True))
    assert renderer.poll_key() == -1
    assert pumped == [True]


def test_poll_key_edge_triggered(renderer: GLRenderer,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored key press is consumed exactly once."""
    monkeypatch.setattr(glfw, "poll_events", lambda: None)
    renderer._window = object()
    renderer._last_key = glfw.KEY_R
    assert renderer.poll_key() == glfw.KEY_R
    assert renderer.poll_key() == -1


# ------------------------------------------------------------------ mouse input
def test_mouse_button_tracks_left_and_right(renderer: GLRenderer) -> None:
    """Left/right press sets the drag flags; release clears; others ignored."""
    renderer._on_mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    assert renderer._mouse_left is True and renderer._mouse_right is False
    renderer._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
    assert renderer._mouse_right is True
    renderer._on_mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert renderer._mouse_left is False
    renderer._on_mouse_button(None, glfw.MOUSE_BUTTON_MIDDLE, glfw.PRESS, 0)
    assert renderer._mouse_left is False and renderer._mouse_right is True


def test_cursor_pos_without_snapshot_is_noop(renderer: GLRenderer) -> None:
    """Mouse motion before the first snapshot is ignored entirely."""
    renderer._on_cursor_pos(None, 100, 100)
    assert renderer._mouse_pos == (0.0, 0.0)  # early return before tracking
    assert renderer.camera.azimuth == 0.0


def test_cursor_pos_left_drag_orbits(renderer: GLRenderer) -> None:
    """Left-drag orbits azimuth and clamps elevation away from the poles."""
    renderer._last_snapshot = _snapshot_with_camera(
        elevation=np.pi / 2, azimuth=0.0)
    renderer._mouse_left = True
    renderer._mouse_pos = (0.0, 0.0)
    renderer._on_cursor_pos(None, 100, 0)  # dx=100, dy=0
    cam = renderer.camera
    assert cam.azimuth == pytest.approx(-0.5)  # dx * 0.005
    assert cam.elevation == pytest.approx(np.pi / 2)  # dy=0 keeps elevation
    # dragging far past a pole clamps instead of flipping
    renderer._on_cursor_pos(None, 0, -100_000)  # dy < 0 pulls elevation down
    assert cam.elevation == pytest.approx(0.05)
    renderer._on_cursor_pos(None, 0, 200_000)
    assert cam.elevation == pytest.approx(np.pi - 0.05)


def test_cursor_pos_right_drag_pans(renderer: GLRenderer) -> None:
    """Right-drag pans the camera target within the view plane."""
    renderer._last_snapshot = _snapshot_with_camera(
        elevation=np.pi / 2, azimuth=0.0, distance=10.0)
    renderer._mouse_right = True
    renderer._mouse_pos = (0.0, 0.0)
    before = renderer.camera.target.copy()
    renderer._on_cursor_pos(None, 100, 0)
    after = renderer.camera.target
    assert not np.allclose(after, before)
    # with the camera on +x looking at the origin, right-dragging slides
    # the target along the camera's right vector (-y), so target.y increases
    assert after[0] == pytest.approx(before[0])
    assert after[1] > before[1]


def test_scroll_zooms_and_clamps(renderer: GLRenderer) -> None:
    """Scroll scales the distance and clamps it to [1, 200]."""
    renderer._last_snapshot = _snapshot_with_camera(distance=10.0)
    renderer._on_scroll(None, 0.0, 1.0)
    assert renderer.camera.distance == pytest.approx(9.0)  # scroll in
    renderer._on_scroll(None, 0.0, -1.0)
    assert renderer.camera.distance == pytest.approx(9.9)  # scroll out
    renderer.camera.distance = 190.0
    renderer._on_scroll(None, 0.0, -1.0)
    assert renderer.camera.distance == pytest.approx(200.0)  # clamped high
    renderer.camera.distance = 1.05
    renderer._on_scroll(None, 0.0, 1.0)
    assert renderer.camera.distance == pytest.approx(1.0)  # clamped low
    renderer._last_snapshot = None
    renderer._on_scroll(None, 0.0, 5.0)  # no snapshot: no-op


# --------------------------------------------------------------------- keyboard
def test_key_escape_stops_renderer(renderer: GLRenderer) -> None:
    """ESC marks the renderer not running."""
    renderer._on_key(None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
    assert renderer.running is False


def test_key_r_resets_camera(renderer: GLRenderer) -> None:
    """R restores the configured distance/elevation and zeroes the target."""
    renderer._last_snapshot = _snapshot_with_camera(
        distance=99.0, elevation=1.4, azimuth=2.0,
        target=np.array([5.0, 5.0, 5.0], dtype=np.float32))
    renderer._on_key(None, glfw.KEY_R, 0, glfw.PRESS, 0)
    cam = renderer.camera
    assert cam.distance == renderer.config.cam_distance
    assert cam.elevation == renderer.config.cam_elevation
    assert cam.azimuth == 0.0
    assert np.allclose(cam.target, 0.0)


def test_key_r_without_snapshot_keeps_defaults(renderer: GLRenderer) -> None:
    """R before any snapshot must not touch a camera (there is none yet)."""
    renderer._on_key(None, glfw.KEY_R, 0, glfw.PRESS, 0)  # must not raise


def test_key_release_and_unknown_keys(renderer: GLRenderer) -> None:
    """Only presses are processed; other keys are recorded for poll_key."""
    renderer._on_key(None, glfw.KEY_A, 0, glfw.RELEASE, 0)
    assert getattr(renderer, "_last_key", -1) == -1  # releases are ignored
    renderer._on_key(None, glfw.KEY_A, 0, glfw.PRESS, 0)
    assert renderer._last_key == glfw.KEY_A
    renderer.running = True  # running starts False until init()
    renderer._on_key(None, glfw.KEY_B, 0, glfw.PRESS, 0)
    assert renderer.running is True  # non-ESC keys never stop the loop


def test_resize_without_context_is_noop(renderer: GLRenderer) -> None:
    """Resize callbacks are safe before a context exists."""
    renderer._on_resize(None, 100, 100)  # must not raise


def test_close_without_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """close() clears the running flag; GLFW teardown is stubbed out."""
    monkeypatch.setattr(glfw, "terminate", lambda: None)
    renderer = GLRenderer(DoubleBuffer(), RendererConfig())
    renderer.running = True
    renderer.close()
    assert renderer.running is False
