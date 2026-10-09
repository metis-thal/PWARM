"""Tests for the GUI layer (interface.gui).

Scope: the headless-safe surface only — camera configuration, the GUI
state machine (pause/step/select/camera), and the orbit-camera controller
math. ``GUI.run()`` is an infinite loop by design and is never invoked.
Scene interactions are exercised with a mock scene.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pwarm.physics  # noqa: F401  (breaks the interface<->physics cycle)
from pwarm.interface.gui import GUI, CameraConfig, CameraController


@pytest.fixture
def mock_scene() -> SimpleNamespace:
    return SimpleNamespace(step=lambda: None, get_render_snapshot=lambda: None)


def test_camera_config_defaults() -> None:
    """CameraConfig carries the documented defaults."""
    cfg = CameraConfig()
    assert cfg.position == (5.0, 5.0, 5.0)
    assert cfg.fov == 60.0
    assert cfg.near == 0.1
    assert cfg.far == 1000.0
    assert (cfg.width, cfg.height) == (1280, 720)


def test_gui_defaults(mock_scene: SimpleNamespace) -> None:
    """A GUI starts paused=False, running=False with a default camera."""
    gui = GUI(mock_scene)
    assert gui.camera.fov == 60.0
    assert gui.running is False
    assert gui.paused is False
    assert gui.show_ui is True
    assert gui.selected_entity is None


def test_gui_toggle_pause_and_step_once(mock_scene: SimpleNamespace) -> None:
    """step_once advances the scene only while paused."""
    steps: list[int] = []
    scene = SimpleNamespace(step=lambda: steps.append(1),
                            get_render_snapshot=lambda: None)
    gui = GUI(scene)
    gui.step_once()
    assert steps == []  # not paused -> no-op
    gui.toggle_pause()
    assert gui.paused is True
    gui.step_once()
    assert len(steps) == 1


def test_gui_set_camera_partial_update(mock_scene: SimpleNamespace) -> None:
    """set_camera updates only the truthy arguments."""
    gui = GUI(mock_scene)
    gui.set_camera(position=(1.0, 2.0, 3.0))
    assert gui.camera.position == (1.0, 2.0, 3.0)
    assert gui.camera.target == (0.0, 0.0, 0.0)  # untouched
    gui.set_camera(target=(0.0, 0.0, -1.0), fov=45.0)
    assert gui.camera.target == (0.0, 0.0, -1.0)
    assert gui.camera.fov == 45.0


def test_gui_select_entity(mock_scene: SimpleNamespace) -> None:
    """select_entity stores the id verbatim."""
    gui = GUI(mock_scene)
    gui.select_entity(42)
    assert gui.selected_entity == 42


def test_gui_render_without_snapshot_is_noop(mock_scene: SimpleNamespace) -> None:
    """_render returns early when the scene has no snapshot."""
    gui = GUI(mock_scene)
    gui._render()  # must not raise


def test_gui_render_with_snapshot_calls_ui(mock_scene: SimpleNamespace) -> None:
    """_render pulls a snapshot and invokes the (stub) UI pass."""
    calls: list[str] = []
    scene = SimpleNamespace(
        step=lambda: None,
        get_render_snapshot=lambda: object(),
    )
    gui = GUI(scene)
    gui._render_ui = lambda: calls.append("ui")  # type: ignore[method-assign]
    gui._render()
    assert calls == ["ui"]


def test_gui_screenshot_is_noop(mock_scene: SimpleNamespace) -> None:
    """screenshot is a documented no-op that writes nothing."""
    gui = GUI(mock_scene)
    gui.screenshot("unused.png")  # must not raise or create files


def test_camera_controller_defaults() -> None:
    """The orbit controller starts at the documented spherical pose."""
    cam = CameraConfig()
    ctrl = CameraController(cam)
    assert (ctrl.distance, ctrl.yaw, ctrl.pitch) == (10.0, 45.0, -30.0)


def test_camera_controller_orbit_left_drag() -> None:
    """Left-drag orbits yaw/pitch; pitch clamps to ±89 degrees."""
    cam = CameraConfig()
    ctrl = CameraController(cam)
    ctrl.update(dx=20.0, dy=10.0, dz=0.0, buttons={"left": True})
    assert ctrl.yaw == 55.0  # +dx * 0.5
    assert ctrl.pitch == -35.0  # -dy * 0.5
    ctrl.update(dx=0.0, dy=-500.0, dz=0.0, buttons={"left": True})
    assert ctrl.pitch == 89.0  # clamped


def test_camera_controller_zoom() -> None:
    """Scroll (dz) zooms with a 0.1 minimum distance, no buttons needed."""
    cam = CameraConfig()
    ctrl = CameraController(cam)
    ctrl.update(dx=0.0, dy=0.0, dz=4.0, buttons={})
    assert ctrl.distance == 8.0  # distance - dz*0.5
    ctrl.update(dx=0.0, dy=0.0, dz=100.0, buttons={})
    assert ctrl.distance == 0.1  # clamped


def test_camera_controller_right_drag_pans_nothing() -> None:
    """Right-drag pan is a documented no-op; pose math still updates."""
    cam = CameraConfig(target=(0.0, 0.0, 0.0))
    ctrl = CameraController(cam)
    ctrl.update(dx=5.0, dy=5.0, dz=0.0, buttons={"right": True})
    assert cam.target == (0.0, 0.0, 0.0)
    # position was still recomputed from the unchanged spherical pose
    assert cam.position[2] == pytest.approx(
        10.0 * __import__("math").sin(__import__("math").radians(-30.0)))


def test_camera_controller_position_math() -> None:
    """Spherical pose converts to the documented Cartesian camera position."""

    cam = CameraConfig(target=(1.0, 2.0, 3.0))
    ctrl = CameraController(cam)
    ctrl.yaw = 0.0
    ctrl.pitch = 0.0
    ctrl.distance = 10.0
    ctrl._update_camera_position()
    assert cam.position == pytest.approx((11.0, 2.0, 3.0))
