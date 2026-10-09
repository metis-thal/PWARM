"""Tests for the pyvista viewers (viz.viewer, viz.viewer3d).

These modules still import the removed ``pwarm.kernel`` package; tests/conftest.py
stubs it, and the worlds themselves are duck-typed fakes. Rendering uses VTK
off-screen mode. Scope: actor construction per body kind, HUD text, the
timer/pause/speed/reset controls, frame rendering, and PNG recording.

Environment preconditions (skips are narrow, never exception-wide):
- pyvista must be importable (it ships with the ``viz`` extra);
- on Linux, VTK off-screen rendering needs X/Wayland (or an OSMesa VTK
  build) — without a display the module skips. Real Linux-CI behaviour was
  NOT verified as of 2026-10-09 (no Linux machine available); run under
  xvfb to exercise these tests there.

Every viewer fixture is function-scoped: each test builds its own world and
plotter, so no test depends on execution order or leftover state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from pwarm.kernel.bodies import Body

from pwarm.viz.viewer import PhysicsViewer, ViewerConfig
from pwarm.viz.viewer3d import PhysicsViewer3D, ViewerConfig3D

pv = pytest.importorskip("pyvista", reason="viz extra not installed")

if sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    pytest.skip(
        "VTK off-screen rendering on Linux requires X/Wayland (or an OSMesa "
        "VTK build); no display detected — run under xvfb to exercise these "
        "tests",
        allow_module_level=True,
    )


class FakeWorld:
    """Duck-typed 2D world: exactly the surface the viewer reads."""

    def __init__(self, bodies: list[Body]) -> None:
        self.bodies = bodies
        self.t = 1.5
        self.step_count = 30
        self.steps_requested: list[int] = []

    def step(self, n: int = 1) -> None:
        self.steps_requested.append(n)
        self.step_count += n
        self.t += 1 / 60

    def total_kinetic_energy(self) -> float:
        return 42.0

    def total_momentum(self) -> np.ndarray:
        return np.array([1.0, -2.0])


def circle_body(x: float = 0.0) -> Body:
    return Body(pos=np.array([x, 0.0]), vel=np.array([1.0, 0.0]), radius=0.5,
                mass=1.0)


def polygon_body() -> Body:
    square = [np.array([-1.0, -1.0]), np.array([1.0, -1.0]),
              np.array([1.0, 1.0]), np.array([-1.0, 1.0])]
    return Body(pos=np.array([0.0, 3.0]), vertices=square, mass=4.0,
                static=True)


@pytest.fixture
def small_config() -> ViewerConfig:
    return ViewerConfig(window_size=(160, 120))


@pytest.fixture
def viewer(small_config: ViewerConfig) -> PhysicsViewer:
    world = FakeWorld([circle_body(), polygon_body(), Body(pos=np.zeros(2))])
    return PhysicsViewer(world, config=small_config, off_screen=True)


def test_actor_for_circle_and_polygon(viewer: PhysicsViewer) -> None:
    """Circles become spheres, polygons extruded meshes, others are skipped."""
    circle = circle_body()
    poly = polygon_body()
    plain = Body(pos=np.zeros(2))  # neither circle nor polygon
    assert viewer._actor_for(circle) is not None
    assert viewer._actor_for(poly) is not None
    assert viewer._actor_for(plain) is None


def test_build_actors_skips_unrenderable(viewer: PhysicsViewer) -> None:
    """The actor map holds entries only for renderable bodies."""
    world = FakeWorld([circle_body(), Body(pos=np.zeros(2))])
    viewer2 = PhysicsViewer(world, config=ViewerConfig(window_size=(160, 120)),
                            off_screen=True)
    # unrenderable bodies get a None placeholder entry
    assert sum(1 for a in viewer2._actors.values() if a is not None) == 1


def test_panel_text_reports_world_state(viewer: PhysicsViewer) -> None:
    """The HUD carries time, step count, body count, energy and pause state."""
    text = viewer._panel_text()
    assert "1.50" in text  # world.t
    assert "30" in text  # step_count
    assert "3" in text  # body count
    assert "42.00" in text  # kinetic energy
    assert "RUNNING" in text


def _bare_actor():
    """A real vtk actor from a throwaway off-screen plotter."""
    plotter = pv.Plotter(off_screen=True, window_size=(80, 60))
    actor = plotter.add_mesh(pv.Sphere(radius=0.5))
    plotter.close()
    return actor


def test_place_actor_positions_and_rotates() -> None:
    """Actors are placed at the body position with its z rotation."""
    actor = _bare_actor()
    body = Body(pos=np.array([2.0, 3.0]), angle=0.5)
    PhysicsViewer._place_actor(actor, body)
    assert actor.position == pytest.approx((2.0, 3.0, 0.0))
    assert actor.orientation == pytest.approx((0.0, 0.0, np.degrees(0.5)))


def test_pause_and_speed_controls(viewer: PhysicsViewer) -> None:
    """Pause toggles; speed scales stay within the configured bounds."""
    assert viewer.paused is False
    viewer._toggle_pause()
    assert viewer.paused is True
    viewer._toggle_pause()
    viewer.speed = 2.0
    viewer._speed_up()
    assert viewer.speed == pytest.approx(2.5)
    viewer.speed = 3.5
    viewer._speed_up()
    assert viewer.speed == pytest.approx(viewer.config.max_speed)  # capped 4.0
    viewer._speed_down()
    assert viewer.speed == pytest.approx(3.2)
    for _ in range(20):
        viewer._speed_down()
    assert viewer.speed == pytest.approx(viewer.config.min_speed)  # floor 0.25


def test_reset_zeroes_counters_only(viewer: PhysicsViewer) -> None:
    """_reset clears world time/steps; body poses are intentionally kept."""
    viewer.world.t = 9.0
    viewer.world.step_count = 77
    viewer._reset()
    assert viewer.world.t == 0.0
    assert viewer.world.step_count == 0
    assert viewer.world.bodies[0].pos[0] == pytest.approx(0.0)  # unchanged


def test_timer_steps_only_when_unpaused(viewer: PhysicsViewer) -> None:
    """_on_timer advances the world unless paused."""
    before = viewer.world.step_count
    viewer._on_timer()
    assert viewer.world.step_count > before
    viewer.paused = True
    fixed = viewer.world.step_count
    viewer._on_timer()
    assert viewer.world.step_count == fixed


def test_render_frame_returns_image(viewer: PhysicsViewer) -> None:
    """Off-screen rendering returns an (H, W, 3) uint8 image."""
    frame = viewer.render_frame()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_record_writes_pngs(viewer: PhysicsViewer, tmp_path: Path) -> None:
    """record() steps, renders and saves frame_%04d.png files."""
    paths = viewer.record(n_frames=2, out_dir=str(tmp_path / "frames"),
                          steps_per_frame=3)
    assert len(paths) == 2
    for i, path in enumerate(paths):
        assert Path(path).name == f"frame_{i:04d}.png"
        assert Path(path).stat().st_size > 0
    # exactly two frames x 3 steps each on this test's own fresh world
    assert viewer.world.steps_requested == [3, 3]


# --------------------------------------------------------------------------- 3D
class FakeShape:
    def __init__(self, kind: str, **kwargs) -> None:
        self.kind = kind
        self.__dict__.update(kwargs)


class FakeBody3D:
    def __init__(self, pos, shape, orn=(1.0, 0.0, 0.0, 0.0),
                 static: bool = False) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(3)
        self.orn = np.asarray(orn, dtype=float)
        self.ang_vel = np.zeros(3)
        self.shape = shape
        self.static = static
        self.temperature = 293.15
        self.mass = 1.0

    def is_sphere(self) -> bool:
        return self.shape.kind == "sphere"

    def is_box(self) -> bool:
        return self.shape.kind == "box"

    def is_cylinder(self) -> bool:
        return self.shape.kind == "cylinder"


class FakeWorld3D(FakeWorld):
    def total_momentum(self) -> np.ndarray:
        return np.array([1.0, -2.0, 0.5])

    def total_angular_momentum(self) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def small_config_3d() -> ViewerConfig3D:
    return ViewerConfig3D(window_size=(160, 120))


@pytest.fixture
def viewer3d(small_config_3d: ViewerConfig3D) -> PhysicsViewer3D:
    world = FakeWorld3D([
        FakeBody3D([0.0, 0.0, 1.0], FakeShape("sphere", radius=0.5)),
        FakeBody3D([2.0, 0.0, 0.25], FakeShape("box",
                                               half_extents=np.array([1.0, 1.0, 0.25])),
                   static=True),
        FakeBody3D([-2.0, 0.0, 0.5], FakeShape("cylinder", radius=0.3,
                                               half_height=0.5)),
        FakeBody3D([9.0, 9.0, 9.0], FakeShape("cone")),  # unsupported kind
    ])
    return PhysicsViewer3D(world, config=small_config_3d, off_screen=True)


def test_actor_3d_per_shape(viewer3d: PhysicsViewer3D) -> None:
    """Sphere/box/cylinder bodies get actors; other shapes are skipped."""
    sphere = FakeBody3D([0.0, 0, 0], FakeShape("sphere", radius=0.5))
    box = FakeBody3D([0.0, 0, 0], FakeShape("box", half_extents=np.ones(3)))
    cyl = FakeBody3D([0.0, 0, 0], FakeShape("cylinder", radius=0.5,
                                            half_height=1.0))
    cone = FakeBody3D([0.0, 0, 0], FakeShape("cone"))
    assert viewer3d._actor_for(sphere) is not None
    assert viewer3d._actor_for(box) is not None
    assert viewer3d._actor_for(cyl) is not None
    assert viewer3d._actor_for(cone) is None


def test_place_actor_3d_uses_quaternion() -> None:
    """3D actors take the position verbatim and the quaternion orientation."""
    actor = _bare_actor()
    body = FakeBody3D([1.0, 2.0, 3.0], FakeShape("sphere", radius=0.5),
                      orn=(np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2))
    PhysicsViewer3D._place_actor(actor, body)
    assert actor.position == pytest.approx((1.0, 2.0, 3.0))
    # 90-degree z rotation rendered as axis * degrees
    assert actor.orientation == pytest.approx((0.0, 0.0, 90.0), abs=1e-4)


def test_panel_text_3d(viewer3d: PhysicsViewer3D) -> None:
    """The 3D HUD adds the angular momentum line."""
    text = viewer3d._panel_text()
    assert "ang. mom" in text
    assert "RUNNING" in text


def test_render_frame_3d_returns_image(viewer3d: PhysicsViewer3D) -> None:
    """Off-screen 3D rendering returns an (H, W, 3) uint8 image."""
    frame = viewer3d.render_frame()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_reset_3d(viewer3d: PhysicsViewer3D) -> None:
    """The 3D reset zeroes the world counters."""
    viewer3d.world.t = 5.0
    viewer3d.world.step_count = 12
    viewer3d._reset()
    assert viewer3d.world.t == 0.0
    assert viewer3d.world.step_count == 0
