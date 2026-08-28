"""2D physics visualization with PyVista.

Renders a 2D World on the z=0 plane viewed from above (orthographic), with a
live data panel (time, energy, per-body state) and keyboard playback controls.

Two modes:
  - Interactive: opens a window with keyboard controls (Space=pause, +/-=speed,
    R=reset, Esc=quit).
  - Headless/record: renders frames to PNG images for automated verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from pymo.kernel.bodies import Body
from pymo.kernel.world import World


@dataclass
class ViewerConfig:
    window_size: tuple[int, int] = (900, 700)
    bg_color: str = "black"
    body_color: str = "lightblue"
    ground_color: str = "gray"
    dt_render: int = 16          # timer callback period in ms (approx 60 fps)
    max_speed: float = 4.0
    min_speed: float = 0.25


class PhysicsViewer:
    """PyVista-based viewer for a 2D physics World."""

    def __init__(self, world: World, config: ViewerConfig | None = None, off_screen: bool = False):
        self.world = world
        self.config = config or ViewerConfig()
        self.off_screen = off_screen
        self.paused = False
        self.speed = 1.0
        self._actors: dict[int, object] = {}  # body id -> actor

        self.plotter = pv.Plotter(
            off_screen=off_screen, window_size=self.config.window_size
        )
        self.plotter.set_background(self.config.bg_color)
        self.plotter.enable_parallel_projection()
        self._setup_camera()
        self._build_actors()
        self._text_panel = self.plotter.add_text(
            "", position=(10, 10), font_size=12, color="white"
        )

    # -- setup --------------------------------------------------------------

    def _setup_camera(self) -> None:
        cam = self.plotter.camera
        cam_position = np.array([0.0, 0.0, 50.0])
        cam_focal = np.array([0.0, 0.0, 0.0])
        cam_up = np.array([0.0, 1.0, 0.0])
        cam.position = cam_position
        cam.focal_point = cam_focal
        cam.up = cam_up
        self.plotter.camera = cam

    def _build_actors(self) -> None:
        for body in self.world.bodies:
            self._actors[id(body)] = self._actor_for(body)

    def _actor_for(self, body: Body):
        if body.is_circle():
            mesh = pv.Sphere(radius=body.radius, theta_resolution=32, phi_resolution=16)
            color = self.config.body_color
        elif body.is_polygon():
            verts = np.asarray(body.vertices, dtype=float)
            points = np.column_stack([verts[:, 0], verts[:, 1], np.zeros(len(verts))])
            mesh = pv.PolyData(points)
            mesh = mesh.delaunay_2d()
            mesh = mesh.extrude((0, 0, 0.5), capping=True)
            color = self.config.ground_color if body.static else self.config.body_color
        else:
            return None
        actor = self.plotter.add_mesh(mesh, color=color, show_edges=True)
        self._place_actor(actor, body)
        return actor

    @staticmethod
    def _place_actor(actor, body: Body) -> None:
        actor.SetPosition(float(body.pos[0]), float(body.pos[1]), 0.0)
        actor.SetOrientation(0.0, 0.0, float(np.degrees(body.angle)))

    # -- per-frame update ---------------------------------------------------

    def _sync_actors(self) -> None:
        for body in self.world.bodies:
            actor = self._actors.get(id(body))
            if actor is None:
                continue
            self._place_actor(actor, body)

    def _panel_text(self) -> str:
        ke = self.world.total_kinetic_energy()
        mom = self.world.total_momentum()
        lines = [
            f"time      : {self.world.t:.3f} s",
            f"step      : {self.world.step_count}",
            f"bodies    : {len(self.world.bodies)}",
            f"kinetic E : {ke:.4f}",
            f"momentum  : ({mom[0]:.4f}, {mom[1]:.4f})",
            f"speed     : {self.speed:.2f}x",
            f"state     : {'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "Controls: Space=pause  +/-=speed  R=reset  Esc=quit",
        ]
        # Per-body detail (first few)
        for i, b in enumerate(self.world.bodies[:4]):
            lines.append(
                f"  b{i}: pos=({b.pos[0]:.2f},{b.pos[1]:.2f}) "
                f"vel=({b.vel[0]:.2f},{b.vel[1]:.2f})"
            )
        return "\n".join(lines)

    def render_frame(self) -> np.ndarray:
        """Synchronize actors, update the data panel, render one frame.

        Returns the rendered image (H, W, 3) uint8 for headless use.
        """
        self._sync_actors()
        self._text_panel.SetInput(self._panel_text())
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    # -- interactive loop ---------------------------------------------------

    def _on_timer(self) -> None:
        if not self.paused:
            # Step the world by speed * base_dt
            steps = max(1, round(self.speed))
            self.world.step(steps)
        self.render_frame()

    def add_controls(self) -> None:
        """Register keyboard callbacks."""
        self.plotter.add_key_callback(self._toggle_pause, key="space")
        self.plotter.add_key_callback(self._speed_up, key="plus")
        self.plotter.add_key_callback(self._speed_down, key="minus")
        self.plotter.add_key_callback(self._reset, key="r")

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _speed_up(self) -> None:
        self.speed = min(self.config.max_speed, self.speed * 1.25)

    def _speed_down(self) -> None:
        self.speed = max(self.config.min_speed, self.speed / 1.25)

    def _reset(self) -> None:
        # Rebuild actors to initial positions (assumes world bodies hold state)
        self.world.step_count = 0
        self.world.t = 0.0
        self._sync_actors()

    def run(self) -> None:
        """Start the interactive visualization loop."""
        self.add_controls()
        self.plotter.add_timer_callback(self._on_timer, self.config.dt_render)
        self.render_frame()
        self.plotter.show(auto_close=False)
        self.plotter.close()

    # -- headless record ----------------------------------------------------

    def record(self, n_frames: int, out_dir: str, steps_per_frame: int = 1) -> list[str]:
        """Render `n_frames` frames to PNGs in `out_dir`. Returns file paths."""
        import os

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i in range(n_frames):
            if not self.paused:
                self.world.step(steps_per_frame)
            self.render_frame()
            p = os.path.join(out_dir, f"frame_{i:04d}.png")
            self.plotter.screenshot(p)
            paths.append(p)
        return paths
