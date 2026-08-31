"""3D physics visualization with PyVista.

Renders a 3D World on the z=0 plane viewed from above (orthographic) with
a live data panel (time, energy, per-body state) and keyboard playback controls.

Two modes:
  - Interactive: opens a window with keyboard controls (Space=pause, +/-=speed,
    R=reset, Esc=quit).
  - Headless/record: renders frames to PNG images for automated verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from pymo.kernel.math3d import quat_to_axis_angle
from pymo.kernel.world3d import World3D


@dataclass
class ViewerConfig3D:
    window_size: tuple[int, int] = (900, 700)
    bg_color: str = "black"
    body_color: str = "lightblue"
    ground_color: str = "gray"
    dt_render: int = 16          # timer callback period in ms (approx 60 fps)
    max_speed: float = 4.0
    min_speed: float = 0.25


class PhysicsViewer3D:
    """PyVista-based viewer for a 3D physics World."""

    def __init__(self, world: World3D, config: ViewerConfig3D | None = None, off_screen: bool = False):
        self.world = world
        self.config = config or ViewerConfig3D()
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

    def _actor_for(self, body):
        if body.is_sphere():
            mesh = pv.Sphere(radius=body.shape.radius, theta_resolution=32, phi_resolution=16)
            color = self.config.body_color
        elif body.is_box():
            extents = body.shape.half_extents
            mesh = pv.Box(bounds=[
                -extents[0], extents[0],
                -extents[1], extents[1],
                -extents[2], extents[2]
            ])
            color = self.config.ground_color if body.static else self.config.body_color
        elif body.is_cylinder():
            mesh = pv.Cylinder(radius=body.shape.radius, height=body.shape.half_height * 2)
            color = self.config.body_color
        else:
            return None
        actor = self.plotter.add_mesh(mesh, color=color, show_edges=True)
        self._place_actor(actor, body)
        return actor

    @staticmethod
    def _place_actor(actor, body) -> None:
        """Position and orient the actor according to body state."""
        # Apply translation
        actor.SetPosition(*body.pos)
        # Apply rotation (convert quaternion to orientation)
        # Note: PyVista uses Euler angles in degrees, we need to convert
        axis, angle = quat_to_axis_angle(body.orn)
        angle_deg = np.degrees(angle)
        actor.SetOrientation(axis * angle_deg)

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
        ang_mom = self.world.total_angular_momentum()
        lines = [
            f"time      : {self.world.t:.3f} s",
            f"step      : {self.world.step_count}",
            f"bodies    : {len(self.world.bodies)}",
            f"kinetic E : {ke:.4f}",
            f"momentum  : ({mom[0]:.4f}, {mom[1]:.4f}, {mom[2]:.4f})",
            f"ang. mom  : ({ang_mom[0]:.4f}, {ang_mom[1]:.4f}, {ang_mom[2]:.4f})",
            f"speed     : {self.speed:.2f}x",
            f"state     : {'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "Controls: Space=pause  +/-=speed  R=reset  Esc=quit",
        ]
        # Per-body detail (first few)
        for i, b in enumerate(self.world.bodies[:4]):
            lines.append(
                f"  b{i}: pos=({b.pos[0]:.2f},{b.pos[1]:.2f},{b.pos[2]:.2f}) "
                f"vel=({b.vel[0]:.2f},{b.vel[1]:.2f},{b.vel[2]:.2f})"
            )
        return "\n".join(lines)

    def render_frame(self) -> np.ndarray | None:
        """Synchronize actors, update the data panel, render one frame.

        Returns the rendered image (H, W, 3) uint8 in headless mode, else None.
        """
        self._sync_actors()
        self._text_panel.SetInput(self._panel_text())
        self.plotter.render()
        if self.off_screen:
            return self.plotter.screenshot(return_img=True)
        return None

    # -- interactive loop ---------------------------------------------------

    def _on_timer(self) -> None:
        if not self.paused:
            steps = max(1, round(self.speed))
            self.world.step(steps)
        self._sync_actors()
        self._text_panel.SetInput(self._panel_text())
        self.plotter.render()

    def add_controls(self) -> None:
        """Register keyboard callbacks."""
        self.plotter.add_key_event("space", lambda: self._toggle_pause())
        self.plotter.add_key_event("plus", lambda: self._speed_up())
        self.plotter.add_key_event("minus", lambda: self._speed_down())
        self.plotter.add_key_event("r", lambda: self._reset())

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
        self.plotter.add_timer_event(max_steps=0, duration=self.config.dt_render, callback=self._on_timer)
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


# Convenience function for quick demos
def demo_3d() -> PhysicsViewer3D:
    """Run an interactive 3D physics demo."""
    import numpy as np

    from pymo.kernel.bodies3d import Material, box_body, sphere_body
    from pymo.kernel.world3d import World3D

    w = World3D(gravity=np.array([0.0, 0.0, -9.81]), dt=1/120.0, solver_iterations=10)
    # Ground
    w.add(box_body([0.0, 0.0, -0.5], np.array([12.0, 12.0, 0.5]), static=True))
    # A few static platforms
    for x in (-6.0, -1.0, 4.0):
        w.add(box_body([x, 0.5, 0.0], np.array([2.0, 2.0, 0.25]), static=True))
    # Falling balls with varied materials
    bouncy = Material(restitution=0.9, friction=0.1)
    heavy = Material(restitution=0.2, friction=0.5)
    for i, (x, mat) in enumerate(
        [(-5.5, bouncy), (-2.5, bouncy), (0.5, heavy), (3.5, bouncy), (6.5, heavy)]
    ):
        w.add(sphere_body([x, 0.0, 6.0 + i * 0.5], 0.4, mass=1.0 + 0.5 * i, material=mat))
    # A stack of boxes on the ground
    for j in range(3):
        w.add(box_body([-4.0, 0.0, 1.0 + j * 1.0], np.array([0.5, 0.5, 0.5]), mass=1.0))
    return PhysicsViewer3D(w)


if __name__ == "__main__":
    demo_3d().run()