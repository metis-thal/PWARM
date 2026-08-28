"""Tests for the 2D PyVista viewer (headless rendering)."""

from __future__ import annotations

import numpy as np

from pymo.kernel.bodies import box_body, circle_body
from pymo.kernel.world import World
from pymo.viz.viewer import PhysicsViewer


def test_viewer_renders_frame_offscreen():
    """The viewer can render the world state to an image headlessly."""
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    w.add(box_body([0.0, -0.5], 5.0, 0.5, static=True))
    w.add(circle_body([0.0, 2.0], 0.5, mass=1.0))
    viewer = PhysicsViewer(w, off_screen=True)
    img = viewer.render_frame()
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    viewer.plotter.close()


def test_viewer_advances_and_renders():
    """Rendering after stepping shows updated positions without error."""
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    w.add(box_body([0.0, -0.5], 5.0, 0.5, static=True))
    w.add(circle_body([0.0, 3.0], 0.5, mass=1.0))
    viewer = PhysicsViewer(w, off_screen=True)
    w.step(10)
    img = viewer.render_frame()
    assert img is not None
    viewer.plotter.close()


def test_viewer_record_produces_files(tmp_path):
    """record() writes PNG files and advances the simulation."""
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    w.add(box_body([0.0, -0.5], 5.0, 0.5, static=True))
    w.add(circle_body([0.0, 3.0], 0.5, mass=1.0))
    viewer = PhysicsViewer(w, off_screen=True)
    out = tmp_path / "frames"
    paths = viewer.record(5, str(out), steps_per_frame=5)
    assert len(paths) == 5
    for p in paths:
        import os

        assert os.path.exists(p)
    # world advanced
    assert w.step_count == 5 * 5
    viewer.plotter.close()
