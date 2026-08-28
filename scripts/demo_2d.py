"""Phase 1.2 demo: a 2D physics scene (falling/bouncing/stacking) visualized.

Run interactively:
    .\\.venv\\Scripts\\python.exe scripts/demo_2d.py

Or render headless frames to images:
    .\\.venv\\Scripts\\python.exe scripts/demo_2d.py --record --frames 120
"""

from __future__ import annotations

import argparse

import numpy as np

from pymo.kernel.bodies import Material, box_body, circle_body
from pymo.kernel.world import World
from pymo.viz.viewer import PhysicsViewer


def build_demo_world() -> World:
    """A small 2D scene: ground, a ramp of boxes, and bouncing balls."""
    w = World(gravity=np.array([0.0, -9.81]), dt=1 / 120.0, solver_iterations=10)
    # Ground
    ground = box_body([0.0, -0.5], 12.0, 0.5, static=True)
    w.add(ground)
    # A few static platforms
    for x in (-6.0, -1.0, 4.0):
        w.add(box_body([x, 0.5], 2.0, 0.25, static=True))
    # Falling balls with varied materials
    bouncy = Material(restitution=0.9, friction=0.1)
    heavy = Material(restitution=0.2, friction=0.5)
    for i, (x, mat) in enumerate(
        [(-5.5, bouncy), (-2.5, bouncy), (0.5, heavy), (3.5, bouncy), (6.5, heavy)]
    ):
        w.add(circle_body([x, 6.0 + i * 0.5], 0.4, mass=1.0 + 0.5 * i, material=mat))
    # A stack of boxes on the ground
    for j in range(3):
        w.add(box_body([-4.0, 1.0 + j * 1.0], 0.5, 0.5, mass=1.0))
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="headless record to images")
    ap.add_argument("--frames", type=int, default=120, help="frames to record")
    ap.add_argument("--out", default="out/frames", help="output dir for record mode")
    args = ap.parse_args()

    world = build_demo_world()
    viewer = PhysicsViewer(world, off_screen=args.record)

    if args.record:
        paths = viewer.record(args.frames, args.out)
        print(f"Recorded {len(paths)} frames to {args.out}")
    else:
        print("Interactive viewer. Press Space to pause, +/- speed, R reset, Esc quit.")
        viewer.run()


if __name__ == "__main__":
    main()
