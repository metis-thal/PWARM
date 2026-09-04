"""3D Physics Visualization — WorldEngine + OpenGL GPU-Instanced Rendering.

Demonstrates the new Genesis-inspired physics engine with real-time 3D rendering:
- Rigid body dynamics (spheres, boxes falling under gravity)
- GPU-instanced rendering (one DrawCall per mesh type)
- PBR lighting with temperature emission
- Orbit camera with mouse control
- Double-buffered snapshot exchange (sim/render separation)

Controls:
    Mouse drag     — Rotate view
    Mouse scroll   — Zoom in/out
    Space          — Pause/Resume simulation
    R              — Reset simulation
    1/2/3          — Add sphere/box/capsule at random position
    Q / Esc        — Quit

Usage:
    python scripts/demo_3d_physics.py
    # or built as EXE: dist/PWARM_3DPhysics.exe
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pymo.physics import WorldEngine, WorldEngineConfig
from pymo.viz.snapshot import (
    CameraState, DoubleBuffer, SceneSnapshot,
    build_snapshot_from_physics_engine, build_model_matrix,
    InstanceData, MeshType, compute_aabb_sphere, compute_aabb_box,
)


def create_demo_scene() -> WorldEngine:
    """Create a demo scene with various rigid bodies."""
    config = WorldEngineConfig(
        dt=1/60,
        substeps=1,
        gravity=(0.0, 0.0, -9.81),
        rigid={'enabled': True},
    )
    engine = WorldEngine(config)

    # Ground plane (static)
    engine.create_rigid_body(
        (0, 0, -0.5), mass=0.0,
        shape='box', shape_params={'half_extents': [20, 20, 0.5]}
    )

    # Several spheres at different heights
    colors = [
        (0, 0, 8),   # high
        (2, 0, 6),   # medium
        (-2, 0, 10), # very high
        (0, 2, 5),   # medium
        (0, -2, 12), # very high
    ]
    for pos in colors:
        engine.create_rigid_body(
            pos, mass=1.0,
            shape='sphere', shape_params={'radius': 0.5}
        )

    # A few boxes
    engine.create_rigid_body(
        (3, 3, 7), mass=2.0,
        shape='box', shape_params={'half_extents': [0.4, 0.4, 0.4]}
    )
    engine.create_rigid_body(
        (-3, -3, 9), mass=1.5,
        shape='box', shape_params={'half_extents': [0.3, 0.6, 0.3]}
    )

    engine.finalize_setup()
    return engine


def main():
    try:
        import glfw
        import moderngl
    except ImportError:
        print("ERROR: Missing OpenGL dependencies.")
        print("Install with: pip install moderngl glfw PyOpenGL")
        input("\nPress Enter to exit...")
        return 1

    from pymo.viz.gl_renderer import GLRenderer, RendererConfig

    print("=" * 60)
    print("  PWARM 3D Physics Visualization")
    print("  Genesis-inspired Engine + OpenGL GPU Instancing")
    print("=" * 60)

    # Create engine and scene
    print("\n[1] Creating physics scene...")
    engine = create_demo_scene()
    print(f"    Entities: {len(engine.scene.entities)}")
    print(f"    Solvers: {list(engine.scene.solvers.keys())}")

    # Create double buffer
    buffer = DoubleBuffer()

    # Initial snapshot
    snapshot = build_snapshot_from_physics_engine(engine)
    buffer.write(snapshot)

    # Create renderer
    print("\n[2] Initializing OpenGL renderer...")
    config = RendererConfig(window_size=(1280, 720))
    renderer = GLRenderer(buffer, config)
    renderer.init()

    # Simulation state
    paused = False
    sim_time = 0.0
    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0

    print("\n[3] Starting render loop...")
    print("    Controls: Mouse drag=rotate, Scroll=zoom, Space=pause, R=reset, Q=quit")
    print("              1=add sphere, 2=add box, 3=add capsule")

    # Main loop
    while renderer.running:
        # Handle input
        key = renderer.poll_key()
        if key == ord('Q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):
            paused = not paused
            print(f"    {'PAUSED' if paused else 'RUNNING'}")
        elif key == ord('R'):
            engine = create_demo_scene()
            sim_time = 0.0
            frame_count = 0
            print("    RESET")
        elif key == ord('1'):
            # Add sphere
            pos = (np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(8, 15))
            engine.create_rigid_body(pos, mass=1.0, shape='sphere', shape_params={'radius': 0.5})
            engine.finalize_setup()
            print(f"    Added sphere at {pos}")
        elif key == ord('2'):
            # Add box
            pos = (np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(8, 15))
            engine.create_rigid_body(pos, mass=1.5, shape='box', shape_params={'half_extents': [0.4, 0.4, 0.4]})
            engine.finalize_setup()
            print(f"    Added box at {pos}")

        # Step simulation
        if not paused:
            engine.tick()
            sim_time += engine.config.dt
            frame_count += 1

        # Build snapshot from engine state
        snapshot = build_snapshot_from_physics_engine(engine, renderer.camera)
        buffer.write(snapshot)

        # Render
        renderer.render_frame()

        # FPS counter
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now
            renderer.set_title(f"PWARM 3D Physics | t={sim_time:.2f}s | {fps:.0f} FPS | {len(engine.scene.entities)} bodies")

    print("\n[4] Shutting down...")
    renderer.close()
    print("    Done!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
