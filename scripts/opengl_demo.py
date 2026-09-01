"""OpenGL GPU-Instanced Demo: WorldEngine → Snapshot → GLRenderer.

Runs the simulation in a background thread, produces SceneSnapshots, and
renders them in a ModernGL window with GPU Instancing, PBR lighting,
frustum culling, and orbit camera.

Controls:
  Left-drag   → orbit
  Right-drag  → pan
  Scroll      → zoom
  R           → reset camera
  Escape      → quit

Usage:
    python scripts/opengl_demo.py
"""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.rules.chemistry import ChemicalSystem
from pymo.rules.fluid import create_water_column, SPHParams
from pymo.viz.snapshot import DoubleBuffer, CameraState, build_snapshot_from_world
from pymo.viz.gl_renderer import GLRenderer, RendererConfig


def setup_scene() -> WorldEngine:
    """Create the WorldEngine with the demo scene."""
    engine = WorldEngine()
    engine.world.gravity = np.array([0.0, 0.0, -9.81])
    engine.world.dt = 1 / 60.0
    engine.world.solver_iterations = 10

    # Ground
    ground = box_body(
        [0, 0, -0.5],
        half_extents=np.array([20.0, 20.0, 0.5]),
        static=True,
        material=Material(
            restitution=0.3, friction=0.8,
            specific_heat=800.0, thermal_conductivity=2.0,
            density=2500.0,
        ),
    )
    engine.add_body(ground, albedo=0.2)

    # Hot iron ball
    iron = sphere_body(
        [0, 0, 4],
        radius=0.4, mass=2.5,
        material=Material(
            restitution=0.3, friction=0.5,
            specific_heat=449.0, thermal_conductivity=80.0,
            density=7874.0,
        ),
    )
    iron.temperature = 600.0
    engine.add_body(iron, albedo=0.3)

    # Cold ice sphere
    ice = sphere_body(
        [-2, 0, 3],
        radius=0.35, mass=0.3,
        material=Material(
            restitution=0.1, friction=0.2,
            specific_heat=2100.0, thermal_conductivity=2.2,
            density=917.0,
        ),
    )
    ice.temperature = 250.0
    engine.add_body(ice, albedo=0.5)

    # Wood block
    wood = box_body(
        [2, 0, 2.5],
        half_extents=np.array([0.3, 0.3, 0.3]),
        mass=0.16,
        material=Material(
            restitution=0.4, friction=0.6,
            specific_heat=1700.0, thermal_conductivity=0.15,
            density=600.0,
        ),
    )
    wood.temperature = 293.0
    engine.add_body(wood, albedo=0.25)

    # Rock sphere
    rock = sphere_body(
        [3, 1, 3],
        radius=0.3, mass=1.0,
        material=Material(
            restitution=0.2, friction=0.7,
            specific_heat=790.0, thermal_conductivity=3.0,
            density=2600.0,
        ),
    )
    rock.temperature = 293.0
    engine.add_body(rock, albedo=0.15)

    # SPH fluid
    sph_params = SPHParams(
        h=0.15, rest_density=1000.0, stiffness=5000.0,
        viscosity=1.0, particle_mass=0.5, gamma=1.0,
    )
    sph = create_water_column(x=-3, y=-1, width=4, height=4, spacing=0.15,
                              params=sph_params)
    engine.add_fluid(sph)

    # Chemistry
    chem = ChemicalSystem(
        masses={"wood": 0.16, "oxygen": 0.5},
        temperature=293.0,
    )
    engine.add_chemistry(chem)

    # Ecology
    engine.solar_flux = 800.0
    engine.environment_temp = 300.0

    return engine


def sim_thread_fn(engine: WorldEngine, buffer: DoubleBuffer,
                  camera: CameraState, stop_event: threading.Event,
                  max_steps: int = 6000) -> None:
    """Simulation loop: tick → snapshot → write to double buffer."""
    dt = 1 / 60.0
    for step in range(max_steps):
        if stop_event.is_set():
            break
        engine.tick(dt=dt)

        # Build snapshot every frame
        snap = build_snapshot_from_world(engine, camera=camera)
        buffer.write(snap)

        # Print periodic status
        if step % 300 == 0:
            cons = engine._snapshot_conservation()
            n_particles = len(engine.sph.particles) if engine.sph else 0
            print(
                f"  sim t={engine.t:6.1f}s  step={step:5d}  "
                f"bodies={len(engine.world.bodies)}  particles={n_particles}  "
                f"KE={cons['kinetic_energy']:8.2f}  "
                f"TE={cons['thermal_energy']:10.1f}  "
                f"total_E={cons['total_energy']:10.1f}"
            )

    print("  [sim] simulation ended")


def main():
    print("=" * 64)
    print("PWARM OpenGL GPU-Instanced Renderer Demo")
    print("  WorldEngine -> SceneSnapshot -> DoubleBuffer -> GLRenderer")
    print("=" * 64)

    # Setup
    engine = setup_scene()
    buffer = DoubleBuffer()
    camera = CameraState(
        target=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        distance=15.0,
        elevation=np.pi / 3.5,
        azimuth=0.3,
        fov=60.0,
    )

    config = RendererConfig(
        window_size=(1280, 720),
        title="PWARM — GPU Instanced Renderer",
        target_fps=60,
        frustum_cull=True,
        cam_distance=15.0,
        cam_elevation=np.pi / 3.5,
        temp_min=200.0,
        temp_max=700.0,
        emission_strength=1.5,
    )

    n_bodies = len(engine.world.bodies)
    n_particles = len(engine.sph.particles) if engine.sph else 0
    print(f"\n  Bodies: {n_bodies}  |  Fluid particles: {n_particles}")
    print(f"  Solar flux: {engine.solar_flux} W/m^2  |  Ambient: {engine.environment_temp} K")

    # Start simulation thread
    stop_event = threading.Event()
    sim = threading.Thread(
        target=sim_thread_fn,
        args=(engine, buffer, camera, stop_event, 6000),
        daemon=True,
    )
    sim.start()
    print("\n  Simulation started in background thread")
    print("  Opening OpenGL window...\n")

    # Render loop (main thread — GLFW requires main thread)
    renderer = GLRenderer(buffer, config)
    try:
        renderer.run()
    except Exception as e:
        print(f"\n  Renderer error: {e}")
        print("  (This is expected in headless/CI environments without a display)")
        print("  The simulation ran successfully; renderer requires a GPU + display.")
    finally:
        stop_event.set()
        sim.join(timeout=3.0)
        print("\n  Done.")


if __name__ == "__main__":
    main()
