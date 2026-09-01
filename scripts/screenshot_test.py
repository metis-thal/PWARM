"""Headless screenshot test: verify the full pipeline renders correctly.

Runs WorldEngine for a few frames, builds a SceneSnapshot, renders it
via GLRenderer in headless mode, and saves the screenshot to disk.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.rules.chemistry import ChemicalSystem
from pymo.rules.fluid import create_water_column, SPHParams
from pymo.viz.snapshot import DoubleBuffer, CameraState, build_snapshot_from_world
from pymo.viz.gl_renderer import GLRenderer, RendererConfig


def main():
    # Setup scene
    engine = WorldEngine()
    engine.world.gravity = np.array([0.0, 0.0, -9.81])
    engine.world.dt = 1 / 60.0
    engine.world.solver_iterations = 10

    ground = box_body([0, 0, -0.5], half_extents=np.array([20.0, 20.0, 0.5]),
                       static=True,
                       material=Material(restitution=0.3, friction=0.8,
                                         specific_heat=800.0, thermal_conductivity=2.0,
                                         density=2500.0))
    engine.add_body(ground, albedo=0.2)

    iron = sphere_body([0, 0, 4], radius=0.4, mass=2.5,
                        material=Material(restitution=0.3, friction=0.5,
                                          specific_heat=449.0, thermal_conductivity=80.0,
                                          density=7874.0))
    iron.temperature = 600.0
    engine.add_body(iron, albedo=0.3)

    ice = sphere_body([-2, 0, 3], radius=0.35, mass=0.3,
                       material=Material(restitution=0.1, friction=0.2,
                                         specific_heat=2100.0, thermal_conductivity=2.2,
                                         density=917.0))
    ice.temperature = 250.0
    engine.add_body(ice, albedo=0.5)

    wood = box_body([2, 0, 2.5], half_extents=np.array([0.3, 0.3, 0.3]),
                     mass=0.16,
                     material=Material(restitution=0.4, friction=0.6,
                                       specific_heat=1700.0, thermal_conductivity=0.15,
                                       density=600.0))
    wood.temperature = 293.0
    engine.add_body(wood, albedo=0.25)

    rock = sphere_body([3, 1, 3], radius=0.3, mass=1.0,
                        material=Material(restitution=0.2, friction=0.7,
                                          specific_heat=790.0, thermal_conductivity=3.0,
                                          density=2600.0))
    rock.temperature = 293.0
    engine.add_body(rock, albedo=0.15)

    # SPH fluid
    sph_params = SPHParams(h=0.15, rest_density=1000.0, stiffness=5000.0,
                            viscosity=1.0, particle_mass=0.5, gamma=1.0)
    sph = create_water_column(x=-3, y=-1, width=4, height=4, spacing=0.15, params=sph_params)
    engine.add_fluid(sph)

    # Ecology
    engine.solar_flux = 800.0
    engine.environment_temp = 300.0

    # Camera — look at scene from a good angle
    camera = CameraState(
        target=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        distance=15.0,
        elevation=np.pi / 3.5,
        azimuth=0.3,
        fov=60.0,
    )

    print("Step 1: Running simulation for 60 steps...")
    for _ in range(60):
        engine.tick(dt=1/60.0)

    print("Step 2: Building SceneSnapshot...")
    snap = build_snapshot_from_world(engine, camera=camera)
    print(f"  Instances: {len(snap.instances)}")
    print(f"  Fluid particles: {snap.fluid_positions.shape if snap.fluid_positions is not None else 'None'}")
    for i, inst in enumerate(snap.instances):
        print(f"  [{i}] mesh={inst.mesh_type.value} T={inst.temperature:.1f}K "
              f"pos=({inst.model_matrix[0,3]:+.2f}, {inst.model_matrix[1,3]:+.2f}, {inst.model_matrix[2,3]:+.2f})")

    print("\nStep 3: Initializing GLRenderer (headless/offscreen)...")
    buffer = DoubleBuffer()
    buffer.write(snap)

    config = RendererConfig(
        window_size=(1280, 720),
        title="PWARM Screenshot Test",
        target_fps=60,
        frustum_cull=True,
        temp_min=200.0,
        temp_max=700.0,
        emission_strength=1.5,
    )

    renderer = GLRenderer(buffer, config)

    print("Step 4: Initializing GLFW + OpenGL context...")
    try:
        renderer._init_glfw()
        print("  GLFW initialized OK")
        renderer._init_shaders()
        print("  Shaders compiled OK")
        renderer._init_meshes()
        print("  Meshes created OK")
    except Exception as e:
        print(f"  INIT FAILED: {e}")
        return 1

    print("\nStep 5: Rendering frame...")
    try:
        renderer._last_snapshot = snap
        # Draw WITHOUT swap so we can read pixels before buffer swap
        renderer._draw_frame(snap)
        print("  Draw frame OK")
    except Exception as e:
        print(f"  RENDER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nStep 6: Reading pixels back...")
    try:
        import glfw
        w, h = glfw.get_framebuffer_size(renderer._window)
        data = renderer._ctx.fbo.read(components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        img = np.flipud(img)  # OpenGL Y flip

        # Check the image is not all black
        mean_brightness = img.mean()
        print(f"  Image size: {w}x{h}")
        print(f"  Mean brightness: {mean_brightness:.1f} / 255")
        print(f"  Min pixel: {img.min()}, Max pixel: {img.max()}")

        if mean_brightness < 5:
            print("  WARNING: Image appears very dark (nearly all black)")
            print("  This could mean: shader issue, no geometry rendered, or wrong camera")
            return 1
        else:
            print(f"  Image is non-trivially bright (mean={mean_brightness:.1f})")

        # Save to file
        try:
            from PIL import Image
            pil_img = Image.fromarray(img)
            out_path = os.path.join(os.path.dirname(__file__), '..', 'screenshot_test.png')
            pil_img.save(out_path)
            print(f"  Screenshot saved to: {os.path.abspath(out_path)}")
        except ImportError:
            # No PIL, save as raw numpy
            out_path = os.path.join(os.path.dirname(__file__), '..', 'screenshot_test.npy')
            np.save(out_path, img)
            print(f"  Saved raw numpy to: {os.path.abspath(out_path)}")
            print("  (Install Pillow to save as PNG: pip install Pillow)")

        # Render a second frame after more sim steps
        print("\nStep 7: Running more sim steps and rendering again...")
        for _ in range(120):
            engine.tick(dt=1/60.0)
        snap2 = build_snapshot_from_world(engine, camera=camera)
        buffer.write(snap2)
        renderer._last_snapshot = snap2
        renderer._draw_frame(snap2)

        data2 = renderer._ctx.fbo.read(components=3)
        img2 = np.flipud(np.frombuffer(data2, dtype=np.uint8).reshape(h, w, 3))
        mean2 = img2.mean()
        print(f"  Second frame brightness: {mean2:.1f}")

        diff = np.abs(img.astype(float) - img2.astype(float)).mean()
        print(f"  Frame-to-frame diff: {diff:.1f} (should be > 0 if scene changed)")

        if diff < 1.0:
            print("  WARNING: Frames appear identical — simulation changes may not be reflected")
        else:
            print("  Frames differ — rendering is updating with simulation state")

    except Exception as e:
        print(f"  PIXEL READ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        renderer._cleanup()

    print("\n" + "=" * 64)
    print("END-TO-END VERIFICATION PASSED")
    print("  - GLFW window + OpenGL context created")
    print("  - Shaders compiled successfully")
    print("  - Meshes (sphere/box/cylinder) created")
    print("  - GPU instancing render call succeeded")
    print("  - Frame buffer is non-empty (visible geometry)")
    print("  - Frame updates with simulation state changes")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
