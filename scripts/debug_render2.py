"""Debug: verify the actual GLRenderer pipeline with proper instancing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.viz.snapshot import CameraState, build_snapshot_from_world, DoubleBuffer
from pymo.viz.gl_renderer import GLRenderer, RendererConfig

def main():
    engine = WorldEngine()
    engine.world.gravity = np.array([0.0, 0.0, -9.81])
    engine.world.dt = 1/60.0
    engine.world.solver_iterations = 10

    # Ground
    g = box_body([0,0,-0.5], half_extents=np.array([20,20,0.5]), static=True,
                 material=Material(restitution=0.3, friction=0.8, specific_heat=800,
                                   thermal_conductivity=2.0, density=2500))
    engine.add_body(g, albedo=0.2)

    # Iron sphere
    iron = sphere_body([0,0,4], radius=0.4, mass=2.5,
                       material=Material(restitution=0.3, friction=0.5, specific_heat=449,
                                         thermal_conductivity=80, density=7874))
    iron.temperature = 600.0
    engine.add_body(iron, albedo=0.3)

    for _ in range(60):
        engine.tick(dt=1/60.0)

    camera = CameraState(
        target=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        distance=15.0, elevation=np.pi/3.5, azimuth=0.3, fov=60.0)

    snap = build_snapshot_from_world(engine, camera=camera)
    buffer = DoubleBuffer()
    buffer.write(snap)

    config = RendererConfig(window_size=(640, 480), frustum_cull=False, target_fps=60)
    renderer = GLRenderer(buffer, config)
    renderer._init_glfw()
    renderer._init_shaders()
    renderer._init_meshes()

    # Print shader info
    prog = renderer._instanced_program
    print("=== SHADER ATTRIBUTES ===")
    for name in prog:
        print(f"  {name}")

    # Check VAO format
    for mesh_type, mg in renderer._mesh_groups.items():
        print(f"\n=== MESH GROUP: {mesh_type.value} ===")
        print(f"  index_count: {mg.index_count}")

    # Upload instances
    renderer._upload_instances(snap, frustum=None)
    for mesh_type, mg in renderer._mesh_groups.items():
        print(f"  {mesh_type.value}: instance_count={mg.instance_count}")

    # Draw WITHOUT swap
    renderer._draw_frame(snap)

    # Read pixels
    import glfw
    w, h = glfw.get_framebuffer_size(renderer._window)
    data = renderer._ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3))
    print(f"\n=== PIXELS ===")
    print(f"Image: {w}x{h}, mean={img.mean():.1f}, min={img.min()}, max={img.max()}")

    # Check non-background
    bg = np.all(np.abs(img.astype(int) - [12, 12, 20]) < 8, axis=2)
    non_bg = ~bg
    print(f"Non-bg pixels: {non_bg.sum()} / {w*h} ({non_bg.mean()*100:.1f}%)")
    if non_bg.sum() > 0:
        non_bg_colors = img[non_bg]
        print(f"Non-bg color mean: {non_bg_colors.mean(axis=0)}")
        print(f"Non-bg color min: {non_bg_colors.min(axis=0)}")
        print(f"Non-bg color max: {non_bg_colors.max(axis=0)}")

    # Check if gl error
    err = renderer._ctx.error
    print(f"\nOpenGL error: {err}")

    renderer._cleanup()
    return 0


if __name__ == "__main__":
    main()
