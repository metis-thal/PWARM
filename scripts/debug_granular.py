"""Granular debug: check GL error after every operation in the render pipeline."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import moderngl
import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.viz.snapshot import CameraState, build_snapshot_from_world, DoubleBuffer
from pymo.viz.gl_renderer import GLRenderer, RendererConfig

def check(ctx, label):
    err = ctx.error
    status = "OK" if err == "GL_NO_ERROR" else f"ERROR: {err}"
    print(f"  [{label}] {status}")
    return err == "GL_NO_ERROR"

def main():
    engine = WorldEngine()
    engine.world.gravity = np.array([0.0, 0.0, -9.81])
    engine.world.dt = 1/60.0
    engine.world.solver_iterations = 10

    g = box_body([0,0,-0.5], half_extents=np.array([20,20,0.5]), static=True,
                 material=Material(restitution=0.3, friction=0.8, specific_heat=800,
                                   thermal_conductivity=2.0, density=2500))
    engine.add_body(g, albedo=0.2)

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

    ctx = renderer._ctx

    import glfw
    w, h = glfw.get_framebuffer_size(renderer._window)
    ctx.viewport = (0, 0, w, h)

    # Step 1: Clear
    check(ctx, "after viewport")
    bg = config.bg_color
    ctx.clear(bg[0], bg[1], bg[2])
    check(ctx, "after clear")

    # Step 2: Set uniforms
    cam = snap.camera
    view = cam.view_matrix()
    proj = cam.projection_matrix(w / h)
    cam_pos = cam.position()

    prog = renderer._instanced_program
    prog['u_view'].write(view.T.tobytes())
    check(ctx, "after u_view")
    prog['u_projection'].write(proj.T.tobytes())
    check(ctx, "after u_projection")
    prog['u_light_dir'].value = (0.5, 0.8, 1.0)
    prog['u_light_color'].value = (1.0, 0.98, 0.92)
    prog['u_ambient_color'].value = (0.4, 0.45, 0.5)
    prog['u_camera_pos'].value = tuple(cam_pos.tolist())
    prog['u_temp_min'].value = config.temp_min
    prog['u_temp_max'].value = config.temp_max
    prog['u_emission_strength'].value = config.emission_strength
    check(ctx, "after uniforms")

    # Step 3: Upload instances
    renderer._upload_instances(snap, frustum=None)
    check(ctx, "after upload instances")

    # Step 4: Check what we have
    for mesh_type, mg in renderer._mesh_groups.items():
        print(f"\n  {mesh_type.value}: instance_count={mg.instance_count}")

    # Step 5: Disable face culling to test
    print("\n  === DISABLING CULL FACE ===")
    ctx.disable(moderngl.CULL_FACE)
    check(ctx, "after disable cull face")

    # Step 6: Draw each mesh group individually
    for mesh_type, mg in renderer._mesh_groups.items():
        if mg.instance_count > 0:
            print(f"\n  === DRAWING {mesh_type.value} ({mg.instance_count} instances) ===")
            print(f"    VAO: {mg.vao}")
            print(f"    index_count: {mg.index_count}")
            check(ctx, f"before {mesh_type.value} draw")
            mg.vao.render(moderngl.TRIANGLES, instances=mg.instance_count)
            check(ctx, f"after {mesh_type.value} draw")

    # Step 7: Read pixels
    data = ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3))
    print(f"\n=== PIXELS ===")
    print(f"mean={img.mean():.1f}, min={img.min()}, max={img.max()}")

    non_bg = img.mean(axis=2) > 30
    print(f"Non-bg pixels: {non_bg.sum()} / {w*h}")

    if non_bg.sum() > 0:
        print(f"Color mean: {img[non_bg].mean(axis=0)}")
        print(f"Color min: {img[non_bg].min(axis=0)}")
        print(f"Color max: {img[non_bg].max(axis=0)}")

    renderer._cleanup()
    return 0

if __name__ == "__main__":
    main()
