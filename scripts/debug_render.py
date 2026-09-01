"""Debug: check why the rendered image is all black."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.viz.snapshot import CameraState, build_snapshot_from_world
from pymo.viz.gl_renderer import GLRenderer, RendererConfig, FrustumPlanes, DoubleBuffer

def main():
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

    # Step a few frames
    for _ in range(60):
        engine.tick(dt=1/60.0)

    camera = CameraState(
        target=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        distance=15.0,
        elevation=np.pi / 3.5,
        azimuth=0.3,
        fov=60.0,
    )

    snap = build_snapshot_from_world(engine, camera=camera)

    # Debug: check snapshot
    print("=== SNAPSHOT DEBUG ===")
    print(f"  t={snap.t:.2f}, step={snap.step}")
    print(f"  instances: {len(snap.instances)}")
    for i, inst in enumerate(snap.instances):
        print(f"    [{i}] mesh={inst.mesh_type.value} T={inst.temperature:.1f}")
        print(f"         model_matrix diag: {np.diag(inst.model_matrix)}")
        print(f"         model_matrix pos: {inst.model_matrix[:3, 3]}")
        print(f"         aabb: {inst.aabb_min} -> {inst.aabb_max}")
        print(f"         base_color: {inst.base_color}")
        print(f"         temperature: {inst.temperature}")

    # Debug: check camera
    print("\n=== CAMERA DEBUG ===")
    pos = camera.position()
    print(f"  position: {pos}")
    print(f"  target: {camera.target}")
    print(f"  distance: {camera.distance}")
    print(f"  elevation: {camera.elevation:.3f} rad = {np.degrees(camera.elevation):.1f} deg")
    print(f"  azimuth: {camera.azimuth:.3f} rad = {np.degrees(camera.azimuth):.1f} deg")
    view = camera.view_matrix()
    proj = camera.projection_matrix(1280/720)
    print(f"  view matrix:\n{view}")
    print(f"  projection matrix:\n{proj}")

    # Debug: frustum
    vp = proj @ view
    frustum = FrustumPlanes.from_view_projection(vp)
    print(f"\n=== FRUSTUM DEBUG ===")
    for i, name in enumerate(['Left', 'Right', 'Bottom', 'Top', 'Near', 'Far']):
        print(f"  {name}: {frustum.planes[i]}")

    # Test each instance against frustum
    print(f"\n=== FRUSTUM CULLING ===")
    for i, inst in enumerate(snap.instances):
        inside = frustum.is_aabb_inside(inst.aabb_min, inst.aabb_max)
        print(f"  [{i}] {inst.mesh_type.value} aabb={inst.aabb_min}->{inst.aabb_max} inside_frustum={inside}")

    # Debug: init renderer
    print("\n=== RENDERER INIT ===")
    buffer = DoubleBuffer()
    buffer.write(snap)
    config = RendererConfig(window_size=(1280, 720), frustum_cull=False)  # disable culling for debug
    renderer = GLRenderer(buffer, config)
    renderer._init_glfw()
    renderer._init_shaders()
    renderer._init_meshes()

    # Manually check instance upload
    print("\n=== INSTANCE UPLOAD (no frustum) ===")
    renderer._upload_instances(snap, frustum=None)
    for mesh_type, mg in renderer._mesh_groups.items():
        print(f"  {mesh_type.value}: instance_count={mg.instance_count}")

    # Render (draw only, no swap yet)
    print("\n=== RENDER ===")
    renderer._last_snapshot = snap
    renderer._draw_frame(snap)

    # Read pixels BEFORE swap
    import glfw
    w, h = glfw.get_framebuffer_size(renderer._window)
    data = renderer._ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3))
    print(f"  Image: {w}x{h}, mean={img.mean():.1f}, min={img.min()}, max={img.max()}")

    # Check specific regions
    center = img[300:420, 500:780]  # center-ish region
    print(f"  Center region mean: {center.mean():.1f}")

    # Check if clear color appears
    # bg_color = (0.05, 0.05, 0.08) -> ~12, 12, 20 in uint8
    bg_pixels = np.all(np.abs(img.astype(int) - [12, 12, 20]) < 5, axis=2)
    print(f"  Background-colored pixels: {bg_pixels.sum()} / {w*h} ({bg_pixels.mean()*100:.1f}%)")

    # Check for non-background pixels (geometry!)
    non_bg = ~bg_pixels
    print(f"  Non-background pixels: {non_bg.sum()} / {w*h} ({non_bg.mean()*100:.1f}%)")
    if non_bg.sum() > 0:
        non_bg_colors = img[non_bg]
        print(f"  Non-bg color mean: {non_bg_colors.mean(axis=0)}")
        print(f"  Non-bg color min: {non_bg_colors.min(axis=0)}")
        print(f"  Non-bg color max: {non_bg_colors.max(axis=0)}")

    # Now swap
    renderer._swap_buffers()

    renderer._cleanup()
    print("\nDone.")

if __name__ == "__main__":
    main()
