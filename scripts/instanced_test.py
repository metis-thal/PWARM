"""Test instanced rendering: draw a red sphere and a green sphere using GPU instancing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import glfw
import moderngl

# Minimal vertex shader for instancing
VS = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;

in vec4 in_model_0;
in vec4 in_model_1;
in vec4 in_model_2;
in vec4 in_model_3;

in vec3 in_color;
in float in_temperature;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_color;
out float v_temp;

void main() {
    mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    v_color = in_color;
    v_temp = in_temperature;
}
"""

FS = """
#version 330 core
in vec3 v_color;
in float v_temp;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


def make_sphere(radius=1.0, sectors=12, stacks=8):
    verts, norms, idx = [], [], []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            x, y, z = np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)
            verts.append([x*radius, y*radius, z*radius])
            norms.append([x, y, z])
    for i in range(stacks):
        for j in range(sectors):
            f = i*(sectors+1)+j
            s = f + sectors + 1
            idx.extend([f, s, f+1, s, s+1, f+1])
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(idx, np.uint32)


def make_model_matrix(pos, scale=1.0):
    m = np.eye(4, dtype=np.float32)
    m[0,0] = m[1,1] = m[2,2] = scale
    m[:3, 3] = pos
    return m


def main():
    if not glfw.init(): return 1

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(640, 480, "Instanced Test", None, None)
    if not window:
        glfw.terminate()
        return 1

    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    verts, norms, idx = make_sphere(1.0, 12, 8)
    n_verts = len(verts)
    interleaved = np.zeros((n_verts, 6), np.float32)
    interleaved[:, :3] = verts
    interleaved[:, 3:6] = norms

    vbo = ctx.buffer(interleaved.tobytes())
    ibo = ctx.buffer(idx.tobytes())

    # 2 instances
    instance_buf = ctx.buffer(reserve=2 * 64)
    color_buf = ctx.buffer(reserve=2 * 12)
    temp_buf = ctx.buffer(reserve=2 * 4)

    vao = ctx.vertex_array(prog, [
        (vbo, '3f 3f', 'in_position', 'in_normal'),
        (instance_buf, '4f 4f 4f 4f /i', 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
        (color_buf, '3f /i', 'in_color'),
        (temp_buf, '1f /i', 'in_temperature'),
    ], ibo)

    # Upload instance data
    # Red sphere at x=-2, Green sphere at x=+2
    mat_data = np.array([
        make_model_matrix([-2.0, 0.0, 0.0], 0.5).flatten(),
        make_model_matrix([ 2.0, 0.0, 0.0], 0.5).flatten(),
    ], np.float32)
    instance_buf.write(mat_data.tobytes())

    color_data = np.array([
        [1.0, 0.2, 0.2],  # red
        [0.2, 1.0, 0.2],  # green
    ], np.float32)
    color_buf.write(color_data.tobytes())

    temp_data = np.array([600.0, 250.0], np.float32)
    temp_buf.write(temp_data.tobytes())

    # Camera: looking at origin from z=5
    view = np.eye(4, dtype=np.float32)
    view[2, 3] = -5.0  # translate camera back

    aspect = 640 / 480
    fov_rad = np.radians(60)
    f = 1.0 / np.tan(fov_rad / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (100 + 0.1) / (0.1 - 100)
    proj[2, 3] = (2 * 100 * 0.1) / (0.1 - 100)
    proj[3, 2] = -1.0

    # Render
    ctx.clear(0.15, 0.15, 0.15)
    prog['u_view'].write(view.tobytes())
    prog['u_projection'].write(proj.tobytes())
    vao.render(moderngl.TRIANGLES, instances=2)

    # Read pixels BEFORE swap
    data = ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3))

    print(f"Image: mean={img.mean():.1f}, min={img.min()}, max={img.max()}")

    # Check left half (should have red) and right half (should have green)
    left = img[:, :320]
    right = img[:, 320:]

    left_non_bg = left.mean(axis=2) > 30
    right_non_bg = right.mean(axis=2) > 30

    print(f"Left non-bg pixels: {left_non_bg.sum()}")
    print(f"Right non-bg pixels: {right_non_bg.sum()}")

    if left_non_bg.sum() > 0:
        left_colors = left[left_non_bg]
        print(f"Left mean color: {left_colors.mean(axis=0)}")
    if right_non_bg.sum() > 0:
        right_colors = right[right_non_bg]
        print(f"Right mean color: {right_colors.mean(axis=0)}")

    glfw.swap_buffers(window)
    vao.release()
    vbo.release()
    ibo.release()
    instance_buf.release()
    color_buf.release()
    temp_buf.release()
    glfw.terminate()

    total_non_bg = left_non_bg.sum() + right_non_bg.sum()
    if total_non_bg > 100:
        print("\nINSTANCED TEST PASSED")
        return 0
    else:
        print("\nINSTANCED TEST FAILED: no geometry visible")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
