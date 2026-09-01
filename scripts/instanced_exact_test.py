"""Exact reproduction of GLRenderer's instanced rendering with the same shaders."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import glfw
import moderngl

# Same vertex shader as gl_renderer.py
VS = """#version 330 core
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
out vec3 v_normal;
out vec3 v_frag_pos;
out vec3 v_color;
out float v_temperature;
void main() {
    mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    mat3 normal_mat = transpose(inverse(mat3(model)));
    v_normal = normalize(normal_mat * in_normal);
    v_frag_pos = world_pos.xyz;
    v_color = in_color;
    v_temperature = in_temperature;
}"""

FS = """#version 330 core
in vec3 v_normal;
in vec3 v_frag_pos;
in vec3 v_color;
in float v_temperature;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient_color;
uniform vec3 u_camera_pos;
uniform float u_temp_min;
uniform float u_temp_max;
uniform float u_emission_strength;
out vec4 frag_color;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    float diff = max(dot(N, L), 0.0);
    float ambient = 0.25;
    vec3 color = v_color * (ambient * u_ambient_color + diff * u_light_color);
    color = pow(color, vec3(1.0/2.2));
    frag_color = vec4(color, 1.0);
}"""


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


def main():
    if not glfw.init(): return 1
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(640, 480, "Instanced Debug", None, None)
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    print(f"GL: {ctx.info['GL_VERSION']}")

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    # Interleaved vertex+normal buffer (EXACT same as GLRenderer)
    verts, norms, idx = make_sphere(1.0, 16, 12)
    n_verts = len(verts)
    interleaved = np.zeros((n_verts, 6), np.float32)
    interleaved[:, :3] = verts
    interleaved[:, 3:6] = norms

    vbo = ctx.buffer(interleaved.tobytes())
    ibo = ctx.buffer(idx.tobytes())

    # Instance buffers (EXACT same layout as GLRenderer)
    max_inst = 4096
    instance_buf = ctx.buffer(reserve=max_inst * 64)
    color_buf = ctx.buffer(reserve=max_inst * 12)
    temp_buf = ctx.buffer(reserve=max_inst * 4)

    # VAO creation (EXACT same format strings as GLRenderer)
    print("Creating VAO...")
    try:
        vao = ctx.vertex_array(prog, [
            (vbo, '3f 3f', 'in_position', 'in_normal'),
            (instance_buf, '4f 4f 4f 4f /i', 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
            (color_buf, '3f /i', 'in_color'),
            (temp_buf, '1f /i', 'in_temperature'),
        ], ibo)
        print(f"VAO created: {vao}")
    except Exception as e:
        print(f"VAO creation FAILED: {e}")
        return 1

    # Upload 1 instance: red sphere at origin, scale 0.5
    m = np.eye(4, np.float32)
    m[0,0] = m[1,1] = m[2,2] = 0.5
    instance_buf.write(m.tobytes())
    color_buf.write(np.array([1.0, 0.2, 0.2], np.float32).tobytes())
    temp_buf.write(np.array([600.0], np.float32).tobytes())

    # Camera
    view = np.eye(4, np.float32)
    view[2, 3] = -5.0
    aspect = 640/480
    fov_rad = np.radians(60)
    f = 1.0 / np.tan(fov_rad / 2)
    proj = np.zeros((4, 4), np.float32)
    proj[0,0] = f/aspect
    proj[1,1] = f
    proj[2,2] = (100+0.1)/(0.1-100)
    proj[2,3] = (2*100*0.1)/(0.1-100)
    proj[3,2] = -1.0

    # Render
    print("Rendering...")
    ctx.clear(0.15, 0.15, 0.15)
    prog['u_view'].write(view.tobytes())
    prog['u_projection'].write(proj.tobytes())
    prog['u_light_dir'].value = (0.5, 0.8, 1.0)
    prog['u_light_color'].value = (1.0, 0.98, 0.92)
    prog['u_ambient_color'].value = (0.4, 0.45, 0.5)
    prog['u_camera_pos'].value = (0.0, 0.0, 5.0)
    prog['u_temp_min'].value = 200.0
    prog['u_temp_max'].value = 800.0
    prog['u_emission_strength'].value = 1.5

    print(f"Error before draw: {ctx.error}")
    vao.render(moderngl.TRIANGLES, instances=1)
    print(f"Error after draw: {ctx.error}")

    # Read pixels
    data = ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3))
    print(f"Image: mean={img.mean():.1f}, min={img.min()}, max={img.max()}")

    non_bg = img.mean(axis=2) > 30
    print(f"Non-bg pixels: {non_bg.sum()} / {640*480}")

    if non_bg.sum() > 0:
        colors = img[non_bg]
        print(f"Color mean: {colors.mean(axis=0)}")
        print(f"Color min: {colors.min(axis=0)}")
        print(f"Color max: {colors.max(axis=0)}")

    glfw.swap_buffers(window)
    vao.release()
    glfw.terminate()

    if non_bg.sum() > 100:
        print("\nINSTANCED RENDER TEST PASSED")
        return 0
    else:
        print("\nINSTANCED RENDER TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
