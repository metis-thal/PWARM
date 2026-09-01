"""Raw OpenGL instanced rendering test — bypass moderngl VAO abstraction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders

VS = """#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_model_0;
layout(location = 3) in vec4 in_model_1;
layout(location = 4) in vec4 in_model_2;
layout(location = 5) in vec4 in_model_3;
layout(location = 6) in vec3 in_color;
layout(location = 7) in float in_temperature;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec3 v_color;
void main() {
    mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    v_color = in_color;
}"""

FS = """#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
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

    window = glfw.create_window(640, 480, "Raw GL Test", None, None)
    glfw.make_context_current(window)

    print(f"GL: {glGetString(GL_VERSION).decode()}")

    # Compile shaders
    vs = shaders.compileShader(VS, GL_VERTEX_SHADER)
    fs = shaders.compileShader(FS, GL_FRAGMENT_SHADER)
    prog = shaders.compileProgram(vs, fs)
    print(f"Program: {prog}")

    # Get attribute locations
    locs = {}
    for name in ['in_position', 'in_normal', 'in_model_0', 'in_model_1',
                  'in_model_2', 'in_model_3', 'in_color', 'in_temperature']:
        loc = glGetAttribLocation(prog, name)
        locs[name] = loc
        print(f"  {name}: location={loc}")

    # Mesh data
    verts, norms, idx = make_sphere(1.0, 12, 8)
    n_verts = len(verts)

    # Create VBO for vertices (interleaved pos+normal)
    interleaved = np.zeros((n_verts, 6), np.float32)
    interleaved[:, :3] = verts
    interleaved[:, 3:6] = norms

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)

    # Instance data: 2 red spheres
    n_instances = 2
    mat_data = np.zeros((n_instances, 4, 4), np.float32)
    # Red sphere at x=-2, scale=0.5
    mat_data[0] = np.eye(4, dtype=np.float32)
    mat_data[0, 0, 0] = mat_data[0, 1, 1] = mat_data[0, 2, 2] = 0.5
    mat_data[0, 0, 3] = -2.0
    # Green sphere at x=+2, scale=0.5
    mat_data[1] = np.eye(4, dtype=np.float32)
    mat_data[1, 0, 0] = mat_data[1, 1, 1] = mat_data[1, 2, 2] = 0.5
    mat_data[1, 0, 3] = 2.0

    # CRITICAL: mat4(col0,col1,col2,col3) in GLSL builds from COLUMNS
    # numpy stores row-major, so we transpose to get column-major layout
    mat_flat = mat_data.transpose(0, 2, 1).reshape(n_instances, 16).astype(np.float32)
    mat_buf = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, mat_buf)
    glBufferData(GL_ARRAY_BUFFER, mat_flat.nbytes, mat_flat, GL_STATIC_DRAW)

    color_data = np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2]], np.float32)
    color_buf = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_buf)
    glBufferData(GL_ARRAY_BUFFER, color_data.nbytes, color_data, GL_STATIC_DRAW)

    temp_data = np.array([600.0, 250.0], np.float32)
    temp_buf = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, temp_buf)
    glBufferData(GL_ARRAY_BUFFER, temp_data.nbytes, temp_data, GL_STATIC_DRAW)

    # Create VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # EBO MUST be bound while VAO is active — it's part of VAO state
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

    # Bind VBO and set vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # in_position: location 0, 3 floats
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    # in_normal: location 1, 3 floats
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    # Instance matrix: locations 2-5, 4 vec4s each, stride=64
    glBindBuffer(GL_ARRAY_BUFFER, mat_buf)
    for i, loc in enumerate([2, 3, 4, 5]):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(i * 16))
        glVertexAttribDivisor(loc, 1)  # per instance

    # in_color: location 6, 3 floats, stride=12
    glBindBuffer(GL_ARRAY_BUFFER, color_buf)
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glVertexAttribDivisor(6, 1)

    # in_temperature: location 7, 1 float, stride=4
    glBindBuffer(GL_ARRAY_BUFFER, temp_buf)
    glEnableVertexAttribArray(7)
    glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
    glVertexAttribDivisor(7, 1)

    glBindVertexArray(0)

    # Camera
    view = np.eye(4, dtype=np.float32)
    view[2, 3] = -5.0
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
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glUseProgram(prog)
    loc_view = glGetUniformLocation(prog, "u_view")
    loc_proj = glGetUniformLocation(prog, "u_projection")
    glUniformMatrix4fv(loc_view, 1, GL_TRUE, view)
    glUniformMatrix4fv(loc_proj, 1, GL_TRUE, proj)

    glBindVertexArray(vao)
    print(f"Error before draw: {glGetError()}")
    glDrawElementsInstanced(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, None, n_instances)
    err = glGetError()
    print(f"Error after draw: {err}")
    glBindVertexArray(0)

    # Read pixels
    w, h = glfw.get_framebuffer_size(window)
    data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)

    print(f"Image: {w}x{h}, mean={img.mean():.1f}, min={img.min()}, max={img.max()}")
    non_bg = img.mean(axis=2) > 30
    print(f"Non-bg pixels: {non_bg.sum()} / {w*h}")
    if non_bg.sum() > 0:
        print(f"Color mean: {img[non_bg].mean(axis=0)}")
        print(f"Color min: {img[non_bg].min(axis=0)}")
        print(f"Color max: {img[non_bg].max(axis=0)}")

    glfw.swap_buffers(window)
    glfw.terminate()

    if non_bg.sum() > 100:
        print("\nRAW GL INSTANCED TEST PASSED")
        return 0
    else:
        print("\nRAW GL INSTANCED TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
