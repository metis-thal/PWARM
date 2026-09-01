"""Minimal test: does moderngl's /i divisor format string work on this Intel GPU?"""

import sys, os
import numpy as np
import glfw
import moderngl

# Minimal shader: one per-vertex attr, one per-instance attr
VS = """#version 330 core
in vec3 in_pos;
in vec3 in_color;
uniform float u_time;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    v_color = in_color;
}"""
FS = """#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}"""


def main():
    if not glfw.init(): return 1
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(200, 200, "Divisor Test", None, None)
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    print(f"GL: {ctx.info['GL_VERSION']}")

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    # 2 triangles forming a quad (4 verts, 6 indices)
    verts = np.array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.5,  0.5, 0.0,
        -0.5,  0.5, 0.0,
    ], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    vbo = ctx.buffer(verts.tobytes())
    ibo = ctx.buffer(idx.tobytes())

    # Instance data: 2 instances with different colors
    # Each instance: 3 floats (color), stride = 12
    color_data = np.array([
        [1.0, 0.2, 0.2],   # red
        [0.2, 0.2, 1.0],   # blue
    ], dtype=np.float32)
    instance_buf = ctx.buffer(color_data.tobytes())

    print(f"instance_buf size: {instance_buf.size}")

    # Approach 1: /i divisor
    print("\n--- Approach 1: moderngl /i divisor ---")
    try:
        vao = ctx.vertex_array(prog, [
            (vbo, '3f', 'in_pos'),
            (instance_buf, '3f /i', 'in_color'),
        ], ibo)
        print(f"  VAO created OK")

        ctx.clear(0.1, 0.1, 0.1)
        print(f"  Error before draw: {ctx.error}")
        vao.render(moderngl.TRIANGLES, instances=2)
        print(f"  Error after draw: {ctx.error}")

        data = ctx.fbo.read(components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(200, 200, 3)
        img = np.flipud(img)
        non_bg = img.mean(axis=2) > 30
        print(f"  Non-bg pixels: {non_bg.sum()}")
        vao.release()
    except Exception as e:
        print(f"  FAILED: {e}")

    # Approach 2: /i with multiple per-vertex + multiple per-instance
    print("\n--- Approach 2: interleaved vertex + instanced ---")
    try:
        # Interleaved pos+normal for vertices
        verts2 = np.array([
            # pos           # normal
            -0.5, -0.5, 0.0,  0, 0, 1,
             0.5, -0.5, 0.0,  0, 0, 1,
             0.5,  0.5, 0.0,  0, 0, 1,
            -0.5,  0.5, 0.0,  0, 0, 1,
        ], dtype=np.float32)
        vbo2 = ctx.buffer(verts2.tobytes())

        # 2 instance colors + temps
        inst_color = np.array([
            [1.0, 0.2, 0.2],
            [0.2, 0.2, 1.0],
        ], dtype=np.float32)
        inst_buf2 = ctx.buffer(inst_color.tobytes())

        # Shader with multiple attrs
        vs2 = """#version 330 core
        in vec3 in_pos;
        in vec3 in_normal;
        in vec3 in_color;
        uniform float u_time;
        out vec3 v_color;
        void main() {
            gl_Position = vec4(in_pos, 1.0);
            v_color = in_color;
        }"""
        prog2 = ctx.program(vertex_shader=vs2, fragment_shader=FS)

        vao2 = ctx.vertex_array(prog2, [
            (vbo2, '3f 3f', 'in_pos', 'in_normal'),
            (inst_buf2, '3f /i', 'in_color'),
        ], ibo)
        print(f"  VAO created OK")

        ctx.clear(0.1, 0.1, 0.1)
        print(f"  Error before draw: {ctx.error}")
        vao2.render(moderngl.TRIANGLES, instances=2)
        print(f"  Error after draw: {ctx.error}")

        data = ctx.fbo.read(components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(200, 200, 3)
        img = np.flipud(img)
        non_bg = img.mean(axis=2) > 30
        print(f"  Non-bg pixels: {non_bg.sum()}")
        vao2.release()
    except Exception as e:
        print(f"  FAILED: {e}")

    glfw.terminate()
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
