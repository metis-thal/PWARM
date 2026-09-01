"""Minimal OpenGL test: draw a single colored triangle to verify the pipeline."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import glfw
import moderngl

def main():
    if not glfw.init():
        print("FAILED: glfw.init()")
        return 1

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(640, 480, "Minimal Test", None, None)
    if not window:
        print("FAILED: create_window")
        glfw.terminate()
        return 1

    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    print(f"OpenGL version: {ctx.version_code}")
    print(f"GLSL version: {ctx.info['GL_VERSION']}")

    # Simple shader: red triangle
    prog = ctx.program(
        vertex_shader="""
        #version 330 core
        in vec3 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            gl_Position = vec4(in_pos, 1.0);
            v_color = in_color;
        }
        """,
        fragment_shader="""
        #version 330 core
        in vec3 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = vec4(v_color, 1.0);
        }
        """,
    )

    # Triangle vertices
    vertices = np.array([
        # x, y, z, r, g, b
        -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
         0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
         0.0,  0.5, 0.0,  0.0, 0.0, 1.0,
    ], dtype=np.float32)

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_color')])

    ctx.clear(0.1, 0.1, 0.1)
    vao.render(moderngl.TRIANGLES)

    # Read pixels
    data = ctx.fbo.read(components=3)
    img = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
    img = np.flipud(img)

    print(f"Image mean: {img.mean():.1f}, min: {img.min()}, max: {img.max()}")

    # Check for red/green/blue pixels
    center = img[200:280, 260:380]
    print(f"Center region mean: {center.mean(axis=0)}")

    # Check for non-dark pixels
    non_dark = img.mean(axis=2) > 30
    print(f"Non-dark pixels: {non_dark.sum()} / {640*480}")

    glfw.swap_buffers(window)
    vao.release()
    vbo.release()
    prog.release()
    glfw.terminate()

    if non_dark.sum() > 100:
        print("MINIMAL TEST PASSED: geometry renders correctly")
        return 0
    else:
        print("MINIMAL TEST FAILED: no geometry visible")
        return 1

if __name__ == "__main__":
    sys.exit(main() or 0)
