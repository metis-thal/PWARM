"""Test: does creating a buffer with actual data vs reserve make a difference?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import glfw
import moderngl

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


def make_model(pos, scale=1.0):
    m = np.eye(4, dtype=np.float32)
    m[0,0] = m[1,1] = m[2,2] = scale
    m[:3, 3] = pos
    return m


def try_approach(label, ctx, prog, verts, norms, idx, instances_data):
    """Try a specific buffer creation approach."""
    print(f"\n{'='*60}")
    print(f"APPROACH: {label}")
    print(f"{'='*60}")

    n_verts = len(verts)
    interleaved = np.zeros((n_verts, 6), np.float32)
    interleaved[:, :3] = verts
    interleaved[:, 3:6] = norms

    vbo = ctx.buffer(interleaved.tobytes())
    ibo = ctx.buffer(idx.tobytes())

    n = len(instances_data)
    mat_data = np.array([inst['mat'].flatten() for inst in instances_data], np.float32)
    color_data = np.array([inst['color'] for inst in instances_data], np.float32)
    temp_data = np.array([inst['temp'] for inst in instances_data], np.float32)

    # Create instance buffers with ACTUAL data (not reserve)
    instance_buf = ctx.buffer(mat_data.tobytes())
    color_buf = ctx.buffer(color_data.tobytes())
    temp_buf = ctx.buffer(temp_data.tobytes())

    print(f"  instance_buf size: {instance_buf.size}, data: {mat_data.nbytes}")
    print(f"  color_buf size: {color_buf.size}, data: {color_data.nbytes}")
    print(f"  temp_buf size: {temp_buf.size}, data: {temp_data.nbytes}")

    try:
        vao = ctx.vertex_array(prog, [
            (vbo, '3f 3f', 'in_position', 'in_normal'),
            (instance_buf, '4f 4f 4f 4f /i', 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
            (color_buf, '3f /i', 'in_color'),
            (temp_buf, '1f /i', 'in_temperature'),
        ], ibo)
        print(f"  VAO created OK")
    except Exception as e:
        print(f"  VAO FAILED: {e}")
        vbo.release(); ibo.release()
        instance_buf.release(); color_buf.release(); temp_buf.release()
        return False

    # Render
    ctx.clear(0.1, 0.1, 0.1)
    print(f"  Error before draw: {ctx.error}")
    try:
        vao.render(moderngl.TRIANGLES, instances=n)
        print(f"  Error after draw: {ctx.error}")
    except Exception as e:
        print(f"  RENDER FAILED: {e}")
        vao.release(); vbo.release(); ibo.release()
        instance_buf.release(); color_buf.release(); temp_buf.release()
        return False

    # Read pixels
    import glfw
    w, h = glfw.get_framebuffer_size(ctx._window_info._window if hasattr(ctx, '_window_info') else glfw.get_current_context())
    w, h = glfw.get_framebuffer_size(glfw.get_current_context())
    data = ctx.fbo.read(components=3)
    img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3))

    non_bg = img.mean(axis=2) > 30
    n_pixels = non_bg.sum()
    print(f"  Non-bg pixels: {n_pixels} / {w*h}")

    if n_pixels > 0:
        colors = img[non_bg]
        print(f"  Color mean: {colors.mean(axis=0)}")
        print(f"  Color min: {colors.min(axis=0)}")
        print(f"  Color max: {colors.max(axis=0)}")

    # Cleanup
    vao.release(); vbo.release(); ibo.release()
    instance_buf.release(); color_buf.release(); temp_buf.release()

    return n_pixels > 100


def main():
    if not glfw.init(): return 1
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(640, 480, "Buffer Test", None, None)
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    print(f"GL: {ctx.info['GL_VERSION']}")

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    verts, norms, idx = make_sphere(1.0, 12, 8)

    instances = [
        {'mat': make_model([-2, 0, 0], 0.5), 'color': np.array([1.0, 0.2, 0.2], np.float32), 'temp': 600.0},
        {'mat': make_model([2, 0, 0], 0.5), 'color': np.array([0.2, 1.0, 0.2], np.float32), 'temp': 250.0},
    ]

    results = []
    results.append(try_approach("data buffers (no reserve)", ctx, prog, verts, norms, idx, instances))

    glfw.swap_buffers(window)

    # Test 2: same but with depth test disabled
    ctx.disable(moderngl.DEPTH_TEST)
    results.append(try_approach("data buffers, no depth test", ctx, prog, verts, norms, idx, instances))
    ctx.enable(moderngl.DEPTH_TEST)

    # Test 3: reserve + write
    def try_reserve(label):
        print(f"\n{'='*60}")
        print(f"APPROACH: {label}")
        print(f"{'='*60}")
        n_verts = len(verts)
        interleaved = np.zeros((n_verts, 6), np.float32)
        interleaved[:, :3] = verts
        interleaved[:, 3:6] = norms
        vbo = ctx.buffer(interleaved.tobytes())
        ibo = ctx.buffer(idx.tobytes())

        n = len(instances)
        mat_data = np.array([inst['mat'].flatten() for inst in instances], np.float32)
        color_data = np.array([inst['color'] for inst in instances], np.float32)
        temp_data = np.array([inst['temp'] for inst in instances], np.float32)

        # reserve + write
        instance_buf = ctx.buffer(reserve=4096 * 64)
        instance_buf.write(mat_data.tobytes())
        color_buf = ctx.buffer(reserve=4096 * 12)
        color_buf.write(color_data.tobytes())
        temp_buf = ctx.buffer(reserve=4096 * 4)
        temp_buf.write(temp_data.tobytes())

        try:
            vao = ctx.vertex_array(prog, [
                (vbo, '3f 3f', 'in_position', 'in_normal'),
                (instance_buf, '4f 4f 4f 4f /i', 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
                (color_buf, '3f /i', 'in_color'),
                (temp_buf, '1f /i', 'in_temperature'),
            ], ibo)
            print(f"  VAO OK")
        except Exception as e:
            print(f"  VAO FAILED: {e}")
            return False

        ctx.clear(0.1, 0.1, 0.1)
        print(f"  Error before: {ctx.error}")
        vao.render(moderngl.TRIANGLES, instances=n)
        print(f"  Error after: {ctx.error}")

        w, h = glfw.get_framebuffer_size(glfw.get_current_context())
        data = ctx.fbo.read(components=3)
        img = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3))
        non_bg = img.mean(axis=2) > 30
        n_px = non_bg.sum()
        print(f"  Non-bg pixels: {n_px}")
        if n_px > 0:
            colors = img[non_bg]
            print(f"  Color mean: {colors.mean(axis=0)}")

        vao.release(); vbo.release(); ibo.release()
        instance_buf.release(); color_buf.release(); temp_buf.release()
        return n_px > 100

    results.append(try_reserve("reserve + write"))

    glfw.terminate()

    print(f"\n{'='*60}")
    print(f"RESULTS: {results}")
    if any(results):
        print("AT LEAST ONE APPROACH WORKS")
        return 0
    else:
        print("ALL APPROACHES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main() or 0)
