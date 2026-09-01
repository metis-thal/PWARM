"""GLRenderer: ModernGL + glfw renderer with GPU Instancing.

Renders SceneSnapshots at 60fps using:
- GPU Instancing: one DrawCall per mesh type (sphere, box, cylinder)
- GLSL shaders: vertex transform via instance matrix, PBR lighting
- Orbit camera with mouse control
- Frustum culling on CPU before submitting to GPU

Architecture:
  Simulation thread → SceneSnapshot → DoubleBuffer → GLRenderer (this module)
  GLRenderer reads snapshots lock-free and renders at display refresh rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import glfw
import moderngl
import numpy as np

from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    InstanceData,
    MeshType,
    SceneSnapshot,
)
from pymo.geology import GeologySolver, get_material_properties_for_gpu


# ---------------------------------------------------------------------------
# GLSL Shaders
# ---------------------------------------------------------------------------

VERTEX_SHADER = """
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
}
"""

FRAGMENT_SHADER = """
#version 330 core

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
    vec3 V = normalize(u_camera_pos - v_frag_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    float ambient = 0.25;

    float t_norm = clamp((v_temperature - u_temp_min) / (u_temp_max - u_temp_min), 0.0, 1.0);
    vec3 emission = vec3(0.0);
    if (t_norm > 0.01) {
        emission = mix(
            vec3(1.0, 0.4, 0.1),
            vec3(1.0, 0.95, 0.9),
            t_norm
        ) * t_norm * u_emission_strength;
    }

    vec3 base = v_color;
    vec3 color = base * (ambient * u_ambient_color + diff * u_light_color)
               + spec * u_light_color * 0.3
               + emission;

    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
"""

PARTICLE_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;

out vec3 v_color;

void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
    v_color = in_color;
}
"""

PARTICLE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_color;
out vec4 frag_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) discard;

    vec3 color = pow(v_color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 0.8);
}
"""

# ============================================================
# Geology shaders
# ============================================================

GEOLOGY_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

in vec4 in_model_0;
in vec4 in_model_1;
in vec4 in_model_2;
in vec4 in_model_3;

in uint in_rock_id;
in float in_temperature;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_frag_pos;
out uint v_rock_id;
out float v_temperature;

void main() {
    mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;

    mat3 normal_mat = transpose(inverse(mat3(model)));
    v_normal = normalize(normal_mat * in_normal);
    v_frag_pos = world_pos.xyz;
    v_rock_id = in_rock_id;
    v_temperature = in_temperature;
}
"""

GEOLOGY_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;
in uint v_rock_id;
in float v_temperature;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient_color;
uniform vec3 u_camera_pos;
uniform float u_temp_min;
uniform float u_temp_max;
uniform float u_emission_strength;
uniform vec3 u_rock_colors[32];  // max 32 rock types

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_frag_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    float ambient = 0.25;

    // Get rock base color from uniform array
    uint rid = min(v_rock_id, 31u);
    vec3 base = u_rock_colors[rid];

    float t_norm = clamp((v_temperature - u_temp_min) / (u_temp_max - u_temp_min), 0.0, 1.0);
    vec3 emission = vec3(0.0);
    if (t_norm > 0.01) {
        emission = mix(
            vec3(1.0, 0.4, 0.1),
            vec3(1.0, 0.95, 0.9),
            t_norm
        ) * t_norm * u_emission_strength;
    }

    vec3 color = base * (ambient * u_ambient_color + diff * u_light_color)
               + spec * u_light_color * 0.3
               + emission;

    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
"""

# Slice plane shader for cross-section view
SLICE_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

in vec4 in_model_0;
in vec4 in_model_1;
in vec4 in_model_2;
in vec4 in_model_3;

in uint in_rock_id;
in float in_temperature;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_slice_plane;  // (normal.x, normal.y, normal.z, distance)
uniform int u_discard_side;  // 0 = keep positive, 1 = keep negative

out vec3 v_normal;
out vec3 v_frag_pos;
out uint v_rock_id;
out float v_temperature;
out float v_slice_dist;

void main() {
    mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
    vec4 world_pos = model * vec4(in_position, 1.0);
    
    // Compute signed distance to slice plane
    v_slice_dist = dot(world_pos.xyz, u_slice_plane.xyz) - u_slice_plane.w;
    
    // Discard fragments on wrong side of slice
    if (u_discard_side == 0) {
        if (v_slice_dist < 0.0) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // behind camera, will be clipped
        } else {
            gl_Position = u_projection * u_view * world_pos;
        }
    } else {
        if (v_slice_dist > 0.0) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        } else {
            gl_Position = u_projection * u_view * world_pos;
        }
    }

    mat3 normal_mat = transpose(inverse(mat3(model)));
    v_normal = normalize(normal_mat * in_normal);
    v_frag_pos = world_pos.xyz;
    v_rock_id = in_rock_id;
    v_temperature = in_temperature;
}
"""

SLICE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;
in uint v_rock_id;
in float v_temperature;
in float v_slice_dist;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient_color;
uniform vec3 u_camera_pos;
uniform float u_temp_min;
uniform float u_temp_max;
uniform float u_emission_strength;
uniform vec3 u_rock_colors[32];
uniform vec4 u_slice_plane;

out vec4 frag_color;

void main() {
    // Highlight slice plane with a thin line
    float slice_thickness = 0.5;  // meters
    float on_slice = 1.0 - smoothstep(-slice_thickness, slice_thickness, abs(v_slice_dist));
    
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_frag_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    float ambient = 0.25;

    uint rid = min(v_rock_id, 31u);
    vec3 base = u_rock_colors[rid];

    float t_norm = clamp((v_temperature - u_temp_min) / (u_temp_max - u_temp_min), 0.0, 1.0);
    vec3 emission = vec3(0.0);
    if (t_norm > 0.01) {
        emission = mix(
            vec3(1.0, 0.4, 0.1),
            vec3(1.0, 0.95, 0.9),
            t_norm
        ) * t_norm * u_emission_strength;
    }

    vec3 color = base * (ambient * u_ambient_color + diff * u_light_color)
               + spec * u_light_color * 0.3
               + emission;

    // Add slice highlight (bright white line)
    color += on_slice * vec3(1.0, 1.0, 1.0) * 0.5;

    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def _make_sphere_mesh(radius: float = 1.0, sectors: int = 16, stacks: int = 12):
    vertices = []
    normals = []
    indices = []

    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            vertices.append([x * radius, y * radius, z * radius])
            normals.append([x, y, z])

    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


def _make_box_mesh(hx: float = 1.0, hy: float = 1.0, hz: float = 1.0):
    v = np.array([
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        [-hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz], [ hx, -hy, -hz],
        [-hx,  hy, -hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx, -hy,  hz], [-hx, -hy,  hz],
        [ hx, -hy, -hz], [ hx,  hy, -hz], [ hx,  hy,  hz], [ hx, -hy,  hz],
        [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy,  hz], [-hx,  hy, -hz],
    ], dtype=np.float32)

    n = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0,-1], [0, 0,-1], [0, 0,-1], [0, 0,-1],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0,-1, 0], [0,-1, 0], [0,-1, 0], [0,-1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [-1, 0, 0],[-1, 0, 0],[-1, 0, 0],[-1, 0, 0],
    ], dtype=np.float32)

    idx = []
    for face in range(6):
        base = face * 4
        idx.extend([base, base+1, base+2, base, base+2, base+3])

    return v, n, np.array(idx, dtype=np.uint32)


def _make_cylinder_mesh(radius: float = 1.0, half_height: float = 1.0,
                         segments: int = 16):
    vertices = []
    normals = []
    indices = []

    for i in range(segments + 1):
        theta = 2.0 * np.pi * i / segments
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        nx, ny = np.cos(theta), np.sin(theta)
        vertices.append([x, y, -half_height])
        normals.append([nx, ny, 0])
        vertices.append([x, y, half_height])
        normals.append([nx, ny, 0])

    for i in range(segments):
        b = i * 2
        indices.extend([b, b+1, b+2])
        indices.extend([b+1, b+3, b+2])

    vertices.append([0, 0, -half_height])
    normals.append([0, 0, -1])
    bottom_idx = len(vertices) - 1
    for i in range(segments):
        v0 = i * 2
        v1 = ((i + 1) % segments) * 2
        indices.extend([bottom_idx, v1, v0])

    vertices.append([0, 0, half_height])
    normals.append([0, 0, 1])
    top_idx = len(vertices) - 1
    for i in range(segments):
        v0 = i * 2 + 1
        v1 = ((i + 1) % segments) * 2 + 1
        indices.extend([top_idx, v0, v1])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# Frustum culling (Gribb-Hartmann)
# ---------------------------------------------------------------------------

@dataclass
class FrustumPlanes:
    """Six frustum planes extracted from view-projection matrix.
    Each plane is (a, b, c, d) where ax + by + cz + d >= 0 means inside."""

    planes: np.ndarray  # (6, 4) float32

    @classmethod
    def from_view_projection(cls, vp: np.ndarray) -> "FrustumPlanes":
        """Extract 6 frustum planes from combined view-projection matrix (Gribb-Hartmann).

        Given clip = VP * P, the clip-space conditions are:
            -w <= x <= w  →  left:  row0 + row3,  right: row3 - row0
            -w <= y <= w  →  bottom: row1 + row3, top:   row3 - row1
            -w <= z <= w  →  near:  row2 + row3,  far:   row3 - row2
        """
        planes = np.zeros((6, 4), dtype=np.float32)
        # vp[i] is the i-th ROW of the VP matrix (numpy row-major)
        planes[0] = vp[0] + vp[3]  # left
        planes[1] = vp[3] - vp[0]  # right
        planes[2] = vp[1] + vp[3]  # bottom
        planes[3] = vp[3] - vp[1]  # top
        planes[4] = vp[2] + vp[3]  # near
        planes[5] = vp[3] - vp[2]  # far

        for i in range(6):
            n = np.linalg.norm(planes[i, :3])
            if n > 1e-8:
                planes[i] /= n

        return cls(planes=planes)

    def is_aabb_inside(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        """Test if an AABB intersects the frustum. Returns True if visible."""
        for i in range(6):
            p = self.planes[i]
            nx, ny, nz = p[0], p[1], p[2]
            # P-vertex: vertex most in the positive direction of the plane normal
            px = aabb_max[0] if nx >= 0 else aabb_min[0]
            py = aabb_max[1] if ny >= 0 else aabb_min[1]
            pz = aabb_max[2] if nz >= 0 else aabb_min[2]
            if nx * px + ny * py + nz * pz + p[3] < 0:
                return False
        return True


# ---------------------------------------------------------------------------
# MeshGroup
# ---------------------------------------------------------------------------

@dataclass
class MeshGroup:
    mesh_type: MeshType
    vao: moderngl.VertexArray
    instance_buffer: moderngl.Buffer
    color_buffer: moderngl.Buffer
    temp_buffer: moderngl.Buffer
    index_count: int
    instance_count: int = 0


# ---------------------------------------------------------------------------
# RendererConfig
# ---------------------------------------------------------------------------

@dataclass
class RendererConfig:
    window_size: tuple[int, int] = (1280, 720)
    title: str = "PWARM Renderer"
    bg_color: tuple[float, float, float] = (0.05, 0.05, 0.08)
    target_fps: int = 60
    point_size: float = 4.0
    cam_distance: float = 25.0
    cam_elevation: float = np.pi / 4
    cam_fov: float = 60.0
    temp_min: float = 200.0
    temp_max: float = 800.0
    emission_strength: float = 1.5
    frustum_cull: bool = True


# ---------------------------------------------------------------------------
# GLRenderer
# ---------------------------------------------------------------------------

class GLRenderer:
    def __init__(self, buffer: DoubleBuffer, config: RendererConfig | None = None):
        self.buffer = buffer
        self.config = config or RendererConfig()
        self.running = False
        self._last_snapshot: SceneSnapshot | None = None

        self._mouse_left = False
        self._mouse_right = False
        self._mouse_pos = (0.0, 0.0)

        self._ctx: moderngl.Context | None = None
        self._window = None
        self._mesh_groups: dict[MeshType, MeshGroup] = {}
        self._particle_program: moderngl.Program | None = None
        self._particle_vao: moderngl.VertexArray | None = None
        self._particle_buffer: moderngl.Buffer | None = None
        self._particle_color_buffer: moderngl.Buffer | None = None
        self._instanced_program: moderngl.Program | None = None

    def _init_glfw(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        w, h = self.config.window_size
        self._window = glfw.create_window(w, h, self.config.title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_framebuffer_size_callback(self._window, self._on_resize)

    def _init_shaders(self) -> None:
        self._instanced_program = self._ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self._particle_program = self._ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER,
            fragment_shader=PARTICLE_FRAGMENT_SHADER,
        )

    def _init_meshes(self) -> None:
        prog = self._instanced_program
        verts, norms, idx = _make_sphere_mesh(1.0, 16, 12)
        self._init_mesh_group(MeshType.SPHERE, verts, norms, idx, prog)
        verts, norms, idx = _make_box_mesh(1.0, 1.0, 1.0)
        self._init_mesh_group(MeshType.BOX, verts, norms, idx, prog)
        verts, norms, idx = _make_cylinder_mesh(1.0, 1.0, 16)
        self._init_mesh_group(MeshType.CYLINDER, verts, norms, idx, prog)

    def _init_mesh_group(self, mesh_type, vertices, normals, indices, program):
        ctx = self._ctx
        n_verts = len(vertices)
        interleaved = np.zeros((n_verts, 6), dtype=np.float32)
        interleaved[:, :3] = vertices
        interleaved[:, 3:6] = normals

        vbo = ctx.buffer(interleaved.tobytes())
        ibo = ctx.buffer(indices.tobytes())

        max_instances = 4096
        # Initialize with dummy data (Intel GPU drivers may not handle
        # writes to reserved-only buffers correctly via GL_INVALID_OPERATION)
        instance_buf = ctx.buffer(np.zeros((max_instances, 16), dtype=np.float32).tobytes())
        color_buf = ctx.buffer(np.zeros((max_instances, 3), dtype=np.float32).tobytes())
        temp_buf = ctx.buffer(np.zeros(max_instances, dtype=np.float32).tobytes())

        vao = ctx.vertex_array(
            program,
            [
                (vbo, '3f 3f', 'in_position', 'in_normal'),
                (instance_buf, '4f 4f 4f 4f /i', 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
                (color_buf, '3f /i', 'in_color'),
                (temp_buf, '1f /i', 'in_temperature'),
            ],
            ibo,
        )

        self._mesh_groups[mesh_type] = MeshGroup(
            mesh_type=mesh_type,
            vao=vao,
            instance_buffer=instance_buf,
            color_buffer=color_buf,
            temp_buffer=temp_buf,
            index_count=len(indices),
        )

    def _upload_instances(self, snapshot: SceneSnapshot,
                          frustum: FrustumPlanes | None = None) -> None:
        for mg in self._mesh_groups.values():
            mg.instance_count = 0

        grouped: dict[MeshType, list[InstanceData]] = {
            MeshType.SPHERE: [],
            MeshType.BOX: [],
            MeshType.CYLINDER: [],
        }
        for inst in snapshot.instances:
            if frustum is not None and inst.aabb_min is not None and inst.aabb_max is not None:
                if not frustum.is_aabb_inside(inst.aabb_min, inst.aabb_max):
                    continue
            grouped[inst.mesh_type].append(inst)

        ctx = self._ctx

        for mesh_type, instances in grouped.items():
            mg = self._mesh_groups.get(mesh_type)
            if mg is None or len(instances) == 0:
                continue

            n = len(instances)
            mg.instance_count = n

            # GLSL mat4(col0,col1,col2,col3) constructs from COLUMNS,
            # so we must transpose numpy's row-major layout to column-major
            mat_data = np.array(
                [inst.model_matrix.T.flatten() for inst in instances],
                dtype=np.float32,
            )
            if n * 64 > mg.instance_buffer.size:
                mg.instance_buffer.release()
                mg.instance_buffer = ctx.buffer(mat_data.tobytes())
            else:
                mg.instance_buffer.write(mat_data.tobytes())

            color_data = np.array(
                [inst.base_color for inst in instances], dtype=np.float32
            )
            mg.color_buffer.write(color_data.tobytes())

            temp_data = np.array(
                [inst.temperature for inst in instances], dtype=np.float32
            )
            mg.temp_buffer.write(temp_data.tobytes())

        if snapshot.fluid_positions is not None and len(snapshot.fluid_positions) > 0:
            pos_data = snapshot.fluid_positions.astype(np.float32)
            col_data = np.tile(
                snapshot.fluid_color if snapshot.fluid_color is not None else np.array([0.2, 0.5, 1.0], dtype=np.float32),
                (len(pos_data), 1),
            ).astype(np.float32)

            if self._particle_buffer is None or pos_data.nbytes > self._particle_buffer.size:
                if self._particle_buffer is not None:
                    self._particle_buffer.release()
                self._particle_buffer = ctx.buffer(pos_data.tobytes())
            else:
                self._particle_buffer.write(pos_data.tobytes())

            if self._particle_color_buffer is None or col_data.nbytes > self._particle_color_buffer.size:
                if self._particle_color_buffer is not None:
                    self._particle_color_buffer.release()
                self._particle_color_buffer = ctx.buffer(col_data.tobytes())
            else:
                self._particle_color_buffer.write(col_data.tobytes())

            if self._particle_vao is not None:
                self._particle_vao.release()

            self._particle_vao = ctx.vertex_array(
                self._particle_program,
                [
                    (self._particle_buffer, '3f', 'in_position'),
                    (self._particle_color_buffer, '3f', 'in_color'),
                ],
            )

    def _draw_frame(self, snapshot: SceneSnapshot) -> None:
        """Draw the frame to the back buffer WITHOUT swapping. Call _swap_buffers() after."""
        ctx = self._ctx
        w, h = glfw.get_framebuffer_size(self._window)
        if w == 0 or h == 0:
            return

        ctx.viewport = (0, 0, w, h)
        bg = self.config.bg_color
        ctx.clear(bg[0], bg[1], bg[2])

        cam = snapshot.camera
        view = cam.view_matrix()
        proj = cam.projection_matrix(w / h)
        cam_pos = cam.position()

        frustum = None
        if self.config.frustum_cull:
            vp = proj @ view
            frustum = FrustumPlanes.from_view_projection(vp)

        self._upload_instances(snapshot, frustum)

        prog = self._instanced_program
        # numpy row-major → GLSL column-major: transpose before writing
        # (moderngl .write() sends raw bytes, unlike glUniformMatrix4fv's GL_TRUE flag)
        prog['u_view'].write(view.T.tobytes())
        prog['u_projection'].write(proj.T.tobytes())
        prog['u_light_dir'].value = (0.5, 0.8, 1.0)
        prog['u_light_color'].value = (1.0, 0.98, 0.92)
        prog['u_ambient_color'].value = (0.4, 0.45, 0.5)
        prog['u_camera_pos'].value = tuple(cam_pos.tolist())
        prog['u_temp_min'].value = self.config.temp_min
        prog['u_temp_max'].value = self.config.temp_max
        prog['u_emission_strength'].value = self.config.emission_strength

        for mg in self._mesh_groups.values():
            if mg.instance_count > 0:
                mg.vao.render(moderngl.TRIANGLES, instances=mg.instance_count)

        if self._particle_vao is not None and self._particle_buffer is not None:
            pprog = self._particle_program
            pprog['u_view'].write(view.T.tobytes())
            pprog['u_projection'].write(proj.T.tobytes())
            pprog['u_point_size'].value = self.config.point_size
            self._particle_vao.render(moderngl.POINTS)

    def _swap_buffers(self) -> None:
        """Swap front/back buffers (call after _draw_frame + optional pixel read)."""
        glfw.swap_buffers(self._window)

    def _render_frame(self, snapshot: SceneSnapshot) -> None:
        """Draw + swap in one call (used by the main render loop)."""
        self._draw_frame(snapshot)
        self._swap_buffers()

    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = (action == glfw.PRESS)

    def _on_cursor_pos(self, window, xpos, ypos):
        if self._last_snapshot is None:
            return
        dx = xpos - self._mouse_pos[0]
        dy = ypos - self._mouse_pos[1]
        self._mouse_pos = (xpos, ypos)

        cam = self._last_snapshot.camera
        if self._mouse_left:
            cam.azimuth -= dx * 0.005
            cam.elevation = np.clip(cam.elevation + dy * 0.005, 0.05, np.pi - 0.05)
        if self._mouse_right:
            forward = cam.position() - cam.target
            forward = forward / (np.linalg.norm(forward) + 1e-12)
            right = np.cross(forward, np.array([0, 0, 1]))
            right = right / (np.linalg.norm(right) + 1e-12)
            up = np.cross(right, forward)
            scale = cam.distance * 0.002
            cam.target -= right * dx * scale + up * dy * scale

    def _on_scroll(self, window, xoffset, yoffset):
        if self._last_snapshot is not None:
            self._last_snapshot.camera.distance *= (1.0 - yoffset * 0.1)
            self._last_snapshot.camera.distance = np.clip(
                self._last_snapshot.camera.distance, 1.0, 200.0
            )

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            self.running = False
        elif key == glfw.KEY_R:
            if self._last_snapshot is not None:
                self._last_snapshot.camera.distance = self.config.cam_distance
                self._last_snapshot.camera.elevation = self.config.cam_elevation
                self._last_snapshot.camera.azimuth = 0.0
                self._last_snapshot.camera.target = np.zeros(3, dtype=np.float32)

    def _on_resize(self, window, width, height):
        if self._ctx and width > 0 and height > 0:
            self._ctx.viewport = (0, 0, width, height)

    def run(self) -> None:
        self._init_glfw()
        self._init_shaders()
        self._init_meshes()

        self.running = True
        frame_interval = 1.0 / self.config.target_fps

        while self.running and not glfw.window_should_close(self._window):
            t0 = time.perf_counter()
            snap = self.buffer.read()
            if snap is not None:
                self._last_snapshot = snap
                self._render_frame(snap)
            glfw.poll_events()
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._cleanup()

    def _cleanup(self) -> None:
        for mg in self._mesh_groups.values():
            mg.vao.release()
            mg.instance_buffer.release()
            mg.color_buffer.release()
            mg.temp_buffer.release()
        if self._particle_vao is not None:
            self._particle_vao.release()
        if self._particle_buffer is not None:
            self._particle_buffer.release()
        if self._particle_color_buffer is not None:
            self._particle_color_buffer.release()
        glfw.terminate()
