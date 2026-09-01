"""SceneSnapshot: immutable world state snapshot for decoupled rendering.

The rendering thread reads snapshots (read-only) while the simulation thread
writes new snapshots. A DoubleBuffer ensures the renderer always sees a
consistent, complete frame — never a half-updated state.

Architecture (Unity-inspired):
  Simulation loop  →  generates SceneSnapshot  →  writes to DoubleBuffer
  Render loop      →  reads from DoubleBuffer  →  groups by mesh type → GPU Instancing

All data in a SceneSnapshot is immutable after creation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MeshType(str, Enum):
    """Supported renderable mesh types."""
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"


@dataclass
class CameraState:
    """Orbit camera state — spherical coordinates.

    position = target + distance * (sin(elev)*cos(azim), sin(elev)*sin(azim), cos(elev))
    """
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    distance: float = 20.0
    elevation: float = np.pi / 4   # radians from horizontal
    azimuth: float = 0.0           # radians around vertical
    fov: float = 60.0              # degrees
    near: float = 0.1
    far: float = 500.0

    def position(self) -> np.ndarray:
        """Compute camera world position from spherical coordinates."""
        se = np.sin(self.elevation)
        ce = np.cos(self.elevation)
        sa = np.sin(self.azimuth)
        ca = np.cos(self.azimuth)
        return self.target + self.distance * np.array(
            [se * ca, se * sa, ce], dtype=np.float32
        )

    def view_matrix(self) -> np.ndarray:
        """Compute 4×4 look-at view matrix."""
        eye = self.position()
        center = self.target
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        f = center - eye
        f = f / (np.linalg.norm(f) + 1e-12)
        s = np.cross(f, up)
        s = s / (np.linalg.norm(s) + 1e-12)
        u = np.cross(s, f)

        vm = np.eye(4, dtype=np.float32)
        vm[0, :3] = s
        vm[1, :3] = u
        vm[2, :3] = -f
        vm[0, 3] = -float(np.dot(s, eye))
        vm[1, 3] = -float(np.dot(u, eye))
        vm[2, 3] = float(np.dot(f, eye))
        return vm

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Compute 4×4 perspective projection matrix."""
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        pm = np.zeros((4, 4), dtype=np.float32)
        pm[0, 0] = f / aspect
        pm[1, 1] = f
        pm[2, 2] = (self.far + self.near) / (self.near - self.far)
        pm[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        pm[3, 2] = -1.0
        return pm


@dataclass
class InstanceData:
    """One renderable instance — position/rotation/shape from simulation.

    The model_matrix is a 4×4 float32 matrix encoding position + rotation + scale.
    It is computed from the Body3D state and is immutable after snapshot creation.
    """
    model_matrix: np.ndarray          # (4, 4) float32
    mesh_type: MeshType
    temperature: float = 293.15       # K — for shader color mapping
    base_color: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.7, 0.8], dtype=np.float32)
    )
    # AABB for frustum culling (min_corner, max_corner in world space)
    aabb_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    aabb_max: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))


@dataclass
class SceneSnapshot:
    """Immutable snapshot of the entire world state for rendering.

    Created by the simulation thread, consumed by the render thread.
    After creation, NO field should be modified — this is the contract
    that makes double-buffering safe without locks on the read side.
    """
    # Time
    t: float = 0.0
    step: int = 0

    # Rigid body instances (grouped by mesh_type for GPU Instancing)
    instances: list[InstanceData] = field(default_factory=list)

    # SPH fluid particles — flat array for point rendering
    # SPH is 2D (x, y) — pad z=0 for 3D rendering
    fluid_positions: np.ndarray | None = None   # (N, 3) float32
    fluid_color: np.ndarray | None = None       # (3,) float32

    # Camera
    camera: CameraState = field(default_factory=CameraState)

    # Global scalars for HUD
    kinetic_energy: float = 0.0
    thermal_energy: float = 0.0
    total_energy: float = 0.0
    total_mass: float = 0.0
    environment_temp: float = 293.15
    solar_flux: float = 0.0

    # Chemistry state
    chemistry_masses: dict[str, float] = field(default_factory=dict)

    # Geology state (Phase 1: stratigraphy + thermal)
    geology_grid: "GeologyGrid | None" = None
    geology_cell_size: float = 0.0
    geology_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geology_slice_axis: int = 2  # 0=YZ, 1=XZ, 2=XY (horizontal)
    geology_slice_index: int = -1  # which slice to render (-1 = auto middle)
    geology_rock_colors: np.ndarray | None = None  # (N_rocks, 3) for GPU
    geology_material_props: dict | None = None  # erosion, thermal, etc for GPU


def _quat_to_matrix(orn: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = orn
    # Normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/n, x/n, y/n, z/n

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def build_model_matrix(pos: np.ndarray, orn: np.ndarray,
                        scale: np.ndarray | None = None) -> np.ndarray:
    """Build 4×4 model matrix from position, quaternion rotation, and optional scale."""
    R = _quat_to_matrix(orn)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    if scale is not None:
        m[0, 0] *= scale[0]
        m[1, 1] *= scale[1]
        m[2, 2] *= scale[2]
    m[:3, 3] = pos.astype(np.float32)
    return m


def compute_aabb_sphere(radius: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AABB for a sphere: center ± radius."""
    r = np.float32(radius)
    return (pos - r).astype(np.float32), (pos + r).astype(np.float32)


def compute_aabb_box(half_extents: np.ndarray, pos: np.ndarray,
                     orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AABB for a rotated box: compute all 8 corners, take min/max."""
    R = _quat_to_matrix(orn)
    hx, hy, hz = half_extents
    # 8 corners in local space
    corners_local = np.array([
        [ hx,  hy,  hz], [ hx,  hy, -hz],
        [ hx, -hy,  hz], [ hx, -hy, -hz],
        [-hx,  hy,  hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [-hx, -hy, -hz],
    ], dtype=np.float32)
    corners_world = corners_local @ R.T + pos.astype(np.float32)
    return corners_world.min(axis=0), corners_world.max(axis=0)


def build_snapshot_from_world(engine: Any, camera: CameraState | None = None) -> SceneSnapshot:
    """Build a SceneSnapshot from a WorldEngine (or World3D).

    This is called by the simulation thread after each batch of ticks.
    It reads the authoritative body state and produces an immutable snapshot.
    """
    from pymo.kernel.bodies3d import SphereShape, BoxShape, CylinderShape

    world = engine.world if hasattr(engine, 'world') else engine
    instances: list[InstanceData] = []

    for i, body in enumerate(world.bodies):
        # Determine mesh type and compute AABB
        shape = body.shape
        if isinstance(shape, SphereShape):
            mesh_type = MeshType.SPHERE
            r = shape.radius
            aabb_min, aabb_max = compute_aabb_sphere(r, body.pos)
            # Unit sphere mesh has radius=1; scale to actual radius
            scale = np.array([r, r, r], dtype=np.float32)
        elif isinstance(shape, BoxShape):
            mesh_type = MeshType.BOX
            hx, hy, hz = shape.half_extents
            aabb_min, aabb_max = compute_aabb_box(shape.half_extents, body.pos, body.orn)
            # Unit box mesh spans [-1,1]; scale by half_extents
            scale = np.array([hx, hy, hz], dtype=np.float32)
        elif isinstance(shape, CylinderShape):
            mesh_type = MeshType.CYLINDER
            r = shape.radius
            hh = shape.half_height
            # Conservative AABB for cylinder
            aabb_min = body.pos - np.array([r, r, hh], dtype=np.float32)
            aabb_max = body.pos + np.array([r, r, hh], dtype=np.float32)
            # Unit cylinder mesh has radius=1, half_height=1; scale to actual dims
            scale = np.array([r, r, hh], dtype=np.float32)
        else:
            continue

        model = build_model_matrix(body.pos, body.orn, scale)

        # Base color from temperature (will be overridden by PBR shader later)
        # For now: gray with temperature indicator
        temp = body.temperature
        instances.append(InstanceData(
            model_matrix=model,
            mesh_type=mesh_type,
            temperature=temp,
            base_color=np.array([0.6, 0.7, 0.8], dtype=np.float32),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        ))

    # SPH fluid particles
    fluid_pos = None
    fluid_col = None
    if hasattr(engine, 'sph') and engine.sph is not None and len(engine.sph.particles) > 0:
        # SPH is 2D (x, y) — pad z=0 for 3D rendering
        pos_2d = np.array([p.pos for p in engine.sph.particles], dtype=np.float32)
        if pos_2d.ndim == 2 and pos_2d.shape[1] == 2:
            fluid_pos = np.column_stack([pos_2d, np.zeros(len(pos_2d), dtype=np.float32)])
        else:
            fluid_pos = pos_2d
        fluid_col = np.array([0.2, 0.5, 1.0], dtype=np.float32)  # water blue

    # Camera
    if camera is None:
        camera = CameraState()

    # Chemistry
    chem_masses = {}
    if hasattr(engine, 'chemistry') and engine.chemistry is not None:
        chem_masses = {k: float(v) for k, v in engine.chemistry.masses.items()}

    # Geology (Phase 1)
    geology_grid = None
    geology_cell_size = 0.0
    geology_origin = (0.0, 0.0, 0.0)
    geology_slice_axis = 2
    geology_slice_index = -1
    geology_rock_colors = None
    geology_material_props = None

    if hasattr(engine, 'enable_geology') and engine.enable_geology and engine.geology_solver is not None:
        solver = engine.geology_solver
        geology_grid = solver.grid
        geology_cell_size = solver.grid_config.cell_size
        geology_origin = solver.grid_config.origin
        # Default to horizontal slice at middle
        geology_slice_axis = 2  # XY plane
        geology_slice_index = solver.grid_config.nz // 2
        geology_rock_colors = solver.get_material_properties_for_gpu()["colors"]
        geology_material_props = solver.get_material_properties_for_gpu()

    return SceneSnapshot(
        t=world.t,
        step=world.step_count,
        instances=instances,
        fluid_positions=fluid_pos,
        fluid_color=fluid_col,
        camera=camera,
        kinetic_energy=world.total_kinetic_energy() if hasattr(world, 'total_kinetic_energy') else 0.0,
        thermal_energy=sum(
            (b.mass * b.material.specific_heat * b.temperature)
            for b in world.bodies
        ) if world.bodies else 0.0,
        total_energy=0.0,  # computed below if engine
        total_mass=sum(b.mass for b in world.bodies),
        environment_temp=getattr(engine, 'environment_temp', 293.15),
        solar_flux=getattr(engine, 'solar_flux', 0.0),
        chemistry_masses=chem_masses,
        geology_grid=geology_grid,
        geology_cell_size=geology_cell_size,
        geology_origin=geology_origin,
        geology_slice_axis=geology_slice_axis,
        geology_slice_index=geology_slice_index,
        geology_rock_colors=geology_rock_colors,
        geology_material_props=geology_material_props,
    )


# ---------------------------------------------------------------------------
# DoubleBuffer: lock-free read, locked write
# ---------------------------------------------------------------------------

class DoubleBuffer:
    """Thread-safe double buffer for scene snapshots.

    The simulation thread calls `write(snapshot)` to publish a new frame.
    The render thread calls `read()` to get the latest complete snapshot.

    Implementation uses a simple pointer swap with a write lock.
    Reads are lock-free — they always see the last completed snapshot.
    """

    def __init__(self) -> None:
        self._front: SceneSnapshot | None = None
        self._back: SceneSnapshot | None = None
        self._write_lock = threading.Lock()
        self._dirty = False

    def write(self, snapshot: SceneSnapshot) -> None:
        """Publish a new snapshot (called by simulation thread)."""
        with self._write_lock:
            self._back = snapshot
            self._front = self._back
            self._dirty = True

    def read(self) -> SceneSnapshot | None:
        """Read the latest snapshot (called by render thread, lock-free)."""
        return self._front

    def is_dirty(self) -> bool:
        """Check if a new snapshot has been published since last read."""
        val = self._dirty
        return val

    def mark_clean(self) -> None:
        """Mark the buffer as clean (called by render thread after reading)."""
        self._dirty = False

    @property
    def has_snapshot(self) -> bool:
        return self._front is not None
