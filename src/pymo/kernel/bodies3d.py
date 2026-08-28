"""Rigid body representation for the pymo physics kernel (2D and 3D).

A Body carries its kinematic state (position, velocity, orientation, angular
velocity), dynamic properties (mass, inertia tensor) and a collision shape.
Bodies are the ground-truth entities that the rules layer and AI observer read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.linalg import det, inv, norm

from pymo.kernel.math3d import (
    cross3,
    mat3_from_quat,
    quat_from_axis_angle,
    quat_identity,
)


class ShapeType(Enum):
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    CONVEX_HULL = "convex_hull"


@dataclass
class Material:
    """Physical material properties used for contact response, thermal rules,
    and fracture mechanics."""

    restitution: float = 0.3       # bounciness 0..1 (lower = settles better)
    friction: float = 0.4          # Coulomb friction coefficient
    density: float = 1000.0        # kg/m^3
    # Thermal properties (used by rules/thermal.py)
    specific_heat: float = 1000.0   # J/(kg*K) — energy to raise 1 kg by 1 K
    thermal_conductivity: float = 0.0  # W/(m*K) — 0 = insulator (no conduction)
    # Mechanical properties (used by rules/fracture.py)
    young_modulus: float = 1e9     # Pa — stiffness (elasticity)
    poisson_ratio: float = 0.3     # dimensionless — lateral contraction
    hardness: float = 1e6          # Pa — resistance to permanent deformation
    fracture_toughness: float = 1e6   # Pa*sqrt(m) — resistance to crack propagation
    brittleness: float = 0.5       # 0=ductile, 1=brittle — fracture tendency


class Shape:
    """Base class for collision shapes."""

    shape_type: ShapeType

    def compute_mass_properties(self, density: float) -> tuple[float, np.ndarray]:
        """Compute mass and inertia tensor from density.
        Returns (mass, inertia_tensor_3x3).
        """
        raise NotImplementedError

    def support_point(self, direction: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        """Find the point on the shape furthest in the given direction.
        Returns world-space coordinates.
        """
        raise NotImplementedError

    def get_vertices(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        """Get world-space vertices for rendering."""
        raise NotImplementedError


@dataclass
class SphereShape(Shape):
    radius: float

    shape_type: ShapeType = ShapeType.SPHERE

    def compute_mass_properties(self, density: float) -> tuple[float, np.ndarray]:
        r = self.radius
        volume = 4.0 / 3.0 * np.pi * r**3
        mass = density * volume
        I = 2.0 / 5.0 * mass * self.radius**2
        inertia = np.eye(3) * I
        return mass, inertia

    def support_point(self, direction: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        dir_norm = norm(direction)
        if dir_norm < 1e-12:
            return pos
        local_dir = direction / dir_norm
        return pos + local_dir * self.radius

    def get_vertices(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        # Return a few points on the sphere for rendering
        return np.array([pos])


@dataclass
class BoxShape(Shape):
    half_extents: np.ndarray  # (3,) half-width, half-height, half-depth

    shape_type: ShapeType = ShapeType.BOX

    def __post_init__(self):
        self.half_extents = np.asarray(self.half_extents, dtype=float)

    def compute_mass_properties(self, density: float) -> tuple[float, np.ndarray]:
        hx, hy, hz = self.half_extents
        volume = 8.0 * hx * hy * hz
        mass = density * volume
        # Inertia tensor for box about center
        Ixx = mass / 3.0 * (hy**2 + hz**2)
        Iyy = mass / 3.0 * (hx**2 + hz**2)
        Izz = mass / 3.0 * (hx**2 + hy**2)
        inertia = np.diag([Ixx, Iyy, Izz])
        return mass, inertia

    def support_point(self, direction: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        # Transform direction to local space
        R = mat3_from_quat(orn)
        local_dir = R.T @ direction
        # For box, support point is at corner with max dot product
        signs = np.sign(local_dir)
        local_point = self.half_extents * signs
        return pos + R @ local_point

    def get_vertices(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        R = mat3_from_quat(orn)
        hx, hy, hz = self.half_extents
        local_verts = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]
        ])
        return (R @ local_verts.T).T + pos


@dataclass
class CylinderShape(Shape):
    radius: float
    half_height: float

    shape_type: ShapeType = ShapeType.CYLINDER

    def compute_mass_properties(self, density: float) -> tuple[float, np.ndarray]:
        r = self.radius
        h = self.half_height * 2
        volume = np.pi * r**2 * h
        mass = density * volume
        Ixx = mass * (3 * self.radius**2 + h**2) / 12.0
        Iyy = mass * (3 * self.radius**2 + h**2) / 12.0
        Izz = mass * self.radius**2 / 2.0
        inertia = np.diag([Ixx, Iyy, Izz])
        return mass, inertia

    def support_point(self, direction: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        R = mat3_from_quat(orn)
        local_dir = R.T @ direction
        # Cylinder support: handle end caps and side
        dir_norm = norm(direction)
        if dir_norm < 1e-12:
            return pos
        local_dir = direction / dir_norm
        
        # Check if direction is more aligned with cylinder axis (z)
        if abs(local_dir[2]) > 0.5:
            # End cap
            z = self.half_height * np.sign(local_dir[2])
            # Project to circle edge
            xy = local_dir[:2]
            xy_norm = norm(xy)
            if xy_norm > 1e-12:
                xy = xy / xy_norm * self.radius
            else:
                xy = np.array([self.radius, 0.0])
            local_point = np.array([xy[0], xy[1], z])
        else:
            # Side
            xy = local_dir[:2]
            xy_norm = norm(xy)
            if xy_norm > 1e-12:
                xy = xy / xy_norm * self.radius
            else:
                xy = np.array([self.radius, 0.0])
            local_point = np.array([xy[0], xy[1], np.clip(local_dir[2], -self.half_height, self.half_height)])
        
        return pos + mat3_from_quat(orn) @ local_point

    def get_vertices(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        R = mat3_from_quat(orn)
        # Generate cylinder vertices
        n_segments = 16
        angles = np.linspace(0, 2*np.pi, n_segments)
        verts = []
        for z in [-self.half_height, self.half_height]:
            for a in angles:
                x = self.radius * np.cos(a)
                y = self.radius * np.sin(a)
                verts.append([x, y, z])
        local_verts = np.array(verts)
        return (R @ local_verts.T).T + pos


@dataclass
class ConvexHullShape(Shape):
    """Convex hull defined by local vertices and faces."""
    vertices: np.ndarray  # (N, 3) local vertices
    faces: np.ndarray     # (M, 3) triangle face indices (CCW)

    shape_type: ShapeType = ShapeType.CONVEX_HULL

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=int)

    def compute_mass_properties(self, density: float) -> tuple[float, np.ndarray]:
        # Compute volume and center of mass using tetrahedron decomposition
        faces = self.faces
        
        mass = 0.0
        com = np.zeros(3)
        inertia_local = np.zeros((3, 3))
        
        for face in faces:
            v0, v1, v2 = self.vertices[face]
            # Tetrahedron from face to origin
            mat = np.column_stack([v0, v1, v2])
            vol = det(mat) / 6.0
            if vol <= 0:
                continue
            tet_mass = density * vol
            mass += tet_mass
            # Center of mass of tetrahedron
            tet_com = (v0 + v1 + v2) / 4.0
            com += tet_mass * tet_com
            
            # Inertia of tetrahedron about origin
            r = tet_com
            r2 = r[0]**2 + r[1]**2 + r[2]**2
            I = tet_mass * (r2 * np.eye(3) - np.outer(r, r))
            inertia_local += I

        if mass <= 0:
            return 1.0, np.eye(3)
        
        com /= mass
        # Shift inertia to COM
        r = com
        r2 = r[0]**2 + r[1]**2 + r[2]**2
        inertia_com = inertia_local - mass * (r2 * np.eye(3) - np.outer(r, r))
        
        return mass, inertia_com

    def support_point(self, direction: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        R = mat3_from_quat(orn)
        local_dir = R.T @ direction
        # Find vertex with max dot product
        dots = self.vertices @ local_dir
        idx = np.argmax(dots)
        return pos + R @ self.vertices[idx]

    def get_vertices(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        R = mat3_from_quat(orn)
        return (R @ self.vertices.T).T + pos


@dataclass
class Body:
    """A rigid body with position/velocity (linear + angular) and a shape.
    
    Supports 3D shapes: sphere, box, cylinder, convex hull.
    """

    # Kinematic state
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    orn: np.ndarray = field(default_factory=quat_identity)  # quaternion (w, x, y, z)
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    # Dynamic properties
    mass: float = 1.0
    inv_mass: float = field(init=False, default=0.0)
    inertia: np.ndarray = field(init=False, default_factory=lambda: np.eye(3))
    inv_inertia: np.ndarray = field(init=False, default_factory=lambda: np.eye(3))

    # Shape
    shape: Shape = field(default_factory=lambda: SphereShape(radius=0.5))
    material: Material = field(default_factory=Material)
    static: bool = False
    is_sensor: bool = False

    # Accumulated forces/torques
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    # Thermal state
    temperature: float = 293.15
    heat: float = 0.0

    def __post_init__(self) -> None:
        self.recompute_mass_properties()

    def recompute_mass_properties(self) -> None:
        """Set inverse mass / inertia from shape and material density."""
        if self.static:
            self.inv_mass = 0.0
            self.inv_inertia = np.zeros((3, 3))
            # Use a large but finite inertia for static bodies to avoid NaN
            self.inertia = np.eye(3) * 1e12
            return
        
        self.inv_mass = 1.0 / self.mass if self.mass > 0 else 0.0
        
        if hasattr(self.shape, 'compute_mass_properties'):
            mass, inertia = self.shape.compute_mass_properties(self.material.density)
            self.mass = mass
            self.inertia = inertia
            self.inv_inertia = inv(inertia)
        else:
            # Fallback
            self.inertia = np.eye(3)
            self.inv_inertia = np.eye(3)

    # -- shape helpers ------------------------------------------------------

    def world_support_point(self, direction: np.ndarray) -> np.ndarray:
        """Find the point on the shape furthest in the given world direction."""
        return self.shape.support_point(direction, self.pos, self.orn)

    def get_world_vertices(self) -> np.ndarray:
        """Return shape vertices transformed to world coordinates."""
        return self.shape.get_vertices(self.pos, self.orn)

    def is_sphere(self) -> bool:
        return self.shape.shape_type == ShapeType.SPHERE

    def is_box(self) -> bool:
        return self.shape.shape_type == ShapeType.BOX

    def is_cylinder(self) -> bool:
        return self.shape.shape_type == ShapeType.CYLINDER

    def is_convex_hull(self) -> bool:
        return self.shape.shape_type == ShapeType.CONVEX_HULL

    def apply_force(self, f: np.ndarray) -> None:
        """Apply a force at the center of mass (no torque)."""
        self.force += np.asarray(f, dtype=float)

    def apply_force_at(self, f: np.ndarray, point: np.ndarray) -> None:
        """Apply a force at a world point (produces torque about COM)."""
        f = np.asarray(f, dtype=float)
        self.force += f
        r = np.asarray(point, dtype=float) - self.pos
        self.torque += cross3(r, f)

    def apply_torque(self, tau: np.ndarray) -> None:
        self.torque += np.asarray(tau, dtype=float)

    def clear_forces(self) -> None:
        self.force[:] = 0.0
        self.torque[:] = 0.0

    def kinetic_energy(self) -> float:
        """Translational + rotational kinetic energy."""
        return (0.5 * self.mass * float(np.dot(self.vel, self.vel)) + 
                0.5 * float(self.ang_vel @ self.inertia @ self.ang_vel))

    def angular_momentum(self) -> np.ndarray:
        return self.inertia @ self.ang_vel


# Factory functions ---------------------------------------------------------

def sphere_body(
    pos, radius: float = 0.5, mass: float = 1.0, 
    material: Material | None = None, static: bool = False
) -> Body:
    return Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        shape=SphereShape(radius=radius),
        material=material or Material(),
        static=static,
    )


def box_body(
    pos,
    half_extents: np.ndarray,
    mass: float = 1.0,
    material: Material | None = None,
    static: bool = False,
    angle: float = 0.0,
) -> Body:
    return Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        shape=BoxShape(half_extents=half_extents),
        material=material or Material(),
        static=static,
        orn=quat_from_axis_angle(np.array([0, 0, 1]), angle) if angle != 0 else quat_identity(),
    )


def cylinder_body(
    pos,
    radius: float,
    half_height: float,
    mass: float = 1.0,
    material: Material | None = None,
    static: bool = False,
    axis: np.ndarray | None = None,
    angle: float = 0.0,
) -> Body:
    if axis is None:
        axis = np.array([0, 0, 1])
    orn = quat_identity()
    if angle != 0:
        orn = quat_from_axis_angle(axis, angle)
    return Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        shape=CylinderShape(radius=radius, half_height=half_height),
        material=material or Material(),
        static=static,
        orn=orn,
    )


def convex_hull_body(
    pos,
    vertices: np.ndarray,
    faces: np.ndarray,
    mass: float = 1.0,
    material: Material | None = None,
    static: bool = False,
) -> Body:
    return Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        shape=ConvexHullShape(vertices=vertices, faces=faces),
        material=material or Material(),
        static=static,
    )