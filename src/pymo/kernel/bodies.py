"""Rigid body representation for the pymo 2D physics kernel.

A Body carries its kinematic state (position, velocity, angle, angular
velocity), dynamic properties (mass, inertia) and a collision shape. Bodies are
the ground-truth entities that the rules layer and AI observer read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Material:
    """Physical material properties used for contact response, thermal rules,
    and fracture mechanics."""

    restitution: float = 0.3       # bounciness 0..1 (lower = settles better)
    friction: float = 0.4          # Coulomb friction coefficient
    density: float = 1.0           # mass per unit area
    # Thermal properties (used by rules/thermal.py)
    specific_heat: float = 1000.0   # J/(kg*K) — energy to raise 1 kg by 1 K
    thermal_conductivity: float = 0.0  # W/(m*K) — 0 = insulator (no conduction)
    # Mechanical properties (used by rules/fracture.py)
    young_modulus: float = 1e9     # Pa — stiffness (elasticity)
    poisson_ratio: float = 0.3     # dimensionless — lateral contraction
    hardness: float = 1e6          # Pa — resistance to permanent deformation
    fracture_toughness: float = 1e6   # Pa*sqrt(m) — resistance to crack propagation
    brittleness: float = 0.5       # 0=ductile, 1=brittle — fracture tendency


@dataclass
class Body:
    """A rigid body with position/velocity (linear + angular) and a shape.

    Shape is either a circle (radius) or a convex polygon (vertices in local
    coordinates, counter-clockwise).
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    angle: float = 0.0
    ang_vel: float = 0.0

    mass: float = 1.0
    inv_mass: float = field(init=False, default=0.0)
    inertia: float = field(init=False, default=0.0)
    inv_inertia: float = field(init=False, default=0.0)

    radius: float = 0.0          # if > 0, circle shape
    vertices: np.ndarray | None = None  # (N, 2) local polygon vertices
    material: Material = field(default_factory=Material)
    static: bool = False         # static bodies have infinite mass (immovable)
    is_sensor: bool = False      # sensors report overlaps but produce no impulse

    # Accumulated force for the current step (linear + torque)
    force: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    torque: float = 0.0

    # Thermal state (used by rules/thermal.py); internal energy J
    temperature: float = 293.15   # Kelvin
    heat: float = 0.0             # accumulated thermal energy J

    def __post_init__(self) -> None:
        self.recompute_inertia()

    def recompute_inertia(self) -> None:
        """Set inverse mass / inertia from shape and density."""
        if self.static:
            self.inv_mass = 0.0
            self.inv_inertia = 0.0
            self.inertia = np.inf
            return
        self.inv_mass = 1.0 / self.mass if self.mass > 0 else 0.0
        if self.radius > 0:
            # Solid circle: I = 0.5 * m * r^2
            self.inertia = 0.5 * self.mass * self.radius**2
        elif self.vertices is not None:
            self.inertia = polygon_inertia(self.vertices, self.mass)
        else:
            self.inertia = 1.0
        self.inv_inertia = 1.0 / self.inertia if self.inertia > 0 else 0.0

    # -- shape helpers ------------------------------------------------------

    def world_vertices(self) -> np.ndarray:
        """Return polygon vertices transformed to world coordinates."""
        if self.vertices is None:
            return np.empty((0, 2))
        c, s = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([[c, -s], [s, c]])
        return self.vertices @ rot.T + self.pos

    def is_circle(self) -> bool:
        return self.radius > 0

    def is_polygon(self) -> bool:
        return self.vertices is not None

    def apply_force(self, f: np.ndarray) -> None:
        """Apply a force at the center of mass (no torque)."""
        self.force = self.force + np.asarray(f, dtype=float)

    def apply_force_at(self, f: np.ndarray, point: np.ndarray) -> None:
        """Apply a force at a world point (produces torque about COM)."""
        f = np.asarray(f, dtype=float)
        self.force = self.force + f
        r = np.asarray(point, dtype=float) - self.pos
        self.torque += cross(r, f)

    def clear_forces(self) -> None:
        self.force[:] = 0.0
        self.torque = 0.0

    def kinetic_energy(self) -> float:
        """Translational + rotational kinetic energy."""
        return 0.5 * self.mass * float(np.dot(self.vel, self.vel)) + 0.5 * self.inertia * self.ang_vel**2


def cross(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (scalar z-component)."""
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_inertia(vertices: np.ndarray, mass: float) -> float:
    """Moment of inertia of a convex polygon about its centroid.

    Uses the standard polygonal inertia formula.
    """
    verts = np.asarray(vertices, dtype=float)
    n = len(verts)
    area = 0.0
    ixx = iyy = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        ixx += (y0 * y0 + y0 * y1 + y1 * y1) * cross
        iyy += (x0 * x0 + x0 * x1 + x1 * x1) * cross
    if abs(area) < 1e-12:
        return 1.0
    area *= 0.5
    # Density = mass / area
    density = mass / abs(area)
    # Inertia about origin; we assume vertices are centered at centroid.
    return density * (ixx + iyy) / 12.0


def circle_body(
    pos, radius: float, mass: float = 1.0, material: Material | None = None, static: bool = False
) -> Body:
    """Convenience factory for a circle body."""
    b = Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        radius=radius,
        material=material or Material(),
        static=static,
    )
    return b


def box_body(
    pos,
    half_w: float,
    half_h: float,
    mass: float = 1.0,
    material: Material | None = None,
    static: bool = False,
    angle: float = 0.0,
) -> Body:
    """Convenience factory for an axis-aligned box body (convex quad)."""
    hw, hh = half_w, half_h
    verts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    return Body(
        pos=np.asarray(pos, dtype=float),
        mass=mass,
        vertices=verts,
        material=material or Material(),
        static=static,
        angle=angle,
    )
