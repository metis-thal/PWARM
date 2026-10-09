"""Shared test fixtures and legacy-dependency stubs.

``pwarm.kernel`` was removed from the repository (commit 75a9e04) while the
rule modules (``rules.chemistry/thermal/ecology/fracture/materials``), the
2D/3D viewers and ``parallel.ray_parallel`` still import it.  So those
modules can be exercised, this conftest installs a lightweight stand-in that
mirrors the dataclass interface those modules rely on, as recorded at
commit e1a8551.  The stub is installed only when the real package cannot be
imported, so a future restoration of ``pwarm.kernel`` takes precedence.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _install_kernel_stubs() -> None:
    kernel = types.ModuleType("pwarm.kernel")
    bodies = types.ModuleType("pwarm.kernel.bodies")
    collision = types.ModuleType("pwarm.kernel.collision")
    bodies3d = types.ModuleType("pwarm.kernel.bodies3d")
    world = types.ModuleType("pwarm.kernel.world")
    world3d = types.ModuleType("pwarm.kernel.world3d")
    math3d = types.ModuleType("pwarm.kernel.math3d")

    # ------------------------------------------------------------------ 2D
    @dataclass
    class Material:
        restitution: float = 0.3
        friction: float = 0.4
        density: float = 1.0
        specific_heat: float = 1000.0
        thermal_conductivity: float = 0.0
        young_modulus: float = 1e9
        poisson_ratio: float = 0.3
        hardness: float = 1e6
        fracture_toughness: float = 1e6
        brittleness: float = 0.5

    @dataclass
    class Body:
        pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
        vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
        angle: float = 0.0
        ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))
        mass: float = 1.0
        inv_mass: float = field(init=False, default=0.0)
        inertia: float = field(init=False, default=0.0)
        inv_inertia: float = field(init=False, default=0.0)
        radius: float = 0.0
        vertices: list | None = None
        material: Material = field(default_factory=Material)
        static: bool = False
        is_sensor: bool = False
        force: np.ndarray = field(default_factory=lambda: np.zeros(2))
        torque: float = 0.0
        temperature: float = 293.15
        heat: float = 0.0

        def __post_init__(self) -> None:
            self.recompute_inertia()

        def recompute_inertia(self) -> None:
            if self.static or self.mass <= 0:
                self.inv_mass = 0.0
            else:
                self.inv_mass = 1.0 / self.mass
            if self.radius > 0:
                self.inertia = 0.5 * self.mass * self.radius**2
            elif self.vertices is not None and len(self.vertices) > 0:
                rel = np.asarray(self.vertices, dtype=float) - np.asarray(self.pos)
                self.inertia = 0.5 * self.mass * float(np.mean(np.sum(rel * rel, axis=1)))
            else:
                self.inertia = 0.0
            if self.static or self.inertia <= 0:
                self.inv_inertia = 0.0
            else:
                self.inv_inertia = 1.0 / self.inertia

        def is_circle(self) -> bool:
            return self.radius > 0

        def is_polygon(self) -> bool:
            return (self.vertices is not None and len(self.vertices) > 0
                    and not self.is_circle())

        def world_vertices(self) -> np.ndarray | None:
            if self.vertices is None or len(self.vertices) == 0:
                return None
            c, s = np.cos(self.angle), np.sin(self.angle)
            rot = np.array([[c, -s], [s, c]])
            return np.asarray(self.vertices, dtype=float) @ rot.T + np.asarray(self.pos)

        def kinetic_energy(self) -> float:
            return 0.5 * self.mass * float(np.dot(self.vel, self.vel))

    def circle_body(pos, radius, mass=1.0, material=None, static=False):
        return Body(pos=np.asarray(pos, dtype=float), radius=radius, mass=mass,
                    material=material or Material(), static=static)

    @dataclass
    class Contact:
        a: Body
        b: Body
        point: np.ndarray
        normal: np.ndarray
        penetration: float
        restitution: float
        friction: float

    bodies.Material = Material
    bodies.Body = Body
    bodies.circle_body = circle_body
    collision.Contact = Contact

    # ------------------------------------------------------------------ 3D
    class ShapeType(Enum):
        SPHERE = "sphere"
        BOX = "box"
        CYLINDER = "cylinder"

    @dataclass
    class Material3D(Material):
        pass

    @dataclass
    class SphereShape:
        radius: float
        shape_type: ShapeType = ShapeType.SPHERE

    @dataclass
    class BoxShape:
        half_extents: np.ndarray
        shape_type: ShapeType = ShapeType.BOX

    @dataclass
    class CylinderShape:
        radius: float
        half_height: float
        shape_type: ShapeType = ShapeType.CYLINDER

    @dataclass
    class Body3D:
        pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
        vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
        orn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
        ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
        mass: float = 1.0
        static: bool = False
        temperature: float = 293.15
        shape: SphereShape | BoxShape | CylinderShape | None = None
        material: Material3D = field(default_factory=Material3D)

    def sphere_body(pos, radius=0.5, mass=1.0, material=None, static=False):
        return Body3D(pos=np.asarray(pos, dtype=float), mass=mass, static=static,
                      material=material or Material3D(), shape=SphereShape(radius))

    def box_body(pos, half_extents=(0.5, 0.5, 0.5), mass=1.0, material=None,
                 static=False, angle=0.0):
        del angle  # accepted for interface parity; stub keeps identity orientation
        return Body3D(pos=np.asarray(pos, dtype=float), mass=mass, static=static,
                      material=material or Material3D(),
                      shape=BoxShape(np.asarray(half_extents, dtype=float)))

    def cylinder_body(pos, radius=0.5, half_height=0.5, mass=1.0, material=None,
                      static=False):
        return Body3D(pos=np.asarray(pos, dtype=float), mass=mass, static=static,
                      material=material or Material3D(),
                      shape=CylinderShape(radius, half_height))

    bodies3d.Body = Body3D
    bodies3d.Material = Material3D
    bodies3d.ShapeType = ShapeType
    bodies3d.SphereShape = SphereShape
    bodies3d.BoxShape = BoxShape
    bodies3d.CylinderShape = CylinderShape
    bodies3d.sphere_body = sphere_body
    bodies3d.box_body = box_body
    bodies3d.cylinder_body = cylinder_body

    class World:
        pass

    class World3D:
        pass

    world.World = World
    world3d.World3D = World3D

    def quat_to_axis_angle(q) -> tuple[np.ndarray, float]:
        q = np.asarray(q, dtype=float)
        w = float(np.clip(q[0], -1.0, 1.0))
        angle = 2.0 * float(np.arccos(w))
        s = float(np.sqrt(max(1e-12, 1.0 - w * w)))
        axis = q[1:4] / s
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0]), angle
        return axis / norm, angle

    math3d.quat_to_axis_angle = quat_to_axis_angle

    kernel.__path__ = []  # mark as package so submodule imports resolve
    for name, mod in [("bodies", bodies), ("collision", collision),
                      ("bodies3d", bodies3d), ("world", world),
                      ("world3d", world3d), ("math3d", math3d)]:
        sys.modules[f"pwarm.kernel.{name}"] = mod
        setattr(kernel, name, mod)
    sys.modules["pwarm.kernel"] = kernel


try:  # a restored pwarm.kernel always wins over the stub
    import pwarm.kernel  # noqa: F401
except ModuleNotFoundError:
    _install_kernel_stubs()
