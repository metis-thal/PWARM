"""3D physics world with Verlet integration.

Extends the 2D world concept to 3D with proper rigid body dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymo.kernel.math3d import integrate_angular_velocity, quat_normalize


@dataclass
class World3D:
    """A 3D physics scene with rigid body dynamics."""

    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    dt: float = 1 / 60.0
    solver_iterations: int = 10
    substeps: int = 1

    bodies: list = field(default_factory=list)
    t: float = 0.0
    step_count: int = 0
    history: list = field(default_factory=list)

    def add(self, *bodies) -> None:
        self.bodies.extend(bodies)

    def clear(self) -> None:
        self.bodies.clear()
        self.history.clear()
        self.t = 0.0
        self.step_count = 0

    def _integrate(self, h: float) -> None:
        # 1. Apply gravity + accumulated forces to get new velocities (half-step for angular)
        for b in self.bodies:
            if b.static:
                continue
            b.vel += (self.gravity + b.force * b.inv_mass) * h
            b.ang_vel += b.inv_inertia @ b.torque * h

        # 2. Integrate positions and orientations (semi-implicit Euler)
        for b in self.bodies:
            if b.static:
                continue
            b.pos += b.vel * h
            b.orn = quat_normalize(integrate_angular_velocity(b.orn, b.ang_vel, h))

        # 3. Collision detection
        contacts = []
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                # For now, use simplified AABB broad phase
                contacts.extend(self._narrow_phase(self.bodies[i], self.bodies[j]))

        # 4. Resolve contacts (impulse-based)
        if contacts:
            self._solve_contacts(contacts)

        # 5. Reset accumulated forces
        for b in self.bodies:
            b.clear_forces()

    def _narrow_phase(self, a, b):
        """Simple SAT-based collision for now; will be replaced with GJK/EPA."""
        # Quick AABB check first
        aabb_a = self._compute_aabb(a)
        aabb_b = self._compute_aabb(b)
        if not self._aabb_overlap(aabb_a, aabb_b):
            return []
        
        # For now, use simplified collision
        # TODO: Replace with GJK/EPA
        return []

    def _compute_aabb(self, body):
        """Compute axis-aligned bounding box."""
        verts = body.get_world_vertices()
        if len(verts) == 0:
            return np.zeros((2, 3))
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        return np.stack([min_corner, max_corner])

    def _aabb_overlap(self, a, b):
        return np.all(a[0] <= b[1]) and np.all(b[0] <= a[1])

    def _solve_contacts(self, contacts):
        """Simple impulse-based contact resolution."""
        for c in contacts:
            # Placeholder - will implement proper impulse resolution
            pass

    def step(self, n: int = 1) -> None:
        """Advance the world by `n` fixed timesteps."""
        for _ in range(n):
            for _s in range(self.substeps):
                h = self.dt / self.substeps
                self._integrate(h)
            self.step_count += 1
            self.t += self.dt

    # -- observation / energy ----------------------------------------------

    def total_kinetic_energy(self) -> float:
        return float(sum(b.kinetic_energy() for b in self.bodies))

    def total_momentum(self) -> np.ndarray:
        return sum((b.mass * b.vel for b in self.bodies), np.zeros(3))

    def total_angular_momentum(self) -> np.ndarray:
        return sum((b.angular_momentum() for b in self.bodies), np.zeros(3))

    def record(self) -> None:
        """Append a full state snapshot to history (for replay/AI observer)."""
        self.history.append(
            {
                "t": self.t,
                "pos": np.array([b.pos.copy() for b in self.bodies]),
                "vel": np.array([b.vel.copy() for b in self.bodies]),
                "orn": np.array([b.orn.copy() for b in self.bodies]),
                "ang_vel": np.array([b.ang_vel.copy() for b in self.bodies]),
                "ke": self.total_kinetic_energy(),
            }
        )