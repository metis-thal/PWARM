"""3D physics world with Verlet integration.

Extends the 2D world concept to 3D with proper rigid body dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm

from pymo.kernel.collision3d import Contact3D, detect_collision, detect_all_collisions
from pymo.kernel.math3d import cross3, integrate_angular_velocity, quat_normalize


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
        # 1. Apply gravity + accumulated forces to get new velocities
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

        # 3. Collision detection: AABB broad phase + GJK/EPA narrow phase
        contacts = self._detect_collisions()

        # 4. Resolve contacts (impulse-based)
        if contacts:
            self._solve_contacts(contacts)

        # 5. Reset accumulated forces
        for b in self.bodies:
            b.clear_forces()

    # -- broad phase ----------------------------------------------------------

    def _compute_aabb(self, body):
        """Compute axis-aligned bounding box (inflated for spheres)."""
        # For spheres, use center ± radius to get proper AABB
        if body.is_sphere():
            r = body.shape.radius
            center = body.pos
            return np.stack([center - r, center + r])
        verts = body.get_world_vertices()
        if len(verts) == 0:
            return np.zeros((2, 3))
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        return np.stack([min_corner, max_corner])

    def _aabb_overlap(self, a, b):
        return np.all(a[0] <= b[1]) and np.all(b[0] <= a[1])

    def _detect_collisions(self) -> list[Contact3D]:
        """AABB broad phase + GJK/EPA narrow phase."""
        # Broad phase
        aabbs = [(self._compute_aabb(b), i) for i, b in enumerate(self.bodies)]
        candidate_pairs = []
        for i in range(len(aabbs)):
            for j in range(i + 1, len(aabbs)):
                if self._aabb_overlap(aabbs[i][0], aabbs[j][0]):
                    candidate_pairs.append((aabbs[i][1], aabbs[j][1]))

        # Narrow phase: GJK/EPA for each candidate pair
        contacts = []
        for i, j in candidate_pairs:
            contacts.extend(detect_collision(self.bodies[i], self.bodies[j]))
        return contacts

    # -- contact resolution ---------------------------------------------------

    def _solve_contacts(self, contacts: list[Contact3D]) -> None:
        """Sequential impulse solver with positional correction (Baumgarte)."""
        baumgarte = 0.05

        for _ in range(self.solver_iterations):
            for c in contacts:
                self._apply_impulse(c)

        for c in contacts:
            self._positional_correction(c, baumgarte)

    def _apply_impulse(self, c: Contact3D) -> None:
        a, b = c.a, c.b
        n = c.normal
        if a.inv_mass == 0 and b.inv_mass == 0:
            return

        ra = c.point - a.pos
        rb = c.point - b.pos

        # Relative velocity at contact point (3D)
        va = a.vel + cross3(a.ang_vel, ra)
        vb = b.vel + cross3(b.ang_vel, rb)
        rv = vb - va

        # Normal component
        vn = float(np.dot(rv, n))
        if vn > 0.0:
            return  # separating

        # Effective mass: 1/m_eff = 1/m_a + 1/m_b + (r_a×n)·(I_a⁻¹·(r_a×n)) + (r_b×n)·(I_b⁻¹·(r_b×n))
        raxn = cross3(ra, n)
        rbxn = cross3(rb, n)
        inv_mass_eff = a.inv_mass + b.inv_mass
        if not a.static:
            inv_mass_eff += float(raxn @ a.inv_inertia @ raxn)
        if not b.static:
            inv_mass_eff += float(rbxn @ b.inv_inertia @ rbxn)

        if inv_mass_eff < 1e-12:
            return

        e = c.restitution
        j = -(1.0 + e) * vn / inv_mass_eff
        impulse = j * n

        # Apply equal-and-opposite impulses (conserves momentum)
        if not a.static:
            a.vel -= a.inv_mass * impulse
            a.ang_vel -= a.inv_inertia @ cross3(ra, impulse)
        if not b.static:
            b.vel += b.inv_mass * impulse
            b.ang_vel += b.inv_inertia @ cross3(rb, impulse)

        # Friction (tangential)
        tangent = rv - vn * n
        tnorm = norm(tangent)
        if tnorm > 1e-12:
            tangent = tangent / tnorm
            vt = float(np.dot(rv, tangent))
            # Effective mass for tangential
            raxt = cross3(ra, tangent)
            rbxt = cross3(rb, tangent)
            inv_mass_tan = a.inv_mass + b.inv_mass
            if not a.static:
                inv_mass_tan += float(raxt @ a.inv_inertia @ raxt)
            if not b.static:
                inv_mass_tan += float(rbxt @ b.inv_inertia @ rbxt)
            jt = -vt / inv_mass_tan
            max_friction = c.friction * abs(j)
            jt = np.clip(jt, -max_friction, max_friction)
            friction_impulse = jt * tangent
            if not a.static:
                a.vel -= a.inv_mass * friction_impulse
                a.ang_vel -= a.inv_inertia @ cross3(ra, friction_impulse)
            if not b.static:
                b.vel += b.inv_mass * friction_impulse
                b.ang_vel += b.inv_inertia @ cross3(rb, friction_impulse)

    def _positional_correction(self, c: Contact3D, baumgarte: float) -> None:
        """Push overlapping bodies apart along the contact normal."""
        a, b = c.a, c.b
        total_inv = a.inv_mass + b.inv_mass
        if total_inv < 1e-12 or c.penetration <= 0.0:
            return
        slop = 0.005
        correction = max(c.penetration - slop, 0.0) / total_inv * baumgarte
        corr = correction * c.normal
        if not a.static:
            a.pos -= a.inv_mass * corr
        if not b.static:
            b.pos += b.inv_mass * corr

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