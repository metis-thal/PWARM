"""2D physics world: integrates bodies, applies forces, resolves collisions.

The World is the ground-truth container for a simulated scene. It advances all
bodies each step (integration -> gravity/forces -> collision detection ->
impulse solving) and records state for observation/replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bodies import Body
from .collision import detect_collision
from .solver import solve_contacts


@dataclass
class World:
    """A 2D physics scene.

    `thermal` is an optional rule object (e.g. rules.thermal.BodyThermalSystem)
    used to conduct heat across contacting bodies each step. It is optional so
    the pure-mechanics kernel stays dependency-free; pass it in to enable
    multi-physics coupling.
    """

    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))
    dt: float = 1 / 60.0
    solver_iterations: int = 10
    substeps: int = 1
    thermal: object | None = None   # optional BodyThermalSystem for heat conduction

    bodies: list[Body] = field(default_factory=list)
    t: float = 0.0
    step_count: int = 0
    history: list[dict] = field(default_factory=list)

    def add(self, *bodies: Body) -> None:
        self.bodies.extend(bodies)

    def clear(self) -> None:
        self.bodies.clear()
        self.history.clear()
        self.t = 0.0
        self.step_count = 0

    def step(self, n: int = 1) -> None:
        """Advance the world by `n` fixed timesteps."""
        for _ in range(n):
            for _s in range(self.substeps):
                h = self.dt / self.substeps
                self._integrate(h)
            self.step_count += 1
            self.t += self.dt

    def _integrate(self, h: float) -> None:
        # 1. Apply gravity (constant force) + accumulate forces
        for b in self.bodies:
            if b.static:
                continue
            b.vel = b.vel + (self.gravity + b.force * b.inv_mass) * h

        # 2. Integrate positions (semi-implicit Euler for rigid bodies)
        for b in self.bodies:
            if b.static:
                continue
            b.pos = b.pos + b.vel * h
            b.angle += b.ang_vel * h

        # 3. Collision detection
        contacts = []
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                contacts.extend(detect_collision(self.bodies[i], self.bodies[j]))

        # 4. Resolve contacts (conserves momentum)
        if contacts:
            solve_contacts(contacts, iterations=self.solver_iterations)

        # 5. Thermal coupling: conduct heat across contacting bodies
        if self.thermal is not None and contacts:
            self.thermal.step_contacts(self.bodies, contacts, h)

        # 6. Reset accumulated forces
        for b in self.bodies:
            b.clear_forces()

    # -- observation / energy ----------------------------------------------

    def total_kinetic_energy(self) -> float:
        return float(sum(b.kinetic_energy() for b in self.bodies))

    def total_momentum(self) -> np.ndarray:
        return sum((b.mass * b.vel for b in self.bodies), np.zeros(2))

    def record(self) -> None:
        """Append a full state snapshot to history (for replay/AI observer)."""
        self.history.append(
            {
                "t": self.t,
                "pos": np.array([b.pos.copy() for b in self.bodies]),
                "vel": np.array([b.vel.copy() for b in self.bodies]),
                "angle": np.array([b.angle for b in self.bodies]),
                "ang_vel": np.array([b.ang_vel for b in self.bodies]),
                "ke": self.total_kinetic_energy(),
            }
        )
