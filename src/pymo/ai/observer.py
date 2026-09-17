"""World observer: collects time-series data from a simulated World.

Simulates real sensors reading the world state. The observer records per-body
positions and velocities at regular intervals, producing a structured dataset
that the AI layer uses to discover physical laws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymo.kernel.world import World


@dataclass
class Observation:
    """A single observed quantity's time series."""

    name: str                 # e.g. "body0.pos.y", "body0.vel.x"
    t: np.ndarray             # (N,) time samples
    values: np.ndarray        # (N,) values


@dataclass
class TimeSeriesDataset:
    """Collection of observations from a world run."""

    observations: list[Observation] = field(default_factory=list)

    def get(self, name: str) -> np.ndarray | None:
        for obs in self.observations:
            if obs.name == name:
                return obs.values
        return None

    def time(self) -> np.ndarray:
        if not self.observations:
            return np.array([])
        return self.observations[0].t

    def names(self) -> list[str]:
        return [o.name for o in self.observations]


class WorldObserver:
    """Collects time-series data from a World by stepping it and sampling state."""

    def __init__(self, world: World, sample_every: int = 1):
        self.world = world
        self.sample_every = max(1, sample_every)

    def observe(self, n_steps: int) -> TimeSeriesDataset:
        """Run the world for `n_steps`, sampling every `sample_every` steps.

        Records per-body pos (x,y) and vel (x,y) plus total KE/momentum.
        """
        times = []
        pos_x: dict[int, list[float]] = {}
        pos_y: dict[int, list[float]] = {}
        vel_x: dict[int, list[float]] = {}
        vel_y: dict[int, list[float]] = {}
        kes: list[float] = []

        for step in range(n_steps):
            self.world.step()
            if step % self.sample_every == 0:
                times.append(self.world.t)
                kes.append(self.world.total_kinetic_energy())
                for i, b in enumerate(self.world.bodies):
                    pos_x.setdefault(i, []).append(float(b.pos[0]))
                    pos_y.setdefault(i, []).append(float(b.pos[1]))
                    vel_x.setdefault(i, []).append(float(b.vel[0]))
                    vel_y.setdefault(i, []).append(float(b.vel[1]))

        obs = [Observation("time", np.array(times), np.array(times))]
        for i, xs in pos_x.items():
            obs.append(Observation(f"body{i}.pos.x", np.array(times), np.array(xs)))
            obs.append(Observation(f"body{i}.pos.y", np.array(times), np.array(pos_y[i])))
            obs.append(Observation(f"body{i}.vel.x", np.array(times), np.array(vel_x[i])))
            obs.append(Observation(f"body{i}.vel.y", np.array(times), np.array(vel_y[i])))
        obs.append(Observation("total.ke", np.array(times), np.array(kes)))
        return TimeSeriesDataset(obs)


def collect_free_fall(n_steps: int = 300, dt: float = 0.01, g: float = 9.81) -> TimeSeriesDataset:
    """Convenience: observe a single free-falling body and return the dataset."""
    from pymo.kernel.bodies import circle_body
    from pymo.kernel.world import World

    w = World(gravity=np.array([0.0, -g]), dt=dt)
    w.add(circle_body([0.0, 10.0], 0.5, mass=1.0))
    obs = WorldObserver(w).observe(n_steps)
    return obs
