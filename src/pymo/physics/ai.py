"""
AI Layer for New Physics Architecture — Observer, Law Discovery, Closed-Loop Experimentation.

Works with pymo.physics.WorldEngine (not the old kernel.World).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymo.physics import WorldEngine


@dataclass
class Observation:
    """A single observed quantity's time series."""
    name: str
    t: np.ndarray
    values: np.ndarray


@dataclass
class TimeSeriesDataset:
    """Collection of observations from a simulation run."""
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
    """
    Collects time-series data from a WorldEngine by stepping it and sampling state.
    
    The observer reads from the ground-truth physics engine (WorldEngine) —
    it never influences the simulation, only records.
    """
    
    def __init__(self, engine: WorldEngine, sample_every: int = 1):
        self.engine = engine
        self.sample_every = max(1, sample_every)

    def observe(self, n_steps: int) -> TimeSeriesDataset:
        """Run the engine for `n_steps`, sampling every `sample_every` steps."""
        times = []
        rigid_pos: dict[int, list[np.ndarray]] = {}
        rigid_vel: dict[int, list[np.ndarray]] = {}
        rigid_angvel: dict[int, list[np.ndarray]] = {}
        total_ke: list[float] = []
        total_momentum: list[np.ndarray] = []
        
        for step in range(n_steps):
            state = self.engine.tick()
            if step % self.sample_every == 0:
                times.append(state.t)
                
                # Global quantities
                if state.global_quantities:
                    total_ke.append(state.global_quantities.total_kinetic_energy)
                    total_momentum.append(state.global_quantities.total_linear_momentum.copy())
                
                # Per-rigid-body data
                if state.rigid_pos is not None and len(state.rigid_pos) > 0:
                    for i in range(len(state.rigid_pos)):
                        rigid_pos.setdefault(i, []).append(state.rigid_pos[i].copy())
                        if state.rigid_linvel is not None:
                            rigid_vel.setdefault(i, []).append(state.rigid_linvel[i].copy())
                        if state.rigid_angvel is not None:
                            rigid_angvel.setdefault(i, []).append(state.rigid_angvel[i].copy())
                
                # SPH fluid data (aggregate)
                if state.sph_pos is not None and len(state.sph_pos) > 0:
                    pass  # Could record fluid center of mass, etc.
        
        obs = [Observation("time", np.array(times), np.array(times))]
        
        for i, xs in rigid_pos.items():
            xs_arr = np.array(xs)  # (T, 3)
            obs.append(Observation(f"rigid{i}.pos.x", np.array(times), xs_arr[:, 0]))
            obs.append(Observation(f"rigid{i}.pos.y", np.array(times), xs_arr[:, 1]))
            obs.append(Observation(f"rigid{i}.pos.z", np.array(times), xs_arr[:, 2]))
        
        for i, xs in rigid_vel.items():
            xs_arr = np.array(xs)
            obs.append(Observation(f"rigid{i}.vel.x", np.array(times), xs_arr[:, 0]))
            obs.append(Observation(f"rigid{i}.vel.y", np.array(times), xs_arr[:, 1]))
            obs.append(Observation(f"rigid{i}.vel.z", np.array(times), xs_arr[:, 2]))
        
        for i, xs in rigid_angvel.items():
            xs_arr = np.array(xs)
            obs.append(Observation(f"rigid{i}.angvel.x", np.array(times), xs_arr[:, 0]))
            obs.append(Observation(f"rigid{i}.angvel.y", np.array(times), xs_arr[:, 1]))
            obs.append(Observation(f"rigid{i}.angvel.z", np.array(times), xs_arr[:, 2]))
        
        if total_ke:
            obs.append(Observation("total.ke", np.array(times), np.array(total_ke)))
        
        if total_momentum:
            mom_arr = np.array(total_momentum)  # (T, 3)
            obs.append(Observation("total.momentum.x", np.array(times), mom_arr[:, 0]))
            obs.append(Observation("total.momentum.y", np.array(times), mom_arr[:, 1]))
            obs.append(Observation("total.momentum.z", np.array(times), mom_arr[:, 2]))
        
        return TimeSeriesDataset(obs)


class WorldObserver3D(WorldObserver):
    """Extended observer for 3D physics (rigid + SPH + FEM + etc.)."""
    
    def observe(self, n_steps: int) -> TimeSeriesDataset:
        base = super().observe(n_steps)
        
        # Add SPH observations
        base.time()
        for step in range(n_steps):
            if step % self.sample_every == 0:
                state = self.engine.get_state()
                if state and state.sph_pos is not None and len(state.sph_pos) > 0:
                    # SPH center of mass
                    np.mean(state.sph_pos, axis=0)
                    # Add to observations
        
        return base


def create_free_fall_experiment(engine: WorldEngine, height: float = 10.0, 
                                mass: float = 1.0, n_steps: int = 300) -> TimeSeriesDataset:
    """Convenience: observe a single free-falling body."""
    engine.create_rigid_body((0, 0, height), mass=mass, shape='sphere', shape_params={'radius': 0.5})
    engine.finalize_setup()
    observer = WorldObserver(engine, sample_every=1)
    return observer.observe(n_steps)


def create_collision_experiment(engine: WorldEngine, 
                                pos1: tuple = (-2, 0, 0), vel1: tuple = (2, 0, 0),
                                pos2: tuple = (2, 0, 0), vel2: tuple = (-2, 0, 0),
                                n_steps: int = 300) -> TimeSeriesDataset:
    """Create a two-body collision experiment."""
    engine.create_rigid_body(pos1, mass=1.0, shape='sphere', shape_params={'radius': 0.5})
    engine.create_rigid_body(pos2, mass=1.0, shape='sphere', shape_params={'radius': 0.5})
    engine.finalize_setup()
    
    # Set initial velocities via the write buffer
    state = engine.scene.double_buffer_write
    if state is not None and state.rigid_linvel is not None:
        state.rigid_linvel[0] = np.array(vel1, dtype=np.float32)
        state.rigid_linvel[1] = np.array(vel2, dtype=np.float32)
    
    observer = WorldObserver(engine, sample_every=1)
    return observer.observe(n_steps)