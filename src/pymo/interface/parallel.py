"""
Parallel Environments — Heterogeneous parallel simulation (Ray/MPS).

Vectorized step across multiple environments with different configurations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..physics.core.scene import Scene
    from ..physics.core.state import State


@dataclass
class EnvConfig:
    """Configuration for a single environment."""
    seed: int = 0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 1.0 / 60.0
    substeps: int = 1
    # Solver enables
    enable_rigid: bool = True
    enable_sph: bool = False
    enable_fem: bool = False
    enable_mpm: bool = False
    enable_pbd: bool = False
    enable_thermal: bool = False
    enable_chemistry: bool = False
    enable_geology: bool = False
    # Coupling
    rigid_sph: bool = True
    rigid_fem: bool = True
    thermal_all: bool = False
    # Custom params
    custom: dict = None


class ParallelEnv:
    """
    Heterogeneous parallel environments.
    
    Uses Ray for distributed, multiprocessing for local.
    Each environment can have different solver configurations.
    """
    
    def __init__(self, 
                 num_envs: int,
                 scene_factory: Callable[[EnvConfig], Scene],
                 configs: list[EnvConfig] | None = None,
                 backend: str = "multiprocessing"):  # "ray" or "multiprocessing"
        self.num_envs = num_envs
        self.scene_factory = scene_factory
        self.backend = backend
        
        if configs is None:
            configs = [EnvConfig(seed=i) for i in range(num_envs)]
        elif len(configs) != num_envs:
            raise ValueError(f"Expected {num_envs} configs, got {len(configs)}")
        self.configs = configs
        
        self._scenes: list[Scene] = []
        self._workers = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Create all environments."""
        if self.backend == "ray":
            self._init_ray()
        else:
            self._init_multiprocessing()
        self._initialized = True
    
    def _init_ray(self) -> None:
        """Initialize with Ray."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            @ray.remote
            class EnvWorker:
                def __init__(self, factory, config):
                    self.scene = factory(config)
                
                def step(self):
                    return self.scene.step()
                
                def reset(self):
                    self.scene = self.factory(self.config)
                    return self.scene.get_render_snapshot()
                
                def get_state(self):
                    return self.scene.double_buffer_read
                
                def set_state(self, state):
                    self.scene.double_buffer_write = state
                    self.scene.double_buffer_read = state
            
            self._workers = [
                EnvWorker.remote(self.scene_factory, config) 
                for config in self.configs
            ]
        except ImportError:
            print("Ray not available, falling back to multiprocessing")
            self.backend = "multiprocessing"
            self._init_multiprocessing()
    
    def _init_multiprocessing(self) -> None:
        """Initialize with multiprocessing."""
        from multiprocessing import Process, Queue
        
        self._queues = []
        self._processes = []
        
        for config in self.configs:
            in_queue = Queue()
            out_queue = Queue()
            self._queues.append((in_queue, out_queue))
            
            proc = Process(target=self._worker_loop, args=(in_queue, out_queue, config))
            proc.start()
            self._processes.append(proc)
    
    @staticmethod
    def _worker_loop(in_queue, out_queue, config: EnvConfig):
        """Worker process loop."""
        # Import here to avoid pickling issues
        scene = None
        
        while True:
            cmd, data = in_queue.get()
            if cmd == "init":
                scene = data(config)
                out_queue.put(("ready", None))
            elif cmd == "step":
                state = scene.step()
                out_queue.put(("state", state))
            elif cmd == "reset":
                scene = data(config)
                out_queue.put(("state", scene.get_render_snapshot()))
            elif cmd == "get_state":
                out_queue.put(("state", scene.get_render_snapshot()))
            elif cmd == "set_state":
                scene.double_buffer_write = data
                scene.double_buffer_read = data
                out_queue.put(("ok", None))
            elif cmd == "close":
                break
    
    def step(self, actions: list[Any] | None = None) -> list[State]:
        """Step all environments. Returns list of new states."""
        if not self._initialized:
            self.initialize()
        
        if self.backend == "ray":
            import ray
            futures = [w.step.remote() for w in self._workers]
            results = ray.get(futures)
            return results
        else:
            for in_queue, _ in self._queues:
                in_queue.put(("step", None))
            results = []
            for _, out_queue in self._queues:
                _cmd, state = out_queue.get()
                results.append(state)
            return results
    
    def reset(self, indices: list[int] | None = None) -> list[State]:
        """Reset specified environments (or all)."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        if self.backend == "ray":
            import ray
            futures = [self._workers[i].reset.remote() for i in indices]
            return ray.get(futures)
        else:
            for i in indices:
                in_queue, _ = self._queues[i]
                in_queue.put(("reset", self.scene_factory))
            results = []
            for i in indices:
                _, out_queue = self._queues[i]
                _cmd, state = out_queue.get()
                results.append(state)
            return results
    
    def get_states(self) -> list[State]:
        """Get current render snapshots from all environments."""
        if self.backend == "ray":
            import ray
            futures = [w.get_state.remote() for w in self._workers]
            return ray.get(futures)
        else:
            for in_queue, _ in self._queues:
                in_queue.put(("get_state", None))
            results = []
            for _, out_queue in self._queues:
                _cmd, state = out_queue.get()
                results.append(state)
            return results
    
    def set_states(self, states: list[State]) -> None:
        """Set states for all environments."""
        if self.backend == "ray":
            import ray
            futures = [w.set_state.remote(s) for w, s in zip(self._workers, states)]
            ray.get(futures)
        else:
            for (in_queue, _), state in zip(self._queues, states):
                in_queue.put(("set_state", state))
            for _, out_queue in self._queues:
                out_queue.get()  # wait for ok
    
    def close(self) -> None:
        """Shutdown all environments."""
        if self.backend == "ray":
            import ray
            for w in self._workers:
                w.__ray_terminate__.remote()
            ray.shutdown()
        else:
            for in_queue, _ in self._queues:
                in_queue.put(("close", None))
            for proc in self._processes:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VectorizedEnv:
    """
    Single-process vectorized environment (no IPC overhead).
    
    All environments share same solver config but different initial states.
    Steps all envs in a single loop (SIMD-style).
    """
    
    def __init__(self, num_envs: int, scene: Scene):
        self.num_envs = num_envs
        self.scene = scene
        self.states: list[State] = []
    
    def reset(self, seeds: list[int] | None = None) -> list[State]:
        """Reset all environments with different seeds."""
        if seeds is None:
            seeds = list(range(self.num_envs))
        
        self.states = []
        for seed in seeds:
            np.random.seed(seed)
            # Randomize initial conditions
            state = self.scene._create_initial_state()
            self._randomize_state(state)
            self.states.append(state)
        
        return self.states
    
    def _randomize_state(self, state: State) -> None:
        """Apply domain randomization to state."""
        # Randomize positions, velocities, etc.
        if state.rigid_pos is not None:
            state.rigid_pos += np.random.uniform(-0.1, 0.1, state.rigid_pos.shape)
        if state.rigid_linvel is not None:
            state.rigid_linvel += np.random.uniform(-0.5, 0.5, state.rigid_linvel.shape)
    
    def step(self, actions: list | None = None) -> list[State]:
        """Step all environments."""
        new_states = []
        for state in self.states:
            self.scene.double_buffer_write = state
            new_state = self.scene.step()
            new_states.append(new_state)
        self.states = new_states
        return new_states
    
    def get_observations(self) -> list[np.ndarray]:
        """Get observations for all environments (for RL)."""
        obs = []
        for state in self.states:
            # Concatenate relevant state vectors
            vec = []
            if state.rigid_pos is not None:
                vec.append(state.rigid_pos.flatten())
            if state.rigid_linvel is not None:
                vec.append(state.rigid_linvel.flatten())
            if state.sph_pos is not None:
                vec.append(state.sph_pos.flatten())
            obs.append(np.concatenate(vec) if vec else np.array([]))
        return obs