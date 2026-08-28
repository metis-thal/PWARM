"""Ray-based parallel simulation infrastructure for pymo.

Provides distributed simulation execution with fault tolerance,
checkpointing, and result aggregation.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ray

from pymo.kernel.bodies3d import Body, Material
from pymo.kernel.world3d import World3D


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    # World parameters
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 1 / 60.0
    solver_iterations: int = 10
    substeps: int = 1
    
    # Simulation control
    max_steps: int = 1000
    record_every: int = 10
    
    # Body definitions (serializable)
    bodies: list[dict] = field(default_factory=list)
    
    # Metadata
    config_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    seed: int | None = None


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    config_id: str
    success: bool
    steps_completed: int
    final_time: float
    final_state: dict
    metrics: dict
    error: str | None = None
    duration_seconds: float = 0.0
    checkpoint_paths: list[str] = field(default_factory=list)


@ray.remote
class SimulationWorker:
    """Ray actor for running physics simulations."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.current_sim: World3D | None = None
        self.current_config: SimulationConfig | None = None
        self.step_count = 0
        self.start_time = 0.0
    
    def run_simulation(self, config: SimulationConfig, 
                       checkpoint_dir: str | None = None,
                       checkpoint_interval: int = 100) -> SimulationResult:
        """Run a complete simulation from config."""
        self.current_config = config
        self.step_count = 0
        self.start_time = time.time()
        
        # Build world from config
        world = self._build_world(config)
        self.current_sim = world
        
        checkpoint_paths = []
        last_checkpoint_step = 0
        
        try:
            for step in range(config.max_steps):
                world.step(1)
                self.step_count += 1
                
                # Periodic checkpointing
                if checkpoint_dir and (step - last_checkpoint_step) >= checkpoint_interval:
                    ckpt_path = self._save_checkpoint(checkpoint_dir, step)
                    checkpoint_paths.append(ckpt_path)
                    last_checkpoint_step = step
                
                # Early termination check
                if self._should_terminate_early(world):
                    break
            
            duration = time.time() - self.start_time
            result = SimulationResult(
                config_id=config.config_id,
                success=True,
                steps_completed=self.step_count,
                final_time=world.t,
                final_state=self._serialize_world(world),
                metrics=self._compute_metrics(world),
                duration_seconds=duration,
                checkpoint_paths=checkpoint_paths
            )
            return result
            
        except (RuntimeError, ValueError, TypeError) as e:
            duration = time.time() - self.start_time
            return SimulationResult(
                config_id=config.config_id,
                success=False,
                steps_completed=self.step_count,
                final_time=world.t if world else 0.0,
                final_state={},
                metrics={},
                error=str(e),
                duration_seconds=duration
            )
    
    def _build_world(self, config: SimulationConfig) -> World3D:
        """Build World3D from SimulationConfig."""
        world = World3D(
            gravity=np.array(config.gravity),
            dt=config.dt,
            solver_iterations=config.solver_iterations,
            substeps=config.substeps
        )
        
        for body_def in config.bodies:
            body = self._create_body(body_def)
            world.add(body)
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        return world
    
    def _create_body(self, body_def: dict) -> Body:
        """Create Body from definition dict."""
        from pymo.kernel.bodies3d import box_body, cylinder_body, sphere_body
        
        shape_type = body_def.get("shape", "sphere")
        pos = np.array(body_def.get("pos", [0, 0, 0]), dtype=float)
        mass = body_def.get("mass", 1.0)
        static = body_def.get("static", False)
        
        material = Material(
            restitution=body_def.get("restitution", 0.3),
            friction=body_def.get("friction", 0.4),
            density=body_def.get("density", 1000.0),
            specific_heat=body_def.get("specific_heat", 1000.0),
            thermal_conductivity=body_def.get("thermal_conductivity", 0.0),
            young_modulus=body_def.get("young_modulus", 1e9),
            hardness=body_def.get("hardness", 1e6),
            fracture_toughness=body_def.get("fracture_toughness", 1e6),
            brittleness=body_def.get("brittleness", 0.5),
        )
        
        if shape_type == "sphere":
            return sphere_body(
                pos=pos,
                radius=body_def.get("radius", 0.5),
                mass=mass,
                material=material,
                static=static
            )
        elif shape_type == "box":
            half_extents = body_def.get("half_extents", [0.5, 0.5, 0.5])
            angle = body_def.get("angle", 0.0)
            return box_body(
                pos=pos,
                half_extents=np.array(half_extents, dtype=float),
                mass=mass,
                material=Material(**material.__dict__),
                static=static,
                angle=angle
            )
        elif shape_type == "cylinder":
            return cylinder_body(
                pos=pos,
                radius=body_def.get("radius", 0.5),
                half_height=body_def.get("half_height", 0.5),
                mass=mass,
                material=Material(**material.__dict__),
                static=static
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
    def _should_terminate_early(self, world: World3D) -> bool:
        """Check if simulation should terminate early (all bodies at rest)."""
        for body in world.bodies:
            if not body.static and (np.linalg.norm(body.vel) > 0.01 or np.linalg.norm(body.ang_vel) > 0.01):
                return False
        return True
    
    def _save_checkpoint(self, checkpoint_dir: str, step: int) -> str:
        """Save simulation state to checkpoint file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = Path(checkpoint_dir) / f"checkpoint_{self.current_config.config_id}_step{step}.pkl"
        
        state = {
            "config": self.current_config,
            "step": self.step_count,
            "world_state": self._serialize_world(self.current_sim),
            "timestamp": time.time()
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        return str(path)
    
    def _serialize_world(self, world: World3D) -> dict:
        """Serialize world state to dictionary."""
        return {
            "t": world.t,
            "step_count": world.step_count,
            "bodies": [
                {
                    "pos": b.pos.tolist(),
                    "vel": b.vel.tolist(),
                    "orn": b.orn.tolist(),
                    "ang_vel": b.ang_vel.tolist(),
                    "mass": b.mass,
                    "static": b.static,
                    "temperature": b.temperature,
                    "shape_type": b.shape.shape_type.value if b.shape else "none",
                    "radius": getattr(b.shape, "radius", None),
                    "half_extents": getattr(b.shape, "half_extents", None).tolist() if hasattr(b.shape, "half_extents") and b.shape.half_extents is not None else None,
                }
                for b in world.bodies
            ]
        }
    
    def _compute_metrics(self, world: World3D) -> dict:
        """Compute simulation metrics."""
        ke = world.total_kinetic_energy()
        mom = world.total_momentum()
        ang_mom = world.total_angular_momentum()
        
        return {
            "kinetic_energy": float(ke),
            "momentum": mom.tolist(),
            "angular_momentum": ang_mom.tolist(),
            "num_bodies": len(world.bodies),
            "static_bodies": sum(1 for b in world.bodies if b.static),
        }


class SimulationBatch:
    """Manages batch execution of simulations across Ray workers."""
    
    def __init__(self, num_workers: int | None = None, ray_address: str | None = None):
        """Initialize Ray cluster and worker pool."""
        if not ray.is_initialized():
            ray.init(address=ray_address, num_cpus=num_workers, ignore_reinit_error=True)
        
        self.num_workers = num_workers or ray.cluster_resources().get("CPU", 4)
        self.workers = [SimulationWorker.remote(i) for i in range(self.num_workers)]
        self.worker_queue = list(self.workers)
        self.results = []
    
    def submit(self, config: SimulationConfig, 
               checkpoint_dir: str | None = None,
               checkpoint_interval: int = 100) -> ray.ObjectRef:
        """Submit a simulation config to the worker pool."""
        worker = self.worker_queue.pop(0)
        self.worker_queue.append(worker)
        
        return worker.run_simulation.remote(config, checkpoint_dir, checkpoint_interval)
    
    def submit_batch(self, configs: list[SimulationConfig],
                     checkpoint_dir: str | None = None,
                     checkpoint_interval: int = 100) -> list[ray.ObjectRef]:
        """Submit multiple simulations."""
        refs = []
        for config in configs:
            ref = self.submit(config, checkpoint_dir, checkpoint_interval)
            refs.append(ref)
        return refs
    
    def gather(self, refs: list[ray.ObjectRef], timeout: float | None = None) -> list[SimulationResult]:
        """Wait for and collect results."""
        results = ray.get(refs, timeout=timeout)
        self.results.extend(results)
        return results
    
    def get_completed(self, refs: list[ray.ObjectRef]) -> tuple[list[SimulationResult], list[ray.ObjectRef]]:
        """Get completed results, return (completed, remaining)."""
        ready, remaining = ray.wait(refs, num_returns=len(refs), timeout=0)
        completed = ray.get(ready) if ready else []
        return completed, remaining
    
    def shutdown(self):
        """Shutdown workers and Ray."""
        ray.shutdown()


class CheckpointManager:
    """Manages simulation checkpoints with automatic cleanup."""
    
    def __init__(self, base_dir: str, max_checkpoints: int = 10):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, config_id: str, step: int, state: dict) -> Path:
        """Save a checkpoint."""
        path = self.base_dir / f"{config_id}_step{step}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"step": step, "state": state, "timestamp": time.time()}, f)
        self._cleanup_old()
        return path
    
    def load_latest(self, config_id: str) -> dict | None:
        """Load the latest checkpoint for a config."""
        checkpoints = sorted(self.base_dir.glob(f"{config_id}_step*.pkl"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        with open(latest, "rb") as f:
            return pickle.load(f)
    
    def _cleanup_old(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = sorted(self.base_dir.glob("*.pkl"), key=os.path.getmtime)
        while len(checkpoints) > self.max_checkpoints:
            checkpoints.pop(0).unlink()


class ExperimentRunner:
    """High-level experiment orchestration with Ray."""
    
    def __init__(self, num_workers: int | None = None, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch = SimulationBatch(num_workers=num_workers)
        self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")
    
    def run_sweep(self, param_grid: dict[str, list], 
                  base_config: SimulationConfig,
                  max_steps: int = 1000) -> list[SimulationResult]:
        """Run parameter sweep across grid."""
        configs = self._expand_grid(base_config, param_grid, max_steps)
        
        # Submit all
        refs = self.batch.submit_batch(configs)
        
        # Wait for completion with progress
        results = []
        remaining = refs
        while remaining:
            ready, remaining = self.batch.get_completed(remaining)
            for r in ready:
                self._save_result(r)
                results.append(r)
            print(f"Progress: {len(results)}/{len(configs)}")
        
        return results
    
    def _expand_grid(self, base: SimulationConfig, grid: dict[str, list], max_steps: int) -> list[SimulationConfig]:
        """Expand parameter grid into list of configs."""
        import itertools
        
        keys = list(grid.keys())
        values = list(grid.values())
        configs = []
        
        for combo in itertools.product(*values):
            config = SimulationConfig(
                gravity=base.gravity,
                dt=base.dt,
                solver_iterations=base.solver_iterations,
                substeps=base.substeps,
                max_steps=max_steps,
                bodies=base.bodies,
                config_id=str(uuid.uuid4())[:8]
            )
            for key, value in zip(keys, combo):
                setattr(config, key, value)
            configs.append(config)
        return configs
    
    def _save_result(self, result: SimulationResult):
        """Save result to output directory."""
        path = self.output_dir / f"result_{result.config_id}.json"
        with open(path, "w") as f:
            json.dump({
                "config_id": result.config_id,
                "success": result.success,
                "steps": result.steps_completed,
                "final_time": result.final_time,
                "metrics": result.metrics,
                "error": result.error,
                "duration": result.duration_seconds
            }, f, indent=2)


# Convenience functions
def run_single_simulation(config: SimulationConfig) -> SimulationResult:
    """Run a single simulation synchronously (no Ray)."""
    worker = SimulationWorker.remote(0)
    return ray.get(worker.run_simulation.remote(config))


def run_parallel_sims(configs: list[SimulationConfig], num_workers: int = 4) -> list[SimulationResult]:
    """Run multiple simulations in parallel using Ray."""
    ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    try:
        batch = SimulationBatch(num_workers=num_workers)
        refs = batch.submit_batch(configs)
        return batch.gather(refs)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    # Demo: run a simple sweep
    import uuid
    
    base_config = SimulationConfig(
        bodies=[
            {"shape": "sphere", "pos": [0, 0, 5], "radius": 0.5, "mass": 1.0},
            {"shape": "box", "pos": [0, 0, -0.5], "half_extents": [5, 5, 0.5], "static": True, "mass": 0}
        ],
        max_steps=200
    )
    
    param_grid = {
        "gravity": [(0, 0, -9.81), (0, 0, -1.62)],
        "dt": [1/60.0, 1/120.0],
    }
    
    runner = ExperimentRunner(num_workers=2)
    results = runner.run_sweep(param_grid, base_config, max_steps=200)
    
    print(f"Completed {len(results)} simulations")
    for r in results:
        print(f"  {r.config_id}: success={r.success}, steps={r.steps_completed}, ke={r.metrics.get('kinetic_energy', 0):.2f}")