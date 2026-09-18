"""AI Autonomous Experiment module.

Enables the AI layer to autonomously design and run experiments:
- Define parameter spaces (initial conditions, body properties)
- Generate experiment configurations via grid/random search
- Execute experiments and collect results
- Analyze outcomes against hypotheses
- Iterate on experimental design

This implements the Phase 3.4 autonomous experimentation loop:
    hypothesis -> design experiment -> execute -> observe -> analyze -> refine
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymo.ai.observer import TimeSeriesDataset, WorldObserver
from pymo.kernel.bodies import Material, circle_body
from pymo.kernel.world import World


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Experiment metadata
    experiment_id: str = ""
    description: str = ""
    
    # World parameters
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))
    dt: float = 0.01
    
    # Body definitions: list of (pos, radius, mass, material_props)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    
    # Observation parameters
    n_steps: int = 300
    sample_every: int = 1
    
    # Custom initial conditions
    initial_conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    
    config: ExperimentConfig
    dataset: TimeSeriesDataset
    metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class Hypothesis:
    """A hypothesis about the world's behavior."""
    
    name: str
    description: str
    expected_behavior: Callable[[TimeSeriesDataset], bool]
    expected_metrics: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1, updated after experiments


class ParameterSpace:
    """Defines the space of experiment parameters."""
    
    def __init__(self):
        self.parameters: dict[str, list[Any]] = {}
    
    def add_parameter(self, name: str, values: list[Any]) -> ParameterSpace:
        """Add a parameter with possible values."""
        self.parameters[name] = values
        return self
    
    def add_range(self, name: str, start: float, end: float, 
                  step: float | None = None, n_points: int = 10) -> ParameterSpace:
        """Add a numeric range parameter."""
        if step is not None:
            values = list(np.arange(start, end + step/2, step))
        else:
            values = list(np.linspace(start, end, n_points))
        self.parameters[name] = values
        return self
    
    def grid_search(self) -> list[dict[str, Any]]:
        """Generate all combinations (grid search)."""
        if not self.parameters:
            return [{}]
        
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs
    
    def random_search(self, n_samples: int = 10, 
                      seed: int | None = None) -> list[dict[str, Any]]:
        """Generate random samples from the parameter space."""
        rng = np.random.default_rng(seed)
        configs = []
        
        for _ in range(n_samples):
            sample = {}
            for name, values in self.parameters.items():
                sample[name] = values[rng.integers(0, len(values))]
            configs.append(sample)
        return configs
    
    def __len__(self) -> int:
        """Total number of combinations."""
        if not self.parameters:
            return 0
        result = 1
        for values in self.parameters.values():
            result *= len(values)
        return result


class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute a single experiment."""
        start_time = time.time()
        
        try:
            # Create world
            world = World(gravity=config.gravity, dt=config.dt)
            
            # Add bodies
            for body_def in config.bodies:
                pos = body_def.get("pos", [0.0, 0.0])
                radius = body_def.get("radius", 0.5)
                mass = body_def.get("mass", 1.0)
                static = body_def.get("static", False)
                
                mat_props = body_def.get("material", {})
                material = Material(
                    restitution=mat_props.get("restitution", 0.3),
                    friction=mat_props.get("friction", 0.4),
                    specific_heat=mat_props.get("specific_heat", 1000.0),
                )
                
                body = circle_body(pos, radius, mass=mass, material=material, static=static)
                
                # Apply initial velocity if specified
                if "vel" in body_def:
                    body.vel = np.array(body_def["vel"], dtype=float)
                
                world.add(body)
            
            # Apply custom initial conditions
            for key, value in config.initial_conditions.items():
                if hasattr(world, key):
                    setattr(world, key, value)
            
            # Observe
            observer = WorldObserver(world, sample_every=config.sample_every)
            dataset = observer.observe(config.n_steps)
            
            # Compute metrics
            metrics = self._compute_metrics(dataset, world)
            
            duration = time.time() - start_time
            
            result = ExperimentResult(
                config=config,
                dataset=dataset,
                metrics=metrics,
                duration_seconds=duration,
                success=True
            )
            
            # Save if output directory specified
            if self.output_dir:
                self._save_result(result)
            
            return result
            
        except Exception as e:  # noqa: BLE001 — isolate one failed experiment
            duration = time.time() - start_time
            return ExperimentResult(
                config=config,
                dataset=TimeSeriesDataset(),
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )
    
    def run_batch(self, configs: list[ExperimentConfig], 
                  progress_callback: Callable[[int, int], None] | None = None
                  ) -> list[ExperimentResult]:
        """Run multiple experiments."""
        results = []
        for i, config in enumerate(configs):
            result = self.run_experiment(config)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(configs))
        return results
    
    def _compute_metrics(self, dataset: TimeSeriesDataset, world: World) -> dict[str, float]:
        """Compute summary metrics from experiment data."""
        metrics = {}
        
        t = dataset.time()
        if len(t) == 0:
            return metrics
        
        # Time metrics
        metrics["total_time"] = float(t[-1]) if len(t) > 0 else 0.0
        metrics["n_samples"] = len(t)
        
        # Energy metrics
        ke = dataset.get("total.ke")
        if ke is not None and len(ke) > 0:
            metrics["ke_mean"] = float(np.mean(ke))
            metrics["ke_max"] = float(np.max(ke))
            metrics["ke_min"] = float(np.min(ke))
            metrics["ke_final"] = float(ke[-1])
        
        # Position metrics for each body
        for obs in dataset.observations:
            if ".pos.y" in obs.name:
                values = obs.values
                metrics[f"{obs.name}_range"] = float(np.ptp(values))
                metrics[f"{obs.name}_final"] = float(values[-1])
                metrics[f"{obs.name}_max"] = float(np.max(values))
                metrics[f"{obs.name}_min"] = float(np.min(values))
        
        return metrics
    
    def _save_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        if not self.output_dir:
            return
        
        path = self.output_dir / f"{result.config.experiment_id}.json"
        
        # Convert dataset to serializable format
        data = {
            "experiment_id": result.config.experiment_id,
            "description": result.config.description,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "metrics": result.metrics,
            "config": {
                "gravity": result.config.gravity.tolist(),
                "dt": result.config.dt,
                "n_steps": result.config.n_steps,
                "bodies": result.config.bodies,
            }
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class AutonomousExperimenter:
    """AI-driven autonomous experimentation loop."""
    
    def __init__(self, runner: ExperimentRunner | None = None):
        self.runner = runner or ExperimentRunner()
        self.hypotheses: list[Hypothesis] = []
        self.experiment_history: list[ExperimentResult] = []
        self.parameter_history: list[dict[str, Any]] = []
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to test."""
        self.hypotheses.append(hypothesis)
    
    def design_experiments(self, parameter_space: ParameterSpace, 
                          n_experiments: int = 10,
                          strategy: str = "grid") -> list[ExperimentConfig]:
        """Design experiments to test hypotheses."""
        if strategy == "grid":
            param_combos = parameter_space.grid_search()
        elif strategy == "random":
            param_combos = parameter_space.random_search(n_experiments)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Limit to n_experiments
        param_combos = param_combos[:n_experiments]
        
        configs = []
        for i, params in enumerate(param_combos):
            config = ExperimentConfig(
                experiment_id=f"exp_{i:04d}",
                description=f"Autonomous experiment {i}",
                bodies=self._build_bodies_from_params(params),
                initial_conditions=params,
            )
            configs.append(config)
        
        return configs
    
    def _build_bodies_from_params(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Build body definitions from parameter dict."""
        bodies = []
        
        # Extract body parameters
        n_bodies = params.get("n_bodies", 1)
        
        for i in range(n_bodies):
            body = {
                "pos": [params.get(f"body{i}_x", 0.0), 
                        params.get(f"body{i}_y", 10.0)],
                "radius": params.get(f"body{i}_radius", 0.5),
                "mass": params.get(f"body{i}_mass", 1.0),
                "vel": [params.get(f"body{i}_vx", 0.0),
                        params.get(f"body{i}_vy", 0.0)],
            }
            bodies.append(body)
        
        return bodies
    
    def run_experiment_campaign(self, parameter_space: ParameterSpace,
                               n_experiments: int = 10,
                               strategy: str = "random") -> list[ExperimentResult]:
        """Run a full experiment campaign."""
        # Design experiments
        configs = self.design_experiments(parameter_space, n_experiments, strategy)
        
        # Run experiments
        results = self.runner.run_batch(configs)
        
        # Store history
        self.experiment_history.extend(results)
        self.parameter_history.extend([c.initial_conditions for c in configs])
        
        # Evaluate hypotheses
        self._evaluate_hypotheses(results)
        
        return results
    
    def _evaluate_hypotheses(self, results: list[ExperimentResult]) -> None:
        """Evaluate hypotheses against experiment results."""
        for hypothesis in self.hypotheses:
            passed_count = 0
            for result in results:
                if result.success:
                    try:
                        if hypothesis.expected_behavior(result.dataset):
                            passed_count += 1
                    except Exception:  # noqa: BLE001, S110 — a broken check must not veto verification
                        pass
            
            # Update confidence based on pass rate
            if results:
                hypothesis.confidence = passed_count / len(results)
    
    def get_best_parameters(self, metric: str = "ke_mean", 
                           maximize: bool = True) -> dict[str, Any] | None:
        """Get parameters that produced the best metric value."""
        if not self.experiment_history:
            return None
        
        best_idx = None
        best_value = None
        
        for i, result in enumerate(self.experiment_history):
            if not result.success or metric not in result.metrics:
                continue
            
            value = result.metrics[metric]
            if best_value is None or (maximize and value > best_value) or \
               (not maximize and value < best_value):
                best_value = value
                best_idx = i
        
        if best_idx is not None:
            return self.parameter_history[best_idx]
        return None
    
    def summary(self) -> str:
        """Generate summary of experiment campaign."""
        lines = ["Autonomous Experiment Campaign Summary:"]
        lines.append(f"  Total experiments: {len(self.experiment_history)}")
        
        successful = sum(1 for r in self.experiment_history if r.success)
        lines.append(f"  Successful: {successful}")
        lines.append(f"  Failed: {len(self.experiment_history) - successful}")
        
        if self.hypotheses:
            lines.append("\nHypothesis Results:")
            for h in self.hypotheses:
                status = "SUPPORTED" if h.confidence > 0.7 else "NEEDS MORE DATA"
                lines.append(f"  {h.name}: {h.confidence:.1%} confidence [{status}]")
        
        return "\n".join(lines)


def create_free_fall_experiment(height: float = 10.0, 
                                mass: float = 1.0,
                                n_steps: int = 300) -> ExperimentConfig:
    """Create a standard free-fall experiment config."""
    return ExperimentConfig(
        experiment_id="free_fall",
        description=f"Free fall from height {height}m",
        gravity=np.array([0.0, -9.81]),
        dt=0.01,
        bodies=[{
            "pos": [0.0, height],
            "radius": 0.5,
            "mass": mass,
            "vel": [0.0, 0.0],
        }],
        n_steps=n_steps,
    )


def create_collision_experiment(v1: float = 5.0, v2: float = -5.0,
                                separation: float = 4.0,
                                n_steps: int = 300) -> ExperimentConfig:
    """Create a two-body collision experiment."""
    return ExperimentConfig(
        experiment_id="collision",
        description=f"Collision with v1={v1}, v2={v2}",
        gravity=np.array([0.0, 0.0]),  # no gravity
        dt=0.01,
        bodies=[
            {
                "pos": [-separation/2, 0.0],
                "radius": 0.5,
                "mass": 1.0,
                "vel": [v1, 0.0],
            },
            {
                "pos": [separation/2, 0.0],
                "radius": 0.5,
                "mass": 1.0,
                "vel": [v2, 0.0],
            },
        ],
        n_steps=n_steps,
    )


def create_parameter_sweep_experiment() -> tuple[ParameterSpace, list[ExperimentConfig]]:
    """Create a parameter sweep for gravity and mass."""
    space = ParameterSpace()
    space.add_parameter("gravity_y", [-9.81, -4.9, -1.0])
    space.add_parameter("body0_mass", [0.5, 1.0, 2.0])
    space.add_parameter("body0_y", [5.0, 10.0, 15.0])
    
    configs = []
    for params in space.grid_search():
        config = ExperimentConfig(
            experiment_id=f"sweep_{hash(str(params)) % 10000:04d}",
            description=f"Sweep: g={params['gravity_y']}, m={params['body0_mass']}, h={params['body0_y']}",
            gravity=np.array([0.0, params["gravity_y"]]),
            dt=0.01,
            bodies=[{
                "pos": [0.0, params["body0_y"]],
                "radius": 0.5,
                "mass": params["body0_mass"],
            }],
            n_steps=300,
        )
        configs.append(config)
    
    return space, configs
