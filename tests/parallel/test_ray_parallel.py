"""Tests for the Ray-based experiment runner (parallel.ray_parallel).

Ray is an optional extra and is NOT installed in CI, so this module installs
a minimal fake ``ray`` (identity ``remote`` decorator) before importing — but
ONLY when ray is genuinely absent (``importlib.util.find_spec``). If a real
ray is installed the whole module skips instead of shadowing it: with the
real ``@ray.remote`` the worker becomes an ActorClass and the direct-call
style of these tests does not apply. Real-ray integration (SimulationBatch,
``run_single_*``) remains out of scope and untested either way.

With the identity decorator ``SimulationWorker`` is a plain class whose
methods are exercised directly. The ``pwarm.kernel`` stubs from
tests/conftest.py are upgraded with a functional World3D fake first.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("ray") is not None:
    pytest.skip(
        "real ray is installed: these tests exercise the ray-free logic via "
        "an identity-remote fake and must not shadow it",
        allow_module_level=True,
    )

_fake_ray = types.ModuleType("ray")

def _remote(cls):  # identity decorator: class stays directly usable
    return cls

_fake_ray.remote = _remote
sys.modules["ray"] = _fake_ray

# Upgrade the kernel stub's World3D with a functional fake before the
# ray_parallel import binds it.
_kernel_world3d = sys.modules["pwarm.kernel.world3d"]


class FakeWorld3D:
    """Just enough World3D for SimulationWorker: step, serialize, metrics."""

    def __init__(self, gravity=None, dt=1 / 60, solver_iterations=10,
                 substeps=1) -> None:
        self.gravity = gravity
        self.dt = dt
        self.solver_iterations = solver_iterations
        self.substeps = substeps
        self.t = 0.0
        self.step_count = 0
        self.bodies: list = []

    def add(self, body) -> None:
        self.bodies.append(body)

    def add_body(self, body) -> None:
        self.add(body)

    def step(self, n: int = 1) -> None:
        g = np.asarray(self.gravity, dtype=float)
        for _ in range(n):
            self.step_count += 1
            self.t += self.dt
            for body in self.bodies:
                if not body.static:
                    body.vel = body.vel + g * self.dt  # gravity accelerates
                body.pos = body.pos + body.vel * self.dt

    def total_kinetic_energy(self) -> float:
        return float(sum(0.5 * b.mass * float(np.dot(b.vel, b.vel))
                         for b in self.bodies))

    def total_momentum(self) -> np.ndarray:
        return np.sum([b.mass * b.vel for b in self.bodies], axis=0)

    def total_angular_momentum(self) -> np.ndarray:
        return np.zeros(3)


_kernel_world3d.World3D = FakeWorld3D

from pwarm.parallel.ray_parallel import (
    CheckpointManager,
    ExperimentRunner,
    SimulationConfig,
    SimulationResult,
    SimulationWorker,
)


def moving_config(max_steps: int = 5, **overrides) -> SimulationConfig:
    body = {"shape": "sphere", "pos": [0.0, 0.0, 5.0], "mass": 1.0,
            "radius": 0.5, "vel": [0.0, 0.0, -1.0]}
    return SimulationConfig(max_steps=max_steps, bodies=[body], **overrides)


def test_simulation_config_defaults() -> None:
    """SimulationConfig carries documented defaults and a short config id."""
    config = SimulationConfig()
    assert config.gravity == (0.0, 0.0, -9.81)
    assert config.max_steps == 1000
    assert config.record_every == 10
    assert config.seed is None
    assert len(config.config_id) == 8


def test_run_simulation_success() -> None:
    """A moving body runs to max_steps and reports state plus metrics."""
    worker = SimulationWorker(0)
    result = worker.run_simulation(moving_config(max_steps=5))
    assert result.success is True
    assert result.steps_completed == 5
    assert result.config_id == result.config_id  # sanity
    assert result.final_time == pytest.approx(5 / 60)
    assert result.final_state["bodies"][0]["shape_type"] == "sphere"
    assert result.metrics["num_bodies"] == 1
    assert result.metrics["kinetic_energy"] > 0.0
    assert result.duration_seconds >= 0.0


def test_run_simulation_terminates_early_at_rest() -> None:
    """A fully resting scene stops after the first step."""
    resting = {"shape": "sphere", "pos": [0.0, 0.0, 5.0]}
    worker = SimulationWorker(1)
    result = worker.run_simulation(
        SimulationConfig(max_steps=100, gravity=(0.0, 0.0, 0.0),
                         bodies=[resting]))
    assert result.success is True
    assert result.steps_completed == 1


def test_run_simulation_unknown_shape_propagates() -> None:
    """_build_world runs outside the try block: ValueError escapes raw."""
    worker = SimulationWorker(0)
    bad = SimulationConfig(max_steps=3,
                           bodies=[{"shape": "torus", "pos": [0, 0, 0]}])
    with pytest.raises(ValueError, match="Unknown shape type"):
        worker.run_simulation(bad)


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("bad"),
                                 TypeError("nope")])
def test_run_simulation_failure_result(
        exc: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors raised by the world during the loop become failed results."""
    def explode(self, n: int = 1) -> None:
        raise exc

    monkeypatch.setattr(FakeWorld3D, "step", explode)
    worker = SimulationWorker(0)
    result = worker.run_simulation(moving_config(max_steps=3))
    assert result.success is False
    assert result.error == str(exc)
    assert result.final_state == {} and result.metrics == {}
    assert result.checkpoint_paths == []


def test_serialize_world_fields() -> None:
    """Serialization exposes per-body pose/vel/material-free fields."""
    worker = SimulationWorker(0)
    world = worker._build_world(moving_config())
    state = worker._serialize_world(world)
    assert state["t"] == 0.0 and state["step_count"] == 0
    body = state["bodies"][0]
    assert body["shape_type"] == "sphere"
    assert body["radius"] == 0.5
    assert body["half_extents"] is None  # spheres have no half extents
    assert body["static"] is False
    assert np.allclose(body["pos"], [0.0, 0.0, 5.0])


def test_serialize_box_half_extents() -> None:
    """Box bodies report their half extents."""
    worker = SimulationWorker(0)
    box_config = SimulationConfig(
        bodies=[{"shape": "box", "pos": [0, 0, 0],
                 "half_extents": [1.0, 2.0, 3.0]}])
    body = worker._serialize_world(worker._build_world(box_config))["bodies"][0]
    assert body["shape_type"] == "box"
    assert np.allclose(body["half_extents"], [1.0, 2.0, 3.0])


def test_compute_metrics_shape() -> None:
    """Metrics carry energy, momentum, angular momentum and body counts."""
    worker = SimulationWorker(0)
    metrics = worker._compute_metrics(worker._build_world(moving_config()))
    assert set(metrics) == {"kinetic_energy", "momentum", "angular_momentum",
                            "num_bodies", "static_bodies"}
    assert metrics["num_bodies"] == 1
    assert metrics["static_bodies"] == 0


def test_should_terminate_early_variants() -> None:
    """Termination needs every non-static body below the motion floor."""
    worker = SimulationWorker(0)
    resting = SimulationConfig(bodies=[{"shape": "sphere", "pos": [0, 0, 0],
                                        "vel": [0, 0, 0]}])
    assert worker._should_terminate_early(worker._build_world(resting)) is True
    moving_world = worker._build_world(
        SimulationConfig(gravity=(0.0, 0.0, 0.0),
                         bodies=[{"shape": "sphere", "pos": [0, 0, 0]}]))
    moving_world.bodies[0].vel = np.array([1.0, 0.0, 0.0])
    assert worker._should_terminate_early(moving_world) is False
    static_only = SimulationConfig(bodies=[{"shape": "box", "pos": [0, 0, 0],
                                            "static": True}])
    assert worker._should_terminate_early(
        worker._build_world(static_only)) is True


def test_checkpoints_written_during_run(tmp_path: Path) -> None:
    """checkpoint_interval triggers periodic pickled checkpoints."""
    worker = SimulationWorker(0)
    checkpoint_dir = tmp_path / "checkpoints"
    result = worker.run_simulation(moving_config(max_steps=5),
                                   checkpoint_dir=str(checkpoint_dir),
                                   checkpoint_interval=2)
    assert result.checkpoint_paths  # at least one checkpoint saved
    assert all(Path(p).exists() for p in result.checkpoint_paths)


def test_checkpoint_manager_roundtrip(tmp_path: Path) -> None:
    """save() writes pickles load_latest() can read back."""
    manager = CheckpointManager(tmp_path, max_checkpoints=10)
    path = manager.save("cfg", 7, {"position": [1.0, 2.0]})
    assert path.exists()
    loaded = manager.load_latest("cfg")
    assert loaded is not None
    assert loaded["step"] == 7
    assert loaded["state"] == {"position": [1.0, 2.0]}


def test_checkpoint_manager_empty(tmp_path: Path) -> None:
    """load_latest() without files returns None."""
    manager = CheckpointManager(tmp_path)
    assert manager.load_latest("nothing") is None


def test_checkpoint_manager_lexicographic_latest(tmp_path: Path) -> None:
    """Pinned quirk: 'latest' is lexicographic, so step10 sorts before step2."""
    manager = CheckpointManager(tmp_path)
    manager.save("cfg", 2, {"which": "step2"})
    manager.save("cfg", 10, {"which": "step10"})
    loaded = manager.load_latest("cfg")
    assert loaded["state"] == {"which": "step2"}  # 'step2' > 'step10' as text


def test_checkpoint_manager_cleanup(tmp_path: Path) -> None:
    """Saving beyond max_checkpoints deletes the oldest by mtime."""
    manager = CheckpointManager(tmp_path, max_checkpoints=2)
    first = manager.save("cfg", 1, {})
    second = manager.save("cfg", 2, {})
    third = manager.save("cfg", 3, {})
    assert not first.exists()
    assert second.exists() and third.exists()


def test_expand_grid_cartesian_product() -> None:
    """_expand_grid builds the cartesian product with fresh config ids."""
    base = SimulationConfig(bodies=[{"shape": "sphere"}], record_every=5,
                            seed=3)
    grid = {"gravity": [(0.0, 0.0, -9.81), (0.0, 0.0, -1.62)],
            "dt": [1 / 60, 1 / 120]}
    runner = ExperimentRunner.__new__(ExperimentRunner)  # skip ray init
    configs = runner._expand_grid(base, grid, max_steps=50)
    assert len(configs) == 4
    for config in configs:
        assert config.max_steps == 50
        assert config.bodies == base.bodies
        assert config.config_id != base.config_id
    assert len({c.config_id for c in configs}) == 4
    # pinned quirk: record_every and seed are NOT propagated to the sweep
    assert all(c.record_every != base.record_every for c in configs)
    assert all(c.seed is None for c in configs)
    gravities = {c.gravity for c in configs}
    assert gravities == {(0.0, 0.0, -9.81), (0.0, 0.0, -1.62)}


def test_save_result_writes_json(tmp_path: Path) -> None:
    """_save_result writes the documented JSON summary keys."""
    runner = ExperimentRunner.__new__(ExperimentRunner)  # skip ray init
    runner.output_dir = tmp_path
    result = SimulationResult(
        config_id="abcd1234", success=True, steps_completed=10,
        final_time=0.5, final_state={}, metrics={"ke": 1.0},
        duration_seconds=2.0)
    ExperimentRunner._save_result(runner, result)
    payload = json.loads((tmp_path / "result_abcd1234.json").read_text())
    assert payload["config_id"] == "abcd1234"
    assert payload["success"] is True
    assert payload["steps"] == 10
    assert payload["final_time"] == 0.5
    assert payload["metrics"] == {"ke": 1.0}
    assert payload["duration"] == 2.0
    assert payload["error"] is None
