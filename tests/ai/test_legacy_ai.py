"""Tests for the legacy AI package (pwarm.ai): PINNs, observer, closed loop,
autonomous experimentation.

The package's experiment/observer modules still import the removed
``pwarm.kernel``; tests/conftest.py stubs it, and this module upgrades the
World stub with a functional 2-D fake before importing. PINN tests use tiny
networks and few epochs to stay fast on CPU.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Upgrade the kernel stub's 2-D World with a functional fake BEFORE the
# pwarm.ai.experiment import binds it (the stub class accepts no kwargs).
_kernel_world = sys.modules["pwarm.kernel.world"]


class FakeWorld2D:
    """Just enough World for the observer/experiment layer."""

    def __init__(self, gravity=None, dt: float = 0.01) -> None:
        self.gravity = np.asarray(gravity, dtype=float)
        self.dt = dt
        self.t = 0.0
        self.bodies: list = []

    def add(self, body) -> None:
        self.bodies.append(body)

    def step(self) -> None:
        self.t += self.dt
        for body in self.bodies:
            if not body.static:
                body.vel = body.vel + self.gravity * self.dt
            body.pos = body.pos + body.vel * self.dt

    def total_kinetic_energy(self) -> float:
        return float(sum(0.5 * b.mass * float(np.dot(b.vel, b.vel))
                         for b in self.bodies))


_kernel_world.World = FakeWorld2D

from pwarm.ai.closed_loop import ClosedLoopAI
from pwarm.ai.experiment import (
    AutonomousExperimenter,
    ExperimentConfig,
    ExperimentRunner,
    Hypothesis,
    ParameterSpace,
    create_collision_experiment,
    create_free_fall_experiment,
)
from pwarm.ai.observer import WorldObserver, collect_free_fall
from pwarm.ai.pinn import (
    Activation,
    HeatPINN,
    NavierStokesPINN,
    PINNConfig,
    PINNTrainer,
    PINNWrapper,
    WavePINN,
    train_wave_pinn,
)

SMALL = PINNConfig(hidden_layers=[8], epochs=2, batch_size=16)


# ------------------------------------------------------------------------ PINN
def test_activation_dispatch_and_unknown() -> None:
    """Known activations build; an unknown name is rejected."""
    for name in ("tanh", "sin", "relu", "swish"):
        assert Activation(name)(torch.zeros(2)).shape == (2,)
    with pytest.raises(ValueError, match="Unknown activation"):
        Activation("selu")


def test_heat_pinn_residual_is_finite() -> None:
    """The heat-equation residual differentiates through the network."""
    model = HeatPINN(SMALL)
    x = torch.rand(16, 1)
    t = torch.rand(16, 1)
    residual = model.pde_residual(x, t)
    assert residual.shape == (16, 1)
    assert bool(torch.isfinite(residual).all())


def test_heat_pinn_loss_components() -> None:
    """loss() returns a scalar tensor and the three component values."""
    model = HeatPINN(SMALL)
    total, parts = model.loss(
        torch.rand(16, 1), torch.rand(16, 1),
        torch.zeros(4, 1), torch.rand(4, 1), torch.zeros(4, 1),
        torch.rand(4, 1), torch.zeros(4, 1), torch.zeros(4, 1))
    assert total.ndim == 0
    assert set(parts) == {"pde", "bc", "ic", "total"}
    assert parts["total"] == pytest.approx(total.item())


def test_wave_pinn_residual() -> None:
    """The wave-equation residual uses the second time derivative."""
    model = WavePINN(SMALL)
    residual = model.pde_residual(torch.rand(8, 1), torch.rand(8, 1))
    assert residual.shape == (8, 1)


def test_navier_stokes_residual_shapes() -> None:
    """NS residuals return momentum x/y and continuity fields."""
    model = NavierStokesPINN(SMALL)
    mx, my, cont = model.pde_residual(torch.rand(8, 1), torch.rand(8, 1),
                                      torch.rand(8, 1))
    for tensor in (mx, my, cont):
        assert tensor.shape == (8, 1)


def test_pinn_trainer_samples_and_step() -> None:
    """Samplers honour the domain bounds; one step changes the loss dict."""
    config = PINNConfig(hidden_layers=[8], x_min=-1.0, x_max=2.0,
                        t_min=0.5, t_max=1.5)
    trainer = PINNTrainer(HeatPINN(config), config)
    x, t = trainer.sample_pde_points(32)
    assert x.shape == (32, 1) and t.shape == (32, 1)
    assert x.min() >= -1.0 and x.max() <= 2.0
    x_bc, _t_bc, _u_bc = trainer.sample_bc_points(8)
    assert len(x_bc) == 8  # 4 left + 4 right
    assert set(np.round(np.unique(x_bc), 6)) == {-1.0, 2.0}
    _x_ic, _t_ic, u_ic = trainer.sample_ic_points(8, lambda xx: xx * 0.0 + 1.0)
    assert np.allclose(u_ic, 1.0)
    losses = trainer.train_step(n_pde=32, n_bc=8, n_ic=8)
    assert set(losses) == {"pde", "bc", "ic", "total"}
    history = trainer.train(epochs=2, n_pde=32, n_bc=8, n_ic=8, verbose=False)
    assert len(history) == 2 and trainer.history is history


def test_pinn_wrapper_train_predict_save_load(tmp_path) -> None:
    """The wrapper trains, predicts, and round-trips through disk."""
    wrapper = PINNWrapper("heat", PINNConfig(hidden_layers=[8], epochs=0))
    wrapper.train(ic_func=lambda x: np.sin(np.pi * x), epochs=1)
    u = wrapper.predict(np.array([0.25, 0.5]), np.array([0.1, 0.1]))
    assert u.shape == (2,)
    path = tmp_path / "heat.pt"
    wrapper.save(str(path))
    revived = PINNWrapper("heat", SMALL)
    revived.load(str(path))
    u2 = revived.predict(np.array([0.25]), np.array([0.1]))
    assert u2.shape == (1,)


def test_pinn_wrapper_errors() -> None:
    """Unknown PDE types and untrained predictions/save fail loudly."""
    with pytest.raises(ValueError, match="Unknown PDE type"):
        PINNWrapper("schrodinger").train()
    wrapper = PINNWrapper("heat", SMALL)
    with pytest.raises(RuntimeError, match="not trained"):
        wrapper.predict(np.array([0.5]), np.array([0.1]))
    with pytest.raises(RuntimeError, match="No model"):
        wrapper.save("/tmp/unused.pt")


def test_train_wave_pinn_returns_untrained_model() -> None:
    """The wave convenience function returns a fresh model (no training yet)."""
    model = train_wave_pinn(SMALL)
    assert isinstance(model, WavePINN)


# --------------------------------------------------------------- observer (2D)
def test_legacy_observer_records_free_fall() -> None:
    """The 2-D observer records per-body channels plus total KE."""
    world = FakeWorld2D(gravity=[0.0, -9.81], dt=0.01)
    world.add(type("B", (), {
        "pos": np.array([0.0, 10.0]), "vel": np.array([0.0, 0.0]),
        "mass": 1.0, "static": False})())
    data = WorldObserver(world, sample_every=10).observe(n_steps=100)
    assert "body0.pos.y" in data.names()
    assert "body0.vel.y" in data.names()
    y = data.get("body0.pos.y")
    assert y[-1] < y[0] - 0.5  # the body fell
    ke = data.get("total.ke")
    assert ke[-1] > 0.0


def test_collect_free_fall_convenience() -> None:
    """collect_free_fall wires the stub world, body, and observer."""
    data = collect_free_fall(n_steps=50, dt=0.01, g=9.81)
    y = data.get("body0.pos.y")
    assert y is not None and y[-1] < y[0]


# ----------------------------------------------------------------- closed loop
def test_closed_loop_recovers_quadratic_law() -> None:
    """A quadratic time series is re-predicted within the pass threshold."""
    t = np.linspace(0.0, 1.0, 100)
    y = -0.5 * 9.81 * t**2
    dataset = type("DS", (), {
        "observations": [type("O", (), {"name": "body0.pos.y", "t": t,
                                        "values": y})()],
        "time": lambda self: t,
        "get": lambda self, q: y,
    })()
    ai = ClosedLoopAI()
    results = ai.run(dataset)
    assert len(results) == 1
    result = results[0]
    assert result.quantity == "body0.pos.y"
    assert result.passed(0.05) is True
    assert result.law.r2 > 0.99
    text = ai.summary()
    assert "PASS" in text and "body0.pos.y" in text


def test_closed_loop_skips_short_or_missing_quantities() -> None:
    """Quantities absent from the dataset (or too short) are skipped."""
    t = np.linspace(0.0, 1.0, 100)

    class DS:
        observations: ClassVar[list] = []

        def time(self):
            return t

        def get(self, q):
            return None

    ai = ClosedLoopAI()
    assert ai.run(DS(), quantities=["missing", "tiny"]) == []
    assert ai.summary().startswith("Closed-loop AI discovery results:")


def test_closed_loop_flat_series_fails_gracefully() -> None:
    """A constant series has zero span; the loop still reports a result."""
    t = np.linspace(0.0, 1.0, 50)
    y = np.zeros(50)

    class DS:
        observations: ClassVar[list] = []

        def time(self):
            return t

        def get(self, q):
            return y

    result = ClosedLoopAI().run(DS(), quantities=["flat"])[0]
    assert result.passed(0.05) is True  # perfect fit of a constant


# ------------------------------------------------------- autonomous experiments
def test_parameter_space_grid_and_random() -> None:
    """Grid search enumerates the product; random search honours the seed."""
    space = ParameterSpace()
    space.add_parameter("g", [-9.81, -1.62])
    space.add_range("m", 1.0, 2.0, step=0.5)  # [1.0, 1.5, 2.0]
    space.add_range("h", 0.0, 1.0, n_points=3)
    assert len(space) == 18  # 2 g x 3 m x 3 h
    combos = space.grid_search()
    assert len(combos) == 18
    assert {"g", "m", "h"} == set(combos[0])
    a = space.random_search(5, seed=7)
    b = space.random_search(5, seed=7)
    assert a == b
    assert ParameterSpace().grid_search() == [{}]
    assert len(ParameterSpace()) == 0


def test_experiment_runner_free_fall(tmp_path) -> None:
    """A free-fall experiment runs, records metrics, and saves JSON."""
    runner = ExperimentRunner(output_dir=tmp_path)
    result = runner.run_experiment(create_free_fall_experiment(height=10.0,
                                                               n_steps=100))
    assert result.success is True
    assert result.metrics["n_samples"] == 100
    assert result.metrics["ke_final"] > 0.0
    assert result.metrics["body0.pos.y_final"] < 10.0
    saved = tmp_path / "free_fall.json"
    assert saved.exists()
    text = saved.read_text()
    assert '"experiment_id": "free_fall"' in text


def test_experiment_runner_collision_and_failure(tmp_path) -> None:
    """A zero-gravity collision runs; a bad body def fails gracefully."""
    runner = ExperimentRunner()
    result = runner.run_experiment(create_collision_experiment(n_steps=60))
    assert result.success is True
    # metrics are computed for the .pos.y channels (both bodies ride y=0)
    assert "body0.pos.y_range" in result.metrics
    # two 1 kg bodies at +-5 m/s carry 25 J of kinetic energy
    assert result.metrics["ke_mean"] == pytest.approx(25.0)

    bad = ExperimentConfig(
        experiment_id="bad", n_steps=5,
        bodies=[{"pos": [0.0, 1.0], "radius": "not-a-number"}])
    failure = runner.run_experiment(bad)
    assert failure.success is False
    assert failure.error_message != ""
    assert failure.dataset.observations == []


def test_run_batch_calls_progress() -> None:
    """run_batch reports (done, total) per experiment."""
    runner = ExperimentRunner()
    configs = [create_free_fall_experiment(n_steps=10) for _ in range(3)]
    seen: list[tuple[int, int]] = []
    results = runner.run_batch(configs, progress_callback=lambda i, n: seen.append((i, n)))
    assert len(results) == 3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_autonomous_experimenter_campaign(tmp_path) -> None:
    """The full loop: design -> run -> evaluate hypotheses -> summarize."""
    experimenter = AutonomousExperimenter(ExperimentRunner(output_dir=tmp_path))
    experimenter.add_hypothesis(Hypothesis(
        name="falls down",
        description="bodies fall",
        expected_behavior=lambda ds: (ds.get("body0.pos.y")[-1]
                                      < ds.get("body0.pos.y")[0]),
    ))
    space = ParameterSpace()
    space.add_parameter("body0_y", [5.0, 10.0])
    results = experimenter.run_experiment_campaign(space, n_experiments=2,
                                                   strategy="grid")
    assert len(results) == 2
    assert len(experimenter.experiment_history) == 2
    assert experimenter.hypotheses[0].confidence == pytest.approx(1.0)
    assert experimenter.get_best_parameters("ke_mean") is not None
    summary = experimenter.summary()
    assert "SUPPORTED" in summary or "NEEDS MORE DATA" in summary


def test_design_experiments_strategies() -> None:
    """Grid/random strategies build configs; unknown ones raise."""
    experimenter = AutonomousExperimenter()
    space = ParameterSpace()
    space.add_parameter("body0_y", [5.0, 10.0])
    space.add_parameter("n_bodies", [1])
    configs = experimenter.design_experiments(space, n_experiments=2,
                                              strategy="grid")
    assert len(configs) == 2
    assert configs[0].experiment_id == "exp_0000"
    assert configs[0].bodies[0]["pos"][1] in (5.0, 10.0)
    random_configs = experimenter.design_experiments(space, n_experiments=1,
                                                     strategy="random")
    assert len(random_configs) == 1
    with pytest.raises(ValueError, match="Unknown strategy"):
        experimenter.design_experiments(space, strategy="bayesian")


def test_build_bodies_from_params() -> None:
    """Body defs pick up per-body parameter keys with defaults."""
    experimenter = AutonomousExperimenter()
    bodies = experimenter._build_bodies_from_params({
        "n_bodies": 2, "body0_x": 1.0, "body1_vy": -2.0})
    assert len(bodies) == 2
    assert bodies[0]["pos"] == [1.0, 10.0]  # y defaults to 10
    assert bodies[1]["vel"] == [0.0, -2.0]


def test_get_best_parameters_empty_and_minimize() -> None:
    """Empty history returns None; minimize picks the smallest metric."""
    experimenter = AutonomousExperimenter()
    assert experimenter.get_best_parameters() is None
    runner = ExperimentRunner()
    # different step counts give different final speeds (and thus KE)
    results = runner.run_batch([
        create_free_fall_experiment(height=5.0, n_steps=30),
        create_free_fall_experiment(height=5.0, n_steps=60)])
    experimenter.experiment_history = results
    experimenter.parameter_history = [{"n": 30}, {"n": 60}]
    best_min = experimenter.get_best_parameters("ke_final", maximize=False)
    best_max = experimenter.get_best_parameters("ke_final", maximize=True)
    assert best_min == {"n": 30}
    assert best_max == {"n": 60}
