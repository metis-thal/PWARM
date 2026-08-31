"""Tests for AI autonomous experiment module."""

import numpy as np
import pytest

from pymo.ai.experiment import (
    AutonomousExperimenter,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    Hypothesis,
    ParameterSpace,
    create_collision_experiment,
    create_free_fall_experiment,
    create_parameter_sweep_experiment,
)
from pymo.ai.observer import TimeSeriesDataset


class TestParameterSpace:
    def test_add_parameter(self):
        space = ParameterSpace()
        space.add_parameter("x", [1, 2, 3])
        assert "x" in space.parameters
        assert space.parameters["x"] == [1, 2, 3]

    def test_add_range(self):
        space = ParameterSpace()
        space.add_range("y", 0.0, 10.0, n_points=5)
        assert len(space.parameters["y"]) == 5
        assert space.parameters["y"][0] == pytest.approx(0.0)
        assert space.parameters["y"][-1] == pytest.approx(10.0)

    def test_grid_search(self):
        space = ParameterSpace()
        space.add_parameter("a", [1, 2])
        space.add_parameter("b", [3, 4])
        grid = space.grid_search()
        assert len(grid) == 4
        assert {"a": 1, "b": 3} in grid
        assert {"a": 2, "b": 4} in grid

    def test_random_search(self):
        space = ParameterSpace()
        space.add_parameter("x", [10, 20, 30])
        samples = space.random_search(n_samples=5, seed=42)
        assert len(samples) == 5
        for s in samples:
            assert s["x"] in [10, 20, 30]

    def test_len(self):
        space = ParameterSpace()
        assert len(space) == 0
        space.add_parameter("a", [1, 2, 3])
        assert len(space) == 3
        space.add_parameter("b", [4, 5])
        assert len(space) == 6


class TestExperimentConfig:
    def test_default_config(self):
        config = ExperimentConfig()
        assert config.experiment_id == ""
        assert config.dt == 0.01
        assert len(config.bodies) == 0

    def test_free_fall_config(self):
        config = create_free_fall_experiment(height=5.0, mass=2.0)
        assert config.experiment_id == "free_fall"
        assert config.gravity[1] == pytest.approx(-9.81)
        assert len(config.bodies) == 1
        assert config.bodies[0]["pos"][1] == pytest.approx(5.0)
        assert config.bodies[0]["mass"] == pytest.approx(2.0)

    def test_collision_config(self):
        config = create_collision_experiment(v1=10.0, v2=-10.0)
        assert config.experiment_id == "collision"
        assert len(config.bodies) == 2
        assert config.bodies[0]["vel"][0] == pytest.approx(10.0)
        assert config.bodies[1]["vel"][0] == pytest.approx(-10.0)


class TestExperimentRunner:
    def test_run_free_fall(self):
        config = create_free_fall_experiment(height=10.0, n_steps=100)
        runner = ExperimentRunner()
        result = runner.run_experiment(config)
        
        assert result.success
        assert result.duration_seconds >= 0  # may be 0 for fast runs
        assert "ke_mean" in result.metrics
        assert "body0.pos.y_final" in result.metrics

    def test_run_collision(self):
        config = create_collision_experiment(n_steps=200)
        runner = ExperimentRunner()
        result = runner.run_experiment(config)
        
        assert result.success
        assert result.dataset.time().shape[0] > 0

    def test_run_batch(self):
        configs = [create_free_fall_experiment(height=h) for h in [5.0, 10.0, 15.0]]
        runner = ExperimentRunner()
        results = runner.run_batch(configs)
        
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_metrics_computed(self):
        config = create_free_fall_experiment(n_steps=100)
        runner = ExperimentRunner()
        result = runner.run_experiment(config)
        
        assert result.metrics["n_samples"] > 0
        assert result.metrics["total_time"] > 0


class TestHypothesis:
    def test_hypothesis_creation(self):
        def always_true(dataset):
            return True
        
        h = Hypothesis(
            name="test_hypothesis",
            description="A test hypothesis",
            expected_behavior=always_true,
            confidence=0.0
        )
        assert h.name == "test_hypothesis"
        assert h.confidence == 0.0

    def test_hypothesis_evaluation(self):
        def body_falls(dataset):
            y = dataset.get("body0.pos.y")
            if y is None or len(y) < 2:
                return False
            return y[-1] < y[0]  # final height less than initial
        
        h = Hypothesis(
            name="free_fall",
            description="Body falls under gravity",
            expected_behavior=body_falls
        )
        
        config = create_free_fall_experiment(height=10.0, n_steps=100)
        runner = ExperimentRunner()
        result = runner.run_experiment(config)
        
        assert h.expected_behavior(result.dataset)


class TestAutonomousExperimenter:
    def test_creation(self):
        experimenter = AutonomousExperimenter()
        assert len(experimenter.hypotheses) == 0
        assert len(experimenter.experiment_history) == 0

    def test_add_hypothesis(self):
        experimenter = AutonomousExperimenter()
        h = Hypothesis(
            name="test",
            description="test",
            expected_behavior=lambda d: True
        )
        experimenter.add_hypothesis(h)
        assert len(experimenter.hypotheses) == 1

    def test_design_experiments_grid(self):
        experimenter = AutonomousExperimenter()
        space = ParameterSpace()
        space.add_parameter("height", [5.0, 10.0])
        space.add_parameter("mass", [1.0, 2.0])
        
        configs = experimenter.design_experiments(space, strategy="grid")
        assert len(configs) == 4

    def test_design_experiments_random(self):
        experimenter = AutonomousExperimenter()
        space = ParameterSpace()
        space.add_parameter("height", [5.0, 10.0, 15.0])
        
        configs = experimenter.design_experiments(space, n_experiments=2, strategy="random")
        assert len(configs) == 2

    def test_run_campaign(self):
        experimenter = AutonomousExperimenter()
        space = ParameterSpace()
        space.add_parameter("body0_y", [5.0, 10.0])
        
        def body_falls(dataset):
            y = dataset.get("body0.pos.y")
            return y is not None and len(y) > 1 and y[-1] < y[0]
        
        h = Hypothesis(
            name="falls",
            description="body falls",
            expected_behavior=body_falls
        )
        experimenter.add_hypothesis(h)
        
        results = experimenter.run_experiment_campaign(space, n_experiments=2)
        assert len(results) == 2
        assert h.confidence > 0

    def test_summary(self):
        experimenter = AutonomousExperimenter()
        summary = experimenter.summary()
        assert "Autonomous Experiment Campaign Summary" in summary
        assert "Total experiments: 0" in summary


class TestParameterSweep:
    def test_create_sweep(self):
        space, configs = create_parameter_sweep_experiment()
        assert len(configs) == 27  # 3 * 3 * 3
        assert len(space.parameters) == 3
