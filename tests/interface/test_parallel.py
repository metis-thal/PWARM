"""Tests for parallel environment wrappers (interface.parallel).

Scope: configuration validation, the single-process ``VectorizedEnv``,
and the multiprocessing ``ParallelEnv`` lifecycle (reset -> step -> states
-> close). The ray backend is only exercised through its ImportError
fallback (ray is an optional extra and is not installed in CI).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

import pwarm.physics  # noqa: F401  (breaks the interface<->physics cycle)
from pwarm.interface.parallel import EnvConfig, ParallelEnv, VectorizedEnv


def make_scene(cfg: EnvConfig) -> FakeScene:
    """Module-level factory: multiprocessing workers need picklable callables."""
    return FakeScene()


def test_env_config_defaults() -> None:
    """EnvConfig carries the documented physics defaults."""
    cfg = EnvConfig()
    assert cfg.gravity == (0.0, 0.0, -9.81)
    assert cfg.dt == pytest.approx(1 / 60)
    assert cfg.substeps == 1
    assert cfg.enable_rigid is True
    assert cfg.enable_sph is False
    assert cfg.custom is None


def test_parallel_env_config_count_mismatch() -> None:
    """A configs list whose length differs from num_envs is rejected."""
    with pytest.raises(ValueError, match="Expected 3 configs, got 1"):
        ParallelEnv(3, lambda cfg: object(), configs=[EnvConfig()])


def test_parallel_env_default_configs() -> None:
    """Without configs, one seeded EnvConfig per env is generated."""
    env = ParallelEnv(4, lambda cfg: object())
    assert [c.seed for c in env.configs] == [0, 1, 2, 3]
    assert env._initialized is False


@dataclass
class FakeState:
    t: float = 0.0
    step: int = 0
    rigid_pos: np.ndarray | None = field(default_factory=lambda: np.zeros(3))
    rigid_linvel: np.ndarray | None = field(default_factory=lambda: np.zeros(3))
    sph_pos: np.ndarray | None = None


class FakeScene:
    """Minimal duck-typed scene for VectorizedEnv and worker processes."""

    def __init__(self) -> None:
        self.double_buffer_write: FakeState | None = None
        self._step_count = 0

    def _create_initial_state(self) -> FakeState:
        return FakeState()

    def step(self) -> FakeState:
        self._step_count += 1
        return FakeState(t=float(self._step_count), step=self._step_count)

    def get_render_snapshot(self) -> FakeState:
        return FakeState(step=self._step_count)


def test_vectorized_env_reset_randomizes() -> None:
    """reset() creates one randomized state per seed."""
    env = VectorizedEnv(3, FakeScene())
    states = env.reset(seeds=[10, 20, 30])
    assert len(states) == 3
    assert len(env.states) == 3


def test_vectorized_env_step_before_reset_is_empty() -> None:
    """step()/get_observations() before reset() are safe no-ops."""
    env = VectorizedEnv(2, FakeScene())
    assert env.step() == []
    assert env.get_observations() == []


def test_vectorized_env_step_and_observe() -> None:
    """step() advances each state; observations concatenate pos+vel."""
    env = VectorizedEnv(2, FakeScene())
    env.reset(seeds=[1, 2])
    states = env.step()
    assert len(states) == 2
    # The fake scene counts steps globally across both env states
    assert states[0].step == 1 and states[1].step == 2
    obs = env.get_observations()
    # rigid_pos(3) + rigid_linvel(3); sph_pos is None -> excluded
    assert all(o.shape == (6,) for o in obs)


def test_vectorized_env_randomization_respects_none_fields() -> None:
    """_randomize_state skips fields that are None."""
    env = VectorizedEnv(1, FakeScene())
    state = FakeState(rigid_pos=None, rigid_linvel=None)
    env._randomize_state(state)  # must not raise


def test_parallel_env_multiprocessing_lifecycle() -> None:
    """mp backend: reset -> step -> get_states -> set_states -> close."""
    env = ParallelEnv(2, make_scene)
    env.initialize()
    assert env._initialized is True
    states = env.reset()
    assert len(states) == 2
    stepped = env.step()
    assert len(stepped) == 2
    again = env.get_states()
    assert len(again) == 2
    env.set_states([FakeState(t=9.0), FakeState(t=9.0)])
    env.close()
    assert all(not p.is_alive() for p in env._processes)


def test_parallel_env_context_manager() -> None:
    """The context manager initializes on entry and closes on exit."""
    with ParallelEnv(1, make_scene) as env:
        assert env._initialized is True
        states = env.reset()
        assert len(states) == 1
    assert all(not p.is_alive() for p in env._processes)


def test_parallel_env_actions_are_ignored() -> None:
    """step(actions=...) ignores the actions argument (documented)."""
    env = ParallelEnv(1, make_scene)
    env.initialize()
    env.reset()
    states = env.step(actions=[{"motor": 1.0}])
    assert len(states) == 1
    env.close()


def test_parallel_env_partial_reset() -> None:
    """reset(indices=[i]) resets only the requested envs."""
    env = ParallelEnv(3, make_scene)
    env.initialize()
    states = env.reset(indices=[2])
    assert len(states) == 1
    env.close()


def test_parallel_env_ray_fallback_to_multiprocessing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ray import falls back to the multiprocessing backend."""
    monkeypatch.setitem(sys.modules, "ray", None)  # `import ray` raises
    env = ParallelEnv(1, make_scene, backend="ray")
    env.initialize()
    assert env.backend == "multiprocessing"
    assert len(env.reset()) == 1
    env.close()
