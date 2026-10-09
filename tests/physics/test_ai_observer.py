"""Tests for the AI observation layer (physics.ai).

Scope: TimeSeriesDataset accessors, WorldObserver time-series collection
against the real WorldEngine, and the free-fall / collision experiment
helpers. The 3D observer extension and its SPH branch are covered with a
duck-typed engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.physics import WorldEngine, WorldEngineConfig
from pwarm.physics.ai import (
    Observation,
    TimeSeriesDataset,
    WorldObserver,
    WorldObserver3D,
    create_collision_experiment,
    create_free_fall_experiment,
)

G = 9.81
DT = 1 / 120


def make_engine() -> WorldEngine:
    config = WorldEngineConfig(dt=DT, substeps=1, gravity=(0.0, 0.0, -G),
                               rigid={"enabled": True})
    return WorldEngine(config)


def test_time_series_dataset_accessors() -> None:
    """get()/names()/time() expose observations; missing names return None."""
    obs = TimeSeriesDataset([
        Observation("time", np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        Observation("ke", np.array([0.0, 1.0]), np.array([5.0, 3.0])),
    ])
    assert obs.names() == ["time", "ke"]
    assert np.allclose(obs.get("ke"), [5.0, 3.0])
    assert obs.get("missing") is None
    assert np.allclose(obs.time(), [0.0, 1.0])
    assert TimeSeriesDataset().time().size == 0


def test_observer_records_free_fall_time_series() -> None:
    """The observer samples position/velocity/energy without disturbing physics."""
    engine = make_engine()
    engine.create_rigid_body((0.0, 0.0, 10.0), mass=1.0, shape="sphere",
                             shape_params={"radius": 0.5})
    engine.finalize_setup()
    observer = WorldObserver(engine, sample_every=5)
    data = observer.observe(n_steps=50)
    assert "rigid0.pos.z" in data.names()
    assert "rigid0.vel.z" in data.names()
    assert "total.ke" in data.names()
    z = data.get("rigid0.pos.z")
    assert z is not None and len(z) == 10  # 50 steps / sample_every 5
    assert z[-1] < z[0]  # the body fell
    vel = data.get("rigid0.vel.z")
    assert vel is not None and vel[-1] < vel[0] - 1.0
    # sampling never influences the ground truth: analytic free fall
    t = data.get("time")
    expected = 10.0 - 0.5 * G * (float(t[-1])) ** 2
    assert z[-1] == pytest.approx(expected, rel=2e-2)


def test_observer_without_global_quantities() -> None:
    """States without global_quantities still produce body observations."""
    engine = make_engine()
    engine.create_rigid_body((0.0, 0.0, 1.0), mass=1.0, shape="sphere",
                             shape_params={"radius": 0.5})
    engine.finalize_setup()
    observer = WorldObserver(engine)
    data = observer.observe(n_steps=3)
    assert "rigid0.pos.x" in data.names()
    assert np.allclose(data.get("rigid0.pos.x"), 0.0)


def test_observer_empty_scene_records_zero_globals() -> None:
    """An engine with no bodies still records the (zero) global channels."""
    engine = make_engine()
    engine.finalize_setup()
    data = WorldObserver(engine).observe(n_steps=3)
    assert data.names() == ["time", "total.ke",
                            "total.momentum.x", "total.momentum.y",
                            "total.momentum.z"]
    assert np.allclose(data.get("total.ke"), 0.0)
    assert len(data.get("time")) == 3  # type: ignore[index]


def _with_get_state(engine: WorldEngine) -> WorldEngine:
    """WorldObserver3D expects engine.get_state(); WorldEngine lacks it, so
    provide the intended contract (read the current read buffer)."""
    engine.get_state = lambda: engine.scene.double_buffer_read  # type: ignore[attr-defined]
    return engine


def test_observer_3d_extension_returns_base_data() -> None:
    """WorldObserver3D replays the run and returns the base dataset."""
    engine = _with_get_state(make_engine())
    engine.create_rigid_body((0.0, 0.0, 10.0), mass=1.0, shape="sphere",
                             shape_params={"radius": 0.5})
    engine.finalize_setup()
    data = WorldObserver3D(engine, sample_every=2).observe(n_steps=6)
    assert "rigid0.pos.z" in data.names()


def test_observer_3d_with_sph_particles() -> None:
    """The SPH branch of the 3D observer runs against an SPH-enabled engine."""
    engine = _with_get_state(make_engine())
    positions = np.array([[x, y, z]
                          for x in (0.0, 0.2)
                          for y in (0.0, 0.2)
                          for z in (5.0, 5.2)])
    engine.create_sph_fluid(positions)
    engine.finalize_setup()
    data = WorldObserver3D(engine, sample_every=1).observe(n_steps=3)
    # SPH aggregates are placeholders today: no sph.* channels are emitted
    assert not any(name.startswith("sph") for name in data.names())
    assert "time" in data.names()


def test_create_free_fall_experiment() -> None:
    """The helper wires body + observer and returns the fall dataset."""
    engine = make_engine()
    data = create_free_fall_experiment(engine, height=5.0, n_steps=60)
    z = data.get("rigid0.pos.z")
    assert z is not None and z[-1] < z[0] - 1.0


def test_create_collision_experiment() -> None:
    """Two approaching bodies are observed and their KE exchange recorded."""
    engine = make_engine()
    data = create_collision_experiment(engine, n_steps=120)
    names = data.names()
    assert "rigid0.pos.x" in names and "rigid1.pos.x" in names
    x0 = data.get("rigid0.pos.x")
    assert x0 is not None and x0[-1] > x0[0]  # body 0 moved in +x
