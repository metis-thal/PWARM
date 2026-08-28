"""Tests for the AI closed-loop reasoning module."""

from __future__ import annotations

import numpy as np

from pymo.ai.closed_loop import ClosedLoopAI
from pymo.ai.law_discovery import LawDiscovery
from pymo.ai.observer import WorldObserver, collect_free_fall
from pymo.kernel.bodies import circle_body
from pymo.kernel.world import World


def test_observer_collects_free_fall_series():
    """The observer records position/velocity time series for a falling body."""
    data = collect_free_fall(n_steps=100, dt=0.01, g=9.81)
    assert "body0.pos.y" in data.names()
    y = data.get("body0.pos.y")
    assert y is not None and len(y) == 100
    # y should decrease (falling) and match 0.5*g*t^2 approximately
    assert y[0] > y[-1]
    # velocity increases downward
    vy = data.get("body0.vel.y")
    assert vy is not None and vy[-1] < vy[0] < 0


def test_law_discovery_recovers_gravity():
    """Symbolic regression recovers g from free-fall data (P0.2 validated)."""
    t = np.linspace(0, 1, 200)
    g = 9.81
    y = 10.0 - 0.5 * g * t**2
    law = LawDiscovery().discover(t, y)
    assert law.r2 > 0.999
    # The discovered law should predict accurately
    pred = law.predict(t)
    assert np.allclose(pred, y, atol=0.5)


def test_closed_loop_free_fall_prediction_under_5pct():
    """The closed loop predicts held-out free-fall positions with <5% error.

    This is the Phase 1 acceptance criterion.
    """
    data = collect_free_fall(n_steps=300, dt=0.01, g=9.81)
    ai = ClosedLoopAI(train_fraction=0.6)
    results = ai.run(data, quantities=["body0.pos.y"])
    assert len(results) == 1
    r = results[0]
    assert r.passed(threshold=0.05), f"rel err {r.relative_error} >= 5%"
    # The discovered law should be a downward quadratic
    assert "t^2" in r.law.expression


def test_closed_loop_velocity_prediction():
    """Velocity prediction also works (linear law v = -g*t)."""
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    w.add(circle_body([0.0, 10.0], 0.5, mass=1.0))
    data = WorldObserver(w).observe(200)
    ai = ClosedLoopAI(train_fraction=0.6)
    results = ai.run(data, quantities=["body0.vel.y"])
    assert len(results) == 1
    assert results[0].passed(threshold=0.05)
