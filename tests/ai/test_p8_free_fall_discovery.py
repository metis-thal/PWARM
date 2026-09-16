"""P8 closed-loop verification: physics -> observation -> AI discovery -> evaluation.

The AI only ever sees recorded (t, z) samples from the engine; it must recover
g from data alone. These tests pin the contract that makes P8 the project's
"undeniable result": the discovered constant matches ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from experiments.free_fall_ai.experiment import FreeFallConfig, FreeFallExperiment


@pytest.fixture(scope="module")
def report():
    exp = FreeFallExperiment(FreeFallConfig(height=10.0, mass=1.0))
    return exp.run_headless(n_steps=100)


def test_discovers_quadratic_law(report):
    assert report.expression, "AI produced no expression"
    assert "t^2" in report.expression.replace(" ", ""), report.expression
    assert report.r2 > 0.9999


def test_recovers_gravity_within_tolerance(report):
    # The user-visible target is ~0.003% on clean data; allow 0.1%.
    assert report.g_ai == pytest.approx(9.81, rel=1e-3)
    assert report.error_pct < 0.1


def test_coefficients_match_free_fall(report):
    # z(t) = c + b*t + a*t^2 with a = -g/2 exactly.
    assert report.a == pytest.approx(-9.81 / 2.0, rel=1e-3)
    assert report.c == pytest.approx(10.0, abs=1e-2)
    # Symplectic Euler adds a small linear drift term b ~ -g*dt/2 (~-0.082),
    # so b is near zero but not machine-zero.
    assert abs(report.b) < 0.2


def test_landing_detected(report):
    assert report.phase in ("LANDED", "VERIFIED")


def test_discover_before_samples_is_safe():
    exp = FreeFallExperiment(FreeFallConfig())
    exp.setup()
    rep = exp.discover()
    assert rep.phase == "OBSERVING"
    assert rep.n_samples == 0
