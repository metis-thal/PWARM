"""Mission 004 Day 1 verification: the unknown liquid becomes measurable.

Day 1 delivers the measurement infrastructure only: universe_004 hides an
ambient fluid density; the immersion rig releases a certified reference
sphere (known density — an instrument specification) into it; the existing
hydrostatic buoyancy physics produces a descent record. No derive, no
claim routing, no law discovery yet — those are Day 2.

Truth-handling rule for this file: universe secrets are read HERE, on the
human evaluation side, to verify the physics is right. No truth value may
flow into any AI-side object (records, REGISTRY modules, knowledge).
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from pymo.scientist import (
    REGISTRY,
    ExperimentSpec,
    Laboratory,
    ObservationRecord,
)
from pymo.scientist.experiments import immersion_test
from pymo.universes import load_universe


def _universe():
    return load_universe("universe_004")


@pytest.fixture(scope="module")
def lab():
    return Laboratory(_universe())


# --- human evaluation side ONLY: the truth, for verifying the physics ---
@pytest.fixture(scope="module")
def truth():
    u = _universe()
    return {"g": float(u.secrets.gravity),
            "rho_fluid": float(u.secrets.materials["ambient_fluid"]["density"])}


def test_immersion_experiment_runs_and_returns_record(lab):
    rec = lab.run_experiment(
        ExperimentSpec(kind="immersion_test", drop_height=2.0))
    assert isinstance(rec, ObservationRecord)
    assert rec.experiment_id == "immersion_test_d2"
    assert len(rec.t) > 10 and len(rec.t) == len(rec.z)
    assert rec.steps > 0
    assert rec.vx is None           # no horizontal channel in this apparatus
    assert rec.t[-1] > rec.t[0]     # a real time series
    assert rec.z[0] > rec.z[-1]     # the sphere descends to the tank floor


def test_record_is_measurement_only(lab, truth):
    """The record must not carry the ambient fluid's density — by field,
    by attribute name, or by value leaking through repr."""
    rec = lab.run_experiment(
        ExperimentSpec(kind="immersion_test", drop_height=2.0))
    assert {f.name for f in dataclasses.fields(ObservationRecord)} == {
        "experiment_id", "t", "z", "vx", "steps"}
    blob = repr(rec)
    assert "density" not in blob
    assert "ambient" not in blob
    assert "fluid" not in blob
    # the hidden VALUE itself (read human-side above) must not appear
    assert str(truth["rho_fluid"]) not in blob


def test_same_design_is_deterministic(lab):
    spec = ExperimentSpec(kind="immersion_test", drop_height=4.0)
    a, b = lab.run_experiment(spec), lab.run_experiment(spec)
    assert a.experiment_id == b.experiment_id
    assert a.steps == b.steps
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.z, b.z)


def test_trajectory_matches_hydrostatic_buoyancy(lab, truth):
    """z(t) is the descent parabola with curvature
    a = g (1 - rho_fluid / rho_sample): Mission 003's verified mechanism,
    with the KNOWN density now on the sample side (truth read human-side)."""
    rec = lab.run_experiment(
        ExperimentSpec(kind="immersion_test", drop_height=2.0))
    a_expected = truth["g"] * (1.0 - truth["rho_fluid"]
                               / immersion_test.SAMPLE_DENSITY)
    c2, c1, c0 = np.polyfit(rec.t, rec.z, 2)
    a_fit = -2.0 * float(c2)
    assert a_fit == pytest.approx(a_expected, rel=0.01)
    pred = np.polynomial.polynomial.polyval(rec.t, [c0, c1, c2])
    r2 = 1.0 - float(np.sum((rec.z - pred) ** 2)) / (
        float(np.sum((rec.z - rec.z.mean()) ** 2)) + 1e-12)
    assert r2 > 0.999


def test_registered_module_follows_day1_protocol(truth):
    """Registered for EXECUTION only: design grid + sample standard. The
    designer-facing derive/claim protocol is Day 2 — it must NOT exist yet,
    and the certified standard must never alias the hidden liquid."""
    assert REGISTRY["immersion_test"] is immersion_test
    assert not hasattr(immersion_test, "derive")      # Day 2
    assert not hasattr(immersion_test, "claim_of")    # Day 2
    grid = immersion_test.design_grid()
    assert grid == [{"depth": float(d)} for d in immersion_test.DESIGN_DEPTHS]
    for design in grid:
        spec = ExperimentSpec(kind="immersion_test",
                              drop_height=design["depth"])
        assert spec.id.startswith("immersion_test_d")
    # the standard is an independent instrument value, not the hidden
    # liquid's density under another name (human-side comparison)
    assert immersion_test.SAMPLE_DENSITY != truth["rho_fluid"]


def test_immersion_module_never_imports_universes():
    """The new AI-side module obeys the structural contract: no truth
    access, not even by import (mirrors the test_api_contract guard)."""
    src = Path(immersion_test.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pymo.universes")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("pymo.universes")
    assert "secrets" not in src      # contract guard for experiments/ modules


def test_universe_without_ambient_fluid_is_refused():
    """The rig needs the ambient liquid to exist (physics-side check); the
    error names the missing PARAMETER, never a value."""
    lab = Laboratory(load_universe("universe_001"))
    with pytest.raises(ValueError, match="ambient_fluid"):
        lab.run_experiment(
            ExperimentSpec(kind="immersion_test", drop_height=2.0))


def test_mission_003_still_measures_density():
    """Regression canary: immersion shares the Fluid Tank session code —
    Mission 003's buoyancy payoff must be untouched."""
    lab = Laboratory(load_universe("universe_003"))
    rec = lab.run_experiment(ExperimentSpec(
        kind="buoyancy_test", drop_height=2.0, material="material_A"))
    hypothesis, _ = REGISTRY["buoyancy_test"].derive(rec, 9.79, "material_A")
    assert hypothesis.value == pytest.approx(7800.0, rel=0.05)
    assert hypothesis.r2 > 0.99
