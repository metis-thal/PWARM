"""Reproducibility verification — the scientific claims are CI-checked.

For each mission envelope: run a FRESH scientist through the same entry
point the CLI uses, then assert the derived hypotheses sit within threshold
of the hidden truth. Ground truth is read here on the EVALUATION side only
(this test is the human-facing narrator, like scientific_evaluation/) — the
AI itself never sees it (enforced by tests/scientist/test_api_contract.py).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / "reproducibility"

# Evaluation-side thresholds: 001's parabola fits resolve g to ~1e-6 relative,
# material properties are trusted to the knowledge threshold (5% rel width),
# matching what the scientist itself claims.
_THRESHOLD = {"001": 1e-3, "002": 0.05, "003": 0.05}
_UNIVERSE = {"001": "universe_001", "002": "universe_002", "003": "universe_003"}


def _run_envelope(mission: str) -> tuple:
    with tempfile.TemporaryDirectory() as td:
        sys.path.insert(0, str(_ROOT / "src"))
        from pymo.cli import _run
        result = _run(mission, Path(td) / "knowledge.json")
    report = result["report"]
    assert report.status == "DISCOVERED", report.summary[-1500:]
    return report, result["truth"]


def _truth() -> dict:
    from pymo.universes import load_universe
    truth: dict = {}
    for uid in _UNIVERSE.values():
        u = load_universe(uid)
        if u.secrets.gravity is not None:
            truth[f"{uid}.gravity"] = u.secrets.gravity
        for name, props in u.secrets.materials.items():
            for prop, value in props.items():
                truth[f"{uid}.{name}.{prop}"] = value
    return truth


@pytest.mark.parametrize("mission", ["001", "002", "003"])
def test_envelope_rediscovers_truth(mission: str) -> None:
    report, _ = _run_envelope(mission)
    truth = _truth()
    assert report.hypotheses, "no hypotheses derived"
    for hyp in report.hypotheses:
        key = f"{_UNIVERSE[mission]}.{hyp['claim']}"
        assert key in truth, f"claim {key} is not a hidden parameter"
        err = abs(hyp["value"] - truth[key]) / abs(truth[key])
        assert err < _THRESHOLD[mission], (key, hyp["value"], truth[key], err)
        assert hyp["r2"] > 0.95


def test_mission_002_honestly_reports_density() -> None:
    report, _ = _run_envelope("002")
    claims = {h["claim"] for h in report.hypotheses}
    # The equivalence-principle gap must be visible, never papered over.
    assert not any(c.endswith(".density") for c in claims)
    assert "unidentifiable" in report.summary.lower()
    assert "material_A.density" in report.summary


def test_mission_003_runs_the_instrument_arc() -> None:
    report, _ = _run_envelope("003")
    claims = {h["claim"] for h in report.hypotheses}
    assert "material_A.density" in claims       # the arc closes the gap
    assert "fluid_tank" in report.summary


def test_ai_artifacts_carry_no_secrets() -> None:
    """Structural leak check: artifacts must contain measurements, never
    answers — no secrets plumbing, and (in the mission where density is
    unidentifiable) no density numbers at all."""
    truth = _truth()
    density_secrets = {k: v for k, v in truth.items()
                       if k.endswith(".density") and "universe_002" in k}
    for mission in ("001", "002", "003"):
        results = _ENV / f"mission_{mission}" / "results"
        for path in results.glob("*.json"):
            blob = path.read_text(encoding="utf-8")
            for marker in ("UniverseSecrets", "ground_truth", '"secrets"'):
                assert marker not in blob, (path, marker)
    for path in (_ENV / "mission_002" / "results").glob("*.json"):
        blob = path.read_text(encoding="utf-8")
        for key, value in density_secrets.items():
            assert f"{value:g}" not in blob, f"secret {key} leaked into {path}"
