"""Tests for the uncertainty bookkeeping and mission reporting modules.

Scope: relative width, the three certainty classes, the aggregated
uncertainty gauge, the dashboard knowledge lines, and the Mission/MissionReport
value objects (frozen mission, report formatting branches).
"""

from __future__ import annotations

import dataclasses

import pytest

from pwarm.scientist.mission import Mission, MissionReport
from pwarm.scientist.state import Belief
from pwarm.scientist.uncertainty import (
    certainty_class,
    knowledge_lines,
    relative_width,
    total_uncertainty,
)


class FakeState:
    """Duck-typed ScientistState: only .beliefs is read."""

    def __init__(self, beliefs: dict[str, Belief]) -> None:
        self.beliefs = beliefs


def test_relative_width_fresh_prior() -> None:
    """A prior spanning [0, 2] around 1 has relative width 2."""
    belief = Belief(name="g", lo=0.0, hi=2.0)
    assert relative_width(belief) == pytest.approx(2.0)


def test_relative_width_narrow_interval() -> None:
    """A tight interval around a large value has a small relative width."""
    belief = Belief(name="g", lo=9.8, hi=10.0)
    assert relative_width(belief) == pytest.approx(0.2 / 9.9)


def test_relative_width_zero_midpoint_guard() -> None:
    """A zero midpoint falls back to the 1e-9 guard instead of dividing by 0."""
    belief = Belief(name="x", lo=-1.0, hi=1.0)
    assert relative_width(belief) == pytest.approx(2.0 / 1e-9)


def test_certainty_classes() -> None:
    """Status maps directly to the three certainty classes."""
    assert certainty_class(Belief(name="a", lo=0, hi=1, status="known")) == "known"
    assert certainty_class(Belief(name="a", lo=0, hi=1,
                                  status="unidentifiable")) == "unidentifiable"
    assert certainty_class(Belief(name="a", lo=0, hi=1,
                                  status="constrained")) == "measurable"
    assert certainty_class(Belief(name="a", lo=0, hi=1,
                                  status="unknown")) == "measurable"


def test_total_uncertainty_sums_clamped_widths() -> None:
    """Non-known beliefs contribute min(width, 1); known ones nothing."""
    state = FakeState({
        "known": Belief(name="known", lo=9.99, hi=10.0, status="known"),
        "wide": Belief(name="wide", lo=0.0, hi=5.0, status="unknown"),
        "narrow": Belief(name="narrow", lo=9.9, hi=10.1, status="constrained"),
    })
    # wide: width 5 clamped to 1.0; narrow: 0.2/10.0 = 0.02
    assert total_uncertainty(state) == pytest.approx(1.0 + 0.02)


def test_total_uncertainty_all_known_is_zero() -> None:
    """A scientist who knows everything has zero remaining uncertainty."""
    state = FakeState({"g": Belief(name="g", lo=9.8, hi=10.0, status="known")})
    assert total_uncertainty(state) == 0.0


def test_knowledge_lines_formats_by_class() -> None:
    """Each certainty class renders its own dashboard line and colour."""
    state = FakeState({
        "alpha": Belief(name="alpha", lo=9.8, hi=10.0, status="known"),
        "beta": Belief(name="beta", lo=0.0, hi=2.0, status="unknown"),
        "gamma": Belief(name="gamma", lo=0.0, hi=0.0, status="unidentifiable"),
    })
    lines = dict(knowledge_lines(state))  # keyed by rendered text
    texts = list(lines)
    assert any("alpha = 9.9" in t and "✓" in t for t in texts)
    assert any("beta in [0, 2" in t for t in texts)
    assert any("gamma = ?  UNIDENTIFIABLE" in t for t in texts)
    # sorted by parameter name
    assert texts == sorted(texts)


# ---------------------------------------------------------------------- mission
def test_mission_is_frozen() -> None:
    """Missions are immutable value objects."""
    mission = Mission(id="001", title="Gravity", objective="Find g",
                      target="gravity", unit="m/s^2")
    with pytest.raises(dataclasses.FrozenInstanceError):
        mission.target = "something else"  # type: ignore[misc]


def test_mission_report_defaults() -> None:
    """A fresh report is INCOMPLETE-ish: zeroed counters, no value."""
    report = MissionReport(mission_id="001", status="INCOMPLETE")
    assert report.experiments_run == 0
    assert report.value is None
    assert report.knowledge_saved is False


def test_mission_report_str_without_value() -> None:
    """Without a value the report skips the formula/confidence lines."""
    report = MissionReport(mission_id="001", status="INCOMPLETE")
    text = str(report)
    assert text.startswith("Mission 001: INCOMPLETE")
    assert "experiments run: 0" in text
    assert "confidence" not in text


def test_mission_report_str_with_value() -> None:
    """With a value the report shows formula, value, unit and confidence."""
    report = MissionReport(
        mission_id="001", status="DISCOVERED", experiments_run=4,
        value=9.81, unit="m/s^2", confidence=0.999,
        formula="y = 0.5 * g * t^2", knowledge_saved=True,
        summary="gravity established")
    text = str(report)
    assert "Mission 001: DISCOVERED" in text
    assert "y = 0.5 * g * t^2" in text
    assert "9.810000 m/s^2" in text
    assert "99.90%" in text
    assert "knowledge saved: True" in text
    assert text.endswith("gravity established")


def test_mission_report_str_unitless() -> None:
    """An empty unit renders without a trailing unit label."""
    report = MissionReport(mission_id="002", status="DISCOVERED", value=0.5,
                           confidence=0.99)
    assert "0.500000" in str(report)
