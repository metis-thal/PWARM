"""Tests for thermal-physics integration (heat conducts across contacts)."""

from __future__ import annotations

import numpy as np
import pytest

from pymo.kernel.bodies import Material, box_body, circle_body
from pymo.kernel.world import World
from pymo.rules.thermal import BodyThermalSystem, internal_energy


def test_heat_conducts_between_contacting_bodies_in_world():
    """Two bodies in contact in a simulated world exchange heat."""
    m = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    # A hot circle resting on a cold ground box (they stay in contact)
    ground = box_body([0.0, -0.5], 5.0, 0.5, static=True, material=m)
    ball = circle_body([0.0, 0.2], 0.5, mass=1.0, material=m)
    ball.temperature = 500.0
    ground.temperature = 300.0

    thermal = BodyThermalSystem()
    w = World(gravity=np.array([0.0, -9.81]), dt=1 / 120.0, thermal=thermal)
    w.add(ground, ball)

    e0 = internal_energy(ground) + internal_energy(ball)
    w.step(300)  # ball falls onto ground and stays in contact

    # Ball cooled, ground warmed (heat flowed ball -> ground)
    assert ball.temperature < 500.0
    assert ground.temperature > 300.0
    # Energy conserved across the coupled system
    assert internal_energy(ground) + internal_energy(ball) == pytest.approx(e0, rel=1e-6)


def test_no_conduction_without_thermal_system():
    """Without a thermal system, temperatures are unchanged during simulation."""
    m = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    ground = box_body([0.0, -0.5], 5.0, 0.5, static=True, material=m)
    ball = circle_body([0.0, 0.2], 0.5, mass=1.0, material=m)
    ball.temperature = 500.0
    ground.temperature = 300.0
    w = World(gravity=np.array([0.0, -9.81]), dt=1 / 120.0)  # no thermal
    w.add(ground, ball)
    w.step(300)
    assert ball.temperature == pytest.approx(500.0)
    assert ground.temperature == pytest.approx(300.0)


def test_insulating_contact_blocks_heat_in_world():
    """A zero-conductivity body blocks heat flow even in contact."""
    m_ins = Material(thermal_conductivity=0.0, specific_heat=1000.0)
    m_cond = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    ground = box_body([0.0, -0.5], 5.0, 0.5, static=True, material=m_cond)
    ball = circle_body([0.0, 0.2], 0.5, mass=1.0, material=m_ins)  # insulator
    ball.temperature = 500.0
    ground.temperature = 300.0
    thermal = BodyThermalSystem()
    w = World(gravity=np.array([0.0, -9.81]), dt=1 / 120.0, thermal=thermal)
    w.add(ground, ball)
    w.step(300)
    # No conduction: ball stays hot, ground stays cold
    assert ball.temperature == pytest.approx(500.0, abs=1.0)
    assert ground.temperature == pytest.approx(300.0, abs=1.0)
