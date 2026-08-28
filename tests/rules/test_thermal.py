"""Tests for the thermodynamics rules module (P2.2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from pymo.kernel.bodies import Material, circle_body
from pymo.rules.thermal import (
    BodyThermalSystem,
    TemperatureField,
    diffuse_field,
    heat_capacity,
    internal_energy,
)


def test_heat_capacity_and_energy():
    """C = m*c; internal energy E = C*T."""
    m = Material(specific_heat=500.0)
    b = circle_body([0.0, 0.0], 0.5, mass=2.0, material=m)
    b.temperature = 300.0
    assert heat_capacity(b) == pytest.approx(2.0 * 500.0)
    assert internal_energy(b) == pytest.approx(2.0 * 500.0 * 300.0)


def test_conduction_reaches_equilibrium_and_conserves_energy():
    """Two contacting conductors equalize temperature; total energy conserved."""
    m = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    hot = circle_body([0.0, 0.0], 0.5, mass=1.0, material=m)
    cold = circle_body([1.0, 0.0], 0.5, mass=1.0, material=m)
    hot.temperature = 400.0
    cold.temperature = 300.0

    e0 = internal_energy(hot) + internal_energy(cold)
    system = BodyThermalSystem()
    for _ in range(60000):
        system.conduct_between(hot, cold, dt=0.001)

    # Reached equilibrium (equal temperatures)
    assert hot.temperature == pytest.approx(cold.temperature, abs=1.0)
    # Equilibrium temp = mass-weighted average = 350 for equal masses
    assert hot.temperature == pytest.approx(350.0, abs=1.0)
    # Energy conserved
    assert internal_energy(hot) + internal_energy(cold) == pytest.approx(e0, rel=1e-6)


def test_insulator_blocks_conduction():
    """A body with zero conductivity does not conduct heat."""
    ins = Material(thermal_conductivity=0.0, specific_heat=1000.0)
    cond = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    hot = circle_body([0.0, 0.0], 0.5, mass=1.0, material=ins)
    cold = circle_body([1.0, 0.0], 0.5, mass=1.0, material=cond)
    hot.temperature = 400.0
    cold.temperature = 300.0
    system = BodyThermalSystem()
    q = system.conduct_between(hot, cold, dt=0.001)
    assert q == 0.0
    assert hot.temperature == pytest.approx(400.0)
    assert cold.temperature == pytest.approx(300.0)


def test_conduction_heat_flows_hot_to_cold():
    """Heat flows from hot to cold: hot cools, cold warms."""
    m = Material(thermal_conductivity=10.0, specific_heat=1000.0)
    hot = circle_body([0.0, 0.0], 0.5, mass=1.0, material=m)
    cold = circle_body([1.0, 0.0], 0.5, mass=1.0, material=m)
    hot.temperature = 500.0
    cold.temperature = 250.0
    system = BodyThermalSystem()
    q = system.conduct_between(hot, cold, dt=0.01)
    assert q > 0  # positive = hot loses, cold gains
    assert hot.temperature < 500.0
    assert cold.temperature > 250.0


def test_field_diffusion_conserves_energy():
    """Diffusion conserves total field energy (sum of T)."""
    field = TemperatureField(nx=20, ny=20, dx=0.1, dy=0.1)
    # Hot spot in the center
    field.set(1.0, 1.0, 500.0)
    e0 = field.total_energy()
    alpha = 0.5
    dt = 0.001
    assert dt < field.dx**2 / (4.0 * alpha)  # stability condition
    for _ in range(1000):
        diffuse_field(field, alpha, dt)
    # Energy conserved
    assert field.total_energy() == pytest.approx(e0, rel=1e-6)
    # Heat spread out: center cooled, edges warmed
    assert field.temperatures[10, 10] < 500.0


def test_field_diffusion_converges_to_flat():
    """Diffusion smooths the field toward a uniform steady state."""
    field = TemperatureField(nx=16, ny=16, dx=0.1, dy=0.1)
    field.set(0.8, 0.8, 300.0)
    alpha = 0.5
    dt = 0.001
    for _ in range(30000):
        diffuse_field(field, alpha, dt)
    # Nearly uniform
    T = field.temperatures
    std = np.std(T)
    assert std < 1.0
