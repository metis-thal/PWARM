"""Tests for the body thermal rule module (rules.thermal).

``pwarm.kernel`` is stubbed by tests/conftest.py. Scope: conduction between
bodies (insulator / zero-capacity / equilibrium early-returns, harmonic
conductivity, stability clip, energy conservation), contact iteration over
duck-typed pairs, pairwise stepping with a contact predicate, and the
temperature-field diffusion step.
"""

from __future__ import annotations

import numpy as np
import pytest
from pwarm.kernel.bodies import Body, Material

from pwarm.rules.thermal import (
    BodyThermalSystem,
    TemperatureField,
    diffuse_field,
    heat_capacity,
    internal_energy,
)


def make_body(mass: float = 1.0, conductivity: float = 1.0,
              specific_heat: float = 1000.0, temperature: float = 300.0) -> Body:
    return Body(pos=np.zeros(2), mass=mass,
                material=Material(thermal_conductivity=conductivity,
                                  specific_heat=specific_heat),
                temperature=temperature)


def test_heat_capacity_and_internal_energy() -> None:
    """heat_capacity = m*c; internal_energy = m*c*T."""
    body = make_body(mass=2.0, specific_heat=500.0, temperature=300.0)
    assert heat_capacity(body) == pytest.approx(1000.0)
    assert internal_energy(body) == pytest.approx(300_000.0)


def test_conduct_between_insulators_blocked() -> None:
    """A zero-conductivity body blocks conduction entirely."""
    a = make_body(temperature=400.0, conductivity=1.0)
    b = make_body(temperature=300.0, conductivity=0.0)
    q = BodyThermalSystem().conduct_between(a, b, dt=1.0)
    assert q == 0.0
    assert a.temperature == 400.0 and b.temperature == 300.0


def test_conduct_between_zero_capacity_blocked() -> None:
    """A zero-mass (or zero-c) body participates in no heat exchange."""
    a = make_body(temperature=400.0)
    b = make_body(mass=0.0, temperature=300.0)
    assert BodyThermalSystem().conduct_between(a, b, dt=1.0) == 0.0
    assert a.temperature == 400.0


def test_conduct_between_equilibrium_noop() -> None:
    """Bodies already at the same temperature exchange nothing."""
    a = make_body(temperature=350.0)
    b = make_body(temperature=350.0)
    assert BodyThermalSystem().conduct_between(a, b, dt=1.0) == 0.0


def test_conduct_between_conserves_energy() -> None:
    """Heat lost by the hot body equals heat gained by the cold one."""
    a = make_body(mass=2.0, temperature=400.0)
    b = make_body(mass=3.0, temperature=300.0)
    ca, cb = heat_capacity(a), heat_capacity(b)
    e0 = ca * a.temperature + cb * b.temperature
    q = BodyThermalSystem(contact_area=1.0, contact_distance=0.1).conduct_between(a, b, dt=1.0)
    assert q > 0.0  # hot -> cold
    assert a.heat == pytest.approx(-q)
    assert b.heat == pytest.approx(q)
    assert ca * a.temperature + cb * b.temperature == pytest.approx(e0)


def test_conduct_between_harmonic_conductivity() -> None:
    """k_eff is the harmonic mean: equal conductivities -> k_eff = k."""
    a = make_body(temperature=400.0, conductivity=2.0)
    b = make_body(temperature=300.0, conductivity=2.0)
    # q = k_eff * area * dT / distance * dt = 2*1*100/0.1*1 = 2000 J
    q = BodyThermalSystem(contact_area=1.0, contact_distance=0.1).conduct_between(a, b, dt=1.0)
    assert q == pytest.approx(2000.0)


def test_conduct_between_stability_clip() -> None:
    """The flux is clipped to half the hot body's heat capacity * dT."""
    a = make_body(mass=1.0, temperature=310.0)  # ca = 1000, max_q = 500
    b = make_body(mass=1e9, temperature=300.0)  # huge cold reservoir
    q = BodyThermalSystem(contact_area=1e6, contact_distance=1e-6).conduct_between(a, b, dt=1e6)
    assert q <= 0.5 * heat_capacity(a) * 10.0 + 1e-6


def test_step_contacts_duck_typed_and_tuples() -> None:
    """step_contacts accepts contact objects (.a/.b) and (a, b) tuples."""
    system = BodyThermalSystem()
    a = make_body(temperature=400.0)
    b = make_body(temperature=300.0)
    c = make_body(temperature=200.0)
    d = make_body(temperature=250.0)

    class ObjContact:
        def __init__(self, a: Body, b: Body) -> None:
            self.a = a
            self.b = b

    total = system.step_contacts([a, b, c, d],
                                 [ObjContact(a, b), (c, d)], dt=1.0)
    assert total > 0.0  # (a, b) transfers +1000 J, (c, d) transfers -500 J
    assert a.temperature < 400.0 and b.temperature > 300.0
    # positive q means a loses / b gains: cold c warms, warm d cools
    assert c.temperature > 200.0 and d.temperature < 250.0


def test_step_contacts_empty() -> None:
    """No contacts -> zero total heat."""
    assert BodyThermalSystem().step_contacts([make_body()], [], dt=1.0) == 0.0


def test_step_pairwise_with_predicate() -> None:
    """step_pairwise conducts only pairs accepted by the contact predicate."""
    system = BodyThermalSystem()
    a = make_body(temperature=400.0)
    b = make_body(temperature=300.0)
    c = make_body(temperature=200.0)
    total = system.step_pairwise([a, b, c], dt=1.0,
                                 contact_fn=lambda x, y: y is c)
    # only (a, c) and (b, c) conduct; (a, b) is filtered out
    assert a.temperature < 400.0
    assert b.temperature < 300.0
    assert total > 0.0


def test_step_pairwise_all_pairs() -> None:
    """Without a predicate every unordered pair conducts."""
    system = BodyThermalSystem()
    a = make_body(temperature=400.0, conductivity=10.0)
    b = make_body(temperature=300.0, conductivity=10.0)
    total = system.step_pairwise([a, b], dt=1.0)
    # compare against a fresh pair (the originals just equilibrated)
    expected = BodyThermalSystem().conduct_between(
        make_body(temperature=400.0, conductivity=10.0),
        make_body(temperature=300.0, conductivity=10.0), dt=1.0)
    assert total == pytest.approx(expected)


def test_temperature_field_index_clamps() -> None:
    """Out-of-domain coordinates clamp to the field bounds."""
    field = TemperatureField(nx=4, ny=3, dx=0.5, dy=0.5)
    assert field.index(-10.0, -10.0) == (0, 0)
    assert field.index(100.0, 100.0) == (3, 2)
    assert field.index(1.1, 0.6) == (2, 1)  # truncated, then clipped


def test_temperature_field_set_and_energy() -> None:
    """set() writes y-major; total_energy sums the field."""
    field = TemperatureField(nx=4, ny=3, dx=0.5, dy=0.5)
    field.set(1.0, 2.0, 500.0)  # ix=2, iy=4->clip 2
    assert field.temperatures[2, 2] == 500.0
    assert field.total_energy() == pytest.approx(500.0)


def test_diffuse_field_uniform_is_stationary() -> None:
    """A uniform field does not change under diffusion."""
    field = TemperatureField(nx=6, ny=5, dx=0.1, dy=0.1)
    field.temperatures[:] = 300.0
    change = diffuse_field(field, alpha=1e-4, dt=0.1)
    assert change == 0.0
    assert np.all(field.temperatures == 300.0)


def test_diffuse_field_conserves_total() -> None:
    """Total 'energy' is conserved across diffusion steps (Neumann BCs)."""
    field = TemperatureField(nx=8, ny=8, dx=0.1, dy=0.1)
    field.set(0.4, 0.4, 600.0)
    e0 = field.total_energy()
    for _ in range(20):
        diffuse_field(field, alpha=1e-3, dt=0.01)
    assert field.total_energy() == pytest.approx(e0, rel=1e-5)


def test_diffuse_field_zero_alpha_noop() -> None:
    """alpha = 0 leaves the field untouched."""
    field = TemperatureField(nx=4, ny=4, dx=0.1, dy=0.1)
    field.temperatures[2, 2] = 900.0
    before = field.temperatures.copy()
    assert diffuse_field(field, alpha=0.0, dt=0.1) == 0.0
    assert np.array_equal(field.temperatures, before)


def test_diffuse_field_spreads_hot_spot() -> None:
    """A hot spot cools while its neighbours warm up."""
    field = TemperatureField(nx=7, ny=7, dx=0.1, dy=0.1)
    field.temperatures[3, 3] = 500.0  # centre cell (avoid float grid traps)
    before = field.temperatures.copy()
    diffuse_field(field, alpha=1e-2, dt=0.05)
    assert field.temperatures[3, 3] < before[3, 3]
    assert field.temperatures[3, 4] > before[3, 4]
    assert field.temperatures[3, 2] > before[3, 2]
