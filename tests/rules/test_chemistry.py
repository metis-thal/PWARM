"""Tests for the chemistry rule module (rules.chemistry).

The module still imports the removed ``pwarm.kernel`` package; tests/conftest.py
installs a stub for it, so these tests exercise the real chemistry logic
(substance properties, Arrhenius kinetics, reaction stepping, thermal coupling)
against the historical body interface.

Note: tests pin the behaviour AS IMPLEMENTED, including documented quirks
(ideal-gas density ignores ``density_gas``, ``iron_oxidation`` re-adds a zero
``iron`` key, ``apply_phase_transitions`` is a placeholder).
"""

from __future__ import annotations

import numpy as np
import pytest
from pwarm.kernel.bodies import Body, Material

from pwarm.rules.chemistry import (
    REACTIONS,
    SUBSTANCES,
    ChemicalSystem,
    Phase,
    Reaction,
    Substance,
    apply_phase_transitions,
    couple_thermal_chemical,
    phase_transition_heat,
    step_chemistry,
)


# --------------------------------------------------------------------- Substance
def test_molar_mass_sums_composition() -> None:
    """Molar mass is the composition-weighted sum of atomic masses."""
    water = SUBSTANCES["water"]
    expected = 2 * 1.008e-3 + 15.999e-3  # H2O
    assert water.molar_mass() == pytest.approx(expected, rel=1e-6)


def test_molar_mass_unknown_element_fallback() -> None:
    """An element outside the table falls back to 0.01 kg/mol."""
    exotic = Substance(name="x", composition={"Xx": 2})
    assert exotic.molar_mass() == pytest.approx(0.02)


def test_molar_mass_empty_composition() -> None:
    """Empty composition gives 0.0, which consumers must special-case."""
    assert Substance(name="empty", composition={}).molar_mass() == 0.0


def test_phase_at_boundaries() -> None:
    """T == melting -> liquid; T == boiling -> gas (strict comparisons)."""
    s = Substance(name="t", composition={"H": 1},
                  melting_point=100.0, boiling_point=200.0)
    assert s.phase_at(50.0) is Phase.SOLID
    assert s.phase_at(100.0) is Phase.LIQUID
    assert s.phase_at(150.0) is Phase.LIQUID
    assert s.phase_at(200.0) is Phase.GAS


def test_specific_heat_follows_phase() -> None:
    """specific_heat_at dispatches on the phase at that temperature."""
    s = Substance(name="t", composition={"H": 1},
                  melting_point=100.0, boiling_point=200.0,
                  specific_heat_solid=800.0, specific_heat_liquid=1200.0,
                  specific_heat_gas=2000.0)
    assert s.specific_heat_at(50.0) == 800.0
    assert s.specific_heat_at(150.0) == 1200.0
    assert s.specific_heat_at(300.0) == 2000.0


def test_density_at_uses_ideal_gas_law() -> None:
    """Gas density is p*M/(R*T); the density_gas field is ignored."""
    s = SUBSTANCES["water"]
    m = s.molar_mass()
    expected = 101325.0 * m / (8.314 * 400.0)
    assert s.density_at(400.0) == pytest.approx(expected, rel=1e-6)
    liquid = s.density_at(350.0)  # between 273.15 and 373.15
    assert liquid == s.density_liquid


def test_reaction_rate_constant_arrhenius() -> None:
    """rate_constant follows k = A * exp(-Ea / (R*T))."""
    r = Reaction(name="r", reactants={"a": 1}, products={"b": 1},
                 activation_energy=5e4, pre_exponential=1e6)
    expected = 1e6 * np.exp(-5e4 / (8.314 * 300.0))
    assert r.rate_constant(300.0) == pytest.approx(expected, rel=1e-9)


def test_reaction_rate_missing_reactant_is_zero() -> None:
    """rate() returns 0 when a species in reaction_order has no concentration."""
    r = Reaction(name="r", reactants={"a": 1}, products={"b": 1},
                 activation_energy=0.0, pre_exponential=2.0,
                 reaction_order={"a": 1, "missing": 1})
    assert r.rate(300.0, {"a": 5.0}) == 0.0
    assert r.rate(300.0, {"a": 5.0, "missing": 0.0}) == 0.0
    assert r.rate(300.0, {"a": 5.0, "missing": -1.0}) == 0.0


def test_reaction_rate_first_order() -> None:
    """rate() = k * prod(conc**order) for present, positive concentrations."""
    r = Reaction(name="r", reactants={"a": 1}, products={"b": 1},
                 activation_energy=0.0, pre_exponential=3.0,
                 reaction_order={"a": 2.0})
    assert r.rate(300.0, {"a": 2.0}) == pytest.approx(3.0 * 4.0)


def test_reaction_heat_release_sign() -> None:
    """heat_release = -delta_h * extent: exothermic (delta_h<0) releases > 0."""
    exo = Reaction(name="exo", reactants={"a": 1}, products={"b": 1},
                   activation_energy=0.0, pre_exponential=1.0, delta_h=-890e3)
    assert exo.heat_release(2.0) == pytest.approx(1.78e6)


# --------------------------------------------------------------- ChemicalSystem
def test_chemical_system_moles_unknown_substance() -> None:
    """moles() of an unregistered substance or absent mass is 0.0."""
    system = ChemicalSystem(masses={"water": 0.018})
    assert system.moles("nope") == 0.0
    assert system.moles("iron") == 0.0
    assert system.moles("water") == pytest.approx(0.018 / SUBSTANCES["water"].molar_mass())


def test_chemical_system_concentrations() -> None:
    """concentrations() returns mol per named substance; empty for no mass."""
    system = ChemicalSystem(masses={"water": 0.018, "unobtainium": 1.0})
    conc = system.concentrations()
    assert set(conc) == {"water"}
    assert conc["water"] == pytest.approx(system.moles("water"))
    assert ChemicalSystem().concentrations() == {}


def test_chemical_system_element_masses_conservation() -> None:
    """Element masses of a substance sum to its total mass."""
    system = ChemicalSystem(masses={"methane": 0.016})
    elements = system.element_masses()
    assert set(elements) == {"C", "H"}
    assert sum(elements.values()) == pytest.approx(0.016, rel=1e-9)


# ---------------------------------------------------------- phase_transition_heat
def test_phase_transition_heat_directions() -> None:
    """Fusion/vaporization absorb heat; freezing/condensation release it."""
    s = Substance(name="t", composition={"H": 1},
                  latent_heat_fusion=334e3, latent_heat_vaporization=2260e3)
    assert phase_transition_heat(s, 1.0, Phase.SOLID, Phase.LIQUID) == pytest.approx(334e3)
    assert phase_transition_heat(s, 1.0, Phase.LIQUID, Phase.GAS) == pytest.approx(2260e3)
    assert phase_transition_heat(s, 1.0, Phase.LIQUID, Phase.SOLID) == pytest.approx(-334e3)
    assert phase_transition_heat(s, 1.0, Phase.GAS, Phase.LIQUID) == pytest.approx(-2260e3)
    assert phase_transition_heat(s, 1.0, Phase.SOLID, Phase.GAS) == 0.0
    assert phase_transition_heat(s, 1.0, Phase.SOLID, Phase.SOLID) == 0.0


# ---------------------------------------------------------------- step_chemistry
def _fresh_system() -> ChemicalSystem:
    return ChemicalSystem(
        masses={"methane": 0.016, "oxygen": 0.032, "wood": 0.1,
                "iron": 0.05, "carbon_dioxide": 0.0, "water": 0.0},
        temperature=600.0)


def test_step_chemistry_consumes_reactants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combustion consumes methane/oxygen and produces CO2 and water."""
    monkeypatch.setitem(REACTIONS, "wood_pyrolysis",
                        Reaction(name="off", reactants={"wood": 1},
                                 products={"methane": 1}, activation_energy=1e30,
                                 pre_exponential=1.0))
    monkeypatch.setitem(REACTIONS, "iron_oxidation",
                        Reaction(name="off2", reactants={"iron": 1},
                                 products={"rust": 1}, activation_energy=1e30,
                                 pre_exponential=1.0))
    system = _fresh_system()
    m_ch4_0 = system.masses["methane"]
    heat = step_chemistry(system, dt=0.1)
    assert heat > 0.0  # methane combustion is exothermic
    assert system.masses["methane"] < m_ch4_0
    assert system.masses["oxygen"] < 0.032
    assert system.masses["carbon_dioxide"] > 0.0
    assert system.masses["water"] > 0.0


def test_step_chemistry_skips_reaction_with_missing_reactant(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A reaction whose reactants are absent contributes no heat."""
    monkeypatch.setitem(REACTIONS, "methane_combustion",
                        Reaction(name="m", reactants={"methane": 1, "oxygen": 2},
                                 products={"carbon_dioxide": 1, "water": 2},
                                 activation_energy=5e4, pre_exponential=1e8,
                                 reaction_order={"methane": 1, "oxygen": 1},
                                 delta_h=-890e3))
    monkeypatch.setitem(REACTIONS, "wood_pyrolysis",
                        Reaction(name="w", reactants={"wood": 1}, products={"x": 1},
                                 activation_energy=1e30, pre_exponential=1.0))
    monkeypatch.setitem(REACTIONS, "iron_oxidation",
                        Reaction(name="i", reactants={"iron": 1}, products={"x": 1},
                                 activation_energy=1e30, pre_exponential=1.0))
    system = ChemicalSystem(masses={}, temperature=600.0)
    assert step_chemistry(system, dt=0.1) == 0.0
    assert system.masses == {}


def test_step_chemistry_extent_limited_by_available_moles(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A huge dt consumes at most all of the limiting reactant."""
    monkeypatch.setitem(REACTIONS, "methane_combustion",
                        Reaction(name="m", reactants={"methane": 1},
                                 products={"water": 1}, activation_energy=0.0,
                                 pre_exponential=1e12, reaction_order={"methane": 1},
                                 delta_h=-100.0))
    for other in ("wood_pyrolysis", "iron_oxidation"):
        monkeypatch.setitem(REACTIONS, other,
                            Reaction(name=other, reactants={"z": 1}, products={"z": 1},
                                     activation_energy=1e30, pre_exponential=1.0))
    system = ChemicalSystem(masses={"methane": 1e-6}, temperature=1000.0)
    step_chemistry(system, dt=1e6)
    # methane fully consumed; the <1e-12 cleanup removes the key
    assert system.masses.get("methane", 0.0) < 1e-12
    assert system.masses["water"] > 0.0


def test_couple_thermal_chemical_heats_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Released heat raises body temperature, body heat, and syncs the system."""
    monkeypatch.setattr("pwarm.rules.chemistry.step_chemistry",
                        lambda system, dt: 500.0)
    body = Body(pos=np.zeros(2), mass=2.0, material=Material(specific_heat=500.0))
    body.temperature = 300.0
    system = ChemicalSystem(masses={}, temperature=300.0)
    heat = couple_thermal_chemical(body, system, dt=0.1)
    assert heat == 500.0
    assert body.heat == 500.0
    assert body.temperature == pytest.approx(300.5)  # 500 / (2*500)
    assert system.temperature == pytest.approx(300.5)


def test_couple_thermal_chemical_no_reactions_is_noop() -> None:
    """With no available reactants, coupling transfers no heat."""
    body = Body(pos=np.zeros(2), mass=2.0, material=Material(specific_heat=500.0))
    body.temperature = 300.0
    system = ChemicalSystem(masses={}, temperature=300.0)
    heat = couple_thermal_chemical(body, system, dt=0.1)
    assert heat == 0.0
    assert body.temperature == 300.0
    assert system.temperature == 300.0  # synced to the unchanged body


def test_couple_thermal_chemical_zero_capacity_noop() -> None:
    """Zero heat capacity leaves the body and system untouched."""
    body = Body(pos=np.zeros(2), mass=0.0, material=Material(specific_heat=500.0))
    body.temperature = 300.0
    system = ChemicalSystem(masses={}, temperature=350.0)
    couple_thermal_chemical(body, system, dt=0.1)
    assert body.temperature == 300.0
    assert system.temperature == 350.0  # not synced


def test_apply_phase_transitions_placeholder() -> None:
    """apply_phase_transitions is a placeholder: returns 0, mutates nothing."""
    body = Body(pos=np.zeros(2), mass=1.0)
    system = ChemicalSystem(masses={"water": 1.0}, temperature=400.0)
    before = dict(system.masses)
    assert apply_phase_transitions(body, system) == 0.0
    assert system.masses == before
    assert system.temperature == 400.0
