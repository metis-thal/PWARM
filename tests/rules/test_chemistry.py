"""Tests for chemistry module: substances, reactions, phase transitions."""

import numpy as np
import pytest

from pymo.rules.chemistry import (
    ChemicalSystem,
    Phase,
    Reaction,
    Substance,
    REACTIONS,
    SUBSTANCES,
    couple_thermal_chemical,
    phase_transition_heat,
    step_chemistry,
)


class TestSubstance:
    def test_water_properties(self):
        water = SUBSTANCES["water"]
        assert water.name == "water"
        assert water.molar_mass() == pytest.approx(0.018, rel=1e-2)
        assert water.composition == {"H": 2, "O": 1}

    def test_phase_at_temperature(self):
        water = SUBSTANCES["water"]
        assert water.phase_at(200.0) == Phase.SOLID
        assert water.phase_at(300.0) == Phase.LIQUID
        assert water.phase_at(400.0) == Phase.GAS

    def test_specific_heat_by_phase(self):
        water = SUBSTANCES["water"]
        assert water.specific_heat_at(200.0) == water.specific_heat_solid
        assert water.specific_heat_at(300.0) == water.specific_heat_liquid
        assert water.phase_at(400.0) == Phase.GAS

    def test_iron_properties(self):
        iron = SUBSTANCES["iron"]
        assert iron.molar_mass() == pytest.approx(0.055845, rel=1e-2)
        assert iron.phase_at(1500.0) == Phase.SOLID
        assert iron.phase_at(2000.0) == Phase.LIQUID

    def test_wood_properties(self):
        wood = SUBSTANCES["wood"]
        assert wood.molar_mass() > 0
        assert wood.specific_heat_solid == 1700.0


class TestReaction:
    def test_methane_combustion_rate(self):
        r = REACTIONS["methane_combustion"]
        # Higher temperature → faster rate
        rate_300 = r.rate(300.0, {"methane": 1.0, "oxygen": 2.0})
        rate_600 = r.rate(600.0, {"methane": 1.0, "oxygen": 2.0})
        assert rate_600 > rate_300

    def test_rate_zero_without_reactants(self):
        r = REACTIONS["methane_combustion"]
        rate = r.rate(600.0, {"methane": 0.0, "oxygen": 2.0})
        assert rate == 0.0

    def test_heat_release(self):
        r = REACTIONS["methane_combustion"]
        # Exothermic: delta_h < 0, so heat_release > 0
        heat = r.heat_release(1.0)
        assert heat > 0

    def test_rate_constant_arrhenius(self):
        r = REACTIONS["methane_combustion"]
        k300 = r.rate_constant(300.0)
        k1000 = r.rate_constant(1000.0)
        assert k1000 > k300


class TestChemicalSystem:
    def test_total_mass(self):
        cs = ChemicalSystem(masses={"water": 1.0, "iron": 2.0})
        assert cs.total_mass() == 3.0

    def test_moles(self):
        cs = ChemicalSystem(masses={"water": 0.018})
        moles = cs.moles("water")
        assert moles == pytest.approx(1.0, rel=0.01)

    def test_concentrations(self):
        cs = ChemicalSystem(masses={"water": 1.0})
        conc = cs.concentrations()
        assert "water" in conc
        assert conc["water"] > 0

    def test_element_masses_conservation(self):
        cs = ChemicalSystem(masses={"water": 1.0})
        elems = cs.element_masses()
        assert "H" in elems
        assert "O" in elems
        assert elems["H"] > 0
        assert elems["O"] > 0


class TestPhaseTransitionHeat:
    def test_melting_absorbs_heat(self):
        water = SUBSTANCES["water"]
        heat = phase_transition_heat(water, 1.0, Phase.SOLID, Phase.LIQUID)
        assert heat > 0  # absorbs heat

    def test_freezing_releases_heat(self):
        water = SUBSTANCES["water"]
        heat = phase_transition_heat(water, 1.0, Phase.LIQUID, Phase.SOLID)
        assert heat < 0  # releases heat

    def test_boiling_absorbs_heat(self):
        water = SUBSTANCES["water"]
        heat = phase_transition_heat(water, 1.0, Phase.LIQUID, Phase.GAS)
        assert heat > 0


class TestStepChemistry:
    def test_exothermic_reaction_releases_heat(self):
        cs = ChemicalSystem(
            masses={"methane": 0.1, "oxygen": 1.0},
            temperature=800.0  # above activation energy
        )
        heat = step_chemistry(cs, 1.0)
        # Should release heat (positive = exothermic)
        assert heat > 0

    def test_no_reaction_below_activation(self):
        cs = ChemicalSystem(
            masses={"methane": 0.1, "oxygen": 1.0},
            temperature=300.0  # below activation energy
        )
        heat_before = step_chemistry(cs, 1.0)
        # Very little reaction at room temperature
        assert heat_before < 1.0

    def test_mass_conservation(self):
        cs = ChemicalSystem(
            masses={"methane": 0.01, "oxygen": 0.1},
            temperature=800.0
        )
        total_before = cs.total_mass()
        step_chemistry(cs, 0.1)
        total_after = cs.total_mass()
        # Total mass should be approximately conserved
        assert total_after == pytest.approx(total_before, rel=0.01)
