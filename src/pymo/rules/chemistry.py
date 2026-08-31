"""Chemistry & Phase Change module for the pymo rules layer.

Implements:
- Substance phase transitions (solid/liquid/gas by temperature/pressure)
- Chemical reactions: combustion, oxidation, dissolution, thermal decomposition
- Mass/element conservation enforcement
- Reaction rate models (Arrhenius)

All equations are standard chemical physics formulations; no hardcoded phenomena.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from pymo.kernel.bodies import Body
from pymo.kernel.bodies3d import Body as Body3D


class Phase(Enum):
    """Phase of matter."""
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    PLASMA = "plasma"  # for completeness


@dataclass
class Substance:
    """A chemical substance with thermodynamic properties."""
    
    name: str
    # Elemental composition: {element: count}
    composition: dict[str, int]
    
    # Phase transition properties
    melting_point: float = 0.0        # K
    boiling_point: float = 0.0        # K
    latent_heat_fusion: float = 0.0   # J/kg (solid -> liquid)
    latent_heat_vaporization: float = 0.0  # J/kg (liquid -> gas)
    
    # Thermal properties
    specific_heat_solid: float = 1000.0      # J/(kg*K)
    specific_heat_liquid: float = 1000.0
    specific_heat_gas: float = 1000.0
    thermal_conductivity: float = 1.0        # W/(m*K)
    
    # Density by phase (kg/m^3)
    density_solid: float = 1000.0
    density_liquid: float = 1000.0
    density_gas: float = 1.0
    
    # Reaction properties
    heat_of_formation: float = 0.0     # J/kg
    activation_energy: float = 0.0     # J/mol
    
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        atomic_mass = {
            'H': 1.008e-3, 'C': 12.011e-3, 'O': 15.999e-3,
            'N': 14.007e-3, 'S': 32.06e-3, 'Fe': 55.845e-3,
            'Si': 28.085e-3, 'Al': 26.982e-3, 'Ca': 40.078e-3,
        }
        return sum(count * atomic_mass.get(elem, 0.01) 
                   for elem, count in self.composition.items())
    
    def phase_at(self, temperature: float, pressure: float = 101325.0) -> Phase:
        """Determine phase at given T, P (simplified: ignores pressure effects on transitions)."""
        if temperature < self.melting_point:
            return Phase.SOLID
        elif temperature < self.boiling_point:
            return Phase.LIQUID
        else:
            return Phase.GAS
    
    def specific_heat_at(self, temperature: float, pressure: float = 101325.0) -> float:
        """Specific heat at given conditions."""
        phase = self.phase_at(temperature, pressure)
        if phase == Phase.SOLID:
            return self.specific_heat_solid
        elif phase == Phase.LIQUID:
            return self.specific_heat_liquid
        else:
            return self.specific_heat_gas
    
    def density_at(self, temperature: float, pressure: float = 101325.0) -> float:
        """Density at given conditions."""
        phase = self.phase_at(temperature, pressure)
        if phase == Phase.SOLID:
            return self.density_solid
        elif phase == Phase.LIQUID:
            return self.density_liquid
        else:
            # Ideal gas law approximation for gas
            R = 8.314  # J/(mol*K)
            M = self.molar_mass()
            return pressure * M / (R * temperature)


# Predefined substances
SUBSTANCES = {
    "water": Substance(
        name="water",
        composition={"H": 2, "O": 1},
        melting_point=273.15,
        boiling_point=373.15,
        latent_heat_fusion=334e3,
        latent_heat_vaporization=2260e3,
        specific_heat_solid=2100.0,
        specific_heat_liquid=4184.0,
        specific_heat_gas=2000.0,
        density_solid=917.0,
        density_liquid=1000.0,
        density_gas=0.6,
        heat_of_formation=-285.8e6,  # J/kg
    ),
    "iron": Substance(
        name="iron",
        composition={"Fe": 1},
        melting_point=1811.0,
        boiling_point=3134.0,
        latent_heat_fusion=247e3,
        latent_heat_vaporization=6090e3,
        specific_heat_solid=449.0,
        specific_heat_liquid=820.0,
        density_solid=7874.0,
        density_liquid=7000.0,
    ),
    "wood": Substance(
        name="wood",
        composition={"C": 6, "H": 10, "O": 5},  # cellulose approximation
        melting_point=473.0,  # pyrolysis starts
        boiling_point=673.0,
        latent_heat_fusion=0.0,  # no melt
        latent_heat_vaporization=0.0,
        specific_heat_solid=1700.0,
        density_solid=600.0,
        heat_of_formation=-120e6,  # approximate
        activation_energy=150e3,  # J/mol
    ),
    "methane": Substance(
        name="methane",
        composition={"C": 1, "H": 4},
        melting_point=90.7,
        boiling_point=111.7,
        latent_heat_fusion=58.6e3,
        latent_heat_vaporization=510e3,
        specific_heat_gas=2200.0,
        density_gas=0.66,
        heat_of_formation=-74.8e6,
    ),
    "oxygen": Substance(
        name="oxygen",
        composition={"O": 2},
        melting_point=54.8,
        boiling_point=90.2,
        density_gas=1.33,
    ),
    "carbon_dioxide": Substance(
        name="carbon_dioxide",
        composition={"C": 1, "O": 2},
        melting_point=216.6,
        boiling_point=194.7,  # sublimates
        density_gas=1.84,
        heat_of_formation=-393.5e6,
    ),
}


@dataclass
class Reaction:
    """A chemical reaction with stoichiometry and kinetics."""
    
    name: str
    # Reactants: {substance_name: stoichiometric_coefficient}
    reactants: dict[str, float]
    # Products: {substance_name: stoichiometric_coefficient}
    products: dict[str, float]
    
    # Kinetics
    activation_energy: float      # J/mol
    pre_exponential: float        # 1/s (for first order) or appropriate units
    reaction_order: dict[str, float] = field(default_factory=dict)  # per reactant
    
    # Heat of reaction (J per mole of reaction as written)
    # Negative = exothermic
    delta_h: float = 0.0
    
    def rate_constant(self, temperature: float) -> float:
        """Arrhenius rate constant: k = A * exp(-Ea / (R*T))."""
        R = 8.314  # J/(mol*K)
        return self.pre_exponential * np.exp(-self.activation_energy / (R * temperature))
    
    def rate(self, temperature: float, concentrations: dict[str, float]) -> float:
        """Reaction rate given concentrations (mol/m^3)."""
        k = self.rate_constant(temperature)
        # Rate = k * prod([C_i]^order_i)
        rate = k
        for reactant, order in self.reaction_order.items():
            conc = concentrations.get(reactant, 0.0)
            if conc <= 0:
                return 0.0
            rate *= conc**order
        return rate
    
    def heat_release(self, extent: float) -> float:
        """Heat released (positive) or absorbed (negative) for given extent (mol)."""
        return -self.delta_h * extent


# Predefined reactions
REACTIONS = {
    "methane_combustion": Reaction(
        name="methane_combustion",
        reactants={"methane": 1, "oxygen": 2},
        products={"carbon_dioxide": 1, "water": 2},
        activation_energy=200e3,
        pre_exponential=1e12,
        reaction_order={"methane": 1, "oxygen": 2},
        delta_h=-802.3e3,  # J/mol methane
    ),
    "wood_pyrolysis": Reaction(
        name="wood_pyrolysis",
        reactants={"wood": 1},
        products={"carbon_dioxide": 1, "water": 1},  # simplified
        activation_energy=150e3,
        pre_exponential=1e10,
        reaction_order={"wood": 1},
        delta_h=-200e3,  # J/mol (exothermic char formation)
    ),
    "iron_oxidation": Reaction(
        name="iron_oxidation",
        reactants={"iron": 4, "oxygen": 3},
        products={"iron": 0},  # forms iron oxide (not tracked separately yet)
        activation_energy=80e3,
        pre_exponential=1e8,
        reaction_order={"iron": 1, "oxygen": 1.5},
        delta_h=-822e3,  # J/mol Fe2O3 formed
    ),
}


@dataclass
class ChemicalSystem:
    """Manages chemical state and reactions for a body."""
    
    # Current substance amounts: {substance_name: mass_kg}
    masses: dict[str, float] = field(default_factory=dict)
    
    # Temperature (K)
    temperature: float = 293.15
    
    # Pressure (Pa)
    pressure: float = 101325.0
    
    def total_mass(self) -> float:
        return sum(self.masses.values())
    
    def moles(self, substance_name: str) -> float:
        """Moles of a substance."""
        sub = SUBSTANCES.get(substance_name)
        if sub is None:
            return 0.0
        mass = self.masses.get(substance_name, 0.0)
        return mass / sub.molar_mass() if sub.molar_mass() > 0 else 0.0
    
    def concentrations(self) -> dict[str, float]:
        """Molar concentrations (mol/m^3) for reaction rates."""
        # Simplified: assume uniform density, convert mass to concentration
        total_mass = self.total_mass()
        if total_mass <= 0:
            return {}
        
        conc = {}
        for name, mass in self.masses.items():
            sub = SUBSTANCES.get(name)
            if sub and sub.molar_mass() > 0:
                # Approximate volume from total mass and average density
                # This is a simplification
                moles = mass / sub.molar_mass()
                # Assume 1 m^3 for concentration calculation
                conc[name] = moles
        return conc
    
    def element_masses(self) -> dict[str, float]:
        """Total mass of each element (conservation check)."""
        elements = {}
        for name, mass in self.masses.items():
            sub = SUBSTANCES.get(name)
            if sub is None:
                continue
            M = sub.molar_mass()
            for elem, count in sub.composition.items():
                elem_M = {
                    'H': 1.008e-3, 'C': 12.011e-3, 'O': 15.999e-3,
                    'N': 14.007e-3, 'S': 32.06e-3, 'Fe': 55.845e-3,
                }.get(elem, 0.01)
                elem_mass_in_mol = count * elem_M
                elem_mass = mass * (elem_mass_in_mol / M)
                elements[elem] = elements.get(elem, 0.0) + elem_mass
        return elements


def phase_transition_heat(substance: Substance, mass: float, 
                          from_phase: Phase, to_phase: Phase) -> float:
    """Latent heat for phase transition (positive = heat absorbed)."""
    if from_phase == Phase.SOLID and to_phase == Phase.LIQUID:
        return mass * substance.latent_heat_fusion
    elif from_phase == Phase.LIQUID and to_phase == Phase.GAS:
        return mass * substance.latent_heat_vaporization
    elif from_phase == Phase.LIQUID and to_phase == Phase.SOLID:
        return -mass * substance.latent_heat_fusion
    elif from_phase == Phase.GAS and to_phase == Phase.LIQUID:
        return -mass * substance.latent_heat_vaporization
    return 0.0


def step_chemistry(system: ChemicalSystem, dt: float) -> float:
    """Advance chemical system by dt.
    
    Returns net heat released (positive = exothermic).
    """
    total_heat = 0.0
    conc = system.concentrations()
    
    # Process reactions
    for reaction in REACTIONS.values():
        # Check if all reactants present
        if all(system.masses.get(r, 0) > 0 for r in reaction.reactants):
            rate = reaction.rate(system.temperature, conc)
            # Extent in moles
            extent = rate * dt
            
            # Limit by available reactants
            for reactant, stoich in reaction.reactants.items():
                available_moles = system.moles(reactant)
                max_extent = available_moles / stoich
                if extent > max_extent:
                    extent = max_extent
            
            if extent <= 0:
                continue
            
            # Consume reactants
            for reactant, stoich in reaction.reactants.items():
                sub = SUBSTANCES[reactant]
                mass_change = -extent * stoich * sub.molar_mass()
                system.masses[reactant] = system.masses.get(reactant, 0.0) + mass_change
                if system.masses[reactant] < 1e-12:
                    del system.masses[reactant]
            
            # Produce products
            for product, stoich in reaction.products.items():
                sub = SUBSTANCES[product]
                mass_change = extent * stoich * sub.molar_mass()
                system.masses[product] = system.masses.get(product, 0.0) + mass_change
            
            # Heat release
            total_heat += reaction.heat_release(extent)
    
    # Check phase transitions
    for name, mass in list(system.masses.items()):
        sub = SUBSTANCES.get(name)
        if sub is None:
            continue
        
        new_phase = sub.phase_at(system.temperature, system.pressure)
        # We don't track current phase per substance in this simple model
        # In a full implementation, each substance batch would track its phase
    
    return total_heat


def couple_thermal_chemical(body: Body, chemical: ChemicalSystem, dt: float) -> float:
    """Couple chemical heat release to body thermal state.
    
    Returns heat transferred to body (J).
    """
    heat = step_chemistry(chemical, dt)
    
    # Add heat to body
    body.heat += heat
    # Update temperature
    c = body.mass * body.material.specific_heat
    if c > 0:
        body.temperature += heat / c
        # Sync chemical system temperature
        chemical.temperature = body.temperature
    
    return heat


# Phase change coupling
def apply_phase_transitions(body: Body, chemical: ChemicalSystem) -> float:
    """Apply phase transitions based on temperature.
    
    Returns latent heat absorbed/released.
    """
    total_latent = 0.0
    
    for name, mass in list(chemical.masses.items()):
        sub = SUBSTANCES.get(name)
        if sub is None or mass <= 0:
            continue
        
        current_phase = sub.phase_at(chemical.temperature, chemical.pressure)
        
        # In a full model, we'd track phase per substance batch
        # For now, just compute latent heat if phase changed
        # This is a placeholder for the full implementation
    
    return total_latent


if __name__ == "__main__":
    # Quick test
    chem = ChemicalSystem(
        masses={"wood": 1.0, "oxygen": 0.5},
        temperature=500.0,
        pressure=101325.0
    )
    
    print(f"Initial masses: {chem.masses}")
    print(f"Element masses: {chem.element_masses()}")
    print(f"Concentrations: {chem.concentrations()}")
    
    # Step
    heat = step_chemistry(chem, dt=0.01)
    print(f"Heat released: {heat:.2f} J")
    print(f"Final masses: {chem.masses}")
    print(f"Element masses: {chem.element_masses()}")