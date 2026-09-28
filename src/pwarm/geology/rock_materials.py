"""Rock material library for geology simulation.

Each rock type has a complete set of physical parameters for:
- Mechanical behavior (density, elastic modulus, cohesion, friction angle)
- Thermal behavior (conductivity)
- Erosion/weathering resistance
- Melting point for magmatic processes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RockMaterial:
    """Immutable rock material definition.

    All parameters in SI units unless noted.
    """
    name: str
    rock_id: int           # Unique identifier for GPU indexing

    # Mechanical
    density: float                 # kg/m³
    young_modulus: float           # Pa (杨氏模量)
    poisson_ratio: float           # 无量纲
    cohesion: float                # Pa (黏聚力)
    friction_angle: float          # radians (内摩擦角)

    # Thermal
    thermal_conductivity: float    # W/(m·K)
    specific_heat: float           # J/(kg·K)
    melting_point: float           # K (熔点)

    # Surface processes
    erosion_resistance: float      # 无量纲相对系数，1.0 = 基准(花岗岩)
    weathering_rate: float         # m/yr (风化速率基准)

    # Visual (for GPU rendering)
    color: tuple[float, float, float]   # RGB in [0,1] for rock_id shader

    def shear_strength(self, normal_stress: float) -> float:
        """Mohr-Coulomb shear strength: τ = c + σ tan(φ)"""
        return self.cohesion + normal_stress * np.tan(self.friction_angle)

    def is_molten(self, temperature: float) -> bool:
        return temperature >= self.melting_point


# ============================================================
# Built-in rock library (rock_id assigned by registration order)
# ============================================================

_ROCK_LIBRARY: dict[str, RockMaterial] = {}
_ID_TO_ROCK: list[RockMaterial] = []


def _register(material: RockMaterial) -> RockMaterial:
    """Register a rock material, assign rock_id sequentially."""
    material = RockMaterial(
        name=material.name,
        rock_id=len(_ID_TO_ROCK),
        density=material.density,
        young_modulus=material.young_modulus,
        poisson_ratio=material.poisson_ratio,
        cohesion=material.cohesion,
        friction_angle=material.friction_angle,
        thermal_conductivity=material.thermal_conductivity,
        specific_heat=material.specific_heat,
        melting_point=material.melting_point,
        erosion_resistance=material.erosion_resistance,
        weathering_rate=material.weathering_rate,
        color=material.color,
    )
    _ROCK_LIBRARY[material.name] = material
    _ID_TO_ROCK.append(material)
    return material


# Igneous / Plutonic
_register(RockMaterial(
    name="granite",
    rock_id=0,  # placeholder, will be overwritten
    density=2700.0,
    young_modulus=70e9,
    poisson_ratio=0.25,
    cohesion=20e6,
    friction_angle=np.radians(35),
    thermal_conductivity=3.0,
    specific_heat=790.0,
    melting_point=1215 + 273.15,
    erosion_resistance=1.0,
    weathering_rate=0.01,
    color=(0.55, 0.50, 0.48),  # speckled gray-pink
))

_register(RockMaterial(
    name="basalt",
    rock_id=0,
    density=2900.0,
    young_modulus=90e9,
    poisson_ratio=0.23,
    cohesion=25e6,
    friction_angle=np.radians(38),
    thermal_conductivity=2.0,
    specific_heat=840.0,
    melting_point=1150 + 273.15,
    erosion_resistance=1.2,
    weathering_rate=0.02,
    color=(0.25, 0.22, 0.20),
))

# Sedimentary
_register(RockMaterial(
    name="sandstone",
    rock_id=0,
    density=2300.0,
    young_modulus=20e9,
    poisson_ratio=0.28,
    cohesion=5e6,
    friction_angle=np.radians(30),
    thermal_conductivity=2.5,
    specific_heat=920.0,
    melting_point=1600 + 273.15,  # quartz cement
    erosion_resistance=0.6,
    weathering_rate=0.05,
    color=(0.75, 0.65, 0.50),  # sandy beige
))

_register(RockMaterial(
    name="shale",
    rock_id=0,
    density=2400.0,
    young_modulus=10e9,
    poisson_ratio=0.35,
    cohesion=2e6,
    friction_angle=np.radians(20),
    thermal_conductivity=1.5,
    specific_heat=880.0,
    melting_point=1200 + 273.15,
    erosion_resistance=0.3,
    weathering_rate=0.15,
    color=(0.40, 0.35, 0.32),  # dark gray
))

_register(RockMaterial(
    name="limestone",
    rock_id=0,
    density=2600.0,
    young_modulus=50e9,
    poisson_ratio=0.28,
    cohesion=10e6,
    friction_angle=np.radians(32),
    thermal_conductivity=2.2,
    specific_heat=900.0,
    melting_point=825 + 273.15,  # calcite decomp
    erosion_resistance=0.5,
    weathering_rate=0.08,
    color=(0.82, 0.80, 0.75),  # light gray-white
))

_register(RockMaterial(
    name="sediment",
    rock_id=0,  # unconsolidated surface sediment
    density=1800.0,
    young_modulus=1e9,
    poisson_ratio=0.40,
    cohesion=0.1e6,
    friction_angle=np.radians(15),
    thermal_conductivity=1.0,
    specific_heat=1200.0,
    melting_point=1000 + 273.15,
    erosion_resistance=0.1,
    weathering_rate=0.50,
    color=(0.55, 0.50, 0.45),
))

# Metamorphic
_register(RockMaterial(
    name="schist",
    rock_id=0,
    density=2750.0,
    young_modulus=40e9,
    poisson_ratio=0.25,
    cohesion=15e6,
    friction_angle=np.radians(33),
    thermal_conductivity=2.8,
    specific_heat=800.0,
    melting_point=650 + 273.15,
    erosion_resistance=0.8,
    weathering_rate=0.03,
    color=(0.50, 0.45, 0.48),
))

_register(RockMaterial(
    name="gneiss",
    rock_id=0,
    density=2800.0,
    young_modulus=60e9,
    poisson_ratio=0.24,
    cohesion=18e6,
    friction_angle=np.radians(36),
    thermal_conductivity=3.2,
    specific_heat=780.0,
    melting_point=700 + 273.15,
    erosion_resistance=0.9,
    weathering_rate=0.02,
    color=(0.48, 0.45, 0.50),
))

# Magmatic
_register(RockMaterial(
    name="magma",
    rock_id=0,
    density=2600.0,
    young_modulus=1e6,   # effectively fluid
    poisson_ratio=0.45,
    cohesion=1e3,
    friction_angle=np.radians(5),
    thermal_conductivity=1.5,
    specific_heat=1100.0,
    melting_point=1200 + 273.15,
    erosion_resistance=0.01,
    weathering_rate=1.0,
    color=(1.0, 0.4, 0.1),  # glowing orange
))


def get_rock(name: str) -> RockMaterial:
    """Get rock material by name."""
    if name not in _ROCK_LIBRARY:
        raise KeyError(f"Unknown rock: {name}. Available: {list(_ROCK_LIBRARY.keys())}")
    return _ROCK_LIBRARY[name]


def get_rock_by_id(rock_id: int) -> RockMaterial:
    """Get rock material by rock_id (for GPU indexing)."""
    if rock_id < 0 or rock_id >= len(_ID_TO_ROCK):
        raise IndexError(f"Invalid rock_id: {rock_id}, max={len(_ID_TO_ROCK)-1}")
    return _ID_TO_ROCK[rock_id]


def all_rocks() -> list[RockMaterial]:
    """Return all registered rocks in rock_id order."""
    return list(_ID_TO_ROCK)


def rock_count() -> int:
    return len(_ID_TO_ROCK)


def get_color_array() -> np.ndarray:
    """Return (N, 3) float32 array of rock colors for GPU uniform buffer."""
    return np.array([r.color for r in _ID_TO_ROCK], dtype=np.float32)


def get_erosion_resistance_array() -> np.ndarray:
    """Return (N,) float32 array of erosion resistance for GPU."""
    return np.array([r.erosion_resistance for r in _ID_TO_ROCK], dtype=np.float32)


def get_thermal_conductivity_array() -> np.ndarray:
    """Return (N,) float32 array of thermal conductivity."""
    return np.array([r.thermal_conductivity for r in _ID_TO_ROCK], dtype=np.float32)


def get_melting_point_array() -> np.ndarray:
    """Return (N,) float32 array of melting points in Kelvin."""
    return np.array([r.melting_point for r in _ID_TO_ROCK], dtype=np.float32)


if __name__ == "__main__":
    print("Registered rocks:")
    for r in all_rocks():
        print(f"  [{r.rock_id}] {r.name}: ρ={r.density:.0f} E={r.young_modulus/1e9:.1f}GPa "
              f"c={r.cohesion/1e6:.1f}MPa φ={np.degrees(r.friction_angle):.0f}° "
              f"k={r.thermal_conductivity:.1f} eros={r.erosion_resistance:.2f} "
              f"color={r.color}")