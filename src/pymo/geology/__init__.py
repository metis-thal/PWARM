"""Geology module for PWARM.

Geological processes: stratigraphy, thermal conduction, erosion, tectonics.
"""

from .geology_grid import (
    GeologyGrid,
    GeologyGridConfig,
    create_stratified_grid,
)
from .geology_solver import (
    GeologySolver,
    GeologySolverConfig,
)
from .processes.sedimentation import (
    SedimentationConfig,
    add_sediment_layer,
    compute_surface_elevation,
    create_initial_stratigraphy,
)
from .processes.thermal import (
    ThermalConfig,
    solve_steady_state,
    solve_thermal_step,
)
from .rock_materials import (
    RockMaterial,
    all_rocks,
    get_color_array,
    get_erosion_resistance_array,
    get_melting_point_array,
    get_rock,
    get_rock_by_id,
    get_thermal_conductivity_array,
    rock_count,
)

__all__ = [
    "GeologyGrid",
    # geology_grid
    "GeologyGridConfig",
    "GeologySolver",
    # geology_solver
    "GeologySolverConfig",
    # rock_materials
    "RockMaterial",
    # sedimentation
    "SedimentationConfig",
    # thermal
    "ThermalConfig",
    "add_sediment_layer",
    "all_rocks",
    "compute_surface_elevation",
    "create_initial_stratigraphy",
    "create_stratified_grid",
    "get_color_array",
    "get_erosion_resistance_array",
    "get_melting_point_array",
    "get_rock",
    "get_rock_by_id",
    "get_thermal_conductivity_array",
    "rock_count",
    "solve_steady_state",
    "solve_thermal_step",
]