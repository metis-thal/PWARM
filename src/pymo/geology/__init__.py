"""Geology module for PWARM.

Geological processes: stratigraphy, thermal conduction, erosion, tectonics.
"""

from .rock_materials import (
    RockMaterial,
    get_rock,
    get_rock_by_id,
    all_rocks,
    rock_count,
    get_color_array,
    get_erosion_resistance_array,
    get_thermal_conductivity_array,
    get_melting_point_array,
)

from .geology_grid import (
    GeologyGridConfig,
    GeologyGrid,
    create_stratified_grid,
)

from .geology_solver import (
    GeologySolverConfig,
    GeologySolver,
)

from .processes.thermal import (
    ThermalConfig,
    solve_thermal_step,
    solve_steady_state,
)

from .processes.sedimentation import (
    SedimentationConfig,
    create_initial_stratigraphy,
    add_sediment_layer,
    compute_surface_elevation,
)

__all__ = [
    # rock_materials
    "RockMaterial",
    "get_rock",
    "get_rock_by_id",
    "all_rocks",
    "rock_count",
    "get_color_array",
    "get_erosion_resistance_array",
    "get_thermal_conductivity_array",
    "get_melting_point_array",
    # geology_grid
    "GeologyGridConfig",
    "GeologyGrid",
    "create_stratified_grid",
    # geology_solver
    "GeologySolverConfig",
    "GeologySolver",
    # thermal
    "ThermalConfig",
    "solve_thermal_step",
    "solve_steady_state",
    # sedimentation
    "SedimentationConfig",
    "create_initial_stratigraphy",
    "add_sediment_layer",
    "compute_surface_elevation",
]