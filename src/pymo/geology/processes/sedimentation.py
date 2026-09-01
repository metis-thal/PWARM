"""Initial stratigraphic layering and sedimentation processes.

Phase 1: Simple horizontal layer deposition from surface down.
Phase 2 (future): Fluvial/deltaic deposition with lateral variation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..rock_materials import RockMaterial, get_rock
from ..geology_grid import GeologyGrid, GeologyGridConfig, create_stratified_grid


@dataclass(slots=True)
class SedimentationConfig:
    """Configuration for initial stratigraphy."""
    # List of (thickness_m, rock_name) from surface down
    layers: List[Tuple[float, str]]
    surface_temp: float = 293.15
    geothermal_gradient: float = 0.025  # K/m


def create_initial_stratigraphy(
    config: GeologyGridConfig,
    sed_config: SedimentationConfig,
) -> GeologyGrid:
    """Create geology grid with initial horizontal layers.

    layers: [(thickness_m, rock_name), ...] from top (surface) down.
    The bottom layer extends to fill the rest of the domain.
    """
    rock_layers = []
    for thickness, rock_name in sed_config.layers:
        rock = get_rock(rock_name)
        rock_layers.append((thickness, rock.rock_id))

    # create_stratified_grid processes from surface down, so use layers as-is
    grid = create_stratified_grid(
        config,
        rock_layers,
        surface_temp=sed_config.surface_temp,
        geothermal_gradient=sed_config.geothermal_gradient,
    )
    return grid


def add_sediment_layer(
    grid: GeologyGrid,
    thickness_m: float,
    rock_name: str,
    surface_temp: float = 293.15,
) -> None:
    """Add a new sediment layer at the surface (top).

    Shifts existing layers down, removes bottom cells if domain full.
    For Phase 2 dynamic sedimentation.
    """
    rock = get_rock(rock_name)
    nz_new = int(thickness_m / grid.config.cell_size)
    if nz_new <= 0:
        return

    nz = grid.nz
    # Shift everything down by nz_new
    if nz_new >= nz:
        # Domain completely filled with new sediment
        grid.rock_id[:] = rock.rock_id
        grid.set_temperature_gradient(surface_temp, 0.025)
        return

    # Shift arrays
    shift = nz_new * grid.nx * grid.ny
    grid.rock_id[shift:] = grid.rock_id[:-shift]
    grid.temperature[shift:] = grid.temperature[:-shift]
    grid.porosity[shift:] = grid.porosity[:-shift]
    grid.uplift_rate[shift:] = grid.uplift_rate[:-shift]
    grid.stress[shift:] = grid.stress[:-shift]

    # Fill new top layers
    grid.rock_id[:shift] = rock.rock_id
    # Temperature for new surface layers
    for iz in range(nz_new):
        base = iz * grid.nx * grid.ny
        z = (nz - 1 - iz) * grid.config.cell_size
        grid.temperature[base:base + grid.nx * grid.ny] = surface_temp + 0.025 * z

    # Adjust porosity (new sediment has higher porosity)
    grid.porosity[:shift] = 0.3


def compute_surface_elevation(grid: GeologyGrid) -> np.ndarray:
    """Compute surface elevation (top of first non-sediment rock, or top of grid).

    Returns (ny, nx) array of elevations in meters.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    elev = np.full((ny, nx), nz * grid.config.cell_size, dtype=np.float32)

    sediment_id = get_rock("sediment").rock_id

    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz - 1, -1, -1):
                i = ix + nx * (iy + ny * iz)
                if grid.rock_id[i] != sediment_id:
                    elev[iy, ix] = (iz + 1) * grid.config.cell_size
                    break
    return elev


if __name__ == "__main__":
    from ..geology_grid import GeologyGridConfig
    config = GeologyGridConfig(nx=32, ny=32, nz=16, cell_size=10.0)
    sed_config = SedimentationConfig(
        layers=[
            (20.0, "sediment"),
            (50.0, "sandstone"),
            (80.0, "shale"),
            (60.0, "granite"),
        ],
        surface_temp=293.15,
    )
    grid = create_initial_stratigraphy(config, sed_config)

    print("Stratigraphy test:")
    slice_top = grid.get_slice_xy(config.nz - 1)
    slice_mid = grid.get_slice_xy(config.nz // 2)
    slice_bot = grid.get_slice_xy(0)
    print(f"  Surface: unique rocks = {np.unique(slice_top['rock_id'])}")
    print(f"  Mid: unique rocks = {np.unique(slice_mid['rock_id'])}")
    print(f"  Base: unique rocks = {np.unique(slice_bot['rock_id'])}")
    print(f"  Temp range: {grid.temperature.min():.1f} - {grid.temperature.max():.1f} K")

    elev = compute_surface_elevation(grid)
    print(f"  Surface elev range: {elev.min():.1f} - {elev.max():.1f} m")