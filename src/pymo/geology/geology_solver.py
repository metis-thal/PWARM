"""Main geology solver integrating all geological processes.

Orchestrates:
- Thermal conduction (geothermal gradient)
- Initial stratigraphy
- (Future) Erosion/sedimentation (Stream Power Law)
- (Future) Tectonic deformation (Mohr-Coulomb failure)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .rock_materials import RockMaterial, all_rocks
from .geology_grid import GeologyGrid, GeologyGridConfig, create_stratified_grid
from .processes.thermal import (
    ThermalConfig,
    solve_thermal_step,
    solve_steady_state,
    get_material_dict,
)
from .processes.sedimentation import SedimentationConfig, create_initial_stratigraphy


@dataclass(slots=True)
class GeologySolverConfig:
    """Full configuration for GeologySolver."""
    # Grid
    domain_size: tuple[float, float, float] = (1000.0, 1000.0, 500.0)  # meters (X, Y, Z)
    cell_resolution: float = 10.0   # meters per cell

    # Initial stratigraphy
    initial_layers: list[tuple[float, str]] = field(default_factory=lambda: [
        (20.0, "sediment"),
        (50.0, "sandstone"),
        (80.0, "shale"),
        (200.0, "granite"),
    ])

    # Thermal
    surface_temp: float = 293.15          # K
    mantle_heat_flux: float = 0.065       # W/m²
    radiogenic_heat: float = 1.0e-6       # W/m³
    geothermal_gradient: float = 0.025    # K/m initial gradient

    # Solver
    thermal_dt: float = 1000.0            # years per thermal step
    steady_state_init: bool = True        # solve steady-state on init


class GeologySolver:
    """Main geology solver.

    Call step(dt) each simulation tick to advance geological processes.
    """

    def __init__(self, config: Optional[GeologySolverConfig] = None):
        self.config = config or GeologySolverConfig()
        self._initialized = False

        # Grid
        nx = int(self.config.domain_size[0] / self.config.cell_resolution)
        ny = int(self.config.domain_size[1] / self.config.cell_resolution)
        nz = int(self.config.domain_size[2] / self.config.cell_resolution)
        self.grid_config = GeologyGridConfig(
            nx=nx, ny=ny, nz=nz,
            cell_size=self.config.cell_resolution,
            origin=(0.0, 0.0, 0.0)
        )
        self.grid: Optional[GeologyGrid] = None

        # Material library (rock_id -> RockMaterial)
        self.materials = get_material_dict()

        # Thermal config
        self.thermal_config = ThermalConfig(
            dt=self.config.thermal_dt,
            surface_temp=self.config.surface_temp,
            mantle_heat_flux=self.config.mantle_heat_flux,
            radiogenic_heat=self.config.radiogenic_heat,
        )

        # Time tracking
        self.time = 0.0        # years
        self.step_count = 0

    def initialize(self) -> None:
        """Create grid, build initial stratigraphy, solve initial steady-state thermal."""
        if self._initialized:
            return

        # Create initial stratigraphy
        sed_config = SedimentationConfig(
            layers=self.config.initial_layers,
            surface_temp=self.config.surface_temp,
            geothermal_gradient=self.config.geothermal_gradient,
        )
        self.grid = create_initial_stratigraphy(self.grid_config, sed_config)

        # Solve steady-state thermal if requested
        if self.config.steady_state_init:
            solve_steady_state(
                self.grid,
                surface_temp=self.config.surface_temp,
                mantle_heat_flux=self.config.mantle_heat_flux,
                rock_materials=self.materials,
            )

        self._initialized = True

    def step(self, dt: float) -> None:
        """Advance all geological processes by dt (years)."""
        if not self._initialized:
            self.initialize()

        # Thermal conduction (sub-step if needed)
        thermal_dt = self.config.thermal_dt
        remaining = dt
        while remaining > 1e-6:
            step_dt = min(thermal_dt, remaining)
            self.thermal_config.dt = step_dt
            solve_thermal_step(self.grid, self.thermal_config, self.materials)
            remaining -= step_dt

        # TODO: Phase 2 - Stream Power Law erosion/sedimentation
        # TODO: Phase 3 - Tectonic stress, Mohr-Coulomb failure

        self.time += dt
        self.step_count += 1

    def get_slice_xy(self, iz: int) -> dict:
        """Get XY slice for rendering."""
        if self.grid is None:
            return {}
        return self.grid.get_slice_xy(iz)

    def get_slice_xz(self, iy: int) -> dict:
        """Get XZ slice for rendering."""
        if self.grid is None:
            return {}
        return self.grid.get_slice_xz(iy)

    def get_slice_yz(self, ix: int) -> dict:
        """Get YZ slice for rendering."""
        if self.grid is None:
            return {}
        return self.grid.get_slice_yz(ix)

    def get_surface_elevation(self) -> np.ndarray:
        """Get surface elevation map (ny, nx) in meters."""
        if self.grid is None:
            return np.zeros((self.grid_config.ny, self.grid_config.nx), dtype=np.float32)
        from .processes.sedimentation import compute_surface_elevation
        return compute_surface_elevation(self.grid)

    def get_material_properties_for_gpu(self) -> dict:
        """Return material property arrays for GPU upload."""
        rocks = all_rocks()
        n = len(rocks)
        return {
            "colors": np.array([r.color for r in rocks], dtype=np.float32),
            "erosion_resistance": np.array([r.erosion_resistance for r in rocks], dtype=np.float32),
            "thermal_conductivity": np.array([r.thermal_conductivity for r in rocks], dtype=np.float32),
            "melting_point": np.array([r.melting_point for r in rocks], dtype=np.float32),
            "density": np.array([r.density for r in rocks], dtype=np.float32),
            "count": n,
        }


if __name__ == "__main__":
    # Quick integration test
    config = GeologySolverConfig(
        domain_size=(320.0, 320.0, 160.0),
        cell_resolution=10.0,
        initial_layers=[
            (20.0, "sediment"),
            (50.0, "sandstone"),
            (80.0, "shale"),
            (60.0, "granite"),
        ],
        steady_state_init=True,
    )
    solver = GeologySolver(config)
    print(f"Grid: {solver.grid_config.shape} cells")
    print(f"Domain: {solver.grid_config.domain_size} m")

    solver.initialize()
    print(f"Initialized. Temp range: {solver.grid.temperature.min():.1f} - {solver.grid.temperature.max():.1f} K")

    # Step a few times
    for _ in range(5):
        solver.step(1000.0)  # 1000 years per step
    print(f"After 5000 years: Temp range: {solver.grid.temperature.min():.1f} - {solver.grid.temperature.max():.1f} K")

    # Check slice
    slice_data = solver.get_slice_xy(solver.grid_config.nz // 2)
    print(f"Mid slice rocks: {np.unique(slice_data['rock_id'])}")