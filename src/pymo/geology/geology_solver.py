"""Main geology solver integrating all geological processes.

Orchestrates:
- Thermal conduction (geothermal gradient)
- Initial stratigraphy
- (Future) Erosion/sedimentation (Stream Power Law)
- (Future) Tectonic deformation (Mohr-Coulomb failure)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geology_grid import GeologyGrid, GeologyGridConfig
from .processes.erosion import ErosionConfig, apply_stream_power_erosion
from .processes.sedimentation import SedimentationConfig, create_initial_stratigraphy
from .processes.tectonics import TectonicConfig, generate_uplift_field
from .processes.thermal import (
    ThermalConfig,
    get_material_dict,
    solve_steady_state,
    solve_thermal_step,
)
from .rock_materials import all_rocks


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

    # Erosion (Stream Power Law)
    enable_erosion: bool = True
    erosion_K: float = 1.0e-5             # erodibility
    erosion_m: float = 0.5                # area exponent
    erosion_n: float = 1.0                # slope exponent
    diffusion_coeff: float = 0.01         # hillslope diffusion (m²/yr)

    # Tectonics
    enable_tectonics: bool = True
    base_uplift_rate: float = 0.005       # m/yr (5 mm/yr for Himalayas)
    uplift_front_position: float = 0.5    # fraction of domain
    uplift_front_width: float = 0.3       # fraction of domain

    # Solver
    thermal_dt: float = 1000.0            # years per thermal step
    steady_state_init: bool = True        # solve steady-state on init


class GeologySolver:
    """Main geology solver.

    Call step(dt) each simulation tick to advance geological processes.
    """

    def __init__(self, config: GeologySolverConfig | None = None):
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
        self.grid: GeologyGrid | None = None

        # Material library (rock_id -> RockMaterial)
        self.materials = get_material_dict()

        # Thermal config
        self.thermal_config = ThermalConfig(
            dt=self.config.thermal_dt,
            surface_temp=self.config.surface_temp,
            mantle_heat_flux=self.config.mantle_heat_flux,
            radiogenic_heat=self.config.radiogenic_heat,
        )

        # Erosion config
        self.erosion_config = ErosionConfig(
            K=self.config.erosion_K,
            m=self.config.erosion_m,
            n=self.config.erosion_n,
            diffusion_coeff=self.config.diffusion_coeff,
        )

        # Tectonic config
        self.tectonic_config = TectonicConfig(
            base_uplift_rate=self.config.base_uplift_rate,
            front_position=self.config.uplift_front_position,
            front_width=self.config.uplift_front_width,
        )

        # Pre-computed uplift field (reused across steps)
        self._uplift_field: np.ndarray | None = None

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

        # 1. Tectonic uplift (pushes surface upward)
        if self.config.enable_tectonics:
            if self._uplift_field is None:
                self._uplift_field = generate_uplift_field(
                    self.grid_config.nx, self.grid_config.ny,
                    self.tectonic_config, seed=42,
                )
            self._apply_tectonics(dt)

        # 2. Thermal conduction (sub-step if needed)
        thermal_dt = self.config.thermal_dt
        remaining = dt
        while remaining > 1e-6:
            step_dt = min(thermal_dt, remaining)
            self.thermal_config.dt = step_dt
            solve_thermal_step(self.grid, self.thermal_config, self.materials)
            remaining -= step_dt

        # 3. Erosion (Stream Power Law + hillslope diffusion)
        if self.config.enable_erosion:
            self._apply_erosion(dt)

        self.time += dt
        self.step_count += 1

    def _apply_tectonics(self, dt: float) -> None:
        """Apply tectonic uplift to surface cells."""
        nx, ny = self.grid_config.nx, self.grid_config.ny
        nz = self.grid_config.nz
        cell_size = self.grid_config.cell_size

        # Find surface elevation and apply uplift
        for iy in range(ny):
            for ix in range(nx):
                # Find topmost non-air cell
                iz_surface = -1
                for iz in range(nz - 1, -1, -1):
                    idx = ix + nx * (iy + ny * iz)
                    if self.grid.rock_id[idx] != 0:  # non-air
                        iz_surface = iz
                        break

                if iz_surface < 0:
                    continue

                # Get uplift amount
                flat_idx = ix + nx * iy
                uplift_amount = self._uplift_field[flat_idx] * dt

                # Apply elevation feedback
                elev = self.grid_config.origin[2] + (iz_surface + 0.5) * cell_size
                if self.tectonic_config.elevation_feedback:
                    factor = 1.0 - self.tectonic_config.feedback_strength * min(
                        elev / self.tectonic_config.max_elevation, 1.0
                    )
                    uplift_amount *= max(factor, 0.1)

                # Convert to cells and shift surface
                cell_uplift = int(uplift_amount / cell_size)
                if cell_uplift > 0 and iz_surface + cell_uplift < nz:
                    # Shift columns upward
                    for dy in range(cell_uplift):
                        iz_from = iz_surface - dy
                        iz_to = iz_surface + cell_uplift - dy
                        if iz_from >= 0 and iz_to < nz:
                            for ix2 in range(max(0, ix - 1), min(nx, ix + 2)):
                                idx_from = ix2 + nx * (iy + ny * iz_from)
                                idx_to = ix2 + nx * (iy + ny * iz_to)
                                self.grid.rock_id[idx_to] = self.grid.rock_id[idx_from]
                                self.grid.temperature[idx_to] = self.grid.temperature[idx_from]

    def _apply_erosion(self, dt: float) -> None:
        """Apply Stream Power Law erosion to surface."""
        nx, ny = self.grid_config.nx, self.grid_config.ny
        nz = self.grid_config.nz
        cell_size = self.grid_config.cell_size

        # Extract surface elevation
        surface_elev = np.zeros(nx * ny, dtype=np.float32)
        surface_rock = np.zeros(nx * ny, dtype=np.uint16)

        for iy in range(ny):
            for ix in range(nx):
                iz_surface = -1
                for iz in range(nz - 1, -1, -1):
                    idx = ix + nx * (iy + ny * iz)
                    if self.grid.rock_id[idx] != 0:
                        iz_surface = iz
                        break
                flat_idx = ix + nx * iy
                if iz_surface >= 0:
                    surface_elev[flat_idx] = self.grid_config.origin[2] + (iz_surface + 0.5) * cell_size
                    surface_rock[flat_idx] = self.grid.rock_id[ix + nx * (iy + ny * iz_surface)]

        # Compute erosion
        erosion = apply_stream_power_erosion(
            surface_elev, surface_rock,
            nx, ny, cell_size,
            self.erosion_config, dt,
        )

        # Apply erosion to grid (remove material from surface cells)
        for iy in range(ny):
            for ix in range(nx):
                flat_idx = ix + nx * iy
                erosion_m = erosion[flat_idx]
                cells_to_remove = int(erosion_m / cell_size)

                if cells_to_remove <= 0:
                    continue

                # Find surface
                iz_surface = -1
                for iz in range(nz - 1, -1, -1):
                    idx = ix + nx * (iy + ny * iz)
                    if self.grid.rock_id[idx] != 0:
                        iz_surface = iz
                        break

                if iz_surface < 0:
                    continue

                # Remove cells from surface
                for dy in range(min(cells_to_remove, iz_surface + 1)):
                    idx = ix + nx * (iy + ny * (iz_surface - dy))
                    self.grid.rock_id[idx] = 0  # air
                    self.grid.temperature[idx] = self.config.surface_temp

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