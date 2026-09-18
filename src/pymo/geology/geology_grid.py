"""3D Geology grid data structure.

Structured grid of GeologyCell, each cell stores:
- rock_id (int)
- temperature (float, K)
- stress tensor (6 components, symmetric 3x3)
- porosity (float)
- uplift_rate (float, m/yr)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GeologyGridConfig:
    """Configuration for 3D geology grid."""
    nx: int
    ny: int
    nz: int
    cell_size: float          # meters per cell (uniform)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world coords of (0,0,0) cell center

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def domain_size(self) -> tuple[float, float, float]:
        return (self.nx * self.cell_size, self.ny * self.cell_size, self.nz * self.cell_size)

    def world_to_grid(self, pos: np.ndarray) -> tuple[int, int, int]:
        """Convert world position to grid indices (floor)."""
        x = int((pos[0] - self.origin[0]) / self.cell_size)
        y = int((pos[1] - self.origin[1]) / self.cell_size)
        z = int((pos[2] - self.origin[2]) / self.cell_size)
        return np.clip(x, 0, self.nx - 1), np.clip(y, 0, self.ny - 1), np.clip(z, 0, self.nz - 1)

    def grid_to_world_center(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Get world coordinates of cell center."""
        return np.array([
            self.origin[0] + (ix + 0.5) * self.cell_size,
            self.origin[1] + (iy + 0.5) * self.cell_size,
            self.origin[2] + (iz + 0.5) * self.cell_size,
        ], dtype=np.float32)


# ============================================================
# Structured grid storage: flat arrays for GPU-friendly layout
# ============================================================

class GeologyGrid:
    """3D structured grid of geology cells.

    Memory layout: flat arrays indexed as [ix + nx*(iy + ny*iz)]
    All arrays are float32 except rock_id (uint16).
    """

    def __init__(self, config: GeologyGridConfig, default_rock_id: int = 0):
        self.config = config
        n = config.total_cells
        self.nx, self.ny, self.nz = config.nx, config.ny, config.nz

        # Core fields
        self.rock_id = np.full(n, default_rock_id, dtype=np.uint16)
        self.temperature = np.full(n, 293.15, dtype=np.float32)  # K
        self.porosity = np.zeros(n, dtype=np.float32)
        self.uplift_rate = np.zeros(n, dtype=np.float32)         # m/yr

        # Stress tensor: 6 components (xx, yy, zz, xy, yz, zx) for symmetric 3x3
        self.stress = np.zeros((n, 6), dtype=np.float32)

        # Derived / cache
        self._stress_xx = self.stress[:, 0]
        self._stress_yy = self.stress[:, 1]
        self._stress_zz = self.stress[:, 2]
        self._stress_xy = self.stress[:, 3]
        self._stress_yz = self.stress[:, 4]
        self._stress_zx = self.stress[:, 5]

    def _idx(self, ix: int, iy: int, iz: int) -> int:
        return ix + self.nx * (iy + self.ny * iz)

    def get_cell_data(self, ix: int, iy: int, iz: int) -> dict:
        """Get all fields for a cell as dict (for debugging)."""
        i = self._idx(ix, iy, iz)
        return {
            "rock_id": int(self.rock_id[i]),
            "temperature": float(self.temperature[i]),
            "porosity": float(self.porosity[i]),
            "uplift_rate": float(self.uplift_rate[i]),
            "stress": self.stress[i].copy(),
        }

    def set_rock_layer(self, z_range: tuple[int, int], rock_id: int):
        """Set rock_id for all cells in z range [z_min, z_max)."""
        z_min, z_max = z_range
        z_min = max(0, z_min)
        z_max = min(self.nz, z_max)
        for iz in range(z_min, z_max):
            base = iz * self.nx * self.ny
            self.rock_id[base:base + self.nx * self.ny] = rock_id

    def set_temperature_gradient(self, surface_temp: float = 293.15, gradient: float = 0.025):
        """Set initial linear geothermal gradient: T(z) = surface + gradient * depth.

        gradient in K/m (typical 0.025 K/m = 25°C/km).
        Depth measured from surface (z = nz-1) downward.
        """
        for iz in range(self.nz):
            z_depth = (self.nz - 1 - iz) * self.config.cell_size
            temp = surface_temp + gradient * z_depth
            base = iz * self.nx * self.ny
            self.temperature[base:base + self.nx * self.ny] = temp

    def get_slice_xy(self, iz: int) -> dict:
        """Get XY slice at given Z index for rendering.

        Returns dict with 2D arrays (nx, ny) for each field.
        """
        iz = np.clip(iz, 0, self.nz - 1)
        base = iz * self.nx * self.ny
        end = base + self.nx * self.ny
        return {
            "rock_id": self.rock_id[base:end].reshape(self.ny, self.nx).copy(),
            "temperature": self.temperature[base:end].reshape(self.ny, self.nx).copy(),
            "porosity": self.porosity[base:end].reshape(self.ny, self.nx).copy(),
            "stress": self.stress[base:end].copy(),
        }

    def get_slice_xz(self, iy: int) -> dict:
        """Get XZ slice at given Y index."""
        iy = np.clip(iy, 0, self.ny - 1)
        # Non-contiguous in flat array, need to gather
        rock_id = np.zeros((self.nz, self.nx), dtype=np.uint16)
        temp = np.zeros((self.nz, self.nx), dtype=np.float32)
        for iz in range(self.nz):
            for ix in range(self.nx):
                i = self._idx(ix, iy, iz)
                rock_id[iz, ix] = self.rock_id[i]
                temp[iz, ix] = self.temperature[i]
        return {"rock_id": rock_id, "temperature": temp}

    def get_slice_yz(self, ix: int) -> dict:
        """Get YZ slice at given X index."""
        ix = np.clip(ix, 0, self.nx - 1)
        rock_id = np.zeros((self.nz, self.ny), dtype=np.uint16)
        temp = np.zeros((self.nz, self.ny), dtype=np.float32)
        for iz in range(self.nz):
            for iy in range(self.ny):
                i = self._idx(ix, iy, iz)
                rock_id[iz, iy] = self.rock_id[i]
                temp[iz, iy] = self.temperature[i]
        return {"rock_id": rock_id, "temperature": temp}


# ============================================================
# Helper: create grid with initial stratigraphy
# ============================================================

def create_stratified_grid(
    config: GeologyGridConfig,
    layers: list[tuple[float, int]],  # list of (thickness_m, rock_id) from surface down
    surface_temp: float = 293.15,
    geothermal_gradient: float = 0.025,
) -> GeologyGrid:
    """Create grid with horizontal layers.

    layers: [(thickness_m, rock_id), ...] from top (surface) down.
    Remaining cells filled with last rock_id.
    """
    grid = GeologyGrid(config)

    # Fill layers from surface DOWN (top of grid)
    # Grid iz=0 is bottom, iz=nz-1 is surface
    current_iz = config.nz - 1  # start at surface
    for thickness, rock_id in layers:
        nz_layer = int(thickness / config.cell_size)
        nz_layer = min(nz_layer, current_iz + 1)
        if nz_layer <= 0:
            continue
        z_start = current_iz - nz_layer + 1
        z_end = current_iz + 1  # exclusive
        grid.set_rock_layer((z_start, z_end), rock_id)
        current_iz = z_start - 1
        if current_iz < 0:
            break

    # Fill any remaining cells at bottom with last rock_id
    if current_iz >= 0 and layers:
        last_rock = layers[-1][1]
        grid.set_rock_layer((0, current_iz + 1), last_rock)

    # Set initial temperature
    grid.set_temperature_gradient(surface_temp, geothermal_gradient)

    return grid


if __name__ == "__main__":
    # Quick test
    config = GeologyGridConfig(nx=32, ny=32, nz=16, cell_size=10.0)
    layers = [(50.0, 2), (100.0, 0), (60.0, 1)]  # sandstone, granite, basalt
    grid = create_stratified_grid(config, layers)
    print(f"Grid: {grid.config.shape}, cell_size={grid.config.cell_size}m")
    print(f"Domain: {grid.config.domain_size}m")
    print(f"Rock IDs at surface (z={config.nz-1}): {grid.get_slice_xy(config.nz-1)['rock_id'][0,0]}")
    print(f"Rock IDs at bottom (z=0): {grid.get_slice_xy(0)['rock_id'][0,0]}")
    slice_mid = grid.get_slice_xy(config.nz//2)
    print(f"Mid slice unique rocks: {np.unique(slice_mid['rock_id'])}")
    print(f"Temperature range: {grid.temperature.min():.1f} - {grid.temperature.max():.1f} K")