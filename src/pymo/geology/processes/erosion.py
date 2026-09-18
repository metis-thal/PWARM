"""Stream Power Law erosion process.

Erosion rate: E = K * A^m * S^n

where:
    K = erodibility coefficient (depends on rock type, climate)
    A = upstream drainage area (m²)
    S = local slope (dimensionless)
    m, n = exponents (typically m/n ≈ 0.5)

This drives river incision into the landscape, creating valleys and canyons.
Combined with tectonic uplift, it shapes the balance between mountain building
and destruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ErosionConfig:
    """Configuration for Stream Power Law erosion."""
    K: float = 1.0e-5          # erodibility coefficient (m^(1-2m)/yr)
    m: float = 0.5             # area exponent
    n: float = 1.0             # slope exponent
    min_slope: float = 1.0e-4  # minimum slope to avoid singularity
    max_erosion_rate: float = 0.01  # m/yr cap to prevent numerical instability
    diffusion_coeff: float = 0.01   # hillslope diffusion (m²/yr)


def compute_drainage_area(
    elevation: np.ndarray,
    nx: int, ny: int,
    cell_size: float,
) -> np.ndarray:
    """Compute upstream drainage area using D8 flow direction.

    Each cell contributes its area to the downslope neighbor.
    Returns drainage area in m² for each cell.
    """
    n = nx * ny
    area = np.full(n, cell_size * cell_size, dtype=np.float32)  # each cell starts with its own area

    # Sort cells by elevation (highest first) for priority-flood
    elevation.reshape(ny, nx)
    sorted_indices = np.argsort(-elevation)  # descending

    # D8 neighbor offsets (8-connected)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    dist = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)

    for idx in sorted_indices:
        ix = idx % nx
        iy = idx // nx
        e0 = elevation[idx]

        # Find steepest downslope neighbor
        best_slope = -1.0
        best_neighbor = -1

        for d in range(8):
            jx = ix + dx[d]
            jy = iy + dy[d]
            if 0 <= jx < nx and 0 <= jy < ny:
                j_idx = jx + nx * jy
                slope = (e0 - elevation[j_idx]) / (cell_size * dist[d])
                if slope > best_slope:
                    best_slope = slope
                    best_neighbor = j_idx

        # Add this cell's area to the downslope neighbor
        if best_neighbor >= 0 and best_slope > 0:
            area[best_neighbor] += area[idx]

    return area


def compute_local_slope(
    elevation: np.ndarray,
    nx: int, ny: int,
    cell_size: float,
) -> np.ndarray:
    """Compute local slope magnitude using central differences.

    Returns slope (dimensionless) for each cell.
    """
    elev_2d = elevation.reshape(ny, nx)
    slope_y, slope_x = np.gradient(elev_2d, cell_size)
    slope_mag = np.sqrt(slope_x**2 + slope_y**2)
    return slope_mag.ravel().astype(np.float32)


def compute_hillslope_diffusion(
    elevation: np.ndarray,
    nx: int, ny: int,
    cell_size: float,
    diffusion_coeff: float,
    dt: float,
) -> np.ndarray:
    """Compute hillslope diffusion (soil creep).

    This smooths the landscape, representing sediment transport
    from high to low areas by gravity-driven creep.

    Returns elevation change per cell (m).
    """
    elev_2d = elevation.reshape(ny, nx)
    h2 = cell_size * cell_size

    # Laplacian via finite differences
    laplacian = np.zeros_like(elev_2d)
    laplacian[1:-1, 1:-1] = (
        elev_2d[2:, 1:-1] + elev_2d[:-2, 1:-1] +
        elev_2d[1:-1, 2:] + elev_2d[1:-1, :-2] -
        4 * elev_2d[1:-1, 1:-1]
    ) / h2

    # Boundary conditions (zero flux)
    laplacian[0, :] = 0
    laplacian[-1, :] = 0
    laplacian[:, 0] = 0
    laplacian[:, -1] = 0

    return (diffusion_coeff * laplacian * dt).ravel().astype(np.float32)


def apply_stream_power_erosion(
    elevation: np.ndarray,
    rock_id: np.ndarray,
    nx: int, ny: int,
    cell_size: float,
    config: ErosionConfig,
    dt: float,
) -> np.ndarray:
    """Apply Stream Power Law erosion to surface elevation.

    Args:
        elevation: (N,) surface elevation in meters
        rock_id: (N,) rock type IDs (affects erodibility)
        nx, ny: grid dimensions
        cell_size: meters per cell
        config: erosion parameters
        dt: time step in years

    Returns:
        erosion_amount: (N,) meters of material removed (positive = erosion)
    """
    nx * ny

    # Compute drainage area and slope
    area = compute_drainage_area(elevation, nx, ny, cell_size)
    slope = compute_local_slope(elevation, nx, ny, cell_size)

    # Clamp slope to avoid singularity
    slope = np.maximum(slope, config.min_slope)

    # Stream Power Law erosion rate
    # E = K * A^m * S^n
    erosion_rate = config.K * np.power(area, config.m) * np.power(slope, config.n)

    # Cap erosion rate for stability
    erosion_rate = np.minimum(erosion_rate, config.max_erosion_rate)

    # Rock type modifier (harder rocks erode slower)
    # granite=0: 0.3x, basalt=1: 0.5x, sandstone=2: 1.0x, shale=3: 1.5x, sediment=5: 2.0x
    rock_factor = np.ones(rock_id.max() + 1, dtype=np.float32)
    rock_factor[0] = 0.3   # granite - very resistant
    rock_factor[1] = 0.5   # basalt - resistant
    rock_factor[2] = 1.0   # sandstone - moderate
    rock_factor[3] = 1.5   # shale - weak
    if len(rock_factor) > 5:
        rock_factor[5] = 2.0   # sediment - very weak

    rock_mod = rock_factor[rock_id]
    erosion_rate *= rock_mod

    # Total erosion over dt
    erosion = erosion_rate * dt

    # Add hillslope diffusion (smooths the landscape)
    diffusion = compute_hillslope_diffusion(
        elevation, nx, ny, cell_size, config.diffusion_coeff, dt
    )

    # Net erosion (positive = material removed)
    net_erosion = erosion - diffusion

    return net_erosion.astype(np.float32)
