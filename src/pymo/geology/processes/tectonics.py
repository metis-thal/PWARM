"""Tectonic uplift process for mountain building.

Models the effect of plate collision driving material upward.
For the Himalayas, this represents the India-Eurasia collision
pushing the Tibetan Plateau and Himalayan range upward.

Uplift rate varies spatially based on:
- Distance from the main collision front
- Existing elevation (isostatic feedback)
- Rock strength (weaker rocks deform more easily)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TectonicConfig:
    """Configuration for tectonic uplift."""
    # Base uplift rate (m/yr)
    base_uplift_rate: float = 0.005  # 5 mm/yr for Himalayas

    # Spatial variation
    # The uplift is strongest at the collision front and decreases northward
    front_position: float = 0.5      # fraction of domain where collision front is
    front_width: float = 0.3         # width of active uplift zone (fraction)
    decay_rate: float = 2.0          # exponential decay away from front

    # Elevation feedback (isostatic)
    # Higher areas uplift slower (mountain root compensation)
    elevation_feedback: bool = True
    max_elevation: float = 9000.0    # meters, above which uplift decreases
    feedback_strength: float = 0.5   # how strongly elevation reduces uplift

    # Lateral variation (rand factor for realism)
    noise_amplitude: float = 0.2     # 20% variation
    noise_scale: float = 0.01        # spatial frequency of noise


def generate_uplift_field(
    nx: int, ny: int,
    config: TectonicConfig,
    seed: int = 42,
) -> np.ndarray:
    """Generate a spatially varying uplift rate field.

    Returns uplift_rate: (nx*ny,) in m/yr
    """
    nx * ny
    rng = np.random.default_rng(seed)

    # Create coordinate grids
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    xx, _yy = np.meshgrid(x, y)  # (ny, nx)

    # Distance from collision front (along x-axis)
    front_x = config.front_position
    dist_from_front = np.abs(xx - front_x)

    # Gaussian-like uplift zone centered on front
    sigma = config.front_width / 2.0
    uplift_spatial = np.exp(-0.5 * (dist_from_front / sigma) ** 2)

    # Add some noise for realism
    noise = rng.normal(1.0, config.noise_amplitude, (ny, nx)).astype(np.float32)
    noise = np.clip(noise, 0.5, 1.5)

    # Combine
    uplift_field = config.base_uplift_rate * uplift_spatial * noise

    return uplift_field.ravel().astype(np.float32)


def apply_tectonic_uplift(
    elevation: np.ndarray,
    nx: int, ny: int,
    cell_size: float,
    config: TectonicConfig,
    dt: float,
    uplift_field: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply tectonic uplift to surface elevation.

    Args:
        elevation: (N,) surface elevation in meters
        nx, ny: grid dimensions
        cell_size: meters per cell
        config: tectonic parameters
        dt: time step in years
        uplift_field: (N,) pre-computed uplift rates (m/yr), or None to generate

    Returns:
        uplift_amount: (N,) meters of material added (positive = uplift)
        new_uplift_rates: (N,) actual uplift rates used (m/yr)
    """
    nx * ny

    if uplift_field is None:
        uplift_field = generate_uplift_field(nx, ny, config)

    rates = uplift_field.copy()

    # Elevation feedback: reduce uplift at high elevations
    if config.elevation_feedback:
        elev_factor = 1.0 - config.feedback_strength * np.clip(
            elevation / config.max_elevation, 0, 1
        )
        rates *= elev_factor

    # Total uplift over dt
    uplift = rates * dt

    return uplift.astype(np.float32), rates.astype(np.float32)


def apply_isostatic_adjustment(
    elevation: np.ndarray,
    thickness: np.ndarray,
    density_crust: float = 2700.0,
    density_mantle: float = 3300.0,
    compensation_depth: float = 100000.0,  # 100 km
) -> np.ndarray:
    """Apply isostatic adjustment (Airy isostasy).

    Thicker crust floats higher on the mantle, like an iceberg.

    Args:
        elevation: (N,) surface elevation
        thickness: (N,) crustal thickness
        density_crust: crust density (kg/m³)
        density_mantle: mantle density (kg/m³)
        compensation_depth: depth of compensation

    Returns:
        adjustment: (N,) elevation change from isostasy
    """
    # Airy isostasy: h = (ρ_m - ρ_c) / ρ_c * thickness
    density_ratio = (density_mantle - density_crust) / density_crust
    adjustment = density_ratio * thickness * 0.001  # scale to reasonable units

    return adjustment.astype(np.float32)
