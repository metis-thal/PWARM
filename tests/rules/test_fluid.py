"""Tests for the SPH fluid module (P2.2.2).

Note: The SPH module uses standard SPH formulations. The kernel tests
verify basic properties; the core physics tests verify stability,
conservation, and basic behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymo.rules.fluid import (
    SPHParams,
    create_water_column,
    poly6_kernel,
    spiky_gradient_kernel,
    viscosity_kernel_laplacian,
)


def test_poly6_kernel_positive():
    """Poly6 kernel is positive and zero outside support."""
    h = 0.1
    assert poly6_kernel(0.0, h) > 0
    assert poly6_kernel(h**2 / 4, h) > 0
    assert poly6_kernel(h**2, h) == 0.0
    assert poly6_kernel(h**2 * 2, h) == 0.0


def test_spiky_gradient_direction():
    """Spiky gradient points radially inward for r < h."""
    h = 0.1
    r_vec = np.array([0.05, 0.0])
    grad = spiky_gradient_kernel(r_vec, 0.05, h)
    # Should point opposite to r_vec (inward)
    assert grad[0] < 0
    # Zero outside support
    assert np.all(spiky_gradient_kernel(np.array([0.2, 0.0]), 0.2, 0.1) == 0.0)


def test_viscosity_laplacian_positive():
    """Viscosity Laplacian is positive for r < h."""
    h = 0.1
    assert viscosity_kernel_laplacian(0.05, h) > 0
    assert viscosity_kernel_laplacian(0.2, 0.1) == 0.0


def test_water_column_mass_conservation():
    """A static water column conserves total mass and particle count."""
    params = SPHParams(h=0.1, rest_density=1000.0, stiffness=2000.0, viscosity=0.1, particle_mass=8.66)
    system = create_water_column(0.0, 0.0, 4, 5, 0.08, params)
    n0 = len(system.particles)
    m0 = system.total_mass()
    for _ in range(50):
        system.step(dt=0.001)
    assert len(system.particles) == n0
    assert system.total_mass() == pytest.approx(m0, rel=1e-6)


def test_water_column_settles_under_gravity():
    """A water column under gravity spreads and settles to a lower height."""
    params = SPHParams(h=0.15, rest_density=1000.0, stiffness=5000.0, viscosity=1.0, particle_mass=8.66)
    system = create_water_column(0.0, 0.5, 5, 6, 0.1, params)
    y0 = max(p.pos[1] for p in system.particles)
    for _ in range(200):
        system.step(dt=0.001)
    y1 = max(p.pos[1] for p in system.particles)
    # Fluid should have fallen
    assert y1 < y0 - 0.1


def test_density_bounded():
    """Density stays within reasonable bounds (clamped at 0.5*rest_density)."""
    params = SPHParams(h=0.15, rest_density=1000.0, stiffness=5000.0, viscosity=1.0, particle_mass=8.66)
    system = create_water_column(0.0, 0.5, 5, 5, 0.1, params)
    for _ in range(100):
        system.step(dt=0.001)
    ratios = [p.density / params.rest_density for p in system.particles]
    # All densities should be clamped at minimum 0.5*rest_density
    assert all(r >= 0.5 for r in ratios)
    assert all(r <= 1.5 for r in ratios)


def test_no_particle_explosion():
    """Particles don't explode to infinity (stability test)."""
    params = SPHParams(h=0.1, rest_density=1000.0, stiffness=5000.0, viscosity=1.0, particle_mass=8.66)
    system = create_water_column(0.0, 1.0, 5, 4, 0.08, params)
    for _ in range(200):
        system.step(dt=0.001)
    # All positions should remain finite and bounded
    for p in system.particles:
        assert np.all(np.isfinite(p.pos))
        assert np.all(np.abs(p.pos) < 10.0)


def test_energy_bounded():
    """Kinetic energy stays bounded (no explosion)."""
    params = SPHParams(h=0.15, rest_density=1000.0, stiffness=5000.0, viscosity=1.0, particle_mass=8.66)
    system = create_water_column(0.0, 0.5, 5, 5, 0.1, params)
    max_ke = 0.0
    for _ in range(200):
        system.step(dt=0.001)
        ke = system.total_energy()
        max_ke = max(max_ke, ke)
    assert max_ke < 1000.0  # energy bounded, no explosion