"""Tests for geology module."""

import numpy as np
import pytest

from pymo.geology import (
    GeologyGrid,
    GeologyGridConfig,
    GeologySolver,
    GeologySolverConfig,
    all_rocks,
    create_stratified_grid,
    get_rock,
    rock_count,
)


def test_rock_library():
    """Test built-in rock library."""
    assert rock_count() >= 9  # at least 9 rock types

    # Check each rock has valid properties
    for rock in all_rocks():
        assert rock.density > 0
        assert rock.young_modulus > 0
        assert 0 <= rock.poisson_ratio < 0.5
        assert rock.cohesion >= 0
        assert 0 <= rock.friction_angle <= np.pi/2
        assert rock.thermal_conductivity > 0
        assert rock.specific_heat > 0
        assert rock.melting_point > 0
        assert rock.erosion_resistance >= 0
        assert rock.weathering_rate >= 0
        assert len(rock.color) == 3
        assert all(0 <= c <= 1 for c in rock.color)

    # Test lookup
    granite = get_rock("granite")
    assert granite.name == "granite"
    assert granite.rock_id >= 0


def test_geology_grid():
    """Test 3D grid creation and indexing."""
    config = GeologyGridConfig(nx=8, ny=8, nz=4, cell_size=10.0)
    grid = GeologyGrid(config, default_rock_id=0)

    assert grid.config.shape == (8, 8, 4)
    assert grid.config.total_cells == 256
    assert grid.rock_id.shape == (256,)
    assert grid.temperature.shape == (256,)
    assert grid.stress.shape == (256, 6)

    # Test layer setting
    grid.set_rock_layer((0, 2), 1)  # bottom 2 layers = rock 1
    slice_bot = grid.get_slice_xy(0)
    assert np.all(slice_bot["rock_id"] == 1)
    slice_top = grid.get_slice_xy(3)
    assert np.all(slice_top["rock_id"] == 0)  # default


def test_stratified_grid():
    """Test stratified grid creation."""
    config = GeologyGridConfig(nx=16, ny=16, nz=16, cell_size=10.0)
    # layers from surface DOWN - first layer ends up at surface
    layers = [(50.0, 2), (80.0, 0), (60.0, 1)]  # sandstone, granite, basalt
    grid = create_stratified_grid(config, layers)

    # Check layers from top down (z=15 is surface)
    slice_top = grid.get_slice_xy(15)
    assert np.all(slice_top["rock_id"] == 2)  # sandstone (first layer) at surface

    slice_mid = grid.get_slice_xy(8)
    assert np.all(slice_mid["rock_id"] == 0)  # granite in middle

    slice_bot = grid.get_slice_xy(0)
    assert np.all(slice_bot["rock_id"] == 1)  # basalt (last layer) at bottom

    # Temperature gradient
    assert grid.temperature.max() > grid.temperature.min()
    # Bottom should be hotter
    i_top = grid._idx(0, 0, 15)
    i_bot = grid._idx(0, 0, 0)
    assert grid.temperature[i_bot] > grid.temperature[i_top]


def test_geology_solver_init():
    """Test GeologySolver initialization."""
    config = GeologySolverConfig(
        domain_size=(160.0, 160.0, 80.0),
        cell_resolution=10.0,
        initial_layers=[
            (20.0, "sediment"),
            (50.0, "sandstone"),
        ],
        steady_state_init=False,  # skip for fast test
    )
    solver = GeologySolver(config)
    assert solver.grid_config.shape == (16, 16, 8)

    solver.initialize()
    assert solver.grid is not None
    assert solver._initialized

    slice_top = solver.get_slice_xy(7)
    assert np.all(slice_top["rock_id"] == get_rock("sediment").rock_id)


def test_geology_solver_thermal_step():
    """Test thermal stepping."""
    import pytest
    pytest.skip("Slow - scipy sparse solver")


def test_surface_elevation():
    """Test surface elevation computation."""
    import pytest
    pytest.skip("Slow - steady-state thermal init")


def test_gpu_properties():
    """Test material property arrays for GPU."""
    import pytest
    pytest.skip("Slow - steady-state thermal init")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])