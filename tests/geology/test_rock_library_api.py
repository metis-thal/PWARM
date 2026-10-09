"""Tests for the rock library and grid coordinate/stratigraphy API.

Complements tests/geology/test_geology.py (which covers the constructor and
basic stratigraphy): error paths of the registry, coordinate round-trips,
slice accessors, temperature gradients, and the stratigraphy edge branches.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pwarm.geology import (
    GeologyGrid,
    GeologyGridConfig,
    create_stratified_grid,
    get_rock,
    get_rock_by_id,
)
from pwarm.geology.rock_materials import (
    all_rocks,
    get_color_array,
    get_erosion_resistance_array,
    get_melting_point_array,
    get_thermal_conductivity_array,
    rock_count,
)


# ------------------------------------------------------------------ rock registry
def test_rock_count_is_nine_and_ids_ordered() -> None:
    """Nine registered rocks with contiguous ids 0..8."""
    rocks = all_rocks()
    assert rock_count() == 9
    assert [r.rock_id for r in rocks] == list(range(9))


def test_get_rock_unknown_name_raises_keyerror() -> None:
    """An unknown rock name raises KeyError naming the valid options."""
    with pytest.raises(KeyError, match="Unknown rock"):
        get_rock("unobtainium")


def test_get_rock_by_id_roundtrip_and_bounds() -> None:
    """get_rock_by_id round-trips; negative and too-large ids raise IndexError."""
    granite = get_rock("granite")
    assert get_rock_by_id(granite.rock_id) is granite
    with pytest.raises(IndexError, match="Invalid rock_id"):
        get_rock_by_id(-1)
    with pytest.raises(IndexError, match="Invalid rock_id"):
        get_rock_by_id(rock_count())


def test_shear_strength_mohr_coulomb() -> None:
    """shear_strength = cohesion + normal_stress * tan(friction_angle)."""
    granite = get_rock("granite")
    expected = granite.cohesion + 1e6 * np.tan(granite.friction_angle)
    assert granite.shear_strength(1e6) == pytest.approx(expected, rel=1e-9)


def test_is_molten_inclusive_boundary() -> None:
    """is_molten is True at exactly the melting point."""
    magma = get_rock("magma")
    assert magma.is_molten(magma.melting_point) is True
    assert magma.is_molten(magma.melting_point - 1.0) is False


def test_rock_material_is_frozen() -> None:
    """RockMaterial instances are immutable."""
    granite = get_rock("granite")
    with pytest.raises(dataclasses.FrozenInstanceError):
        granite.density = 1.0  # type: ignore[misc]


def test_gpu_property_arrays() -> None:
    """The four array getters return float32 arrays ordered by rock id."""
    n = rock_count()
    colors = get_color_array()
    assert colors.shape == (n, 3) and colors.dtype == np.float32
    for getter in (get_erosion_resistance_array, get_thermal_conductivity_array,
                   get_melting_point_array):
        arr = getter()
        assert arr.shape == (n,) and arr.dtype == np.float32
    # values match the registry order
    assert get_thermal_conductivity_array()[0] == pytest.approx(
        get_rock("granite").thermal_conductivity)


# ------------------------------------------------------------- grid coordinates
@pytest.fixture
def config() -> GeologyGridConfig:
    return GeologyGridConfig(nx=4, ny=3, nz=5, cell_size=10.0,
                             origin=(100.0, 200.0, 300.0))


def test_config_properties(config: GeologyGridConfig) -> None:
    """shape/total_cells/domain_size derive from the config fields."""
    assert config.shape == (4, 3, 5)
    assert config.total_cells == 60
    assert config.domain_size == (40.0, 30.0, 50.0)


def test_world_to_grid_truncates_and_clamps(config: GeologyGridConfig) -> None:
    """Coordinates map by truncation and clamp to the domain."""
    GeologyGrid(config)  # construction is part of the contract; mapping is config-side
    assert config.world_to_grid(np.array([125.0, 205.0, 340.0])) == (2, 0, 4)
    # out-of-domain clamps per axis
    assert config.world_to_grid(np.array([-50.0, 500.0, 1000.0])) == (0, 2, 4)


def test_grid_to_world_center_roundtrip(config: GeologyGridConfig) -> None:
    """Cell centers map back into the same cell via world_to_grid."""
    GeologyGrid(config)
    for ix, iy, iz in [(0, 0, 0), (3, 2, 4), (1, 1, 2)]:
        center = config.grid_to_world_center(ix, iy, iz)
        assert tuple(config.world_to_grid(center)) == (ix, iy, iz)
    assert config.grid_to_world_center(0, 0, 0) == pytest.approx(
        [105.0, 205.0, 305.0])


def test_get_cell_data(config: GeologyGridConfig) -> None:
    """get_cell_data exposes rock id, temperature, porosity and stress copy."""
    grid = GeologyGrid(config, default_rock_id=2)
    grid.temperature[grid._idx(1, 1, 1)] = 500.0
    cell = grid.get_cell_data(1, 1, 1)
    assert cell["rock_id"] == 2
    assert cell["temperature"] == pytest.approx(500.0)
    assert cell["porosity"] == 0.0
    assert cell["uplift_rate"] == 0.0
    assert cell["stress"].shape == (6,)


def test_set_rock_layer_clamps_and_ignores_empty(config: GeologyGridConfig) -> None:
    """Layer ranges clamp to the grid; empty ranges are silent no-ops."""
    grid = GeologyGrid(config, default_rock_id=0)
    grid.set_rock_layer((-5, 2), 3)  # clamped to (0, 2)
    assert np.all(grid.rock_id[: 2 * 4 * 3] == 3)
    assert np.all(grid.rock_id[2 * 4 * 3:] == 0)
    before = grid.rock_id.copy()
    grid.set_rock_layer((4, 2), 7)  # empty range
    assert np.array_equal(grid.rock_id, before)


def test_set_temperature_gradient_exact_values(config: GeologyGridConfig) -> None:
    """Top layer is exactly surface_temp; deeper layers add gradient*depth."""
    grid = GeologyGrid(config)
    grid.set_temperature_gradient(surface_temp=290.0, gradient=0.03)
    nxy = 4 * 3
    top = grid.temperature[4 * nxy:]  # iz = nz-1
    assert np.all(top == pytest.approx(290.0))
    bottom = grid.temperature[0:nxy]  # iz = 0 -> depth (nz-1)*cell
    assert np.all(bottom == pytest.approx(290.0 + 0.03 * 4 * 10.0))


def test_slice_accessors(config: GeologyGridConfig) -> None:
    """Slices clip their index and return (copies of) 2-D layer data."""
    grid = GeologyGrid(config, default_rock_id=1)
    xy = grid.get_slice_xy(99)  # clipped to nz-1
    assert xy["rock_id"].shape == (3, 4)
    assert np.all(xy["rock_id"] == 1)
    assert set(xy) == {"rock_id", "temperature", "porosity", "stress"}
    xz = grid.get_slice_xz(1)
    assert xz["rock_id"].shape == (5, 4)
    yz = grid.get_slice_yz(-7)  # clipped to 0
    assert yz["rock_id"].shape == (5, 3)
    yz["rock_id"][0, 0] = 99  # copies, not views
    assert grid.rock_id[0] == 1


# ------------------------------------------------------------ stratigraphy edges
def test_stratified_grid_empty_layers(config: GeologyGridConfig) -> None:
    """No layers -> default rock everywhere, temperature gradient still set."""
    grid = create_stratified_grid(config, layers=[], surface_temp=280.0,
                                  geothermal_gradient=0.02)
    assert np.all(grid.rock_id == 0)
    # the gradient is still applied: surface (min) is exact, bottom is hotter
    assert grid.temperature.min() == pytest.approx(280.0)
    assert grid.temperature.max() == pytest.approx(280.0 + 0.02 * 4 * 10.0)


def test_stratified_grid_zero_thickness_skipped(config: GeologyGridConfig) -> None:
    """Layers thinner than one cell are silently skipped."""
    grid = create_stratified_grid(
        config, layers=[(5.0, 2), (40.0, 1)],  # 5 m < 10 m cell -> skipped
        surface_temp=280.0)
    top = grid.get_slice_xy(config.nz - 1)["rock_id"]
    assert np.all(top == 1)  # only the second layer appears


def test_stratified_grid_bottom_remainder_fill(config: GeologyGridConfig) -> None:
    """Cells below the listed layers fill with the LAST layer's rock."""
    grid = create_stratified_grid(
        config, layers=[(20.0, 2), (10.0, 1)],  # 2 cells + 1 cell listed
        surface_temp=280.0)
    ids = grid.rock_id.reshape(config.nz, -1)[:, 0]  # one column, bottom-up
    # remainder (2 cells) is filled with the LAST listed layer's rock (1)
    assert list(ids) == [1, 1, 1, 2, 2]
