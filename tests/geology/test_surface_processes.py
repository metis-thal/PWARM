"""Tests for the surface-process modules: erosion, sedimentation, tectonics.

Scope: D8 drainage accumulation, slope and hillslope diffusion, the stream
power law (rock factor, cap, and the rock_id<3 sizing defect), sediment layer
addition branches, surface elevation extraction, and the tectonic uplift /
isostatic adjustment fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.geology import GeologyGrid, GeologyGridConfig, get_rock
from pwarm.geology.processes.erosion import (
    ErosionConfig,
    apply_stream_power_erosion,
    compute_drainage_area,
    compute_hillslope_diffusion,
    compute_local_slope,
)
from pwarm.geology.processes.sedimentation import (
    SedimentationConfig,
    add_sediment_layer,
    compute_surface_elevation,
    create_initial_stratigraphy,
)
from pwarm.geology.processes.tectonics import (
    TectonicConfig,
    apply_isostatic_adjustment,
    apply_tectonic_uplift,
    generate_uplift_field,
)


# --------------------------------------------------------------------- erosion
def test_drainage_area_accumulates_downhill() -> None:
    """A 1-D ramp accumulates drainage area from ridge to outlet."""
    elev = np.array([3.0, 2.0, 1.0])
    area = compute_drainage_area(elev, nx=3, ny=1, cell_size=1.0)
    # every cell drains to the outlet at index 2, which accumulates all area
    assert area == pytest.approx([1.0, 2.0, 3.0])


def test_drainage_area_flat_terrain_no_flow() -> None:
    """A flat surface has no downslope direction: every cell keeps its own."""
    elev = np.full(6, 5.0)
    area = compute_drainage_area(elev, nx=3, ny=2, cell_size=2.0)
    assert np.all(area == pytest.approx(4.0))


def test_local_slope_ramp_and_flat() -> None:
    """A unit ramp has slope 1; flat terrain has slope 0."""
    ramp = np.tile(np.array([3.0, 2.0, 1.0]), 2)
    slope = compute_local_slope(ramp, nx=3, ny=2, cell_size=1.0)
    assert np.allclose(slope, 1.0, atol=1e-6)
    flat = np.zeros(6)
    assert np.all(compute_local_slope(flat, nx=3, ny=2, cell_size=1.0) == 0.0)


def test_hillslope_diffusion_smooths_peak_and_fills_pit() -> None:
    """Peaks erode (negative) and pits fill (positive) under diffusion."""
    elev = np.zeros(9)
    elev[4] = 10.0  # centre peak in a 3x3 grid
    diff = compute_hillslope_diffusion(elev, nx=3, ny=3, cell_size=1.0,
                                       diffusion_coeff=1.0, dt=1.0)
    assert diff[4] < 0.0  # peak is eroded
    pit = np.zeros(9)
    pit[4] = -10.0
    diff_pit = compute_hillslope_diffusion(pit, nx=3, ny=3, cell_size=1.0,
                                           diffusion_coeff=1.0, dt=1.0)
    assert diff_pit[4] > 0.0  # pit is filled


def test_stream_power_erosion_scales_with_rock_resistance() -> None:
    """Shale (id 3, factor 1.5) erodes faster than sandstone (id 2, 1.0)."""
    # the rock-factor table is sized by rock_id.max(), so every grid here
    # must contain at least one cell with id >= 3 (shale) to be indexable
    elev = np.array([3.0, 2.0, 1.0, 0.5])
    sand = apply_stream_power_erosion(
        elev, np.array([2, 2, 2, 3]), nx=2, ny=2, cell_size=1.0,
        config=ErosionConfig(), dt=1000.0)
    shale = apply_stream_power_erosion(
        elev, np.array([3, 3, 3, 3]), nx=2, ny=2, cell_size=1.0,
        config=ErosionConfig(), dt=1000.0)
    assert np.all(shale >= sand)  # weaker rock never erodes less
    assert np.any(shale > sand)  # ... and strictly more on sand cells


def test_stream_power_erosion_low_rock_id_raises() -> None:
    """The rock-factor table is sized by rock_id.max(): ids < 3 raise IndexError."""
    elev = np.array([3.0, 2.0, 1.0, 0.5])
    with pytest.raises(IndexError):
        apply_stream_power_erosion(elev, np.array([0, 0, 0, 0]), nx=2, ny=2,
                                   cell_size=1.0, config=ErosionConfig(), dt=1.0)


def test_stream_power_erosion_empty_input_raises() -> None:
    """Empty elevation arrays fail on the rock-factor size reduction."""
    with pytest.raises(ValueError):
        apply_stream_power_erosion(np.array([]), np.array([], dtype=int),
                                   nx=0, ny=0, cell_size=1.0,
                                   config=ErosionConfig(), dt=1.0)


def test_stream_power_erosion_capped_rate() -> None:
    """Even extreme parameters stay below max_erosion_rate * dt."""
    elev = np.array([100.0, 0.0, 0.0, 0.0])
    config = ErosionConfig(K=1e3, max_erosion_rate=0.01)
    result = apply_stream_power_erosion(elev, np.array([3, 3, 3, 3]),
                                        nx=2, ny=2, cell_size=1.0,
                                        config=config, dt=1e6)
    # the rate cap is applied before the rock factor (shale x1.5)
    assert np.all(result <= config.max_erosion_rate * 1.5 * 1e6 + 1e-3)


# --------------------------------------------------------------- sedimentation
def _grid(nz: int = 4) -> GeologyGrid:
    config = GeologyGridConfig(nx=2, ny=2, nz=nz, cell_size=10.0)
    grid = GeologyGrid(config, default_rock_id=5)  # sediment everywhere
    return grid


def test_add_sediment_layer_sub_cell_thickness_is_noop() -> None:
    """A layer thinner than one cell leaves the grid untouched."""
    grid = _grid()
    before = grid.rock_id.copy()
    add_sediment_layer(grid, 5.0, "sandstone")  # 5 m < 10 m cell
    assert np.array_equal(grid.rock_id, before)


def test_add_sediment_layer_thicker_than_grid_fills_all() -> None:
    """A layer at least as thick as the whole column replaces everything."""
    grid = _grid()
    grid.set_rock_layer((0, 2), 1)
    add_sediment_layer(grid, 100.0, "limestone")
    assert np.all(grid.rock_id == get_rock("limestone").rock_id)


def test_add_sediment_layer_shifts_column_down() -> None:
    """Adding one cell of sediment rewrites the first flat block.

    Pinned as implemented: in the flat layout (iz slowest) ``rock_id[:shift]``
    is the BOTTOM layer, so the new rock lands at the column base even though
    the docstring says "adds at surface". The docstring/layout mismatch is a
    known quirk, not part of the contract.
    """
    grid = _grid(nz=3)
    grid.rock_id[:] = 1  # sandstone column
    add_sediment_layer(grid, 10.0, "sediment")
    assert list(grid.rock_id.reshape(3, -1)[:, 0]) == [5, 1, 1]  # bottom-up
    assert np.all(grid.porosity[:4] == 0.3)  # first flat block gets porosity


def test_add_sediment_layer_unknown_rock_raises() -> None:
    """An unknown rock name propagates KeyError."""
    with pytest.raises(KeyError):
        add_sediment_layer(_grid(), 10.0, "unobtainium")


def test_create_initial_stratigraphy_maps_names_to_ids() -> None:
    """Name-based layers map through the rock registry."""
    config = GeologyGridConfig(nx=2, ny=2, nz=4, cell_size=10.0)
    sed = SedimentationConfig(layers=[(10.0, "sediment"), (30.0, "shale")])
    grid = create_initial_stratigraphy(config, sed)
    ids = grid.rock_id.reshape(4, -1)[:, 0]
    assert list(ids) == [get_rock("shale").rock_id, get_rock("shale").rock_id,
                         get_rock("shale").rock_id, get_rock("sediment").rock_id]


def test_compute_surface_elevation_columns() -> None:
    """Elevation is the top of the uppermost non-sediment cell per column."""
    config = GeologyGridConfig(nx=3, ny=1, nz=4, cell_size=10.0)
    grid = GeologyGrid(config, default_rock_id=5)  # all sediment
    elev = compute_surface_elevation(grid)
    assert np.all(elev == 40.0)  # fully sediment columns keep the grid top
    # column 1: non-sediment below two sediment cells -> surface at 2 cells
    grid.rock_id[grid._idx(1, 0, 1)] = 2
    elev = compute_surface_elevation(grid)
    assert elev[0, 1] == pytest.approx(20.0)
    assert elev[0, 0] == pytest.approx(40.0)
    # column 2: non-sediment at the very top still reports the grid top
    grid.rock_id[grid._idx(2, 0, 3)] = 2
    assert compute_surface_elevation(grid)[0, 2] == pytest.approx(40.0)


# ------------------------------------------------------------------- tectonics
def test_uplift_field_deterministic_and_bounded() -> None:
    """Same seed -> identical field; rates stay within [0, base * 1.5]."""
    config = TectonicConfig(base_uplift_rate=0.005)
    a = generate_uplift_field(8, 6, config, seed=42)
    b = generate_uplift_field(8, 6, config, seed=42)
    assert np.array_equal(a, b)
    assert a.shape == (48,)
    assert np.all(a >= 0.0)
    assert np.all(a <= 0.005 * 1.5 + 1e-12)


def test_uplift_field_peaks_at_front() -> None:
    """Without noise the Gaussian peaks at the collision front."""
    config = TectonicConfig(base_uplift_rate=0.005, front_position=0.5,
                            front_width=0.4, noise_amplitude=0.0)
    field = generate_uplift_field(11, 1, config, seed=0).reshape(1, 11)
    assert field[0, 5] == pytest.approx(0.005)  # x=0.5 -> dist 0
    assert field[0, 0] < 0.001  # far from the front


def test_apply_tectonic_uplift_with_feedback() -> None:
    """High elevation damps uplift by the feedback strength."""
    config = TectonicConfig(base_uplift_rate=0.005, elevation_feedback=True,
                            feedback_strength=0.5, max_elevation=1000.0)
    uplift_field = np.full(4, 0.005)
    low = apply_tectonic_uplift(np.zeros(4), 2, 2, 1.0, config, dt=1000.0,
                                uplift_field=uplift_field)
    high = apply_tectonic_uplift(np.full(4, 1000.0), 2, 2, 1.0, config,
                                 dt=1000.0, uplift_field=uplift_field)
    assert np.all(low[0] == pytest.approx(5.0))  # 0.005 * 1000, no damping
    assert np.all(high[0] == pytest.approx(2.5))  # damped by factor 0.5
    assert np.all(high[1] == pytest.approx(0.0025))  # damped rates returned


def test_apply_tectonic_uplift_generates_field_when_none() -> None:
    """Without a field, one is generated deterministically (seed 42)."""
    config = TectonicConfig(base_uplift_rate=0.005, noise_amplitude=0.0)
    uplift, rates = apply_tectonic_uplift(np.zeros(4), 2, 2, 1.0, config,
                                          dt=1000.0, uplift_field=None)
    expected_field = generate_uplift_field(2, 2, config, seed=42)
    assert np.allclose(rates, expected_field, atol=1e-12)
    assert np.allclose(uplift, rates * 1000.0)


def test_isostatic_adjustment_scales_with_thickness() -> None:
    """Airy adjustment = (rho_m - rho_c)/rho_c * thickness * 1e-3."""
    thickness = np.array([0.0, 1000.0])
    adjust = apply_isostatic_adjustment(np.zeros(2), thickness)
    assert adjust[0] == 0.0
    assert adjust[1] == pytest.approx(600.0 / 2700.0 * 1000.0 * 0.001)
