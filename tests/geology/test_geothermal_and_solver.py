"""Tests for the geology thermal solver and the GeologySolver facade.

Scope: implicit thermal matrix assembly, the single thermal step (including
its melting-point clamp quirk), the steady-state Gauss-Seidel solver with its
boundary conditions and console output, and the solver facade (lazy init,
step accounting, process toggles, slice/elevation/GPU accessors).
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.geology import (
    GeologyGrid,
    GeologyGridConfig,
    GeologySolver,
    GeologySolverConfig,
    all_rocks,
    get_rock,
)
from pwarm.geology.processes.thermal import (
    ThermalConfig,
    build_thermal_matrices,
    get_material_dict,
    solve_steady_state,
    solve_thermal_step,
)


@pytest.fixture
def small_grid() -> GeologyGrid:
    config = GeologyGridConfig(nx=3, ny=3, nz=3, cell_size=1.0)
    return GeologyGrid(config, default_rock_id=0)  # granite everywhere


def test_thermal_config_requires_dt() -> None:
    """ThermalConfig has no default dt."""
    with pytest.raises(TypeError):
        ThermalConfig()  # type: ignore[call-arg]


def test_build_thermal_matrices_stencil(small_grid: GeologyGrid) -> None:
    """Uniform media give the 7-point stencil: diag 1+6a, off-diag -a."""
    n = 27
    rock_k = np.full(n, 2.0)
    rho_c = np.full(n, 2.0e6)
    A, b = build_thermal_matrices(small_grid, rock_k, rho_c, dt=1.0, cell_size=1.0)
    assert b.shape == (n,) and np.all(b == 0.0)
    assert A.shape == (n, n)
    alpha_k = 1.0 / 2.0e6 * 2.0  # dt/(rho_c h^2) * k_face (harmonic mean = k)
    centre = small_grid._idx(1, 1, 1)
    # inspect the centre row of the csr matrix directly
    row_start = A.indptr[centre]
    row_end = A.indptr[centre + 1]
    cols = A.indices[row_start:row_end]
    vals = A.data[row_start:row_end]
    assert len(cols) == 7  # self + 6 neighbours
    diag_mask = cols == centre
    assert vals[diag_mask][0] == pytest.approx(1.0 + 6.0 * alpha_k, rel=1e-5)
    assert np.allclose(vals[~diag_mask], -alpha_k)


def test_solve_thermal_step_clamps_to_melting(small_grid: GeologyGrid) -> None:
    """Pinned as implemented: every cell ends at least at its rock's melting
    point (the clamp uses np.maximum, the opposite of superheat prevention)."""
    small_grid.temperature[:] = 300.0
    materials = get_material_dict()
    solve_thermal_step(small_grid, ThermalConfig(dt=10.0), materials)
    melt = np.array([materials[rid].melting_point for rid in small_grid.rock_id])
    assert np.all(small_grid.temperature >= melt - 1e-3)


def test_solve_steady_state_profile_and_bcs(small_grid: GeologyGrid) -> None:
    """Top layer pinned to surface temp; bottom is hotter; converges."""
    solve_steady_state(small_grid, surface_temp=293.15, mantle_heat_flux=0.065,
                       max_iter=500, tolerance=1e-8)
    nxy = 9
    top = small_grid.temperature[2 * nxy:]
    assert np.all(top == pytest.approx(293.15, abs=1e-3))
    bottom = small_grid.temperature[:nxy].mean()
    assert bottom > 293.15  # mantle heat flux warms the base


def test_solve_steady_state_reports_non_convergence(
        small_grid: GeologyGrid, capsys: pytest.CaptureFixture[str]) -> None:
    """A tolerance below reach prints the non-convergence message."""
    solve_steady_state(small_grid, max_iter=2, tolerance=1e-30)
    out = capsys.readouterr().out
    assert "did not converge" in out


def test_solve_steady_state_reports_convergence(
        small_grid: GeologyGrid, capsys: pytest.CaptureFixture[str]) -> None:
    """A reachable tolerance prints the convergence message."""
    solve_steady_state(small_grid, max_iter=2000, tolerance=1e-5)
    out = capsys.readouterr().out
    assert "converged" in out


def test_get_material_dict_covers_all_rocks() -> None:
    """The material dictionary maps every registered rock id."""
    materials = get_material_dict()
    assert set(materials) == {r.rock_id for r in all_rocks()}
    assert materials[0] is get_rock("granite")


# ------------------------------------------------------------------ solver API
def _solver_config() -> GeologySolverConfig:
    return GeologySolverConfig(
        domain_size=(40.0, 40.0, 40.0), cell_resolution=10.0,
        initial_layers=[(10.0, "sediment"), (20.0, "sandstone")],
        steady_state_init=False)


def test_default_config_is_constructible() -> None:
    """The default config resolves to the documented 100x100x50 grid."""
    config = GeologySolverConfig()
    assert config.domain_size == (1000.0, 1000.0, 500.0)
    assert config.cell_resolution == 10.0
    solver = GeologySolver(config)
    assert solver.grid_config.shape == (100, 100, 50)


def test_initialize_is_idempotent() -> None:
    """A second initialize() call reuses the same grid object."""
    solver = GeologySolver(_solver_config())
    solver.initialize()
    grid_first = solver.grid
    assert grid_first is not None
    assert solver._initialized is True
    solver.initialize()
    assert solver.grid is grid_first


def test_step_accounting_and_thermal_clamp() -> None:
    """step() advances time/step_count and runs the thermal solve."""
    solver = GeologySolver(_solver_config())
    solver.step(dt=0.0)  # lazy init + no thermal substeps at dt <= 1e-6
    assert solver.step_count == 1
    assert solver.time == 0.0
    solver.step(dt=1.5)
    assert solver.step_count == 2
    assert solver.time == pytest.approx(1.5)
    # the thermal step applies its melting clamp (documented quirk)
    materials = solver.materials
    melt = np.array([materials[rid].melting_point for rid in solver.grid.rock_id])
    assert np.all(solver.grid.temperature >= melt - 1e-3)


def test_step_with_processes_disabled() -> None:
    """With tectonics/erosion disabled only the thermal solver runs."""
    config = _solver_config()
    config.enable_tectonics = False
    config.enable_erosion = False
    solver = GeologySolver(config)
    solver.step(dt=1.0)
    assert solver.step_count == 1 and solver.time == pytest.approx(1.0)


def test_slices_before_initialize_are_empty() -> None:
    """Slice accessors return {} until a grid exists."""
    solver = GeologySolver(_solver_config())
    assert solver.get_slice_xy(0) == {}
    assert solver.get_slice_xz(0) == {}
    assert solver.get_slice_yz(0) == {}


def test_surface_elevation_and_gpu_properties() -> None:
    """Elevation needs a grid; GPU properties always describe all rocks."""
    solver = GeologySolver(_solver_config())
    assert solver.get_surface_elevation().shape == (4, 4)
    assert np.all(solver.get_surface_elevation() == 0.0)  # no grid yet
    props = solver.get_material_properties_for_gpu()
    assert props["count"] == 9
    assert props["colors"].shape == (9, 3)
    assert props["density"].shape == (9,)
    solver.initialize()
    elevation = solver.get_surface_elevation()
    assert elevation.shape == (4, 4)
    # one sediment cell on top, sandstone below -> surface at 3 cells * 10 m
    assert np.all(elevation == 30.0)
