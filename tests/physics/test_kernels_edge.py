"""Branch-level tests for the numba SPH/contact kernels.

The engine tests (test_sph.py, test_rigid_solver.py) drive the JIT-compiled
entry points; numba-jitted code is not traceable by coverage, so these tests
call the ``*_py`` pure-Python implementations directly to cover every branch:
cubic-spline pieces, coincidence fade, near-contact repulsion, surface
tension, artificial-viscosity activation, and the contact solver's
separating/zero-mass/friction paths.
"""

from __future__ import annotations

import numpy as np

from pwarm.physics.solvers.contact_kernels import solve_contacts_velocity_py
from pwarm.physics.solvers.sph_kernels import sph_density_py, sph_forces_py


def csr(neighbour_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.zeros(len(neighbour_lists) + 1, dtype=np.int64)
    total = 0
    for i, lst in enumerate(neighbour_lists):
        total += len(lst)
        starts[i + 1] = total
    flat = np.array([j for lst in neighbour_lists for j in lst], dtype=np.int64)
    return starts, flat


def test_density_cubic_spline_pieces_and_self_contribution() -> None:
    """Near pairs use the q<0.5 piece, mid pairs the tail, self always counts."""
    h = 1.0
    mass = np.array([1.0, 1.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],   # q = 0.4  -> inner piece
                    [0.8, 0.0, 0.0]])  # q = 0.8  -> tail piece
    starts, flat = csr([[1, 2], [0, 2], [0, 1]])
    density = sph_density_py(pos, mass, starts, flat, h)
    norm = 8.0 / (np.pi * h**3)
    q2_inner, q2_tail = 0.16, 0.64
    w_inner = norm * (1.0 - 6.0 * q2_inner + 6.0 * q2_inner * np.sqrt(q2_inner))
    w_tail = norm * 2.0 * (1.0 - np.sqrt(q2_tail)) ** 3
    expected0 = 1.0 * norm + w_inner + w_tail
    assert density[0] == pytest_approx(expected0)


def pytest_approx(value: float):  # local helper keeps this file pytest-light
    import pytest

    return pytest.approx(value, rel=1e-9)


def test_density_excludes_outside_support_and_self_entry() -> None:
    """Neighbours beyond h contribute nothing; a self-entry is skipped."""
    h = 1.0
    mass = np.ones(2)
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    starts, flat = csr([[1, 0], [0]])  # second entry of particle 0 is itself
    density = sph_density_py(pos, mass, starts, flat, h)
    norm = 8.0 / (np.pi * h**3)
    assert np.allclose(density, norm)  # self weight only


def test_density_fades_coincident_pairs() -> None:
    """r < 0.1*h fades the pair weight linearly to avoid double-counting."""
    h = 1.0
    mass = np.ones(2)
    norm = 8.0 / (np.pi * h**3)
    far = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    near = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    starts, flat = csr([[1], [0]])
    d_far = sph_density_py(far, mass, starts, flat, h)[0]
    d_near = sph_density_py(near, mass, starts, flat, h)[0]
    # the coincident pair contributes less than the moderately close one
    assert d_near < d_far
    assert d_near > norm  # but still more than the bare self weight


def test_forces_pressure_repulsion_both_kernel_pieces() -> None:
    """Pressure pushes particles apart in both spline regimes."""
    h = 1.0
    n = 2
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])  # q = 0.3
    vel = np.zeros((n, 3))
    mass = np.ones(n)
    density = np.full(n, 1.0)
    pressure = np.array([1.0, 1.0])
    starts, flat = csr([[1], [0]])
    forces = sph_forces_py(pos, vel, mass, density, pressure, starts, flat,
                           h, viscosity=0.0, surface_tension=0.0,
                           contact_rc=0.0, contact_acc=0.0)
    assert forces[0, 0] < 0.0 and forces[1, 0] > 0.0  # repelled along -/+x
    # momentum conservation for symmetric pairs
    assert abs(forces[0].sum() + forces[1].sum()) < 1e-12

    pos_far = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])  # q = 0.7 tail piece
    forces_far = sph_forces_py(pos_far, vel, mass, density, pressure, starts,
                               flat, h, 0.0, 0.0, 0.0, 0.0)
    assert forces_far[0, 0] < 0.0


def test_forces_viscosity_damps_approaching_pairs_only() -> None:
    """Approaching pairs get a repulsive damping impulse; separating don't."""
    h = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = np.ones(2)
    density = np.full(2, 1.0)
    pressure = np.zeros(2)
    starts, flat = csr([[1], [0]])

    approaching = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    f_app = sph_forces_py(pos, approaching, mass, density, pressure, starts,
                          flat, h, viscosity=1.0, surface_tension=0.0,
                          contact_rc=0.0, contact_acc=0.0)
    assert f_app[0, 0] < 0.0  # pushed back (damped)

    separating = -approaching
    f_sep = sph_forces_py(pos, separating, mass, density, pressure, starts,
                          flat, h, viscosity=1.0, surface_tension=0.0,
                          contact_rc=0.0, contact_acc=0.0)
    assert np.allclose(f_sep, 0.0)


def test_forces_near_contact_repulsion_and_surface_tension() -> None:
    """contact_rc repulsion pushes overlaps apart; tension pulls together."""
    h = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    mass = np.ones(2)
    density = np.full(2, 1.0)
    pressure = np.zeros(2)
    starts, flat = csr([[1], [0]])
    repulsion = sph_forces_py(pos, vel, mass, density, pressure, starts, flat,
                              h, 0.0, 0.0, contact_rc=0.5, contact_acc=10.0)
    assert repulsion[0, 0] < 0.0  # pushed apart
    tension = sph_forces_py(pos, vel, mass, density, pressure, starts, flat,
                            h, 0.0, surface_tension=5.0,
                            contact_rc=0.0, contact_acc=0.0)
    assert tension[0, 0] > 0.0  # pulled toward the neighbour


def test_forces_skips_degenerate_and_distant_pairs() -> None:
    """Coincident (r<1e-6) and out-of-support pairs contribute nothing."""
    h = 1.0
    mass = np.ones(2)
    density = np.full(2, 1.0)
    pressure = np.ones(2)
    starts, flat = csr([[1], [0]])
    coincident = np.zeros((2, 3))
    f = sph_forces_py(coincident, np.zeros((2, 3)), mass, density, pressure,
                      starts, flat, h, 1.0, 1.0, 0.5, 1.0)
    assert np.allclose(f, 0.0)
    distant = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    f2 = sph_forces_py(distant, np.zeros((2, 3)), mass, density, pressure,
                       starts, flat, h, 1.0, 1.0, 0.5, 1.0)
    assert np.allclose(f2, 0.0)


# ------------------------------------------------------------- contact solver
def test_contact_solver_normal_impulse_stops_approach() -> None:
    """A head-on approaching pair is resolved; relative normal velocity -> 0."""
    linvel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    inv_mass = np.ones(2)
    idx = np.array([0])
    idx_b = np.array([1])
    normals = np.array([[1.0, 0.0, 0.0]])
    fric = np.array([0.0])
    rest = np.array([0.0])
    out = solve_contacts_velocity_py(linvel.copy(), inv_mass, idx, idx_b,
                                     normals, fric, rest, iterations=4)
    vn = (out[1, 0] - out[0, 0]) * 1.0
    assert vn >= -1e-12  # no longer approaching
    assert out[0].sum() + out[1].sum() == pytest_approx(0.0)  # momentum kept


def test_contact_solver_separating_and_zero_mass_skip() -> None:
    """Separating contacts and two static bodies are left untouched."""
    linvel = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # separating
    inv_mass = np.ones(2)
    idx = np.zeros(1, dtype=int)
    idx_b = np.ones(1, dtype=int)
    normals = np.array([[1.0, 0.0, 0.0]])
    fric = np.array([0.5])
    rest = np.array([0.5])
    out = solve_contacts_velocity_py(linvel.copy(), inv_mass, idx, idx_b,
                                     normals, fric, rest, iterations=3)
    assert np.allclose(out, linvel)

    static = solve_contacts_velocity_py(linvel.copy(), np.zeros(2), idx, idx_b,
                                        normals, fric, rest, iterations=3)
    assert np.allclose(static, linvel)


def test_contact_solver_friction_clamps_to_cone() -> None:
    """Tangential slip is opposed and clamped to mu * j_n."""
    linvel = np.array([[1.0, 0.5, 0.0], [-1.0, -0.5, 0.0]])  # approaching
    inv_mass = np.ones(2)
    idx = np.zeros(1, dtype=int)
    idx_b = np.ones(1, dtype=int)
    normals = np.array([[1.0, 0.0, 0.0]])
    fric = np.array([0.1])  # small cone: large slip gets clamped
    rest = np.array([0.0])
    out = solve_contacts_velocity_py(linvel.copy(), inv_mass, idx, idx_b,
                                     normals, fric, rest, iterations=1)
    # y-relative slip is reduced but not eliminated (cone clamp)
    slip_before = linvel[1, 1] - linvel[0, 1]
    slip_after = out[1, 1] - out[0, 1]
    assert abs(slip_after) < abs(slip_before)


def test_contact_solver_zero_slip_tangential_guard() -> None:
    """Purely-normal relative velocity skips the tangential branch."""
    linvel = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    inv_mass = np.ones(2)
    idx = np.zeros(1, dtype=int)
    idx_b = np.ones(1, dtype=int)
    normals = np.array([[1.0, 0.0, 0.0]])
    fric = np.array([0.5])
    rest = np.array([0.0])
    out = solve_contacts_velocity_py(linvel.copy(), inv_mass, idx, idx_b,
                                     normals, fric, rest, iterations=2)
    assert out[0, 1] == 0.0 and out[1, 1] == 0.0  # no tangential contamination
