"""Tests for SPH kernels (numba JIT + pure-Python fallback) and the fluid pipeline.

The kernels are validated against independently written brute-force O(n²)
reference loops using the documented cubic-spline formulas, so the CSR
conversion, kernel math, and pair iteration are all cross-checked.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pymo.physics import WorldEngine, WorldEngineConfig
from pymo.physics.solvers.sph_kernels import (
    neighbors_to_csr,
    sph_density,
    sph_forces,
)

H = 0.05  # smoothing length matching SPHOptions defaults (2 * 0.025)


def reference_density(pos: np.ndarray, mass: np.ndarray, h: float) -> np.ndarray:
    """Direct O(n^2) cubic-spline density (self-contribution included)."""
    n = len(pos)
    density = np.zeros(n)
    norm = 8.0 / (np.pi * h ** 3)
    for i in range(n):
        acc = mass[i] * norm  # kernel(0, h)
        for j in range(n):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            r2 = float(np.dot(diff, diff))
            if r2 < h * h:
                q2 = r2 / h ** 2
                # Cubic-spline pieces split at q = 0.5 (q2 = 0.25).
                if q2 < 0.25:
                    w = norm * (1.0 - 6.0 * q2 + 6.0 * q2 * np.sqrt(q2))
                else:
                    t = 1.0 - np.sqrt(q2)
                    w = norm * 2.0 * t ** 3
                # Coincidence guard (matches sph_kernels.sph_density): fade
                # the pair contribution to zero as r -> 0 (full weight only
                # beyond r = 0.1*h) so an exact overlap cannot double-count
                # the self-weight.
                fade = r2 / (0.01 * h * h)
                if fade < 1.0:
                    w *= fade
                acc += mass[j] * w
        density[i] = acc
    return density


def reference_forces(pos, vel, mass, density, pressure, h, viscosity,
                     surface_tension, contact_rc, contact_acc) -> np.ndarray:
    """Direct O(n^2) pair forces: pressure + artificial viscosity +
    near-contact repulsion + surface tension, following the formulas
    documented in SPHSolver."""
    n = len(pos)
    forces = np.zeros((n, 3))
    h2 = h * h
    norm = 48.0 / (np.pi * h ** 3)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = pos[j] - pos[i]
            r_norm = float(np.linalg.norm(r))
            if r_norm >= h or r_norm < 1e-6:
                continue
            # Correct cubic-spline gradient: dW/dr = norm*(3q^2-2q)/h (q<0.5),
            # -norm*(1-q)^2/h (q>=0.5); vanishes linearly as q->0.
            q = r_norm / h
            if q < 0.5:
                factor = norm * (3.0 * q * q - 2.0 * q) / h
            else:
                t = 1.0 - q
                factor = -norm * t * t / h
            grad_w = r * factor / r_norm

            # Pressure (repulsive): grad_w = dW/dr * r_hat(j-i) points i <- j
            # (dW/dr < 0), which equals -grad_i W(r_i - r_j), so the Monaghan
            # force F_i = -m_j * p_term * grad_i W is +m_j * p_term * grad_w.
            p_term = (pressure[i] / (density[i] ** 2 + 1e-6)
                      + pressure[j] / (density[j] ** 2 + 1e-6))
            forces[i] += mass[j] * p_term * grad_w

            # Viscosity (Monaghan 1992, approaching pairs only):
            # mu_ij = h * (v_i - v_j) . (r_i - r_j) / (|r_ij|^2 + 0.01 h^2)
            # is negative for approaching pairs; vanishes linearly as r -> 0.
            v_ij = vel[i] - vel[j]
            v_dot_r = float(np.dot(v_ij, -r))
            if v_dot_r < 0:
                mu = h * v_dot_r / (r_norm ** 2 + 0.01 * h2)
                pi_ij = (-viscosity * mu + 0.1 * mu ** 2) / \
                        ((density[i] + density[j]) * 0.5)
                forces[i] += mass[j] * pi_ij * grad_w

            # Near-contact repulsion (matches sph_forces): the EOS is clamped
            # at p=0 for rho < rest density, so sparse neighborhoods (free
            # surface, wall piles) cannot resist interpenetration without it.
            if r_norm < contact_rc:
                red_mass = (mass[i] * mass[j]) / (mass[i] + mass[j])
                fc = red_mass * contact_acc * (1.0 - r_norm / contact_rc)
                forces[i] -= fc * r / r_norm

            # Surface tension (central cohesion: attractive, along -grad_w)
            if surface_tension > 0:
                forces[i] -= surface_tension * mass[j] * grad_w
    return forces


def make_particle_data(n=40, seed=7):
    """Deterministic random particles inside a unit-ish box."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 0.3, size=(n, 3))
    vel = rng.uniform(-0.5, 0.5, size=(n, 3))
    mass = np.full(n, 0.125)
    return pos, vel, mass


def kernel_inputs(pos, h):
    """KD-tree neighbors -> CSR arrays."""
    tree = cKDTree(pos)
    neighbors = tree.query_ball_tree(tree, h)
    return neighbors_to_csr(neighbors)


class TestNeighborsToCSR:
    def test_conversion_correctness(self):
        lists = [[0, 1, 2], [1], [], [3, 0]]
        starts, flat = neighbors_to_csr(lists)
        assert starts.tolist() == [0, 3, 4, 4, 6]
        assert flat.tolist() == [0, 1, 2, 1, 3, 0]

    def test_empty(self):
        starts, flat = neighbors_to_csr([])
        assert starts.tolist() == [0]
        assert len(flat) == 0


class TestDensityKernel:
    def test_matches_brute_force(self):
        pos, _vel, mass = make_particle_data()
        starts, flat = kernel_inputs(pos, H)
        got = sph_density(pos.astype(np.float64), mass.astype(np.float64),
                          starts, flat, H)
        expected = reference_density(pos, mass, H)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_self_contribution_single_particle(self):
        """Single isolated particle: density = mass * kernel(0, h)
        = mass * 8 / (pi * h^3). Locks the self-contribution fix (the old
        loop only applied it to the last particle due to indentation)."""
        pos = np.array([[0.1, 0.2, 0.3]])
        mass = np.array([0.125])
        starts = np.array([0, 0], dtype=np.int64)
        flat = np.array([], dtype=np.int64)
        d = sph_density(pos.astype(np.float64), mass.astype(np.float64),
                        starts, flat, H)
        expected = 0.125 * 8.0 / (np.pi * H ** 3)
        assert d[0] == pytest.approx(expected, rel=1e-9)


class TestForcesKernel:
    def test_matches_brute_force(self):
        pos, vel, mass = make_particle_data()
        starts, flat = kernel_inputs(pos, H)
        density = reference_density(pos, mass, H)
        pressure = np.maximum(1000.0 * (density / 1000.0 - 1.0), 0.0)
        got = sph_forces(pos.astype(np.float64), vel.astype(np.float64),
                         mass.astype(np.float64), density, pressure,
                         starts, flat, H, 0.1, 0.072, 0.35 * H, 400.0)
        expected = reference_forces(pos, vel, mass, density, pressure,
                                    H, 0.1, 0.072, 0.35 * H, 400.0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)

    def test_zero_viscosity_zero_tension_pure_pressure(self):
        pos, vel, mass = make_particle_data()
        starts, flat = kernel_inputs(pos, H)
        density = reference_density(pos, mass, H)
        pressure = np.maximum(1000.0 * (density / 1000.0 - 1.0), 0.0)
        got = sph_forces(pos.astype(np.float64), vel.astype(np.float64),
                         mass.astype(np.float64), density, pressure,
                         starts, flat, H, 0.0, 0.0, 0.35 * H, 400.0)
        expected = reference_forces(pos, vel, mass, density, pressure,
                                    H, 0.0, 0.0, 0.35 * H, 400.0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)


class TestSPHThroughEngine:
    def test_sph_column_settles(self):
        """5x5x5 particle block dropped above the floor: no NaN, particles
        contained above the floor boundary, net fall, bounded velocities.

        Uses the standard WCSPH lattice (spacing d = particle_radius = h/2),
        the default mass (rest_density * d^3 = 0.015625 kg), a softer Tait
        B = 1429 (sound speed ≈ 3.16 m/s — keeps the discrete pair-stiffness
        oscillation limit above the CFL substep so the pure-Python path can
        afford the substep count), Monaghan viscosity with the full r_ij
        dot, and the solver's CFL-based internal substepping.
        """
        config = WorldEngineConfig(
            dt=1 / 60, substeps=1, gravity=(0.0, 0.0, -9.81),
            sph={"enabled": True, "viscosity": 0.2, "stiffness": 1429.0, "surface_tension": 0.0},
            rigid={"enabled": False},
        )
        engine = WorldEngine(config)
        # Standard WCSPH lattice: spacing d = particle_radius = h/2, so ~26
        # neighbors sit inside the kernel support (spacing = h is degenerate:
        # every neighbor sits exactly on the support boundary with W = 0).
        xs = np.arange(5, dtype=np.float64) * 0.025
        zs = 0.1 + np.arange(5, dtype=np.float64) * 0.025
        grid = np.array([[x, y, z] for x in xs for y in xs for z in zs],
                        dtype=np.float32)
        engine.create_sph_fluid(grid)
        engine.finalize_setup()
        initial_mean_z = float(grid[:, 2].mean())

        for _ in range(60):  # 1 s (fall ~0.124 s + settle)
            state = engine.tick()

        assert np.isfinite(state.sph_pos).all()
        assert np.isfinite(state.sph_vel).all()
        # Bounded speeds rule out an instability blow-up (which reaches 1e5+)
        assert float(np.max(np.linalg.norm(state.sph_vel, axis=1))) < 5.0
        assert float(state.sph_pos[:, 2].min()) >= 0.02  # floor at 0.025
        assert float(state.sph_pos[:, 2].mean()) < initial_mean_z
        assert float(state.sph_pos[:, 2].max()) < 1.0

    def test_entity_to_sph_mapping(self):
        config = WorldEngineConfig(
            dt=1 / 60, substeps=1, gravity=(0.0, 0.0, -9.81),
            sph={"enabled": True}, rigid={"enabled": False},
        )
        engine = WorldEngine(config)
        grid = np.array([[0.0, 0.0, 0.1], [0.04, 0.0, 0.1]], dtype=np.float32)
        engine.create_sph_fluid(grid)
        engine.finalize_setup()
        mapping = engine.scene.double_buffer_read.entity_to_sph
        assert len(mapping) == 1
        assert list(mapping.values()) == [0]
        expected_mass = 1000.0 * 0.025 ** 3  # default mass (rest_density * d^3)
        np.testing.assert_allclose(engine.scene.double_buffer_read.sph_mass,
                                   [expected_mass, expected_mass], rtol=1e-6)
