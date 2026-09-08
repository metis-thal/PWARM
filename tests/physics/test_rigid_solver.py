"""Tests for the rigid body solver through WorldEngine.

Locks in the core dynamics contract of the SDK:
- Free fall matches semi-implicit Euler gravity integration
- Restitution causes strictly decreasing bounces (no energy injection)
- Static bodies never move
- Momentum is conserved exactly across collisions (impulse symmetry)
- Determinism: identical initial conditions produce identical trajectories
"""

import numpy as np
import pytest

from pymo.physics import WorldEngine, WorldEngineConfig


G = 9.81
DT = 1 / 60


def make_engine(*bodies, gravity=(0.0, 0.0, -G)):
    config = WorldEngineConfig(dt=DT, substeps=1, gravity=gravity, rigid={"enabled": True})
    engine = WorldEngine(config)
    for pos, mass, shape, params in bodies:
        engine.create_rigid_body(pos, mass=mass, shape=shape, shape_params=params)
    engine.finalize_setup()
    return engine


# ---------------------------------------------------------------------------
# Free fall
# ---------------------------------------------------------------------------

class TestFreeFall:
    def test_matches_semi_implicit_euler(self):
        """Position after n steps must match semi-implicit Euler integration
        of constant gravity (what the solver implements)."""
        z0 = 10.0
        engine = make_engine(((0, 0, z0), 1.0, "sphere", {"radius": 0.5}))
        n = 120  # 2 seconds
        for _ in range(n):
            state = engine.tick()
        expected_drop = G * DT * DT * n * (n + 1) / 2
        got = float(state.rigid_pos[0][2])
        assert got == pytest.approx(z0 - expected_drop, rel=1e-4)

    def test_near_analytic_parabola(self):
        z0 = 10.0
        engine = make_engine(((0, 0, z0), 1.0, "sphere", {"radius": 0.5}))
        for _ in range(60):
            state = engine.tick()
        t = 60 * DT
        analytic = z0 - 0.5 * G * t * t
        got = float(state.rigid_pos[0][2])
        assert got == pytest.approx(analytic, rel=0.02)  # O(dt) discretization

    def test_velocity_accumulates_linearly(self):
        engine = make_engine(((0, 0, 10), 1.0, "sphere", {"radius": 0.5}))
        for _ in range(60):
            state = engine.tick()
        assert float(state.rigid_linvel[0][2]) == pytest.approx(-G * 1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Collision response through the full tick pipeline
# ---------------------------------------------------------------------------

class TestBounce:
    def test_bounce_apexes_strictly_decreasing(self):
        """e=0.6 drop from z=8: apexes must decay monotonically. Locks the
        regression where an inverted contact normal injected energy."""
        engine = make_engine(
            ((0, 0, 8), 1.0, "sphere", {"radius": 0.5, "restitution": 0.6}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5], "restitution": 0.6}),
        )
        apexes = []
        last_v = 0.0
        cur_apex = 8.0
        for _ in range(900):  # 15 s
            state = engine.tick()
            z = float(state.rigid_pos[0][2])
            v = float(state.rigid_linvel[0][2])
            if v > 0:
                cur_apex = max(cur_apex, z)
            elif last_v > 0:  # velocity sign flip: apex reached
                apexes.append(cur_apex)
                cur_apex = z
            else:
                cur_apex = z
            last_v = v
        assert len(apexes) >= 3, f"expected multiple bounces, got {apexes}"
        for a, b in zip(apexes, apexes[1:]):
            assert b < a + 1e-6, f"energy injected: apex {a} -> {b}"

    def test_rest_height_on_ground(self):
        """Ball (r=0.5) on ground (top at z=0) must settle at z≈0.5 with
        near-zero velocity, never sinking below z=0.25."""
        engine = make_engine(
            ((0, 0, 8), 1.0, "sphere", {"radius": 0.5}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5]}),
        )
        min_z = np.inf
        for i in range(600):  # 10 s
            state = engine.tick()
            z = float(state.rigid_pos[0][2])
            if i > 300:
                min_z = min(min_z, z)
        final = float(engine.scene.double_buffer_read.rigid_pos[0][2])
        assert min_z > 0.44, f"sank below rest pose: {min_z}"
        assert abs(final - 0.5) < 0.1, f"rest height wrong: {final}"

    def test_no_deep_penetration_at_impact(self):
        """At 12 m/s impact with dt=1/60 the one-frame penetration bound is
        v*dt = 0.2 m. Ball center must never drop below 0.5-0.25."""
        engine = make_engine(
            ((0, 0, 8), 1.0, "sphere", {"radius": 0.5, "restitution": 0.6}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5], "restitution": 0.6}),
        )
        min_z = np.inf
        for _ in range(900):
            state = engine.tick()
            min_z = min(min_z, float(state.rigid_pos[0][2]))
        assert min_z > 0.25, f"tunneled: min z = {min_z}"


class TestStaticBodies:
    def test_ground_never_moves(self):
        engine = make_engine(
            ((0, 0, 8), 1.0, "sphere", {"radius": 0.5, "restitution": 0.6}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5], "restitution": 0.6}),
        )
        for _ in range(300):
            state = engine.tick()
        ground = state.rigid_pos[1]
        np.testing.assert_allclose(ground, [0, 0, -0.5], atol=1e-6)
        assert float(state.rigid_linvel[1][2]) == 0.0

    def test_static_has_zero_inv_mass(self):
        engine = make_engine(((0, 0, -0.5), 0.0, "box", {"half_extents": [1, 1, 1]}))
        assert float(engine.scene.double_buffer_read.rigid_inv_mass[0]) == 0.0


class TestMomentum:
    def test_head_on_collision_conserves_momentum(self):
        """Two equal spheres in zero gravity: A given initial velocity toward
        resting B. Total linear momentum must be conserved exactly (impulses
        are equal and opposite)."""
        engine = make_engine(
            ((-2, 0, 0.5), 1.0, "sphere", {"radius": 0.5}),
            ((2, 0, 0.5), 1.0, "sphere", {"radius": 0.5}),
            gravity=(0.0, 0.0, 0.0),
        )
        # Inject initial velocity into A via the write buffer (test setup only)
        engine.scene.double_buffer_write.rigid_linvel[0][:] = [2.0, 0.0, 0.0]
        p_before = np.array([2.0, 0.0, 0.0]) * 1.0  # m_A * v_A

        p_exchanged = False
        for _ in range(180):
            state = engine.tick()
            gap = float(np.linalg.norm(state.rigid_pos[0] - state.rigid_pos[1]))
            if gap < 1.0:
                p_exchanged = True
                p_after = (state.rigid_linvel[0] * state.rigid_mass[0]
                           + state.rigid_linvel[1] * state.rigid_mass[1])
                np.testing.assert_allclose(p_after, p_before, atol=1e-4)
        assert p_exchanged, "bodies never met"


class TestFriction:
    def _sliding_ball(self, friction):
        return make_engine(
            ((0, 0, 0.49), 1.0, "sphere", {"radius": 0.5, "friction": friction}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5], "friction": friction}),
        )

    def test_friction_stops_sliding_ball(self):
        """Ball sliding at 5 m/s on ground with mu=0.5 must decelerate to a
        stop (friction impulse per frame is clamped to mu * normal impulse)."""
        engine = self._sliding_ball(friction=0.5)
        engine.scene.double_buffer_write.rigid_linvel[0][:] = [5.0, 0.0, 0.0]
        for _ in range(240):  # 4 s
            state = engine.tick()
        vx = float(state.rigid_linvel[0][0])
        assert abs(vx) < 0.3, f"friction failed to stop ball: vx={vx}"
        assert abs(float(state.rigid_pos[0][0])) < 8.0, "ball slid unreasonably far"

    def test_zero_friction_ball_slides_forever(self):
        """With mu=0 the tangential impulse must be exactly zero: the ball
        keeps its horizontal speed (normal impulse is vertical-only)."""
        engine = self._sliding_ball(friction=0.0)
        engine.scene.double_buffer_write.rigid_linvel[0][:] = [5.0, 0.0, 0.0]
        for _ in range(240):  # 4 s
            state = engine.tick()
        vx = float(state.rigid_linvel[0][0])
        assert vx == pytest.approx(5.0, abs=1e-4), f"mu=0 must not decay vx: {vx}"


class TestDeterminism:
    def test_identical_conditions_identical_trajectory(self):
        """Core project rule: same initial conditions -> identical evolution."""
        def run():
            engine = make_engine(
                ((0, 0, 8), 1.0, "sphere", {"radius": 0.5, "restitution": 0.6}),
                ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5], "restitution": 0.6}),
            )
            frames = []
            for _ in range(240):
                state = engine.tick()
                frames.append(state.rigid_pos.copy())
            return frames

        a, b = run(), run()
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            np.testing.assert_array_equal(fa, fb)
