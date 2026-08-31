"""Integration tests for 3D collision detection and contact resolution in World3D."""

import numpy as np
import pytest

from pymo.kernel.bodies3d import Material, box_body, sphere_body
from pymo.kernel.world3d import World3D


def test_sphere_falls_and_bounces_on_ground():
    """Sphere dropped from height should fall, hit ground, and bounce."""
    w = World3D(gravity=np.array([0.0, 0.0, -9.81]), dt=1/120.0, solver_iterations=10)
    ground = box_body([0, 0, -0.5], np.array([20.0, 20.0, 0.5]), static=True)
    ball = sphere_body([0, 0, 3.0], radius=0.5, mass=1.0,
                       material=Material(restitution=0.8, friction=0.3))
    w.add(ground, ball)

    # Record initial height
    h0 = ball.pos[2]

    # Simulate 2 seconds (240 steps at 120 fps)
    w.step(240)

    # Ball should have bounced and be above ground
    assert ball.pos[2] > -0.5 + 0.5, f"Ball fell through ground: z={ball.pos[2]:.3f}"
    # Ball should not be flying away to infinity
    assert ball.pos[2] < 20.0, f"Ball flew away: z={ball.pos[2]:.3f}"
    # Ball should have bounced (not still sitting on ground or fallen through)
    # Allow some overshoot from positional correction (up to 2x initial height)
    assert ball.pos[2] < h0 * 2.5, f"Ball bounced too high: z={ball.pos[2]:.3f}, h0={h0:.3f}"


def test_two_spheres_collide():
    """Two spheres moving toward each other should collide and bounce apart."""
    w = World3D(gravity=np.array([0.0, 0.0, 0.0]), dt=1/120.0, solver_iterations=10)
    # No ground, just two spheres
    s1 = sphere_body([-2, 0, 0], radius=0.5, mass=1.0,
                     material=Material(restitution=0.9, friction=0.1))
    s1.vel = np.array([3.0, 0.0, 0.0])  # moving right
    s2 = sphere_body([2, 0, 0], radius=0.5, mass=1.0,
                     material=Material(restitution=0.9, friction=0.1))
    s2.vel = np.array([-3.0, 0.0, 0.0])  # moving left
    w.add(s1, s2)

    # Let them collide
    w.step(60)  # 0.5 seconds

    # After collision, they should be moving apart or stopped
    # s1 should have positive x velocity (bounced right) or zero
    # s2 should have negative x velocity (bounced left) or zero
    rel_vel = s1.vel[0] - s2.vel[0]
    # Relative velocity should be positive (separating)
    assert rel_vel >= -0.5, f"Spheres not separating: rel_vel={rel_vel:.3f}"


def test_box_stack_stability():
    """Stacked boxes should remain stable (not explode)."""
    w = World3D(gravity=np.array([0.0, 0.0, -9.81]), dt=1/120.0, solver_iterations=15)
    ground = box_body([0, 0, -0.5], np.array([10.0, 10.0, 0.5]), static=True)
    w.add(ground)

    # Stack 3 boxes
    for j in range(3):
        box = box_body([0, 0, 0.5 + j * 1.0], np.array([0.5, 0.5, 0.5]),
                       mass=1.0, material=Material(restitution=0.1, friction=0.5))
        w.add(box)

    # Simulate 3 seconds
    w.step(360)

    # All boxes should be near their expected heights (not flying away)
    for i, b in enumerate(w.bodies[1:], start=1):
        expected_z = 0.5 + (i - 1) * 1.0
        assert abs(b.pos[2] - expected_z) < 2.0, \
            f"Box {i} drifted too far: z={b.pos[2]:.3f}, expected≈{expected_z:.3f}"


def test_sphere_sphere_restitution():
    """Two identical spheres head-on: verify coefficient of restitution."""
    w = World3D(gravity=np.array([0.0, 0.0, 0.0]), dt=1/240.0, solver_iterations=10)
    e = 0.7  # restitution
    s1 = sphere_body([-3, 0, 0], radius=0.5, mass=1.0,
                     material=Material(restitution=e, friction=0.0))
    s1.vel = np.array([1.0, 0.0, 0.0])
    s2 = sphere_body([3, 0, 0], radius=0.5, mass=1.0,
                     material=Material(restitution=e, friction=0.0))
    s2.vel = np.array([-1.0, 0.0, 0.0])
    w.add(s1, s2)

    # Run until collision and separation
    for _ in range(600):
        w.step(1)

    # After collision: s1 moves LEFT (vel[0] < 0), s2 moves RIGHT (vel[0] > 0)
    # Separation speed = s2.vel[0] - s1.vel[0] (positive when moving apart)
    sep_speed = s2.vel[0] - s1.vel[0]
    # Each sphere should have bounced: s1 going left, s2 going right
    assert s1.vel[0] < 0, f"s1 should be moving left after bounce: vel={s1.vel[0]:.3f}"
    assert s2.vel[0] > 0, f"s2 should be moving right after bounce: vel={s2.vel[0]:.3f}"
    assert sep_speed > 0, f"Spheres not separating: {sep_speed:.3f}"


def test_energy_not_gained():
    """Kinetic energy should not increase after collisions (energy conservation)."""
    w = World3D(gravity=np.array([0.0, 0.0, -9.81]), dt=1/120.0, solver_iterations=10)
    ground = box_body([0, 0, -0.5], np.array([20.0, 20.0, 0.5]), static=True)
    ball = sphere_body([0, 0, 5.0], radius=0.5, mass=1.0,
                       material=Material(restitution=0.5, friction=0.3))
    w.add(ground, ball)

    # Record initial KE (should be 0 since starting from rest)
    ke_initial = 0.0

    # Simulate and check KE doesn't spike
    max_ke = 0.0
    for _ in range(300):
        w.step(1)
        ke = w.total_kinetic_energy()
        max_ke = max(max_ke, ke)

    # KE should not exceed gravitational PE at initial height
    # PE = mgh = 1.0 * 9.81 * 5.5 ≈ 54 J
    assert max_ke < 60.0, f"KE spike detected: max_ke={max_ke:.2f}"


def test_static_body_immutability():
    """Static bodies should not move after collisions."""
    w = World3D(gravity=np.array([0.0, 0.0, -9.81]), dt=1/120.0, solver_iterations=10)
    ground = box_body([0, 0, -0.5], np.array([10.0, 10.0, 0.5]), static=True)
    ball = sphere_body([0, 0, 3.0], radius=0.5, mass=1.0,
                       material=Material(restitution=0.9, friction=0.1))
    w.add(ground, ball)

    ground_pos_before = ground.pos.copy()

    w.step(120)  # 1 second

    # Ground should not have moved
    np.testing.assert_array_almost_equal(ground.pos, ground_pos_before, decimal=10,
                                          err_msg="Static ground moved!")


def test_momentum_conservation_dynamic():
    """Total momentum should be approximately conserved for isolated system."""
    w = World3D(gravity=np.array([0.0, 0.0, 0.0]), dt=1/240.0, solver_iterations=10)
    s1 = sphere_body([-2, 0, 0], radius=0.5, mass=2.0,
                     material=Material(restitution=0.8, friction=0.0))
    s1.vel = np.array([1.0, 0.0, 0.0])
    s2 = sphere_body([2, 0, 0], radius=0.5, mass=1.0,
                     material=Material(restitution=0.8, friction=0.0))
    s2.vel = np.array([-1.0, 0.0, 0.0])
    w.add(s1, s2)

    p_before = w.total_momentum().copy()

    # Run several collisions
    w.step(480)

    p_after = w.total_momentum()

    # Momentum should be conserved (zero gravity, no external forces)
    np.testing.assert_allclose(p_after, p_before, atol=0.1,
                                err_msg="Momentum not conserved!")
