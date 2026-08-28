"""Tests for the 2D physics kernel: mechanics, collision, conservation."""

from __future__ import annotations

import numpy as np
import pytest

from pymo.kernel.bodies import Material, box_body, circle_body
from pymo.kernel.collision import detect_collision
from pymo.kernel.world import World


def test_free_fall_gravity():
    """A body under gravity accelerates at g (semi-implicit Euler)."""
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    b = circle_body([0.0, 10.0], radius=0.5, mass=1.0)
    w.add(b)
    n_steps = 100
    w.step(n_steps)
    t = n_steps * w.dt
    # v = g*t; displacement = 0.5*g*t^2
    expected_vy = -9.81 * t
    assert b.vel[1] == pytest.approx(expected_vy, rel=0.02)
    expected_y = 10.0 - 0.5 * 9.81 * t**2
    assert b.pos[1] == pytest.approx(expected_y, rel=0.02)


def test_elastic_collision_momentum_and_energy():
    """Head-on elastic collision of two equal masses conserves momentum & KE.

    Equal masses, e=1: velocities swap.
    """
    mat = Material(restitution=1.0, friction=0.0)
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat)
    b = circle_body([2.0, 0.0], 0.5, mass=1.0, material=mat)
    a.vel = np.array([1.0, 0.0])
    b.vel = np.array([0.0, 0.0])
    w = World(gravity=np.array([0.0, 0.0]), dt=0.001)
    w.add(a, b)
    p0 = w.total_momentum().copy()
    ke0 = w.total_kinetic_energy()
    # Run until they collide and separate
    for _ in range(3000):
        w.step()
        if a.vel[0] < 0 and b.vel[0] > 0:  # separated after bounce
            break
    # Momentum conserved
    assert np.allclose(w.total_momentum(), p0, atol=1e-9)
    # Energy conserved (elastic)
    assert w.total_kinetic_energy() == pytest.approx(ke0, rel=0.05)


def test_inelastic_collision_momentum_conserved():
    """Perfectly inelastic collision (e=0): bodies stick, momentum conserved."""
    mat = Material(restitution=0.0, friction=0.0)
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat)
    b = circle_body([2.0, 0.0], 0.5, mass=2.0, material=mat)
    a.vel = np.array([3.0, 0.0])
    b.vel = np.array([0.0, 0.0])
    w = World(gravity=np.array([0.0, 0.0]), dt=0.001)
    w.add(a, b)
    p0 = w.total_momentum().copy()
    for _ in range(3000):
        w.step()
        if b.vel[0] > 0 and a.vel[0] <= b.vel[0] + 1e-6:
            break
    assert np.allclose(w.total_momentum(), p0, atol=1e-9)
    # Combined velocity = p_total / m_total = 3 / 3 = 1
    assert a.vel[0] == pytest.approx(1.0, abs=0.02)
    assert b.vel[0] == pytest.approx(1.0, abs=0.02)


def test_circle_polygon_detection():
    """A circle overlapping a static box is detected as a contact."""
    ground = box_body([0.0, -1.0], 5.0, 0.5, static=True)  # top at y=-0.5
    ball = circle_body([0.0, -0.1], 0.5)  # bottom at -0.6 -> overlaps top by 0.1
    contacts = detect_collision(ball, ground)
    assert len(contacts) >= 1
    assert contacts[0].penetration > 0


def test_stacking_no_tunneling():
    """Two boxes stacked: lower one stays put, upper rests on top (stable)."""
    ground = box_body([0.0, -0.5], 5.0, 0.5, static=True)  # top at y=0
    low = box_body([0.0, 0.5], 0.5, 0.5, mass=1.0)          # rests on y=0 -> center 0.5
    high = box_body([0.0, 1.5], 0.5, 0.5, mass=1.0)         # rests on low top(1.0) -> center 1.5
    w = World(gravity=np.array([0.0, -9.81]), dt=1 / 120.0)
    w.add(ground, low, high)
    w.step(600)
    # low rests on ground top at y=0 -> center near 0.5
    assert low.pos[1] == pytest.approx(0.5, abs=0.1)
    # high rests on low (top at 1.0) -> center near 1.5
    assert high.pos[1] == pytest.approx(1.5, abs=0.1)
    # neither tunnels below the ground (low bottom stays >= 0)
    assert low.pos[1] - 0.5 > -0.05
    assert high.pos[1] > low.pos[1]  # ordering preserved


def test_static_body_immovable():
    """Static bodies never move regardless of collisions."""
    ground = box_body([0.0, 0.0], 5.0, 0.5, static=True)
    ball = circle_body([0.0, 3.0], 0.5)
    w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
    w.add(ground, ball)
    w.step(200)
    assert np.allclose(ground.pos, [0.0, 0.0], atol=1e-12)
    assert ground.vel[0] == 0 and ground.vel[1] == 0
