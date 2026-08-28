"""Tests for 3D collision detection (GJK/EPA)."""

from __future__ import annotations

import numpy as np
import pytest

from pymo.kernel.bodies3d import Material, box_body, sphere_body
from pymo.kernel.collision3d import (
    detect_all_collisions,
    detect_collision,
    gjk_intersect,
)


def test_sphere_sphere_intersection():
    """Two overlapping spheres should intersect."""
    a = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0)
    b = sphere_body([1.5, 0.0, 0.0], 1.0, mass=1.0)  # centers 1.5 apart, radii 1.0
    intersecting, _simplex = gjk_intersect(a, b)
    assert intersecting is True


def test_sphere_sphere_no_intersection():
    """Two separated spheres should not intersect."""
    a = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0)
    b = sphere_body([3.0, 0.0, 0.0], 1.0, mass=1.0)  # centers 3.0 apart, radii 1.0
    intersecting, _simplex = gjk_intersect(a, b)
    assert intersecting is False


def test_sphere_box_intersection():
    """Sphere intersecting box."""
    sphere = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0)
    box = box_body([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], mass=1.0)  # box contains sphere center
    intersecting, _simplex = gjk_intersect(sphere, box)
    assert intersecting is True


def test_sphere_box_no_intersection():
    """Sphere far from box."""
    sphere = sphere_body([5.0, 0.0, 0.0], 1.0, mass=1.0)
    box = box_body([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], mass=1.0)
    intersecting, _simplex = gjk_intersect(sphere, box)
    assert intersecting is False


def test_box_box_intersection():
    """Two intersecting boxes."""
    a = box_body([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], mass=1.0)
    b = box_body([1.5, 0.0, 0.0], [1.0, 1.0, 1.0], mass=1.0)
    intersecting, _simplex = gjk_intersect(a, b)
    assert intersecting is True


def test_box_box_no_intersection():
    """Two separated boxes."""
    a = box_body([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], mass=1.0)
    b = box_body([5.0, 0.0, 0.0], [1.0, 1.0, 1.0], mass=1.0)
    intersecting, _simplex = gjk_intersect(a, b)
    assert intersecting is False


def test_detect_collision_returns_contacts():
    """detect_collision returns contact list for intersecting bodies."""
    a = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0)
    b = sphere_body([1.5, 0.0, 0.0], 1.0, mass=1.0)
    contacts = detect_collision(a, b)
    assert len(contacts) >= 1
    c = contacts[0]
    assert c.a is a
    assert c.b is b
    assert c.penetration > 0
    assert np.linalg.norm(c.normal) == pytest.approx(1.0)


def test_detect_all_collisions():
    """detect_all_collisions finds all pairwise contacts."""
    bodies = [
        sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0),
        sphere_body([1.5, 0.0, 0.0], 1.0, mass=1.0),
        sphere_body([0.0, 3.0, 0.0], 1.0, mass=1.0),  # far away
    ]
    contacts = detect_all_collisions(bodies)
    # Should find 1 contact (between first two spheres)
    assert len(contacts) == 1


def test_collision_restitution_and_friction():
    """Contact includes correct restitution and friction from materials."""
    mat = Material(restitution=0.8, friction=0.6)
    a = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0, material=mat)
    b = sphere_body([1.5, 0.0, 0.0], 1.0, mass=1.0, material=mat)
    contacts = detect_collision(a, b)
    c = contacts[0]
    assert c.restitution == pytest.approx(0.8)
    assert c.friction == pytest.approx(0.6)


def test_sphere_sphere_penetration_depth():
    """Penetration depth for overlapping spheres."""
    a = sphere_body([0.0, 0.0, 0.0], 1.0, mass=1.0)
    b = sphere_body([1.0, 0.0, 0.0], 1.0, mass=1.0)  # overlap = 1.0
    contacts = detect_collision(a, b)
    assert len(contacts) >= 1
    c = contacts[0]
    assert c.penetration == pytest.approx(1.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])