"""Tests for the fracture mechanics module (P2.2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from pymo.kernel.bodies import Body, Material, box_body, circle_body
from pymo.kernel.collision import Contact
from pymo.rules.fracture import (
    FractureParams,
    check_fracture,
    estimate_contact_stress,
    principal_stress_angle,
    principal_stresses,
    process_fracture,
    split_body,
)


def test_principal_stresses():
    """Principal stresses computed correctly for simple cases."""
    # Pure tension in x
    s1, s2 = principal_stresses(100.0, 0.0, 0.0)
    assert s1 == pytest.approx(100.0)
    assert s2 == pytest.approx(0.0)

    # Pure shear
    s1, s2 = principal_stresses(0.0, 0.0, 50.0)
    assert s1 == pytest.approx(50.0)
    assert s2 == pytest.approx(-50.0)

    # Equal biaxial
    s1, s2 = principal_stresses(100.0, 100.0, 0.0)
    assert s1 == pytest.approx(100.0)
    assert s2 == pytest.approx(100.0)


def test_principal_stress_angle():
    """Principal stress angle computed correctly."""
    # Uniaxial tension in x -> angle 0
    angle = principal_stress_angle(100.0, 0.0, 0.0)
    assert angle == pytest.approx(0.0, abs=1e-6)

    # Uniaxial tension in y -> angle pi/2
    angle = principal_stress_angle(0.0, 100.0, 0.0)
    assert angle == pytest.approx(np.pi / 2, abs=1e-6)

    # Pure shear -> angle pi/4
    angle = principal_stress_angle(0.0, 0.0, 50.0)
    assert angle == pytest.approx(np.pi / 4, abs=1e-6)


def test_estimate_contact_stress():
    """Contact stress estimation produces reasonable values."""
    mat = Material(thermal_conductivity=0.0, young_modulus=1e9, hardness=1e6)
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat)
    b = circle_body([1.0, 0.0], 0.5, mass=1.0, material=mat)
    c = Contact(a, b, point=np.array([0.5, 0.0]), normal=np.array([1.0, 0.0]),
                penetration=0.01, restitution=0.5, friction=0.3)

    sx, sy, sxy = estimate_contact_stress(c)
    # Should produce negative normal stress (compression)
    assert sx < 0  # compression in x direction
    # sy should be 0 (no stress in tangent direction for this contact)
    assert float(sy) == pytest.approx(0.0, abs=1e-6)
    # sxy should be non-zero due to friction (shear stress)
    assert float(sxy) != pytest.approx(0.0, abs=1e-6)


def test_fracture_threshold_brittle():
    """A brittle material fractures under sufficient stress."""
    mat_brittle = Material(
        thermal_conductivity=0.0,
        young_modulus=1e9,
        hardness=1e6,
        fracture_toughness=1e3,  # very low toughness for testing
        brittleness=0.9  # very brittle
    )
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat_brittle)
    b = circle_body([1.0, 0.0], 0.5, mass=1.0, material=mat_brittle)
    c = Contact(a, b, point=np.array([0.5, 0.0]), normal=np.array([1.0, 0.0]),
                penetration=0.05, restitution=0.5, friction=0.3)
    # With deep penetration and brittle material, should fracture
    assert check_fracture(c) is True


def test_no_fracture_ductile():
    """A ductile material does not fracture under same stress."""
    mat_ductile = Material(
        thermal_conductivity=0.0,
        young_modulus=1e9,
        hardness=1e6,
        fracture_toughness=1e6,
        brittleness=0.1  # ductile
    )
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat_ductile)
    b = circle_body([1.0, 0.0], 0.5, mass=1.0, material=mat_ductile)
    c = Contact(a, b, point=np.array([0.5, 0.0]), normal=np.array([1.0, 0.0]),
                penetration=0.001, restitution=0.5, friction=0.3)
    # Ductile material should not fracture at same stress
    assert check_fracture(c) is False


def test_no_fracture_static_body():
    """Static bodies do not fracture."""
    mat = Material(thermal_conductivity=0.0, young_modulus=1e9, fracture_toughness=1e6)
    a = circle_body([0.0, 0.0], 0.5, mass=1.0, material=mat)
    b = box_body([0.0, -1.0], 5.0, 0.5, static=True, material=mat)
    c = Contact(a, b, point=np.array([0.0, -0.5]), normal=np.array([0.0, 1.0]),
                penetration=0.01, restitution=0.5, friction=0.3)
    assert check_fracture(c) is False


def test_split_circle():
    """Splitting a circle produces two fragments."""
    c = circle_body([0.0, 0.0], 0.5, mass=2.0, material=Material())
    angle = 0.0
    fragments = split_body(c, angle, FractureParams())
    assert len(fragments) == 2
    assert sum(f.mass for f in fragments) == pytest.approx(2.0)
    assert all(f.vertices is not None for f in fragments)


def test_split_polygon():
    """Splitting a polygon produces two fragments."""
    # Create a square
    verts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    b = Body(pos=np.array([0.0, 0.0]), mass=2.0, vertices=verts, material=Material())
    fragments = split_body(b, 0.0, FractureParams())
    assert len(fragments) == 2
    assert sum(f.mass for f in fragments) == pytest.approx(2.0)


def test_process_fracture_removes_fractured_bodies():
    """process_fracture removes fractured bodies and adds fragments."""
    mat = Material(thermal_conductivity=0.0, young_modulus=1e9,
                   fracture_toughness=1e3, brittleness=0.9)  # very brittle
    a = circle_body([0.0, 0.0], 0.5, mass=2.0, material=mat)
    b = circle_body([1.0, 0.0], 0.5, mass=2.0, material=mat)
    c = Contact(a, b, point=np.array([0.5, 0.0]), normal=np.array([1.0, 0.0]),
                penetration=0.05, restitution=0.5, friction=0.3)

    bodies = [a, b]
    params = FractureParams(min_fragment_mass=0.1)
    new_bodies = process_fracture([c], bodies, params)

    # Both original bodies should be replaced by fragments (check by identity)
    assert not any(obj is a for obj in new_bodies)
    assert not any(obj is b for obj in new_bodies)
    assert len(new_bodies) >= 4  # at least 2 fragments per body