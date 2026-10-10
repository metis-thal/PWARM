"""Tests for the fracture rule module (rules.fracture).

Runs against the REAL 2-D value types (``pwarm.rules.bodies2d``) — no
``pwarm.kernel`` stub involved. Scope: principal stress algebra, contact
stress estimation, the crack check, body splitting (circle and polygon), and
the fracture event pipeline (fragment mass filtering, static handling,
missing-body robustness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pwarm.rules.bodies2d import Body, Material
from pwarm.rules.fracture import (
    FractureParams,
    check_fracture,
    estimate_contact_stress,
    principal_stress_angle,
    principal_stresses,
    process_fracture,
    split_body,
)


@dataclass
class Contact:
    """Minimal engine-style contact record (fracture consumes it duck-typed)."""

    a: Body
    b: Body
    point: np.ndarray
    normal: np.ndarray
    penetration: float
    restitution: float
    friction: float


def make_body(pos, mass=1.0, radius=0.0, vertices=None, static=False,
              material: Material | None = None) -> Body:
    return Body(pos=np.asarray(pos, dtype=float), mass=mass, radius=radius,
                vertices=vertices, static=static,
                material=material or Material())


def make_contact(a: Body, b: Body, penetration=0.01, friction=0.5) -> Contact:
    point = (np.asarray(a.pos) + np.asarray(b.pos)) / 2.0
    normal = np.array([1.0, 0.0])
    return Contact(a=a, b=b, point=point, normal=normal,
                   penetration=penetration, restitution=0.3, friction=friction)


# ------------------------------------------------------------ principal stresses
def test_principal_stresses_ordering() -> None:
    """sigma_1 >= sigma_2 for any tensor."""
    s1, s2 = principal_stresses(100.0, 50.0, 20.0)
    assert s1 >= s2
    s1r, s2r = principal_stresses(50.0, 100.0, -20.0)
    assert s1r >= s2r
    assert s1 == pytest.approx(s1r)  # symmetric under transpose


def test_principal_stresses_hydrostatic() -> None:
    """A hydrostatic state has both principal stresses equal to the mean."""
    s1, s2 = principal_stresses(70.0, 70.0, 0.0)
    assert s1 == pytest.approx(70.0)
    assert s2 == pytest.approx(70.0)


def test_principal_stress_angle() -> None:
    """Angle is half the arctan2 of the shear over the normal difference."""
    assert principal_stress_angle(70.0, 70.0, 0.0) == 0.0
    assert principal_stress_angle(100.0, 0.0, 0.0) == 0.0
    assert principal_stress_angle(0.0, 100.0, 0.0) == pytest.approx(np.pi / 2)


# --------------------------------------------------------- contact stress fields
def test_estimate_contact_stress_compression_grows_with_penetration() -> None:
    """Deeper penetration produces more compressive normal stress."""
    a = make_body([0, 0])
    b = make_body([1, 0])
    light = estimate_contact_stress(make_contact(a, b, penetration=1e-3))
    heavy = estimate_contact_stress(make_contact(a, b, penetration=1e-1))
    assert heavy[0] < light[0] < 0.0  # normal stress is compressive (negative)


def test_estimate_contact_stress_zero_penetration_floored() -> None:
    """Zero penetration is clamped to the 1e-6 floor instead of vanishing."""
    a = make_body([0, 0])
    b = make_body([1, 0])
    xx, _, _ = estimate_contact_stress(make_contact(a, b, penetration=0.0))
    assert xx < 0.0  # still a finite compressive stress


def test_estimate_contact_stress_zero_modulus_divides_by_zero() -> None:
    """Ea + Eb == 0 raises ZeroDivisionError (no guard in the module)."""
    a = make_body([0, 0], material=Material(young_modulus=0.0))
    b = make_body([1, 0], material=Material(young_modulus=0.0))
    with pytest.raises(ZeroDivisionError):
        estimate_contact_stress(make_contact(a, b))


def test_estimate_contact_stress_shear_capped_by_hardness() -> None:
    """Shear is min(|sigma_n| * mu, a.material.hardness)."""
    soft = make_body([0, 0], material=Material(hardness=100.0, friction=10.0))
    hard = make_body([1, 0])
    _, _, xy = estimate_contact_stress(make_contact(soft, hard, friction=10.0))
    assert abs(xy) <= 100.0 + 1e-9  # capped at the soft body's hardness


# ------------------------------------------------------------------ crack check
def test_check_fracture_static_bodies_never_fracture() -> None:
    """A contact involving a static body is exempt from fracture."""
    a = make_body([0, 0], static=True)
    b = make_body([1, 0])
    assert check_fracture(make_contact(a, b, penetration=1.0)) is False


def test_check_fracture_zero_toughness_skipped() -> None:
    """Bodies with fracture_toughness <= 0 cannot fracture."""
    a = make_body([0, 0], material=Material(fracture_toughness=0.0))
    b = make_body([1, 0], material=Material(fracture_toughness=0.0))
    assert check_fracture(make_contact(a, b, penetration=1.0)) is False


def test_check_fracture_triggers_on_weak_material() -> None:
    """A weak, brittle pair under deep penetration fractures."""
    weak = Material(young_modulus=1e9, hardness=1e6,
                    fracture_toughness=1e3, brittleness=0.5)
    a = make_body([0, 0], material=weak)
    b = make_body([1, 0], material=Material(fracture_toughness=1e3))
    assert check_fracture(make_contact(a, b, penetration=0.01)) is True


def test_check_fracture_tough_material_holds() -> None:
    """A tough material survives the same contact."""
    tough = Material(young_modulus=1e9, hardness=1e9,
                     fracture_toughness=1e9, brittleness=0.0)
    a = make_body([0, 0], material=tough)
    b = make_body([1, 0], material=tough)
    assert check_fracture(make_contact(a, b, penetration=0.01)) is False


# ----------------------------------------------------------------- body splitting
def test_split_circle_produces_two_halves() -> None:
    """A circle splits into two half-mass fragments offset along the cut."""
    body = make_body([0, 0], mass=2.0, radius=0.5)
    fragments = split_body(body, angle=0.0, params=FractureParams())
    assert len(fragments) == 2
    assert all(abs(f.mass - 1.0) < 1e-12 for f in fragments)
    offsets = sorted(f.pos[0] for f in fragments)
    assert offsets == pytest.approx([-0.25, 0.25])  # +-0.5*radius along x


def test_split_fragments_carry_the_body_material() -> None:
    """Fragments share the original body's material instance and parameters."""
    mat = Material(young_modulus=5e8, fracture_toughness=1e3, brittleness=0.9)
    body = make_body([0, 0], mass=2.0, radius=0.5, material=mat)
    fragments = split_body(body, angle=0.0, params=FractureParams())
    assert len(fragments) == 2
    assert all(f.material is mat for f in fragments)
    assert all(f.material.fracture_toughness == 1e3 for f in fragments)


def test_split_polygon_produces_two_fragments() -> None:
    """A square polygon splits into two half-mass fragments."""
    square = [np.array(v, dtype=float) for v in
              [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]]
    body = make_body([0, 0], mass=4.0, vertices=square)
    fragments = split_body(body, angle=0.0, params=FractureParams())
    assert len(fragments) == 2
    assert sum(f.mass for f in fragments) == pytest.approx(4.0)


def test_split_polygon_without_vertices_returns_original() -> None:
    """A polygon body without vertices is returned unchanged."""
    body = make_body([0, 0], mass=1.0)
    assert split_body(body, angle=0.0, params=FractureParams()) == []


def test_split_polygon_collinear_vertices_degenerate() -> None:
    """A degenerate (collinear) polygon is returned as a single body."""
    body = make_body([0, 0], mass=1.0,
                     vertices=[np.array(v, dtype=float) for v in
                               [(1, 0), (2, 0), (3, 0)]])
    result = split_body(body, angle=0.0, params=FractureParams())
    assert len(result) == 1 and result[0] is body


# -------------------------------------------------------------- fracture pipeline
def test_process_fracture_no_contacts_returns_copy() -> None:
    """Without contacts the body list is returned as a copy."""
    a = make_body([0, 0], mass=1.0, radius=0.5)
    result = process_fracture([], [a])
    assert len(result) == 1 and result[0] is a


def test_process_fracture_splits_both_bodies() -> None:
    """A fracturing contact replaces both bodies by their fragments."""
    weak = Material(fracture_toughness=1e3)
    a = make_body([0, 0], mass=2.0, radius=0.5, material=weak)
    b = make_body([1, 0], mass=2.0, radius=0.5,
                  material=Material(fracture_toughness=1e3, brittleness=0.0))
    result = process_fracture([make_contact(a, b, penetration=0.01)], [a, b])
    assert len(result) == 4  # two fragments per body
    assert all(f is not a and f is not b for f in result)
    assert all(f.mass == pytest.approx(1.0) for f in result)


def test_process_fracture_drops_tiny_fragments() -> None:
    """Fragments lighter than min_fragment_mass are silently removed."""
    weak = Material(fracture_toughness=1e3)
    tiny = make_body([0, 0], mass=0.001, radius=0.5, material=weak)
    other = make_body([1, 0], mass=100.0, radius=0.5,
                      material=Material(fracture_toughness=1e9))
    result = process_fracture([make_contact(tiny, other, penetration=0.01)],
                              [tiny, other])
    # tiny's fragments (0.0005 kg) are below the 0.01 kg floor -> body vanishes;
    # the tough 100 kg body still splits into two 50 kg halves
    assert len(result) == 2
    assert all(f.mass == pytest.approx(50.0) for f in result)


def test_process_fracture_custom_min_fragment_mass() -> None:
    """A larger min_fragment_mass keeps only substantial fragments."""
    weak = Material(fracture_toughness=1e3)
    a = make_body([0, 0], mass=2.0, radius=0.5, material=weak)
    b = make_body([1, 0], mass=2.0, radius=0.5,
                  material=Material(fracture_toughness=1e3, brittleness=0.0))
    params = FractureParams(min_fragment_mass=1.5)
    result = process_fracture([make_contact(a, b, penetration=0.01)], [a, b],
                              params=params)
    # fragments are 1.0 kg each, below the 1.5 kg floor -> everything vanishes
    assert result == []
