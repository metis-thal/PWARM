"""Tests for the 2-D rigid-body value types (rules.bodies2d).

These types were restored from the pre-deletion legacy kernel (commit
``e1a8551``) as pure data — no World, collision, solver or integrator is
involved or stubbed.  Scope: construction, geometry helpers, the documented
mass/inertia semantics (analytic circle and polygon values, static/zero-mass
fallbacks), world-space transforms, force accumulation and kinetic energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.rules.bodies2d import (
    Body,
    Material,
    box_body,
    circle_body,
    cross,
    polygon_inertia,
)

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


# ------------------------------------------------------------------ Material
def test_material_defaults_are_the_documented_ones() -> None:
    """Every default matches the documented semantics (kernel-era contract)."""
    m = Material()
    assert m.restitution == 0.3
    assert m.friction == 0.4
    assert m.density == 1.0
    assert m.specific_heat == 1000.0
    assert m.thermal_conductivity == 0.0
    assert m.young_modulus == 1e9
    assert m.poisson_ratio == 0.3
    assert m.hardness == 1e6
    assert m.fracture_toughness == 1e6
    assert m.brittleness == 0.5


def test_material_custom_params_round_trip() -> None:
    """Custom parameters are stored verbatim."""
    m = Material(restitution=0.9, friction=0.1, density=2.5, specific_heat=500.0,
                 thermal_conductivity=50.0, young_modulus=2e8, poisson_ratio=0.1,
                 hardness=1e4, fracture_toughness=1e3, brittleness=1.0)
    assert (m.restitution, m.friction, m.density) == (0.9, 0.1, 2.5)
    assert (m.specific_heat, m.thermal_conductivity) == (500.0, 50.0)
    assert (m.young_modulus, m.poisson_ratio) == (2e8, 0.1)
    assert (m.hardness, m.fracture_toughness, m.brittleness) == (1e4, 1e3, 1.0)


# ------------------------------------------------------------- construction
def test_circle_body_factory_and_analytic_inertia() -> None:
    """A circle body is a circle, not a polygon; I = 1/2 m r^2 exactly."""
    b = circle_body([3.0, 4.0], 0.5, mass=2.0)
    assert b.is_circle() and not b.is_polygon()
    assert np.allclose(b.pos, [3.0, 4.0])
    assert b.inertia == pytest.approx(0.5 * 2.0 * 0.5**2)
    assert b.inv_mass == pytest.approx(0.5)
    assert b.inv_inertia == pytest.approx(1.0 / 0.25)
    assert b.material == Material()  # default material substituted


def test_box_body_factory_builds_a_convex_quad() -> None:
    """box_body produces the four axis-aligned corners in local frame."""
    b = box_body([1.0, 2.0], 0.5, 0.25, mass=3.0)
    assert b.is_polygon() and not b.is_circle()
    assert np.allclose(b.vertices, [[-0.5, -0.25], [0.5, -0.25],
                                    [0.5, 0.25], [-0.5, 0.25]])
    assert b.mass == 3.0


# ------------------------------------------------------ mass/inertia semantics
def test_polygon_inertia_square_matches_analytic_value() -> None:
    """Solid square, side 2, mass 4: I = m(a^2 + b^2)/12 = 8/3."""
    assert polygon_inertia(SQUARE, 4.0) == pytest.approx(8.0 / 3.0)


def test_polygon_body_inertia_uses_the_polygon_formula() -> None:
    """A polygon body derives inertia from its vertices and mass."""
    b = Body(pos=np.zeros(2), mass=4.0, vertices=SQUARE)
    assert b.inertia == pytest.approx(8.0 / 3.0)


def test_degenerate_polygon_falls_back_to_unit_inertia() -> None:
    """A zero-area (collinear) polygon gets the documented unit inertia."""
    collinear = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert polygon_inertia(collinear, 1.0) == 1.0
    assert Body(mass=1.0, vertices=collinear).inertia == 1.0


def test_shapeless_body_falls_back_to_unit_inertia() -> None:
    """No radius and no vertices: point mass with unit inertia."""
    b = Body(mass=2.0)
    assert b.inertia == 1.0 and b.inv_inertia == 1.0


def test_static_body_has_infinite_mass() -> None:
    """Static bodies are immovable: zero inverse quantities, infinite I."""
    b = Body(mass=5.0, radius=1.0, static=True)
    assert b.inv_mass == 0.0
    assert b.inv_inertia == 0.0
    assert b.inertia == np.inf


def test_non_positive_mass_means_infinite_mass() -> None:
    """mass <= 0 is the documented infinite-mass case (inv_mass 0, no crash)."""
    for mass in (0.0, -5.0):
        b = Body(mass=mass, radius=1.0)
        assert b.inv_mass == 0.0


def test_recompute_inertia_after_in_place_mutation() -> None:
    """Mutating mass/shape in place requires an explicit recompute."""
    b = circle_body([0, 0], 1.0, mass=1.0)
    b.mass = 3.0
    b.radius = 2.0
    b.recompute_inertia()
    assert b.inertia == pytest.approx(0.5 * 3.0 * 4.0)


# ------------------------------------------------------------------ geometry
def test_world_vertices_rotates_and_translates() -> None:
    """Local square at pos (10, 20), angle pi/2 lands at rotated+translated."""
    b = Body(pos=np.array([10.0, 20.0]), vertices=SQUARE, angle=np.pi / 2)
    world = b.world_vertices()
    # world_vertices applies the CCW rotation R(angle): local @ R(angle).T + pos
    for local, w in zip(SQUARE, world):
        assert np.allclose(w, local @ _rot(np.pi / 2).T + [10, 20])
    # a 90-degree turn maps the local (1, -1) vertex onto (1, 1) + pos
    assert np.allclose(world[1], [11.0, 21.0])


def _rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def test_world_vertices_without_shape_is_empty() -> None:
    """A shapeless body has no world vertices (empty (0, 2) array)."""
    out = Body().world_vertices()
    assert out.shape == (0, 2)


# -------------------------------------------------------------- dynamics API
def test_kinetic_energy_is_translational_plus_rotational() -> None:
    """KE = 1/2 m |v|^2 + 1/2 I omega^2."""
    b = circle_body([0, 0], 0.5, mass=2.0)
    b.vel = np.array([3.0, 4.0])
    assert b.kinetic_energy() == pytest.approx(25.0)
    b.ang_vel = 2.0  # I = 0.25 -> rotational part 0.5
    assert b.kinetic_energy() == pytest.approx(25.5)


def test_apply_force_accumulates_and_clear_resets() -> None:
    """apply_force adds at the COM; apply_force_at adds torque; clear resets."""
    b = Body(pos=np.zeros(2))
    b.apply_force([1.0, 0.0])
    b.apply_force([0.5, 0.0])
    assert np.allclose(b.force, [1.5, 0.0])
    b.apply_force_at(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert b.torque == pytest.approx(-1.0)  # cross([0,1],[1,0]) = -1
    b.clear_forces()
    assert np.allclose(b.force, [0.0, 0.0]) and b.torque == 0.0


def test_cross_returns_scalar_z_component() -> None:
    """cross((1,0),(0,1)) = 1; cross((0,1),(1,0)) = -1."""
    assert cross(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert cross(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == -1.0
