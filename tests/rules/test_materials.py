"""Tests for the advanced material rule module (rules.materials).

``pwarm.kernel`` is stubbed by tests/conftest.py. Scope: von Mises plasticity
(radial return), fatigue (Miner's rule + Paris law, both endurance-limit unit
conventions), the soft-body tetrahedral FEM helpers, soft-body dynamics with
fixed nodes, box mesh generation, and the deformable-body switch.
"""

from __future__ import annotations

import numpy as np
import pytest
from pwarm.kernel.bodies3d import Body as Body3D
from pwarm.kernel.bodies3d import BoxShape

from pwarm.rules.materials import (
    DeformableBody,
    FatigueParams,
    FatigueState,
    PlasticityParams,
    PlasticState,
    SoftBodyNode,
    SoftBodyParams,
    SoftBodyTetra,
    create_soft_body_box,
    deviatoric_part,
    plastic_correction,
    von_mises_stress,
    von_mises_yield,
)


# ------------------------------------------------------------------ plasticity
def test_deviatoric_part_removes_hydrostatic() -> None:
    """The deviatoric part has zero trace."""
    stress = np.diag([300.0, 200.0, 100.0])
    dev = deviatoric_part(stress)
    assert dev.trace() == pytest.approx(0.0, abs=1e-9)
    assert dev[0, 0] == pytest.approx(100.0)  # mean 200 removed


def test_von_mises_stress_hydrostatic_is_zero() -> None:
    """A purely hydrostatic stress produces no von Mises stress."""
    assert von_mises_stress(np.diag([100.0, 100.0, 100.0])) == pytest.approx(0.0)


def test_von_mises_stress_uniaxial() -> None:
    """Uniaxial stress sigma gives von Mises = |sigma|."""
    assert von_mises_stress(np.diag([250e6, 0.0, 0.0])) == pytest.approx(250e6,
                                                                          rel=1e-9)


def test_von_mises_yield_is_inclusive() -> None:
    """Yielding triggers at exactly the yield stress."""
    stress = np.diag([250e6, 0.0, 0.0])
    assert von_mises_yield(stress, 250e6)
    assert not von_mises_yield(stress, 250e6 + 1.0)


def test_plastic_correction_elastic_unchanged() -> None:
    """Below yield the state is returned untouched."""
    stress = np.diag([100e6, 0.0, 0.0])
    plastic = PlasticState()
    params = PlasticityParams(yield_stress=250e6)
    new_stress, new_plastic = plastic_correction(stress, plastic, params)
    assert np.array_equal(new_stress, stress)
    assert new_plastic.eps_p_eq == 0.0


def test_plastic_correction_returns_stress_to_yield_surface() -> None:
    """Plastic overstress is projected back; hardening raises the yield stress."""
    stress = np.diag([400e6, 0.0, 0.0])
    plastic = PlasticState()
    params = PlasticityParams(yield_stress=250e6, hardening_modulus=1e9)
    new_stress, new_plastic = plastic_correction(stress, plastic, params)
    assert new_plastic.eps_p_eq > 0.0
    assert new_plastic.current_yield == pytest.approx(
        250e6 + 1e9 * new_plastic.eps_p_eq)
    # corrected deviatoric stress is much smaller than the trial stress
    assert von_mises_stress(new_stress) < 400e6
    # the hydrostatic part is preserved exactly
    assert np.trace(new_stress) == pytest.approx(np.trace(stress))


def test_plastic_correction_mutates_state_in_place() -> None:
    """The same PlasticState object is updated and returned."""
    stress = np.diag([400e6, 0.0, 0.0])
    plastic = PlasticState()
    params = PlasticityParams(yield_stress=250e6)
    _, returned = plastic_correction(stress, plastic, params)
    assert returned is plastic


# --------------------------------------------------------------------- fatigue
def test_fatigue_default_endurance_filters_only_tiny_stress() -> None:
    """With default params the effective threshold is 1e-4 MPa (unit quirk)."""
    state = FatigueState()
    state.add_cycle(50e6, FatigueParams())  # 50 MPa > 1e-4 MPa -> damages
    assert state.damage > 0.0


def test_fatigue_endurance_limit_in_pascals_filters_50mpa() -> None:
    """With endurance_limit=100e6 (Pa), 50 MPa cycles do no damage."""
    params = FatigueParams(endurance_limit=100e6)
    state = FatigueState()
    state.add_cycle(50e6, params)
    assert state.damage == 0.0
    assert state.crack_size == params.initial_crack
    state.add_cycle(150e6, params)
    assert state.damage > 0.0


def test_fatigue_miners_rule_damage_magnitude() -> None:
    """One 150 MPa cycle accrues damage 1/Nf = S^m / C."""
    params = FatigueParams(sn_c=1e12, sn_m=3.0, endurance_limit=100e6)
    state = FatigueState()
    state.add_cycle(150e6, params)
    expected = 150.0**3 / 1e12  # S in MPa
    assert state.damage == pytest.approx(expected, rel=1e-9)


def test_fatigue_paris_law_grows_crack() -> None:
    """Crack growth follows da/dN = C * (delta_K)^m."""
    params = FatigueParams(paris_c=1e-12, paris_m=3.0, endurance_limit=100e6,
                           initial_crack=1e-4)
    state = FatigueState(crack_size=1e-4)
    state.add_cycle(150e6, params)
    delta_k = 150.0 * np.sqrt(np.pi * 1e-4)
    expected = 1e-12 * delta_k**3
    assert state.crack_size == pytest.approx(1e-4 + expected, rel=1e-6)


def test_fatigue_negative_stress_is_noop() -> None:
    """A negative amplitude logs nan but accrues no damage or growth."""
    params = FatigueParams(endurance_limit=100e6)
    state = FatigueState()
    with np.errstate(invalid="ignore", divide="ignore"):
        state.add_cycle(-50e6, params)
    assert state.damage == 0.0
    assert state.crack_size == params.initial_crack


def test_fatigue_is_failed_thresholds() -> None:
    """Failure is damage >= 1 or crack >= critical_crack (inclusive)."""
    params = FatigueParams(critical_crack=0.01)
    state = FatigueState(damage=1.0)
    assert state.is_failed(params) is True
    state2 = FatigueState(damage=0.5, crack_size=0.01)
    assert state2.is_failed(params) is True
    assert FatigueState(damage=0.5, crack_size=0.009).is_failed(params) is False


# ------------------------------------------------------------------- soft body
def _single_tetra_params() -> SoftBodyParams:
    return SoftBodyParams(young_modulus=1e5, poisson_ratio=0.3, density=1000.0)


def _unit_tetra_nodes() -> tuple[list[SoftBodyNode], np.ndarray]:
    """A right tetrahedron with one vertex at the origin."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    nodes = [SoftBodyNode(p, mass=0.25) for p in positions]
    return nodes, positions


def test_deformation_gradient_identity_undeformed() -> None:
    """An undeformed tetra yields F = I."""
    nodes, rest = _unit_tetra_nodes()
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    F = tetra.compute_deformation_gradient(nodes, rest)
    assert np.allclose(F, np.eye(3), atol=1e-9)


def test_deformation_gradient_stretch() -> None:
    """Uniaxial stretch of one node shows up as F = diag(s, 1, 1)."""
    nodes, rest = _unit_tetra_nodes()
    nodes[1].pos = np.array([2.0, 0.0, 0.0])
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    F = tetra.compute_deformation_gradient(nodes, rest)
    assert np.allclose(np.diag(F), [2.0, 1.0, 1.0], atol=1e-9)


def test_deformation_gradient_degenerate_returns_identity() -> None:
    """A degenerate rest tetra (det < 1e-12) falls back to F = I."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # coplanar base
    nodes = [SoftBodyNode(p, mass=0.25) for p in positions]
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=0.0)
    F = tetra.compute_deformation_gradient(nodes, positions)
    assert np.allclose(F, np.eye(3))


def test_stress_zero_at_rest_state() -> None:
    """F = I produces zero Cauchy stress."""
    nodes, rest = _unit_tetra_nodes()
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    F = tetra.compute_deformation_gradient(nodes, rest)
    stress = tetra.compute_stress(F, _single_tetra_params())
    assert np.allclose(stress, 0.0, atol=1e-6)


def test_stress_inverted_element_clamped() -> None:
    """An inverted element (J <= 0) is clamped instead of exploding."""
    nodes, rest = _unit_tetra_nodes()
    nodes[1].pos = np.array([-1.0, 0.0, 0.0])  # det(F) < 0
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    F = tetra.compute_deformation_gradient(nodes, rest)
    stress = tetra.compute_stress(F, _single_tetra_params())
    assert np.all(np.isfinite(stress))


def test_stress_incompressible_limit_divides_by_zero() -> None:
    """nu = 0.5 makes the first Lame coefficient diverge."""
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    F = np.diag([1.1, 1.0, 1.0])
    with pytest.raises(ZeroDivisionError):
        tetra.compute_stress(F, SoftBodyParams(poisson_ratio=0.5))


def test_compute_forces_shape_and_equilibrium() -> None:
    """Nodal forces are (4, 3) and vanish in the rest state."""
    nodes, rest = _unit_tetra_nodes()
    tetra = SoftBodyTetra([0, 1, 2, 3], volume=1.0 / 6.0)
    forces = tetra.compute_forces(nodes, rest, _single_tetra_params())
    assert forces.shape == (4, 3)
    assert np.allclose(forces, 0.0, atol=1e-9)


def test_soft_body_box_mesh_counts_and_mass() -> None:
    """resolution=1 gives 8 nodes / 5 tetras and conserves total mass."""
    params = SoftBodyParams(density=1000.0)
    body = create_soft_body_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1, params)
    assert len(body.nodes) == 8
    assert len(body.tetras) == 5
    total_mass = sum(n.mass for n in body.nodes)
    assert total_mass == pytest.approx(1000.0 * 8.0)  # density * volume
    # the bottom layer (j == 0, i.e. y == 0) is pinned
    assert all(n.fixed for n in body.nodes if n.pos[1] == 0.0)
    assert not any(n.fixed for n in body.nodes if n.pos[1] > 0.0)


def test_soft_body_box_zero_resolution_raises() -> None:
    """resolution=0 divides by zero when spacing the grid."""
    with pytest.raises(ZeroDivisionError):
        create_soft_body_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, SoftBodyParams())


def test_soft_body_step_fixed_nodes_do_not_move() -> None:
    """Fixed nodes never move; free nodes sag under gravity."""
    params = SoftBodyParams(density=1000.0, young_modulus=1e5)
    body = create_soft_body_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, params)
    bottom_before = [n.pos.copy() for n in body.nodes if n.fixed]
    top_before = np.mean([n.pos.copy() for n in body.nodes if not n.fixed], axis=0)
    for _ in range(30):
        body.step(dt=1 / 240)
    for node, before in zip([n for n in body.nodes if n.fixed], bottom_before):
        assert np.array_equal(node.pos, before)
    top_after = np.mean([n.pos for n in body.nodes if not n.fixed], axis=0)
    assert top_after[1] < top_before[1]  # gravity is -y


# -------------------------------------------------------------- deformable body
def test_deformable_body_switch_without_rigid_is_noop() -> None:
    """switch_to_deformable without a rigid body changes nothing."""
    deformable = DeformableBody()
    deformable.switch_to_deformable(SoftBodyParams())
    assert deformable.is_deformable is False
    assert deformable.soft_body is None


def test_deformable_body_switch_creates_soft_body() -> None:
    """A box-shaped rigid body converts into a matching soft body."""
    rigid = Body3D(pos=np.zeros(3), shape=BoxShape(np.array([1.0, 0.5, 0.25])))
    deformable = DeformableBody(rigid_body=rigid)
    deformable.switch_to_deformable(SoftBodyParams(), resolution=1)
    assert deformable.is_deformable is True
    assert deformable.soft_body is not None
    assert len(deformable.soft_body.nodes) == 8


def test_deformable_body_step_dispatch() -> None:
    """step() drives the soft body when deformable; else it is a no-op."""
    deformable = DeformableBody()
    deformable.step(dt=1 / 60)  # neither rigid nor soft: no-op

    rigid = Body3D(pos=np.zeros(3), shape=BoxShape(np.array([1.0, 1.0, 1.0])))
    deformable = DeformableBody(rigid_body=rigid)
    deformable.step(dt=1 / 60)  # rigid branch: kernel owns stepping (pass)

    deformable.switch_to_deformable(SoftBodyParams(), resolution=1)
    top_before = np.mean([n.pos for n in deformable.soft_body.nodes
                          if not n.fixed], axis=0)
    deformable.step(dt=1 / 60)
    top_after = np.mean([n.pos for n in deformable.soft_body.nodes
                         if not n.fixed], axis=0)
    assert top_after[1] < top_before[1]
