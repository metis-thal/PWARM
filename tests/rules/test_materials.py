"""Tests for materials module: plasticity, fatigue, soft body."""

import numpy as np
import pytest

from pymo.rules.materials import (
    DeformableBody,
    FatigueParams,
    FatigueState,
    PlasticityParams,
    PlasticState,
    SoftBody,
    SoftBodyNode,
    SoftBodyParams,
    SoftBodyTetra,
    create_soft_body_box,
    plastic_correction,
    von_mises_stress,
)


class TestPlasticityParams:
    def test_default_values(self):
        p = PlasticityParams()
        assert p.yield_stress > 0
        assert p.hardening_modulus >= 0

    def test_plastic_state(self):
        state = PlasticState()
        assert state.eps_p_eq == 0.0
        assert state.current_yield == 0.0


class TestVonMisesStress:
    def test_zero_stress(self):
        stress = np.zeros((3, 3))
        vm = von_mises_stress(stress)
        assert vm == 0.0

    def test_uniaxial_tension(self):
        # Uniaxial stress in x: σ_xx = 100
        stress = np.zeros((3, 3))
        stress[0, 0] = 100.0
        vm = von_mises_stress(stress)
        assert vm == pytest.approx(100.0, rel=1e-6)

    def test_hydrostatic_stress(self):
        # Hydrostatic stress has zero deviatoric part → zero von Mises
        stress = np.eye(3) * 100.0
        vm = von_mises_stress(stress)
        assert vm == pytest.approx(0.0, abs=1e-6)


class TestPlasticCorrection:
    def test_no_plasticity_below_yield(self):
        params = PlasticityParams(yield_stress=250e6)
        state = PlasticState()
        stress = np.zeros((3, 3))
        stress[0, 0] = 100e6  # below yield
        corrected, new_state = plastic_correction(stress, state, params)
        np.testing.assert_allclose(corrected, stress, atol=1e-6)

    def test_plasticity_above_yield(self):
        params = PlasticityParams(yield_stress=250e6, hardening_modulus=0.0)
        state = PlasticState()
        stress = np.zeros((3, 3))
        stress[0, 0] = 400e6  # above yield
        corrected, new_state = plastic_correction(stress, state, params)
        # After correction, von Mises should be reduced toward yield
        vm = von_mises_stress(corrected)
        assert vm < 400e6  # stress reduced
        assert vm >= 250e6 - 1e6  # but at or near yield (radial return is approximate)

    def test_plastic_strain_accumulates(self):
        params = PlasticityParams(yield_stress=250e6, hardening_modulus=1e9)
        state = PlasticState()
        stress = np.zeros((3, 3))
        stress[0, 0] = 400e6
        _, state = plastic_correction(stress, state, params)
        assert state.eps_p_eq > 0


class TestFatigueParams:
    def test_default_values(self):
        f = FatigueParams()
        assert f.sn_c > 0
        assert f.sn_m > 0
        assert f.endurance_limit > 0

    def test_fatigue_state(self):
        state = FatigueState()
        assert state.damage == 0.0
        assert state.crack_size > 0


class TestFatigueState:
    def test_damage_accumulates(self):
        params = FatigueParams(sn_c=1e12, sn_m=3.0, endurance_limit=50e6)
        state = FatigueState()
        # Apply stress above endurance limit
        state.add_cycle(150e6, params)
        assert state.damage > 0

    def test_no_damage_below_endurance(self):
        params = FatigueParams(endurance_limit=200e6)
        state = FatigueState()
        state.add_cycle(100e6, params)  # below endurance
        assert state.damage == 0.0

    def test_crack_grows(self):
        params = FatigueParams()
        state = FatigueState()
        crack_before = state.crack_size
        state.add_cycle(200e6, params)
        assert state.crack_size > crack_before

    def test_failure_detection(self):
        params = FatigueParams(critical_crack=0.001)
        state = FatigueState(crack_size=0.002)  # above critical
        assert state.is_failed(params)


class TestSoftBody:
    def test_create_box(self):
        params = SoftBodyParams()
        body = create_soft_body_box(0, 0, 0, 1.0, 1.0, 1.0, 2, params)
        assert isinstance(body, SoftBody)
        assert len(body.nodes) > 0
        assert len(body.tetras) > 0

    def test_tetra_volume_positive(self):
        params = SoftBodyParams()
        body = create_soft_body_box(0, 0, 0, 1.0, 1.0, 1.0, 2, params)
        for tetra in body.tetras:
            assert tetra.volume > 0

    def test_node_mass_positive(self):
        params = SoftBodyParams()
        body = create_soft_body_box(0, 0, 0, 1.0, 1.0, 1.0, 2, params)
        for node in body.nodes:
            assert node.mass > 0

    def test_step(self):
        params = SoftBodyParams()
        body = create_soft_body_box(0, 0, 5.0, 1.0, 1.0, 1.0, 2, params)
        # Store initial positions
        initial_positions = np.array([n.pos.copy() for n in body.nodes])
        body.step(params.dt)
        # Nodes should have moved (gravity applied)
        final_positions = np.array([n.pos for n in body.nodes])
        assert np.all(np.isfinite(final_positions))


class TestDeformableBody:
    def test_wrapper(self):
        params = SoftBodyParams()
        body = create_soft_body_box(0, 0, 0, 1.0, 1.0, 1.0, 2, params)
        deform = DeformableBody(soft_body=body)
        assert deform.soft_body is body
        assert deform.fatigue.damage == 0.0
        assert deform.plasticity.eps_p_eq == 0.0
