"""Tests for the placeholder multi-physics solvers (physics.solvers.stubs)
and the solver base contract (physics.solvers.base).

The stub solvers integrate with the new Scene/State architecture; these tests
verify their documented simplified behaviour (FEM explicit step, MPM advection,
PBD constraint projection, chemistry clipping, thermal/geology pass-throughs),
the CouplingData message contract, and the Solver lifecycle helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwarm.physics.core.entity import ComponentMask, Entity, EntityID
from pwarm.physics.core.scene import Scene
from pwarm.physics.core.state import State
from pwarm.physics.solvers.base import CouplingData, Solver
from pwarm.physics.solvers.stubs import (
    ChemistryOptions,
    ChemistrySolver,
    FEMOptions,
    FEMSolver,
    GeologyOptions,
    GeologySolver,
    MPMOptions,
    MPMSolver,
    PBDOptions,
    PBDSolver,
    ThermalOptions,
    ThermalSolver,
)


def blank_state() -> State:
    return State(t=0.0, dt=1 / 60)


# ---------------------------------------------------------------- CouplingData
def test_coupling_data_defaults() -> None:
    """CouplingData() lazily initialises its containers."""
    data = CouplingData()
    assert data.forces == {} and data.velocities == {}
    assert data.temperatures == {} and data.concentrations == {}
    assert data.contacts == [] and data.aabbs is None


def test_coupling_data_add_force_accumulates() -> None:
    """add_force sums into an existing target and copies on first add."""
    data = CouplingData()
    f1 = np.array([1.0, 0.0, 0.0])
    data.add_force("rigid", f1)
    f1[0] = 99.0  # first add copies: stored value unaffected
    assert data.forces["rigid"][0] == 1.0
    data.add_force("rigid", np.array([1.0, 2.0, 0.0]))
    assert np.allclose(data.forces["rigid"], [2.0, 2.0, 0.0])


def test_coupling_data_add_velocity_copies() -> None:
    """add_velocity stores a copy both when creating and overwriting."""
    data = CouplingData()
    v = np.array([1.0, 1.0, 1.0])
    data.add_velocity("sph", v)
    v[:] = 0.0
    assert np.all(data.velocities["sph"] == 1.0)
    data.add_velocity("sph", np.array([5.0, 0.0, 0.0]))
    assert np.allclose(data.velocities["sph"], [5.0, 0.0, 0.0])


def test_solver_base_is_abstract() -> None:
    """The Solver ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Solver()  # type: ignore[abstract]


def test_solver_base_helpers() -> None:
    """get_entities needs a scene; repr reports initialization state."""

    class MinimalSolver(Solver):
        name = "minimal"

        def step(self, state, dt, contacts):  # pragma: no cover - not called
            return state

        def get_coupling_data(self, state):  # pragma: no cover
            return CouplingData()

        def apply_coupling(self, state, coupling):  # pragma: no cover
            return state

    solver = MinimalSolver()
    assert solver.get_entities(blank_state()) == []
    assert repr(solver) == "minimalSolver(initialized=False)"
    scene = Scene()
    solver.initialize(scene)
    assert solver.scene is scene and solver._initialized is True
    assert repr(solver) == "minimalSolver(initialized=True)"
    solver.reset()  # no-op on the base class


# ------------------------------------------------------------------------- FEM
def test_fem_solver_explicit_step() -> None:
    """FEM advances node positions with the contact-force acceleration."""
    solver = FEMSolver({"density": 2.0})
    scene = Scene()
    solver.initialize(scene)
    state = blank_state()
    state.fem_pos = np.zeros((2, 3))
    state.fem_vel = np.zeros((2, 3))
    state.fem_contact_forces = np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    new_state = solver.step(state, dt=0.1, contacts=[])
    # accel = force / density = [1, 0, 0] and [0, 2, 0]
    assert np.allclose(new_state.fem_vel, [[0.1, 0, 0], [0, 0.2, 0]])
    assert np.allclose(new_state.fem_pos, [[0.01, 0, 0], [0, 0.02, 0]])


def test_fem_solver_skips_empty_nodes() -> None:
    """Without FEM nodes the step returns an unchanged copy."""
    solver = FEMSolver()
    new_state = solver.step(blank_state(), dt=0.1, contacts=[])
    assert new_state is not None


def test_fem_coupling_roundtrip() -> None:
    """FEM exports node velocities and absorbs rigid forces."""
    solver = FEMSolver()
    state = blank_state()
    state.fem_pos = np.zeros((3, 3))
    state.fem_vel = np.full((3, 3), 0.5)
    data = solver.get_coupling_data(state)
    assert np.allclose(data.velocities["rigid"], 0.5)
    assert data.temperatures["thermal"].shape == (3,)

    coupling = CouplingData()
    coupling.add_force("rigid", np.full((3, 3), 1.0))
    merged = solver.apply_coupling(state, coupling)
    assert np.allclose(merged.fem_contact_forces, 1.0)
    merged2 = solver.apply_coupling(merged, coupling)
    assert np.allclose(merged2.fem_contact_forces, 2.0)  # accumulated


# ------------------------------------------------------------------------- MPM
def test_mpm_solver_advects_particles() -> None:
    """MPM moves particles by velocity * dt (simplified explicit step)."""
    solver = MPMSolver({"cell_size": 0.1})
    state = blank_state()
    state.mpm_pos = np.zeros((2, 3))
    state.mpm_vel = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    new_state = solver.step(state, dt=0.5, contacts=[])
    assert np.allclose(new_state.mpm_pos, [[0.5, 0, 0], [0, 1.0, 0]])


def test_mpm_solver_empty_and_coupling() -> None:
    """Empty particle sets are skipped; coupling exports velocities."""
    solver = MPMSolver()
    assert solver.step(blank_state(), dt=0.1, contacts=[]) is not None
    state = blank_state()
    state.mpm_pos = np.zeros((2, 3))
    state.mpm_vel = np.ones((2, 3))
    data = solver.get_coupling_data(state)
    assert np.all(data.velocities["rigid"] == 1.0)
    coupling = CouplingData()
    coupling.add_force("rigid", np.ones((2, 3)))
    assert solver.apply_coupling(state, coupling) is not None


# ------------------------------------------------------------------------- PBD
def _pbd_state() -> tuple[State, PBDSolver]:
    state = blank_state()
    state.pbd_pos = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    state.pbd_vel = np.zeros((2, 3))
    state.pbd_pred_pos = state.pbd_pos.copy()
    state.pbd_inv_mass = np.array([1.0, 1.0])
    state.pbd_distance_constraints = np.array([[0, 1]])
    state.pbd_distance_rest = np.array([1.0])
    return state, PBDSolver({"constraint_iterations": 20})


def test_pbd_distance_constraint_pulls_particles() -> None:
    """Over-stretched particles are projected toward the rest length."""
    state, solver = _pbd_state()
    new_state = solver.step(state, dt=0.01, contacts=[])
    separation = np.linalg.norm(new_state.pbd_pred_pos[1] - new_state.pbd_pred_pos[0])
    assert separation < 2.0  # pulled toward the 1.0 rest length
    assert separation >= 1.0 - 1e-6


def test_pbd_constraint_guards() -> None:
    """Out-of-range indices and zero total weight are skipped safely."""
    state, solver = _pbd_state()
    state.pbd_distance_constraints = np.array([[0, 5], [0, 1]])
    state.pbd_distance_rest = np.array([1.0, 1.0])
    state.pbd_inv_mass = np.array([0.0, 0.0])  # w_sum == 0
    before = state.pbd_pred_pos.copy()
    solver._solve_distance_constraints(state)
    assert np.array_equal(state.pbd_pred_pos, before)


def test_pbd_external_forces_and_bending_collision_stubs() -> None:
    """External forces feed the prediction; stub passes never raise."""
    state, solver = _pbd_state()
    state.pbd_external_forces = np.full((2, 3), 10.0)
    new_state = solver.step(state, dt=0.1, contacts=[object()])
    assert new_state.pbd_pred_pos is not None
    solver._solve_bending_constraints(new_state)  # placeholder
    solver._solve_collision_constraints(new_state, [object()])  # placeholder


def test_pbd_coupling_roundtrip() -> None:
    """PBD exports velocities and accumulates rigid external forces."""
    solver = PBDSolver()
    state = blank_state()
    state.pbd_pos = np.zeros((2, 3))
    state.pbd_vel = np.full((2, 3), 0.25)
    data = solver.get_coupling_data(state)
    assert np.allclose(data.velocities["rigid"], 0.25)
    coupling = CouplingData()
    coupling.add_force("rigid", np.ones((2, 3)))
    merged = solver.apply_coupling(state, coupling)
    assert np.allclose(merged.pbd_external_forces, 1.0)
    merged2 = solver.apply_coupling(merged, coupling)
    assert np.allclose(merged2.pbd_external_forces, 2.0)


# ------------------------------------------------------------------- chemistry
def test_chemistry_solver_clips_negative_concentrations() -> None:
    """A step clips species concentrations at zero."""
    solver = ChemistrySolver({"num_species": 2})
    state = blank_state()
    state.chem_conc = np.array([[1.0, -0.5], [0.2, 0.0]])
    state.chem_diffusion = np.zeros((2, 2))
    new_state = solver.step(state, dt=0.1, contacts=[])
    assert np.all(new_state.chem_conc >= 0.0)
    assert new_state.chem_conc[0, 0] == pytest.approx(1.0)


def test_chemistry_solver_empty_and_coupling() -> None:
    """Empty concentrations skip the step; coupling exports species sums."""
    solver = ChemistrySolver()
    assert solver.step(blank_state(), dt=0.1, contacts=[]) is not None
    state = blank_state()
    state.chem_conc = np.array([[1.0, 2.0]])
    data = solver.get_coupling_data(state)
    assert np.allclose(data.concentrations["thermal"], [3.0])
    assert data.concentrations["sph"].shape == (1, 2)
    coupling = CouplingData()
    coupling.add_velocity("sph", np.ones((1, 3)))
    merged = solver.apply_coupling(state, coupling)
    assert np.all(merged.chem_advection_vel == 1.0)


def test_chemistry_options_defaults() -> None:
    """ChemistryOptions fills diffusion coefficients lazily."""
    options = ChemistryOptions()
    assert options.diffusion_coeffs == [1e-5] * 4
    assert options.reaction_rates == {}


# --------------------------------------------------------------------- thermal
def test_thermal_solver_pass_through_and_coupling() -> None:
    """The thermal stub copies state and bridges temperatures to solvers."""
    solver = ThermalSolver({"conductivity": {"granite": 3.0}})
    state = blank_state()
    state.thermal_temp = np.array([300.0, 310.0])
    new_state = solver.step(state, dt=0.1, contacts=[])
    assert new_state.thermal_temp is not None
    data = solver.get_coupling_data(state)
    for target in ("rigid", "sph", "fem", "mpm"):
        assert np.allclose(data.temperatures[target], [300.0, 310.0])

    coupling = CouplingData()
    coupling.temperatures["chemistry"] = np.array([500.0, 700.0])
    coupling.temperatures["rigid"] = np.array([320.0])
    merged = solver.apply_coupling(state, coupling)
    assert merged.thermal_flux_from_chemistry == pytest.approx(600.0)
    assert merged.thermal_flux_from_rigid == pytest.approx(320.0)


def test_thermal_options_defaults() -> None:
    """ThermalOptions fills its mapping defaults lazily."""
    options = ThermalOptions()
    assert options.conductivity == {} and options.specific_heat == {}


# --------------------------------------------------------------------- geology
def test_geology_solver_pass_through_and_coupling() -> None:
    """The geology stub copies state and exports a zero velocity field."""
    solver = GeologySolver({"grid_size": (4, 4, 2)})
    state = blank_state()
    state.geo_rock_type = np.zeros((2, 2, 2), dtype=np.int32)
    new_state = solver.step(state, dt=1000.0, contacts=[])
    assert new_state.geo_rock_type is not None
    data = solver.get_coupling_data(state)
    assert np.all(data.velocities["rigid"] == 0.0)
    coupling = CouplingData()
    coupling.temperatures["thermal"] = np.array([400.0])
    assert solver.apply_coupling(state, coupling) is not None


def test_geology_options_defaults() -> None:
    """GeologyOptions carries the documented defaults."""
    options = GeologyOptions()
    assert options.grid_size == (128, 128, 64)
    assert options.cell_size == 10.0


def test_required_components_masks() -> None:
    """Each stub solver declares its owning component mask."""
    assert FEMSolver.required_components == [ComponentMask.FEM_NODE]
    assert MPMSolver.required_components == [ComponentMask.MPM_PARTICLE]
    assert PBDSolver.required_components == [ComponentMask.PBD_PARTICLE]
    assert ChemistrySolver.required_components == [ComponentMask.CHEMISTRY]
    assert ThermalSolver.required_components == [ComponentMask.THERMAL]
    assert GeologySolver.required_components == [ComponentMask.GEOLOGY]
    assert FEMOptions().implicit is True
    assert PBDOptions().constraint_iterations == 10
    assert MPMOptions().grid_size == (64, 64, 64)


def test_entity_mask_helpers_roundtrip() -> None:
    """Entity add/remove manipulate the mask; identity follows EntityID."""
    entity = Entity(id=EntityID(), mask=ComponentMask.NONE)
    assert not entity.has(ComponentMask.RIGID_BODY)
    entity.add(ComponentMask.RIGID_DYNAMIC)
    assert entity.has(ComponentMask.RIGID_BODY)
    assert entity.has(ComponentMask.COLLISION_SHAPE)
    entity.remove(ComponentMask.COLLISION_SHAPE)
    assert not entity.has(ComponentMask.COLLISION_SHAPE)
    assert hash(entity) == hash(Entity(id=entity.id))
    assert entity == Entity(id=entity.id)
    assert entity != "not an entity"
