"""Tests for the unified time stepper and integrators (physics.integrator).

Scope: the three integrators, sparse/dense linear solves, the TimeStepper
pipeline (collision -> per-solver sub-steps -> state merge -> coupling ->
conservation check) including the per-solver DOF merge table and the
conservation report, and the factory helper.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pwarm.physics.core.state import GlobalQuantities, State
from pwarm.physics.integrator import (
    ImplicitEulerIntegrator,
    SymplecticEulerIntegrator,
    TimeStepper,
    TimeStepperOptions,
    VelocityVerletIntegrator,
    create_time_stepper,
)


def state_with_rigid(n: int = 1) -> State:
    state = State(t=0.0, dt=1 / 60)
    state.rigid_pos = np.zeros((n, 3))
    state.rigid_quat = np.tile([1.0, 0, 0, 0], (n, 1))
    state.rigid_linvel = np.zeros((n, 3))
    state.rigid_angvel = np.zeros((n, 3))
    state.rigid_mass = np.ones(n)
    state.rigid_inv_mass = np.ones(n)
    state.rigid_force_accum = np.zeros((n, 3))
    state.rigid_torque_accum = np.zeros((n, 3))
    state.global_quantities = GlobalQuantities()
    return state


# ----------------------------------------------------------------- integrators
def test_velocity_verlet_step() -> None:
    """Half-kick / drift: pos uses the half-step velocity."""
    pos = np.zeros(3)
    vel = np.zeros(3)
    accel = np.array([2.0, 0.0, 0.0])
    pos_new, vel_half = VelocityVerletIntegrator.step(pos, vel, accel, dt=1.0)
    assert np.allclose(vel_half, [1.0, 0.0, 0.0])  # a*dt/2
    assert np.allclose(pos_new, [1.0, 0.0, 0.0])  # v_half*dt
    assert np.allclose(VelocityVerletIntegrator.step_position(pos, vel_half, 1.0),
                       pos_new)
    assert np.allclose(VelocityVerletIntegrator.step_velocity(vel, accel, 1.0),
                       [2.0, 0.0, 0.0])


def test_symplectic_euler_step() -> None:
    """Semi-implicit Euler: velocity first, then position with new velocity."""
    pos, vel = SymplecticEulerIntegrator.step(
        np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), dt=0.5)
    assert np.allclose(vel, [0.5, 0.0, 0.0])
    assert np.allclose(pos, [0.25, 0.0, 0.0])


def test_implicit_euler_solve_dense_and_sparse() -> None:
    """solve_linear dispatches dense (numpy) and sparse (scipy) inputs."""
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    b = np.array([2.0, 8.0])
    x = ImplicitEulerIntegrator.solve_linear(A, b)
    assert np.allclose(x, [1.0, 2.0])
    A_sp = csr_matrix(A)
    assert hasattr(A_sp, "tocsc")
    x_sp = ImplicitEulerIntegrator.solve_linear(A_sp, b)
    assert np.allclose(np.asarray(x_sp), [1.0, 2.0])


# ----------------------------------------------------------------- TimeStepper
class FakeSolver:
    """Records steps and writes a marker array into its own DOF."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.scene = None
        self.calls: list[float] = []

    def step(self, state: State, dt: float, contacts: list) -> State:
        self.calls.append(dt)
        state = state.copy()
        attr = f"{self.name}_pos"
        if hasattr(state, attr):  # thermal/geology own non-pos DOFs
            setattr(state, attr, np.full((1, 3), float(len(self.calls))))
        return state

    def get_coupling_data(self, state: State):
        from pwarm.physics.solvers.base import CouplingData

        return CouplingData()

    def apply_coupling(self, state: State, coupling) -> State:
        return state


class FakeCollision:
    def detect(self, state: State) -> list:
        return []


class FakeCoupler:
    def couple(self, state: State, dt: float) -> State:
        return state


def make_scene(solver_names: list[str]):
    from types import SimpleNamespace

    scene = SimpleNamespace(
        frame=0,
        collision_system=FakeCollision(),
        coupler=FakeCoupler(),
        solvers={name: FakeSolver(name) for name in solver_names},
    )
    return scene


def test_time_stepper_substeps_and_solver_substeps() -> None:
    """dt is divided by substeps; per-solver substeps divide it further."""
    scene = make_scene(["rigid"])
    options = TimeStepperOptions(dt=1.0, substeps=2, coupling_iterations=1,
                                 rigid_substeps=2)
    stepper = TimeStepper(scene, options)  # type: ignore[arg-type]
    state = state_with_rigid()
    new_state = stepper.step(state)
    # pinned quirk: _merge_states adds options.dt per substep, so t advances
    # by dt * substeps for one frame (2.0 here instead of 1.0)
    assert new_state.t == pytest.approx(2.0)
    rigid = scene.solvers["rigid"]
    assert len(rigid.calls) == 4  # 2 substeps x 2 solver substeps
    assert all(dt == pytest.approx(0.25) for dt in rigid.calls)


def test_time_stepper_merges_all_solver_dofs() -> None:
    """The merge table copies every solver's arrays into the merged state."""
    names = ["rigid", "sph", "fem", "mpm", "pbd", "thermal", "chemistry",
             "geology"]
    scene = make_scene(names)
    options = TimeStepperOptions(dt=0.1, substeps=1, coupling_iterations=1)
    stepper = TimeStepper(scene, options)  # type: ignore[arg-type]
    state = state_with_rigid()
    new_state = stepper.step(state)
    for name in names:
        if name == "thermal" or name == "geology":
            continue  # these DOFs are not *_pos arrays
        array = getattr(new_state, f"{name}_pos", None)
        if array is not None:
            assert float(array[0, 0]) == 1.0, name
    # thermal/chemistry/geology merge branches ran without dedicated asserts
    assert new_state.t == pytest.approx(0.1)


def test_check_conservation_logs_and_reports(
        capsys: pytest.CaptureFixture[str]) -> None:
    """_check_conservation logs KE changes; the report summarises them."""
    scene = make_scene(["rigid"])
    stepper = TimeStepper(scene, TimeStepperOptions())  # type: ignore[arg-type]
    old = state_with_rigid()
    old.rigid_mass = np.array([1.0])
    old.rigid_linvel = np.array([[1.0, 0.0, 0.0]])  # KE = 0.5
    new = state_with_rigid()
    new.rigid_mass = np.array([1.0])
    new.rigid_linvel = np.array([[10.0, 0.0, 0.0]])  # KE = 50
    stepper._check_conservation(new, old)
    assert capsys.readouterr().out.count("CONSERVATION WARNING") == 1
    report = stepper.get_conservation_report()
    assert report["total_violations"] == 1
    entry = report["by_quantity"]["total_kinetic_energy"]
    assert entry["count"] == 1 and entry["max"] == pytest.approx(99.0)


def test_state_copy_shares_global_quantities() -> None:
    """Pinned quirk: State.copy shares the GlobalQuantities object.

    Because of this, the TimeStepper's live conservation check always
    compares an object against itself and never logs through step().
    """
    state = state_with_rigid()
    clone = state.copy()
    assert clone.global_quantities is state.global_quantities


def test_conservation_report_empty() -> None:
    """Without violations the report reports no data."""
    scene = make_scene([])
    stepper = TimeStepper(scene, TimeStepperOptions())  # type: ignore[arg-type]
    stepper.step(state_with_rigid())
    assert stepper.get_conservation_report() == {"status": "no_data"}


def test_create_time_stepper_factory() -> None:
    """The factory applies kwargs onto TimeStepperOptions."""
    scene = make_scene([])
    stepper = create_time_stepper(scene, dt=0.01, substeps=3)
    assert stepper.options.dt == 0.01
    assert stepper.options.substeps == 3
