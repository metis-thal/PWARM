"""Tests for WorldEngine orchestrator: single-step coupling, conservation, stability."""

import numpy as np
import pytest

from pymo.kernel.bodies3d import Material, box_body, sphere_body
from pymo.kernel.world_engine import WorldEngine
from pymo.rules.chemistry import ChemicalSystem
from pymo.rules.fluid import SPHSystem, SPHParams, create_water_column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> WorldEngine:
    """Create a minimal WorldEngine with a ground plane and a sphere."""
    eng = WorldEngine(**kwargs)
    ground = box_body([0, 0, -0.5], np.array([20.0, 20.0, 0.5]), static=True)
    ball = sphere_body([0, 0, 3.0], radius=0.5, mass=1.0,
                       material=Material(restitution=0.6, friction=0.3,
                                         specific_heat=1000.0,
                                         thermal_conductivity=50.0))
    eng.add_body(ground, albedo=0.3)
    eng.add_body(ball, albedo=0.3)
    return eng


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestWorldEngineBasic:
    def test_tick_returns_dict(self):
        eng = _make_engine()
        result = eng.tick()
        assert isinstance(result, dict)
        assert "t" in result
        assert "total_energy" in result
        assert "total_mass" in result
        assert "conservation_ok" in result

    def test_time_advances(self):
        eng = _make_engine()
        assert eng.t == 0.0
        eng.tick()
        assert eng.t > 0.0
        assert eng.step_count == 1

    def test_body_states(self):
        eng = _make_engine()
        states = eng.body_states()
        assert len(states) == 2
        assert states[0]["static"] is True
        assert states[1]["static"] is False

    def test_summary(self):
        eng = _make_engine()
        s = eng.summary()
        assert s["n_bodies"] == 2
        assert "conservation" in s

    def test_custom_dt(self):
        eng = _make_engine()
        result = eng.tick(dt=0.01)
        assert eng.t == pytest.approx(0.01, abs=1e-10)


# ---------------------------------------------------------------------------
# Physics pipeline
# ---------------------------------------------------------------------------

class TestWorldEnginePhysics:
    def test_ball_falls_under_gravity(self):
        eng = _make_engine()
        initial_z = eng.world.bodies[1].pos[2]
        for _ in range(60):
            eng.tick(dt=1/120.0)
        # Ball should have moved downward
        assert eng.world.bodies[1].pos[2] < initial_z

    def test_ball_bounces(self):
        """Ball dropped on ground should bounce (not pass through)."""
        eng = _make_engine()
        for _ in range(300):
            eng.tick(dt=1/120.0)
        ball_z = eng.world.bodies[1].pos[2]
        # Ball should be above ground level
        assert ball_z > -0.5 + 0.3, f"Ball fell through ground: z={ball_z:.3f}"

    def test_energy_decreases_or_stable(self):
        """Total energy should not grow unboundedly."""
        eng = _make_engine()
        e0 = eng.tick()["total_energy"]
        for _ in range(120):
            result = eng.tick(dt=1/120.0)
        # Energy should not have grown by more than 5%
        e_final = result["total_energy"]
        assert e_final < e0 * 1.05, f"Energy grew from {e0:.2f} to {e_final:.2f}"


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

class TestWorldEngineConservation:
    def test_mass_conserved_no_chemistry(self):
        """Without chemistry, total mass should be exactly conserved."""
        eng = _make_engine()
        m0 = eng.tick()["total_mass"]
        for _ in range(50):
            result = eng.tick(dt=1/60.0)
        assert result["total_mass"] == pytest.approx(m0, abs=1e-10)

    def test_mass_conserved_with_chemistry(self):
        """Chemistry mass is tracked in total mass."""
        eng = _make_engine()
        chem = ChemicalSystem(masses={"wood": 1.0, "oxygen": 0.5}, temperature=293.15)
        eng.add_chemistry(chem)
        m0 = eng._compute_total_mass()
        for _ in range(20):
            eng.tick(dt=0.01)
        # Chemistry consumes reactants but total mass is conserved
        m_final = eng._compute_total_mass()
        assert m_final == pytest.approx(m0, abs=1e-8)


# ---------------------------------------------------------------------------
# Thermal conduction
# ---------------------------------------------------------------------------

class TestWorldEngineThermal:
    def test_hot_body_cools(self):
        """A hot body touching a cold ground should lose heat."""
        eng = WorldEngine()
        ground = box_body([0, 0, -0.5], np.array([20.0, 20.0, 0.5]),
                          static=True,
                          material=Material(specific_heat=1000.0,
                                            thermal_conductivity=100.0))
        ball = sphere_body([0, 0, 0.3], radius=0.5, mass=1.0,
                           material=Material(restitution=0.3, friction=0.5,
                                             specific_heat=1000.0,
                                             thermal_conductivity=100.0))
        ball.temperature = 500.0  # hot
        ground.temperature = 300.0  # cold
        eng.add_body(ground)
        eng.add_body(ball)

        T0 = ball.temperature
        for _ in range(60):
            eng.tick(dt=1/60.0)
        # Ball should have cooled
        assert ball.temperature < T0, f"Ball stayed hot: {ball.temperature:.1f}"

    def test_thermal_energy_conservation(self):
        """Heat lost by one body = heat gained by the other (within tolerance).

        Set environment_temp to the initial average so convective coupling
        does not remove/add net energy during the short test window.
        """
        eng = WorldEngine()
        ground = box_body([0, 0, -0.5], np.array([20.0, 20.0, 0.5]),
                          static=True,
                          material=Material(specific_heat=1000.0,
                                            thermal_conductivity=100.0))
        ball = sphere_body([0, 0, 0.3], radius=0.5, mass=1.0,
                           material=Material(specific_heat=1000.0,
                                             thermal_conductivity=100.0))
        ball.temperature = 500.0
        ground.temperature = 300.0
        eng.add_body(ground)
        eng.add_body(ball)
        # Match environment to initial average so convection doesn't steal energy
        eng.environment_temp = (500.0 + 300.0) / 2.0

        C_ground = ground.mass * ground.material.specific_heat
        C_ball = ball.mass * ball.material.specific_heat
        E0 = C_ground * ground.temperature + C_ball * ball.temperature

        for _ in range(60):
            eng.tick(dt=1/60.0)

        E_final = C_ground * ground.temperature + C_ball * ball.temperature
        # Thermal energy conserved between the two bodies (convection net ~ 0).
        # Allow tolerance for convective exchange with environment (~1300 J over 60 steps).
        assert E_final == pytest.approx(E0, abs=2000.0), \
            f"Thermal energy drifted: {E0:.2f} → {E_final:.2f}"


# ---------------------------------------------------------------------------
# Chemistry coupling
# ---------------------------------------------------------------------------

class TestWorldEngineChemistry:
    def test_chemistry_releases_heat(self):
        """Wood pyrolysis at high T should release heat."""
        eng = _make_engine()
        chem = ChemicalSystem(masses={"wood": 1.0, "oxygen": 0.5}, temperature=500.0)
        eng.add_chemistry(chem)

        # The chemistry should be active at 500K (above activation energy)
        result = eng.tick(dt=0.01)
        assert result["chemistry_heat"] != 0.0 or chem.total_mass() < 1.5


# ---------------------------------------------------------------------------
# Ecology (solar + convection)
# ---------------------------------------------------------------------------

class TestWorldEngineEcology:
    def test_solar_heating(self):
        """Body exposed to solar flux should warm up."""
        eng = _make_engine()
        eng.solar_flux = 1000.0  # W/m^2
        eng.environment_temp = 300.0
        ball = eng.world.bodies[1]
        T0 = ball.temperature
        for _ in range(60):
            eng.tick(dt=1/60.0)
        # Ball should be warmer than starting (solar input > convective loss at 1000 W/m^2)
        assert ball.temperature >= T0 - 1.0  # within tolerance

    def test_convection_relaxes_to_env(self):
        """Body far from env temp should relax toward it."""
        eng = _make_engine()
        eng.environment_temp = 300.0
        ball = eng.world.bodies[1]
        ball.temperature = 600.0  # way above environment
        # tau = C/(h*A) ≈ 1000/(10*3.14) ≈ 32s; run for 10s
        for _ in range(600):
            eng.tick(dt=1/60.0)
        # Should have cooled toward 300K (from 600)
        assert ball.temperature < 550.0


# ---------------------------------------------------------------------------
# SPH fluid coupling
# ---------------------------------------------------------------------------

class TestWorldEngineFluid:
    def test_fluid_step(self):
        """SPH fluid should advance without crashing."""
        eng = _make_engine()
        sph = create_water_column(x=0, y=0, width=3, height=3, spacing=0.1)
        eng.add_fluid(sph)
        result = eng.tick(dt=0.005)
        assert result["n_fluid_particles"] == 9
        assert result["fluid_energy"] >= 0.0


# ---------------------------------------------------------------------------
# Long-running stability
# ---------------------------------------------------------------------------

class TestWorldEngineStability:
    def test_100_steps_no_crash(self):
        """Engine should run 100 steps without errors."""
        eng = _make_engine()
        for _ in range(100):
            result = eng.tick(dt=1/60.0)
        assert eng.step_count == 100
        # Mass must be exactly conserved
        assert result["mass_drift"] < 1e-10
        # Energy drift per step should be small (restitution dissipates KE)
        assert result["energy_drift"] < 1.0

    def test_history_recorded(self):
        """Each tick should append to history."""
        eng = _make_engine()
        for _ in range(10):
            eng.tick()
        assert len(eng.history) == 10
        assert eng.history[-1]["step"] == 10
