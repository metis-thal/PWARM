"""Tests for the ecology rule module (rules.ecology).

``pwarm.kernel`` is stubbed by tests/conftest.py. Scope: solar geometry and
surface flux, the day/night thermal model, weather classification thresholds,
body-environment coupling (evaporation, precipitation, capacity guard), and
the ecology system loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from pwarm.kernel.bodies import Body, Material

from pwarm.rules.ecology import (
    AtmosphereParams,
    EcologicalBody,
    EnvironmentState,
    SolarParams,
    WeatherType,
    couple_body_environment,
    create_ecology_system,
    solar_flux_at_surface,
    solar_position,
    update_environment,
)


# ---------------------------------------------------------------- solar geometry
def test_solar_position_noon_equator_equinox() -> None:
    """At equatorial noon the sun is (near) overhead."""
    elev, azim = solar_position(81.0, 12.0, 0.0, 0.0, SolarParams())  # equinox
    assert elev == pytest.approx(np.pi / 2, abs=1e-6)
    assert -np.pi <= azim <= np.pi


def test_solar_position_afternoon_azimuth_negative() -> None:
    """Afternoon (hour_angle > 0) flips the azimuth sign (westward)."""
    _, azim_morning = solar_position(100.0, 6.0, 0.3, 0.0, SolarParams())
    _, azim_evening = solar_position(100.0, 18.0, 0.3, 0.0, SolarParams())
    assert azim_morning > 0.0
    assert azim_evening < 0.0


def test_solar_position_polar_latitude_is_clamped() -> None:
    """At the poles the azimuth arccos argument is clamped, not invalid."""
    elev, azim = solar_position(100.0, 12.0, np.pi / 2, 0.0, SolarParams())
    assert np.isfinite(elev) and np.isfinite(azim)
    assert -1.0 <= np.sin(elev) <= 1.0


def test_solar_flux_at_surface_night_is_zero() -> None:
    """No flux at or below the horizon."""
    params = SolarParams()
    assert solar_flux_at_surface(0.0, 0.0, params) == 0.0
    assert solar_flux_at_surface(-0.5, 0.0, params) == 0.0


def test_solar_flux_clear_sky_positive_and_limited() -> None:
    """Clear-sky flux is positive, bounded by the solar constant."""
    params = SolarParams(solar_constant=1361.0, albedo=0.3)
    noon = solar_flux_at_surface(np.pi / 2, 0.0, params)
    assert 0.0 < noon <= 1361.0
    low_sun = solar_flux_at_surface(np.radians(10.0), 0.0, params)
    assert 0.0 < low_sun < noon  # thinner air mass at zenith


def test_solar_flux_heavy_cloud_clamped_to_zero() -> None:
    """Cloud cover beyond 1 saturates the (1 - 0.75*c^1.5) term; max() clamps."""
    params = SolarParams()
    assert solar_flux_at_surface(np.pi / 2, 2.0, params) == 0.0


# ------------------------------------------------------------------ thermal model
def _state_at_night() -> EnvironmentState:
    return EnvironmentState(time=81 * 86400.0)  # doy 81, midnight


def _state_at_noon() -> EnvironmentState:
    return EnvironmentState(time=81 * 86400.0 + 12 * 3600.0)


def test_update_environment_day_heats_toward_equilibrium() -> None:
    """Day branch: temperature relaxes toward the radiative equilibrium."""
    state = _state_at_noon()
    state.temperature = 250.0
    state.cloud_cover = 0.0
    update_environment(state, dt=1.0, latitude=0.0, longitude=0.0,
                       solar_params=SolarParams(),
                       atmo_params=AtmosphereParams())
    assert state.solar_elevation > 0.0
    assert state.temperature > 250.0  # warmed by strong noon flux


def test_update_environment_night_cools() -> None:
    """Night branch: radiative loss cools a 293 K surface."""
    state = _state_at_night()
    state.temperature = 293.15
    state.cloud_cover = 0.0
    t0 = state.temperature
    update_environment(state, dt=100.0, latitude=0.0, longitude=0.0,
                       solar_params=SolarParams(),
                       atmo_params=AtmosphereParams())
    assert state.solar_elevation < 0.0
    assert state.temperature < t0


def test_update_environment_time_and_pressure() -> None:
    """Time advances; hour/day fractions and the pressure wobble update."""
    state = _state_at_noon()
    update_environment(state, dt=3600.0, latitude=0.0, longitude=0.0,
                       solar_params=SolarParams(),
                       atmo_params=AtmosphereParams())
    assert state.time == 81 * 86400.0 + 12 * 3600.0 + 3600.0
    assert state.hour_of_day == pytest.approx(13.0)
    assert state.day_fraction == pytest.approx(13.0 / 24.0)
    assert state.pressure == pytest.approx(
        101325.0 * (1 + 0.003 * np.sin(2 * np.pi * 13.0 / 24)))


@pytest.mark.parametrize(("cloud", "humidity", "temp", "expected", "precip"), [
    (0.9, 0.95, 270.0, WeatherType.SNOW, 4.5),
    (0.9, 0.95, 280.0, WeatherType.RAIN, 4.5),
    (0.7, 0.5, 293.0, WeatherType.CLOUDY, 0.0),
    (0.5, 0.5, 293.0, WeatherType.CLEAR, 0.0),  # >0.3 cloud still reports CLEAR
    (0.1, 0.5, 293.0, WeatherType.CLEAR, 0.0),
])
def test_update_environment_weather_thresholds(
        cloud: float, humidity: float, temp: float,
        expected: WeatherType, precip: float) -> None:
    """Weather classification follows the documented cloud/humidity gates."""
    state = _state_at_night()
    state.cloud_cover = cloud
    state.humidity = humidity
    state.temperature = temp
    update_environment(state, dt=1.0, latitude=0.0, longitude=0.0,
                       solar_params=SolarParams(),
                       atmo_params=AtmosphereParams())
    assert state.weather is expected
    assert state.precipitation_rate == pytest.approx(precip)


# ------------------------------------------------------- body-environment coupling
def _eco_body(mass: float = 1.0, temperature: float = 300.0,
              water: float = 0.5, max_water: float = 1.0) -> EcologicalBody:
    body = Body(pos=np.zeros(2), mass=mass,
                material=Material(specific_heat=1000.0), radius=0.5,
                temperature=temperature)
    return EcologicalBody(body=body, water_mass=water, max_water=max_water)


def test_coupling_zero_capacity_is_noop() -> None:
    """A zero-mass body is skipped entirely by the coupling."""
    eco = _eco_body(mass=0.0)
    env = EnvironmentState(temperature=350.0, solar_elevation=1.0,
                           solar_flux=800.0, humidity=0.5)
    couple_body_environment(eco, env, dt=1.0)
    assert eco.body.temperature == 300.0
    assert eco.water_mass == 0.5


def test_coupling_evaporates_water() -> None:
    """Dry air evaporates water; the latent heat cools the body."""
    eco = _eco_body(water=0.5)
    env = EnvironmentState(temperature=320.0, solar_elevation=-1.0,
                           solar_flux=0.0, humidity=0.5)
    couple_body_environment(eco, env, dt=10.0)
    assert eco.water_mass < 0.5


def test_coupling_humidity_one_stops_evaporation() -> None:
    """Saturated air (humidity == 1) evaporates nothing."""
    eco = _eco_body(water=0.5)
    env = EnvironmentState(temperature=320.0, solar_elevation=-1.0,
                           humidity=1.0)
    couple_body_environment(eco, env, dt=10.0)
    assert eco.water_mass == 0.5


def test_coupling_precipitation_fills_to_max() -> None:
    """Rain adds water, capped at max_water."""
    eco = _eco_body(water=0.0, max_water=0.2)
    env = EnvironmentState(temperature=290.0, solar_elevation=-1.0,
                           humidity=0.5, precipitation_rate=50.0)
    couple_body_environment(eco, env, dt=100.0)
    assert eco.water_mass == pytest.approx(0.2)


def test_coupling_solar_heating_warms_body() -> None:
    """Sunlight (elevation > 0) heats a cold body toward equilibrium."""
    eco = _eco_body(temperature=250.0, water=0.0)
    env = EnvironmentState(temperature=300.0, solar_elevation=1.0,
                           solar_flux=900.0, humidity=0.5)
    couple_body_environment(eco, env, dt=60.0)
    assert eco.body.temperature > 250.0


def test_coupling_temperature_floor() -> None:
    """The updated body temperature never drops below 50 K."""
    eco = _eco_body(temperature=60.0, water=0.0)
    env = EnvironmentState(temperature=50.0, solar_elevation=-1.0,
                           solar_flux=0.0, humidity=0.5)
    couple_body_environment(eco, env, dt=1000.0)
    assert eco.body.temperature >= 50.0


# ------------------------------------------------------------------ system wiring
def test_create_ecology_system_converts_degrees() -> None:
    """create_ecology_system stores latitude/longitude in radians."""
    system = create_ecology_system(latitude_deg=45.0, longitude_deg=90.0)
    assert system.latitude == pytest.approx(np.pi / 4)
    assert system.longitude == pytest.approx(np.pi / 2)


def test_ecology_system_step_updates_env_and_bodies() -> None:
    """step() advances the environment and couples every registered body."""
    system = create_ecology_system(latitude_deg=0.0)
    eco = system.add_body(Body(pos=np.zeros(2), mass=1.0, radius=0.5),
                          water_mass=0.5)
    assert isinstance(eco, EcologicalBody)
    t0 = system.environment.time
    temp0 = eco.body.temperature
    system.step(dt=60.0)
    assert system.environment.time == pytest.approx(t0 + 60.0)
    assert eco.body.temperature != temp0  # some thermal exchange occurred
