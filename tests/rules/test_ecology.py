"""Tests for ecology module: environment, solar, atmospheric coupling."""

import numpy as np
import pytest

from pymo.kernel.bodies import circle_body, Material
from pymo.rules.ecology import (
    AtmosphereParams,
    EcologicalBody,
    EcologySystem,
    EnvironmentState,
    SolarParams,
    WeatherType,
    couple_body_environment,
    create_ecology_system,
    solar_flux_at_surface,
    solar_position,
    update_environment,
)


class TestSolarPosition:
    def test_solar_elevation_noon(self):
        """At solar noon, elevation should be highest."""
        params = SolarParams()
        elev, azim = solar_position(
            day_of_year=172,  # summer solstice
            hour_of_day=12.0,
            latitude=np.radians(40.0),
            longitude=0.0,
            params=params
        )
        assert elev > 0  # above horizon

    def test_solar_elevation_midnight(self):
        """At midnight, elevation should be negative (below horizon)."""
        params = SolarParams()
        elev, azim = solar_position(
            day_of_year=172,
            hour_of_day=0.0,
            latitude=np.radians(40.0),
            longitude=0.0,
            params=params
        )
        assert elev < 0

    def test_solar_elevation_varies_with_season(self):
        """Summer should have higher noon elevation than winter."""
        params = SolarParams()
        elev_summer, _ = solar_position(172, 12.0, np.radians(40.0), 0.0, params)
        elev_winter, _ = solar_position(0, 12.0, np.radians(40.0), 0.0, params)
        assert elev_summer > elev_winter


class TestSolarFlux:
    def test_flux_zero_at_night(self):
        params = SolarParams()
        flux = solar_flux_at_surface(elevation=-0.1, cloud_cover=0.0, params=params)
        assert flux == 0.0

    def test_flux_positive_in_day(self):
        params = SolarParams()
        flux = solar_flux_at_surface(elevation=np.radians(45), cloud_cover=0.0, params=params)
        assert flux > 0

    def test_clouds_reduce_flux(self):
        params = SolarParams()
        flux_clear = solar_flux_at_surface(np.radians(45), 0.0, params)
        flux_cloudy = solar_flux_at_surface(np.radians(45), 0.8, params)
        assert flux_cloudy < flux_clear


class TestEnvironmentState:
    def test_default_state(self):
        env = EnvironmentState()
        assert env.temperature == 293.15
        assert env.humidity == 0.5
        assert env.weather == WeatherType.CLEAR

    def test_update_advances_time(self):
        env = EnvironmentState()
        solar = SolarParams()
        atmo = AtmosphereParams()
        update_environment(env, 3600.0, 0.0, 0.0, solar, atmo)
        assert env.time == 3600.0


class TestEcologySystem:
    def test_create_system(self):
        eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
        assert eco.latitude == pytest.approx(np.radians(40.0), rel=1e-6)
        assert eco.longitude == pytest.approx(np.radians(-74.0), rel=1e-6)

    def test_add_body(self):
        eco = create_ecology_system()
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        eco_body = eco.add_body(ball, albedo=0.3)
        assert isinstance(eco_body, EcologicalBody)
        assert eco_body.albedo == 0.3
        assert len(eco.ecological_bodies) == 1

    def test_step_updates_environment(self):
        eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
        t_before = eco.environment.time
        eco.step(3600.0)
        assert eco.environment.time > t_before

    def test_body_temperature_couples_to_env(self):
        eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        ball.temperature = 300.0  # warmer than environment
        eco.add_body(ball, albedo=0.3)

        # Step several hours
        for _ in range(12):
            eco.step(3600.0)

        # Body should have cooled toward environment
        assert ball.temperature < 300.0


class TestCoupleBodyEnvironment:
    def test_hot_body_cools(self):
        eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        ball.temperature = 350.0
        eco_body = EcologicalBody(body=ball, albedo=0.3, water_mass=0.0)

        eco.environment.temperature = 293.0
        eco.environment.solar_elevation = -0.5  # night

        couple_body_environment(eco_body, eco.environment, 3600.0)
        assert ball.temperature < 350.0

    def test_cold_body_warms_in_sun(self):
        eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        ball.temperature = 200.0
        eco_body = EcologicalBody(body=ball, albedo=0.3, water_mass=0.0)

        eco.environment.temperature = 293.0
        eco.environment.solar_elevation = np.radians(45)  # daytime
        eco.environment.solar_flux = 500.0

        couple_body_environment(eco_body, eco.environment, 3600.0)
        assert ball.temperature > 200.0

    def test_evaporation_cools_body(self):
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        ball.temperature = 300.0
        eco_body = EcologicalBody(body=ball, albedo=0.3, water_mass=0.5)

        env = EnvironmentState()
        env.temperature = 293.0
        env.humidity = 0.3  # dry air
        env.solar_elevation = -0.5

        water_before = eco_body.water_mass
        couple_body_environment(eco_body, env, 3600.0)
        assert eco_body.water_mass < water_before  # water evaporated

    def test_precipitation_adds_water(self):
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        eco_body = EcologicalBody(body=ball, albedo=0.3, water_mass=0.0, max_water=1.0)

        env = EnvironmentState()
        env.temperature = 293.0
        env.solar_elevation = -0.5
        env.precipitation_rate = 10.0  # heavy rain

        couple_body_environment(eco_body, env, 3600.0)
        assert eco_body.water_mass > 0

    def test_body_temperature_never_negative(self):
        ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
        ball.temperature = 100.0
        eco_body = EcologicalBody(body=ball, albedo=0.3, water_mass=0.0)

        env = EnvironmentState()
        env.temperature = 50.0  # very cold
        env.solar_elevation = -0.5

        # Run many steps
        for _ in range(100):
            couple_body_environment(eco_body, env, 3600.0)

        assert ball.temperature >= 50.0  # won't go below environment
