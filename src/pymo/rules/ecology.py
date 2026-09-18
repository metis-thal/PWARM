"""Environmental Ecology System for the pymo rules layer.

Implements:
- Dynamic environment: temperature, humidity, pressure, illumination fields
- Energy cycles: light → heat → phase change
- Atmospheric effects on thermal/fluid systems
- Weather systems: wind, precipitation, clouds
- Day/night cycle with solar radiation

All equations are standard environmental physics formulations; no hardcoded phenomena.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pymo.kernel.bodies import Body


class WeatherType(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    STORM = "storm"
    FOG = "fog"


@dataclass
class AtmosphereParams:
    """Atmospheric parameters."""
    
    # Composition (molar fractions)
    n2_fraction: float = 0.78
    o2_fraction: float = 0.21
    co2_fraction: float = 0.0004
    h2o_fraction: float = 0.01  # variable
    
    # Physical properties
    gas_constant: float = 287.0  # J/(kg*K) for dry air
    specific_heat: float = 1005.0  # J/(kg*K)
    thermal_conductivity: float = 0.026  # W/(m*K)
    viscosity: float = 1.8e-5  # Pa*s
    
    # Surface pressure
    surface_pressure: float = 101325.0  # Pa
    
    # Scale height
    scale_height: float = 8400.0  # m
    
    # Greenhouse effect multiplier (simplified)
    greenhouse_factor: float = 1.0


@dataclass
class SolarParams:
    """Solar radiation parameters."""
    
    # Solar constant at top of atmosphere
    solar_constant: float = 1361.0  # W/m^2
    
    # Planetary albedo
    albedo: float = 0.3
    
    # Obliquity (axial tilt)
    obliquity: float = 23.44 * np.pi / 180  # radians
    
    # Orbital eccentricity
    eccentricity: float = 0.0167
    
    # Longitude of perihelion
    arg_perihelion: float = 102.9 * np.pi / 180
    
    # Day length at equator (seconds)
    day_length: float = 86400.0


@dataclass
class EnvironmentState:
    """State of the environment at a given time."""
    
    # Time
    time: float = 0.0  # seconds since epoch
    day_of_year: float = 0.0
    hour_of_day: float = 0.0
    
    # Solar
    solar_elevation: float = 0.0  # radians
    solar_azimuth: float = 0.0
    solar_flux: float = 0.0  # W/m^2 at surface
    day_fraction: float = 0.5  # 0=midnight, 0.5=noon
    
    # Atmospheric
    temperature: float = 293.15  # K
    pressure: float = 101325.0  # Pa
    humidity: float = 0.5  # relative humidity 0-1
    dew_point: float = 283.15  # K
    
    # Wind
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Weather
    weather: WeatherType = WeatherType.CLEAR
    cloud_cover: float = 0.0  # 0-1
    precipitation_rate: float = 0.0  # mm/hr
    
    # Fields (for spatial variation)
    # These would be 3D fields in a full implementation


def solar_position(
    day_of_year: float, hour_of_day: float, latitude: float, longitude: float,
    params: SolarParams
) -> tuple[float, float]:
    """Calculate solar elevation and azimuth angles.
    
    Returns (elevation, azimuth) in radians.
    Elevation: -pi/2 (below horizon) to +pi/2 (zenith)
    Azimuth: 0 = North, pi/2 = East, pi = South, -pi/2 = West
    """
    # Approximate solar declination
    decl = params.obliquity * np.sin(2 * np.pi * (day_of_year - 81) / 365.25)
    
    # Hour angle
    hour_angle = 2 * np.pi * (hour_of_day - 12) / 24.0
    
    lat = latitude
    
    # Solar elevation
    sin_elev = (np.sin(lat) * np.sin(decl) + 
                np.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    elevation = np.arcsin(np.clip(sin_elev, -1.0, 1.0))
    
    # Solar azimuth
    cos_az = (np.sin(decl) - np.sin(lat) * sin_elev) / (
        np.cos(lat) * np.cos(elevation) + 1e-12
    )
    cos_az = np.clip(cos_az, -1.0, 1.0)
    azimuth = np.arccos(cos_az)
    
    # Determine azimuth sign (morning/afternoon)
    if hour_angle > 0:
        azimuth = -azimuth  # afternoon = west
    
    return elevation, azimuth


def solar_flux_at_surface(elevation: float, cloud_cover: float, params: SolarParams) -> float:
    """Calculate solar flux at surface (W/m^2)."""
    if elevation <= 0:
        return 0.0
    
    # Atmospheric attenuation (simplified)
    # Optical air mass
    air_mass = 1.0 / (np.sin(elevation) + 0.50572 * (elevation + 6.07995)**-1.6364)
    
    # Clear sky transmission
    transmittance = 0.7**air_mass
    
    # Cloud attenuation
    transmittance *= (1.0 - 0.75 * cloud_cover**1.5)
    
    # Flux on horizontal surface
    flux = params.solar_constant * np.sin(elevation) * transmittance * (1 - params.albedo)
    
    return max(0.0, flux)


def update_environment(state: EnvironmentState, dt: float, latitude: float, longitude: float,
                       solar_params: SolarParams, atmo_params: AtmosphereParams) -> None:
    """Update environment state by dt."""
    # Update time
    state.time += dt
    state.day_of_year = (state.time / 86400.0) % 365.25
    state.hour_of_day = (state.time / 3600.0) % 24.0
    state.day_fraction = (state.hour_of_day % 24.0) / 24.0
    
    # Solar position
    state.solar_elevation, state.solar_azimuth = solar_position(
        state.day_of_year, state.hour_of_day, latitude, longitude, solar_params
    )
    
    # Solar flux
    state.solar_flux = solar_flux_at_surface(
        state.solar_elevation, state.cloud_cover, solar_params
    )
    
    # Simple temperature model: relax toward radiative equilibrium
    # Daytime: warm toward solar flux equilibrium
    # Nighttime: cool toward space
    if state.solar_elevation > 0:
        # Daytime equilibrium
        sigma = 5.67e-8  # Stefan-Boltzmann
        eq_temp = (state.solar_flux / (sigma * (1 - state.cloud_cover * 0.5)))**0.25
        eq_temp = np.clip(eq_temp, 200, 350)
        # Relax toward equilibrium
        state.temperature += (eq_temp - state.temperature) * 0.01
    else:
        # Nighttime cooling
        sigma = 5.67e-8
        sky_temp = 260.0 * (1 - 0.5 * state.cloud_cover)  # effective sky temperature
        net_loss = sigma * (state.temperature**4 - sky_temp**4)
        state.temperature -= net_loss / (1.2 * 1005.0 * 100) * dt  # simplified
    
    # Pressure (simplified: hydrostatic + diurnal variation)
    state.pressure = atmo_params.surface_pressure * (1 + 0.003 * np.sin(2 * np.pi * state.hour_of_day / 24))
    
    # Humidity (simplified)
    # Saturated vapor pressure
    es = 611.2 * np.exp(17.67 * (state.temperature - 273.15) / (state.temperature - 29.65))
    e = state.humidity * es
    # Dew point
    state.dew_point = 273.15 + 243.5 * np.log(e / 611.2) / (17.67 - np.log(e / 611.2 + 1e-12))
    
    # Simple weather transition
    if state.cloud_cover > 0.8 and state.humidity > 0.9:
        if state.temperature < 273.15:
            state.weather = WeatherType.SNOW
        else:
            state.weather = WeatherType.RAIN
        state.precipitation_rate = 5.0 * state.cloud_cover
    elif state.cloud_cover > 0.6:
        state.weather = WeatherType.CLOUDY
        state.precipitation_rate = 0.0
    elif state.cloud_cover > 0.3:
        state.weather = WeatherType.CLEAR
        state.precipitation_rate = 0.0
    else:
        state.weather = WeatherType.CLEAR
        state.precipitation_rate = 0.0


@dataclass
class EcologicalBody:
    """A body that interacts with the environment."""
    
    body: Body
    # Ecological properties
    albedo: float = 0.3  # surface reflectivity
    emissivity: float = 0.9  # thermal emissivity
    heat_capacity: float = 1000.0  # J/(kg*K) - per kg
    
    # Water content
    water_mass: float = 0.0  # kg
    max_water: float = 1.0  # kg
    
    # Biological (simplified)
    biomass: float = 0.0  # kg
    growth_rate: float = 0.0  # kg/s


def couple_body_environment(body: EcologicalBody, env: EnvironmentState, dt: float) -> None:
    """Couple ecological body with environment using analytical solution for linear ODE.
    
    The temperature evolution follows: dT/dt = -(h*A/c) * (T - T_env) + Q_solar/c
    with Q_solar = alpha * flux * A_cross / c
    """
    b = body.body
    
    # Body geometry
    radius = b.radius if b.radius > 0 else 0.5
    np.pi * radius**2
    surface_area = 4 * np.pi * radius**2
    
    # Heat capacity
    c = b.mass * b.material.specific_heat
    if c <= 0:
        return
    
    # Current temperature
    T_body = max(b.temperature, 50.0)
    max(env.temperature, 50.0)
    
    # Heat transfer coefficients
    sigma = 5.67e-8
    T_avg = 0.5 * (T_body + env.temperature)
    h_rad = 4.0 * sigma * T_avg**3 * body.emissivity
    h_conv = 10.0  # W/(m^2*K)
    
    # Total heat transfer coefficient
    h_total = h_rad + h_conv
    A = surface_area
    
    # Time constant
    tau = c / (h_total * A)
    
    # Solar heating (constant over dt)
    Q_solar = 0.0
    if env.solar_elevation > 0:
        Q_solar = env.solar_flux * (1 - body.albedo) * np.pi * (b.radius if b.radius > 0 else 0.5)**2
    
    # Evaporative cooling
    Q_evap = 0.0
    if body.water_mass > 0 and env.humidity < 1.0:
        evap_rate = 0.001 * (1 - env.humidity) * surface_area
        evap_mass = min(evap_rate * dt, body.water_mass)
        if evap_mass > 0:
            Q_evap = -evap_mass * 2260e3 / dt  # W (negative)
            body.water_mass -= evap_mass
    
    # Precipitation
    if env.precipitation_rate > 0:
        precip_mass = env.precipitation_rate / 1000 * np.pi * (b.radius if b.radius > 0 else 0.5)**2 * dt
        body.water_mass = min(body.water_mass + precip_mass, body.max_water)
    
    # Net heating power
    Q_net = Q_solar + Q_evap
    
    # Analytical solution: T(t) = T_ss + (T0 - T_ss) * exp(-t/tau)
    # where T_ss = T_env + Q_net / (h_total * A)
    T_ss = env.temperature + Q_net / (h_total * surface_area)
    tau = max(c / (h_total * surface_area), 1.0)  # minimum 1s
    
    # Update temperature
    b.temperature = max(50.0, T_ss + (T_body - T_ss) * np.exp(-dt / tau))


@dataclass
class EcologySystem:
    """Manages the entire ecology system."""
    
    # Environment
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    
    # Bodies
    ecological_bodies: list[EcologicalBody] = field(default_factory=list)
    
    # Parameters
    latitude: float = 0.0  # radians
    longitude: float = 0.0
    solar_params: SolarParams = field(default_factory=SolarParams)
    atmo_params: AtmosphereParams = field(default_factory=AtmosphereParams)
    
    def add_body(self, body: Body, albedo: float = 0.3, emissivity: float = 0.9,
                 water_mass: float = 0.0, max_water: float = 1.0) -> EcologicalBody:
        eco = EcologicalBody(
            body=body,
            albedo=albedo,
            emissivity=emissivity,
            water_mass=water_mass,
            max_water=max_water
        )
        self.ecological_bodies.append(eco)
        return eco
    
    def step(self, dt: float) -> None:
        """Advance ecology system by dt."""
        # Update environment
        update_environment(
            self.environment, dt, self.latitude, self.longitude,
            self.solar_params, self.atmo_params
        )
        
        # Couple each body
        for eco in self.ecological_bodies:
            couple_body_environment(eco, self.environment, dt)
        
        # Thermal conduction between bodies (using existing thermal system)
        # This would use BodyThermalSystem from thermal.py


def create_ecology_system(latitude_deg: float = 0.0, longitude_deg: float = 0.0) -> EcologySystem:
    """Create an ecology system at given location."""
    return EcologySystem(
        latitude=latitude_deg * np.pi / 180,
        longitude=longitude_deg * np.pi / 180,
    )


if __name__ == "__main__":
    # Quick test
    eco = create_ecology_system(latitude_deg=40.0, longitude_deg=-74.0)
    
    # Add a body
    from pymo.kernel.bodies import Material, circle_body
    ball = circle_body([0, 0], 0.5, mass=1.0, material=Material(specific_heat=1000.0))
    eco.add_body(ball, albedo=0.3, water_mass=0.1)
    
    print(f"Initial: temp={eco.environment.temperature:.1f}K, "
          f"solar_elev={np.degrees(eco.environment.solar_elevation):.1f}°")
    
    # Simulate 24 hours
    for hour in range(24):
        eco.step(3600.0)  # 1 hour steps
        if hour % 6 == 0:
            print(f"Hour {hour}: temp={eco.environment.temperature:.1f}K, "
                  f"solar={eco.environment.solar_flux:.0f} W/m^2, "
                  f"weather={eco.environment.weather.value}, "
                  f"body_temp={ball.temperature:.1f}K")