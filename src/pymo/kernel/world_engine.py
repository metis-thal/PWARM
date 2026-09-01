"""WorldEngine: top-level orchestrator coupling all physical subsystems.

Runs a strict tick(dt) pipeline where each subsystem operates on shared
body objects — no duplicate state. Cross-module coupling (chemistry→heat,
thermal→chemistry, ecology→thermal, fracture→new bodies) happens through
shared state variables, never direct subsystem-to-subsystem calls.

World3D remains the底层内核; WorldEngine is the上层编排器.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pymo.kernel.bodies3d import Body, Material, sphere_body, box_body
from pymo.kernel.world3d import World3D
from pymo.rules.chemistry import ChemicalSystem, step_chemistry
from pymo.rules.fluid import SPHSystem, SPHParams
from pymo.rules.thermal import BodyThermalSystem
from pymo.geology import GeologySolver, GeologySolverConfig


# ---------------------------------------------------------------------------
# Observation data structures for the AI layer
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single observed quantity's time series."""
    name: str
    t: np.ndarray
    values: np.ndarray


@dataclass
class TimeSeriesDataset:
    """Collection of observations — compatible with ai.observer interface."""
    observations: list[Observation] = field(default_factory=list)

    def get(self, name: str) -> np.ndarray | None:
        for obs in self.observations:
            if obs.name == name:
                return obs.values
        return None

    def time(self) -> np.ndarray:
        if not self.observations:
            return np.array([])
        return self.observations[0].t

    def names(self) -> list[str]:
        return [o.name for o in self.observations]


def _body_heat_capacity(body: Body) -> float:
    """C = m * c for a Body3D."""
    return body.mass * body.material.specific_heat


def _conduction_between(a: Body, b: Body, thermal: BodyThermalSystem, dt: float) -> float:
    """Fourier heat conduction between two 3D bodies.

    Mirrors BodyThermalSystem.conduct_between but operates on Body3D objects.
    Heat flows hot→cold, energy conserved exactly.
    """
    ka = a.material.thermal_conductivity
    kb = b.material.thermal_conductivity
    if ka <= 0.0 or kb <= 0.0:
        return 0.0
    k_eff = 2.0 * ka * kb / (ka + kb)
    ca = _body_heat_capacity(a)
    cb = _body_heat_capacity(b)
    if ca <= 0 or cb <= 0:
        return 0.0
    dT = a.temperature - b.temperature
    if abs(dT) < 1e-12:
        return 0.0
    q = k_eff * thermal.contact_area * dT / thermal.contact_distance * dt
    max_q = 0.5 * min(ca, cb) * abs(dT)
    q = float(np.clip(q, -max_q, max_q))
    # Transfer
    a.heat -= q
    b.heat += q
    a.temperature -= q / ca
    b.temperature += q / cb
    return q


def _solar_heat_on_body(body: Body, solar_flux: float, albedo: float, dt: float) -> float:
    """Compute solar heating power absorbed by a body (W).

    Q = (1 - albedo) * flux * cross_section_area.
    Returns heat energy added (J) over dt.
    """
    if solar_flux <= 0:
        return 0.0
    # Approximate cross-section from shape
    from pymo.kernel.bodies3d import SphereShape, BoxShape
    if isinstance(body.shape, SphereShape):
        cross_section = np.pi * body.shape.radius**2
    elif isinstance(body.shape, BoxShape):
        hx, hy, _ = body.shape.half_extents
        cross_section = 4.0 * hx * hy  # top-down cross-section
    else:
        cross_section = 0.25  # fallback
    return (1.0 - albedo) * solar_flux * cross_section * dt


def _convective_heat(body: Body, t_env: float, h_conv: float, dt: float) -> float:
    """Convective heat transfer: Q = h * A * (T_env - T_body) * dt.

    Returns heat added to body.
    """
    from pymo.kernel.bodies3d import SphereShape, BoxShape
    if isinstance(body.shape, SphereShape):
        r = body.shape.radius
        surface_area = 4.0 * np.pi * r**2
    elif isinstance(body.shape, BoxShape):
        hx, hy, hz = body.shape.half_extents
        surface_area = 2.0 * (hx*hy + hy*hz + hx*hz) * 4.0  # full box surface
    else:
        surface_area = 1.0
    C = _body_heat_capacity(body)
    if C <= 0:
        return 0.0
    # Newton's law of cooling: analytic step
    tau = C / (h_conv * surface_area) if (h_conv * surface_area) > 0 else 1e12
    tau = max(tau, 1.0)
    T_ss = t_env
    body.temperature = T_ss + (body.temperature - T_ss) * np.exp(-dt / tau)
    return 0.0  # temperature already updated


@dataclass
class WorldEngine:
    """Top-level orchestrator that couples all physical subsystems.

    Pipeline order (order-sensitive):
      1) Force fields / external forces
      2) Verlet integration + GJK-EPA collision + impulse (via World3D._integrate)
      3) Thermal conduction (contact-based Fourier)
      4) SPH fluid step
      5) Material stress/fracture (placeholder for future)
      6) Chemistry Arrhenius ODE + heat coupling
      7) Ecology solar radiation + convection
      8) Conservation check + AI observation
      9) Return state dict

    All subsystems read/write the SAME body objects — zero duplicate state.
    """

    # Core physics kernel (底层内核, kept as-is)
    world: World3D = field(default_factory=World3D)

    # Rule subsystems
    thermal: BodyThermalSystem = field(default_factory=BodyThermalSystem)
    sph: SPHSystem | None = None
    chemistry: ChemicalSystem | None = None

    # Ecology state (simplified — no full EcologySystem dependency to avoid 2D coupling)
    solar_flux: float = 0.0        # W/m^2 at surface
    environment_temp: float = 293.15  # K (ambient)
    body_albedos: dict[int, float] = field(default_factory=dict)  # body_index → albedo

    # Geology (Phase 1: stratigraphy + thermal)
    enable_geology: bool = False
    geology_solver: GeologySolver | None = None
    geology_config: GeologySolverConfig | None = None

    # Conservation tracking
    _initial_total_mass: float = 0.0
    _initial_total_energy: float = 0.0

    # Time
    t: float = 0.0
    step_count: int = 0

    # Tick history
    history: list[dict[str, Any]] = field(default_factory=list)

    # Observation collection (for AI law discovery)
    observe: bool = False           # enable observation recording
    sample_every: int = 1           # record every N ticks
    _obs_times: list[float] = field(default_factory=list)
    _obs_series: dict[str, list[float]] = field(default_factory=dict)

    # --- Body management ---

    def __post_init__(self) -> None:
        """Initialize optional subsystems."""
        if self.enable_geology and self.geology_solver is None:
            self.geology_config = self.geology_config or GeologySolverConfig()
            self.geology_solver = GeologySolver(self.geology_config)
            self.geology_solver.initialize()

    def add_body(self, body: Body, albedo: float = 0.3) -> int:
        """Add a 3D rigid body. Returns its index."""
        self.world.add(body)
        idx = len(self.world.bodies) - 1
        self.body_albedos[idx] = albedo
        return idx

    def add_chemistry(self, chemical: ChemicalSystem) -> None:
        """Attach a chemical system (one per body or global)."""
        self.chemistry = chemical

    def add_fluid(self, sph: SPHSystem) -> None:
        """Attach an SPH fluid system."""
        self.sph = sph

    # --- Conservation bookkeeping ---

    def _compute_total_mass(self) -> float:
        """Sum of rigid body masses + SPH fluid mass + chemistry masses."""
        mass = sum(b.mass for b in self.world.bodies)
        if self.sph is not None:
            mass += self.sph.total_mass()
        if self.chemistry is not None:
            mass += self.chemistry.total_mass()
        return mass

    def _compute_total_energy(self) -> float:
        """KE + thermal energy + fluid energy."""
        ke = self.world.total_kinetic_energy()
        # Thermal energy of all bodies
        thermal_e = sum(
            _body_heat_capacity(b) * b.temperature for b in self.world.bodies
        )
        # Fluid kinetic energy
        fluid_e = self.sph.total_energy() if self.sph is not None else 0.0
        return ke + thermal_e + fluid_e

    def _snapshot_conservation(self) -> dict[str, float]:
        """Return a snapshot of conserved quantities."""
        return {
            "total_mass": self._compute_total_mass(),
            "total_energy": self._compute_total_energy(),
            "kinetic_energy": self.world.total_kinetic_energy(),
            "thermal_energy": sum(
                _body_heat_capacity(b) * b.temperature for b in self.world.bodies
            ),
            "fluid_energy": self.sph.total_energy() if self.sph is not None else 0.0,
            "fluid_mass": self.sph.total_mass() if self.sph is not None else 0.0,
        }

    # --- Main tick pipeline ---

    def tick(self, dt: float | None = None) -> dict[str, Any]:
        """Advance the world by one time step.

        Returns a dict with state and conservation data for AI observation.
        """
        if dt is None:
            dt = self.world.dt

        # Snapshot before step
        cons_before = self._snapshot_conservation()

        # ── Step 1 & 2: Force fields + Verlet integration + GJK/EPA + impulse ──
        # World3D._integrate handles: gravity → velocity → position → collision → impulse.
        # Respect substeps for numerical stability.
        for _ in range(self.world.substeps):
            h = dt / self.world.substeps
            self.world._integrate(h)
        self.world.step_count += 1
        self.world.t += dt

        # ── Step 3: Thermal conduction (contact-based) ──
        contacts = self.world._detect_collisions()
        thermal_transferred = 0.0
        if contacts:
            for c in contacts:
                thermal_transferred += _conduction_between(c.a, c.b, self.thermal, dt)

        # ── Step 4: SPH fluid ──
        fluid_energy_delta = 0.0
        if self.sph is not None:
            e_before = self.sph.total_energy()
            self.sph.step(dt)
            fluid_energy_delta = self.sph.total_energy() - e_before

        # ── Step 5: Material stress / fracture (placeholder) ──
        # Future: compute应力 from contact forces, track fatigue, fracture bodies
        # For now, thermal softening can reduce yield stress (coupling to materials)

        # ── Step 6: Chemistry Arrhenius ODE + heat coupling ──
        chemistry_heat = 0.0
        if self.chemistry is not None:
            chemistry_heat = step_chemistry(self.chemistry, dt)
            # Couple chemical heat to body temperature
            if self.world.bodies:
                # Apply to the first non-static body (or distribute)
                for b in self.world.bodies:
                    if not b.static and b.mass > 0:
                        C = _body_heat_capacity(b)
                        if C > 0:
                            b.temperature += chemistry_heat / C
                            b.heat += chemistry_heat
                        break

        # ── Step 7: Ecology — solar radiation + convection ──
        ecology_heat = 0.0
        for i, body in enumerate(self.world.bodies):
            if body.static:
                continue
            albedo = self.body_albedos.get(i, 0.3)
            # Solar heating
            q_solar = _solar_heat_on_body(body, self.solar_flux, albedo, dt)
            C = _body_heat_capacity(body)
            if C > 0:
                body.temperature += q_solar / C
            ecology_heat += q_solar
            # Convective coupling to environment
            _convective_heat(body, self.environment_temp, h_conv=10.0, dt=dt)

        # ── Step 8: Geology (stratigraphy + thermal conduction) ──
        geology_heat = 0.0
        if self.enable_geology and self.geology_solver is not None:
            # Geology solver works in years; convert dt from seconds
            dt_years = dt / 31557600.0  # 1 year = 365.25 days
            if dt_years > 0:
                self.geology_solver.step(dt_years)
                # Couple geology thermal to surface bodies
                # Surface temperature from geology grid affects ground-level bodies
                if self.geology_solver.grid is not None:
                    surface_temp = float(self.geology_solver.grid.temperature[
                        self.geology_solver.grid._idx(
                            self.geology_solver.grid.nx // 2,
                            self.geology_solver.grid.ny // 2,
                            self.geology_solver.grid.nz - 1
                        )
                    ])
                    # Apply to ground-level bodies (z near surface)
                    for body in self.world.bodies:
                        if body.pos[2] < 10.0 and not body.static:  # within 10m of surface
                            C = _body_heat_capacity(body)
                            if C > 0:
                                dT = surface_temp - body.temperature
                                if abs(dT) > 0.1:
                                    q = min(C * abs(dT), C * 10.0) * (1.0 if dT > 0 else -1.0)
                                    body.temperature += q / C
                                    body.heat += q
                                    geology_heat += q

        # ── Step 9: Conservation check ──
        cons_after = self._snapshot_conservation()
        mass_drift = abs(cons_after["total_mass"] - cons_before["total_mass"])
        energy_drift = abs(cons_after["total_energy"] - cons_before["total_energy"])

        # ── Record state ──
        self.t = self.world.t
        self.step_count = self.world.step_count

        step_data = {
            "t": self.t,
            "step": self.step_count,
            "n_bodies": len(self.world.bodies),
            "n_fluid_particles": len(self.sph.particles) if self.sph else 0,
            "kinetic_energy": cons_after["kinetic_energy"],
            "thermal_energy": cons_after["thermal_energy"],
            "fluid_energy": cons_after["fluid_energy"],
            "total_energy": cons_after["total_energy"],
            "total_mass": cons_after["total_mass"],
            "thermal_transferred": thermal_transferred,
            "chemistry_heat": chemistry_heat,
            "ecology_heat": ecology_heat,
            "geology_heat": geology_heat,
            "fluid_energy_delta": fluid_energy_delta,
            "mass_drift": mass_drift,
            "energy_drift": energy_drift,
            "conservation_ok": mass_drift < 1e-6 and energy_drift < 1e-3,
        }
        self.history.append(step_data)

        # ── Step 9: Observation collection (for AI law discovery) ──
        if self.observe and self.step_count % self.sample_every == 0:
            self._record_observation()

        return step_data

    # --- Observation collection ---

    def _record_observation(self) -> None:
        """Snapshot current state into observation buffers."""
        t = self.t
        self._obs_times.append(t)

        for i, b in enumerate(self.world.bodies):
            prefix = f"body{i}"
            self._obs_series.setdefault(f"{prefix}.pos.x", []).append(float(b.pos[0]))
            self._obs_series.setdefault(f"{prefix}.pos.y", []).append(float(b.pos[1]))
            self._obs_series.setdefault(f"{prefix}.pos.z", []).append(float(b.pos[2]))
            self._obs_series.setdefault(f"{prefix}.vel.x", []).append(float(b.vel[0]))
            self._obs_series.setdefault(f"{prefix}.vel.y", []).append(float(b.vel[1]))
            self._obs_series.setdefault(f"{prefix}.vel.z", []).append(float(b.vel[2]))
            self._obs_series.setdefault(f"{prefix}.temperature", []).append(b.temperature)
            self._obs_series.setdefault(f"{prefix}.mass", []).append(b.mass)

        # Global scalars
        cons = self._snapshot_conservation()
        self._obs_series.setdefault("total.ke", []).append(cons["kinetic_energy"])
        self._obs_series.setdefault("total.thermal_e", []).append(cons["thermal_energy"])
        self._obs_series.setdefault("total.energy", []).append(cons["total_energy"])
        self._obs_series.setdefault("total.mass", []).append(cons["total_mass"])

        # Chemistry
        if self.chemistry is not None:
            for name, mass in self.chemistry.masses.items():
                self._obs_series.setdefault(f"chem.{name}", []).append(mass)

    def collect_observations(self) -> TimeSeriesDataset:
        """Return accumulated observations as a TimeSeriesDataset for the AI layer.

        Call after running the simulation (or during, to get partial data).
        """
        t_arr = np.array(self._obs_times, dtype=float)
        obs_list = [Observation("time", t_arr, t_arr)]
        for name, vals in self._obs_series.items():
            obs_list.append(Observation(name, t_arr, np.array(vals, dtype=float)))
        return TimeSeriesDataset(observations=obs_list)

    def body_states(self) -> list[dict[str, float]]:
        """Return per-body state dict for AI observation."""
        states = []
        for i, b in enumerate(self.world.bodies):
            states.append({
                "index": i,
                "pos_x": float(b.pos[0]),
                "pos_y": float(b.pos[1]),
                "pos_z": float(b.pos[2]),
                "vel_x": float(b.vel[0]),
                "vel_y": float(b.vel[1]),
                "vel_z": float(b.vel[2]),
                "temperature": b.temperature,
                "mass": b.mass,
                "static": b.static,
            })
        return states

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current world state."""
        states = self.body_states()
        return {
            "t": self.t,
            "step": self.step_count,
            "n_bodies": len(self.world.bodies),
            "n_fluid_particles": len(self.sph.particles) if self.sph else 0,
            "bodies": states,
            "conservation": self._snapshot_conservation(),
        }
