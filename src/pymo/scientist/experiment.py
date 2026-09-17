"""Experiment execution — the physics-side laboratory.

``Laboratory`` is the ONLY object the AI scientist receives. It can start
experiments and hand back observation records; it never exposes the
universe's hidden parameters. The secrets are consumed here — on the physics
side — to configure a WorldEngine per experiment; the records the laboratory
returns contain measurements only.

Two execution styles:
    * batch:    ``run_experiment(spec) -> ObservationRecord``
    * stepped:  ``start_experiment(spec) -> ExperimentSession`` (live UI)

Experiment kinds:
    * ``drop``      Mission 001 free fall (records until ground contact)
    * ``drop_test`` material restitution: records fall -> contact -> first
                    bounce apex (the rebound reveals e)
    * ``slide_test`` material friction: launches a box at v0, records the
                    Coulomb deceleration until it stops
    * ``buoyancy_test`` Mission 003 Fluid Tank instrument: releases the
                    sample (built at its TRUE material density) below the
                    water surface; the buoyant force makes the descent
                    acceleration depend on density,
                    a = g (1 - rho_fluid / rho)

Apparatus: the ground is a hard anvil (friction 1.0, restitution 1.0) so the
contact pair minimum passes the MATERIAL's properties through — the measured
bounce and deceleration belong to the material, not the floor.

Core rule: the AI proposes experiments, physics executes them, the AI
observes results. The AI never configures physics directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..universes import Universe

_CONTACT_MARGIN = 0.05     # stop free-fall recording this close to contact
_SETTLE_FRAMES = 45        # "drop": keep simulating briefly after contact
_BOUNCE_TIMEOUT = 120      # "drop_test": frames after contact with no apex
_SLIDE_VX_STOP = 0.05      # "slide_test": below this the body is stopped
_MAX_STEPS = 900           # hard safety cap per experiment

# Fluid Tank instrument geometry (Mission 003) — apparatus, not secrets.
_TANK_SURFACE_Z = 8.0      # water surface height
_TANK_FLOOR_TOP_Z = 0.5    # tank floor top face
_SAMPLE_RADIUS = 0.3       # buoyancy sample sphere


@dataclass(frozen=True)
class ExperimentSpec:
    """A proposed experiment — the AI's design, executed by physics."""

    kind: str                       # "drop" | "drop_test" | "slide_test"
    drop_height: float = 10.0
    mass: float = 1.0
    radius: float = 0.5
    material: str | None = None     # key into universe secrets' materials
    v0: float | None = None         # slide_test launch speed

    @property
    def id(self) -> str:
        if self.kind == "drop_test":
            return f"drop_test_{self.material}_h{self.drop_height:g}"
        if self.kind == "slide_test":
            return f"slide_test_{self.material}_v{self.v0:g}"
        if self.kind == "buoyancy_test":
            return f"buoyancy_test_{self.material}_d{self.drop_height:g}"
        return f"drop_h{self.drop_height:g}_m{self.mass:g}"


@dataclass(frozen=True)
class ObservationRecord:
    """What the AI receives after an experiment: measurements, nothing else."""

    experiment_id: str
    t: np.ndarray
    z: np.ndarray
    vx: np.ndarray | None = None    # slide_test only
    steps: int = 0                  # world ticks consumed (budget accounting)

    @property
    def field_names(self) -> tuple[str, ...]:
        return ("t", "z") + (("vx",) if self.vx is not None else ())


class ExperimentSession:
    """One running experiment; owns its WorldEngine.

    ``engine`` is exposed read-only for visualization (the render path reads
    state snapshots — it never writes).
    """

    def __init__(self, universe: Universe, spec: ExperimentSpec):
        from pymo.physics import WorldEngine, WorldEngineConfig

        gravity = universe.secrets.gravity
        if gravity is None:
            raise ValueError(
                f"universe '{universe.name}' has no gravity secret; "
                "experiments are undefined")
        if spec.kind not in ("drop", "drop_test", "slide_test", "buoyancy_test"):
            raise ValueError(f"unsupported experiment kind: {spec.kind}")

        # Material properties are a physics-side secret use: they flow into
        # the body's contact parameters, never back to the AI as data.
        body_friction, body_restitution = 0.5, 0.0
        if spec.material is not None:
            props = universe.secrets.materials.get(spec.material)
            if props is None:
                raise ValueError(
                    f"unknown material '{spec.material}' in universe '{universe.name}'")
            body_friction = float(props.get("friction", 0.5))
            body_restitution = float(props.get("restitution", 0.0))

        dt = float(universe.rules.get("physics", {}).get("dt", 1.0 / 60.0))
        self.spec = spec
        self.universe = universe
        self._engine = WorldEngine(WorldEngineConfig(
            dt=dt, substeps=1,
            gravity=(0.0, 0.0, -gravity),
            rigid={"enabled": True},
        ))
        self.t: list[float] = []
        self.z: list[float] = []
        self.vx: list[float] = []
        self.running = True
        self._steps = 0
        self._contact_step = -1
        self._recorded_done = False
        self._bounced = False
        self._fluid_volume = 0.0
        self._fluid_density = 0.0
        self._sample_radius = float(spec.radius)

        if spec.kind == "slide_test":
            if spec.v0 is None or spec.v0 <= 0.0:
                raise ValueError("slide_test requires a positive v0")
            # Sliding box + hard-anvil ground (friction 1.0 passes the
            # material's coefficient through the pair minimum). The ground
            # is wide: the box must not slide off the edge mid-measurement
            # (v0=20 with mu=0.25 slides ~81 m).
            self._engine.create_rigid_body(
                (0, 0, 0.3), mass=spec.mass, shape="box",
                shape_params={"half_extents": [0.3, 0.3, 0.3],
                              "friction": body_friction,
                              "restitution": 0.0},
            )
            self._engine.create_rigid_body(
                (0, 0, -0.5), mass=0.0, shape="box",
                shape_params={"half_extents": [100, 100, 0.5],
                              "friction": 1.0, "restitution": 1.0},
            )
            self._engine.finalize_setup()
            # Apparatus: launch the box at v0 (AI designs it, physics applies it)
            state = self._engine.scene.double_buffer_write
            if state is not None and state.rigid_linvel is not None:
                state.rigid_linvel[0] = np.array(
                    [spec.v0, 0.0, 0.0], dtype=np.float32)
        elif spec.kind == "buoyancy_test":
            self._build_fluid_tank(body_friction, body_restitution)
        else:  # "drop" / "drop_test"
            self._engine.create_rigid_body(
                (0, 0, spec.drop_height), mass=spec.mass, shape="sphere",
                shape_params={"radius": spec.radius,
                              "friction": body_friction,
                              "restitution": body_restitution},
            )
            self._engine.create_rigid_body(
                (0, 0, -0.5), mass=0.0, shape="box",
                shape_params={"half_extents": [10, 10, 0.5],
                              "friction": 1.0, "restitution": 1.0},
            )
            self._engine.finalize_setup()

    def _build_fluid_tank(self, body_friction: float,
                          body_restitution: float) -> None:
        """Fluid Tank instrument (Mission 003): a walled tank of water.

        The sample sphere is built at the material's TRUE density
        (mass = rho * V — read from the universe's secrets on the physics
        side); this is the ONLY apparatus whose dynamics depend on density.
        The tank applies the buoyant force each tick through the solver's
        per-body force accumulator.
        """
        from .experiments.buoyancy_test import FLUID_DENSITY

        props = self.universe.secrets.materials.get(self.spec.material or "")
        if props is None or "density" not in props:
            raise ValueError(
                "buoyancy_test requires a material with a density secret: "
                f"{self.spec.material!r}")
        density = float(props["density"])
        volume = 4.0 / 3.0 * math.pi * _SAMPLE_RADIUS ** 3
        self._fluid_volume = volume
        self._fluid_density = float(FLUID_DENSITY)
        self._sample_radius = _SAMPLE_RADIUS

        # Sample FIRST (rigid index 0 — the observation and buoyancy
        # channels address the sample as body 0), at its true material
        # density, released at rest below the surface (spec.drop_height =
        # release depth below the surface).
        release_z = _TANK_SURFACE_Z - float(self.spec.drop_height)
        self._engine.create_rigid_body(
            (0, 0, release_z), mass=density * volume, shape="sphere",
            shape_params={"radius": _SAMPLE_RADIUS,
                          "friction": body_friction,
                          "restitution": body_restitution})

        # Tank fixture: floor slab + four walls (static bodies).
        self._engine.create_rigid_body(
            (0, 0, 0.4), mass=0.0, shape="box",
            shape_params={"half_extents": [2.2, 2.2, 0.1],
                          "friction": 0.5, "restitution": 0.0})
        for cx, cy, hx, hy in ((0.0, 2.0, 2.2, 0.1), (0.0, -2.0, 2.2, 0.1),
                               (2.0, 0.0, 0.1, 2.2), (-2.0, 0.0, 0.1, 2.2)):
            self._engine.create_rigid_body(
                (cx, cy, 4.7), mass=0.0, shape="box",
                shape_params={"half_extents": [hx, hy, 4.2],
                              "friction": 0.1, "restitution": 0.0})
        self._engine.finalize_setup()

    def _apply_buoyancy(self) -> None:
        """Apparatus action: F = rho_fluid * g * V (upward) on the submerged
        sample, via the solver's per-body force accumulator. The world's
        gravity stays untouched — the instrument applies a FORCE, exactly
        like a real tank."""
        write = self._engine.scene.double_buffer_write
        if write is None or write.rigid_pos is None or len(write.rigid_pos) == 0:
            return
        if write.rigid_force_accum is None:
            write.rigid_force_accum = np.zeros_like(write.rigid_pos)
        if float(write.rigid_pos[0][2]) < _TANK_SURFACE_Z:
            g = float(self.universe.secrets.gravity or 9.81)
            f_buoy = self._fluid_density * g * self._fluid_volume
            write.rigid_force_accum[0] = np.array(
                [0.0, 0.0, f_buoy], dtype=np.float32)

    @property
    def engine(self):
        """Read-only engine view for visualization (snapshots, HUD)."""
        return self._engine

    @property
    def state(self) -> tuple[float, float, float]:
        """(t, z, vz) of the primary body — ground truth for display."""
        state = self._engine.scene.double_buffer_read
        if state is None or state.rigid_pos is None or len(state.rigid_pos) == 0:
            return (0.0, float(self.spec.drop_height), 0.0)
        return (float(state.t), float(state.rigid_pos[0][2]),
                float(state.rigid_linvel[0][2]))

    @property
    def recording_done(self) -> bool:
        """True once the observation recording has finished."""
        return self._recorded_done

    def step(self) -> bool:
        """Advance one tick. Returns True while the session is still running."""
        if not self.running:
            return False
        if self.spec.kind == "buoyancy_test":
            self._apply_buoyancy()
        st = self._engine.tick()
        self._steps += 1
        t = float(st.t)
        z = float(st.rigid_pos[0][2])
        vz = float(st.rigid_linvel[0][2])
        vx = float(st.rigid_linvel[0][0])
        kind = self.spec.kind

        if kind == "slide_test":
            if vx > _SLIDE_VX_STOP:
                self.t.append(t)
                self.z.append(z)
                self.vx.append(vx)
            else:
                self._recorded_done = True
                self.running = False
                return False
        elif kind == "drop_test":
            # Record the full interaction: fall -> contact -> first bounce
            # apex (or a no-bounce timeout when restitution ~ 0). The apex
            # can only be declared AFTER a rebound was observed (vz > 0):
            # right after the threshold crossing the ball is still descending
            # into the penetration zone, and the discrete contact impulse
            # lands one frame late.
            self.t.append(t)
            self.z.append(z)
            if self._contact_step < 0:
                if z <= self.spec.radius + _CONTACT_MARGIN:
                    self._contact_step = self._steps
            elif not self._bounced:
                if vz > 0.05:
                    self._bounced = True          # rebound underway
                elif self._steps - self._contact_step >= _BOUNCE_TIMEOUT:
                    self._recorded_done = True    # no bounce: restitution ~ 0
            elif vz < -0.05:
                self._recorded_done = True        # descending past the apex
        elif kind == "buoyancy_test":
            # Record the submerged descent: a clean parabola whose curvature
            # is g*(1 - rho_fluid/rho) — the density observable.
            self.t.append(t)
            self.z.append(z)
            if z <= _TANK_FLOOR_TOP_Z + self._sample_radius:
                self._recorded_done = True    # reached the tank floor
        else:  # "drop" — Mission 001 semantics: free-fall segment only
            if not self._recorded_done:
                if z > self.spec.radius + _CONTACT_MARGIN:
                    self.t.append(t)
                    self.z.append(z)
                else:
                    self._contact_step = self._steps
                    self._recorded_done = True

        # Termination rules
        if (self._steps >= _MAX_STEPS
                or (kind in ("drop_test", "buoyancy_test")
                    and self._recorded_done)
                or (kind == "drop" and self._recorded_done
                    and self._steps - self._contact_step >= _SETTLE_FRAMES)):
            self.running = False
        return self.running

    def finish(self) -> ObservationRecord:
        """End the session and return the observation record."""
        self.running = False
        return ObservationRecord(
            experiment_id=self.spec.id,
            t=np.asarray(self.t, dtype=float),
            z=np.asarray(self.z, dtype=float),
            vx=(np.asarray(self.vx, dtype=float) if self.vx else None),
            steps=self._steps,
        )


class Laboratory:
    """Executes experiments in a universe and returns observations.

    The universe's hidden parameters are used here — on the physics side —
    and are not reachable from the records the laboratory returns.
    """

    def __init__(self, universe: Universe):
        self._universe = universe
        self.supported_experiments = ("drop", "drop_test", "slide_test",
                                      "buoyancy_test")

    @property
    def universe_name(self) -> str:
        """Public world name (safe to show the AI)."""
        return self._universe.name

    def truth_summary(self) -> str:
        """GROUND TRUTH — for humans and benchmarks only, never for the agent."""
        s = self._universe.secrets
        parts = [(f"universe={self._universe.name} gravity={s.gravity} "
                  f"air_density={s.air_density}")]
        for mat, props in s.materials.items():
            parts.append(f"{mat}={props}")
        return " ".join(parts)

    def start_experiment(self, spec: ExperimentSpec) -> ExperimentSession:
        """Stepped execution — used by the live dashboard."""
        return ExperimentSession(self._universe, spec)

    def run_experiment(self, spec: ExperimentSpec,
                       max_steps: int = _MAX_STEPS) -> ObservationRecord:
        """Batch execution — used by headless missions and benchmarks."""
        session = self.start_experiment(spec)
        steps = 0
        while session.running and steps < max_steps:
            session.step()
            steps += 1
        return session.finish()
