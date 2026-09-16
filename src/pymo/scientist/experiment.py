"""Experiment execution — the physics-side laboratory.

``Laboratory`` is the ONLY object the AI scientist receives. It can start
experiments and hand back observation records; it never exposes the
universe's hidden parameters. The secrets are consumed here — on the physics
side — to configure a WorldEngine per experiment; the records the laboratory
returns contain measurements only.

Two execution styles:
    * batch:    ``run_experiment(spec) -> ObservationRecord``
    * stepped:  ``start_experiment(spec) -> ExperimentSession`` (live UI)

Core rule: the AI proposes experiments, physics executes them, the AI
observes results. The AI never configures physics directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..universes import Universe

_CONTACT_MARGIN = 0.05     # stop recording this close to ground contact
_SETTLE_FRAMES = 45        # keep simulating briefly after contact (visual)
_MAX_STEPS = 900           # hard safety cap per experiment


@dataclass(frozen=True)
class ExperimentSpec:
    """A proposed experiment — the AI's design, executed by physics."""

    kind: str                       # "drop"
    drop_height: float
    mass: float = 1.0
    radius: float = 0.5

    @property
    def id(self) -> str:
        return f"drop_h{self.drop_height:g}_m{self.mass:g}"


@dataclass(frozen=True)
class ObservationRecord:
    """What the AI receives after an experiment: measurements, nothing else."""

    experiment_id: str
    t: np.ndarray
    z: np.ndarray
    field_names: tuple[str, ...] = ("t", "z")


class ExperimentSession:
    """One running experiment; owns its WorldEngine.

    ``engine`` is exposed read-only for visualization (the render path reads
    state snapshots — it never writes). ``step()`` advances one tick and
    records an observation while the body is in free fall; after first ground
    contact recording stops (the AI's dataset stays a clean free-fall segment)
    and the session keeps simulating briefly so viewers see the impact.
    """

    def __init__(self, universe: Universe, spec: ExperimentSpec):
        from pymo.physics import WorldEngine, WorldEngineConfig

        gravity = universe.secrets.gravity
        if gravity is None:
            raise ValueError(
                f"universe '{universe.name}' has no gravity secret; "
                "drop experiment is undefined")
        if spec.kind != "drop":
            raise ValueError(f"unsupported experiment kind: {spec.kind}")
        if spec.drop_height <= spec.radius + _CONTACT_MARGIN:
            raise ValueError("drop_height must start above contact margin")

        dt = float(universe.rules.get("physics", {}).get("dt", 1.0 / 60.0))
        self.spec = spec
        self._engine = WorldEngine(WorldEngineConfig(
            dt=dt, substeps=1,
            gravity=(0.0, 0.0, -gravity),
            rigid={"enabled": True},
        ))
        self._engine.create_rigid_body(
            (0, 0, spec.drop_height), mass=spec.mass,
            shape="sphere", shape_params={"radius": spec.radius},
        )
        self._engine.create_rigid_body(
            (0, 0, -0.5), mass=0.0,
            shape="box", shape_params={"half_extents": [10, 10, 0.5]},
        )
        self._engine.finalize_setup()

        self.t: list[float] = []
        self.z: list[float] = []
        self.running = True
        self._steps = 0
        self._recorded_done = False
        self._contact_step = -1

    @property
    def engine(self):
        """Read-only engine view for visualization (snapshots, HUD)."""
        return self._engine

    @property
    def state(self) -> tuple[float, float, float]:
        """(t, z, vz) of the falling body — ground truth for display."""
        state = self._engine.scene.double_buffer_read
        if state is None or state.rigid_pos is None or len(state.rigid_pos) == 0:
            return (0.0, float(self.spec.drop_height), 0.0)
        return (float(state.t), float(state.rigid_pos[0][2]),
                float(state.rigid_linvel[0][2]))

    @property
    def recording_done(self) -> bool:
        """True once ground contact stopped the observation recording."""
        return self._recorded_done

    def step(self) -> bool:
        """Advance one tick. Returns True while the session is still running."""
        if not self.running:
            return False
        st = self._engine.tick()
        self._steps += 1

        if not self._recorded_done:
            z = float(st.rigid_pos[0][2])
            if z > self.spec.radius + _CONTACT_MARGIN:
                self.t.append(float(st.t))
                self.z.append(z)
            else:
                self._recorded_done = True
                self._contact_step = self._steps

        if self._steps >= _MAX_STEPS or self._recorded_done and self._steps - self._contact_step >= _SETTLE_FRAMES:
            self.running = False
        return self.running

    def finish(self) -> ObservationRecord:
        """End the session and return the observation record."""
        self.running = False
        return ObservationRecord(
            experiment_id=self.spec.id,
            t=np.asarray(self.t, dtype=float),
            z=np.asarray(self.z, dtype=float),
        )


class Laboratory:
    """Executes experiments in a universe and returns observations.

    The universe's hidden parameters are used here — on the physics side —
    and are not reachable from the records the laboratory returns.
    """

    def __init__(self, universe: Universe):
        self._universe = universe
        self.supported_experiments = ("drop",)

    @property
    def universe_name(self) -> str:
        """Public world name (safe to show the AI)."""
        return self._universe.name

    def truth_summary(self) -> str:
        """GROUND TRUTH — for humans and benchmarks only, never for the agent."""
        s = self._universe.secrets
        return (f"universe={self._universe.name} "
                f"gravity={s.gravity} air_density={s.air_density}")

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
