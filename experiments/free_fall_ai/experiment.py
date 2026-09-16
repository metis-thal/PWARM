"""P8 — Free Fall → AI Discovers Gravity: the first closed-loop experiment.

Lifecycle (the Experiment Manager seed for the benchmark system):

    setup()      create the world (engine + bodies)
    step()       advance one tick and record an observation
    observe()    hand the recorded data to the AI (its ONLY input)
    discover()   AI fits a law to the observations
    evaluate()   compare the AI's derived constant against ground truth

Core rule (from the project charter): the physics engine is the absolute
ground truth; the AI is a learned approximation. The AI here never reads the
engine's gravity parameter — it sees only ``(t, z)`` samples and must recover
``g`` from data alone.

The fitted model is interrogated as a black box: ``g = -d²z/dt²`` is
evaluated through ``DiscoveredLaw.predict`` (central second difference), so
the derivation works for ANY symbolic backend, not just polynomials.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Standalone-run bootstrap (under pytest, pythonpath=src already covers this).
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pymo.ai import LawDiscovery
from pymo.ai.law_discovery import PolynomialBackend
from pymo.physics import WorldEngine, WorldEngineConfig


@dataclass
class FreeFallConfig:
    """Experiment parameters (engine-side facts the AI is NOT allowed to see
    beyond the recorded samples — gravity especially)."""
    height: float = 10.0
    mass: float = 1.0
    radius: float = 0.5
    gravity: float = 9.81            # ground truth, hidden from the AI
    dt: float = 1.0 / 60.0
    discover_every: int = 10         # frames between AI discovery passes
    min_samples: int = 5
    verify_tol_pct: float = 1.0      # |g_ai - g| / g below this → VERIFIED
    r2_tol: float = 0.99


@dataclass
class DiscoveryReport:
    """Everything the panel/console needs about the AI's current state."""
    phase: str = "OBSERVING"         # OBSERVING | DISCOVERING | VERIFIED | LANDED
    n_samples: int = 0
    t_span: tuple[float, float] = (0.0, 0.0)
    expression: str = ""
    a: float = 0.0                   # z(t) = a*t^2 + b*t + c
    b: float = 0.0
    c: float = 0.0
    g_ai: float = 0.0
    r2: float = 0.0
    error_pct: float = float("inf")
    verified: bool = False


@dataclass
class _Recorder:
    """The observation channel: the only data that flows from the engine to
    the AI. Ground truth writes here; the AI reads here; nothing else."""
    t: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)

    def record(self, t: float, z: float) -> None:
        self.t.append(t)
        self.z.append(z)

    def clear(self) -> None:
        self.t.clear()
        self.z.clear()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.t, dtype=float), np.asarray(self.z, dtype=float)


class FreeFallExperiment:
    """A sphere dropped onto ground; an AI that must discover gravity."""

    def __init__(self, config: FreeFallConfig | None = None):
        self.config = config or FreeFallConfig()
        self.engine: WorldEngine | None = None
        self._recorder = _Recorder()
        self._landed = False
        self._landing_t: float | None = None
        self.report = DiscoveryReport()

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> WorldEngine:
        """Create the world: falling sphere (rigid index 0) + static ground."""
        cfg = self.config
        engine = WorldEngine(WorldEngineConfig(
            dt=cfg.dt,
            substeps=1,
            gravity=(0.0, 0.0, -cfg.gravity),
            rigid={"enabled": True},
        ))
        engine.create_rigid_body(
            (0, 0, cfg.height), mass=cfg.mass,
            shape="sphere", shape_params={"radius": cfg.radius},
        )
        engine.create_rigid_body(
            (0, 0, -0.5), mass=0.0,
            shape="box", shape_params={"half_extents": [10, 10, 0.5]},
        )
        engine.finalize_setup()
        self.engine = engine
        self._recorder.clear()
        self._landed = False
        self._landing_t = None
        self.report = DiscoveryReport()
        return engine

    def step(self) -> DiscoveryReport:
        """Advance one tick and record an observation. Once the body first
        touches the ground, recording stops — the AI's dataset is the clean
        free-fall segment, exactly what a real observer would trust."""
        if self.engine is None:
            raise RuntimeError("call setup() before step()")
        state = self.engine.tick()
        if not self._landed:
            z = float(state.rigid_pos[0][2])
            self._recorder.record(float(state.t), z)
            self.report.n_samples = len(self._recorder.t)
            self.report.t_span = (self._recorder.t[0], self._recorder.t[-1])
            if z < self.config.radius + 0.05:
                self._landed = True
                self._landing_t = float(state.t)
        return self.report

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """The AI's only input: recorded (t, z) samples."""
        return self._recorder.arrays()

    def discover(self) -> DiscoveryReport:
        """AI pass: fit z(t) = a*t² + b*t + c to the observations and derive
        g from the fitted model (black-box second difference)."""
        t, z = self.observe()
        rep = self.report
        if len(t) < self.config.min_samples:
            rep.phase = "OBSERVING"
            return rep

        law = LawDiscovery(PolynomialBackend(degree=2)).discover_from_observation(t, z)
        rep.expression = law.expression
        rep.r2 = float(law.r2)
        if law.coeffs is not None and len(law.coeffs) >= 3:
            rep.c = float(law.coeffs[0])
            rep.b = float(law.coeffs[1])
            rep.a = float(law.coeffs[2])

        # g = -z''(t): z(t) = h - g*t^2/2  =>  z'' = -g. Evaluated through the
        # AI's predict (works for any backend twice-differentiable in t).
        t0 = 0.5 * (float(t[0]) + float(t[-1]))
        h = max(1e-3, 0.01 * (float(t[-1]) - float(t[0])))
        z0 = float(np.asarray(law.predict(np.array([t0]))).ravel()[0])
        zp = float(np.asarray(law.predict(np.array([t0 + h]))).ravel()[0])
        zm = float(np.asarray(law.predict(np.array([t0 - h]))).ravel()[0])
        zpp = (zp - 2.0 * z0 + zm) / (h * h)
        rep.g_ai = -zpp
        return rep

    def evaluate(self) -> DiscoveryReport:
        """Compare the AI's derived gravity against ground truth."""
        rep = self.report
        g_true = self.config.gravity
        if rep.g_ai != 0.0:
            rep.error_pct = abs(rep.g_ai - g_true) / g_true * 100.0
        rep.verified = (rep.r2 >= self.config.r2_tol
                        and rep.error_pct <= self.config.verify_tol_pct)
        if self._landed:
            rep.phase = "VERIFIED" if rep.verified else "LANDED"
        elif rep.verified:
            rep.phase = "VERIFIED"
        elif rep.n_samples >= self.config.min_samples:
            rep.phase = "DISCOVERING"
        else:
            rep.phase = "OBSERVING"
        return rep

    def run_headless(self, n_steps: int = 100) -> DiscoveryReport:
        """Full closed loop without any UI — used by tests and benchmarks."""
        self.setup()
        for i in range(n_steps):
            self.step()
            if i % self.config.discover_every == 0:
                self.discover()
                self.evaluate()
        self.discover()
        self.evaluate()
        return self.report

    # -- introspection for the demo UI ---------------------------------------

    @property
    def landed(self) -> bool:
        return self._landed

    @property
    def landing_t(self) -> float | None:
        return self._landing_t

    def sphere_state(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        """(t, position, velocity) of the falling body — read-only ground
        truth for display. None before setup()/if no bodies exist."""
        if self.engine is None:
            return None
        state = self.engine.scene.double_buffer_read
        if state is None or state.rigid_pos is None or len(state.rigid_pos) == 0:
            return None
        pos = state.rigid_pos[0].copy()
        vel = (state.rigid_linvel[0].copy()
               if state.rigid_linvel is not None else np.zeros(3))
        return float(state.t), pos, vel
