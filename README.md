# PWARM

### An open research environment for autonomous scientific discovery.

Give an AI scientist an unknown simulated world.

It observes.
It designs experiments.
It forms hypotheses.
It tests them.
It accumulates knowledge.

---

## Mission 001 — Can an AI discover gravity without being told its value?

```
python scripts/demo_mission_001.py
```

```
Estimated gravity: 9.81000 m/s²
Ground truth:      9.81000 m/s²
Experiments:       3 (5m, 10m, 20m drops)
Confidence:        100%
Status:            VERIFIED
```

The AI scientist drops a ball from three different heights, fits each trajectory to a quadratic, and cross-verifies that the same gravitational acceleration emerges independently — a law, not a coincidence.

Knowledge persists: a second run concludes from civilization knowledge without re-running a single experiment.

---

## Why PWARM?

**The physics is real.** All phenomena emerge from bottom-up differential equations — no hardcoded animations, no preset events. The AI sees only what a measurement-only channel provides.

**The science is honest.** The AI cannot cheat. It earns knowledge through the scientific method: plan → execute → observe → hypothesize → verify → publish. What it cannot know, it admits.

**The progression is incremental.** Each Mission adds one new capability. The AI's world grows richer as its methods grow more sophisticated.

---

## Quick Start

```bash
git clone https://github.com/metis-thal/PWARM.git
cd PWARM
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[core,ai,viz,parallel,dev]"
```

```bash
# Run the simplest mission — discover gravity
python scripts/demo_mission_001.py

# Autonomous experiment design
python scripts/demo_mission_002.py

# Budget-constrained science with instrument requests
python scripts/demo_mission_003.py
```

```python
from pymo.scientist import ScientistAgent, ScientistState, Mission
from pymo.universes import load_universe

universe = load_universe("universe_001")
# The AI discovers what's hidden inside — on its own.
```

---

## Missions

Each Mission answers one question about artificial scientific intelligence.

### Mission 001 — Discover Gravity

**Can an AI discover a law without being told what to look for?**

The universe hides `g = 9.81 m/s²`. The AI receives only `(t, z)` samples from three drop experiments. It fits each trajectory independently, then cross-verifies that the same gravitational acceleration emerges — turning three measurements into one law.

```
plan → execute → observe → hypothesize → cross-verify → publish
```

| Metric | Result |
|--------|--------|
| Experiments | 3 |
| Measured g | 9.81000 m/s² |
| Ground truth | 9.81000 m/s² |
| Error | 0.0007% |

### Mission 002 — Autonomous Experimentation

**Can an AI decide what to measure?**

The universe hides two materials with unknown `restitution`, `friction`, and `density`. The AI models every unknown as an interval, scores candidate experiments by information gain, and harvests byproducts — one drop derives both gravity and restitution. It also honestly reports what it *cannot* know: density is unidentifiable in a gravity+contact world (equivalence principle).

```
Unknowns → Uncertainty → Experiment Design → Observation → Knowledge
```

| Metric | Result |
|--------|--------|
| Experiments | 4 (self-chosen) |
| Parameters identified | 5 of 6 |
| Unidentifiable | density (honestly reported) |
| Worst error | 0.14% |

### Mission 003 — Science Under Constraints

**Can an AI do science when experiments cost something?**

The universe grants a finite budget: 4 experiments, 1400 simulation steps, 12 cost units. The AI ranks candidates by **value = information gain / cost** — buying the cheapest design that yields knowledge. When density proves unidentifiable with drop/slide apparatus, the AI analyzes the gap, files an instrument request, receives a Fluid Tank, and identifies density via buoyancy.

```
Budget → Value Ranking → Experiment → Gap Analysis → Instrument Request → Grant → New Experiment
```

| Metric | Result |
|--------|--------|
| Experiments | 6 (4 base + 2 granted) |
| Parameters identified | 7 of 7 |
| Budget discipline | 4 + 2 ≤ 6 |
| Instrument arc | fluid_tank granted once |

### Mission 004+ — The Road Ahead

- **Mission 004**: Fluid & Thermodynamics — temperature, viscosity, thermal conductivity
- **Mission 005+**: Chemistry, Materials, Optics, Geology

---

## Architecture

```
src/pymo/
├── kernel/     # Legacy 2D/3D physics kernels (deprecated, kept for compatibility)
├── geology/    # Stratigraphy, thermal conduction, erosion, tectonics
├── rules/      # Mechanics, thermodynamics, fluids, materials, chemistry
├── ai/         # Observer, hypothesis, verification, experimentation
├── physics/    # Unified multi-physics engine (Genesis-inspired)
│   ├── core/   # Scene, State, Entity, Component (single source of truth)
│   ├── solvers/ # Rigid, SPH, FEM, PBD, Thermal, Chemistry, Geology
│   ├── coupling/ # Explicit multi-physics coupler
│   ├── collision/ # SAP + GJK/EPA + CCD
│   └── integrator/ # TimeStepper
├── interface/  # URDF/MJCF/GLTF parsers, GUI, Sensors, Parallel envs
├── scientist/  # The AI scientist layer
│   ├── state/  # Self-model: uncertainty intervals + statuses
│   ├── information/ # Measurement resolution models
│   ├── designer/ # Choose experiments by value = gain / cost
│   ├── instrument/ # Gap analysis → request → catalog grant
│   └── experiments/ # drop_test, slide_test, buoyancy_test
└── viz/        # OpenGL GPU instancing, PBR, ray-tracing
```

### Core Principles

1. **Zero hardcoded phenomena** — combustion, fracture, flow emerge from equations.
2. **Ground truth / AI separation** — physics engine is absolute truth; AI is a learned approximation.
3. **Visual acceptance every phase** — no headless development; every phase demoable.

---

## Reproducibility

```bash
pytest -q                    # all tests
pytest tests/scientist/      # scientist missions only
```

| Test Suite | Tests | Status |
|------------|-------|--------|
| Mission 001 | 5 | ✅ All pass |
| Mission 002 | 8 | ✅ All pass |
| Mission 003 | 8 | ✅ All pass |
| **Total** | **21** | **✅** |

Knowledge persists across sessions. A second mission run loads `knowledge/universe_*.json` and concludes without re-running experiments.

---

## Research

PWARM is designed for studying how artificial agents can acquire scientific knowledge through experimentation. Key research directions:

- **Curriculum learning** for scientific reasoning
- **Metacognition**: when to stop measuring and admit uncertainty
- **Instrument acquisition**: the AI requests new capabilities when stuck
- **Value-of-information**: budget-aware experiment selection

---

## Contributing

Contributions welcome. The project follows these constraints:

- `physics/`, `render/`, `core/`, `solver/` are frozen — no modifications
- All changes must pass `ruff check` and `pytest`
- New Missions require: universe config, experiment modules, tests, demo script, README section

---

## Roadmap

| Mission | Domain | Status |
|---------|--------|--------|
| 001 | Gravity (free-fall) | ✅ Done |
| 002 | Material properties (restitution, friction) | ✅ Done |
| 003 | Budget constraints + instrument acquisition | ✅ Done |
| 004 | Fluid & Thermodynamics | Planned |
| 005+ | Chemistry, Materials, Optics, Geology | Future |

---

## License

MIT
