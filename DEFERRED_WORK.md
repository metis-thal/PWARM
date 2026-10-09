# Deferred Work: Phases 3-5 (Future Continuation)

This document outlines the remaining work from the original 5-phase roadmap that was explicitly deferred.

## Genesis Arc — Open Items (recorded after Step 10, 2026-09)

Real gaps found while building Steps 1–10 (prediction → commitment →
verification → competition state), in priority order:

1. **Adjudication persistence asymmetry** — belief-path predictions are
   committed and their verdicts persisted; competition-path commitments
   exist (Step 6) and their verdicts persist (Step 7), but the
   OutputBinding/ObservationReduction used is NOT part of the commitment —
   a committed prediction can be adjudicated through different channels.
2. **Tolerance epistemology** — model tolerances default to 1.0; where a
   model's uncertainty SHOULD come from (self-declared resolution?
   historical calibration?) is undecided. The g·dt² recording offset must
   be absorbed by tolerance, never corrected away.
3. **Refuted-is-final vs re-run discipline** — `record_verification`
   treats refuted as terminal; re-verification appends records but cannot
   overturn a verdict. A reconciliation semantic (e.g. fluke re-runs) is
   needed before verdicts drive model survival.
4. **Model survival semantics** — no winner/weight/elimination exists by
   design; `CompetitionState` provides facts only. Comparing two states
   (including cross-model discriminating pairs like "A confirmed E1 while
   B refuted E1") is the next decision.
5. **Binding declaration source** — ConditionBinding/OutputBinding are
   declared data, but WHO declares them (human lab vocabulary? learned?)
   is open. `material` (string conditions) is out of vocabulary.
6. **experiment.py immersion residue** — the immersion code path is live
   again post-restore; the Day-2 claim-routing landmine (the designer's
   suffix map would route `ambient_fluid.density` to buoyancy_test) remains.
7. **prediction.py cohesion** — five responsibilities in one module
   (models / commitments / verdicts / bindings / reduction) plus a
   state↔knowledge↔prediction import cycle (TYPE_CHECKING workaround);
   split before Phase 3.
8. **Temporal dimension** — CompetitionState has no timestamp ordering;
   whether stale evidence applies to a revised model is unmodelled.
9. **Production import integrity — dangling `pwarm.kernel` imports** —
   registered 2026-10-09 by the pre-merge test-coverage audit; fixing it is
   a product/architecture decision, NOT done as part of the test change.
   See the dedicated section below.

## Production Import Integrity — Dangling `pwarm.kernel` Imports

**Status: REGISTERED ONLY (2026-10-09). Deliberately not fixed in the
test-coverage change; restoring the kernel or re-pointing imports is gated
on an explicit decision.**

`pwarm.kernel` was deleted in commit `75a9e04`, but the following modules
still import it at module level and therefore **fail to import in any
production (non-test) environment**:

| Module | Missing import(s) |
|---|---|
| `pwarm/rules/chemistry.py` | `pwarm.kernel.bodies.Body` |
| `pwarm/rules/thermal.py` | `pwarm.kernel.bodies.Body` |
| `pwarm/rules/ecology.py` | `pwarm.kernel.bodies.Body` (+ `Material`/`circle_body` in its `__main__` block) |
| `pwarm/rules/fracture.py` | `pwarm.kernel.bodies.Body`, `pwarm.kernel.collision.Contact` |
| `pwarm/rules/materials.py` | `pwarm.kernel.bodies3d.Body` |
| `pwarm/viz/viewer.py` | `pwarm.kernel.bodies.{Body,Material}`, `pwarm.kernel.world.World` |
| `pwarm/viz/viewer3d.py` | `pwarm.kernel.math3d.quat_to_axis_angle`, `pwarm.kernel.world3d.World3D` |
| `pwarm/ai/experiment.py` | `pwarm.kernel.bodies.{Material,circle_body}`, `pwarm.kernel.world.World` |
| `pwarm/parallel/ray_parallel.py` | `pwarm.kernel.bodies3d.{Body,Material}`, `pwarm.kernel.world3d.World3D` (plus `ray`, an optional extra) |

`pwarm/ai/observer.py` and `pwarm/ai/closed_loop.py` touch the kernel only
under `TYPE_CHECKING` and import fine.

**Minimal reproduction** (fresh interpreter, no pytest, no conftest):

```
python -c "import sys; sys.path.insert(0, 'src'); import pwarm.rules.chemistry"
# ModuleNotFoundError: No module named 'pwarm.kernel'
```

**Why the test suite is green anyway:** `tests/conftest.py` installs an
in-memory `sys.modules` stand-in for `pwarm.kernel` (mirroring the
pre-deletion dataclass interface, commit `e1a8551`) whenever the real
package is absent. Imports therefore succeed inside pytest, the modules
execute under tests, and coverage counts them. This is deliberate — it
keeps the surviving logic testable instead of excluding 800+ statements —
but it must be read correctly:

> The coverage numbers (87.9% with torch, ≈82.4% in the current CI
> configuration — see `docs/testing/TEST_POLICY.md`, "Coverage baselines")
> measure logic executed under test stubs. They do NOT measure production
> importability. Import integrity and the coverage metric are separate
> concerns; neither implies the other.

**Disposition options (decision pending; nothing implemented):**

1. **Restore** the kernel package from history (`75a9e04^`; full copy at
   `e1a8551`) and re-home it under `src/pwarm/kernel/`.
2. **Re-point** each module's imports onto the current architecture
   (`physics.core`, `physics.WorldEngine`) — per-module decision; the rules
   modules need 2-D body/contact value types that no longer exist, so this
   includes small design work, not a mechanical rename.
3. **Retire** the affected modules formally (remove them or move them to an
   explicit legacy namespace) — product decision.
4. **Interim documentation**: mark the affected modules as not importable in
   production builds in ARCHITECTURE.md/README until 1–3 is decided.

## Phase 3: Multi-Discipline Coupling + Complex Emergent Scenarios (3-4 weeks)

### 3.1 Chemistry & Phase Change Module
- [ ] Substance phase transitions (solid/liquid/gas by temperature/pressure)
- [ ] Chemical reactions: combustion, oxidation, dissolution, thermal decomposition
- [ ] Mass/element conservation enforcement
- [ ] Reaction rate models (Arrhenius)

### 3.2 Advanced Material Mechanics
- [ ] Stress accumulation, deformation, fracture, wear
- [ ] Soft body simulation: cloth, deformable objects
- [ ] Plastic deformation and permanent deformation
- [ ] Fatigue and damage accumulation

### 3.3 Environmental Ecology System
- [ ] Dynamic environment: temperature, humidity, pressure, illumination fields
- [ ] Energy cycles: light → heat → phase change
- [ ] Atmospheric effects on thermal/fluid systems

### 3.4 AI Autonomous Experimentation
- [ ] AI modifies initial conditions autonomously (temperature, pressure, positions)
- [ ] Automated controlled experiments, hypothesis filtering
- [ ] Iterative physics model updates, error correction

## Phase 4: Large-Scale Parallel Simulation + AI Evolution (3-4 weeks)

### 4.1 Parallel Simulation Architecture
- [ ] Ray-based distributed simulation (100+ concurrent worlds)
- [ ] Batch data collection pipeline for AI training
- [ ] Checkpointing and fault tolerance

### 4.2 Neural Physics Acceleration (PINN/FNO)
- [ ] Physics-Informed Neural Networks replacing expensive PDE solves
- [ ] Fourier Neural Operators for fluid/thermal fields
- [ ] Target: 5-10x speedup vs pure numerical
- [ ] Uncertainty quantification for NN predictions

### 4.3 AI Evolution Loop Hardening
- [ ] Permanent loop: hypothesis → batch sim → error compare → model update
- [ ] Discovery of human-unprescribed derivative laws
- [ ] Auto model selection: retain best, discard degraded, continuous evolution

### 4.4 Full Observability & Replay System
- [ ] Time-series DB: every simulation, AI iteration, discovered law
- [ ] Arbitrary timestep replay, scene reproduction, comparative analysis
- [ ] Experiment lineage tracking

## Phase 5: Real-World Alignment + Production Hardening (2 weeks)

### 5.1 Physics Parameter Calibration
- [ ] Calibrate to real constants: g, atm pressure, specific heats, friction coefficients
- [ ] Fix long-term drift, energy non-conservation, phenomenon deviations
- [ ] Validation against real-world benchmarks

### 5.2 Visualization Polish
- [ ] Photorealistic rendering: lighting, shadows, material textures
- [ ] Simplified operations panel: core observe/control/replay only
- [ ] Zero-config startup

### 5.3 System Packaging
- [ ] Single-command launch: background autonomous sim + foreground live observation
- [ ] Hardened AI evolution loop: zero human intervention, long-term self-improvement
- [ ] Docker/container deployment, CI/CD pipeline

### 5.4 Documentation & Archival
- [ ] Architecture manual, dev log, AI iteration records, physics rule docs
- [ ] All observable cases, data reports, law discovery archive
- [ ] API documentation, user guides

## Technical Debt & Known Issues

### Known Limitations (Documented)
1. **SPH fluid**: Density ratios ~0.7-1.3 (clamped at 0.5×rest), slight energy growth over time
2. **Fracture**: Circle split creates overlapping vertices; polygon split works but generates non-convex fragments
3. **Thermal**: Contact conductance simplified; no radiation/convection
4. **3D collision**: EPA not implemented for edge/face cases; fallback to SAT for box-box
5. **AI closed loop**: Uses gplearn fallback; PySR unavailable (Julia network blocked)

### Architecture Decisions (Locked)
- ✅ Custom Numba kernel (no black-box engines)
- ✅ PyVista for visualization (orthographic + interactive)
- ✅ Pluggable AI backend (gplearn+refine / PySR pluggable)
- ✅ Python 3.10 + Numba + SciPy + PyVista + Ray (planned)
- ✅ Aliyun PyPI mirror for China network

## Resuming Work

### Quick Start
```bash
cd D:\project\code\pwarm
.\.venv\Scripts\activate
python -m pytest -q  # 52 tests should pass
```

### Next Logical Increments (Priority Order)
1. **SPH pressure force fix** — stabilize density oscillations
2. **EPA tetrahedron case** — implement full EPA for GJK tetrahedra
3. **Ray parallel wrapper** — `ray.init(); @ray.remote def sim_worker(...)`
4. **PINN integration** — `torch` + custom loss for PDE residuals
5. **Ray Tune integration** — hyperparameter search for NN architectures

### Environment Notes
- Python 3.10 (pinned; 3.14 has no Numba/SciPy/PyVista wheels)
- Aliyun PyPI mirror configured globally
- Julia unavailable (network blocked) → gplearn fallback active
- Windows 10/11 target; PyVista works on WSL2 too

---

*This document captures the state as of Phase 2 completion. All 52 tests pass, ruff clean, 30+ tests across kernel/viz/AI/rules.*