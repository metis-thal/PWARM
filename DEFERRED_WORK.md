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
   registered 2026-10-09 by the pre-merge test-coverage audit; **partially
   fixed by the P0 remediation later the same day** (4 annotation-only
   modules re-pointed, `pwarm.parallel` package gating added, import gate
   added); the remaining exemptions and two function-level hazards are
   product/architecture decisions.  See the dedicated section below.

## Production Import Integrity — Dangling `pwarm.kernel` Imports

**Status: PARTIALLY FIXED (2026-10-09 P0 remediation; 2026-10-10 P1-1).** Four
annotation-only modules re-pointed, `pwarm.parallel` package gating added,
and a regression gate (`tests/test_production_imports.py`) now enforces the
exemption list.  P1-1 additionally restored the 2-D rigid-body value types
(`pwarm/rules/bodies2d.py`, from the pre-deletion kernel at `e1a8551`) and
re-pointed `rules/fracture.py` onto them — the fracture module is no longer
exempt.  The remaining items below are deliberate, registered exemptions
with disposition plans — not oversights.

`pwarm.kernel` was deleted in commit `75a9e04` (as `pymo/kernel/`; the
`pymo → pwarm` rename in `5f8728e` rewrote the dangling imports verbatim).

**Fixed in P0 (2026-10-09) — no runtime/algorithm changes:**

| Module | Fix |
|---|---|
| `pwarm/rules/chemistry.py` | kernel `Body` was annotation-only (postponed annotations); import moved under `TYPE_CHECKING` with the duck-typed structural contract documented in place. |
| `pwarm/rules/thermal.py` | same; contacts were already duck-typed (`.a`/`.b` objects or `(Body, Body)` tuples). |
| `pwarm/rules/ecology.py` | same for the module body; its `__main__` demo block keeps a local kernel import and remains unusable — pending retirement. |
| `pwarm/rules/materials.py` | same (one `DeformableBody` field annotation only). |
| `pwarm/parallel/__init__.py` | lazy PEP-562 re-export: `import pwarm.parallel` no longer requires ray; touching the ray-backed symbols raises an actionable `ModuleNotFoundError` with the `PWARM[parallel]` install hint; a missing `pwarm.kernel` (still broken inside `ray_parallel`) propagates unmasked. |

**Fixed in P1-1 (2026-10-10):**

| Module | Fix |
|---|---|
| `pwarm/rules/bodies2d.py` (NEW) | the 2-D rigid-body value types (`Material`, `Body`, `circle_body`, `box_body`, `polygon_inertia`, `cross`) restored from the pre-deletion kernel (`75a9e04^` / `e1a8551`) as pure data — numpy-only, no World/collision/solver/integrator coupling; the historical semantics (analytic circle inertia `½mr²`, standard polygon inertia theorem, static ⇒ `inv_mass=0`/`inertia=∞`, non-positive mass ⇒ infinite mass, degenerate polygon ⇒ unit inertia) are preserved verbatim and now pinned by `tests/rules/test_bodies2d.py`. |
| `pwarm/rules/fracture.py` | runtime kernel import replaced by `pwarm.rules.bodies2d`; the consumed contact shape is now a structural `ContactLike` Protocol (a/b/point/normal/penetration/friction) instead of the deleted `kernel.collision.Contact`. Algorithm untouched. |
| `tests/rules/test_fracture.py` | migrated off the conftest kernel stub — constructs real `bodies2d` bodies and a local minimal `Contact` record; all 20 fracture tests now validate production behavior without the stub. |

Natural P1 follow-up (registered, not done in P1-1 to keep its diff minimal):
`rules/{chemistry,thermal,ecology,materials}.py` still carry `TYPE_CHECKING`
imports of the deleted `pwarm.kernel.bodies` for their annotations; each can
now be re-pointed to `pwarm.rules.bodies2d` in one line once their structural
contracts are reviewed against the restored types.

**Minor findings from the independent pre-commit review (2026-10-10) —
registered as follow-ups, deliberately NOT fixed in the P0/P1-1 changeset:**

1. `rules/fracture.py` carries an unreachable duplicated block after the
   first `return` of `estimate_contact_stress` — pre-existing dead code,
   preserved untouched by the remediation.
2. `docs/testing/TEST_POLICY.md` traceability matrix lacks a row for the new
   `tests/rules/test_bodies2d.py → rules/bodies2d`.
3. `bodies2d.polygon_inertia` silently returns NEGATIVE inertia for
   clockwise-wound polygons (`inv_inertia` collapses to 0); the CCW
   convention is documented but not enforced — decide whether to reject CW
   input explicitly at the next value-type contract review.
4. `Body.vertices` is documented as an `(N, 2)` ndarray; list-typed vertices
   work for `rules/fracture.py` but crash `world_vertices()`.
5. The conftest stub's value semantics diverge from `bodies2d` (ang_vel
   type, static inertia, world_vertices return) — no consumer crosses both
   today; the sibling re-point's contract review (above) covers it.
6. Stale untracked `src/pymo.egg-info/SOURCES.txt` still lists the old
   kernel files (local editable-install artifact; regenerates on reinstall).

**Still failing at import — exempted in `tests/test_production_imports.py`
(`LEGACY_EXEMPTIONS`; the list may only shrink):**

| Module | Missing import(s) | Disposition plan |
|---|---|---|
| `pwarm/viz/viewer.py` | `pwarm.kernel.bodies.{Body,Material}`, `pwarm.kernel.world.World` | retire-or-migrate onto the `SceneSnapshot`/GLRenderer pipeline (P2) |
| `pwarm/viz/viewer3d.py` | `pwarm.kernel.math3d.quat_to_axis_angle`, `pwarm.kernel.world3d.World3D` | same as `viewer.py` (P2) |
| `pwarm/ai/experiment.py` | `pwarm.kernel.bodies.{Material,circle_body}`, `pwarm.kernel.world.World` | migrate onto `physics.WorldEngine` + `physics.ai.WorldObserver`, or retire (P2) |
| `pwarm/parallel/ray_parallel.py` | `ray` (optional `parallel` extra) + `pwarm.kernel.bodies3d`/`world3d` | migrate `_build_world`/`_create_body` onto `create_world_engine` (P2); real-ray integration remains untested either way |

**Function-level kernel hazards (module imports fine, calls fail — found by
the 2026-10-09 pre-fix audit; invisible to the import gate):**

- `pwarm/ai/observer.py::collect_free_fall` imports the kernel inside the
  function; calling it in production raises.  (An earlier note here claimed
  observer.py touches the kernel "only under `TYPE_CHECKING`" — that was
  inaccurate.)  Plan: re-point onto `physics.ai.WorldObserver` +
  `create_world_engine` (P1).
- `pwarm/viz/snapshot.py::build_snapshot_from_world` imports kernel shape
  classes for isinstance dispatch.  The production render path does NOT use
  it (`Scene.get_render_snapshot` → `DoubleBuffer` → `GLRenderer`), but its
  docstring promises engine-thread use.  Plan: add a
  `build_snapshot_from_engine(WorldEngine)` path and deprecate the kernel
  one (P1).

**Optional-dependency surface (same audit):**

- `pwarm/ai/pinn.py` imports `torch` at module level — covered by the `ai`
  extra; `import pwarm.ai` does not require it.
- `pwarm/viz/text_overlay.py` imports `PIL` (Pillow) at module level, but
  **Pillow is not declared in any extra** — a bare or CI `[viz,dev]` install
  cannot import it (tests skip via `importorskip`).  Disposition pending:
  declare it (e.g. in `viz`) or drop the dependency.
- `pwarm/viz/gl_renderer.py` imports `glfw`/`moderngl` — covered by `viz`.
- `numba` is a declared core dependency but no module in `src/pwarm` imports
  it at module level (observation only; packaging review out of scope here).

**Regression gate.** `tests/test_production_imports.py` imports every shipped
module in a fresh stub-free subprocess (conftest never loads; the child
asserts `pwarm.kernel` never entered `sys.modules`).  `LEGACY_EXEMPTIONS`
must stay in sync with the table above: an exemption whose module starts
importing fails the gate as stale, and any new module-level broken import
fails the gate by name.

**Why the pytest suite is green anyway:** `tests/conftest.py` installs an
in-memory `sys.modules` stand-in for `pwarm.kernel` (mirroring the
pre-deletion dataclass interface, commit `e1a8551`) whenever the real
package is absent, so the exempted modules stay executable and counted
under tests.  The coverage numbers (87.9% with torch, ≈82.4% in the CI
configuration — see `docs/testing/TEST_POLICY.md`) measure logic executed
under stubs; the import gate measures production importability.  Import
integrity and the coverage metric are separate concerns; neither implies
the other.

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