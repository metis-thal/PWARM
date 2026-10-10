# PWARM Test Policy

**Status:** active · **Applies to:** `src/pwarm/**`, `tests/**` · **Gate:** branch coverage ≥ 80 %

This document records the test plan and test summary information for the
PWARM codebase, **aligned with** the test-documentation set of
ISO/IEC/IEEE 29119-3 and the quality characteristics of ISO/IEC 25010.
This is a **self-assessment**: no formal ISO/IEC/IEEE conformity
assessment has been performed, and none is claimed. It records the test
levels, the coverage criteria, the traceability from test files to
production modules, the environment assumptions, and the deliberately
pinned implementation quirks. CI enforces the coverage gate on every push
and pull request (`.github/workflows/ci.yml`); `make coverage` enforces it
locally.

**Standards-reference caveat:** part-level titles (29119-1 concepts &
vocabulary, 29119-2 test processes, 29119-3 test documentation, 29119-4
test techniques) are reliable. Clause-level numbers were deliberately
removed pending verification against the standard text — the authoring
environment had no network access to the standards (2026-10-09), and no
verification result is claimed.

## 1. Standards alignment

| Concern | Reference (part level) | Self-assessed implementation in PWARM |
|---|---|---|
| Test processes & documentation | ISO/IEC/IEEE 29119-3 (test documentation) | This document serves as the project's test plan in the 29119-3 sense; each test-file docstring is the per-case specification |
| Quality characteristics under test | ISO/IEC 25010 | Functional correctness (physics conservation laws, parser contracts), reliability (error paths, failure isolation), maintainability (this gate), portability (headless/CI runs) |
| Structural coverage criterion | ISO/IEC/IEEE 29119-4 (test techniques) | coverage.py in **branch mode**, `fail_under = 80` in `pyproject.toml` |
| Reproducibility of scientific claims | project doctrine (`pwarm-science-design`) | Deterministic-physics missions are re-run in CI ("stranger test") and asserted in `tests/test_reproducibility.py` |

The 80 % branch-coverage threshold is the agreed release gate. It is a
*minimum* over the whole package, not an average of per-module numbers:
`--cov-fail-under` applies to the combined total, so thin modules must be
compensated elsewhere. 80 % is a **gate, not a fixed coverage number** —
measured coverage differs between environments (see §1a).

## 1a. Coverage baselines (measured 2026-10-09, Windows, CPython 3.10.11)

Two measured baselines exist because CI deliberately installs a smaller
dependency set than a full development environment:

| Configuration | Branch coverage | Collected | Difference source |
|---|---|---|---|
| Full development install (`pip install -e ".[viz,ai,dev]"`, torch present) | **87.9 %** | 795 (792 passed + 3 skipped) | — |
| Current CI install (`pip install -e ".[viz,dev]"`, no torch) | **≈ 82.4 %** | 773 (770 passed + 3 skipped) | `tests/ai/test_legacy_ai.py` (22 tests) skips at module import via `pytest.importorskip("torch")` |

Both baselines pass the 80 % gate (exit 0). The CI number was **simulated
locally** by excluding the torch-gated file (`--ignore=tests/ai/test_legacy_ai.py`);
it is unverified against a real CI run until the workflow executes. The
3 skips in both runs are the pre-existing slow-test markers in
`tests/geology/test_geology.py`.

> **Import integrity is a separate metric.** A handful of modules
> (rules/chemistry·thermal·ecology·fracture·materials, viz.viewer*, the
> legacy ai experiment/observer modules, parallel.ray_parallel) still
> import the removed `pwarm.kernel` and are NOT importable in a production
> environment; the coverage above is measured through the test stubs
> described in §5. See `DEFERRED_WORK.md`, "Production Import Integrity".

## 2. Test levels

1. **Unit tests** — pure functions and classes (`rules/*`, `geology/*`,
   `physics/solvers/*`, `interface/*`, `viz/snapshot`). Numerical properties
   (conservation, bounds, determinism) are asserted against analytic
   expectations, not against the implementation.
2. **Engine-integration tests** — the real `WorldEngine` pipeline
   (`tests/physics/test_rigid_solver.py`, `test_sph.py`, `test_collision.py`,
   `tests/physics/test_ai_observer.py`), including analytic free-fall and
   momentum checks.
3. **System tests** — the scientist loop end-to-end
   (`tests/scientist/test_mission_00*.py`, `test_open_genesis_pilot.py`) and
   the CLI stranger test in CI.
4. **Regression/pinned-behaviour tests** — see §6: defects and quirks are
   pinned with an explicit "pinned as implemented" note so any change is a
   conscious decision.

## 3. Traceability (test file → production module)

| Test file | Covers |
|---|---|
| `tests/rules/test_fluid.py` | `rules/fluid` |
| `tests/rules/test_chemistry.py` | `rules/chemistry` |
| `tests/rules/test_thermal.py` | `rules/thermal` |
| `tests/rules/test_ecology.py` | `rules/ecology` |
| `tests/rules/test_fracture.py` | `rules/fracture` |
| `tests/rules/test_materials.py` | `rules/materials` |
| `tests/geology/test_geology.py` | `geology/grid`, `geology/rock_materials`, `geology/solver` (smoke) |
| `tests/geology/test_rock_library_api.py` | `geology/rock_materials`, `geology/grid` coordinates/stratigraphy |
| `tests/geology/test_surface_processes.py` | `geology/processes/{erosion,sedimentation,tectonics}` |
| `tests/geology/test_geothermal_and_solver.py` | `geology/processes/thermal`, `geology/solver` |
| `tests/physics/test_collision.py` | `physics/collision` |
| `tests/physics/test_rigid_solver.py` | `physics/solvers/rigid`, `physics/__init__` |
| `tests/physics/test_sph.py` | `physics/solvers/sph`, `sph_kernels` (JIT paths) |
| `tests/physics/test_kernels_edge.py` | `sph_kernels`, `contact_kernels` (pure-Python branches) |
| `tests/physics/test_solver_stubs.py` | `physics/solvers/{stubs,base}` |
| `tests/physics/test_integrator.py` | `physics/integrator` |
| `tests/physics/test_entity_and_scene.py` | `physics/core/{entity,scene}` |
| `tests/physics/test_ai_observer.py` | `physics/ai`, `physics/__init__` (engine) |
| `tests/physics/test_package_imports.py` | `physics/__init__` lazy interface re-export (contract + fresh-process import order) |
| `tests/ai/test_legacy_ai.py` | `ai/{pinn,observer,closed_loop,experiment,law_discovery}` (skips without torch — e.g. current CI) |
| `tests/interface/test_asset_parser.py` | `interface/asset_parser` |
| `tests/interface/test_sensors.py` | `interface/sensors` |
| `tests/interface/test_gui.py` | `interface/gui` |
| `tests/interface/test_parallel.py` | `interface/parallel` |
| `tests/parallel/test_ray_parallel.py` | `parallel/ray_parallel` (ray-free logic; module skips if a real ray is installed) |
| `tests/viz/test_gl_renderer.py` | `viz/gl_renderer` (meshes, frustum) |
| `tests/viz/test_gl_renderer_windowless.py` | `viz/gl_renderer` (input handling, windowless) |
| `tests/viz/test_snapshot.py` | `viz/snapshot` |
| `tests/viz/test_text_overlay.py` | `viz/text_overlay` (needs a standalone GL context) |
| `tests/viz/test_viewers.py` | `viz/viewer`, `viz/viewer3d` (off-screen pyvista; skips on Linux without a display — verified green on Linux CI 2026-10-10; skips on CI Windows runners, where GPU-less software OpenGL makes VTK's native render access-violate — uncatchable in-process, so the guard is static) |
| `tests/scientist/*` | `scientist/**` |
| `tests/test_reproducibility.py` | reproducibility artifacts |

## 4. Coverage measurement and exclusions

* Runner: `pytest --cov=pwarm --cov-branch --cov-fail-under=80`
  (`[tool.coverage.*]` in `pyproject.toml`).
* Excluded *lines* (not modules), per industry practice:
  `pragma: no cover`, `if TYPE_CHECKING:`, `raise NotImplementedError`,
  `__main__` demo blocks, `@abstractmethod` bodies. The numba ImportError
  fallbacks in the JIT kernel modules carry inline pragmas.
* No whole module is omitted. Modules whose *imports* require unavailable
  optional dependencies are still measured and executed through the stubs
  below (§5).

## 5. Test environment and legacy stubs

`tests/conftest.py` installs a stand-in for `pwarm.kernel` (removed in
commit 75a9e04) **only when the real package is absent**, mirroring the
historical dataclass interface (`Material`, `Body`, `Contact`, 3-D shapes,
`quat_to_axis_angle`). This keeps `rules.{chemistry,thermal,ecology,fracture,materials}`,
`viz.viewer*`, `parallel.ray_parallel` and `ai.{experiment,observer}` —
which still import the removed kernel — executable and measured instead of
excluded.

Additional environment contracts:

* **ray** is optional (`parallel` extra, not installed in CI):
  `tests/parallel/test_ray_parallel.py` checks `importlib.util.find_spec("ray")`.
  When ray is absent it injects an identity-`remote` fake before import;
  when a real ray IS installed the module **skips instead of shadowing** it
  (the direct-call test style requires the fake). Live-ray paths
  (SimulationBatch, run_single_*) are out of scope and untested either way.
* **OpenGL**: `tests/viz/test_text_overlay.py` needs a standalone moderngl
  context; the skip guard wraps only the context-creation call (no
  assertions inside), so it cannot mask assertion failures.
* **pyvista/VTK**: viewer tests run in off-screen mode with function-scoped
  fixtures (no shared state between tests). On Linux without
  `DISPLAY`/`WAYLAND_DISPLAY` the module skips — a narrowly identified
  environment precondition, not an exception-wide guard. Real Linux-CI
  behaviour has **not** been verified (2026-10-09); run under xvfb there.
* torch ships with the `ai` extra only (not `viz`) and is exercised by the
  PINN tests in `tests/ai/test_legacy_ai.py`; without it that file skips
  (see §1a).

## 6. Pinned implementation quirks (behaviour as implemented)

These are asserted *as implemented* with an explicit note, so a change is a
conscious decision rather than an accident. They are defect candidates, not
contract:

* `geology/processes/thermal.solve_thermal_step` clamps temperatures **up**
  to each rock's melting point (`np.maximum`, opposite of the comment's
  "prevent superheating").
* `geology/processes/sedimentation.add_sediment_layer` writes the new layer
  into the *first* flat block (column base) although the docstring says
  "adds at surface" — flat-layout axis-order mismatch.
* `geology/processes/erosion.apply_stream_power_erosion` sizes its rock-factor
  table by `rock_id.max()` but writes indices 1–3 unconditionally: any surface
  with max id < 3 raises IndexError; empty input raises ValueError.
* `rules/thermal` flux clip bounds only the *hot* body's capacity, so a much
  lighter cold partner can overshoot equilibrium.
* `rules/materials.FatigueState.add_cycle` compares stress in MPa against
  `endurance_limit / 1e6`, i.e. the documented-MPa default acts as 1e-4 MPa.
* `rules/chemistry` ideal-gas density ignores `density_gas`; the
  `iron_oxidation` product row re-creates a zero `iron` key;
  `apply_phase_transitions` is a placeholder.
* `rules/ecology.update_environment` reports CLEAR for 0.3 < cloud ≤ 0.6.
* `physics/integrator` `_merge_states` adds the full frame dt once per
  substep, so time advances by dt × substeps per frame.
* `physics/core/state.State.copy` shares the `GlobalQuantities` object, so
  the live conservation check inside `TimeStepper.step` compares an object
  against itself and never logs.
* `viz/text_overlay.TextPanel._img_height` is never updated (dead field);
  every draw re-rasterizes; a released panel cannot draw again.
* `viz/gl_renderer` elevation is the polar angle from +z (not from the
  horizontal plane, despite the dataclass comment).
* `parallel/ray_parallel.CheckpointManager.load_latest` sorts lexicographically
  (`step10` < `step2`); `ExperimentRunner._expand_grid` does not propagate
  `record_every`/`seed`; `SimulationWorker._build_world` runs outside the
  try-block so unknown shapes escape as raw ValueError.
* `ai/observer.WorldObserver3D.observe` calls `engine.get_state()`, which
  `WorldEngine` does not implement (test supplies the intended contract).

## 7. Defect log (found by this test effort, fixed in source)

| # | Defect | Fix |
|---|---|---|
| 1 | `interface/asset_parser._create_entity_from_link` built `Entity(name=...)` without the required `id` — every URDF load raised TypeError | construct with `EntityID()` |
| 2 | The unconditional `entity.add(ComponentMask.RIGID_BODY)` re-added the rigid bit to zero-mass (static) entities | add the bit only for positive mass |
| 3 | `pwarm.physics.__init__` imported `..interface` at module level, creating a package-import cycle that broke any process whose first pwarm import was `pwarm.interface.*` (including multiprocessing workers) | lazy module-level `__getattr__` re-export; regression pin `tests/physics/test_package_imports.py` |
| 4 | `ai/pinn.PINNWrapper.load` used the torch ≥ 2.6 default `weights_only=True`, which rejects its own config-bearing checkpoints | `weights_only=False` (self-produced files) |

## 8. Commands

```
make test          # full suite
make coverage      # suite + branch-coverage gate (fail_under=80)
ruff check src tests
```
