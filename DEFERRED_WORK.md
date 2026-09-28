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