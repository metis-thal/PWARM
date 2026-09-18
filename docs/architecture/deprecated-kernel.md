# The deprecated kernel — a post-mortem / 遗留内核的存档

`src/pymo/kernel/` (legacy 2D/3D physics) was removed in commit `75a9e04`
(v0.1). This document is the record that deletion advice asks for — kept
because "we deleted it" should always come with "here is why that was safe".

## Why it existed

The kernel was PWARM's first physics implementation: 2D circle/edge worlds
(`world.py`, `bodies.py`) and a 3D extension (`world3d.py`, `bodies3d.py`,
`collision3d.py`, `math3d.py`). It powered the early AI discovery loops
(P0/P8 era) and the first closed-loop gravity recovery demos.

## Why it was replaced

The Genesis-inspired unified engine (`pymo.physics`) superseded it on every
axis that matters:

| | kernel | physics |
|---|--------|---------|
| scope | rigid only, hand-rolled per-dimension | rigid + SPH + FEM + MPM + PBD + thermal + chemistry + geology |
| state | per-solver ad-hoc structs | single `State` / entity components (one source of truth) |
| collision | bespoke | SAP broad-phase + GJK/EPA + CCD, shared by all solvers |
| coupling | none | explicit coupler, sub-stepped integrator |
| AI boundary | ad hoc | `Laboratory` facade + measurement-only records |

## What replaced it

- `pymo.physics` — the unified engine (see README architecture tree)
- `pymo.scientist.experiment.Laboratory` — the AI↔physics door
- `tests/physics/` — engine verification (conservation, collision, architecture)

## Should you use it?

No. It is gone from v0.1; the last full copy lives at tag/commit
`e1a8551` (and everywhere before `75a9e04`) if you need the history. If you
are migrating old experiments: kernel `World.step` maps to
`WorldEngine.tick`, kernel bodies map to `create_rigid_body`, and the
scientist layer has never depended on the kernel — Mission 001–003 all run
on `pymo.physics` only.
