# The AI ↔ Physics Contract / AI 与物理的契约

**核心原则：AI 可以观察世界，但不能读取答案。**
**Core principle: the AI may observe the world, but it can never read the answers.**

```
        AI (ScientistAgent + ScientistState + Designer)
         │
         │  ExperimentSpec          ObservationRecord
         │  ───────────────────►    ◄───────────────────
         │        proposals               measurements
         ▼
┌─────────────────────────┐
│  Scientific World       │   Laboratory (the only door)
│  Laboratory · Knowledge │
└─────────────────────────┘
         │
         │  hidden state (UniverseSecrets — never crosses this line)
         ▼
┌─────────────────────────┐
│  Physics (absolute truth)│
└─────────────────────────┘
```

## The one rule

The AI's ONLY data channel is `ObservationRecord` — measurements, nothing
else. Hidden truth lives in `UniverseSecrets`, loaded exclusively physics-side
by `pymo.universes` and consumed exclusively by `pymo.scientist.experiment`
when it configures a world. `ScientistAgent`, `ScientistState`, and the
designer never import `pymo.universes` and never receive a `Universe` object.

## What the AI may see

| Type | Direction | Contents |
|------|-----------|----------|
| `ExperimentSpec` | AI → physics | a design: kind, height, launch speed, material NAME |
| `ObservationRecord` | physics → AI | `experiment_id`, `t`, `z`, `vx`, `steps` — raw samples |
| `Instrument` catalog | system → AI | what apparatus EXISTS (names + capabilities), which are locked |
| `Budget` | system → AI | the resource envelope (a constraint, not a secret) |
| `KnowledgeBase` | AI's own memory | published laws + material properties |
| `manifest` | system → AI | parameter NAMES only — knowing what you don't know |

## What the AI may never see

- `UniverseSecrets` (gravity value, material property values, air density)
- Any engine internals (`WorldEngine`, scene state, solver options)
- Any "ground truth" comparison outside the evaluation layer
  (`scientific_evaluation/`, CI's reproducibility test, demo epilogues —
  all of which are the human-facing narrator, not the AI)

## How this is enforced

1. **By construction** — `Laboratory` is the only object the AI holds; the
   universe is `Laboratory._universe` (private) and is never returned.
2. **By test** — `tests/scientist/test_api_contract.py`:
   - no AI-side module imports `pymo.universes`;
   - `ObservationRecord` fields are measurement-only;
   - artifacts written by the CLI carry no secret values
     (`tests/test_reproducibility.py::test_ai_artifacts_carry_no_secrets`).
3. **By honesty** — when no observable depends on a parameter, the AI reports
   `UNIDENTIFIABLE` instead of guessing (Mission 002), and requests a new
   instrument (Mission 003).

## Note on demo scripts

`scripts/demo_mission_*.py` ARE the human-facing narrator: they render the
world with `GLRenderer` (reading physics state for the camera) and print
epilogues comparing to truth. That is presentation code, outside the AI —
the same separation as a paper's figure, not part of the scientist.
