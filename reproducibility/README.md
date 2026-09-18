# Reproducibility / 可复现性

Every mission ships a self-contained reproduction envelope. A stranger (or CI)
runs one script and gets the AI-side evidence back:

```bash
bash reproducibility/mission_001/run.sh
```

Each envelope answers the eight credibility questions:

| Question | Where |
|----------|-------|
| What was hidden? | `config.yaml` → `hidden` (names only) + the universe config it points at; VALUES live physics-side only |
| What could the AI observe? | `config.yaml` → `observation_space`; raw samples in `results/observations.json` |
| What experiments could it perform? | `config.yaml` → `instruments` |
| What did it choose? | `results/report.json` → summary (one reasoned line per experiment) |
| What did it discover? | `results/hypotheses.json` |
| What was the ground truth? | `scientific_evaluation/mission_XXX/` (human-facing, NEVER in AI artifacts) |
| What was the error? | `scientific_evaluation/mission_XXX/outcome.json` |
| What random seed? | `seed.txt` — the loop is fully deterministic; numpy is seeded to 0 as belt-and-braces |

Two-sided separation (structurally enforced, see
`docs/architecture/ai-physics-contract.md`):

- `results/` — **AI-side artifacts**: observations, hypotheses, report.
  Ground truth must never appear here.
- `scientific_evaluation/` — **human-facing evaluation**: compares the AI's
  published knowledge against the hidden truth, records honest outcomes
  (`SUCCESS`, `UNIDENTIFIABLE`, `INSUFFICIENT_BUDGET`, ...), including the
  failures. `expected_output.txt` pins the full console transcript of the
  as-run envelope.
