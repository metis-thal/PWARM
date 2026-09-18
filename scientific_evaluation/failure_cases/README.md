# Failure Cases / 失败案例

Failed, incomplete, and honest-unknown runs are recorded here with the same
rigor as successes — a discovery platform that only publishes wins is a demo,
not science.

## v0.1 recorded failures

| Case | Mission | Outcome | Why it matters |
|------|---------|---------|----------------|
| density unidentifiable under drop/slide apparatus | 002 | `UNIDENTIFIABLE` ×2 | equivalence principle: a = g regardless of mass — the apparatus CANNOT see density; the AI said so instead of guessing. Mission 003 was built on this exact gap. |

## How to add a case

Run any mission with a constrained setup (e.g. an empty instrument catalog,
a starved budget) via `pwarm mission run 003 --knowledge <path>`, save the
`report.json`, and write an `outcome.json` next to it using the vocabulary in
[`../README.md`](../README.md). Planned v0.2 cases:

- `INSUFFICIENT_BUDGET` — run Mission 003 with `compute_cost: 6` (the two
  value-ranked contact experiments fit, but no grant can rescue density).
- `INVALID_HYPOTHESIS` — a buoyancy sample denser than ~50× the fluid trips
  the error-amplification guard; the derive refuses rather than publish junk.
