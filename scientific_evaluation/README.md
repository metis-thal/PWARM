# Scientific Evaluation / 科学评估

Human-facing evaluation of every mission run. This layer — unlike the AI —
may read the hidden truth, because evaluation is what humans do with the AI's
published knowledge.

## Outcome vocabulary / 结果词表

| Verdict | Meaning |
|---------|---------|
| `SUCCESS` | every targeted claim identified within threshold |
| `SUCCESS_WITH_HONEST_UNKNOWN` | claims identified; some reported `UNIDENTIFIABLE` instead of guessed |
| `UNIDENTIFIABLE` | a claim no available apparatus observable depends on |
| `INSUFFICIENT_BUDGET` | informative candidates exist but the budget cannot afford them |
| `INVALID_HYPOTHESIS` | a fit failed quality gates (low R², saturated observable) |
| `FAILED` | the AI published knowledge later shown wrong by evaluation |

Every outcome file records: what was measured vs the truth, per-claim relative
error, experiment count, determinism/seed, and the reasoning chain for any
non-SUCCESS claim. **Failures are first-class citizens here** — see
`failure_cases/`.
