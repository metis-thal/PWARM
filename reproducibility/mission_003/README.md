# Mission 003 — Science Under Constraints / 约束下的科学

**Question:** Can an AI do science when experiments cost something?

"Constrained World" grants 4 experiments / 1400 steps / 12 cost units. The
AI buys the cheapest design that crosses the knowledge threshold
(value = expected utility / cost), and when density proves unidentifiable
it runs the full instrument arc:

```
UNIDENTIFIABLE -> gap analysis -> instrument request (fluid_tank)
-> grant + budget extension -> buoyancy_test -> density identified
```

Six experiments recover all 7 hidden parameters. `results/report.json`
contains the budget ledger and the instrument audit trail.

- Reproduce: `bash reproducibility/mission_003/run.sh`
- Evaluation: `../../scientific_evaluation/mission_003/outcome.json`
