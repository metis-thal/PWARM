# Mission 002 — Autonomous Material Discovery / 自主材料发现

**Question:** Can an AI decide WHAT to measure?

"Material World" hides 7 parameters (gravity + density/restitution/friction
for two materials). The AI models every unknown as an interval, scores
candidate designs by expected information gain, respects scientific
dependencies (friction needs gravity first), and harvests byproducts (one
drop derives BOTH g and restitution).

The headline result is the HONEST FAILURE: with drop/slide apparatus the
density observables do not exist (equivalence principle — a = g regardless
of mass), so the AI reports `UNIDENTIFIABLE` instead of guessing. This is a
feature, not a bug — Mission 003 builds on exactly this gap.

- Reproduce: `bash reproducibility/mission_002/run.sh`
- Evaluation (incl. the unidentifiable chain): `../../scientific_evaluation/mission_002/`
