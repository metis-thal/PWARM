# Mission 001 — Discover Gravity / 发现重力

**Question:** Can an AI discover a law without being told what to look for?

The universe ("Unknown Planet") hides `g = 9.81 m/s^2`. The AI receives only
`(t, z)` trajectory samples from three drop experiments it planned itself
(heights 10/20/5 m). It fits each free-fall parabola independently, then
cross-verifies: the same g emerging from independent conditions is a LAW.

- Reproduce: `bash reproducibility/mission_001/run.sh`
- Verify: compare `results/report.json` against
  `../../scientific_evaluation/mission_001/outcome.json`
- Console transcript of the as-run envelope: `expected_output.txt`
