#!/usr/bin/env bash
# Reproduce Mission 004 Day 1: same universe + same design grid ->
# byte-identical ObservationRecords. (No Mission 004 CLI yet by design —
# Day 1 runs through the existing test/experiment entries.)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
export PYTHONPATH=src

python -m pytest tests/scientist/test_mission_004.py -q

mkdir -p reproducibility/mission_004_day1/results
python - <<'EOF' | tee reproducibility/mission_004_day1/results/run_output.txt
import json

from pymo.scientist import ExperimentSpec, Laboratory
from pymo.universes import load_universe

lab = Laboratory(load_universe("universe_004"))
depths = (1.0, 2.0, 4.0)   # immersion_test.DESIGN_DEPTHS


def run_grid():
    out = []
    for depth in depths:
        rec = lab.run_experiment(
            ExperimentSpec(kind="immersion_test", drop_height=depth))
        out.append({
            "experiment_id": rec.experiment_id,
            "steps": rec.steps,
            "t": [round(v, 12) for v in rec.t],
            "z": [round(v, 12) for v in rec.z],
        })
    return out


first = run_grid()
second = run_grid()
assert first == second, "non-deterministic: identical designs, different records"

with open("reproducibility/mission_004_day1/results/records.json", "w",
          encoding="utf-8") as f:
    json.dump(first, f, indent=1)

print("Mission 004 Day 1 reproduction")
print("universe: universe_004 (Liquid World); instrument: immersion rig")
print("design grid: immersion_test depths", depths)
for rec in first:
    print(f"  {rec['experiment_id']}: {rec['steps']} steps, "
          f"{len(rec['t'])} samples, z {rec['z'][0]:.3f} -> {rec['z'][-1]:.3f}")
print("deterministic: two identical runs -> byte-identical ObservationRecords: OK")
print("records written to reproducibility/mission_004_day1/results/records.json")
print("(measurement data only - no universe secrets)")
EOF
