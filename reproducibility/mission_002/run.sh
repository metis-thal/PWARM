#!/usr/bin/env bash
# Reproduce Mission 002 end to end (fresh scientist, deterministic).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
python -m pymo.cli mission run 002 --json reproducibility/mission_002/results
