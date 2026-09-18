#!/usr/bin/env bash
# Reproduce Mission 003 end to end (fresh scientist, deterministic).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
python -m pymo.cli mission run 003 --json reproducibility/mission_003/results
