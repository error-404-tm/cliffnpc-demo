#!/usr/bin/env bash
# Run every experiment; plots and summary.txt files land in experiments/out/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for e in e1_foraging e2_lightcone e3_order e4_holonomy e5_threshold e6_village; do
  echo "=== $e"
  "$PY" "experiments/$e.py"
done
