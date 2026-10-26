#!/usr/bin/env bash
# Run every experiment; plots and summary.txt files land in experiments/out/<name>/.
#
# All experiments start at once, and each spreads its independent runs over
# worker processes (experiments/common.py: pmap), so the wall time is roughly
# that of the slowest experiment rather than the sum.  Every run is seeded, so
# results match a serial run exactly.  CLIFFNPC_WORKERS=1 runs everything serially.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # tiny matrices: BLAS threads only add contention
EXPS=(e1_foraging e2_lightcone e3_order e4_holonomy e5_threshold e6_village e8_sequence_memory e9_life_signature e10_trivector_affect e11_connection_basis e12_learning_gym e13_td_connection e14_cartan_connection e15_pavlovian_valence e16_fear_v2 e17_surprise_gated e18_affect_food e19_confidence e20_induced_geometry)

start=$(date +%s)
declare -A PIDS
for e in "${EXPS[@]}"; do
  mkdir -p "experiments/out/$e"
  "$PY" "experiments/$e.py" > "experiments/out/$e/run.log" 2>&1 &
  PIDS[$e]=$!
done

fail=0
for e in "${EXPS[@]}"; do
  if wait "${PIDS[$e]}"; then
    echo "ok    $e  ($(( $(date +%s) - start )) s)"
  else
    echo "FAIL  $e  (see experiments/out/$e/run.log)"
    fail=1
  fi
done
echo "all experiments: $(( $(date +%s) - start )) s wall"
exit $fail
