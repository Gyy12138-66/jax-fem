#!/usr/bin/env bash
# ============================================================================
# Single-layer solver / platform comparison on the smoke2 cube contract.
#
# Every arm runs the SAME mesh, the SAME layer-1 path (639 rows), the SAME
# contract argv (mechanics on, event-driven cadence); only the platform and
# the linear solver differ. Arms are described by arms/<name>.json
# (conda_env, platform, extra_argv). Serial, idempotent per arm.
#
#   bash run_compare.sh                       # all arms in arms/
#   ARMS="cpu_pardiso gpu_pardiso" bash run_compare.sh
#   PRE=<preflight dir> OUT=<output dir> bash run_compare.sh
#
# Results: <OUT>/<arm>/ (solver products) and, after collect_results.sh,
# results/ in this folder (small files only: profiles, configs, table).
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../../.." && pwd)}"
PRE="${PRE:-/home/user/work/159/output/v2_cube_smoke_smoke2/preflight}"
OUT="${OUT:-/home/user/work/159/output/v2_cube_smoke_smoke2/solver_compare}"
ARM_TIMEOUT="${ARM_TIMEOUT:-7200}"
L1="$PRE/v2_cube_smoke_path_layer1.csv"
LOG="$OUT/solver_compare.log"
mkdir -p "$OUT"
export PYTHONPATH="$REPO" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OMP_NUM_THREADS="${MKL_NUM_THREADS:-8}"
cd "$REPO"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}
[ -f "$L1" ] || { say "layer-1 path missing: $L1 (run model/runs/v_cube_smoke.sh stage 2 first)"; exit 2; }
if [ -z "${ARMS:-}" ]; then
  ARMS=$(ls "$HERE/arms"/*.json | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')
fi
mapfile -t ARGV < <(python3 -c "import json;print('\n'.join(json.load(open('$PRE/runner_contract.json'))['argv']))")
COMMON=(--path-file "$L1" --cooling-steps 0 --no-release-after-cooling --thermal-output-every 100000 --summary-every 100)

say "======== solver comparison on smoke2 layer 1 (ARMS: $ARMS) ========"
for arm in $ARMS; do
  SPEC="$HERE/arms/$arm.json"
  [ -f "$SPEC" ] || { say "no arm spec $SPEC"; exit 2; }
  ENV=$(python3 -c "import json;print(json.load(open('$SPEC'))['conda_env'])")
  PYBIN=$(python3 -c "import json;print(json.load(open('$SPEC'))['python_bin'])")
  PLATFORM=$(python3 -c "import json;print(json.load(open('$SPEC'))['platform'])")
  # drop the empty line an empty extra_argv would produce (argparse rejects '')
  mapfile -t EXTRA < <(python3 -c "import json;print('\n'.join(json.load(open('$SPEC'))['extra_argv']))" | sed '/^$/d')
  D="$OUT/$arm"
  if is_complete "$D"; then say "$arm complete, skip"; continue; fi
  [ -x "$PYBIN" ] || { say "$arm: python_bin not executable: $PYBIN"; exit 2; }
  rm -rf "$D"; mkdir -p "$D"
  # arms are launched by absolute interpreter path: jax-fem-gpu lives under
  # miniconda3, jax-fem-env under miniforge3, and one conda.sh cannot activate both
  export JAX_PLATFORM_NAME="$PLATFORM"
  say "$arm start (env=$ENV python=$PYBIN platform=$PLATFORM extra: ${EXTRA[*]:-})"
  timeout "$ARM_TIMEOUT" "$PYBIN" -m jax_fem_am.simulation.runner "${ARGV[@]}" "${COMMON[@]}" \
    --xla-platform "$PLATFORM" ${EXTRA[@]+"${EXTRA[@]}"} \
    --output-dir "$D" --profile-json "$D/profile.json" --profile-label "solver-compare-$arm" \
    > "$D/run.log" 2>&1
  RC=$?; N=$(wc -l < "$D/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  say "$arm end rc=$RC ledger=$N complete=$(is_complete "$D" && echo yes || echo no)"
done
CMP_PY=$(python3 -c "import json;print(json.load(open('$HERE/arms/cpu_pardiso.json'))['python_bin'])")
"$CMP_PY" "$HERE/compare_solver_arms.py" --root "$OUT" --baseline cpu_pardiso --arms $ARMS 2>&1 | tee -a "$LOG"
say "SOLVER_COMPARE_DONE $OUT"
