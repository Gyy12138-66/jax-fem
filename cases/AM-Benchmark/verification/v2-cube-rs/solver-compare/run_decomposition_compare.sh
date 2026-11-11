#!/usr/bin/env bash
# Repeated CPU/GPU full-Newton versus modified-Newton comparison.
# Safe default: with no --run argument this script only prints the plan.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../../.." && pwd)}"
PRE="${PRE:-/home/user/work/159/output/v2_cube_smoke_smoke2/preflight}"
OUT="${OUT:-/home/user/work/159/output/v2_cube_smoke_smoke2/decomposition_compare}"
STEPS="${STEPS:-16}"
REPEATS="${REPEATS:-4}"
WARMUP="${WARMUP:-1}"
ARM_TIMEOUT="${ARM_TIMEOUT:-}"
MKL_THREADS="${MKL_NUM_THREADS:-8}"
MECHANICS_EVERY_OVERRIDE="${MECHANICS_EVERY:-}"
ARMS="${ARMS:-cpu_full_phase23 cpu_modified_phase33 gpu_full_phase23 gpu_modified_phase33}"
RUN_MODE=""

usage() {
  cat <<'EOF'
Usage:
  bash run_decomposition_compare.sh                 # print plan only
  bash run_decomposition_compare.sh --run quick     # first STEPS=16 rows
  bash run_decomposition_compare.sh --run full      # all 639 layer-1 rows

Environment overrides: PRE OUT STEPS REPEATS WARMUP ARM_TIMEOUT
                       MKL_NUM_THREADS MECHANICS_EVERY ARMS REPO

Quick mode defaults to MECHANICS_EVERY=1 as a solver stress test. Full mode
defaults to the cadence already frozen in runner_contract.json; setting
MECHANICS_EVERY explicitly changes either mode into a cadence override test.

Every arm is a separate process and all arms run serially. Warmups are stored
under warmup-* and excluded from comparison. Existing incomplete run folders
are never deleted or overwritten; choose a new OUT to restart them.
Measured repeats alternate forward/reverse arm order to reduce order bias.
EOF
}

while (($#)); do
  case "$1" in
    --run)
      (($# >= 2)) || { echo "--run requires quick or full" >&2; exit 2; }
      RUN_MODE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$RUN_MODE" in
  ""|quick|full) ;;
  *) echo "--run must be quick or full" >&2; exit 2 ;;
esac
if [[ -z "$ARM_TIMEOUT" ]]; then
  if [[ "$RUN_MODE" == full ]]; then ARM_TIMEOUT=7200; else ARM_TIMEOUT=600; fi
fi
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be >= 1" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be >= 1" >&2; exit 2; }
[[ "$WARMUP" =~ ^[1-9][0-9]*$ ]] || { echo "WARMUP must be >= 1" >&2; exit 2; }
if [[ -n "$MECHANICS_EVERY_OVERRIDE" ]]; then
  [[ "$MECHANICS_EVERY_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || {
    echo "MECHANICS_EVERY must be >= 1 when supplied" >&2
    exit 2
  }
  CADENCE_LABEL="override mechanics-every=$MECHANICS_EVERY_OVERRIDE"
elif [[ "$RUN_MODE" == full ]]; then
  CADENCE_LABEL="frozen runner contract"
else
  MECHANICS_EVERY_OVERRIDE=1
  CADENCE_LABEL="quick stress mechanics-every=1"
fi

LAYER1="$PRE/v2_cube_smoke_path_layer1.csv"
CONTRACT="$PRE/runner_contract.json"
MODE_LABEL="${RUN_MODE:-plan}"
MODE_ROOT="$OUT/$MODE_LABEL"

echo "Thermo-mechanical decomposition comparison"
echo "  mode:       $MODE_LABEL${RUN_MODE:+ (explicit execution enabled)}"
echo "  repo:       $REPO"
echo "  preflight:  $PRE"
echo "  output:     $MODE_ROOT"
echo "  arms:       $ARMS"
echo "  warmups:    $WARMUP per arm (excluded)"
echo "  repeats:    $REPEATS per arm (median reported)"
echo "  quick rows: $STEPS"
echo "  arm timeout: ${ARM_TIMEOUT}s"
echo "  cadence:    $CADENCE_LABEL"
echo "  execution:  separate CPU/GPU processes, serial"
echo "  frozen dir: $HERE/results (never written)"

if [[ -z "$RUN_MODE" ]]; then
  echo
  echo "Plan only. Use --run quick or --run full to execute."
  exit 0
fi

[[ -f "$LAYER1" ]] || { echo "missing layer-1 path: $LAYER1" >&2; exit 2; }
[[ -f "$CONTRACT" ]] || { echo "missing runner contract: $CONTRACT" >&2; exit 2; }

mkdir -p "$MODE_ROOT"
if [[ "$RUN_MODE" == quick ]]; then
  PATH_FILE="$MODE_ROOT/inputs/layer1_first_${STEPS}.csv"
  python3 "$HERE/make_quick_path.py" \
    --input "$LAYER1" --output "$PATH_FILE" --steps "$STEPS"
else
  PATH_FILE="$LAYER1"
fi

mapfile -t CONTRACT_ARGV < <(
  python3 -c "import json; print('\\n'.join(json.load(open('$CONTRACT'))['argv']))"
)
COMMON=(
  --path-file "$PATH_FILE"
  --cooling-steps 0
  --no-release-after-cooling
  --thermal-output-every 100000
  --mechanics-output-every 100000
  --summary-every 1
  --xla-preallocate off
)
if [[ -n "$MECHANICS_EVERY_OVERRIDE" ]]; then
  COMMON+=(--mechanics-every "$MECHANICS_EVERY_OVERRIDE")
fi
read -r -a ARM_ARRAY <<< "$ARMS"
MANIFEST_CADENCE="${MECHANICS_EVERY_OVERRIDE:-contract}"
python3 "$HERE/prepare_experiment_manifest.py" \
  --root "$MODE_ROOT" --repo "$REPO" --mode "$RUN_MODE" \
  --path-file "$PATH_FILE" --contract "$CONTRACT" \
  --mechanics-every "$MANIFEST_CADENCE" --threads "$MKL_THREADS" \
  --warmups "$WARMUP" --repeats "$REPEATS" \
  --arms "${ARM_ARRAY[@]}"

export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS="$MKL_THREADS"
export OMP_NUM_THREADS="$MKL_THREADS"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
is_complete() {
  python3 - "$1" "$2" "$MODE_ROOT/experiment_manifest.json" <<'PY' 2>/dev/null
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
arm = sys.argv[2]
manifest = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
path = root / "thermal_energy_ledger_summary.json"
try:
    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    saved_arm = json.loads((root / "arm.json").read_text(encoding="utf-8"))
    complete = (
        json.loads(path.read_text(encoding="utf-8")).get("complete") is True
        and profile.get("steps") == manifest["expected_steps"]
        and saved_arm == manifest["arm_specs"][arm]
        and (root / "used_config.json").is_file()
        and any(root.glob("step_*.vtu"))
    )
except Exception:
    complete = False
raise SystemExit(0 if complete else 1)
PY
}

run_one() {
  local group="$1"
  local arm="$2"
  local destination="$MODE_ROOT/$group/$arm"
  local spec="$HERE/decomposition-arms/$arm.json"
  [[ -f "$spec" ]] || { echo "missing arm spec: $spec" >&2; return 2; }

  local pybin platform env_name
  pybin=$(python3 -c "import json; print(json.load(open('$spec'))['python_bin'])")
  platform=$(python3 -c "import json; print(json.load(open('$spec'))['platform'])")
  env_name=$(python3 -c "import json; print(json.load(open('$spec'))['conda_env'])")
  [[ -x "$pybin" ]] || { echo "$arm python is not executable: $pybin" >&2; return 2; }
  mapfile -t extra < <(
    python3 -c "import json; print('\\n'.join(json.load(open('$spec'))['extra_argv']))"
  )

  if is_complete "$destination" "$arm"; then
    echo "[$(timestamp)] $group/$arm complete; skip"
    return 0
  fi
  if [[ -d "$destination" ]] && find "$destination" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite incomplete output: $destination" >&2
    echo "choose a new OUT or move the incomplete directory aside" >&2
    return 2
  fi
  mkdir -p "$destination"
  cp "$spec" "$destination/arm.json"

  export JAX_PLATFORM_NAME="$platform"
  export JAX_COMPILATION_CACHE_DIR="$MODE_ROOT/jax-cache/$platform"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  echo "[$(timestamp)] $group/$arm start env=$env_name platform=$platform"
  set +e
  timeout "$ARM_TIMEOUT" "$pybin" -m jax_fem_am.simulation.runner \
    "${CONTRACT_ARGV[@]}" "${COMMON[@]}" --xla-platform "$platform" \
    "${extra[@]}" --output-dir "$destination" \
    --profile-json "$destination/profile.json" \
    --profile-label "decomposition-$RUN_MODE-$group-$arm" \
    > "$destination/run.log" 2>&1
  local rc=$?
  set -e
  echo "[$(timestamp)] $group/$arm end rc=$rc complete=$(is_complete "$destination" "$arm" && echo yes || echo no)"
  if ((rc != 0)); then
    tail -30 "$destination/run.log" || true
    return "$rc"
  fi
  is_complete "$destination" "$arm" || {
    echo "$group/$arm exited without complete, identity-matched evidence" >&2
    return 2
  }
}

for ((index = 1; index <= WARMUP; index++)); do
  printf -v warmup '%02d' "$index"
  for arm in "${ARM_ARRAY[@]}"; do
    run_one "warmup-$warmup" "$arm"
  done
done
for ((index = 1; index <= REPEATS; index++)); do
  printf -v repeat '%02d' "$index"
  if ((index % 2 == 1)); then
    for arm in "${ARM_ARRAY[@]}"; do
      run_one "repeat-$repeat" "$arm"
    done
  else
    for ((arm_index = ${#ARM_ARRAY[@]} - 1; arm_index >= 0; arm_index--)); do
      run_one "repeat-$repeat" "${ARM_ARRAY[$arm_index]}"
    done
  fi
done

COMPARE_PY=$(python3 -c "import json; print(json.load(open('$HERE/decomposition-arms/cpu_full_phase23.json'))['python_bin'])")
"$COMPARE_PY" "$HERE/compare_decomposition_arms.py" \
  --root "$MODE_ROOT" --baseline cpu_full_phase23 --arms "${ARM_ARRAY[@]}"
echo "[$(timestamp)] DECOMPOSITION_COMPARE_DONE $MODE_ROOT"
