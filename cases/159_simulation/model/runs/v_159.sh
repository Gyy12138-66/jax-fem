#!/usr/bin/env bash
# ============================================================================
# 159 (0119 part) -- preflight + energy trial + shakedown + production, serial,
# idempotent, fail-closed. Same contract and stages as v2-cube-rs/model/runs/v_cube_smoke.sh.
#
#   STAGE 1  preflight   per-slab areas from the HEX8 mesh, flash schedule, ledgers, runner contract
#   STAGE 2  energy      thermal-only, physical layer 1 only (flash + hold rows): capture / closure gate
#   STAGE S  shakedown   SHAKEDOWN_SLABS slabs with mechanics, short cooldown, release
#   STAGE 3  production  the full build + 600 s cooldown + raft release, then gate
#
#   STAGES="1 2" bash v_159.sh ; CFG=... TAG=... OUTROOT=... as in the cube script
# ============================================================================
set -u
REPO="${REPO:-/home/user/work/d11_B_tree}"
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH="$REPO" JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CASE="$REPO/cases/159_simulation"
M="$CASE/model"
CFG="${CFG:-$CASE/inputs/0119-flash.json}"
TAG="${TAG:-$(git -C "$REPO" rev-parse --short HEAD)}"
OUTROOT="${OUTROOT:-/home/user/work/159/output/v159_${TAG}}"
PRE="$OUTROOT/preflight"
STAGES="${STAGES:-1 2}"
LOG="$OUTROOT/v159.log"
mkdir -p "$OUTROOT"
cd "$REPO"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
has() { [ "${STAGES#*$1}" != "$STAGES" ]; }
is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}
contract_argv() { python3 -c "import json;print('\n'.join(json.load(open('$PRE/runner_contract.json'))['argv']))"; }

say "======== 159 pid=$$ TAG=$TAG STAGES='$STAGES' CFG=$CFG ========"
say "branch: $(git -C "$REPO" branch --show-current) @ $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --short | wc -l)"

if has 1; then
  if [ -f "$PRE/v2_cube_smoke_ledger.json" ] && [ -f "$PRE/runner_contract.json" ]; then
    say "stage 1: preflight artefacts exist, re-validating fingerprints"
    python3 - "$PRE" <<'PY' || { echo "preflight fingerprints stale -> regenerate"; rm -rf "$PRE"; }
import json, sys, hashlib, os
pre = sys.argv[1]
led = json.load(open(os.path.join(pre, "v2_cube_smoke_ledger.json")))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ok = all(sha(led[k]) == led[k + "_sha256"] for k in ("config", "mesh", "path", "material_config"))
sys.exit(0 if ok else 1)
PY
  fi
  if [ ! -f "$PRE/runner_contract.json" ]; then
    say "stage 1: generating preflight artefacts -> $PRE"
    python "$M/make_159_preflight.py" --config "$CFG" --repo "$REPO" --output "$PRE" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "preflight FAILED, stop"; exit 2; }
  fi
  say "stage 1 gate: $(python3 -c "
import json;l=json.load(open('$PRE/0119_ledger_summary.json'))
print('complete=%s rows=%d steps=%d slabs=%d layers=%d energy=%.1f J build_clock=%.0f s' % (l['complete'], l['path_rows'], l['expected_runner_steps'], l['activation_slabs'], l['physical_layers'], l['nominal_laser_energy_J'], l['build_clock_s']))")"
fi

run_solver() {  # NAME OUT extra-args...
  local NAME=$1 OUT=$2; shift 2
  if is_complete "$OUT"; then say "$NAME already complete, skip (idempotent)"; return 0; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  mapfile -t ARGV < <(contract_argv)
  local PYBIN PLATFORM
  PYBIN=$(python3 -c "import json;c=json.load(open('$PRE/runner_contract.json'));print(c.get('python_bin') or '')")
  PLATFORM=$(python3 -c "import json;c=json.load(open('$PRE/runner_contract.json'));print(c.get('platform') or 'cpu')")
  [ -n "$PYBIN" ] || PYBIN=python
  export JAX_PLATFORM_NAME="$PLATFORM"
  say "$NAME start -> $OUT (python=$PYBIN platform=$PLATFORM extra: $*)"
  "$PYBIN" -m jax_fem_am.simulation.runner "${ARGV[@]}" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" --profile-label "v159-$NAME" \
    "$@" > "$OUT/run.log" 2>&1
  local RC=$? N
  N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  say "$NAME end rc=$RC ledger=$N complete=$(is_complete "$OUT" && echo yes || echo no)"
  if [ "$N" -le 5 ]; then say "$NAME died at start (ledger=$N). See $OUT/run.log"; tail -20 "$OUT/run.log" | tee -a "$LOG"; exit 2; fi
}

if has 2; then
  L1="$PRE/0119_path_layer1.csv"
  if [ ! -f "$L1" ]; then
    python3 - "$PRE/0119_path.csv" "$L1" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], newline="")) if r["physical_layer"] == "1" and r["mode"] != "recoat"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"layer-1 path: {len(rows)} rows, t_end={rows[-1]['time']} s")
PY
  fi
  run_solver energy_layer1 "$OUTROOT/energy_layer1" --path-file "$L1" --mechanics-every 0 --cooling-steps 0 \
    --thermal-output-every 100000 --summary-every 2 --no-release-after-cooling
  say "energy gate:"
  python "$M/check_159.py" --run "$OUTROOT/energy_layer1" --preflight "$PRE" --capture-only 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  say "ENERGY_GATE_RC=${PIPESTATUS[0]}"
  python3 - "$OUTROOT/energy_layer1" "$PRE" <<'PY' | tee -a "$LOG"
import json, sys, os
run, pre = sys.argv[1], sys.argv[2]
g = json.load(open(os.path.join(run, "cube_smoke_gate.json")))
led = json.load(open(os.path.join(pre, "0119_ledger_summary.json")))
e = g["energy"]; f = e["flash"]; s = g["log"]["last_summary"]; p = g["profile"]
print(f"energy trial: deposited {e['laser_deposited_J']:.4f} J vs physical absorbed {f['physical_absorbed_J']:.4f} J "
      f"(ratio {f['deposited_over_physical_absorbed']:.6f}); capture {e['capture_fraction']:.6e} vs mesh {f['capture_fraction_analytic']:.6e} "
      f"(ratio {f['capture_fraction_ledger_over_analytic']:.6f}); balance err max {e['max_relative_balance_error']:.2e}; "
      f"T_max {float(s[10]) if s else float('nan'):.0f} K; {p.get('seconds_per_step', float('nan')):.2f} s/step over {p.get('steps')} steps")
PY
fi

if has S; then
  NS="${SHAKEDOWN_SLABS:-2}"
  SD="$OUTROOT/shakedown_${NS}slabs"
  run_solver "shakedown_${NS}slabs" "$SD" --max-print-layers "$NS" \
    --cooling-steps "${SHAKEDOWN_COOLING_STEPS:-6}" --cooling-dt "${SHAKEDOWN_COOLING_DT:-100}"
  say "shakedown gate (slabs <= $NS):"
  python "$M/check_159.py" --run "$SD" --preflight "$PRE" --max-slabs "$NS" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  say "SHAKEDOWN_GATE_RC=${PIPESTATUS[0]}"
fi

if has 3; then
  run_solver production "$OUTROOT/production"
  say "production gate:"
  python "$M/check_159.py" --run "$OUTROOT/production" --preflight "$PRE" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  say "PRODUCTION_GATE_RC=${PIPESTATUS[0]}"
fi
say "V159_DONE $OUTROOT"
