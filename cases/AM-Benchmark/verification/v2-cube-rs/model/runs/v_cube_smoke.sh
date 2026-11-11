#!/usr/bin/env bash
# ============================================================================
# V2 cube stress reproduction -- stage 1 preflight + stage 2 reduced-height smoke
# (STRESS-REPRODUCTION-PLAN.md, IET-9). Serial, idempotent, fail-closed.
#
#   STAGE 1  preflight    mesh + path + slab activation map + time/energy ledgers
#                         + runner contract (make_v2_cube_preflight.py) + unit tests
#   STAGE 2  capture      thermal-only ONE physical layer, two heat-source
#                         readings (D-V2-10): physical r=50 um half-space vs
#                         the contract's r=200 um slab-band renormalised source.
#                         Measures the capture fraction and the cost per step.
#   STAGE 3  smoke        the full 25-layer / 5-slab sequential thermo-mechanical
#                         smoke with cooldown and substrate release, then gate.
#
# Usage:
#   bash v_cube_smoke.sh                 # all stages
#   STAGES="1 2" bash v_cube_smoke.sh    # subset
#   TAG=<name> bash v_cube_smoke.sh      # output root suffix (default: branch commit)
#
# Red lines: zero calibration, no shared-solver edits, no measurement feedback.
# ============================================================================
set -u
REPO="${REPO:-/home/user/work/d11_B_tree}"
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH="$REPO" JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

V2="$REPO/cases/AM-Benchmark/verification/v2-cube-rs"
M="$V2/model"
CFG="${CFG:-$V2/inputs/cube-stress-smoke.json}"     # CFG=.../cube-stress-production.json for stage-3 work
TAG="${TAG:-$(git -C "$REPO" rev-parse --short HEAD)}"
OUTROOT="${OUTROOT:-/home/user/work/159/output/v2_cube_smoke_${TAG}}"
PRE="$OUTROOT/preflight"
STAGES="${STAGES:-1 2 3}"
LOG="$OUTROOT/cube_smoke.log"
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
contract_argv() {  # print the runner argv from the contract, one per line
  python3 -c "
import json;print('\n'.join(json.load(open('$PRE/runner_contract.json'))['argv']))"
}

say "======== V2 cube smoke pid=$$ TAG=$TAG STAGES='$STAGES' REPO=$REPO ========"
say "branch: $(git -C "$REPO" branch --show-current) @ $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --short | wc -l)"

# ---- stage 1: preflight ----
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
    say "stage 1: unit tests for the preflight"
    python -m pytest -q "$REPO/tests/unit/test_v2_cube_preflight.py" 2>&1 | tail -3 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "preflight unit tests FAILED, stop"; exit 2; }
    say "stage 1: generating preflight artefacts -> $PRE"
    python "$M/make_v2_cube_preflight.py" --config "$CFG" --repo "$REPO" --output "$PRE" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "preflight FAILED, stop"; exit 2; }
  fi
  say "stage 1 gate: $(python3 -c "
import json;l=json.load(open('$PRE/v2_cube_smoke_ledger.json'))
print('complete=%s solver_compatible=%s mesh_check=%s rows=%d steps=%d slabs=%d energy=%.3f J' % (l['complete'], l['solver_compatible'], l['mesh_check']['ok'], l['path_rows'], l['expected_runner_steps'], l['activation_slabs'], l['nominal_laser_energy_J']))")"
fi

run_solver() {  # NAME OUT extra-args...
  local NAME=$1 OUT=$2; shift 2
  if is_complete "$OUT"; then say "$NAME already complete, skip (idempotent)"; return 0; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  mapfile -t ARGV < <(contract_argv)
  # interpreter and platform follow the contract: the GPU env (jax-fem-gpu) lives
  # under miniconda3 and cannot be `conda activate`d from miniforge's conda.sh,
  # so it is addressed by absolute path (L0 / D-11 PYTHON_BIN convention)
  local PYBIN PLATFORM
  PYBIN=$(python3 -c "import json;c=json.load(open('$PRE/runner_contract.json'));print(c.get('python_bin') or '')")
  PLATFORM=$(python3 -c "import json;c=json.load(open('$PRE/runner_contract.json'));print(c.get('platform') or 'cpu')")
  [ -n "$PYBIN" ] || PYBIN=python
  export JAX_PLATFORM_NAME="$PLATFORM"
  say "$NAME start -> $OUT (python=$PYBIN platform=$PLATFORM extra: $*)"
  "$PYBIN" -m jax_fem_am.simulation.runner "${ARGV[@]}" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" --profile-label "v2-cube-$NAME" \
    "$@" > "$OUT/run.log" 2>&1
  local RC=$? N
  N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  say "$NAME end rc=$RC ledger=$N complete=$(is_complete "$OUT" && echo yes || echo no)"
  if [ "$N" -le 5 ]; then say "$NAME died at start (ledger=$N). See $OUT/run.log"; tail -20 "$OUT/run.log" | tee -a "$LOG"; exit 2; fi
}

# ---- stage 2: capture trial (thermal-only, physical layer 1 only) ----
if has 2; then
  L1="$PRE/v2_cube_smoke_path_layer1.csv"
  if [ ! -f "$L1" ]; then
    python3 - "$PRE/v2_cube_smoke_path.csv" "$L1" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], newline=""))
        if r["physical_layer"] == "1" and r["mode"] != "recoat"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"layer-1 path: {len(rows)} rows, t_end={rows[-1]['time']} s")
PY
  fi
  COMMON=(--path-file "$L1" --mechanics-every 0 --cooling-steps 0 --thermal-output-every 100000 \
          --summary-every 100 --no-release-after-cooling)
  # Radius ladder (D-V2-10 quantification). Default = the contract radius only;
  # CAPTURE_RADII="2.0e-4 3.0e-4 4.0e-4 6.0e-4" runs the ladder. Every member
  # keeps the slab-band cutoff + renormalisation of the contract; the physical
  # r=50 um half-space reading of V1 is always run as the reference.
  CONTRACT_R=$(python3 -c "import json;a=json.load(open('$PRE/runner_contract.json'))['argv'];print(a[a.index('--beam-radius')+1])")
  RADII="${CAPTURE_RADII:-$CONTRACT_R}"
  DIRS=()
  for r in $RADII; do
    name="capture_r$(python3 -c "print(int(round(float('$r')*1e6)))")_band"
    run_solver "$name" "$OUTROOT/$name" "${COMMON[@]}" --beam-radius "$r"
    DIRS+=("$name")
  done
  if [ "${SKIP_R50:-0}" != "1" ]; then   # the physical-spot reference is meaningless for a flash contract
    run_solver capture_r50_halfspace "$OUTROOT/capture_r50_halfspace" "${COMMON[@]}" \
      --beam-radius 5.0e-5 --source-depth-cutoff 0 --no-source-cutoff-renormalize
    DIRS+=(capture_r50_halfspace)
  fi
  for d in "${DIRS[@]}"; do
    say "capture gate [$d]:"
    python "$M/check_cube_smoke.py" --run "$OUTROOT/$d" --preflight "$PRE" --capture-only 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  done
  python3 - "$OUTROOT" "${DIRS[@]}" <<'PY' | tee -a "$LOG"
import json, os, sys, re
root, dirs = sys.argv[1], sys.argv[2:]
rows = []
for d in dirs:
    g = json.load(open(os.path.join(root, d, "cube_smoke_gate.json")))
    cfg = json.load(open(os.path.join(root, d, "used_config.json")))
    e, p, s = g["energy"], g["profile"], g["log"]["last_summary"]
    rows.append({"run": d, "beam_radius_m": cfg.get("beam_radius"),
                 "source_depth_cutoff_m": cfg.get("source_depth_cutoff"),
                 "renormalize": cfg.get("source_cutoff_renormalize"),
                 "capture_fraction": e["capture_fraction"],
                 "per_step_capture_min": e["per_step_capture_min"],
                 "per_step_capture_max": e["per_step_capture_max"],
                 "per_step_capture_std": e["per_step_capture_std"],
                 "T_max_last_K": float(s[10]) if s else None,
                 "seconds_per_step": p.get("seconds_per_step"),
                 "max_relative_balance_error": e["max_relative_balance_error"]})
out = os.path.join(root, "capture_ladder.json")
json.dump({"schema": "v2.cube-capture-ladder/1", "members": rows}, open(out, "w"), indent=2)
print("capture ladder -> " + out)
for r in rows:
    print(f"  {r['run']:24s} r={r['beam_radius_m']:.1e} capture={r['capture_fraction']:.4f} "
          f"step[min,max,std]=[{r['per_step_capture_min']:.3f},{r['per_step_capture_max']:.3f},{r['per_step_capture_std']:.3f}] "
          f"Tmax={r['T_max_last_K']:.0f}K {r['seconds_per_step']:.3f}s/step")
PY
fi

# ---- stage S: shakedown truncated to SHAKEDOWN_SLABS slabs (production mesh cost/behaviour) ----
if has S; then
  NS="${SHAKEDOWN_SLABS:-2}"
  SD="$OUTROOT/shakedown_${NS}slabs"
  run_solver "shakedown_${NS}slabs" "$SD" --max-print-layers "$NS" \
    --cooling-steps "${SHAKEDOWN_COOLING_STEPS:-6}" --cooling-dt "${SHAKEDOWN_COOLING_DT:-100}"
  say "shakedown gate (slabs <= $NS):"
  python "$M/check_cube_smoke.py" --run "$SD" --preflight "$PRE" --max-slabs "$NS" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  say "SHAKEDOWN_GATE_RC=${PIPESTATUS[0]}"
  python3 - "$SD/profile.json" "$SD/run.log" "$NS" <<'PY' | tee -a "$LOG"
import json, re, sys
p = json.load(open(sys.argv[1])); ss = p["stage_seconds"]; sc = p["stage_calls"]
text = open(sys.argv[2], encoding="utf-8", errors="replace").read()
mech = len(set(re.findall(r"mechanics_current=1 mechanics_source_step=(\d+)", text)))
steps = p["steps"]; ns = int(sys.argv[3]); layers = 5 * ns
print(f"shakedown: {steps} steps, wall {p['wall_seconds']:.0f} s ({p['wall_seconds']/steps:.2f} s/step); "
      f"assembly {ss.get('assembly',0):.0f} s, solver {ss.get('solver',0):.0f} s, newton {p['meta'].get('newton_wall_seconds',0):.0f} s; "
      f"nonlinear solves {sc.get('nonlinear_solve')}, mechanics solves seen in summaries >= {mech}")
per_layer = p["wall_seconds"] / layers
print(f"per physical layer {per_layer:.0f} s -> 250 layers ~ {250*per_layer/3600:.1f} h (+ cooldown/release); "
      f"NOTE: early slabs are cheaper than late ones (matrix grows with printed volume)")
PY
fi

# ---- stage 3: reduced-height sequential thermo-mechanical smoke ----
if has 3; then
  run_solver smoke "$OUTROOT/smoke"
  say "smoke gate:"
  python "$M/check_cube_smoke.py" --run "$OUTROOT/smoke" --preflight "$PRE" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  say "SMOKE_GATE_RC=${PIPESTATUS[0]}"
fi
say "CUBE_SMOKE_DONE $OUTROOT"
