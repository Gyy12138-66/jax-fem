#!/usr/bin/env bash
# ============================================================================
# 采纳 gate(Kimi 2026-08-06):graded 网格**全程跑通**,而不只是"越过失败点"。
#
# 现有证据只到 v2_trio_ecol 在 2400 s 上限处 ledger=101、Newton 零失败 —— 那是
# "越过了 frac=0.01 原本确定性死掉的 ledger=40",不是跑完 240+40 步。本脚本用
# 采纳口径(v2_material_config.json,已指向 E_collapse + flow_curve_adopted)
# 在 graded 上**不设 timeout** 跑到底。
#
# 预计 graded 速率约为 coarse 的 1/3,240+40 步约 2.5-3 小时。串行、幂等。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=6

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUT=/home/user/work/159/output/v2_gate_graded_adopted
cd /home/user/work/159
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$VT/graded_gate.log"; }

is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$OUT','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}
if is_complete; then say "已完成,跳过(幂等)"; say "GRADED_GATE_DONE"; exit 0; fi

say "graded 全程 gate 开始(采纳口径,无 timeout)"
rm -rf "$OUT"; mkdir -p "$OUT"
python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
  --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
python -m jax_fem_am.simulation.runner \
  --config "$M/v2_material_config.json" \
  --inp "$M/v2_multitrack_c3d8.inp" \
  --output-dir "$OUT" --profile-json "$OUT/profile.json" \
  --profile-label "gate-graded-adopted" \
  --xla-platform cpu --xla-preallocate off --xla-linear-solver pardiso \
  --xla-pardiso-mode phase23 \
  --build-axis z --base-side min --layer-thickness 4.0e-5 --layers 1 \
  --support-thickness 4.0e-4 --path-file "$OUT/path.csv" --path-length-scale 1.0 \
  --source-model legacy --beam-radius 5.0e-5 --source-depth 1.0e-4 \
  --laser-power 50 --dt 1.9e-5 \
  --layer-activation-mode layer_on_scan --layer-activation-geometry intersection \
  --future-layer-mode void --active-window-below-layers 0 --inactive-mass-factor 1.0 \
  --powder-mode powder --surface-selection exterior --boundary-tol 1.0e-6 \
  --quadrature-order 2 --ambient 313.0 --preheat-temperature 353.15 \
  --bottom-thermal-bc fixed --bottom-temperature 353.15 \
  --stress-relaxation-temperature 0 \
  --cooling-steps 40 --cooling-dt 0.02 --final-cooldown-temperature 353.15 \
  --mechanics-model j2_plastic --bottom-mechanics-bc fixed \
  --mechanics-every 20 --mechanics-rel-tol 5e-5 --mechanics-acceptance abaqus \
  --mechanics-temperature-floor 293.15 --thermal-mass-lumping \
  --thermal-output-every 100 --mechanics-output-every 20 --summary-every 2 \
  --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
  --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3 \
  > "$OUT/run.log" 2>&1
RC=$?
N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
NF=$(grep -c 'did not converge' "$OUT/run.log" 2>/dev/null | tr -d '\n')
if is_complete; then V=COMPLETE; else V=INCOMPLETE; fi
say "结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
say "GRADED_GATE_DONE"
