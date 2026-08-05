#!/usr/bin/env bash
# ============================================================================
# 扫描 A:D-V2-17 方案 (c) 的地板 / 坍塌起点敏感度(Fable5 2026-08-05 锁定规格)
#
#   地板扫描  floor-frac ∈ {0.005, 0.01, 0.02} × collapse-frac 0.8
#   起点扫描  collapse-frac ∈ {0.7, 0.9}       × floor-frac 0.01
#   中心点 (0.01, 0.8) 一并重跑,使 5 个成员出自同一 harness、同一驱动。
#
# 串行(Fable5 指定:IET-7 的 B 三连占 GPU + 22 线程)。MKL_NUM_THREADS=6。
# harness 与 v_ecol_coarse.sh(判据③那次)逐字相同,只换 --config,
# 因此 5 个成员之间、以及与三件套验证之间都可逐单元相减。
#
# **幂等**:台账 summary complete==true 的成员直接跳过。本机今天已重启两次,
# 5-10 小时的串行必须能被重启打断后接着跑,而不是从头再来。
# 重启后只需再执行一次本脚本即可续跑。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=6

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/sweepA.log
RES=$VT/sweepA_results.txt
cd /home/user/work/159

TAGS="f0005_c08 f001_c08 f002_c08 f001_c07 f001_c09"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}

run_member() {
  local TAG=$1
  local OUT=$OUTROOT/v2_sweepA_$TAG
  if is_complete "$OUT"; then
    say "$TAG 已完成,跳过(幂等)"
    return 0
  fi
  say "$TAG 开始"
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
  python -m jax_fem_am.simulation.runner \
    --config "$M/v2_material_config_fc_ecol_${TAG}.json" \
    --inp "$M/v2_multitrack_c3d8_coarse.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "sweepA-$TAG" \
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
  local RC=$?
  local N; N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  local NF; NF=$(grep -c 'did not converge' "$OUT/run.log" 2>/dev/null | tr -d '\n')
  local V; if is_complete "$OUT"; then V=COMPLETE; else V=INCOMPLETE; fi
  printf '%-11s %-11s ledger=%-6s newton_nonconv=%-4s rc=%s\n' \
    "$TAG" "$V" "$N" "$NF" "$RC" >> "$RES"
  say "$TAG 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"

  # 起步即死的护栏(Fable5 要求):台账 <= 12 行说明连失败臂的老停点都没过,
  # 那是配置层面的问题,不是收敛的问题 —— 停下,不要浪费后面几个小时。
  if [ "$V" = INCOMPLETE ] && [ "$N" -le 12 ]; then
    say "$TAG 起步即死(ledger=$N)。按规格停止扫描,等人工判读。"
    say "SWEEPA_ABORTED"
    return 2
  fi
  return 0
}

: > "$RES"
say "======== 扫描 A 启动 pid=$$ (串行, MKL_NUM_THREADS=6) ========"
say "成员: $TAGS"
for TAG in $TAGS; do
  run_member "$TAG" || { say "扫描中止于 $TAG"; exit 2; }
done
say "结果表:"
cat "$RES" | tee -a "$LOG"
say "SWEEPA_DONE"
