#!/usr/bin/env bash
# ============================================================================
# D-V2-07 fine 变体 gate(Kimi 修正版实现单第 3 项,2026-08-07)
#
# 与 runs/v_graded_gate.sh 同构,只换网格(fine)并加三件事:
#   * 串行门:等主线 T_cut 带的完成标记,再启动。Kimi 点 3 的实测依据是
#     fine NEED 24 GB(coarse 30k 单元实测 RSS 5.6 GB,fine 110k 单元约 3.7 倍),
#     以及 2026-08-03 的 OOM 教训(fine/graded 是 18.8 GB OOM 的来源)。
#   * 5 分钟冒烟 RSS 水印:头 5 分钟每 10 s 采一次 RSS,记录峰值。既验证内存
#     预算,也让"起步即死"在 5 分钟内暴露,而不是几小时后。
#   * 起步即死护栏:台账 <= 12 行即判定为配置层面的死,停下报告。
#
# MKL_NUM_THREADS=6,与 graded gate 相同 —— 少一个变量(Kimi 点 3)。
# 无 timeout、幂等(complete 即跳过)。不由本脚本决定何时开跑:启动归 Fable5。
#
# 标记路径可覆盖:
#   WAIT_MARKER=/path/to/marker bash v_fine_gate.sh
#   NO_WAIT=1 bash v_fine_gate.sh        # 跳过串行门(仅当确认 GPU 任务已停)
# 注意:截至交付时,仓库/vtmp 里**没有任何脚本会创建 B_tcut.done**。
# 现有先例是 d11_B/B.done(offline_chain_20260806.sh 轮询它)。所以要么由 T_cut
# 带结束时 touch 这个文件,要么启动时用 WAIT_MARKER= 指到真实存在的标记。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=6

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUT=/home/user/work/159/output/v2_gate_fine_adopted
LOG=$VT/fine_gate.log
WAIT_MARKER=${WAIT_MARKER:-/home/user/work/output/d11_B/B_tcut.done}
POLL=${POLL:-300}
CAP_SECONDS=${CAP_SECONDS:-43200}      # 12 h,与 offline_chain 的口径一致
SMOKE_SECONDS=${SMOKE_SECONDS:-300}
NEED_GB=${NEED_GB:-24}

cd /home/user/work/159
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$OUT','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}

if is_complete; then say "已完成,跳过(幂等)"; say "FINE_GATE_DONE"; exit 0; fi

# ---- 串行门 --------------------------------------------------------------
if [ "${NO_WAIT:-0}" = "1" ]; then
  say "NO_WAIT=1,跳过串行门(调用方已确认 GPU 任务停止)"
else
  say "串行门:等待标记 $WAIT_MARKER(每 ${POLL}s 轮询,上限 ${CAP_SECONDS}s)"
  waited=0
  while [ ! -f "$WAIT_MARKER" ]; do
    if [ "$waited" -ge "$CAP_SECONDS" ]; then
      say "等待超过上限仍未见标记。**不强行启动**(fine 需 ${NEED_GB} GB,与 GPU 任务并行会 OOM)。"
      say "FINE_GATE_ABORTED_WAIT_TIMEOUT"
      exit 3
    fi
    sleep "$POLL"; waited=$((waited + POLL))
  done
  say "标记出现,已等待 ${waited}s;沉降 120 s 让上一任务释放内存"
  sleep 120
fi

# ---- 内存预算复核(不是估算,是启动前实测)-------------------------------
AVAIL_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
say "启动前可用内存 ${AVAIL_GB} GB,fine 需要约 ${NEED_GB} GB"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  say "可用内存不足,拒绝启动(宁可不跑,也不要 OOM 掉别人的任务)。"
  say "FINE_GATE_ABORTED_LOW_MEMORY"
  exit 4
fi

say "fine gate 开始:110000 单元 / 122412 节点 / 367236 力学 DOF,采纳口径,无 timeout"
rm -rf "$OUT"; mkdir -p "$OUT"
python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
  --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1

python -m jax_fem_am.simulation.runner \
  --config "$M/v2_material_config.json" \
  --inp "$M/v2_multitrack_c3d8_fine.inp" \
  --output-dir "$OUT" --profile-json "$OUT/profile.json" \
  --profile-label "gate-fine-adopted" \
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
  > "$OUT/run.log" 2>&1 &
RUNNER_PID=$!
say "runner pid=$RUNNER_PID"

# ---- 5 分钟冒烟 + RSS 水印 -----------------------------------------------
PEAK_KB=0; t=0
while [ "$t" -lt "$SMOKE_SECONDS" ]; do
  kill -0 "$RUNNER_PID" 2>/dev/null || break
  RSS_KB=$(ps -o rss= -p "$RUNNER_PID" 2>/dev/null | tr -d ' ')
  [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt "$PEAK_KB" ] && PEAK_KB=$RSS_KB
  sleep 10; t=$((t + 10))
done
say "冒烟窗口(${SMOKE_SECONDS}s)RSS 水印 = $(awk -v k="$PEAK_KB" 'BEGIN{printf "%.2f", k/1048576}') GB"
echo "$PEAK_KB" > "$OUT/rss_watermark_kb.txt"

if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
  LEDGER=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  if [ "$LEDGER" -le 12 ]; then
    say "起步即死(冒烟窗口内退出,ledger=$LEDGER)。停下,等人工判读。"
    say "FINE_GATE_ABORTED_EARLY_DEATH"
    exit 2
  fi
fi

wait "$RUNNER_PID"; RC=$?
N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
NF=$(grep -c 'did not converge' "$OUT/run.log" 2>/dev/null | tr -d '\n')
if is_complete; then V=COMPLETE; else V=INCOMPLETE; fi
say "结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
if [ "$V" = INCOMPLETE ] && [ "$N" -le 12 ]; then
  say "起步即死(ledger=$N)"; say "FINE_GATE_ABORTED_EARLY_DEATH"; exit 2
fi
say "FINE_GATE_DONE"
