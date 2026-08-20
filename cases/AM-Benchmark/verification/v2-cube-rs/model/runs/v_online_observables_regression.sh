#!/usr/bin/env bash
# ============================================================================
# 在线观测量记录器的**行为保持性**回归(IET-20 交付件 3 的红线证据)
#
# 红线:不带 `--online-observables` 时,行为与 HEAD 逐字节相同。
# 验法沿用仓库既有的金标做法(GOLDEN_EQUIVALENCE.txt):同一配置跑两遍,
# 对求解产物**逐字节** cmp。这里跑三遍:
#
#   A  HEAD(9fecd40)代码,不带新旗标           <- 基线
#   B  本分支代码,不带新旗标                   <- 必须与 A 逐字节相同
#   C  本分支代码,**带** --online-observables   <- 求解产物仍须与 A 逐字节相同,
#                                                 且额外产出 online_observables.jsonl
#
# C 是比红线更强的主张:记录器开着也不动物理。
#
# 逐字节比的是全部求解产物:每一个 *.vtu、thermal_energy_ledger.jsonl、
# thermal_energy_ledger_summary.json、path_used.csv。
# used_config.json **不**参与逐字节比 —— 新旗标本来就该出现在配置回显里,
# 藏起来才是问题;它的差异单独列出来给人看。
#
# 规模:1 道 + 5 步冷却 ≈ 83 个求解步,单遍约 5 分钟(与生产同一张 M1 网格,
# 所以走的是真实代码路径,不是玩具网格)。
#
# 用法: bash v_online_observables_regression.sh
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
# 确定性:金标裁决(2026-07-22)要求单线程,多线程 MKL 舍入噪声会自己制造差异
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/.." && pwd)
NEW_REPO=$(cd "$M/../../../../.." && pwd)
HEAD_REPO=${HEAD_REPO:-/home/user/work/159/iet20_head_worktree}
BASE_REF=${BASE_REF:-9fecd40669b8ca23dc6b88bc13a1c971a3e42c59}
OUTROOT=${OUTROOT:-/home/user/work/159/output}
WORK=${WORK:-/home/user/work/159/vtmp/iet20}
mkdir -p "$WORK"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

say "======== 在线观测量:行为保持性回归 ========"
say "  新代码 : $NEW_REPO"
say "  HEAD   : $HEAD_REPO @ $BASE_REF"

if [ ! -d "$HEAD_REPO/jax_fem_am" ]; then
  say "HEAD 工作树不存在,请先创建:git worktree add $HEAD_REPO $BASE_REF"
  exit 2
fi

# 三遍共用的最小算例:1 道,与生产同一张 M1 网格,同一套物理配置。
MESH=$M/v2_multitrack_c3d8.inp
CFG=$M/v2_material_config_thermal_asis.json

run_case() {
  local TAG=$1 REPO=$2 OUT=$OUTROOT/iet20_reg_$1; shift 2
  rm -rf "$OUT"; mkdir -p "$OUT"
  say "---- $TAG (repo=$REPO) ----"
  PYTHONPATH=$REPO python "$M/make_v2_path_multitrack.py" --tracks 1 \
    --power 220 --speed 0.650 --hatch 0.12e-3 \
    --output "$OUT/path.csv" --ledger-json "$OUT/path_ledger.json" \
    > "$OUT/path.log" 2>&1 || { say "$TAG 造路径失败"; exit 2; }
  PYTHONPATH=$REPO python -m jax_fem_am.simulation.runner \
    --config "$CFG" --inp "$MESH" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "iet20-regression-$TAG" \
    --xla-platform cpu --xla-preallocate off --xla-linear-solver pardiso \
    --xla-pardiso-mode phase23 \
    --build-axis z --base-side min --layer-thickness 4.0e-5 --layers 1 \
    --support-thickness 4.0e-4 --path-file "$OUT/path.csv" --path-length-scale 1.0 \
    --source-model legacy --beam-radius 5.0e-5 --source-depth 1.0e-4 \
    --laser-power 220 --dt 7.6923e-5 \
    --layer-activation-mode layer_on_scan --layer-activation-geometry intersection \
    --future-layer-mode void --active-window-below-layers 0 --inactive-mass-factor 1.0 \
    --powder-mode powder --surface-selection exterior --boundary-tol 1.0e-6 \
    --quadrature-order 2 --ambient 313.0 --preheat-temperature 353.15 \
    --bottom-thermal-bc fixed --bottom-temperature 353.15 \
    --cooling-steps 5 --cooling-dt 0.01 \
    --mechanics-every 0 \
    --thermal-mass-lumping --thermal-output-every 10 --summary-every 200 \
    --phase-history-model paper_irreversible \
    "$@" > "$OUT/run.log" 2>&1
  local RC=$?
  say "  $TAG rc=$RC ledger=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)"
  [ $RC -eq 0 ] || { say "$TAG 未正常结束,回归无法判读"; tail -20 "$OUT/run.log"; exit 2; }
}

# 三遍可以分开跑(每遍约 5 分钟):CASES="head" / "offA" / "onB" / "compare"。
# 默认全跑。分开跑是为了让每一遍都能是一次**前台阻塞**调用 —— 丢到后台再回来
# 收结果的做法在这个运行时里会丢结果。
CASES="${CASES:-head offA onB compare}"
wants() { [ "${CASES#*$1}" != "$CASES" ]; }

wants head && run_case head "$HEAD_REPO"
wants offA && run_case offA "$NEW_REPO"
wants onB  && run_case onB  "$NEW_REPO" --online-observables \
                            --online-observables-window 0.0,1.0 \
                            --online-observables-probes "0.0018,0.0017,0.00044;0.0028,0.0017,0.00044;0.0038,0.0017,0.00044"
if ! wants compare; then say "CASES='$CASES' 不含 compare,先到这里"; exit 0; fi

python "$M/check_bytes_identical.py" \
  --baseline "$OUTROOT/iet20_reg_head" \
  --candidate "$OUTROOT/iet20_reg_offA" --label "flag OFF" \
  --output "$WORK/online_observables_regression.json" || exit 2
python "$M/check_bytes_identical.py" \
  --baseline "$OUTROOT/iet20_reg_head" \
  --candidate "$OUTROOT/iet20_reg_onB" --label "flag ON" \
  --output "$WORK/online_observables_regression_on.json" || exit 2

say "---- 记录器产物 ----"
ls -la "$OUTROOT/iet20_reg_onB"/online_observables* 2>/dev/null || {
  say "带旗标那一遍没有产出记录器文件 —— 记录器没生效"; exit 2; }
python "$M/summarize_online_observables.py" \
  "$OUTROOT/iet20_reg_onB/online_observables.jsonl" \
  --meta "$OUTROOT/iet20_reg_onB/online_observables_meta.json" || exit 2

say "======== 回归通过 ========"
