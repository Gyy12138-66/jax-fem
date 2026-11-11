#!/usr/bin/env bash
# ============================================================================
# 三路读数重读:用**现有** box-159 生产场重新出读数、出图、出表。零 GPU。
#
# 不重跑任何求解器 —— 只把 90 帧 x 2 臂的温度场按三种读数定义重读一遍:
#   1  采纳口径(Balbaa Sec 3.3 条件平均)—— 原样保留,一个字不改
#   2  双色辐射合成(仪器物理,D-V2-24)
#   3  全光斑无阈值平均(诊断下界)
#
# 顺带跑一次**采纳口径回归**:新版 analyze_pyrometer.py 的 avg_K / n_hot /
# max_K 必须与 2026-08-19 生产产物逐箱逐帧一致。三路是加法,不是改动。
#
# 用法: bash v_reread_three_readings.sh [OUTDIR]
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env

HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/.." && pwd)
V2=$(cd "$M/.." && pwd)
OUTROOT=${OUTROOT:-/home/user/work/159/output}
VT=${VT:-/home/user/work/159/vtmp}
OUT=${1:-$VT/iet20}
mkdir -p "$OUT"

ASIS=$OUTROOT/v2_thermal_gate_asis
PARITY=$OUTROOT/v2_thermal_gate_parity

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

say "======== 三路读数重读(零 GPU,不重跑求解器)========"
say "  as-is  场: $ASIS"
say "  parity 场: $PARITY"
say "  产物    : $OUT"

# ---- 0. 仪器模型自检:均匀场必须被精确还原 ----
say "---- 双色合成读数:均匀场自检 ----"
python "$M/two_colour_pyrometer.py" || { say "自检未过,停止"; exit 2; }

# ---- 1. 逐臂三路提取 ----
for pair in "asis:$ASIS" "parity:$PARITY"; do
  NAME=${pair%%:*}; DIR=${pair#*:}
  say "---- 提取 $NAME ----"
  python "$M/analyze_pyrometer.py" "$DIR" --arm "$NAME" --tc-sensitivity \
    --output "$OUT/pyro3_gate_${NAME}.json" || exit 2
done

# ---- 2. 采纳口径回归:新版必须与生产产物逐值一致 ----
say "---- 采纳口径回归(vs 2026-08-19 生产产物)----"
for NAME in asis parity; do
  OLD=$VT/pyro_gate_${NAME}.json
  [ -f "$OLD" ] || { say "  跳过 $NAME:找不到 $OLD"; continue; }
  python "$M/check_adopted_reading_unchanged.py" \
    --old "$OLD" --new "$OUT/pyro3_gate_${NAME}.json" --arm "$NAME" || exit 2
done

# ---- 3. 三路并列对比 + 两臂差 + 出图出表 ----
say "---- 三路对比 ----"
python "$M/compare_thermal_gate.py" \
  --parity "$OUT/pyro3_gate_parity.json" \
  --asis   "$OUT/pyro3_gate_asis.json" \
  --experiment "$V2/inputs/balbaa-fig14-pyrometer.json" \
  --output "$OUT/thermal_gate_three_readings.json" \
  --plot   "$OUT/thermal_gate_three_readings.png" \
  --table  "$OUT/thermal_gate_three_readings.csv" || exit 2

say "======== 完成 ========"
ls -la "$OUT"
