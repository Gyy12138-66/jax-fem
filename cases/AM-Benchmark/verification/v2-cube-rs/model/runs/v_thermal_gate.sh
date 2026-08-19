#!/usr/bin/env bash
# ============================================================================
# 热闸门:keff 双臂 vs 双色高温计(D-V2-21 双臂,IET-8)
#
# 四个阶段,**顺序是物理决定的**,不是排版:
#   1  as-is 臂     不加 keff,如实跑
#   2  keff 推导    从 as-is 臂**自己的**熔池半宽 L 算 Balbaa Eq 19-24
#                   (一次 Picard;L 是模型输出,这是 D-V1-10 的鸡生蛋)
#   3  parity 臂    带 keff 重跑,与 as-is 臂只差 k_table_liquid 一个键
#   4  对比         高温计协议提取 x2 + 双臂相减 + 三角形对比 + 出图
#
# **幂等**:已完成的阶段直接跳过(台账 complete==true / 产物已存在)。
# **串行**:两臂不并发 —— 2026-08-03 三个变体并发触发过 18.8 GB RSS 的内核 OOM。
# CPU:热闸门不占 GPU(D-11 矩阵优先占 GPU,IET-7)。
#
# 用法
#   bash v_thermal_gate.sh              生产(整层窗口模型,83 道)
#   SMOKE=1 bash v_thermal_gate.sh      冒烟(6 道,4x4 域,证明管线能起步)
#   STAGES="1 2" bash v_thermal_gate.sh 只跑指定阶段
#
# 红线:fork-only、零标定、不触实测回调、不改共享求解器。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=6

REPO=/home/user/work/159/jax-fem
M=$REPO/cases/AM-Benchmark/verification/v2-cube-rs/model
V2=$REPO/cases/AM-Benchmark/verification/v2-cube-rs
OUTROOT=/home/user/work/159/output
VT=/home/user/work/159/vtmp
SMOKE="${SMOKE:-0}"
STAGES="${STAGES:-1 2 3 4}"
LOG=$VT/thermal_gate.log
mkdir -p "$VT"
cd /home/user/work/159

if [ "$SMOKE" = "1" ]; then
  TAG=smoke; PATH_ARGS="--tracks 6"; MESH=$M/v2_multitrack_c3d8.inp
  OUT_EVERY=10; COOL_STEPS=5
else
  TAG=gate;  PATH_ARGS="--exposure-area 10.0e-3"; MESH=$M/v2_multitrack_c3d8.inp
  # 细步长 dt = 50 um / 0.65 m/s = 7.69e-5 s -> 每 100 步出一帧 = 7.7 ms <= 10 ms 协议增量
  OUT_EVERY=100; COOL_STEPS=40
fi
ASIS=$OUTROOT/v2_thermal_${TAG}_asis
PARITY=$OUTROOT/v2_thermal_${TAG}_parity
KEFF_CSV=$M/tables/k_liquid_keff.csv
KEFF_JSON=$M/derived/keff_${TAG}.json

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
has()  { [ "${STAGES#*$1}" != "$STAGES" ]; }
is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}

# ---- 起飞前体检(Fable5 预检清单:材料表逐值溯源 + 配置回显 + 判据先打印)----
preflight() {
  say "---- 预检 ($1) ----"
  python3 - "$1" <<'PY'
import json, sys, csv, os
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print("  material_name:", cfg["material_name"])
for k in ("rho_solid","rho_powder","rho_liquid","conductivity_liquid","cp_liquid",
          "solidus_temperature","liquidus_temperature","latent_heat",
          "absorptivity","emissivity","convection_h","powder_mode"):
    print(f"    {k:24s} {cfg.get(k)}")
kt = cfg.get("k_table_liquid")
print(f"    {'k_table_liquid':24s} {kt or '(无 -> 用上面的标量,as-is 臂)'}")
if kt:
    if not os.path.isfile(kt):
        print("    !! keff 表不存在 —— 先跑阶段 2"); sys.exit(3)
    rows = list(csv.DictReader(open(kt, encoding="utf-8")))
    vals = sorted({float(r["value"]) for r in rows})
    print(f"    keff 表 {len(rows)} 行, 取值 {vals} W/mK")
    print(f"    来源     {rows[0]['source']}")
    if len(vals) != 1:
        print("    !! keff 表不是常值 —— Balbaa 的 keff 每工况一个数"); sys.exit(3)
for k in ("k_table_solid","cp_table_solid","k_table_powder","cp_table_powder"):
    p = cfg.get(k)
    print(f"    {k:24s} {'OK' if p and os.path.isfile(p) else '缺失!'}  {p}")
    if not (p and os.path.isfile(p)): sys.exit(3)
PY
  [ $? -eq 0 ] || { say "预检未过,停止"; exit 2; }
  say "  判据(开跑前先写死,免得事后挑):"
  say "    - 台账 complete==true 且 rc==0"
  say "    - 能量平衡相对误差与 sweepA 同量级(~1e-6)"
  say "    - 圆内有单元过 1000 degC 的热帧,帧距 <= 10 ms"
  say "    - 两臂逐键只差 k_table_liquid(阶段 3 前已核对)"
}

run_arm() {
  local NAME=$1 CFG=$2 OUT=$3
  if is_complete "$OUT"; then say "$NAME 已完成,跳过(幂等)"; return 0; fi
  preflight "$CFG"
  say "$NAME 开始 -> $OUT"
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" $PATH_ARGS \
    --power 220 --speed 0.650 --hatch 0.12e-3 \
    --output "$OUT/path.csv" --ledger-json "$OUT/path_ledger.json" \
    2>&1 | tee "$OUT/path.log" | sed 's/^/    /'
  python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$MESH" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "thermal-gate-$NAME" \
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
    --front-surface-loss-radiation \
    --cooling-steps "$COOL_STEPS" --cooling-dt 0.01 \
    --mechanics-every 0 \
    --thermal-mass-lumping --thermal-output-every "$OUT_EVERY" --summary-every 200 \
    --phase-history-model paper_irreversible \
    > "$OUT/run.log" 2>&1
  local RC=$? N
  N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  say "$NAME 结束 rc=$RC ledger=$N"
  # 起步即死护栏:台账没几行就是配置层面的问题,别烧几小时再发现
  if [ "$N" -le 12 ]; then
    say "$NAME 起步即死(ledger=$N)。停止,等人工判读。"; say "GATE_ABORTED"; exit 2
  fi
  python -m jax_fem_am.verification.run_audit "$OUT" \
    --output "$OUT/run_audit.json" --ambient 313.0 --quality-threshold 0.05 \
    >> "$OUT/run.log" 2>&1 || say "$NAME WARNING: run_audit 失败"
}

say "======== 热闸门启动 pid=$$ TAG=$TAG STAGES='$STAGES' ========"

# ---- 阶段 1:as-is 臂 ----
if has 1; then
  run_arm asis "$M/v2_material_config_thermal_asis.json" "$ASIS"
fi

# ---- 阶段 2:从 as-is 臂量 L,推 keff ----
if has 2; then
  if [ -f "$KEFF_CSV" ] && [ -f "$KEFF_JSON" ]; then
    say "keff 表已存在,跳过(幂等):$KEFF_CSV"
  else
    say "阶段 2:从 as-is 臂量熔池半宽 -> Balbaa Eq 19-24"
    # 量测窗限制到**单独一条道**上:多道的 y 跨度是整层,不是熔池宽度。
    # 取窗中心那条道(mesh y = 2.0 mm)上下各半个 hatch。
    python "$M/make_keff_table.py" \
      --power 220 --speed 0.650 \
      --from-run "$ASIS" --from-run-window "0.5e-3,3.5e-3,1.94e-3,2.06e-3" \
      --tag "$TAG" --output "$KEFF_CSV" --json "$KEFF_JSON" \
      2>&1 | tee -a "$LOG" | sed 's/^/    /'
    [ -f "$KEFF_CSV" ] || { say "keff 表没生成,停止"; exit 2; }
  fi
fi

# ---- 阶段 3:parity 臂 ----
if has 3; then
  say "两臂配置逐键核对(必须只差 k_table_liquid):"
  python3 - "$M/v2_material_config_thermal_asis.json" \
             "$M/v2_material_config_thermal_keff.json" <<'PY' | tee -a "$LOG" | sed 's/^/    /'
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
IGNORE = {"_comment", "material_name", "_k_table_liquid_note"}
only_b = sorted(set(b) - set(a) - IGNORE)
diff = sorted(k for k in (set(a) & set(b)) - IGNORE if a[k] != b[k])
print("parity 独有键:", only_b)
print("共有键取值不同:", diff)
sys.exit(0 if only_b == ["k_table_liquid"] and not diff else 4)
PY
  [ ${PIPESTATUS[0]} -eq 0 ] || { say "两臂差异不止 keff,停止"; exit 2; }
  run_arm parity "$M/v2_material_config_thermal_keff.json" "$PARITY"
fi

# ---- 阶段 4:对比 ----
if has 4; then
  say "阶段 4:高温计协议提取 + 双臂对比"
  for pair in "asis:$ASIS" "parity:$PARITY"; do
    NAME=${pair%%:*}; DIR=${pair#*:}
    python "$M/analyze_pyrometer.py" "$DIR" --arm "$NAME" \
      --output "$VT/pyro_${TAG}_${NAME}.json" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
  done
  python "$M/compare_thermal_gate.py" \
    --parity "$VT/pyro_${TAG}_parity.json" \
    --asis   "$VT/pyro_${TAG}_asis.json" \
    --experiment "$V2/inputs/balbaa-fig14-pyrometer.json" \
    --output "$VT/thermal_gate_${TAG}.json" \
    --plot   "$VT/thermal_gate_${TAG}.png" 2>&1 | tee -a "$LOG" | sed 's/^/    /'
fi

say "GATE_DONE  产物:$VT/thermal_gate_${TAG}.json  $VT/thermal_gate_${TAG}.png"
