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
#
# 2026-08-20(IET-20)删掉了 `--front-surface-loss-radiation`。它是**空转旗标**:
# thermal.py:292 把整个 front-loss 块(含 :304 的辐射分支)门控在
# `front_surface_loss_h > 0 且 thickness > 0` 之下,而本脚本两者都没传、配置里
# 也没有这两个键(默认 0.0)。生产运行的 used_config.json 记的正是
# h=0.0 / thickness=0.0 / radiation=true,台账里 front_loss_j 全程 8866 步
# 合计 0.0000 J —— 它一点作用都没有,只会让人以为多了一项辐射。
# 真正的对流/辐射走 `--surface-selection exterior` 的表面积分路径,未受影响:
# 同一台账里 surface_loss_j = 0.1297 J,物理照旧。删除**不改变任何数值结果**。
# ============================================================================
set -euo pipefail
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
# Balbaa Sec. 3.3 / scoring spec: response window and three top-layer probes.
OBS_WINDOW="${OBS_WINDOW:-0.45,0.90}"
OBS_PROBES="${OBS_PROBES:-0.001,0.002,0.00042;0.002,0.002,0.00042;0.003,0.002,0.00042}"
LOG=$VT/thermal_gate.log
mkdir -p "$VT"
cd /home/user/work/159

if [ "$SMOKE" = "1" ]; then
  TAG=smoke; PATH_ARGS=(--tracks 6); MESH=$M/v2_multitrack_c3d8.inp
  # Keep the registered 0.45--0.90 s protocol reachable. The short six-track
  # scan is followed by inexpensive cooling steps through the end of the window.
  OUT_EVERY=10; COOL_STEPS=90; COOL_DT=dynamic-to-window-end
else
  TAG=gate; PATH_ARGS=(--exposure-area 10.0e-3); MESH=$M/v2_multitrack_c3d8.inp
  # 细步长 dt = 50 um / 0.65 m/s = 7.69e-5 s -> 每 100 步出一帧 = 7.7 ms <= 10 ms 协议增量
  OUT_EVERY=100; COOL_STEPS=40; COOL_DT=0.01
fi
ASIS=$OUTROOT/v2_thermal_${TAG}_asis
PARITY=$OUTROOT/v2_thermal_${TAG}_parity
KEFF_DIR=$VT/derived
KEFF_CSV=$KEFF_DIR/k_liquid_keff.csv
KEFF_JSON=$KEFF_DIR/keff_${TAG}.json
PARITY_CFG=$KEFF_DIR/v2_material_config_thermal_keff.json
mkdir -p "$KEFF_DIR"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
has()  { [ "${STAGES#*$1}" != "$STAGES" ]; }
RUN_PARAMETERS=$(python3 - "$OBS_WINDOW" "$OBS_PROBES" <<'PY'
import json, sys
window, probes = sys.argv[1:]
print(json.dumps({
    "laser_power_W": 220.0, "scan_speed_m_s": 0.650,
    "hatch_m": 0.12e-3, "layer_thickness_m": 4.0e-5,
    "beam_radius_m": 5.0e-5, "source_depth_m": 1.0e-4,
    "source_depth_cutoff_m": 4.0e-5, "source_cutoff_renormalize": True,
    "fixture_thermal_phase": "follow-temperature", "dt_s": 7.6923e-5,
    "ambient_K": 313.0, "preheat_K": 353.15,
    "bottom_thermal_bc": "fixed", "bottom_temperature_K": 353.15,
    "surface_selection": "exterior", "observation_window": window,
    "observation_probes": probes, "response_bin_ms": 10.0,
}, sort_keys=True))
PY
)

build_manifest() {
  python3 "$M/build_run_manifest.py" --repo "$REPO" --arm "$1" \
    --config "$2" --mesh "$MESH" --path "$3" \
    --parameters-json "$RUN_PARAMETERS" --output "$4"
}

is_complete() {
  local NAME=$1 CFG=$2 OUT=$3 EXPECTED
  [ -s "$OUT/path.csv" ] || return 1
  EXPECTED=$(mktemp)
  build_manifest "$NAME" "$CFG" "$OUT/path.csv" "$EXPECTED" >/dev/null || { rm -f "$EXPECTED"; return 1; }
  python3 - "$OUT" "$EXPECTED" <<'PY' 2>/dev/null
import hashlib, json, os, sys
out, expected_path = sys.argv[1:]
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
try:
    manifest = json.load(open(os.path.join(out, "run_manifest.json"), encoding="utf-8"))
    expected = json.load(open(expected_path, encoding="utf-8"))
    ledger = json.load(open(os.path.join(out, "thermal_energy_ledger_summary.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(out, "online_observables_meta.json"), encoding="utf-8"))
    summary = json.load(open(os.path.join(out, "online_observables_summary.json"), encoding="utf-8"))
    audit = json.load(open(os.path.join(out, "run_audit.json"), encoding="utf-8"))
    rows = os.path.join(out, "online_observables.jsonl")
    run_id = expected["run_id"]
    ok = (manifest.get("run_id") == run_id
          and ledger.get("complete") is True
          and manifest.get("status") == "complete"
          and os.path.getsize(rows) > 0
          and meta.get("run_id") == run_id
          and summary.get("meta", {}).get("run_id") == run_id
          and summary.get("source_jsonl_sha256") == sha256(rows)
          and summary.get("n_rows") == sum(1 for line in open(rows, encoding="utf-8") if line.strip())
          and summary.get("coverage_s", [None, None])[0] <= 0.45 + 1.0e-9
          and summary.get("coverage_s", [None, None])[1] >= 0.90 - 1.0e-9
          and audit.get("thermal_only") is True
          and audit.get("transient", {}).get("all_steps_valid") is True
          and summary.get("response_integrated_series"))
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  local RC=$?
  rm -f "$EXPECTED"
  return "$RC"
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
  if is_complete "$NAME" "$CFG" "$OUT"; then say "$NAME 已完成,跳过(幂等)"; return 0; fi
  preflight "$CFG"
  say "$NAME 开始 -> $OUT"
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" "${PATH_ARGS[@]}" \
    --power 220 --speed 0.650 --hatch 0.12e-3 \
    --output "$OUT/path.csv" --ledger-json "$OUT/path_ledger.json" \
    2>&1 | tee "$OUT/path.log" | sed 's/^/    /'
  local ARM_COOL_DT="$COOL_DT"
  if [ "$SMOKE" = "1" ]; then
    ARM_COOL_DT=$(python3 - "$OUT/path.csv" "$COOL_STEPS" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
steps = int(sys.argv[2])
path_end = float(rows[-1]["time"])
# Stay just inside the inclusive recorder boundary after repeated FP additions;
# is_complete allows 1e-9 s at the registered 0.90 s endpoint.
target = 0.90 - 1.0e-12
remaining = target - path_end
if steps <= 0 or remaining <= 0.0:
    raise SystemExit("smoke path already reaches/exceeds registered window end")
print(f"{remaining / steps:.17g}")
PY
    ) || { say "$NAME 无法把 smoke cooling 精确落在 0.90 s，停止"; exit 2; }
  fi
  local RUN_ID
  RUN_ID=$(build_manifest "$NAME" "$CFG" "$OUT/path.csv" "$OUT/run_manifest.json")
  set +e
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
    --source-depth-cutoff 4.0e-5 --source-cutoff-renormalize \
    --laser-power 220 --dt 7.6923e-5 \
    --layer-activation-mode layer_on_scan --layer-activation-geometry intersection \
    --future-layer-mode void --active-window-below-layers 0 --inactive-mass-factor 1.0 \
    --powder-mode powder --surface-selection exterior --boundary-tol 1.0e-6 \
    --quadrature-order 2 --ambient 313.0 --preheat-temperature 353.15 \
    --bottom-thermal-bc fixed --bottom-temperature 353.15 \
    --cooling-steps "$COOL_STEPS" --cooling-dt "$ARM_COOL_DT" \
    --mechanics-every 0 \
    --thermal-mass-lumping --thermal-output-every "$OUT_EVERY" --summary-every 200 \
    --phase-history-model paper_irreversible \
    --fixture-thermal-phase follow-temperature \
    --online-observables --online-observables-run-id "$RUN_ID" \
    --online-observables-window "$OBS_WINDOW" \
    --online-observables-probes "$OBS_PROBES" \
    > "$OUT/run.log" 2>&1
  local RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    python3 - "$OUT/run_manifest.json" "$RC" <<'PY'
import json, sys
path, rc = sys.argv[1], int(sys.argv[2])
manifest = json.load(open(path, encoding="utf-8"))
manifest["status"] = "failed"
manifest["exit_code"] = rc
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
PY
  fi
  local N
  N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
  say "$NAME 结束 rc=$RC ledger=$N"
  # 起步即死护栏:台账没几行就是配置层面的问题,别烧几小时再发现
  if [ "$N" -le 12 ]; then
    say "$NAME 起步即死(ledger=$N)。停止,等人工判读。"; say "GATE_ABORTED"; exit 2
  fi
  [ "$RC" -eq 0 ] || { say "$NAME 求解失败 rc=$RC，停止"; exit "$RC"; }
  python -m jax_fem_am.verification.run_audit "$OUT" \
    --output "$OUT/run_audit.json" --thermal-only \
    --ambient 313.0 --quality-threshold 0.05 \
    >> "$OUT/run.log" 2>&1 \
    || { say "$NAME run_audit 失败，停止"; exit 2; }
  for artifact in online_observables.jsonl online_observables_meta.json; do
    [ -s "$OUT/$artifact" ] \
      || { say "$NAME 缺少在线观测产物 $artifact，停止"; exit 2; }
  done
  python "$M/summarize_online_observables.py" \
    "$OUT/online_observables.jsonl" \
    --meta "$OUT/online_observables_meta.json" \
    --expected-run-id "$RUN_ID" \
    --output "$OUT/online_observables_summary.json" \
    >> "$OUT/run.log" 2>&1 \
    || { say "$NAME 在线观测汇总失败，停止"; exit 2; }
  python3 - "$OUT/run_manifest.json" <<'PY'
import json, sys
path = sys.argv[1]
manifest = json.load(open(path, encoding="utf-8"))
manifest["status"] = "complete"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
PY
}

say "======== 热闸门启动 pid=$$ TAG=$TAG STAGES='$STAGES' ========"

# ---- 阶段 1:as-is 臂 ----
if has 1; then
  run_arm asis "$M/v2_material_config_thermal_asis.json" "$ASIS"
fi

# ---- 阶段 2:从 as-is 臂量 L,推 keff ----
if has 2; then
  # F2(Fable5 2026-08-19)第一处:as-is 臂没跑完就不许推 keff。
  # 原来只有 ledger<=12 的起步即死护栏,半途死掉(rc!=0 但台账很长)会漏过去。
  if ! is_complete asis "$M/v2_material_config_thermal_asis.json" "$ASIS"; then
    say "as-is 臂未完成($ASIS),阶段 2 拒绝从半成品推 keff。先把阶段 1 跑完。"
    exit 2
  fi
  # F2 第二处:幂等不能只看"文件在不在"。把 as-is 臂的指纹和已有 keff 台账里
  # 记的比一比,不一致(例如 as-is 臂重跑过)就重算,否则 parity 臂会带着
  # 陈旧 keff 跑完全程 —— 而且台账里看不出来。
  STALE=1
  if [ -f "$KEFF_CSV" ] && [ -f "$KEFF_JSON" ]; then
    if python3 - "$KEFF_JSON" "$ASIS" <<'PY'
import json, sys, os, hashlib
rec = json.load(open(sys.argv[1], encoding="utf-8"))
old = (rec.get("characteristic_length") or {}).get("source_run_fingerprint")
if not old:
    print("已有 keff 台账没有 source_run_fingerprint(旧版产物)-> 重算"); sys.exit(1)
s = os.path.join(sys.argv[2], "thermal_energy_ledger_summary.json")
cur = hashlib.sha256(open(s, "rb").read()).hexdigest() if os.path.isfile(s) else None
if old.get("summary_sha256") != cur:
    print(f"as-is 臂指纹变了({old.get('summary_sha256')} -> {cur})-> 重算")
    sys.exit(1)
print("as-is 臂指纹一致 -> keff 表可复用")
PY
    then STALE=0; fi
  fi
  if [ "$STALE" = "0" ]; then
    say "keff 表与当前 as-is 臂一致,跳过(幂等):$KEFF_CSV"
  else
    say "阶段 2:从 as-is 臂量熔池半宽 -> Balbaa Eq 19-24"
    # F1:不再在整层累积场上用 ±hatch/2 的窗做双侧量测 —— 那个量法在 M1 网格上
    # 有 hatch/2 = 60 um 的结构性天花板,而且是静默的。改为:回到某一条穿窗道
    # 刚跑完、下一道还没到的那一刻,做**单侧(上半)**量测。道号/时刻/窗口由
    # pick_keff_measurement.py 从这次实际用过的 path.csv 推出,不硬编码。
    # 先取回再解析:进程替换里的失败退出码 read 是看不见的
    MEAS=$(python "$M/pick_keff_measurement.py" "$ASIS/path.csv" \
             --nth "${MEAS_NTH:-3}" 2>&1) \
      || { say "选取量测道/时刻失败:$MEAS"; exit 2; }
    read -r MY MT MX0 MX1 MY0 MY1 <<< "$MEAS"
    if [ -z "${MY1:-}" ]; then say "量测参数解析失败:$MEAS"; exit 2; fi
    say "  量测:道中心线 y=${MY} m,取样时刻 t=${MT} s,窗 x[${MX0},${MX1}] y[${MY0},${MY1}]"
    python "$M/make_keff_table.py" \
      --power 220 --speed 0.650 \
      --from-run "$ASIS" \
      --from-run-window "${MX0},${MX1},${MY0},${MY1}" \
      --from-run-time "$MT" --from-run-track-y "$MY" \
      --tag "$TAG" --output "$KEFF_CSV" --json "$KEFF_JSON" \
      2>&1 | tee -a "$LOG" | sed 's/^/    /'
    [ -f "$KEFF_CSV" ] || { say "keff 表没生成(多半是饱和/分辨率护栏拦下了,看上面的原因),停止"; exit 2; }
  fi
fi

# ---- 阶段 3:parity 臂 ----
if has 3; then
  [ -s "$KEFF_CSV" ] || { say "缺少阶段 2 的 keff 表:$KEFF_CSV"; exit 2; }
  python3 - "$M/v2_material_config_thermal_keff.json" "$PARITY_CFG" "$KEFF_CSV" <<'PY'
import json, sys
source, output, keff = sys.argv[1:]
config = json.load(open(source, encoding="utf-8"))
config["k_table_liquid"] = keff
config["_k_table_liquid_note"] = "runtime-derived Balbaa Eq 19-24 table; path bound by run manifest"
with open(output, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
PY
  say "两臂配置逐键核对(必须只差 k_table_liquid):"
  python3 - "$M/v2_material_config_thermal_asis.json" \
             "$PARITY_CFG" <<'PY' | tee -a "$LOG" | sed 's/^/    /'
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
  run_arm parity "$PARITY_CFG" "$PARITY"
fi

# ---- 阶段 4:对比 ----
if has 4; then
  say "阶段 4:高温计协议提取 + 双臂对比"
  for triple in "asis:$M/v2_material_config_thermal_asis.json:$ASIS" \
                "parity:$PARITY_CFG:$PARITY"; do
    NAME=${triple%%:*}; REST=${triple#*:}; CFG=${REST%%:*}; DIR=${REST#*:}
    [ -s "$CFG" ] || { say "$NAME 缺少运行配置 $CFG，停止比较"; exit 2; }
    is_complete "$NAME" "$CFG" "$DIR" \
      || { say "$NAME 当前产物未通过 manifest/观测/audit 完整性检查，停止比较"; exit 2; }
    [ -s "$DIR/online_observables_summary.json" ] \
      || { say "$NAME 缺少在线响应积分摘要，停止比较"; exit 2; }
    python "$M/analyze_pyrometer.py" "$DIR" --arm "$NAME" \
      --online-summary "$DIR/online_observables_summary.json" \
      --observation-window "$OBS_WINDOW" \
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
