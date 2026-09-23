#!/usr/bin/env python3
"""D-V2-17 方案 (c) 落地:把 E(T) 坍塌 + 有限地板变成 V2 的**采纳口径**。

决策:yuyao 2026-08-06 —— 按 (c) 与主线 D-11 一并定口径;括号宽度是登记内容,
不是采纳前提。参数冻结 floor-frac 1 %、collapse-frac 0.8·T_sol(扫描 A 的中心点)。

落地方式(刻意让审查者一眼能看出改了什么):
  - `E.csv` **不动**,继续是 D-V2-03 的数据表重建,保留出处;
  - `E_collapse.csv` 是采纳表(floor 0.01 / onset 0.8),已在扫描 A 中作为中心点跑过;
  - 新增 `flow_curve_adopted.csv`(offset 主臂)与 `flow_curve_adopted_cap.csv`
    (cap 括号臂),两者都按采纳的 E 表重建,最小切线仍 0.01(剂量不是扫描变量);
  - `v2_material_config.json` 的 E_table / flow_curve_table 指向上述两者。

为什么另起 adopted 名字而不是覆盖 offset_mt:offset_mt 等实验臂是证据链的一部分,
覆盖它们会让"当时到底跑的是哪张表"事后无法复原。采纳口径单独命名,实验臂原样留存。
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

M = Path("/home/user/work/159/jax-fem/cases/AM-Benchmark/verification"
         "/v2-cube-rs/model")
TB = M / "tables"
PY = "/home/user/miniconda3/envs/jax-fem-env/bin/python"
CFG = M / "v2_material_config.json"

ADOPTED_E = TB / "E_collapse.csv"
FLOOR_FRAC, COLLAPSE_FRAC, MIN_TANGENT = 0.01, 0.8, 0.01


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, cwd=str(M), capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd)); print(r.stdout[-2000:]); print(r.stderr[-2000:])
        sys.exit(1)
    return r.stdout


before = json.loads(CFG.read_text(encoding="utf-8"))
print("=== 采纳前 v2_material_config.json ===")
for k in ("E_table", "flow_curve_table"):
    print(f"  {k}: {Path(before[k]).name}")

# 1) 采纳的 E 表:用冻结参数重新生成一次,确保它确实等于 (0.01, 0.8)
run([PY, "make_e_collapse.py", "--out", str(ADOPTED_E),
     "--floor-frac", str(FLOOR_FRAC), "--collapse-frac", str(COLLAPSE_FRAC)])

# 与扫描 A 的中心点表逐字节比对:采纳的就是被扫过的那一张,不是另造一张
center = TB / "E_collapse_f001_c08.csv"
same_as_center = sha(ADOPTED_E) == sha(center)
print(f"\n采纳 E 表 == 扫描 A 中心点表: {'是' if same_as_center else '否 <- 有问题'}")
if not same_as_center:
    sys.exit(1)

# 2) 采纳的流动曲线(主臂 + 括号臂),按采纳的 E 表构建
for arm, name in (("offset", "adopted"), ("cap", "adopted_cap")):
    run([PY, "make_flow_curve_variants.py", "--arm", arm, "--name", name,
         "--min-tangent-frac", str(MIN_TANGENT), "--e-table", str(ADOPTED_E)])

# 3) 基准配置指向采纳口径
cfg = json.loads(CFG.read_text(encoding="utf-8"))
cfg["E_table"] = str(ADOPTED_E.resolve())
cfg["flow_curve_table"] = str((TB / "flow_curve_adopted.csv").resolve())
cfg["_comment"] = (
    f"D-V2-17 方案 (c) 已采纳(yuyao 2026-08-06,与主线 D-11 一并定口径):"
    f"E(T) 随固相线坍塌 + 有限地板,参数冻结 floor-frac={FLOOR_FRAC:g}、"
    f"collapse-frac={COLLAPSE_FRAC:g}*T_sol;流动曲线按该 E 表重建,"
    f"D-V2-19-R1 最小切线保持 {MIN_TANGENT:g}*E(T)。敏感度括号见 deviations.yaml "
    f"D-V2-17(扫描 A,登记内容而非采纳前提)。" + cfg.get("_comment", ""))
CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 采纳后 v2_material_config.json ===")
for k in ("E_table", "flow_curve_table"):
    print(f"  {k}: {Path(cfg[k]).name}")

print("\n=== 采纳口径的 sha256 ===")
for p in (ADOPTED_E, TB / "flow_curve_adopted.csv", TB / "flow_curve_adopted_cap.csv", CFG):
    print(f"  {sha(p)}  {p.name}")

# 4) 采纳臂 vs 扫描中心点臂:应逐值相同(同 E、同剂量、同 offset 处理)
a = (TB / "flow_curve_adopted.csv").read_text().splitlines()
b = (TB / "flow_curve_ecol_f001_c08.csv").read_text().splitlines()
import csv as _csv


def vals(lines):
    return {(r["temperature_K"], r["equivalent_plastic_strain"]): r["flow_stress_Pa"]
            for r in _csv.DictReader(iter(lines))}


va, vb = vals(a), vals(b)
same = sum(1 for k in va if va[k] == vb.get(k))
print(f"\n采纳流动曲线 vs 扫描中心点臂:{same}/{len(va)} 个格点逐值相同"
      + ("  <- 采纳的就是被扫过的那条曲线" if same == len(va) else "  <- 不一致,需查"))
