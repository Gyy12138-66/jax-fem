#!/usr/bin/env python3
"""扫描 A 角点补充(Kimi 缺口 3,Fable5 2026-08-06):floor x onset 交互检测。

两个物理上最极端的对角:f0005_c07(早坍塌 + 低地板,最软)与 f002_c09
(晚坍塌 + 高地板,最硬)。生成管线与 make_sweep_a.py 逐字相同,只换
MEMBERS;剂量 min-tangent 0.01 不动。原五成员规格文档:

5 个成员(Fable5 2026-08-05 锁定规格):
    地板扫描  floor-frac in {0.005, 0.01, 0.02}  x  collapse-frac 0.8
    起点扫描  collapse-frac in {0.7, 0.9}        x  floor-frac 0.01
中心点 (0.01, 0.8) 与三件套验证同配置,但本批**一并重跑**——Fable5 允许复用,
但要求逐项核对同构;重跑比核对便宜,而且让 5 个成员出自同一 harness、同一驱动,
省掉"中心点是不是真同构"这个此后无法再验的疑问。

每个成员:E 表 -> 流动曲线(min-tangent 0.01,剂量不动)-> 材料配置(E_table 指向
该成员自己的 E 表)。最后打印 sha256,启动前贴到 issue,防事后解释。
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

M = Path("/home/user/work/159/jax-fem/cases/AM-Benchmark/verification"
         "/v2-cube-rs/model")
TB = M / "tables"
PY = "/home/user/miniconda3/envs/jax-fem-env/bin/python"

# (tag, floor_frac, collapse_frac)
MEMBERS = [
    ("f0005_c07", 0.005, 0.7),
    ("f002_c09", 0.020, 0.9),
]
MIN_TANGENT = 0.01   # 剂量固定,本批不动


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, cwd=str(M), capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(1)
    return r.stdout


rows = []
for tag, ff, cf in MEMBERS:
    e_tbl = TB / f"E_collapse_{tag}.csv"
    run([PY, "make_e_collapse.py", "--out", str(e_tbl),
         "--floor-frac", str(ff), "--collapse-frac", str(cf)])
    arm = f"ecol_{tag}"
    run([PY, "make_flow_curve_variants.py", "--arm", "offset", "--name", arm,
         "--min-tangent-frac", str(MIN_TANGENT), "--e-table", str(e_tbl)])
    cfg_p = M / f"v2_material_config_fc_{arm}.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg["E_table"] = str(e_tbl.resolve())
    cfg["_comment"] = (
        f"扫描 A 成员 {tag}: D-V2-17 方案 (c),floor-frac={ff:g}, "
        f"collapse-frac={cf:g}, min-tangent-frac={MIN_TANGENT:g}(剂量不动)。"
        + cfg.get("_comment", ""))
    cfg_p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.append((tag, ff, cf, e_tbl, TB / f"flow_curve_{arm}.csv", cfg_p))

print("扫描 A —— 5 个成员的生成命令与 sha256")
print()
for tag, ff, cf, e_tbl, fc, cfg_p in rows:
    print(f"### {tag}   floor-frac={ff:g}  collapse-frac={cf:g}"
          + ("   <- 中心点" if tag == "f001_c08" else ""))
    print(f"    python make_e_collapse.py --out tables/{e_tbl.name} "
          f"--floor-frac {ff:g} --collapse-frac {cf:g}")
    print(f"    python make_flow_curve_variants.py --arm offset --name ecol_{tag} "
          f"--min-tangent-frac {MIN_TANGENT:g} --e-table tables/{e_tbl.name}")
    for p in (e_tbl, fc, cfg_p):
        print(f"    {sha256(p)}  {p.name}")
    print()

# 各成员在固相线处的 E,便于一眼看出扫描跨度
print("各成员 E(固相线) 与坍塌起点:")
import csv
for tag, ff, cf, e_tbl, _, _ in rows:
    with open(e_tbl) as f:
        rr = list(csv.DictReader(f))
    t_c = float(rr[-2]["T"])
    e_sol = float(rr[-1]["value"])
    print(f"  {tag:>10}  T_c={t_c:>8.2f} K   E(T_sol)={e_sol/1e9:>7.3f} GPa")
