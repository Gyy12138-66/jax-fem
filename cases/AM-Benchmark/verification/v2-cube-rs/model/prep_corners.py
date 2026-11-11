#!/usr/bin/env python3
"""缺口 3(Kimi):星形设计测不出 floor x onset 交互,补角点。

Kimi 提名 f0005_c07 与 f002_c09。补 Kimi 的 v2 收割给出的新证据:**全体包络的
最大值出现在 f0005_c08 vs f001_c09(低地板 x 晚坍塌)这条对角上**(vM 6.34 %),
而不是任一轴的端到端(4.97 % / 4.91 %)。也就是说交互确实存在且可测,而最陡的
方向是"低地板 + 晚坍塌"(坍塌起点晚 -> 掉得更陡),不是 Kimi 直觉的"早坍塌 + 低地板"。

所以四个角点全部备好,由 Fable5 决定跑哪几个:
    f0005_c07  低地板 + 早坍塌   (长而浅的坡,掉到很低)
    f0005_c09  低地板 + 晚坍塌   (最陡的坡)      <- 证据指向这里
    f002_c07   高地板 + 早坍塌   (最缓的坡)
    f002_c09   高地板 + 晚坍塌
只生成表与配置,不跑。
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
MIN_TANGENT = 0.01

CORNERS = [
    ("f0005_c07", 0.005, 0.7),
    ("f0005_c09", 0.005, 0.9),
    ("f002_c07", 0.020, 0.7),
    ("f002_c09", 0.020, 0.9),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, cwd=str(M), capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd)); print(r.stdout[-2000:]); print(r.stderr[-2000:])
        sys.exit(1)
    return r.stdout


print("缺口 3 角点:生成命令与 sha256(只生成,不跑)\n")
for tag, ff, cf in CORNERS:
    e_tbl = TB / f"E_collapse_{tag}.csv"
    run([PY, "make_e_collapse.py", "--out", str(e_tbl),
         "--floor-frac", str(ff), "--collapse-frac", str(cf)])
    arm = f"ecol_{tag}"
    run([PY, "make_flow_curve_variants.py", "--arm", "offset", "--name", arm,
         "--min-tangent-frac", str(MIN_TANGENT), "--e-table", str(e_tbl)])
    p = M / f"v2_material_config_fc_{arm}.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["E_table"] = str(e_tbl.resolve())
    cfg["_comment"] = (f"扫描 A 角点 {tag}(缺口 3,交互检测): floor-frac={ff:g}, "
                       f"collapse-frac={cf:g}。" + cfg.get("_comment", ""))
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"### {tag}   floor={ff:g}  onset={cf:g}")
    print(f"    python make_e_collapse.py --out tables/{e_tbl.name} "
          f"--floor-frac {ff:g} --collapse-frac {cf:g}")
    print(f"    python make_flow_curve_variants.py --arm offset --name {arm} "
          f"--min-tangent-frac {MIN_TANGENT:g} --e-table tables/{e_tbl.name}")
    for q in (e_tbl, TB / f"flow_curve_{arm}.csv", p):
        print(f"    {sha(q)}  {q.name}")
    print()
