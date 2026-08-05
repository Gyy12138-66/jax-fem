#!/usr/bin/env python3
"""扫描 A 的收割:每个成员对中心点 (0.01, 0.8) 逐单元比对。

规格(Fable5 2026-08-05):
  - 量:sigma_xx / yy / zz / von Mises / eqp,外加最大主应力
  - **绝对差(MPa)+ 相对差,双列**(Kimi 建议,已采纳)
  - 主指标:von Mises / 最大主应力

5 个成员同网格、同路径、同求解器参数,只有 E 表(与随之重建的流动曲线)不同,
所以顶层 10000 个件内单元可逐单元直接相减,无需插值。

用法: compare_sweepA.py            # 全部成员对中心点
      compare_sweepA.py --json     # 另存 JSON
"""
import glob
import json
import os
import sys

import meshio
import numpy as np

OUT = "/home/user/work/159/output"
PREFIX = "v2_sweepA_"
CENTER = "f001_c08"
TAGS = ["f0005_c08", "f001_c08", "f002_c08", "f001_c07", "f001_c09"]
NX = NY = 100
TOP = NX * NY


def cf(m, n):
    d = m.cell_data_dict.get(n)
    return None if d is None else np.asarray(list(d.values())[0]).reshape(-1)


def qavg(m, pat):
    parts = [cf(m, pat.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def max_principal(m):
    """最大主应力:对每个单元的对称应力张量求最大特征值(8 个积分点先平均)。"""
    comp = {}
    for c in ("xx", "yy", "zz", "xy", "xz", "yz"):
        comp[c] = qavg(m, f"stress_quadQ_{c}")
        if comp[c] is None:
            return None
    n = comp["xx"].size
    t = np.empty((n, 3, 3))
    t[:, 0, 0] = comp["xx"]; t[:, 1, 1] = comp["yy"]; t[:, 2, 2] = comp["zz"]
    t[:, 0, 1] = t[:, 1, 0] = comp["xy"]
    t[:, 0, 2] = t[:, 2, 0] = comp["xz"]
    t[:, 1, 2] = t[:, 2, 1] = comp["yz"]
    return np.linalg.eigvalsh(t)[:, -1]


def load(tag):
    d = os.path.join(OUT, PREFIX + tag)
    vt = sorted(glob.glob(f"{d}/*.vtu"))
    if not vt:
        return None
    m = meshio.read(vt[-1])
    r = {"tag": tag, "vtu": os.path.basename(vt[-1])}
    for c in ("xx", "yy", "zz"):
        r[f"s{c}"] = qavg(m, f"stress_quadQ_{c}")
    r["vm"] = qavg(m, "vm_quadQ")
    r["p1"] = max_principal(m)
    r["eqp"] = cf(m, "eq_plastic_strain")
    r["ncells"] = r["vm"].size
    sp = os.path.join(d, "thermal_energy_ledger_summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        r["complete"] = s.get("complete")
        r["steps"] = s.get("recorded_step_count")
    return r


def top(a, n):
    return a[n - TOP:]


FIELDS = [("vm", "von Mises", 1e6, "MPa"), ("p1", "max principal", 1e6, "MPa"),
          ("sxx", "sigma_xx", 1e6, "MPa"), ("syy", "sigma_yy", 1e6, "MPa"),
          ("szz", "sigma_zz", 1e6, "MPa"), ("eqp", "eqp", 1.0, "-")]


def main():
    runs = {t: load(t) for t in TAGS}
    have = {t: r for t, r in runs.items() if r}
    print("=" * 96)
    print("扫描 A 收割:各成员 vs 中心点 (floor 0.01, collapse 0.8),件内顶层 10000 单元")
    print("=" * 96)
    for t in TAGS:
        r = have.get(t)
        print(f"  {t:>10}  " + ("缺" if not r else
              f"{r['vtu']:<26} steps={r.get('steps','-')} complete={r.get('complete','-')}"))
    if CENTER not in have:
        print("\n中心点缺失,无法比对")
        return
    ref = have[CENTER]
    summary = {"center": CENTER, "members": {}}
    for t in TAGS:
        if t == CENTER or t not in have:
            continue
        r = have[t]
        print(f"\n--- {t} vs {CENTER} ---")
        print(f"{'量':>14} {'中心RMS':>12} {'成员RMS':>12} "
              f"{'绝对RMS差':>12} {'最大绝对差':>12} {'相对RMS差':>11}")
        rec = {}
        for key, lab, scale, unit in FIELDS:
            a, b = ref.get(key), r.get(key)
            if a is None or b is None or a.shape != b.shape:
                continue
            x, y = top(a, ref["ncells"]), top(b, r["ncells"])
            d = y - x
            rx = float(np.sqrt(np.mean(x ** 2)))
            ry = float(np.sqrt(np.mean(y ** 2)))
            rd = float(np.sqrt(np.mean(d ** 2)))
            mx = float(np.abs(d).max())
            rel = rd / rx * 100 if rx > 0 else float("nan")
            rec[key] = {"center_rms": rx, "member_rms": ry,
                        "abs_rms_diff": rd, "abs_max_diff": mx,
                        "rel_rms_diff_pct": rel, "unit": unit}
            print(f"{lab:>14} {rx/scale:>12.4g} {ry/scale:>12.4g} "
                  f"{rd/scale:>12.4g} {mx/scale:>12.4g} {rel:>10.2f}%")
        summary["members"][t] = rec
    if "--json" in sys.argv:
        p = os.path.join(OUT, "v2_sweepA_summary.json")
        json.dump(summary, open(p, "w"), indent=1)
        print(f"\n已写出: {p}")


if __name__ == "__main__":
    main()
