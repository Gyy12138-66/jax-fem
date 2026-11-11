#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 A 收割补充(Kimi 审查缺口 2/3/4,Fable5 2026-08-06)。

在 compare_sweepA.py(vs 中心点)之上补三样:
  缺口 2:成员间两两包络 + 每轴端到端全幅;单调 vs 谷底的判别用
         半步差场夹角:cos((成员A-中心), (成员B-中心))。≈-1 单调穿过中心,
         ≈+1 中心是谷/脊。
  缺口 3:角点(f0005_c07 / f002_c09)在场时,报角点 vs 中心,并做可分性
         残差:RMS(d_corner - d_floor - d_onset) / RMS(d_corner)。
  缺口 4:top-10000 选择的 z 坐标核验:选中单元质心 z 均值 vs 未选中,
         差值须 >= 一个层厚(4.0e-5 m)。

依赖 compare_sweepA.py 的 load/top 语义(同一目录导入),比对量与其一致。
"""
import itertools
import json
import os
import sys

import meshio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_sweepA import CENTER, FIELDS, OUT, PREFIX, TOP, load, top  # noqa: E402

ALL_TAGS = ["f0005_c08", "f001_c08", "f002_c08", "f001_c07", "f001_c09",
            "f0005_c07", "f002_c09", "f0005_c09", "f002_c07"]
KEY_FIELDS = ["vm", "p1", "sxx"]
LAYER = 4.0e-5


def rel_rms(a, b, n):
    x, y = top(a, n), top(b, n)
    d = y - x
    rx = float(np.sqrt(np.mean(x ** 2)))
    return (float(np.sqrt(np.mean(d ** 2))), rx,
            float(np.sqrt(np.mean(d ** 2))) / rx * 100 if rx > 0 else float("nan"))


def diff_field(runs, t, key):
    ref = runs[CENTER]
    r = runs[t]
    return top(r[key], r["ncells"]) - top(ref[key], ref["ncells"])


def zcheck():
    d = os.path.join(OUT, PREFIX + CENTER)
    import glob
    vt = sorted(glob.glob(f"{d}/*.vtu"))
    m = meshio.read(vt[-1])
    cells = m.cells[0].data  # single block, C3D8
    cz = m.points[cells][:, :, 2].mean(axis=1)
    n = cz.size
    sel, rest = cz[n - TOP:], cz[: n - TOP]
    return {"n_cells": int(n), "sel_mean_z": float(sel.mean()),
            "rest_mean_z": float(rest.mean()),
            "sel_min_z": float(sel.min()), "rest_max_z": float(rest.max()),
            "gap_vs_layer": float((sel.mean() - rest.mean()) / LAYER),
            "disjoint": bool(sel.min() >= rest.max() - 1e-12)}


def main():
    runs = {t: r for t in ALL_TAGS
            if (r := load(t)) and r.get("complete") is True}
    have = sorted(runs)
    print("成员在场(仅 complete==true):", " ".join(have))
    res = {"pairs": {}, "axis": {}, "corners": {}, "zcheck": None}

    print("\n== 缺口 2:两两 rel RMS 差(%),顶层 10000 单元 ==")
    hdr = f"{'pair':>24}" + "".join(f"{k:>8}" for k in KEY_FIELDS)
    print(hdr)
    for a, b in itertools.combinations(have, 2):
        row = {}
        for k in KEY_FIELDS:
            _, _, rel = rel_rms(runs[a][k], runs[b][k], runs[a]["ncells"])
            row[k] = round(rel, 2)
        res["pairs"][f"{a}|{b}"] = row
        print(f"{a + ' | ' + b:>24}" + "".join(f"{row[k]:>8.2f}" for k in KEY_FIELDS))

    print("\n== 缺口 2:轴向诊断(端到端全幅 + 半步差场夹角)==")
    axes = {"floor(c08)": ("f0005_c08", "f002_c08"),
            "onset(f001)": ("f001_c07", "f001_c09")}
    for name, (lo, hi) in axes.items():
        if lo not in runs or hi not in runs:
            continue
        ax = {}
        for k in KEY_FIELDS:
            _, _, e2e = rel_rms(runs[lo][k], runs[hi][k], runs[lo]["ncells"])
            da, db = diff_field(runs, lo, k), diff_field(runs, hi, k)
            cos = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db)))
            ax[k] = {"end_to_end_pct": round(e2e, 2), "cos_half_steps": round(cos, 3)}
            print(f"  {name:>12} {k:>4}: 端到端 {e2e:6.2f} %   cos {cos:+.3f}"
                  f"   ({'单调穿过中心' if cos < -0.5 else '中心是谷/脊' if cos > 0.5 else '混合'})")
        res["axis"][name] = ax

    corners = [t for t in ("f0005_c07", "f002_c09") if t in runs]
    if corners:
        print("\n== 缺口 3:角点 vs 中心 + 可分性残差 ==")
        for t in corners:
            fl = "f0005_c08" if t == "f0005_c07" else "f002_c08"
            on = "f001_c07" if t == "f0005_c07" else "f001_c09"
            rec = {}
            for k in KEY_FIELDS:
                dr, rx, rel = rel_rms(runs[CENTER][k], runs[t][k], runs[CENTER]["ncells"])
                dc = diff_field(runs, t, k)
                pred = diff_field(runs, fl, k) + diff_field(runs, on, k)
                sep = float(np.linalg.norm(dc - pred) / np.linalg.norm(dc))
                rec[k] = {"rel_pct": round(rel, 2),
                          "abs_rms_MPa": round(dr / 1e6, 3) if k != "eqp" else dr,
                          "separability_residual": round(sep, 3)}
                print(f"  {t} {k:>4}: vs中心 {rel:6.2f} % ({dr/1e6:6.2f} MPa)"
                      f"   可分性残差 {sep:.3f}")
            res["corners"][t] = rec

    print("\n== 缺口 4:top-10000 的 z 坐标核验 ==")
    z = zcheck()
    res["zcheck"] = z
    print(f"  选中均值 z = {z['sel_mean_z']:.6e}  其余均值 z = {z['rest_mean_z']:.6e}")
    print(f"  差 = {z['gap_vs_layer']:.2f} 个层厚;z 区间不相交 = {z['disjoint']}")

    p = os.path.join(OUT, "v2_sweepA_pairwise.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n已写出: {p}")


if __name__ == "__main__":
    main()
