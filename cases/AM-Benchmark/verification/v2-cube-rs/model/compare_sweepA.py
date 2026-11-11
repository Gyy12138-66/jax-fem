#!/usr/bin/env python3
"""扫描 A 的收割。

原版只做"每个成员 vs 中心点",Kimi 审查(2026-08-06)指出两个会影响**登记括号**
的代码缺口,本版补上:

  缺口 2  只对中心点比对 => 系统性低估全幅敏感度。若响应对参数单调,轴向端到端
          上界 ~= 两个半幅之和,而不是单个半幅。本版加 --pairwise:算全部 10 对
          成员间差,给出每轴端到端、全体包络,并顺带判断单调 / 非单调(谷底是否
          落在中心点),这两种物理故事的登记含义不同。
  缺口 4  顶层 10000 单元是按**数组位置**选的(a[n-10000:]),从未验证过它真是
          件内顶层。本版加 verify_top_selection():用单元质心 z 直接核实,
          选中集与未选中集的 z 必须至少相隔一个层厚,否则整份收割的前提就是错的。

规格(Fable5 2026-08-05 锁定,保留):绝对差(MPa)+ 相对差双列,
主指标 von Mises / 最大主应力。
"""
import glob
import itertools
import json
import os
import sys

import meshio
import numpy as np

OUT = "/home/user/work/159/output"
PREFIX = "v2_sweepA_"
CENTER = "f001_c08"
TAGS = ["f0005_c08", "f001_c08", "f002_c08", "f001_c07", "f001_c09"]
FLOOR_AXIS = ["f0005_c08", "f001_c08", "f002_c08"]     # 0.005 -> 0.01 -> 0.02
ONSET_AXIS = ["f001_c07", "f001_c08", "f001_c09"]      # 0.7 -> 0.8 -> 0.9
NX = NY = 100
TOP = NX * NY
LAYER_THICKNESS = 4.0e-5


def cf(m, n):
    d = m.cell_data_dict.get(n)
    return None if d is None else np.asarray(list(d.values())[0]).reshape(-1)


def qavg(m, pat):
    parts = [cf(m, pat.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def max_principal(m):
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


def cell_centroid_z(m):
    """每个单元的质心 z。用于核实'最后 10000 个单元 = 件内顶层'。"""
    pts = np.asarray(m.points)
    for block in m.cells:
        conn = np.asarray(block.data)
        if conn.shape[1] == 8:                     # HEX8
            return pts[conn, 2].mean(axis=1)
    return None


def verify_top_selection(m):
    """缺口 4:顶层选择是按数组位置做的,这里用几何直接核实。"""
    z = cell_centroid_z(m)
    if z is None:
        return {"ok": None, "reason": "找不到 HEX8 单元块"}
    n = z.size
    sel, rest = z[n - TOP:], z[:n - TOP]
    if rest.size == 0:
        return {"ok": None, "reason": "没有未选中单元可比"}
    gap = float(sel.min() - rest.max())
    ok = gap >= LAYER_THICKNESS * 0.5
    return {"ok": bool(ok), "n_cells": int(n),
            "sel_z_mean": float(sel.mean()), "rest_z_mean": float(rest.mean()),
            "sel_z_min": float(sel.min()), "rest_z_max": float(rest.max()),
            "gap_m": gap, "layer_thickness_m": LAYER_THICKNESS}


def load(tag):
    d = os.path.join(OUT, PREFIX + tag)
    vt = sorted(glob.glob(f"{d}/*.vtu"))
    if not vt:
        return None
    m = meshio.read(vt[-1])
    r = {"tag": tag, "vtu": os.path.basename(vt[-1]), "_mesh_checked": None}
    for c in ("xx", "yy", "zz"):
        r[f"s{c}"] = qavg(m, f"stress_quadQ_{c}")
    r["vm"] = qavg(m, "vm_quadQ")
    r["p1"] = max_principal(m)
    r["eqp"] = cf(m, "eq_plastic_strain")
    r["ncells"] = r["vm"].size
    r["_mesh_checked"] = verify_top_selection(m)
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


def diff_stats(ref, mem, key):
    a, b = ref.get(key), mem.get(key)
    if a is None or b is None or a.shape != b.shape:
        return None
    x, y = top(a, ref["ncells"]), top(b, mem["ncells"])
    d = y - x
    rx = float(np.sqrt(np.mean(x ** 2)))
    rd = float(np.sqrt(np.mean(d ** 2)))
    return {"ref_rms": rx, "mem_rms": float(np.sqrt(np.mean(y ** 2))),
            "abs_rms_diff": rd, "abs_max_diff": float(np.abs(d).max()),
            "rel_rms_diff_pct": rd / rx * 100 if rx > 0 else float("nan")}


def main():
    runs = {t: load(t) for t in TAGS}
    have = {t: r for t, r in runs.items() if r}
    summary = {"center": CENTER, "members": {}, "pairwise": {},
               "axis_end_to_end": {}, "envelope": {}, "mesh_check": {}}

    print("=" * 100)
    print("扫描 A 收割(补 Kimi 缺口 2/4 版)")
    print("=" * 100)
    for t in TAGS:
        r = have.get(t)
        print(f"  {t:>10}  " + ("缺" if not r else
              f"{r['vtu']:<26} steps={r.get('steps','-')} complete={r.get('complete','-')}"))

    # --- 缺口 4:顶层选择的几何核实 -------------------------------------
    print("\n[缺口 4] 顶层 10000 单元选择的几何核实(单元质心 z):")
    for t in TAGS:
        r = have.get(t)
        if not r:
            continue
        c = r["_mesh_checked"]
        summary["mesh_check"][t] = c
        if c.get("ok") is None:
            print(f"  {t:>10}  无法核实: {c.get('reason')}")
        else:
            print(f"  {t:>10}  选中 z_min={c['sel_z_min']:.6g} m, "
                  f"未选中 z_max={c['rest_z_max']:.6g} m, 间隔={c['gap_m']:.6g} m "
                  f"(层厚 {LAYER_THICKNESS:g}) -> {'通过' if c['ok'] else '不通过'}")

    if CENTER not in have:
        print("\n中心点缺失")
        return

    ref = have[CENTER]
    # --- 原表:各成员 vs 中心点 ------------------------------------------
    for t in TAGS:
        if t == CENTER or t not in have:
            continue
        print(f"\n--- {t} vs {CENTER}(中心点) ---")
        print(f"{'量':>14} {'中心RMS':>12} {'成员RMS':>12} "
              f"{'绝对RMS差':>12} {'最大绝对差':>12} {'相对RMS差':>11}")
        rec = {}
        for key, lab, scale, unit in FIELDS:
            s = diff_stats(ref, have[t], key)
            if not s:
                continue
            rec[key] = dict(s, unit=unit)
            print(f"{lab:>14} {s['ref_rms']/scale:>12.4g} {s['mem_rms']/scale:>12.4g} "
                  f"{s['abs_rms_diff']/scale:>12.4g} {s['abs_max_diff']/scale:>12.4g} "
                  f"{s['rel_rms_diff_pct']:>10.2f}%")
        summary["members"][t] = rec

    # --- 缺口 2:全部成员对 ---------------------------------------------
    print("\n" + "=" * 100)
    print("[缺口 2] 全部 10 对成员间比对(相对中心点场 RMS 归一)")
    print("=" * 100)
    print(f"{'成员对':>26} " + " ".join(f"{lab:>16}" for _, lab, _, _ in FIELDS[:3]))
    for a, b in itertools.combinations([t for t in TAGS if t in have], 2):
        rec = {}
        cells = []
        for key, lab, scale, unit in FIELDS:
            s = diff_stats(have[a], have[b], key)
            if not s:
                continue
            rec[key] = dict(s, unit=unit)
        for key, lab, scale, unit in FIELDS[:3]:
            s = rec.get(key)
            cells.append(f"{s['rel_rms_diff_pct']:>7.2f}% /{s['abs_rms_diff']/scale:>7.3g}"
                         if s else f"{'-':>16}")
        summary["pairwise"][f"{a}|{b}"] = rec
        print(f"{a + ' vs ' + b:>26} " + " ".join(cells))
    print("  (每格 = 相对RMS差 % / 绝对RMS差 MPa)")

    # --- 轴向端到端 + 单调性 --------------------------------------------
    print("\n各轴端到端(登记括号应取这个,而不是单个半幅):")
    for axis_name, axis in (("floor 0.005->0.02", FLOOR_AXIS),
                            ("onset 0.7->0.9", ONSET_AXIS)):
        lo, mid, hi = axis
        if not all(t in have for t in axis):
            continue
        rec = {}
        for key, lab, scale, unit in FIELDS:
            e2e = diff_stats(have[lo], have[hi], key)
            half_lo = diff_stats(have[mid], have[lo], key)
            half_hi = diff_stats(have[mid], have[hi], key)
            if not (e2e and half_lo and half_hi):
                continue
            # 单调则端到端 ~= 两半幅之和;远小于则说明响应非单调(谷底靠近中心)
            ratio = (e2e["abs_rms_diff"] /
                     (half_lo["abs_rms_diff"] + half_hi["abs_rms_diff"])
                     if (half_lo["abs_rms_diff"] + half_hi["abs_rms_diff"]) > 0
                     else float("nan"))
            rec[key] = {"end_to_end": dict(e2e, unit=unit),
                        "half_lo": dict(half_lo, unit=unit),
                        "half_hi": dict(half_hi, unit=unit),
                        "e2e_over_sum_of_halves": ratio}
        summary["axis_end_to_end"][axis_name] = rec
        print(f"\n  {axis_name}")
        print(f"{'量':>14} {'端到端相对':>11} {'端到端绝对':>12} "
              f"{'半幅和绝对':>12} {'端到端/半幅和':>14} {'形态':>10}")
        for key, lab, scale, unit in FIELDS:
            r = rec.get(key)
            if not r:
                continue
            hs = r["half_lo"]["abs_rms_diff"] + r["half_hi"]["abs_rms_diff"]
            shape = ("单调" if r["e2e_over_sum_of_halves"] > 0.8
                     else "非单调(谷底近中心)" if r["e2e_over_sum_of_halves"] < 0.5
                     else "介于")
            print(f"{lab:>14} {r['end_to_end']['rel_rms_diff_pct']:>10.2f}% "
                  f"{r['end_to_end']['abs_rms_diff']/scale:>12.4g} "
                  f"{hs/scale:>12.4g} {r['e2e_over_sum_of_halves']:>14.2f} {shape:>10}")

    # --- 全体包络 --------------------------------------------------------
    print("\n全体包络(所有成员对的最大值,登记用):")
    print(f"{'量':>14} {'最大相对RMS差':>15} {'最大绝对RMS差':>16} {'出现在':>26}")
    for key, lab, scale, unit in FIELDS:
        best = None
        for pair, rec in summary["pairwise"].items():
            s = rec.get(key)
            if s and (best is None or s["abs_rms_diff"] > best[1]["abs_rms_diff"]):
                best = (pair, s)
        if best:
            summary["envelope"][key] = dict(best[1], unit=unit, pair=best[0])
            print(f"{lab:>14} {best[1]['rel_rms_diff_pct']:>14.2f}% "
                  f"{best[1]['abs_rms_diff']/scale:>16.4g} {best[0]:>26}")

    if "--json" in sys.argv:
        p = os.path.join(OUT, "v2_sweepA_summary_v2.json")
        json.dump(summary, open(p, "w"), indent=1, default=str)
        print(f"\n已写出: {p}")


if __name__ == "__main__":
    main()
