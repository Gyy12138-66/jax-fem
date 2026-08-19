#!/usr/bin/env python3
"""热闸门对比:keff 双臂(parity / as-is) x 高温计实测 x Balbaa 的 ABAQUS。

**一等输出是两臂之差**(Fable5 2026-08-18 指定):Marangoni 升级决策
(本 issue metadata,DEFERRED)的输入就是"加不加 keff 到底差多少",
所以两臂差不是副产品,是本脚本的第一张表 —— 即使两臂各自都对不上实测,
两臂之差依然是有效结论。

其余两条腿沿用 V1 的三角形口径:
  code-to-experiment  各臂 vs 高温计实测点(带数字化读数不确定度)
  code-to-code        各臂 vs Balbaa 自己的数值点
并且把 **Balbaa 自己的数值-实测差**也算出来当尺子 —— 我们的差要和他的差比,
而不是和零比(SPEC "Acceptance framing":没有可调的阈值)。

输入:两个 analyze_pyrometer.py 的 JSON + digitize_fig14_pyrometer.py 的 JSON。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

C2K = 273.15


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def series_arrays(doc, key="avg_K"):
    """有效箱的 (t, T);无效箱(圆内无单元过 1000 degC)不参与。"""
    t, v = [], []
    for s in doc["series"]:
        if s.get(key) is not None:
            t.append(s["t_center_s"])
            v.append(s[key])
    return np.asarray(t), np.asarray(v)


def sample_at(t_arr, v_arr, t_query, max_gap_s):
    """把一条箱序列取到 t_query。最近的有效箱超过 max_gap_s 就返回 None ——
    宁可报缺,不外推(外推出来的点会被当成模型预测读,那是假证据)。"""
    if t_arr.size == 0:
        return None, None
    i = int(np.argmin(np.abs(t_arr - t_query)))
    gap = float(abs(t_arr[i] - t_query))
    if gap > max_gap_s:
        return None, gap
    if t_arr.size == 1:
        return float(v_arr[0]), gap
    return float(np.interp(t_query, t_arr, v_arr)), gap


def stats(diff):
    d = np.asarray([x for x in diff if x is not None], float)
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "mean": float(d.mean()),
            "rms": float(np.sqrt(np.mean(d ** 2))),
            "max_abs": float(np.abs(d).max()),
            "min": float(d.min()), "max": float(d.max())}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", type=Path, required=True, help="keff parity 臂 JSON")
    ap.add_argument("--asis", type=Path, required=True, help="as-is 臂 JSON")
    ap.add_argument("--experiment", type=Path, required=True, help="Fig 14 数字化 JSON")
    ap.add_argument("--max-gap-ms", type=float, default=10.0,
                    help="取样时允许的最近有效箱时间间隔上限")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    parity, asis, fig14 = load(args.parity), load(args.asis), load(args.experiment)
    max_gap = args.max_gap_ms * 1e-3

    tp, vp = series_arrays(parity)
    ta, va = series_arrays(asis)

    # ---------------- 一等输出:两臂之差 ----------------
    pbins = {s["bin_index"]: s for s in parity["series"]}
    abins = {s["bin_index"]: s for s in asis["series"]}
    shared = sorted(set(pbins) & set(abins))
    # 逐箱相减的前提是两臂箱号同锚(analyze_pyrometer.py 锚在 t=0)。
    # 万一以后有人改了锚点,这里当场炸,而不是安静地错位一格。
    for idx in shared:
        if abs(pbins[idx]["t_center_s"] - abins[idx]["t_center_s"]) > 1e-9:
            raise SystemExit(
                f"两臂箱 {idx} 的中心时刻不一致 "
                f"({pbins[idx]['t_center_s']} vs {abins[idx]['t_center_s']}):"
                "箱号锚点不同,逐箱相减无效")
    per_bin, both_valid = [], []
    for idx in shared:
        p, a = pbins[idx]["avg_K"], abins[idx]["avg_K"]
        d = (p - a) if (p is not None and a is not None) else None
        per_bin.append({"bin_index": idx, "t_center_s": pbins[idx]["t_center_s"],
                        "parity_K": p, "asis_K": a, "parity_minus_asis_K": d})
        both_valid.append(d)
    arm_diff = {
        "_why_first": "Marangoni 升级决策(issue metadata,DEFERRED)的直接输入",
        "shared_bins": len(shared),
        "bins_both_valid": sum(1 for d in both_valid if d is not None),
        "parity_minus_asis_K": stats(both_valid),
        "peak_bin_avg_K": {"parity": parity.get("peak_avg_K"),
                           "asis": asis.get("peak_avg_K")},
        "peak_cell_K": {"parity": parity.get("peak_K"), "asis": asis.get("peak_K")},
        "per_bin": per_bin,
    }
    for field, node in (("peak_bin_avg_K", arm_diff["peak_bin_avg_K"]),
                        ("peak_cell_K", arm_diff["peak_cell_K"])):
        if node["parity"] is not None and node["asis"] is not None:
            node["parity_minus_asis"] = node["parity"] - node["asis"]

    # ---------------- 三角形的另外两条腿 ----------------
    legs = []
    for pair in fig14["pairs"]:
        t = pair["time_s"]
        exp_K = pair["experimental_C"] + C2K
        bal_K = pair["balbaa_numerical_C"] + C2K
        row = {"time_s": t, "experimental_K": exp_K, "balbaa_numerical_K": bal_K,
               "balbaa_minus_experiment_K": bal_K - exp_K}
        for tag, (t_arr, v_arr) in (("parity", (tp, vp)), ("asis", (ta, va))):
            val, gap = sample_at(t_arr, v_arr, t, max_gap)
            row[f"{tag}_K"] = val
            row[f"{tag}_gap_s"] = gap
            row[f"{tag}_minus_experiment_K"] = (val - exp_K) if val is not None else None
            row[f"{tag}_minus_balbaa_K"] = (val - bal_K) if val is not None else None
        legs.append(row)

    # 数字化读数不确定度:实测腿的残差要和它比
    u_exp = max((p["temperature_C_uncertainty"] for p in fig14["experimental"]),
                default=None)
    u_num = max((p["temperature_C_uncertainty"] for p in fig14["balbaa_numerical"]),
                default=None)

    summary = {
        "code_to_experiment": {
            "parity": stats([r["parity_minus_experiment_K"] for r in legs]),
            "asis": stats([r["asis_minus_experiment_K"] for r in legs]),
            "balbaa_own_abaqus": stats([r["balbaa_minus_experiment_K"] for r in legs]),
            "_yardstick": "我们的残差与 Balbaa 自己的 ABAQUS 残差比,不与零比"
                          "(SPEC Acceptance framing:没有可调阈值)",
        },
        "code_to_code": {
            "parity": stats([r["parity_minus_balbaa_K"] for r in legs]),
            "asis": stats([r["asis_minus_balbaa_K"] for r in legs]),
        },
        "digitization_read_off_uncertainty_C": {
            "experimental_max": u_exp, "balbaa_numerical_max": u_num,
            "_note": "小于该值的残差差异没有判读意义",
        },
    }

    doc = {
        "id": "V2-thermal-gate-comparison",
        "condition": fig14.get("condition"),
        "arms": {"parity": {"json": str(args.parity), "run": parity.get("run_dir"),
                            "protocol": parity.get("protocol")},
                 "asis": {"json": str(args.asis), "run": asis.get("run_dir"),
                          "protocol": asis.get("protocol")}},
        "experiment_source": fig14.get("provenance"),
        "scope_finding": fig14.get("SCOPE_FINDING"),
        "ARM_DIFFERENCE_FIRST_CLASS": arm_diff,
        "triangle_at_figure_sample_times": legs,
        "summary": summary,
        "warnings": (parity.get("warnings", []) + asis.get("warnings", [])),
        "no_threshold_note": "本脚本不判 pass/fail:零标定,差异如实上报",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print("=== 热闸门对比 ===")
    d = arm_diff["parity_minus_asis_K"]
    print(f"[一等输出] 两臂差 parity - asis:共享箱 {arm_diff['shared_bins']},"
          f"双方有效 {arm_diff['bins_both_valid']}")
    if d.get("n"):
        print(f"  均值 {d['mean']:+.1f} K  RMS {d['rms']:.1f} K  "
              f"最大绝对 {d['max_abs']:.1f} K  区间 [{d['min']:+.1f}, {d['max']:+.1f}] K")
    for k in ("peak_bin_avg_K", "peak_cell_K"):
        n = arm_diff[k]
        if n.get("parity_minus_asis") is not None:
            print(f"  {k}: parity {n['parity']:.1f} - asis {n['asis']:.1f} "
                  f"= {n['parity_minus_asis']:+.1f} K")
    print(f"\n[三角形] 在 Fig 14 的 {len(legs)} 个采样时刻(K):")
    print(f"  {'t [s]':>7} {'实测':>8} {'Balbaa':>8} {'parity':>8} {'asis':>8}"
          f" {'par-exp':>9} {'asis-exp':>9} {'Bal-exp':>9}")
    for r in legs:
        fmt = lambda v: f"{v:8.1f}" if v is not None else "     n/a"
        fmt9 = lambda v: f"{v:+9.1f}" if v is not None else "      n/a"
        print(f"  {r['time_s']:7.4f} {r['experimental_K']:8.1f} "
              f"{r['balbaa_numerical_K']:8.1f} {fmt(r['parity_K'])} {fmt(r['asis_K'])}"
              f" {fmt9(r['parity_minus_experiment_K'])}"
              f" {fmt9(r['asis_minus_experiment_K'])}"
              f" {fmt9(r['balbaa_minus_experiment_K'])}")
    print("\n[汇总] 对实测的残差 RMS / 最大绝对 (K):")
    for tag, s in summary["code_to_experiment"].items():
        if isinstance(s, dict) and s.get("n"):
            print(f"  {tag:18s} n={s['n']}  RMS {s['rms']:8.1f}  max|.| {s['max_abs']:8.1f}")
    print(f"  数字化读数不确定度 +/-{u_exp:.1f} degC(实测点):小于此的差异无判读意义")
    for w in doc["warnings"]:
        print(f"  [WARN] {w}")
    print(f"\n写出 {args.output}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] 没装 matplotlib,跳过出图(JSON 已写出)")
            return 0
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        if tp.size:
            ax.plot(tp, vp - C2K, "-", lw=1.2, color="tab:red", label="ours: keff parity")
        if ta.size:
            ax.plot(ta, va - C2K, "-", lw=1.2, color="tab:blue", label="ours: as-is")
        ex_t = [r["time_s"] for r in legs]
        ex_v = [r["experimental_K"] - C2K for r in legs]
        ba_v = [r["balbaa_numerical_K"] - C2K for r in legs]
        ax.errorbar(ex_t, ex_v, yerr=u_exp or 0.0, fmt="s", ms=8, mfc="none",
                    color="k", label="Balbaa Fig 14 experiment (digitized)")
        ax.errorbar(ex_t, ba_v, yerr=u_num or 0.0, fmt="^", ms=8, mfc="none",
                    color="tab:orange", label="Balbaa Fig 14 ABAQUS (digitized)")
        ax.axhline(1000.0, ls=":", color="gray", lw=1)
        ax.set_ylabel("Temperature (degC)")
        ax.set_title("V2 thermal gate: keff two-arm vs two-colour pyrometer\n"
                     "220 W / 650 mm/s / hatch 0.12 mm / layer 0.04 mm")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        dt_ = [b["t_center_s"] for b in per_bin if b["parity_minus_asis_K"] is not None]
        dv_ = [b["parity_minus_asis_K"] for b in per_bin
               if b["parity_minus_asis_K"] is not None]
        ax.plot(dt_, dv_, "-", color="tab:purple", lw=1.2)
        ax.axhline(0.0, ls="-", color="k", lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("parity - as-is (K)")
        ax.set_title("FIRST-CLASS OUTPUT: arm-to-arm difference", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"写出 {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
