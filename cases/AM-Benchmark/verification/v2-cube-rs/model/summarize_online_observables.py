#!/usr/bin/env python3
"""读在线观测量 JSONL:出摘要,并给出 10 ms 响应积分后的高温计读数。

在线记录器每个求解步(dt = 76.9 us)写一行。这个脚本做两件事:

  1. **摘要** —— 行数、时间跨度、实际步距、光束在圆内的步数与驻留段数、
     三路读数的范围。判读"采样是不是真的密了"就看这里。
  2. **10 ms 响应积分** —— 真实仪器的读数是 10 ms 窗口上的积分,不是瞬时值。
     采纳口径按窗内**温度**平均(沿用 Balbaa 的分箱口径),双色路按窗内
     **亮度** S1/S2 平均后反演一次(仪器积分的是亮度)。这是 VTU 帧距
     7.69 ms 时根本做不到的事 —— 那时一个 10 ms 箱里只有 0-1 帧。

同时报**每个箱内的样本数**:它就是拍频有没有被消掉的直接证据。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_colour_pyrometer as tc                       # noqa: E402

C2K = 273.15


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{args.jsonl} 是空的")
    meta = (json.loads(args.meta.read_text(encoding="utf-8"))
            if args.meta and args.meta.is_file() else {})
    wl = tuple(meta.get("two_colour", {}).get("wavelengths_m",
                                              tc.DEFAULT_WAVELENGTHS_M))

    t = np.asarray([r["time_s"] for r in rows])
    dt = np.diff(t) if t.size > 1 else np.asarray([0.0])
    inside = np.asarray([r["beam_inside_spot"] for r in rows])
    # 驻留段数 = 光束进出圆的次数。VTU 帧距下这个数是 0(每段只被抓到一次或没抓到)
    dwells = int(np.sum(inside[1:] & ~inside[:-1])) + int(bool(inside[0]))
    hot = [r for r in rows if r["n_hot"] > 0]

    # ---- 10 ms 响应积分 ----
    bin_s = args.bin_ms * 1.0e-3
    bins = {}
    for r in rows:
        bins.setdefault(int(r["time_s"] // bin_s), []).append(r)
    series = []
    for idx in sorted(bins):
        members = bins[idx]
        adopted = [m["avg_K"] for m in members if m["avg_K"] is not None]
        s1 = float(np.mean([m["two_colour_S1"] for m in members]))
        s2 = float(np.mean([m["two_colour_S2"] for m in members]))
        tc_bin = tc.read_accumulated(s1, s2, wavelengths=wl)
        entry = {
            "bin_index": idx,
            "t_center_s": idx * bin_s + 0.5 * bin_s,
            "n_samples": len(members),
            "n_samples_beam_inside": int(sum(m["beam_inside_spot"] for m in members)),
            "avg_K": float(np.mean(adopted)) if adopted else None,
            "mean_n_hot": float(np.mean([m["n_hot"] for m in members])),
            "two_colour_K": tc_bin["T_K"],
            "two_colour_over_range": tc_bin["over_range"],
            "full_spot_avg_K": float(np.mean([m["full_spot_avg_K"] for m in members])),
            "max_K": float(max(m["max_K"] for m in members)),
        }
        if "probe_K" in members[0]:
            probes = np.asarray([m["probe_K"] for m in members], dtype=float)
            entry["probe_mean_K"] = [float(v) for v in probes.mean(axis=0)]
            entry["probe_max_K"] = [float(v) for v in probes.max(axis=0)]
        series.append(entry)

    per_bin = [s["n_samples"] for s in series]
    doc = {
        "source": str(args.jsonl),
        "meta": meta,
        "n_rows": len(rows),
        "t_range_s": [float(t.min()), float(t.max())],
        "step_dt_s": {"min": float(dt.min()), "max": float(dt.max()),
                      "median": float(np.median(dt))},
        "n_steps_beam_inside_spot": int(inside.sum()),
        "n_dwell_segments": dwells,
        "n_steps_with_hot_cells": len(hot),
        "samples_per_10ms_bin": {"min": int(min(per_bin)), "max": int(max(per_bin)),
                                 "median": float(np.median(per_bin))},
        "_why_samples_per_bin_matters": (
            "这是拍频有没有被消掉的直接证据。VTU 帧距 7.69 ms 时每个 10 ms 箱"
            "只有 0-1 帧,区间平均退化为单点采样(D-V2-25);在线记录之后每个箱"
            "应当有上百个样本,10 ms 响应积分才是真的积分"),
        "response_integrated_series": series,
        "reading_definitions": {
            "avg_K": "采纳口径:窗内各步条件平均温度再平均(Balbaa 分箱口径)",
            "two_colour_K": "双色:窗内 S1/S2 亮度先平均再反演一次(仪器积分亮度)",
            "full_spot_avg_K": "全光斑无阈值,诊断下界",
            "probe_*": "Fig 15/16 定点探针(D-V2-27),无条件平均、无 n_hot",
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    print("=== 在线观测量摘要 ===")
    print(f"  {len(rows)} 行,t = {t.min():.6f} .. {t.max():.6f} s")
    print(f"  步距 {dt.min()*1e3:.4f} .. {dt.max()*1e3:.4f} ms(中位 {np.median(dt)*1e3:.4f} ms)")
    print(f"  光束在圆内的步 {int(inside.sum())},驻留段 {dwells};"
          f"  圆内有热单元的步 {len(hot)}")
    print(f"  每个 {args.bin_ms:.0f} ms 箱的样本数:{min(per_bin)} .. {max(per_bin)}"
          f"(中位 {np.median(per_bin):.0f})")
    valid = [s for s in series if s["avg_K"] is not None]
    if valid:
        print(f"  有效箱 {len(valid)} / {len(series)}")
        print(f"  {'t [s]':>8} {'n样本':>6} {'圆内':>5} {'n_hot':>7} "
              f"{'采纳 K':>9} {'双色 K':>9} {'全斑 K':>9} {'峰值 K':>9}")
        for s in valid[:24]:
            f = lambda v: f"{v:9.1f}" if v is not None else "      n/a"
            print(f"  {s['t_center_s']:8.4f} {s['n_samples']:6d} "
                  f"{s['n_samples_beam_inside']:5d} {s['mean_n_hot']:7.1f} "
                  f"{f(s['avg_K'])} {f(s['two_colour_K'])} "
                  f"{f(s['full_spot_avg_K'])} {f(s['max_K'])}")
        if len(valid) > 24:
            print(f"  ... 另有 {len(valid)-24} 个有效箱")
    if "probe_mean_K" in (series[0] if series else {}):
        n = len(series[0]["probe_mean_K"])
        print(f"  定点探针 {n} 个(Fig 15/16 靶子):峰值 "
              + ", ".join(f"{max(s['probe_max_K'][i] for s in series):.1f} K"
                          for i in range(n)))
    if args.output:
        print(f"  写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
