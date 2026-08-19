#!/usr/bin/env python3
"""热闸门观测量:按 Balbaa 的高温计协议从一个运行目录里提取温度轨迹。

协议(Balbaa2022 Sec 3.3,逐字):
  "the elements temperatures above 1000 degC, i.e. the lower limit of the
   pyrometer, within a 2 mm diameter circle were averaged and compared to
   the pyrometer measurements ... the temperatures calculated from the model
   are extracted and averaged at an increment of 10 ms"

即:2 mm 圆内、温度高于 1000 degC 的**单元**取平均,10 ms 一个增量。

三处他没写死、我们必须自己定并登记的(D-V2-23):
  (a) 圆心位置 —— 默认取网格窗中心(高温计对准件中心是自然读法);
  (b) 深度 —— 高温计是光学表面仪器,默认只取顶层粉层单元;--all-depths 可关;
  (c) "averaged at an increment of 10 ms" 是**区间平均**还是**每 10 ms 采样一次**
      —— 两者都算,默认采用区间平均,差值一并输出。

另外单独报一路 `range_limited`:该高温计量程上限 3000 degC,超出量程的单元
不在其可测范围内。Balbaa 只写了下限,所以这一路是**诊断**不是采纳口径 ——
as-is 臂峰值远在量程之上,这个差值是判读闸门时必须看到的。
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import meshio
import numpy as np

C2K = 273.15


def load_step_times(run_dir):
    """求解器自己记的 step -> time(扫描 dt 由路径行距决定,不是 --dt)。"""
    path = os.path.join(run_dir, "path_used.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{run_dir} 缺 path_used.csv,无法给帧配时间")
    times, laser_on = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            times[step] = float(row["time"])
            if "laser_on" in row and str(row["laser_on"]).strip():
                laser_on[step] = float(row["laser_on"]) > 0.5
    return times, laser_on


def step_of(path):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else -1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--arm", default=None, help="臂标签(parity / asis)")
    ap.add_argument("--spot-center", default=None,
                    help="圆心 x,y [m];缺省 = 网格 xy 包围盒中心")
    ap.add_argument("--spot-diameter", type=float, default=2.0e-3)
    ap.add_argument("--threshold-C", type=float, default=1000.0,
                    help="高温计量程下限")
    ap.add_argument("--range-max-C", type=float, default=3000.0,
                    help="量程上限,仅用于 range_limited 诊断路")
    ap.add_argument("--bin-ms", type=float, default=10.0, help="高温计响应时间")
    ap.add_argument("--all-depths", action="store_true",
                    help="不限顶层,全深度参与平均")
    ap.add_argument("--layer-top-z", type=float, default=None,
                    help="粉层底面 z [m];缺省 = 网格 z 上界减一个层厚")
    ap.add_argument("--layer-thickness", type=float, default=40.0e-6)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    vtus = sorted(glob.glob(os.path.join(args.run_dir, "*.vtu")))
    if not vtus:
        raise SystemExit(f"{args.run_dir} 内没有 VTU")
    times, laser_on = load_step_times(args.run_dir)

    first = meshio.read(vtus[0])
    pts = np.asarray(first.points)
    conn = None
    for block in first.cells:
        if block.data.shape[1] == 8:
            conn = np.asarray(block.data)
            break
    if conn is None:
        raise SystemExit("找不到 HEX8 单元块")
    centers = pts[conn].mean(axis=1)

    if args.spot_center:
        cx, cy = (float(v) for v in args.spot_center.split(","))
    else:
        cx = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
        cy = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    radius = 0.5 * args.spot_diameter
    in_circle = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2 <= radius ** 2

    z_top = pts[:, 2].max()
    layer_bottom = (args.layer_top_z if args.layer_top_z is not None
                    else z_top - args.layer_thickness)
    in_depth = np.ones(len(centers), bool) if args.all_depths else centers[:, 2] >= layer_bottom
    gauge = in_circle & in_depth
    if not gauge.any():
        raise SystemExit("量测集为空:检查 --spot-center / --spot-diameter / 深度选项")

    thr_K = args.threshold_C + C2K
    ceil_K = args.range_max_C + C2K

    frames = []
    for vtu in vtus:
        step = step_of(vtu)
        t = times.get(step)
        if t is None:
            continue
        m = meshio.read(vtu)
        T_node = np.asarray(m.point_data["T"]).reshape(-1)
        T_cell = T_node[conn].mean(axis=1)          # 单元温度 = 8 节点均值
        vals = T_cell[gauge]
        hot = vals >= thr_K
        inrange = hot & (vals <= ceil_K)
        frames.append({
            "step": step, "time_s": t,
            "laser_on": bool(laser_on.get(step, False)),
            "n_hot": int(hot.sum()),
            "avg_K": float(vals[hot].mean()) if hot.any() else None,
            "max_K": float(vals.max()),
            "n_in_range": int(inrange.sum()),
            "avg_range_limited_K": float(vals[inrange].mean()) if inrange.any() else None,
            "n_over_range": int((hot & (vals > ceil_K)).sum()),
        })
    if not frames:
        raise SystemExit("没有一帧能在 path_used.csv 里配到时间")
    frames.sort(key=lambda f: f["time_s"])

    # ---- 10 ms 分箱 ----
    # 箱号锚在 **t = 0**,不是锚在本臂的首帧:两臂的首帧时刻只要差一点,
    # 按各自首帧编号就会让 compare_thermal_gate.py 的逐箱相减错位一格,
    # 而且错得很安静。锚在 0 之后箱号在两臂间天然可比。
    bin_s = args.bin_ms * 1e-3
    bins = {}
    for f in frames:
        bins.setdefault(int(f["time_s"] // bin_s), []).append(f)
    series = []
    for idx in sorted(bins):
        members = bins[idx]
        lo = idx * bin_s
        center = lo + 0.5 * bin_s
        hot = [m for m in members if m["avg_K"] is not None]
        rng = [m for m in members if m["avg_range_limited_K"] is not None]
        nearest = min(members, key=lambda m: abs(m["time_s"] - center))
        series.append({
            "bin_index": idx,
            "t_lo_s": lo, "t_center_s": center,
            "n_frames": len(members),
            "frame_times_s": [m["time_s"] for m in members],
            # 采纳口径:箱内区间平均
            "avg_K": float(np.mean([m["avg_K"] for m in hot])) if hot else None,
            # 备选读法:取最接近箱心的那一帧
            "nearest_frame_avg_K": nearest["avg_K"],
            "avg_range_limited_K": (float(np.mean([m["avg_range_limited_K"] for m in rng]))
                                    if rng else None),
            "max_K": float(max(m["max_K"] for m in members)),
            "mean_n_hot": float(np.mean([m["n_hot"] for m in members])),
            "any_laser_on": any(m["laser_on"] for m in members),
        })

    dts = np.diff([f["time_s"] for f in frames]) if len(frames) > 1 else np.array([0.0])
    # 判读只依赖"圆内有单元过 1000 degC"的那些帧,帧距警告因此按热帧算:
    # 窗外的粗步长本来就该稀疏,拿全局最大帧距报警只会天天误报。
    hot_t = [f["time_s"] for f in frames if f["n_hot"] > 0]
    hot_dts = np.diff(hot_t) if len(hot_t) > 1 else np.array([0.0])
    warnings = []
    if hot_dts.size and float(hot_dts.max()) > bin_s:
        warnings.append(
            f"有效(圆内有热单元)帧的最大帧距 {float(hot_dts.max())*1e3:.2f} ms > "
            f"分箱 {args.bin_ms} ms:这段时间的 10 ms 箱只有 0-1 帧,"
            "区间平均退化为单点采样。生产运行请把 --thermal-output-every "
            "调到热段帧距 <= 10 ms")
    empty = [s["bin_index"] for s in series if s["avg_K"] is None]
    if empty:
        warnings.append(f"{len(empty)} 个箱内没有单元超过 {args.threshold_C} degC"
                        "(高温计在这些时刻同样什么都测不到,不是缺陷)")
    over = sum(f["n_over_range"] for f in frames)
    if over:
        warnings.append(
            f"共 {over} 个单元-帧超出高温计量程上限 {args.range_max_C} degC:"
            "采纳口径按 Balbaa 原文只设下限,故它们仍进了平均;"
            "range_limited 一路给出剔除它们后的对照")

    doc = {
        "arm": args.arm or os.path.basename(os.path.normpath(args.run_dir)),
        "run_dir": os.path.abspath(args.run_dir),
        "protocol": {
            "_source": "Balbaa2022 Sec 3.3 p.13",
            "threshold_C": args.threshold_C,
            "spot_center_m": [cx, cy], "spot_diameter_m": args.spot_diameter,
            "bin_ms": args.bin_ms,
            "bin_index_anchor": "t = 0(两臂箱号可直接对齐)",
            "depth_scope": "全深度" if args.all_depths else f"顶层(z >= {layer_bottom:.6e} m)",
            "cell_temperature": "单元 8 节点温度的算术平均",
            "adopted_bin_reading": "箱内区间平均(nearest_frame_avg_K 为备选读法)",
            "registered_as": "D-V2-23(圆心/深度/分箱读法三项均为我们的约定)",
        },
        "gauge_cells": int(gauge.sum()),
        "gauge_cells_in_circle": int(in_circle.sum()),
        "n_frames": len(frames),
        "frame_dt_s": {"min": float(dts.min()), "max": float(dts.max()),
                       "median": float(np.median(dts)),
                       "max_between_hot_frames": float(hot_dts.max())},
        "t_range_s": [frames[0]["time_s"], frames[-1]["time_s"]],
        "peak_K": float(max(f["max_K"] for f in frames)),
        "peak_avg_K": max((s["avg_K"] for s in series if s["avg_K"] is not None),
                          default=None),
        "series": series,
        "frames": frames,
        "warnings": warnings,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"=== 高温计协议提取:{doc['arm']} ===")
    print(f"  量测单元 {doc['gauge_cells']}(圆内 {doc['gauge_cells_in_circle']}),"
          f"{doc['protocol']['depth_scope']}")
    print(f"  圆心 ({cx*1e3:.3f}, {cy*1e3:.3f}) mm  直径 {args.spot_diameter*1e3:.1f} mm")
    print(f"  帧 {len(frames)},帧距 {dts.min()*1e3:.3f} .. {dts.max()*1e3:.3f} ms,"
          f"t {frames[0]['time_s']:.4f} .. {frames[-1]['time_s']:.4f} s")
    print(f"  10 ms 箱 {len(series)},其中有效 {sum(1 for s in series if s['avg_K'])}")
    if doc["peak_avg_K"]:
        print(f"  峰值箱均温 {doc['peak_avg_K']:.1f} K = {doc['peak_avg_K']-C2K:.1f} degC;"
              f"  单元峰温 {doc['peak_K']:.1f} K = {doc['peak_K']-C2K:.1f} degC")
    for w in warnings:
        print(f"  [WARN] {w}")
    if args.output:
        print(f"  写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
