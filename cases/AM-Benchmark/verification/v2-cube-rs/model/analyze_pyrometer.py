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

## 三路读数(2026-08-20 起为一等输出,IET-20)

场级分析(Opus5 03:57,IET-19)证明采纳口径的形状几乎全部来自它自己的分母:
观测窗内场的峰温趋势只有 -27 K,而采纳读数掉了 919 K,同期 n_hot 从 11 涨到
140。采纳口径因此是**复现 Balbaa 协议的主尺**(必须原样保留,一个字不改),
但它不是仪器的模型,也不是场的下界。三路并列、互不替换:

| 路 | 字段 | 干什么用 | 谁定的口径 |
|---|---|---|---|
| 采纳(条件平均) | `avg_K` | code-to-code:复现他的协议与他的数值曲线 | Balbaa Sec 3.3 |
| 双色合成 | `two_colour_K` | code-to-experiment:按仪器自己的物理读 | 我们,D-V2-24 |
| 全光斑无阈值 | `full_spot_avg_K` | 诊断下界:分母取满时的读数 | 我们,D-V2-24 |

三路共用**同一套几何**(同圆心、同直径、同顶层、同单元温度定义),所以可以
逐列直接相比 —— 差异只可能来自读数定义本身。

零标定:三路都没有可向实测回调的参数;阈值/量程是仪器规格,不是旋钮。
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

import two_colour_pyrometer as tc

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
    ap.add_argument("--observation-window", default="0.45,0.90",
                    help="在线观测窗 t0,t1 [s]，仅用于协议一致性校验")
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
    ap.add_argument("--tc-wavelengths-um", default="0.95,1.05",
                    help="双色合成读数的两个波长 [um](短波在前)")
    ap.add_argument("--tc-sensitivity", action="store_true",
                    help="额外输出波长分离的敏感性括号(峰值帧上)")
    ap.add_argument("--online-summary", type=Path, default=None,
                    help="求解步级在线观测摘要；给出后作为正式 10 ms series，VTU 仅作诊断")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    wl = tuple(float(v) * 1e-6 for v in args.tc_wavelengths_um.split(","))
    if len(wl) != 2:
        raise SystemExit("--tc-wavelengths-um 需要两个值,例如 0.95,1.05")
    # 自检先跑:均匀场必须被精确还原。不过就不许出读数 —— 一个反演错了符号的
    # 仪器模型会安静地产出看起来合理的曲线,那比没有这一路更糟。
    tc_ok, tc_rows = tc.uniform_field_self_check(wavelengths=wl)
    if not tc_ok:
        raise SystemExit("双色合成读数的均匀场自检未通过,拒绝出读数:\n"
                         + json.dumps(tc_rows, indent=2, ensure_ascii=False))

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
        # ---- 三路读数,同一批 vals、同一套几何 ----
        # (1) 采纳口径 avg_K —— Balbaa Sec 3.3,原样保留;
        # (2) 双色合成 —— 仪器物理,无阈值,亮度积分后反演;
        # (3) 全光斑无阈值算术平均 —— 诊断下界,分母取满。
        tcr = tc.read_field(vals, wavelengths=wl,
                            range_min_C=args.threshold_C,
                            range_max_C=args.range_max_C)
        frames.append({
            "step": step, "time_s": t,
            "laser_on": bool(laser_on.get(step, False)),
            "n_hot": int(hot.sum()),
            "avg_K": float(vals[hot].mean()) if hot.any() else None,
            "max_K": float(vals.max()),
            "n_in_range": int(inrange.sum()),
            "avg_range_limited_K": float(vals[inrange].mean()) if inrange.any() else None,
            "n_over_range": int((hot & (vals > ceil_K)).sum()),
            # --- 路 2:双色辐射合成 ---
            "two_colour_K": tcr["T_K"],
            "two_colour_S1": tcr["S1"], "two_colour_S2": tcr["S2"],
            "two_colour_over_range": tcr["over_range"],
            "two_colour_under_range": tcr["under_range"],
            "two_colour_cold_tail_frac": tcr["cold_tail_frac"],
            # --- 路 3:全光斑无阈值平均 ---
            "full_spot_avg_K": float(vals.mean()),
            "n_gauge_cells": int(vals.size),
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
        # 双色的箱读法:仪器 10 ms 响应积分的是**亮度**,不是温度。所以先把
        # 箱内各帧的 S1/S2 平均,再反演一次。先逐帧反演再平均温度是另一个量
        # (也一并给出,两者之差就是这个读法选择的曝光量)。
        S1 = float(np.mean([m["two_colour_S1"] for m in members]))
        S2 = float(np.mean([m["two_colour_S2"] for m in members]))
        tc_bin = tc.read_accumulated(S1, S2, wavelengths=wl,
                                     range_min_C=args.threshold_C,
                                     range_max_C=args.range_max_C)
        tc_frames = [m["two_colour_K"] for m in members if m["two_colour_K"] is not None]
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
            # --- 路 2:双色辐射合成(箱内亮度平均后反演) ---
            "two_colour_K": tc_bin["T_K"],
            "two_colour_over_range": tc_bin["over_range"],
            "two_colour_under_range": tc_bin["under_range"],
            "two_colour_frame_mean_K": (float(np.mean(tc_frames)) if tc_frames else None),
            "n_frames_two_colour_over_range": int(
                sum(1 for m in members if m["two_colour_over_range"])),
            # --- 路 3:全光斑无阈值平均 ---
            "full_spot_avg_K": float(np.mean([m["full_spot_avg_K"] for m in members])),
        })

    online_summary = None
    if args.online_summary is not None:
        if not args.online_summary.is_file():
            raise SystemExit(f"在线观测摘要不存在: {args.online_summary}")
        online_summary = json.loads(args.online_summary.read_text(encoding="utf-8"))
        online_meta = online_summary.get("meta", {})
        expected_meta = {
            "spot_center_m": [float(cx), float(cy)],
            "spot_diameter_m": float(args.spot_diameter),
            "threshold_C": float(args.threshold_C),
            "range_max_C": float(args.range_max_C),
            "window_s": [float(args.observation_window.split(",")[0]),
                         float(args.observation_window.split(",")[1])],
        }
        for key, expected in expected_meta.items():
            actual = online_meta.get(key)
            if actual is None or not np.allclose(actual, expected, rtol=0.0,
                                                  atol=1.0e-12):
                raise SystemExit(
                    f"在线观测摘要协议不匹配 {key}: {actual!r} != {expected!r}")
        actual_wl = online_meta.get("two_colour", {}).get("wavelengths_m")
        if actual_wl is None or not np.allclose(actual_wl, wl, rtol=0.0,
                                                atol=1.0e-12):
            raise SystemExit(f"在线观测摘要波长不匹配: {actual_wl!r} != {wl!r}")
        expected_scope = ("all depths" if args.all_depths else
                          f"top layer (z >= {layer_bottom:.6e} m)")
        if online_meta.get("depth_scope") != expected_scope:
            raise SystemExit("在线观测摘要深度口径不匹配: "
                             f"{online_meta.get('depth_scope')!r} != {expected_scope!r}")
        if not np.isclose(float(online_summary.get("response_bin_ms", np.nan)),
                          args.bin_ms, rtol=0.0, atol=1.0e-12):
            raise SystemExit("在线观测摘要响应箱宽与 --bin-ms 不匹配")
        online_series = online_summary.get("response_integrated_series")
        if not isinstance(online_series, list) or not online_series:
            raise SystemExit("在线观测摘要缺少非空 response_integrated_series")
        required = {"bin_index", "t_center_s", "avg_K", "two_colour_K",
                    "full_spot_avg_K", "max_K"}
        for index, item in enumerate(online_series):
            missing = required - set(item)
            if missing:
                raise SystemExit(f"在线观测摘要第 {index} 箱缺字段: {sorted(missing)}")
            item.setdefault("n_frames", item.get("n_samples", 0))
            item.setdefault("any_laser_on", bool(item.get("n_samples_beam_inside", 0)))
            item.setdefault("avg_range_limited_K", None)
            item.setdefault("nearest_frame_avg_K", None)
        if len(online_series) > 2:
            incomplete = [item["bin_index"] for item in online_series[1:-1]
                          if not np.isclose(item.get("coverage_fraction", np.nan),
                                            1.0, rtol=0.0, atol=1.0e-6)]
            if incomplete:
                raise SystemExit(
                    f"在线观测摘要内部响应箱覆盖不完整: {incomplete[:10]}")
        # The online recorder samples every accepted solver step.  It is the
        # production reading; sparse VTUs remain below only for diagnostics and
        # the optional wavelength-sensitivity field reconstruction.
        series = online_series

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
    tc_over = sum(1 for s in series if s.get("two_colour_over_range", False))
    if tc_over:
        warnings.append(
            f"{tc_over} 个 10 ms 箱的双色合成读数超出仪器上限 "
            f"{args.range_max_C} degC。真机在 Fig 14 的五个时刻**没有**超量程"
            "(实测 1389-1470 degC),所以这不是读数口径问题,是场的绝对水平问题")
    cold = [f["two_colour_cold_tail_frac"] for f in frames
            if f["two_colour_cold_tail_frac"] is not None and f["n_hot"] > 0]
    cold_max = max(cold) if cold else None

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
            "adopted_bin_reading": ("求解步级 dt 覆盖加权响应积分"
                                    if online_summary is not None else
                                    "稀疏 VTU 箱内平均(nearest_frame_avg_K 为备选读法)"),
            "online_summary": (str(args.online_summary) if args.online_summary else None),
            "registered_as": "D-V2-23(圆心/深度/分箱读法三项均为我们的约定)",
        },
        "three_readings": {
            "_why": "采纳口径的形状几乎全部来自它自己的分母(场峰温窗内趋势 "
                    "-27 K,采纳读数掉 919 K,同期 n_hot 11 -> 140)。三路并列、"
                    "互不替换:采纳口径复现他的协议,双色合成按仪器物理读,"
                    "全光斑给分母取满时的下界",
            "adopted_conditional_average": {
                "field": "avg_K",
                "definition": f"2 mm 圆内、顶层、T >= {args.threshold_C} degC 单元的算术平均",
                "source": "Balbaa2022 Sec 3.3 逐字协议 —— 一个字不改",
                "use": "code-to-code(复现他的数值曲线)",
            },
            "two_colour_synthetic": dict(
                tc.PROTOCOL_NOTE,
                field="two_colour_K",
                wavelengths_m=list(wl),
                use="code-to-experiment(按仪器自己的物理读)",
                uniform_field_self_check=tc_rows,
                uniform_field_self_check_passed=tc_ok,
                max_cold_tail_fraction=cold_max,
                cold_tail_meaning="低于 1000 degC 的单元对短波通道信号的最大贡献占比;"
                                  "它证明'不设阈值'在 1 um 波段无害,而不是断言它无害",
            ),
            "full_spot_unthresholded": {
                "field": "full_spot_avg_K",
                "definition": "同一 2 mm 圆内、同一顶层,全部单元的算术平均(无阈值)",
                "use": "诊断下界 —— 分母取满时读数是多少",
                "registered_as": "D-V2-24",
            },
        },
        "gauge_cells": int(gauge.sum()),
        "gauge_cells_in_circle": int(in_circle.sum()),
        "n_frames": len(frames),
        "series_source": ("online_solver_steps" if online_summary is not None
                          else "sparse_vtu_frames"),
        "frame_dt_s": {"min": float(dts.min()), "max": float(dts.max()),
                       "median": float(np.median(dts)),
                       "max_between_hot_frames": float(hot_dts.max())},
        "t_range_s": [frames[0]["time_s"], frames[-1]["time_s"]],
        "peak_K": float(max(f["max_K"] for f in frames)),
        "peak_avg_K": max((s["avg_K"] for s in series if s["avg_K"] is not None),
                          default=None),
        "peak_two_colour_K": max(
            (s["two_colour_K"] for s in series if s["two_colour_K"] is not None),
            default=None),
        "peak_full_spot_avg_K": max(
            (s["full_spot_avg_K"] for s in series
             if s["full_spot_avg_K"] is not None), default=None),
        "n_bins_two_colour_over_range": tc_over,
        "series": series,
        "frames": frames,
        "warnings": warnings,
    }

    if args.tc_sensitivity:
        # 波长分离是我们定的(论文只写 "Si/Si ~1 um")。在采纳口径读数最高的
        # 那一帧上给出括号 —— 那也是这个选择最有可能改变判读的地方。
        peak_frame = max((f for f in frames if f["avg_K"] is not None),
                         key=lambda f: f["avg_K"], default=None)
        if peak_frame is not None:
            m = meshio.read(next(v for v in vtus if step_of(v) == peak_frame["step"]))
            T_cell = np.asarray(m.point_data["T"]).reshape(-1)[conn].mean(axis=1)
            doc["two_colour_wavelength_sensitivity"] = {
                "at_frame_time_s": peak_frame["time_s"],
                "_why": "论文只写 Si/Si ~1 um,两通道的分离由我们定;这是该选择的曝光量",
                "bracket": tc.sensitivity(T_cell[gauge]),
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
    print(f"  双色合成 {wl[0]*1e6:.2f}/{wl[1]*1e6:.2f} um,均匀场自检 "
          f"{'通过' if tc_ok else '失败'}(最大 |err| "
          f"{max(r['abs_error_K'] for r in tc_rows):.2e} K)")
    if cold_max is not None:
        print(f"  冷尾贡献 <= {cold_max*100:.3f}%(低于 1000 degC 的单元对短波通道)"
              " -> 无阈值积分无害")
    print(f"  三路读数(有效箱数 / 峰值 K):")
    for label, key, peak in (("采纳(条件平均)", "avg_K", doc["peak_avg_K"]),
                             ("双色合成", "two_colour_K", doc["peak_two_colour_K"]),
                             ("全光斑无阈值", "full_spot_avg_K",
                              doc["peak_full_spot_avg_K"])):
        n = sum(1 for s in series if s.get(key) is not None)
        pk = f"{peak:8.1f}" if peak is not None else "     n/a"
        print(f"    {label:<16s} n={n:<4d} 峰值 {pk} K")
    if tc_over:
        print(f"    [!] {tc_over} 个箱的双色读数超出 {args.range_max_C} degC 量程上限")
    for w in warnings:
        print(f"  [WARN] {w}")
    if args.output:
        print(f"  写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
