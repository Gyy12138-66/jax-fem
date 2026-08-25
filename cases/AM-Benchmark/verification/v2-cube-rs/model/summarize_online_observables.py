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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_colour_pyrometer as tc                       # noqa: E402

C2K = 273.15


def response_integrated_series(rows, bin_s, wavelengths):
    """Integrate piecewise-constant step observations over fixed time bins.

    A row describes the interval ``(time_s - dt_s, time_s]``.  Splitting that
    interval at bin boundaries is essential when dt changes or a solver step
    straddles a 10 ms instrument-response boundary.
    """
    if not np.isfinite(bin_s) or bin_s <= 0.0:
        raise ValueError("bin_s must be finite and positive")
    bins = {}
    previous_end = None
    tolerance = max(1.0e-12, bin_s * 1.0e-9)
    for row_index, row in enumerate(rows):
        end = float(row["time_s"])
        dt_s = float(row["dt_s"])
        if not np.isfinite(end) or not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"row {row_index} has invalid time_s/dt_s")
        start = end - dt_s
        if previous_end is not None:
            delta = start - previous_end
            if abs(delta) > tolerance:
                kind = "gap" if delta > 0.0 else "overlap"
                raise ValueError(
                    f"row {row_index} introduces a {kind} of {abs(delta):.6e} s")
        previous_end = end
        first = int(np.floor(start / bin_s))
        # Treat an endpoint exactly on a boundary as belonging to the preceding
        # interval; the zero-width overlap with the next bin is then discarded.
        last = int(np.floor(np.nextafter(end, -np.inf) / bin_s))
        for idx in range(first, last + 1):
            lo, hi = idx * bin_s, (idx + 1) * bin_s
            overlap = min(end, hi) - max(start, lo)
            if overlap > 0.0:
                bins.setdefault(idx, []).append((row, overlap))

    series = []
    for idx in sorted(bins):
        pieces = bins[idx]
        total_w = sum(weight for _, weight in pieces)

        def weighted(key, *, present=lambda value: value is not None):
            selected = [(row[key], weight) for row, weight in pieces
                        if present(row.get(key))]
            if not selected:
                return None
            denominator = sum(weight for _, weight in selected)
            return float(sum(float(value) * weight for value, weight in selected)
                         / denominator)

        s1 = weighted("two_colour_S1")
        s2 = weighted("two_colour_S2")
        tc_bin = tc.read_accumulated(s1, s2, wavelengths=wavelengths)
        members = [row for row, _ in pieces]
        coverage = total_w / bin_s
        if coverage > 1.0 + 1.0e-9:
            raise ValueError(f"bin {idx} coverage exceeds one: {coverage}")
        # A2 (a)(规范 v2.1 §3.2,yuyao 2026-08-25):采纳口径的分母只取有效时样
        # (n_hot > 0),NA 时样不进分母 —— 这是 Balbaa 分箱读法,数值与之前完全
        # 相同;但删掉了多少**必须逐箱写出来**,不得静默。时长占比 > 50 % 的箱在
        # 评分矩阵里要加旁注(该读数由少数时样决定)。
        na_pieces = [(row, weight) for row, weight in pieces if int(row["n_hot"]) == 0]
        na_covered = float(sum(weight for _, weight in na_pieces))
        entry = {
            "bin_index": idx,
            "t_center_s": idx * bin_s + 0.5 * bin_s,
            "covered_s": float(total_w),
            "coverage_fraction": float(coverage),
            "n_samples": len(pieces),
            "n_na_samples": len(na_pieces),
            "na_sample_fraction": len(na_pieces) / len(pieces),
            "na_covered_s": na_covered,
            "na_time_fraction": (na_covered / total_w) if total_w > 0.0 else None,
            "na_over_half": (bool(na_covered / total_w > 0.5)
                             if total_w > 0.0 else None),
            "n_samples_beam_inside": int(sum(
                bool(row["beam_inside_spot"]) for row, _ in pieces)),
            "avg_K": weighted("avg_K"),
            "mean_n_hot": weighted("n_hot"),
            "two_colour_K": tc_bin["T_K"],
            "two_colour_over_range": tc_bin["over_range"],
            "full_spot_avg_K": weighted("full_spot_avg_K"),
            "max_K": float(max(row["max_K"] for row in members)),
        }
        if "probe_K" in members[0]:
            probes = np.asarray([row["probe_K"] for row, _ in pieces], dtype=float)
            weights = np.asarray([weight for _, weight in pieces], dtype=float)
            entry["probe_mean_K"] = [float(v) for v in
                                      np.average(probes, axis=0, weights=weights)]
            entry["probe_max_K"] = [float(v) for v in probes.max(axis=0)]
        series.append(entry)
    return series


def validate_protocol(meta, *, expected_run_id=None,
                      summary_window_s=(0.45, 0.90)):
    """Fail closed unless metadata covers the registered summary protocol.

    The recorder may intentionally use a wider window.  That is valid as long as
    the requested summary window is wholly contained in the recorded window.
    """
    errors = []
    if meta.get("schema_version") != "v06.online-observables/1":
        errors.append("schema_version")
    if expected_run_id is not None and meta.get("run_id") != expected_run_id:
        errors.append("run_id")
    expected = {
        "spot_center_m": [0.002, 0.002],
        "spot_diameter_m": 2.0e-3,
        "threshold_C": 1000.0,
        "range_max_C": 3000.0,
        "record_every_n_steps": 1,
    }
    for key, value in expected.items():
        actual = meta.get(key)
        if actual is None or not np.allclose(actual, value, rtol=0.0, atol=1.0e-12):
            errors.append(key)
    recorded_window = meta.get("window_s")
    if (not isinstance(recorded_window, list) or len(recorded_window) != 2
            or recorded_window[0] > summary_window_s[0] + 1.0e-12
            or recorded_window[1] < summary_window_s[1] - 1.0e-12):
        errors.append("window_s")
    if meta.get("depth_scope") != "top layer (z >= 4.000000e-04 m)":
        errors.append("depth_scope")
    probes = meta.get("probes")
    expected_probes = [[0.001, 0.002, 0.00042],
                       [0.002, 0.002, 0.00042],
                       [0.003, 0.002, 0.00042]]
    requested = [item.get("requested_m") for item in probes] if isinstance(probes, list) else []
    if len(requested) != 3 or not np.allclose(requested, expected_probes,
                                               rtol=0.0, atol=1.0e-12):
        errors.append("probes")
    if meta.get("probe_resolution", {}).get("all_probes_contained") is not True:
        errors.append("all_probes_contained")
    wl = meta.get("two_colour", {}).get("wavelengths_m")
    if wl is None or not np.allclose(wl, [0.95e-6, 1.05e-6],
                                     rtol=0.0, atol=1.0e-15):
        errors.append("wavelengths_m")
    if errors:
        raise ValueError("online observable protocol mismatch: " + ", ".join(errors))
    return tuple(float(value) for value in wl)


def crop_rows_to_window(rows, window_s):
    """Clip step intervals to a summary window without inventing samples."""
    start, end = (float(value) for value in window_s)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("summary window must be finite and increasing")
    clipped = []
    for row in rows:
        interval_start = float(row["time_s"]) - float(row["dt_s"])
        interval_end = float(row["time_s"])
        overlap_start = max(interval_start, start)
        overlap_end = min(interval_end, end)
        if overlap_end <= overlap_start:
            continue
        item = dict(row)
        item["time_s"] = overlap_end
        item["dt_s"] = overlap_end - overlap_start
        clipped.append(item)
    return clipped


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--summary-window", default="0.45,0.90",
                    help="汇总窗口 start,end [s]；记录窗口可以更宽，但必须完全覆盖它")
    ap.add_argument("--expected-run-id", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{args.jsonl} 是空的")
    if not args.meta or not args.meta.is_file():
        raise SystemExit("正式在线汇总必须提供 --meta")
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    try:
        summary_window = tuple(float(value) for value in args.summary_window.split(","))
        if len(summary_window) != 2:
            raise ValueError("summary window needs exactly two values")
        wl = validate_protocol(meta, expected_run_id=args.expected_run_id,
                               summary_window_s=summary_window)
        rows = crop_rows_to_window(rows, summary_window)
        if not rows:
            raise ValueError("summary window contains no recorded observations")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    t = np.asarray([r["time_s"] for r in rows])
    dt = np.diff(t) if t.size > 1 else np.asarray([0.0])
    inside = np.asarray([r["beam_inside_spot"] for r in rows])
    # 驻留段数 = 光束进出圆的次数。VTU 帧距下这个数是 0(每段只被抓到一次或没抓到)
    dwells = int(np.sum(inside[1:] & ~inside[:-1])) + int(bool(inside[0]))
    hot = [r for r in rows if r["n_hot"] > 0]

    # ---- 10 ms 响应积分 ----
    bin_s = args.bin_ms * 1.0e-3
    series = response_integrated_series(rows, bin_s, wl)


    per_bin = [s["n_samples"] for s in series]
    doc = {
        "source": str(args.jsonl),
        "meta": meta,
        "recorded_window_s": meta.get("window_s"),
        "summary_window_s": list(summary_window),
        "response_bin_ms": float(args.bin_ms),
        "source_jsonl_sha256": file_sha256(args.jsonl),
        "n_rows": len(rows),
        "coverage_s": [float(min(r["time_s"] - r["dt_s"] for r in rows)),
                       float(max(r["time_s"] for r in rows))],
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
        "na_disclosure": {
            "_spec": "scoring-spec-thermal-gate-v2.1.md §3.2 / §4.1(修正案 A2 (a),"
                     "yuyao 2026-08-25 裁决)",
            "rule": "采纳口径分母只取有效时样(n_hot > 0);NA 时样(圆内顶层无单元 "
                    ">= 1000 degC)不进分母但逐箱披露数量/时长/时长占比;目标箱 NA 时长"
                    "占比 > 50 % 的点加旁注;整箱无有效时样才是 NA / INVALID",
            "n_bins": len(series),
            "n_bins_with_na_samples": sum(1 for s in series if s["n_na_samples"]),
            "n_bins_na_over_half": sum(1 for s in series if s["na_over_half"]),
            "n_bins_all_na": sum(1 for s in series if s["avg_K"] is None),
            "bins_na_over_half": [s["bin_index"] for s in series if s["na_over_half"]],
        },
        "response_integrated_series": series,
        "reading_definitions": {
            "avg_K": "采纳口径:按各步在响应箱内的实际覆盖时长加权平均(Balbaa 分箱口径)",
            "n_na_samples / na_covered_s / na_time_fraction / na_over_half":
                "A2 (a) 强制披露:箱内 n_hot == 0 的时样数、时长、时长占比、是否 > 50 %",
            "two_colour_K": "双色:按实际覆盖时长积分 S1/S2 后反演一次(仪器积分亮度)",
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
              f"{'采纳 K':>9} {'双色 K':>9} {'全斑 K':>9} {'峰值 K':>9} {'NA时长%':>7}")
        for s in valid[:24]:
            f = lambda v: f"{v:9.1f}" if v is not None else "      n/a"
            na = (f"{s['na_time_fraction']*100:6.1f}{'*' if s['na_over_half'] else ' '}"
                  if s["na_time_fraction"] is not None else "    n/a")
            print(f"  {s['t_center_s']:8.4f} {s['n_samples']:6d} "
                  f"{s['n_samples_beam_inside']:5d} {s['mean_n_hot']:7.1f} "
                  f"{f(s['avg_K'])} {f(s['two_colour_K'])} "
                  f"{f(s['full_spot_avg_K'])} {f(s['max_K'])} {na}")
        if len(valid) > 24:
            print(f"  ... 另有 {len(valid)-24} 个有效箱")
        nd = doc["na_disclosure"]
        print(f"  [A2 (a) 披露] {nd['n_bins_with_na_samples']}/{nd['n_bins']} 箱含 NA 时样,"
              f"{nd['n_bins_na_over_half']} 箱 NA 时长占比 > 50 %(上表 *),"
              f"{nd['n_bins_all_na']} 箱整箱无有效时样(采纳读数 n/a)")
    if "probe_K" in rows[0]:
        # 峰值**和峰时刻**一起报(规范 §6.2 要求"直接受热主峰温度与时刻")。
        # 只报温度会误导:单道等速扫描下三个探针看到的是同一束流经过,峰值
        # 本来就该几乎相同,差别全在**时刻**上 —— 相邻探针应差 1 mm / 扫描速度。
        n = len(rows[0]["probe_K"])
        print(f"  定点探针 {n} 个(Fig 15/16 靶子,规范 §6.1 包含单元 8 节点均值):")
        peaks = []
        for i in range(n):
            top = max(rows, key=lambda r: r["probe_K"][i])
            peaks.append((top["probe_K"][i], top["time_s"]))
            print(f"    P{i+1}  主峰 {top['probe_K'][i]:9.3f} K"
                  f"  @ t = {top['time_s']:.5f} s")
        if n > 1:
            gaps = [peaks[i + 1][1] - peaks[i][1] for i in range(n - 1)]
            print("    相邻主峰时刻差 "
                  + ", ".join(f"{g*1e3:.3f} ms" for g in gaps)
                  + "(等速单道下应 ≈ 探针间距 / 扫描速度)")
    if args.output:
        print(f"  写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
