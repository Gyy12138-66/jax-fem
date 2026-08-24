#!/usr/bin/env python3
"""采纳口径回归:三路读数是**加法**,不是改动。

IET-20 的硬约束(Fable5 05:18 / yuyao 08-20):采纳口径(Balbaa Sec 3.3 的
条件平均)是复现他协议与他数值曲线的主尺,**原样保留,一个字不改**。
新增双色合成与全光斑两路之后,必须能证明这一点 —— 证明的方式是拿新版
analyze_pyrometer.py 的输出和 2026-08-19 生产运行留下的产物逐值比。

比的是**采纳口径及其全部输入**:
  帧级  n_hot / avg_K / max_K / n_in_range / avg_range_limited_K / n_over_range
  箱级  avg_K / nearest_frame_avg_K / avg_range_limited_K / max_K / mean_n_hot
  标量  gauge_cells / gauge_cells_in_circle / peak_K / peak_avg_K
容差 0(浮点逐位相等)—— 同一批 VTU、同一段代码路径,任何差异都是回归。
"""
import argparse
import json
import sys
from pathlib import Path

FRAME_KEYS = ("step", "time_s", "laser_on", "n_hot", "avg_K", "max_K",
              "n_in_range", "avg_range_limited_K", "n_over_range")
SERIES_KEYS = ("bin_index", "t_lo_s", "t_center_s", "n_frames", "frame_times_s",
               "avg_K", "nearest_frame_avg_K", "avg_range_limited_K",
               "max_K", "mean_n_hot", "any_laser_on")
SCALAR_KEYS = ("gauge_cells", "gauge_cells_in_circle", "n_frames",
               "peak_K", "peak_avg_K")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True, help="生产产物 JSON(基准)")
    ap.add_argument("--new", type=Path, required=True, help="新版重读 JSON")
    ap.add_argument("--arm", default="")
    args = ap.parse_args()

    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    diffs = []

    for k in SCALAR_KEYS:
        if old.get(k) != new.get(k):
            diffs.append(f"标量 {k}: {old.get(k)!r} -> {new.get(k)!r}")

    for label, key, keys in (("帧", "frames", FRAME_KEYS),
                             ("箱", "series", SERIES_KEYS)):
        a, b = old.get(key, []), new.get(key, [])
        if len(a) != len(b):
            diffs.append(f"{label}数不同: {len(a)} -> {len(b)}")
            continue
        for i, (ra, rb) in enumerate(zip(a, b)):
            for k in keys:
                if ra.get(k) != rb.get(k):
                    diffs.append(f"{label}[{i}].{k}: {ra.get(k)!r} -> {rb.get(k)!r}")

    n_f, n_s = len(new.get("frames", [])), len(new.get("series", []))
    tag = f"[{args.arm}] " if args.arm else ""
    if diffs:
        print(f"  {tag}采纳口径回归 **失败** —— {len(diffs)} 处差异:")
        for d in diffs[:40]:
            print(f"    {d}")
        if len(diffs) > 40:
            print(f"    ... 另有 {len(diffs) - 40} 处")
        return 1
    print(f"  {tag}采纳口径回归通过:{n_f} 帧 x {len(FRAME_KEYS)} 字段 + "
          f"{n_s} 箱 x {len(SERIES_KEYS)} 字段 + {len(SCALAR_KEYS)} 个标量,逐值相等")
    added = sorted(set(new.get("frames", [{}])[0]) - set(old.get("frames", [{}])[0]))
    print(f"  {tag}新增字段(纯加法):{', '.join(added) if added else '(无)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
