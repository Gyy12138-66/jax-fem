#!/usr/bin/env python3
"""为 keff 的 L 量测选"哪一道、哪一刻、什么窗"(Fable5 复审 F1 的修法 (b))。

问题:整层跑完之后的曾熔场是所有道的**并集**。熔池全宽(V1 自己预测 141 um)
大于 hatch(120 um)时,单道半宽在那个并集里原理上量不出来 —— 无论窗口怎么调。

修法:回到某一道刚跑完、下一道还没到的那一刻。那时窗内只有这一道的熔池,
且它朝 +y 的一侧还没有被下一道污染,于是**单侧(上半)**半宽就是 L。

本脚本从 as-is 臂用过的 path.csv 里把这三样算出来,打印成 shell 可以直接读的
一行,不在运行器里硬编码任何魔数 —— 路径参数改了,量测跟着改。

用法:
    python pick_keff_measurement.py <path.csv> [--nth 3] [--domain 4.0e-3]
输出(空格分隔):
    track_y  measure_time  win_x0 win_x1 win_y0 win_y1
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_csv", type=Path)
    ap.add_argument("--nth", type=int, default=3,
                    help="用第几条穿窗道(1 起)。默认 3:第 1 条贴着域边 "
                         "(y 约 0.08 mm),它朝 -y 一侧被域边裁掉、散热环境也和"
                         "内部道不同;往里挪几道两者都避开,而 +y 一侧仍是处女地")
    ap.add_argument("--domain", type=float, default=4.0e-3, help="网格窗边长 [m]")
    ap.add_argument("--hatch", type=float, default=0.12e-3)
    ap.add_argument("--x-margin", type=float, default=0.5e-3,
                    help="量测窗在 x 上避开道端加减速/端点效应的留白")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.path_csv)))
    if not rows:
        raise SystemExit(f"{args.path_csv} 是空的")

    # 按 y 分组成道,保留出现顺序;只要激光开着的行
    tracks, order = {}, []
    for r in rows:
        if float(r["laser_on"]) <= 0.5:
            continue
        y = round(float(r["y"]), 9)
        if y not in tracks:
            tracks[y] = []
            order.append(y)
        tracks[y].append(float(r["time"]))

    # 穿窗道 = 道中心线落在网格窗内的道
    crossing = [y for y in order if 0.0 <= y <= args.domain]
    if len(crossing) < args.nth + 1:
        raise SystemExit(
            f"只有 {len(crossing)} 条穿窗道,取不到第 {args.nth} 条并留出下一道。"
            "把 --nth 调小,或确认这条路径确实是窗口模式")

    y_meas = crossing[args.nth - 1]
    y_next = crossing[args.nth]
    t_end = max(tracks[y_meas])          # 本道最后一行的时刻
    t_next = min(tracks[y_next])         # 下一道第一行的时刻
    if t_next <= t_end:
        raise SystemExit("下一道的起始时刻不晚于本道结束时刻,路径时间不单调?")

    # 取样时刻取本道结束的那一刻:make_keff_table.py 会取 <= 该时刻的最后一帧,
    # 于是绝不会把下一道的沉积算进来。
    t_meas = t_end

    # 量测窗:x 避开道端;y 下界紧贴本道下方半个 hatch(下面是已跑过的道,
    # 单侧量法用不到),上界留到下一道中心线之上 —— 留够空间,好让"曾熔区顶到
    # 窗界"这个饱和护栏在熔池真的那么宽时**能够**触发,而不是被窗口提前掩盖。
    win = (args.x_margin, args.domain - args.x_margin,
           y_meas - 0.5 * args.hatch, y_next + 0.5 * args.hatch)

    print(f"{y_meas:.9e} {t_meas:.9f} {win[0]:.9e} {win[1]:.9e} "
          f"{win[2]:.9e} {win[3]:.9e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
