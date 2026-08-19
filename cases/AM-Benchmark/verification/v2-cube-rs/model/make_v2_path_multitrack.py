#!/usr/bin/env python3
"""V2 多道扫描路径(Balbaa 多道模型,Sec 2.6.2 / Table 3)。

蛇形往返,hatch 沿 y,扫描沿 x。默认工况 = 他的高温计验证点
220 W / 650 mm/s / hatch 0.12 mm(Sec 3.3)。

--tracks 限制道数:基板离散敏感性研究(D-V2-07)只需要少数几道,
结论不依赖整层保真度,因为三个基板变体共用同一条路径。
整层留给热闸门的高温计对照。

条带宽度未给(D-V2-09):4 mm 域小于典型 EOS 条带(5-10 mm),按单条带
处理,即纯蛇形。CSV 列与 runner --path-file 一致(m/s/W)。

--------------------------------------------------------------------------
D-V2-08 窗口模式(2026-08-19 加,热闸门用)
--------------------------------------------------------------------------
Balbaa 的 step time 是 **10x10 mm** 层的曝光时间,而网格只有 4x4 mm。
登记的 leading reading (c):激光走完整的 10x10 蛇形,4x4 网格只是一个
**空间窗**;激光在窗外时高斯尾在域内的沉积 ~0。用法:

    --exposure-area 10.0e-3        # 真实曝光区(蛇形铺满它)
    [--window-area 4.0e-3]         # 网格窗,默认 = AREA_X/Y,居中放在曝光区里

路径坐标输出的是**网格坐标**(曝光区坐标减去窗口原点),所以窗外的行
坐标会落在 [0, 4] mm 之外 —— 这是有意的,不是 bug。

窗外的行没有域内沉积,只有传导冷却,时间分辨率需求低得多,因此窗外用
--coarse-step 粗采样(默认 8 x --sample-step),把步数从 ~1.7 万降到几千。
这是纯粹的时间离散选择,不动任何物理量;能量台账里逐行核对了这一点。
"""
import argparse
import math
from pathlib import Path

AREA_X = 4.0e-3
AREA_Y = 4.0e-3


def gaussian_capture(x, y, win):
    """束心在 (x, y) 时,Eq 18 高斯横向分布落在网格窗矩形内的能量比例。

    I ~ exp(-2 d^2 / r^2) 等价于 sigma = r/2 的二维正态,故窗内比例是两个
    一维 erf 差的乘积。用来给"窗外沉积 ~0"这句话一个数,而不是嘴上说说。
    """
    x0, x1, y0, y1, r = win
    s = math.sqrt(2.0) / r
    fx = 0.5 * (math.erf((x1 - x) * s) - math.erf((x0 - x) * s))
    fy = 0.5 * (math.erf((y1 - y) * s) - math.erf((y0 - y) * s))
    return max(fx, 0.0) * max(fy, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=float, default=220.0)
    ap.add_argument("--speed", type=float, default=0.650)
    ap.add_argument("--hatch", type=float, default=0.12e-3)
    ap.add_argument("--tracks", type=int, default=0, help="0 = 整层")
    ap.add_argument("--sample-step", type=float, default=50.0e-6)
    ap.add_argument("--jump-speed", type=float, default=5.0)
    ap.add_argument("--z", type=float, default=440.0e-6, help="粉层顶面")
    ap.add_argument("--margin", type=float, default=0.1e-3, help="道端距域边留白")
    ap.add_argument("--exposure-area", type=float, default=None,
                    help="D-V2-08 (c):真实曝光区边长 [m](如 10.0e-3);"
                         "缺省 = 网格窗,即旧行为")
    ap.add_argument("--window-area", type=float, default=None,
                    help="网格窗边长 [m],默认 4.0e-3(M1 网格)")
    ap.add_argument("--coarse-step", type=float, default=None,
                    help="窗外采样步长 [m],默认 8 x --sample-step")
    ap.add_argument("--window-pad", type=float, default=0.3e-3,
                    help="窗边外仍用细采样的缓冲带 [m](>= 4 倍束半径即够)")
    ap.add_argument("--coarse-within-track", action="store_true",
                    help="穿窗的道也在其窗外段用粗步长。默认**关**:穿窗的道"
                         "整道细采样,以免道间冷却段的帧距超过 10 ms 协议增量;"
                         "开启可再省约 1/3 步数,代价是热输出时间分辨率不均匀")
    ap.add_argument("--beam-radius", type=float, default=50.0e-6,
                    help="仅用于能量捕获台账,不进入路径")
    ap.add_argument("--ledger-json", type=Path, default=None, help="能量台账 JSON")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    win_side = args.window_area if args.window_area is not None else AREA_X
    exp_side = args.exposure_area if args.exposure_area is not None else win_side
    if exp_side < win_side:
        raise SystemExit("--exposure-area 不能小于 --window-area")
    windowed = exp_side > win_side

    # 蛇形铺满曝光区;网格窗居中放在曝光区里(D-V2-09 采纳约定,见下方回显)
    area_x = area_y = exp_side
    win_x0 = win_y0 = 0.5 * (exp_side - win_side)
    win_x1, win_y1 = win_x0 + win_side, win_y0 + win_side

    n_full = int(round((area_y - 2 * args.margin) / args.hatch)) + 1
    n = n_full if args.tracks <= 0 else min(args.tracks, n_full)
    # 少道数时居中放置,避免贴边效应污染基板研究
    y0 = (area_y - (n - 1) * args.hatch) / 2.0
    x_lo, x_hi = args.margin, area_x - args.margin

    coarse = args.coarse_step if args.coarse_step is not None else 8.0 * args.sample_step
    pad = args.window_pad
    cap_win = (win_x0, win_x1, win_y0, win_y1, args.beam_radius)

    def step_for(y_track):
        """该道使用的采样步长:道整体离窗太远就用粗步长。"""
        if not windowed:
            return args.sample_step
        if y_track < win_y0 - pad or y_track > win_y1 + pad:
            return coarse
        # 该道横穿窗口
        return None if args.coarse_within_track else args.sample_step

    rows = []
    t = 0.0
    prev = None
    for i in range(n):
        y = y0 + i * args.hatch
        a, b = (x_lo, x_hi) if i % 2 == 0 else (x_hi, x_lo)
        if prev is not None:
            jump = abs(y - prev[1]) + abs(a - prev[0])
            t += max(jump / args.jump_speed, 1.0e-5)
            rows.append((t, a, y, 0.0, 0))
        length = abs(b - a)
        uniform = step_for(y)
        if uniform is not None:
            nseg = max(int(round(length / uniform)), 1)
            for s in range(1, nseg + 1):
                frac = s / nseg
                t += (length / nseg) / args.speed
                rows.append((t, a + frac * (b - a), y, args.power, 1))
        else:
            # 沿 x 走:进入 [win_x0-pad, win_x1+pad] 用细步长,其余粗步长。
            # 逐小段推进,保证细/粗切换点落在缓冲带边界上而不是任意位置。
            x_cur, direction = a, (1.0 if b > a else -1.0)
            bounds = sorted([win_x0 - pad, win_x1 + pad])
            while (b - x_cur) * direction > 1e-15:
                fine = bounds[0] <= x_cur + 1e-15 * direction <= bounds[1]
                # 本段终点:下一个边界或道端
                nxt = b
                for bound in bounds:
                    if (bound - x_cur) * direction > 1e-12 and \
                       (bound - x_cur) * direction < (nxt - x_cur) * direction:
                        nxt = bound
                seg_len = abs(nxt - x_cur)
                step = args.sample_step if fine else coarse
                nseg = max(int(round(seg_len / step)), 1)
                for s in range(1, nseg + 1):
                    x_new = x_cur + direction * seg_len * s / nseg
                    t += (seg_len / nseg) / args.speed
                    rows.append((t, x_new, y, args.power, 1))
                x_cur = nxt
        prev = (b, y)

    # ---- 能量台账:窗外真的沉积 ~0 吗?给数,不给形容词 ----
    on_rows = [r for r in rows if r[4]]
    cap = [gaussian_capture(r[1], r[2], cap_win) for r in on_rows]
    dts, prev_t = [], 0.0
    for r in rows:
        dts.append(r[0] - prev_t)
        prev_t = r[0]
    on_dt = [dt for dt, r in zip(dts, rows) if r[4]]
    e_total = sum(dt * r[3] for dt, r in zip(dts, rows) if r[4])
    e_in = sum(dt * r[3] * c for dt, r, c in zip(on_dt, on_rows, cap))
    n_fine = sum(1 for c in cap if c > 1e-6)
    ledger = {
        "reading": "D-V2-08 (c) 窗口模型" if windowed else "4x4 域内整层(旧行为)",
        "exposure_area_m": exp_side, "window_area_m": win_side,
        "window_bounds_exposure_frame_m": [win_x0, win_x1, win_y0, win_y1],
        "tracks": n, "tracks_full_layer": n_full,
        "rows": len(rows), "rows_laser_on": len(on_rows),
        "rows_with_capture_gt_1e-6": n_fine,
        "t_end_s": t,
        "laser_on_time_s": sum(on_dt),
        "dt_min_s": min(dts[1:]) if len(dts) > 1 else 0.0,
        "dt_max_s": max(dts[1:]) if len(dts) > 1 else 0.0,
        "path_energy_J_nominal": e_total,
        "path_energy_J_captured_in_window": e_in,
        "capture_fraction": (e_in / e_total) if e_total > 0 else 0.0,
        "sample_step_m": args.sample_step,
        "coarse_step_m": coarse if windowed else args.sample_step,
        "window_pad_m": pad if windowed else 0.0,
        "convention_D_V2_09": "单条带纯蛇形;首道在 y_min,首道沿 +x;"
                              "网格窗居中于曝光区。stripe 宽度/起始角未给,"
                              "此为采纳约定,登记在 deviations D-V2-09",
        "coordinate_note": "输出为网格坐标 = 曝光区坐标 - 窗口原点;"
                           "窗外行的坐标落在网格外是有意的",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("time,x,y,z,power,laser_on,layer,hatch,mode,front_coord\n")
        for k, (tt, x, y, p, on) in enumerate(rows):
            f.write(f"{tt:.9f},{x - win_x0:.9e},{y - win_y0:.9e},{args.z:.9e},{p},{on},"
                    f"1,{k},scan,{args.z:.9e}\n")

    print(f"wrote {args.output}: {len(rows)} rows, {n}/{n_full} tracks, "
          f"P={args.power} W, v={args.speed*1e3:.0f} mm/s, hatch={args.hatch*1e6:.0f} um, "
          f"t_end={t*1e3:.3f} ms")
    if windowed:
        print(f"  D-V2-08 (c): 曝光区 {exp_side*1e3:.1f} mm, 网格窗 {win_side*1e3:.1f} mm "
              f"居中 [{win_x0*1e3:.2f}, {win_x1*1e3:.2f}] mm")
        print(f"  能量捕获 {ledger['capture_fraction']*100:.2f} % "
              f"({e_in:.4f} / {e_total:.4f} J), 有沉积的行 {n_fine} / {len(on_rows)}")
        print(f"  步长 细 {args.sample_step*1e6:.1f} um / 粗 {coarse*1e6:.1f} um, "
              f"dt {ledger['dt_min_s']:.3e} .. {ledger['dt_max_s']:.3e} s")
    if args.ledger_json:
        import json
        args.ledger_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.ledger_json, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)
        print(f"  写出能量台账 {args.ledger_json}")


if __name__ == "__main__":
    main()
