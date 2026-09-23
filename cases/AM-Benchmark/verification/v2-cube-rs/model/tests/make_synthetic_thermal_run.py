#!/usr/bin/env python3
"""造一个**合成**的热运行目录,用来冒烟 analyze_pyrometer.py / compare_thermal_gate.py。

这不是物理:温度场是一个沿真实路径行走的解析高斯峰,唯一目的是让分析链
(帧->时间配对、圆内取样、10 ms 分箱、双臂相减、三角形对比)在没有求解器的
机器上也能逐行跑通并被审阅者复现。生产判读只认真实运行。

两臂用 --peak-K 区分,模拟 keff 抬高导热后峰值下降、热区变宽的定性效果。
"""
import argparse
import csv
from pathlib import Path

import meshio
import numpy as np

NX = NY = 40          # 4 mm / 40 = 100 um,比 M1 粗,够冒烟
NZ_SUB, NZ_LAYER = 2, 1
LX = LY = 4.0e-3
SUB = 400.0e-6
LAYER = 40.0e-6


def build_mesh():
    xs = np.linspace(0.0, LX, NX + 1)
    ys = np.linspace(0.0, LY, NY + 1)
    zs = np.concatenate([np.linspace(0.0, SUB, NZ_SUB + 1),
                         [SUB + LAYER]])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    nx, ny = NX + 1, NY + 1
    cells = []
    for k in range(len(zs) - 1):
        for j in range(NY):
            for i in range(NX):
                n0 = k * nx * ny + j * nx + i
                n1 = n0 + nx * ny
                cells.append([n0, n0 + 1, n0 + 1 + nx, n0 + nx,
                              n1, n1 + 1, n1 + 1 + nx, n1 + nx])
    return pts, np.asarray(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-file", type=Path, required=True,
                    help="make_v2_path_multitrack.py 的输出")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--peak-K", type=float, default=5000.0, help="束心峰温")
    ap.add_argument("--sigma-m", type=float, default=180.0e-6, help="热区宽度")
    ap.add_argument("--base-K", type=float, default=353.15, help="预热")
    ap.add_argument("--every", type=int, default=8, help="每 N 行输出一帧")
    ap.add_argument("--t-window", default="0.50,0.78",
                    help="只在这个时间窗内出帧,省磁盘")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.path_file)))
    t_lo, t_hi = (float(v) for v in args.t_window.split(","))
    pts, cells = build_mesh()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # path_used.csv:列与 jax_fem_am/io/vtu.py::write_path_output 一致
    with open(args.out_dir / "path_used.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time", "mode", "layer", "hatch", "scan", "x", "y", "z",
                    "front_coord", "power", "laser_on", "dt", "scan_frac",
                    "hatch_frac", "ambient_temperature", "bottom_temperature"])
        for i, r in enumerate(rows):
            w.writerow([i, r["time"], "scan", 1, 1, i + 1, r["x"], r["y"], r["z"],
                        r["front_coord"], r["power"], r["laser_on"],
                        0.0, 0.0, 0.0, 313.0, 353.15])

    tmax_hist = np.full(len(cells), args.base_K)
    centers = pts[cells].mean(axis=1)
    n_written = 0
    for i, r in enumerate(rows):
        t = float(r["time"])
        if t < t_lo or t > t_hi:
            continue
        if i % args.every:
            continue
        bx, by = float(r["x"]), float(r["y"])
        on = float(r["laser_on"]) > 0.5
        d2 = (pts[:, 0] - bx) ** 2 + (pts[:, 1] - by) ** 2
        depth = np.exp(-(pts[:, 2].max() - pts[:, 2]) / 200.0e-6)
        T = args.base_K + (args.peak_K - args.base_K) * np.exp(
            -d2 / (2 * args.sigma_m ** 2)) * depth * (1.0 if on else 0.2)
        T_cell = T[cells].mean(axis=1)
        tmax_hist = np.maximum(tmax_hist, T_cell)
        meshio.write_points_cells(
            str(args.out_dir / f"step_{i:06d}_scan.vtu"),
            pts, [("hexahedron", cells)],
            point_data={"T": T},
            cell_data={"max_temperature_history": [tmax_hist.copy()]},
        )
        n_written += 1
    print(f"合成运行 {args.out_dir}: {n_written} 帧, {len(cells)} 单元, "
          f"peak={args.peak_K} K sigma={args.sigma_m*1e6:.0f} um")
    print(f"  质心 z 上界 {centers[:,2].max():.6e} m(顶层单元中心)")


if __name__ == "__main__":
    main()
