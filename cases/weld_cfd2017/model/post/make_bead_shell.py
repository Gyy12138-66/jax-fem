# -*- coding: utf-8 -*-
"""
把焊缝隆起（余高）做成 ParaView 可读的时间序列（.vtu + .pvd），用作**灰色参照壳**：
力学结果仍画在平板上，隆起只叠在上面看形状，不参与任何计算。

隆起高度取自 2017 年 tecfree 的上表面变形量 h = zr2_top − 6 mm（`bead_height.npz`，6 帧），
在帧之间按时间线性插值；t=0 时为 0，关功率（缺省 1.505 s）之后保持不变。

壳体是一层六面体：下面贴在板面 z=6 mm，上面在 z=6 mm + h，只保留四个角点 h 都超过
`--min-height` 的单元，所以板面其余部分不会被遮住。

用法:
  python make_bead_shell.py <bead_height.npz> <输出目录> --times-pvd <stress.pvd> [--min-height 5e-5]
"""
import argparse
import os
import re

import numpy as np
import meshio


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz")
    p.add_argument("out_dir")
    p.add_argument("--times-pvd", default=None,
                   help="采用某个 .pvd 的时刻（推荐指向要叠加的 stress.pvd，保证两条时间轴逐帧一致）")
    p.add_argument("--times", default=None, help="或用 field_index.txt 的时刻")
    p.add_argument("--min-height", type=float, default=5e-5, help="低于此高度的区域不生成单元 [m]")
    p.add_argument("--scale", type=float, default=1.0, help="隆起高度放大倍数，仅用于观察")
    p.add_argument("--freeze-after", type=float, default=1.505, help="此时刻之后隆起不再生长 [s]")
    a = p.parse_args(argv)

    d = np.load(a.npz)
    ts, H, x, y = d["time"], d["h"], d["x"], d["y"]
    thickness = float(d["thickness"])
    if a.times_pvd:
        out_t = np.array([float(v) for v in re.findall(r'timestep="([^"]+)"', open(a.times_pvd).read())])
    elif a.times:
        out_t = np.loadtxt(a.times, comments="#", ndmin=2)[:, 1]
    else:
        out_t = ts
    tk = np.concatenate([[0.0], ts, [1e6]])
    Hk = np.concatenate([np.zeros((1,) + H.shape[1:], dtype=H.dtype), H, H[-1:]], axis=0)

    os.makedirs(a.out_dir, exist_ok=True)
    ni, nj = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # 单元索引：点编号 i + ni*j（下层），再加 ni*nj（上层）
    ii, jj = np.meshgrid(np.arange(ni - 1), np.arange(nj - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    n0 = ii + ni * jj
    base = np.stack([n0, n0 + 1, n0 + 1 + ni, n0 + ni], axis=1)
    hexa_all = np.concatenate([base, base + ni * nj], axis=1)

    entries, ncells = [], []
    for n, t in enumerate(out_t):
        tc = min(float(t), a.freeze_after)
        k = min(max(int(np.searchsorted(tk, tc, side="right") - 1), 0), len(tk) - 2)
        w = (tc - tk[k]) / (tk[k + 1] - tk[k]) if tk[k + 1] < 1e5 else 0.0
        h = ((1.0 - w) * Hk[k] + w * Hk[k + 1]).astype(np.float64) * a.scale
        hf = h.ravel(order="F")
        keep = np.min(hf[base], axis=1) > a.min_height
        pts = np.concatenate([
            np.stack([X.ravel(order="F"), Y.ravel(order="F"), np.full(ni * nj, thickness)], axis=1),
            np.stack([X.ravel(order="F"), Y.ravel(order="F"), thickness + hf], axis=1)])
        name = "bead_%04d.vtu" % n
        cells = hexa_all[keep]
        if len(cells) == 0:                      # 起始帧还没有焊道，放一个退化单元占位
            cells = hexa_all[:1]
        meshio.write_points_cells(os.path.join(a.out_dir, name), pts, [("hexahedron", cells)],
                                  point_data={"bead_height": np.concatenate([hf, hf])})
        entries.append((float(t), name)); ncells.append(int(keep.sum()))

    with open(os.path.join(a.out_dir, "bead.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for t, name in entries:
            f.write('<DataSet timestep="%.6f" group="" part="0" file="%s"/>\n' % (t, name))
        f.write("</Collection>\n</VTKFile>\n")
    size = sum(os.path.getsize(os.path.join(a.out_dir, n)) for _, n in entries)
    print("写出 %d 帧灰壳, t = %.3f .. %.3f s, 共 %.1f MB -> %s"
          % (len(entries), entries[0][0], entries[-1][0], size / 1e6, os.path.join(a.out_dir, "bead.pvd")))
    print("焊道单元数 %d -> %d（随热源推进生长）；末帧隆起峰值 %.3f mm"
          % (ncells[0], ncells[-1], H[-1].max() * 1e3 * a.scale))


if __name__ == "__main__":
    main()
