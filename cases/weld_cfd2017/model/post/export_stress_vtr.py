# -*- coding: utf-8 -*-
"""
把力学运行的 VTU 序列转成 ParaView 的 RectilinearGrid (.vtr) 时间序列 + .pvd。

力学网格是张量积（渐变但仍是规则网格），所以可以无损转成 .vtr：文件更小、与热学导出
(field.pvd) 同一种格式，便于并排比较。

用法:
  python export_stress_vtr.py <力学结果目录> <输出目录> [--mirror] [--path-csv 路径表]
    --mirror    : 半模型按 x=0 镜像成全板（位移 ux 变号），仅对半模型有意义
    --path-csv  : 步号->时间 的对照表（默认取结果目录里的 path_used.csv）

在 ParaView 打开 <输出目录>/stress.pvd。
  单元场: von_mises, sigma_xx(横向) sigma_yy(纵向) sigma_zz(厚向) sigma_xy/yz/xz,
          eq_plastic_strain, max_temperature_history, stress_free_temperature ...
  点场:   T (K), u (位移矢量, m) -> 用 Warp By Vector 看变形
"""
import argparse
import csv
import glob
import os
import re
import struct

import numpy as np
import meshio


def tensor_axes(coords, n_expected):
    """从坐标列还原张量积的三条轴，并给出每个点/单元的 (i,j,k) 线性序号。"""
    axes = [np.unique(np.round(coords[:, d], 12)) for d in range(3)]
    if int(np.prod([len(a) for a in axes])) != n_expected:
        raise SystemExit("网格不是张量积：%d x %d x %d != %d" % (*[len(a) for a in axes], n_expected))
    idx = [np.searchsorted(axes[d], np.round(coords[:, d], 12)) for d in range(3)]
    n0, n1 = len(axes[0]), len(axes[1])
    lin = idx[0] + n0 * (idx[1] + n1 * idx[2])
    if len(np.unique(lin)) != n_expected:
        raise SystemExit("张量积索引有重复，网格坐标可能不规则")
    return axes, lin


def write_vtr(path, x, y, z, point_arrays, cell_arrays):
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    blobs, offset, heads = [], 0, []

    def add(name, data, ncomp, where):
        nonlocal offset
        raw = np.ascontiguousarray(np.asarray(data, dtype="<f4")).tobytes()
        blobs.append(struct.pack("<Q", len(raw)) + raw)
        heads.append((name, ncomp, offset, where))
        offset += 8 + len(raw)

    for k, (v, nc) in point_arrays.items():
        add(k, v, nc, "point")
    for k, (v, nc) in cell_arrays.items():
        add(k, v, nc, "cell")
    for nm, v in (("x", x), ("y", y), ("z", z)):
        add(nm, v, 1, "coord")

    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" version="1.0" '
                b'byte_order="LittleEndian" header_type="UInt64">\n')
        f.write(('<RectilinearGrid WholeExtent="0 %d 0 %d 0 %d">\n<Piece Extent="0 %d 0 %d 0 %d">\n'
                 % (nx, ny, nz, nx, ny, nz)).encode())
        for tag, where in (("PointData", "point"), ("CellData", "cell")):
            f.write(("<%s>\n" % tag).encode())
            for name, nc, off, w in heads:
                if w == where:
                    f.write(('<DataArray type="Float32" Name="%s" NumberOfComponents="%d" '
                             'format="appended" offset="%d"/>\n' % (name, nc, off)).encode())
            f.write(("</%s>\n" % tag).encode())
        f.write(b"<Coordinates>\n")
        for name, nc, off, w in heads:
            if w == "coord":
                f.write(('<DataArray type="Float32" Name="%s" format="appended" offset="%d"/>\n'
                         % (name, off)).encode())
        f.write(b"</Coordinates>\n</Piece>\n</RectilinearGrid>\n<AppendedData encoding=\"raw\">\n_")
        for b in blobs:
            f.write(b)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("out_dir")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--path-csv", default=None)
    a = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.run_dir, "step_*.vtu")),
                   key=lambda f: int(re.search(r"step_(\d+)_", os.path.basename(f)).group(1)))
    if not files:
        raise SystemExit("没有找到 step_*.vtu：" + a.run_dir)
    steps = [int(re.search(r"step_(\d+)_", os.path.basename(f)).group(1)) for f in files]
    path_csv = a.path_csv or os.path.join(a.run_dir, "path_used.csv")
    rows = list(csv.DictReader(open(path_csv)))
    t_of_step = {k: float(r["time"]) for k, r in enumerate(rows)}
    os.makedirs(a.out_dir, exist_ok=True)

    entries = []
    for f, s in zip(files, steps):
        m = meshio.read(f)
        pts = m.points
        cells = m.cells[0].data
        cen = pts[cells].mean(axis=1)
        (px, py, pz), plin = tensor_axes(pts, len(pts))
        (cx, cy, cz), clin = tensor_axes(cen, len(cells))
        npn = len(pts)
        ncc = len(cells)

        def to_point(v, nc):
            out = np.zeros((npn, nc), dtype=np.float64)
            out[plin] = np.asarray(v, dtype=np.float64).reshape(npn, nc)
            return out

        def to_cell(v):
            out = np.zeros(ncc, dtype=np.float64)
            out[clin] = np.asarray(v, dtype=np.float64).ravel()
            return out

        point_arrays, cell_arrays = {}, {}
        for name, arr in m.point_data.items():
            arr = np.asarray(arr)
            nc = 1 if arr.ndim == 1 else arr.shape[1]
            if name == "sol":
                continue
            point_arrays[name] = [to_point(arr, nc), nc]
        for name, arr in m.cell_data.items():
            cell_arrays[name] = [to_cell(arr[0]), 1]

        X, Y, Z = px, py, pz
        if a.mirror:
            # 半模型（对称面 x=px[0]）镜像成全板；矢量的 x 分量变号
            x0 = px[0]
            X = np.concatenate([-(px[:0:-1] - x0) + x0, px])
            npx = len(px)
            for name, (v, nc) in list(point_arrays.items()):
                g = v.reshape((npx, len(py), len(pz), nc), order="F")
                gm = g[:0:-1].copy()
                if nc == 3:
                    gm[..., 0] *= -1.0
                point_arrays[name] = [np.concatenate([gm, g], axis=0).reshape(-1, nc, order="F"), nc]
            ncx = len(cx)
            for name, (v, nc) in list(cell_arrays.items()):
                g = v.reshape((ncx, len(cy), len(cz)), order="F")
                cell_arrays[name] = [np.concatenate([g[::-1], g], axis=0).ravel(order="F"), nc]

        pa = {k: (v.ravel(order="C") if nc > 1 else v.ravel(), nc) for k, (v, nc) in point_arrays.items()}
        ca = {k: (v, nc) for k, (v, nc) in cell_arrays.items()}
        name = "stress_%04d.vtr" % s
        write_vtr(os.path.join(a.out_dir, name), X, Y, Z, pa, ca)
        entries.append((t_of_step[s], name))

    with open(os.path.join(a.out_dir, "stress.pvd"), "w") as fh:
        fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for t, name in entries:
            fh.write('<DataSet timestep="%.6f" group="" part="0" file="%s"/>\n' % (t, name))
        fh.write("</Collection>\n</VTKFile>\n")
    size = sum(os.path.getsize(os.path.join(a.out_dir, n)) for _, n in entries)
    print("写出 %d 帧, t = %.3f .. %.3f s, 共 %.1f MB -> %s%s"
          % (len(entries), entries[0][0], entries[-1][0], size / 1e6, os.path.join(a.out_dir, "stress.pvd"),
             "  [已镜像成全板]" if a.mirror else "  [半模型]"))
    print("单元场:", ", ".join(sorted(ca)))
    print("点场:", ", ".join("%s(%d)" % (k, nc) for k, (_, nc) in sorted(pa.items())))


if __name__ == "__main__":
    main()
