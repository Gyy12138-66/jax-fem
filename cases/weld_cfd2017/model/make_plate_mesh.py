#!/usr/bin/env python3
"""Graded tensor-product HEX8 mesh for the 2017 CFD bead-on-plate case (weld_cfd2017).

Axes follow the CFD grid exactly, so the mechanical model and the thermal field share one
coordinate system (no swap, no rotation when both are opened in ParaView):

    x : along the weld,      0 .. length            (heat source travels +x)
    y : across the weld,     0 .. half-width        (y = 0 is the symmetry plane / weld centreline)
    z : through thickness,   0 .. thickness         (z = thickness is the top face, the weld side)

Plain rectangular plate: no groove and no reinforcement (the CFD free surface was never
coupled, so the plate stays flat).

Grading: uniform fine spacing next to the weld centreline (y) and below the top face (z),
geometric growth outward and downward. Along the weld (x) the gradients are mild, so the
spacing is uniform by default.

--model half : y from 0 to half-width (matches the CFD half domain, the default).
--model full : y mirrored to the full width, weld centreline at width/2.

--bead-height-file 给出焊道余高场（bead_height.npz，来自 2017 tecfree）时，母板几何完全不变，
只在 h 超过 --bead-min-height 的区域从 z=thickness 往上加 --bead-layers 层单元，层厚随 h(x,y) 变化，
z=thickness 的节点与母板共用（协调网格）。单元集 PLATE / BEAD 分开输出，供 --bead-elsets BEAD 做单元生死。

Output: Abaqus .inp in METERS (C3D8, ELSET ALL, NSET SYMMETRY_Y / TOP_SURFACE) plus a
summary JSON. Node ids are consecutive 1..N with i fastest, as the temperature mapper
(make_prescribed_temperature.py) requires.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def graded_one_sided(d_fine, fine_len, d_max, growth, total):
    """Coordinates 0..total: uniform d_fine over [0, fine_len], then geometric growth."""
    fine_len = min(fine_len, total)
    n_fine = max(int(round(fine_len / d_fine)), 1)
    fine = np.linspace(0.0, fine_len, n_fine + 1)
    rest = total - fine[-1]
    if rest <= 1e-12:
        return fine
    segs, d = [], d_fine
    while sum(segs) < rest:
        d = min(d * growth, d_max)
        segs.append(d)
    segs = np.asarray(segs) * (rest / sum(segs))
    out = np.concatenate([fine, fine[-1] + np.cumsum(segs)])
    out[-1] = total
    return out


def build(a):
    half_width = a.width / 2.0
    # x: along the weld
    if a.dx_max > a.dx and a.x_fine_start > 0.0:
        head = a.x_fine_start - graded_one_sided(a.dx, 0.0, a.dx_max, a.growth, a.x_fine_start)[::-1]
        mid = np.linspace(a.x_fine_start, a.x_fine_end, max(int(round((a.x_fine_end - a.x_fine_start) / a.dx)), 1) + 1)
        tail = a.x_fine_end + graded_one_sided(a.dx, 0.0, a.dx_max, a.growth, a.length - a.x_fine_end)
        x = np.unique(np.concatenate([head, mid, tail]))
    else:
        x = np.linspace(0.0, a.length, int(round(a.length / a.dx)) + 1)
    # y: across the weld, fine at the centreline
    yhalf = graded_one_sided(a.dy_fine, a.y_fine, a.dy_max, a.growth, half_width)
    if a.model == "half":
        y = yhalf
        weld_centre_y = 0.0
    else:
        y = np.concatenate([-yhalf[:0:-1], yhalf]) + half_width
        weld_centre_y = half_width
    # z: through thickness, fine at the top face
    zt = graded_one_sided(a.dz_fine, a.z_fine, a.dz_max, a.growth, a.thickness)
    z = np.sort(a.thickness - zt)

    x, y, z = [np.asarray(v, dtype=np.float64) for v in (x, y, z)]
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")], axis=1)

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(order="F"), j.ravel(order="F"), k.ravel(order="F")
    cells = np.stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                      nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)], axis=1)

    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    total_vol = float(x[-1] - x[0]) * float(y[-1] - y[0]) * float(z[-1] - z[0])
    sum_vol = float(np.sum(np.outer(np.outer(dx, dy).ravel(), dz)))

    n_base_nodes = len(nodes)
    n_base_cells = len(cells)
    bead_cells = np.empty((0, 8), dtype=np.int64)
    bead_info = None
    if a.bead_height_file:
        from scipy.interpolate import RegularGridInterpolator
        d = np.load(a.bead_height_file)
        hx, hy, hall = d["x"] * 1e3, d["y"] * 1e3, d["h"] * 1e3      # 转 mm
        frame = a.bead_frame if a.bead_frame >= 0 else len(hall) + a.bead_frame
        hf = RegularGridInterpolator((hx, hy), hall[frame], method="linear",
                                     bounds_error=False, fill_value=0.0)
        yq = np.abs(y - (0.0 if a.model == "half" else half_width))    # 半模型对称
        HN = np.clip(hf(np.stack(np.meshgrid(x, yq, indexing="ij"), axis=-1)), 0.0, None)
        # 保留四角 h 都超过阈值的 (i,j) 单元列
        keep = (np.minimum.reduce([HN[:-1, :-1], HN[1:, :-1], HN[1:, 1:], HN[:-1, 1:]]) > a.bead_min_height)
        ii, jj = np.nonzero(keep)
        node_ij = np.unique(np.concatenate([ii + (nx + 1) * jj, ii + 1 + (nx + 1) * jj,
                                            ii + (nx + 1) * (jj + 1), ii + 1 + (nx + 1) * (jj + 1)]))
        # 焊道节点：每个 (i,j) 上 a.bead_layers 层，z = thickness + h*m/N
        gi, gj = node_ij % (nx + 1), node_ij // (nx + 1)
        hcol = np.maximum(HN[gi, gj], a.bead_min_height)
        new_pts = []
        for mlay in range(1, a.bead_layers + 1):
            new_pts.append(np.stack([x[gi], y[gj], a.thickness + hcol * mlay / a.bead_layers], axis=1))
        nodes = np.concatenate([nodes] + new_pts, axis=0)
        # 编号：底面用母板顶层节点，上面各层依次接在后面
        rank = -np.ones((nx + 1) * (ny + 1), dtype=np.int64)
        rank[node_ij] = np.arange(len(node_ij))

        def bead_nid(i, j, mlay):
            if mlay == 0:
                return nid(i, j, nz)
            return n_base_nodes + 1 + rank[i + (nx + 1) * j] + (mlay - 1) * len(node_ij)

        blocks = []
        for mlay in range(a.bead_layers):
            blocks.append(np.stack([bead_nid(ii, jj, mlay), bead_nid(ii + 1, jj, mlay),
                                    bead_nid(ii + 1, jj + 1, mlay), bead_nid(ii, jj + 1, mlay),
                                    bead_nid(ii, jj, mlay + 1), bead_nid(ii + 1, jj, mlay + 1),
                                    bead_nid(ii + 1, jj + 1, mlay + 1), bead_nid(ii, jj + 1, mlay + 1)], axis=1))
        bead_cells = np.concatenate(blocks, axis=0)
        cells = np.concatenate([cells, bead_cells], axis=0)
        cw = np.minimum.reduce([HN[:-1, :-1], HN[1:, :-1], HN[1:, 1:], HN[:-1, 1:]])[keep]
        bvol = float(np.sum(np.maximum(cw, a.bead_min_height) * np.outer(dx, dy)[keep]))
        bead_info = dict(source=os.path.abspath(a.bead_height_file), frame=int(frame),
                         time_s=float(d["time"][frame]), layers=int(a.bead_layers),
                         min_height_mm=a.bead_min_height, cells=int(len(bead_cells)),
                         nodes_added=int(len(nodes) - n_base_nodes),
                         height_max_mm=float(HN.max()), volume_mm3=bvol,
                         footprint_x_mm=[float(x[ii.min()]), float(x[ii.max() + 1])],
                         footprint_y_mm=[float(y[jj.min()]), float(y[jj.max() + 1])])
        print("焊道: %d 单元, %d 新节点, 峰高 %.3f mm, 体积 %.2f mm3 (半模型), 足迹 x %.1f..%.1f mm, y %.1f..%.1f mm"
              % (len(bead_cells), len(nodes) - n_base_nodes, HN.max(), bvol,
                 x[ii.min()], x[ii.max() + 1], y[jj.min()], y[jj.max() + 1]))

    sym = np.flatnonzero(np.abs(nodes[:, 1] - y[0]) <= 1e-12) + 1
    top = np.flatnonzero(np.abs(nodes[:, 2] - z[-1]) <= 1e-12) + 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(f"Bead-on-plate {a.model} model for weld_cfd2017 in CFD axes "
                f"(x along weld, y across weld with y=0 the symmetry plane, z through thickness), "
                f"graded tensor-product HEX8, units METERS, generated by make_plate_mesh.py\n")
        f.write("*NODE\n")
        for n, p in enumerate(nodes, start=1):
            f.write("%d, %.9e, %.9e, %.9e\n" % (n, p[0] * 1e-3, p[1] * 1e-3, p[2] * 1e-3))
        f.write("*ELEMENT, TYPE=C3D8, ELSET=ALL\n")
        for e, cc in enumerate(cells, start=1):
            f.write("%d, %s\n" % (e, ", ".join(str(v) for v in cc)))
        for name, lo, hi in (("PLATE", 1, n_base_cells), ("BEAD", n_base_cells + 1, len(cells))):
            if hi < lo:
                continue
            f.write(f"*ELSET, ELSET={name}\n")
            ids = list(range(lo, hi + 1))
            for sblk in range(0, len(ids), 16):
                f.write(", ".join(str(v) for v in ids[sblk:sblk + 16]) + "\n")
        for name, ids in (("SYMMETRY_Y", sym), ("TOP_SURFACE", top)):
            f.write(f"*NSET, NSET={name}\n")
            for s in range(0, len(ids), 16):
                f.write(", ".join(str(v) for v in ids[s:s + 16]) + "\n")

    summary = dict(
        model=a.model, units="inp in meters; this summary in mm",
        axes=dict(x="along the weld", y="across the weld (y=0 symmetry plane)", z="through thickness (top = z max)"),
        geometry_mm=dict(length_x=a.length, width_y=a.width, thickness_z=a.thickness,
                         y_extent=[float(y[0]), float(y[-1])], weld_centre_y=weld_centre_y),
        counts=dict(nx=nx, ny=ny, nz=nz, nodes=len(nodes), cells=len(cells), dof=3 * len(nodes),
                    plate_cells=int(n_base_cells), bead_cells=int(len(bead_cells)),
                    symmetry_nodes=int(len(sym)), top_surface_nodes=int(len(top))),
        bead=bead_info,
        spacing_mm=dict(dx_min=float(dx.min()), dx_max=float(dx.max()),
                        dy_min=float(dy.min()), dy_max=float(dy.max()),
                        dz_min=float(dz.min()), dz_max=float(dz.max())),
        checks=dict(volume_closure=sum_vol / total_vol, node_ids_consecutive=True,
                    all_positive_jacobian=bool(dx.min() > 0 and dy.min() > 0 and dz.min() > 0)),
        output=os.path.abspath(a.out),
    )
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("half", "full"), default="half")
    p.add_argument("--length", type=float, default=44.0, help="plate length along the weld, mm")
    p.add_argument("--width", type=float, default=20.0, help="full plate width across the weld, mm")
    p.add_argument("--thickness", type=float, default=6.0)
    p.add_argument("--dx", type=float, default=0.5, help="spacing along the weld, mm")
    p.add_argument("--dx-max", type=float, default=0.5, help="coarse spacing before/after the weld travel (= dx keeps it uniform)")
    p.add_argument("--x-fine-start", type=float, default=0.0)
    p.add_argument("--x-fine-end", type=float, default=44.0)
    p.add_argument("--dy-fine", type=float, default=0.25, help="transverse spacing next to the weld centreline, mm")
    p.add_argument("--y-fine", type=float, default=6.0, help="transverse extent of the fine zone from the centreline, mm")
    p.add_argument("--dy-max", type=float, default=1.0)
    p.add_argument("--dz-fine", type=float, default=0.25, help="through-thickness spacing at the top face, mm")
    p.add_argument("--z-fine", type=float, default=3.0, help="depth of the fine zone below the top face, mm")
    p.add_argument("--dz-max", type=float, default=0.5)
    p.add_argument("--growth", type=float, default=1.25)
    p.add_argument("--bead-height-file", default=None,
                   help="焊道余高场 npz（time, h[t,i,j], x, y，单位 m）；给出后在板面上加焊道层")
    p.add_argument("--bead-frame", type=int, default=-1, help="用第几帧的 h（-1 = 最后一帧）")
    p.add_argument("--bead-layers", type=int, default=4, help="焊道厚度方向的单元层数")
    p.add_argument("--bead-min-height", type=float, default=0.25,
                   help="低于此高度的区域不生成焊道单元，且足迹内的层厚不低于该值 [mm]")
    p.add_argument("--out", default="inputs/plate_half_cfdaxes_025mm.inp")
    build(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
