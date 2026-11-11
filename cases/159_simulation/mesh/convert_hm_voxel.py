#!/usr/bin/env python3
"""Turn a HyperMesh voxel HEX8 export (mm, build axis = model x, no sets, no raft) into the runner's
layered mesh: keep only voxels whose centre lies inside the TET4 reference surface (HyperMesh mode 3
keeps every voxel that touches the solid, +44 % volume), rotate to build along +Z, scale to metres,
extrude a raft below the base layer, tag ELSET PART / RAFT and NSET BASE, write inp + vtu.

    convert_hm_voxel.py --inp <hm_voxel.inp> --ref-tet 0119_c3d4_only.inp --out-dir <dir> [--raft-layers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import meshio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mesh_check"))
from inp_health import parse, tet_metrics, hex_metrics  # noqa: E402
from make_0119_hex_mesh import boundary_triangles, slice_loops, inside_by_nearest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--ref-tet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raft-layers", type=int, default=2)
    ap.add_argument("--tag", default="0119_hm_voxel")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = meshio.read(args.inp)
    P = np.asarray(m.points, float) * 1e-3                     # mm -> m, HM frame (x = build axis)
    conn = np.asarray(m.cells_dict["hexahedron"], np.int64)
    # voxel size and grid
    e = np.linalg.norm(P[conn[:, 1]] - P[conn[:, 0]], axis=1)
    h = float(np.median(e))
    print(f"voxel size {h*1e3:.4f} mm, {len(conn)} cells, {len(P)} nodes")

    # reference surface (old HyperMesh frame, metres) and inside test per voxel layer
    nodes, elems, _, _ = parse(args.ref_tet)
    ids = np.array(sorted(nodes))
    RP = np.array([nodes[i] for i in ids])
    ridx = {int(n): k for k, n in enumerate(ids)}
    rconn = None
    for (etype, _), rows in elems.items():
        if etype.startswith("C3D4"):
            rconn = np.vectorize(ridx.get)(np.array([r[:5] for r in rows])[:, 1:])
    T = boundary_triangles(RP, rconn)
    tet_vol = float(np.abs(tet_metrics(RP[rconn])[0]).sum())
    ctr = P[conn].mean(1)
    lev = np.round((ctr[:, 0] - ctr[:, 0].min()) / h).astype(int)
    keep = np.zeros(len(conn), bool)
    for k in np.unique(lev):
        sel = np.nonzero(lev == k)[0]
        x0 = float(ctr[sel[0], 0])
        loops = slice_loops(T, x0)
        if loops:
            keep[sel] = inside_by_nearest(loops, ctr[sel][:, 1:])
    print(f"centre-inside voxels: {keep.sum()} of {len(conn)}")
    conn = conn[keep]

    # rotate (x,y,z)_HM -> (X,Y,Z) = (y, z, x); origin: X,Y at min, Z so that the raft bottom is 0
    Q = np.column_stack([P[:, 1], P[:, 2], P[:, 0]])
    used = np.unique(conn)
    Q = Q - Q[used].min(0)
    # snap the grid to integers for node sharing
    G = np.round(Q / h).astype(np.int64)
    if np.abs(Q / h - G).max() > 1e-3:
        raise SystemExit("nodes are not on a uniform voxel grid")
    key = {}
    pts = []

    def nid(g):
        t = (int(g[0]), int(g[1]), int(g[2]))
        if t not in key:
            key[t] = len(pts)
            pts.append(t)
        return key[t]

    cells = []
    region = []
    Gc = G[conn]                                  # (n,8,3) integer corners
    for c in Gc:
        cells.append([nid(g) for g in c])
        region.append(1)
    # raft: replicate the bottom part layer downward
    zmin = Gc[:, :, 2].min()
    bottom = Gc[Gc[:, :, 2].min(1) == zmin]
    for r in range(1, args.raft_layers + 1):
        for c in bottom:
            cc = c.copy()
            cc[:, 2] -= r
            cells.append([nid(g) for g in cc])
            region.append(0)
    cells = np.array(cells, np.int64)
    region = np.array(region, np.int32)
    pts = np.array(pts, float) * h
    pts[:, 2] -= pts[:, 2].min()
    # orientation: HyperMesh C3D8 order should already be positive; verify
    J, L = hex_metrics(pts[cells])
    if (J <= 0).any():
        # flip bottom/top faces if HM wrote the opposite handedness
        cells = cells[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        J, L = hex_metrics(pts[cells])
    part_vol = float((J[region == 1] * 8).sum())
    raft_h = args.raft_layers * h
    zc = pts[cells].mean(1)[:, 2]
    slab = np.where(region == 1, np.floor((zc - raft_h) / h + 1e-6).astype(int), -1)
    rep = {"source": str(args.inp), "voxel_m": h, "cells_in": int(len(keep)), "cells_centre_inside": int(keep.sum()),
           "part_cells": int((region == 1).sum()), "raft_cells": int((region == 0).sum()), "nodes": int(len(pts)),
           "mechanics_dof": int(3 * len(pts)), "jacobian_nonpositive": int((J <= 0).sum()),
           "part_volume_m3": part_vol, "tet_volume_m3": tet_vol, "part_volume_over_tet": part_vol / tet_vol,
           "slabs": int(slab.max() + 1), "raft_height_m": raft_h, "part_base_z_m": raft_h,
           "bbox_m": [pts.min(0).tolist(), pts.max(0).tolist()], "cells_per_slab_first_last": [int((slab == 0).sum()), int((slab == slab.max()).sum())]}
    print(json.dumps(rep, indent=2))
    inp = args.out_dir / f"{args.tag}.inp"
    with inp.open("w", encoding="utf-8", newline="\n") as f:
        f.write("*HEADING\n0119 HyperMesh voxel HEX8 (centre-inside filtered) + raft, build axis +Z, units m\n")
        f.write(f"** source {args.inp.name}; voxel {h*1e3:.3f} mm; raft {args.raft_layers} layers\n*NODE\n")
        for n, (x, y, z) in enumerate(pts, 1):
            f.write(f"{n}, {x:.9e}, {y:.9e}, {z:.9e}\n")
        for name, sel in (("PART", region == 1), ("RAFT", region == 0)):
            f.write(f"*ELEMENT, TYPE=C3D8, ELSET={name}\n")
            for e_, row in zip(np.nonzero(sel)[0] + 1, cells[sel] + 1):
                f.write(f"{e_}, " + ", ".join(str(v) for v in row) + "\n")
        base = np.nonzero(pts[:, 2] < 0.5 * h)[0] + 1
        f.write("*NSET, NSET=BASE\n")
        for i in range(0, len(base), 16):
            f.write(", ".join(str(v) for v in base[i:i + 16]) + "\n")
    meshio.Mesh(pts, [("hexahedron", cells)], cell_data={"region": [region], "slab": [slab.astype(np.int32)]}).write(args.out_dir / f"{args.tag}.vtu")
    (args.out_dir / f"{args.tag}_report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("wrote", inp)


if __name__ == "__main__":
    main()
