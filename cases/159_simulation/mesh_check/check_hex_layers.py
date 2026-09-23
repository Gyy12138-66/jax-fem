#!/usr/bin/env python3
"""Acceptance check for a HEX8 (C3D8) inp meant for the layer-activation runner.

Checks: single C3D8 family; ELSETs PART / RAFT (optional); node levels along the build axis are
multiples of the layer height (dz) and every cell spans exactly one layer; conformity (every face
shared by <= 2 cells, closed boundary); centre Jacobians > 0; cell edge statistics; cells across
the sheet thickness (median of the shortest in-plane edge); volume against a reference TET4 inp.

    check_hex_layers.py mesh.inp [--build-axis z] [--dz 2e-4] [--scale 1.0] [--ref-tet 0119_c3d4_only.inp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inp_health import parse, faces_of, hex_metrics, tet_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("--build-axis", choices=("x", "y", "z"), default="z")
    ap.add_argument("--dz", type=float, default=2e-4, help="layer height [m] after scaling")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply coordinates by this (1e-3 for an inp in mm)")
    ap.add_argument("--ref-tet", type=Path, default=None, help="reference TET4 inp (metres) for the volume check")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    nodes, elems, nsets, elsets = parse(args.inp)
    ids = np.array(sorted(nodes))
    P = np.array([nodes[i] for i in ids]) * args.scale
    idx = {int(n): k for k, n in enumerate(ids)}
    rep = {"inp": str(args.inp), "nodes": int(len(ids))}
    types = {t for (t, _) in elems}
    rep["element_types"] = sorted(types)
    hexes = [(es, rows) for (t, es), rows in elems.items() if t.startswith("C3D8")]
    if not hexes or len(types) != 1:
        rep["verdict"] = f"FAIL: expected only C3D8 blocks, found {sorted(types)}"
        print(json.dumps(rep, indent=2))
        return
    conn = np.vstack([np.vectorize(idx.get)(np.array([r[:9] for r in rows])[:, 1:]) for _, rows in hexes])
    block = np.concatenate([[es or f"block{i}"] * len(rows) for i, (es, rows) in enumerate(hexes)])
    rep["cells"] = int(len(conn))
    rep["elsets"] = {k: int((block == k).sum()) for k in np.unique(block)}
    rep["elsets_declared"] = {k: len(v) for k, v in elsets.items()}
    ax = "xyz".index(args.build_axis)
    z = P[:, ax]
    lev = z / args.dz
    rep["build_axis_levels"] = {"min_m": float(z.min()), "max_m": float(z.max()),
                               "max_offset_from_layer_grid_m": float(np.abs(lev - np.round(lev)).max() * args.dz),
                               "distinct_levels": int(len(np.unique(np.round(lev))))}
    zc = z[conn]
    span = (zc.max(1) - zc.min(1)) / args.dz
    rep["cells_span_one_layer"] = {"fraction": float(np.mean(np.abs(span - 1) < 1e-3)), "span_min_max_layers": [float(span.min()), float(span.max())]}
    rep["cells_per_layer"] = {"first": int((np.round(zc.min(1) / args.dz) == np.round(zc.min(1) / args.dz).min()).sum()),
                              "max": int(np.bincount(np.round(zc.min(1) / args.dz).astype(int) - int(np.round(zc.min(1) / args.dz).min())).max())}
    J, L = hex_metrics(P[conn])
    rep["jacobian"] = {"nonpositive_cells": int((J <= 0).sum()), "volume_m3": float(np.abs(J).sum() * 8)}
    inplane = [e for e in ((0, 1), (1, 2), (2, 3), (3, 0)) ]
    Lin = np.stack([np.linalg.norm(P[conn[:, a]] - P[conn[:, b]], axis=1) for a, b in inplane], axis=1)
    rep["edges"] = {"in_plane_min_med_max_m": [float(Lin.min()), float(np.median(Lin)), float(Lin.max())],
                    "shortest_in_plane_edge_median_m": float(np.median(Lin.min(1))),
                    "aspect_max_over_min_med_max": [float(np.median(L.max(1) / L.min(1))), float((L.max(1) / L.min(1)).max())]}
    F = np.concatenate(faces_of(conn, "C3D8"), axis=0)
    Fu, c = np.unique(F, axis=0, return_counts=True)
    rep["faces"] = {"boundary": int((c == 1).sum()), "interior": int((c == 2).sum()), "nonmanifold": int((c > 2).sum())}
    q = np.round(P, 9)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    rep["duplicate_coordinate_nodes"] = int((cnt - 1).sum())
    if args.ref_tet is not None:
        rn, re_, _, _ = parse(args.ref_tet)
        rid = np.array(sorted(rn))
        RP = np.array([rn[i] for i in rid])
        ridx = {int(n): k for k, n in enumerate(rid)}
        rconn = np.vstack([np.vectorize(ridx.get)(np.array([r[:5] for r in rows])[:, 1:]) for (t, _), rows in re_.items() if t.startswith("C3D4")])
        tv = float(np.abs(tet_metrics(RP[rconn])[0]).sum())
        part = block == "PART" if "PART" in rep["elsets"] else np.ones(len(conn), bool)
        rep["volume_check"] = {"tet_volume_m3": tv, "hex_part_volume_m3": float(np.abs(J[part]).sum() * 8), "ratio": float(np.abs(J[part]).sum() * 8 / tv)}
    problems = []
    if rep["build_axis_levels"]["max_offset_from_layer_grid_m"] > 1e-6:
        problems.append("node levels are not on the layer grid")
    if rep["cells_span_one_layer"]["fraction"] < 0.999:
        problems.append("cells spanning != 1 layer")
    if rep["jacobian"]["nonpositive_cells"]:
        problems.append("non-positive Jacobians")
    if rep["faces"]["nonmanifold"]:
        problems.append("non-manifold faces")
    if rep["duplicate_coordinate_nodes"]:
        problems.append("duplicate nodes (unmerged blocks?)")
    if "PART" not in rep["elsets"]:
        problems.append("no ELSET=PART")
    rep["verdict"] = "PASS" if not problems else "FAIL: " + "; ".join(problems)
    txt = json.dumps(rep, indent=2)
    print(txt)
    if args.json:
        args.json.write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
