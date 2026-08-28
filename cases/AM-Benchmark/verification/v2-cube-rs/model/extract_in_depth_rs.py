#!/usr/bin/env python3
"""In-depth residual-stress profile from a released cube run (Balbaa Sec 3.4 / 4.3.2 protocol).

Balbaa: XRD on the top face after electro-polishing in 0.1 mm increments down to
1 mm; stresses parallel to the laser scan direction and perpendicular to it
(hatch direction); each value averaged over a 2 mm diameter spot at the cube
centre; the model side (Sec 4.3.2) is "averaged across an area with a
diameter of 2 mm" after 600 s cooling and separation from the plate.

This script reads release.vtu, keeps PART cells (centroid above the substrate),
selects cells whose in-plane centroid lies within the 2 mm spot around the cube
centre, bins them by depth below the top face, and averages the quad-averaged
stress components per bin. Depth bins are one cell (200 um) wide: a 0.1 mm
XRD step is finer than the mesh, so the profile is reported at the cell
centres (0.1, 0.3, 0.5, ... mm) and, for the 0.1 mm grid, linearly
interpolated between them (flagged as interpolated). Removal-induced
re-equilibration of the polished layers is NOT modelled (neither by Balbaa).

Under the flash reading (D-V2-11 A') there is no scan direction: sigma_xx and
sigma_yy are both reported; their difference measures the model's in-plane
anisotropy (expected ~0), not a scan/hatch effect.

Usage: extract_in_depth_rs.py --run <dir> --preflight <dir> [--spot-diameter 2e-3] [--depth 1e-3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np


def cell_field(mesh, name):
    d = mesh.cell_data_dict.get(name)
    return None if d is None else np.asarray(list(d.values())[0]).reshape(-1)


def quad_average(mesh, pattern):
    parts = [cell_field(mesh, pattern.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def hex_centroids(mesh):
    pts = np.asarray(mesh.points)
    for block in mesh.cells:
        conn = np.asarray(block.data)
        if conn.shape[1] == 8:
            return pts[conn].mean(axis=1)
    raise SystemExit("no HEX8 block")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--vtu", default="release.vtu", help="frame to read (release.vtu = after plate separation)")
    ap.add_argument("--spot-diameter", type=float, default=2.0e-3)
    ap.add_argument("--depth", type=float, default=1.0e-3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    pre = json.loads((args.preflight / "v2_cube_smoke_ledger.json").read_text(encoding="utf-8"))
    sub_z = float(pre["substrate_z_m"])
    side = float(pre["footprint_m"])
    origin = float(pre["part_origin_xy_m"])
    cell = float(pre["layers"][0]["deposition_z_m"] - sub_z)
    top_z = sub_z + int(pre["activation_slabs"]) * cell
    cx = cy = origin + 0.5 * side
    mesh = meshio.read(args.run / args.vtu)
    c = hex_centroids(mesh)
    part = c[:, 2] > sub_z + 1e-9
    removed = cell_field(mesh, "release_removed")
    if removed is not None:
        part &= removed < 0.5
    in_spot = (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 <= (0.5 * args.spot_diameter) ** 2
    depth = top_z - c[:, 2]
    sel = part & in_spot & (depth >= 0.0) & (depth <= args.depth + 0.5 * cell)
    comps = {name: quad_average(mesh, f"stress_quadQ_{name}") for name in ("xx", "yy", "zz", "xy")}
    vm = quad_average(mesh, "vm_quadQ")
    eqp = cell_field(mesh, "eq_plastic_strain")
    if any(v is None for v in comps.values()) or vm is None:
        raise SystemExit("stress quad fields missing in the VTU")
    # one bin per cell row below the top face
    rows = []
    d_sel = depth[sel]
    levels = np.unique(np.round(d_sel / cell)) * cell   # cell-centre depths: 0.5*cell, 1.5*cell, ...
    for lv in sorted(levels):
        m = sel & (np.abs(depth - lv) < 0.25 * cell)
        rows.append({
            "depth_m": float(lv), "cells": int(m.sum()),
            "sxx_MPa": float(comps["xx"][m].mean() / 1e6), "syy_MPa": float(comps["yy"][m].mean() / 1e6),
            "szz_MPa": float(comps["zz"][m].mean() / 1e6), "sxy_MPa": float(comps["xy"][m].mean() / 1e6),
            "vm_MPa": float(vm[m].mean() / 1e6),
            "sxx_std_MPa": float(comps["xx"][m].std() / 1e6), "syy_std_MPa": float(comps["yy"][m].std() / 1e6),
            "eqp_mean": float(eqp[m].mean()) if eqp is not None else None,
        })
    # 0.1 mm XRD grid by linear interpolation between cell centres (flagged)
    grid = np.arange(0.0, args.depth + 1e-12, 1.0e-4)
    dz = np.array([r["depth_m"] for r in rows]); sxx = np.array([r["sxx_MPa"] for r in rows]); syy = np.array([r["syy_MPa"] for r in rows])
    interp = [{"depth_m": float(g), "sxx_MPa": float(np.interp(g, dz, sxx)), "syy_MPa": float(np.interp(g, dz, syy)),
               "interpolated": bool(not np.any(np.abs(dz - g) < 1e-9))} for g in grid]
    out = {
        "schema": "v2.cube-in-depth-rs/1",
        "protocol": "Balbaa Sec 3.4/4.3.2: 2 mm spot at the cube centre, top face, depth 0-1 mm; cell-row bins of 200 um; "
                    "0.1 mm grid linearly interpolated (flagged); no polishing re-equilibration",
        "frame": args.vtu, "spot_diameter_m": args.spot_diameter, "centre_xy_m": [cx, cy], "top_z_m": top_z,
        "cells_in_spot_per_row": rows[0]["cells"] if rows else 0,
        "profile_cell_rows": rows, "profile_0p1mm_grid": interp,
        "in_plane_anisotropy_note": "flash reading has no scan direction; sxx vs syy difference is model anisotropy only",
        "whole_part": {"vm_mean_MPa": float(vm[part].mean() / 1e6), "vm_max_MPa": float(vm[part].max() / 1e6),
                       "sxx_mean_MPa": float(comps["xx"][part].mean() / 1e6), "eqp_max": float(eqp[part].max()) if eqp is not None else None},
    }
    out_path = args.output or (args.run / "in_depth_rs.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{'depth mm':>9s} {'cells':>5s} {'sxx MPa':>9s} {'syy MPa':>9s} {'szz MPa':>9s} {'vm MPa':>8s} {'eqp':>8s}")
    for r in rows:
        print(f"{r['depth_m']*1e3:9.2f} {r['cells']:5d} {r['sxx_MPa']:9.1f} {r['syy_MPa']:9.1f} {r['szz_MPa']:9.1f} {r['vm_MPa']:8.1f} {r['eqp_mean'] if r['eqp_mean'] is None else round(r['eqp_mean'],4):>8}")
    print(f"whole part: vm mean {out['whole_part']['vm_mean_MPa']:.1f} / max {out['whole_part']['vm_max_MPa']:.1f} MPa; written {out_path}")


if __name__ == "__main__":
    main()
