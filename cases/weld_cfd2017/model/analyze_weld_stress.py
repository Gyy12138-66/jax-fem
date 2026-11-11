#!/usr/bin/env python3
"""Top-surface residual-stress lines for the weld case (Lu 2020 Fig. 7 style).

Works with either axis convention:
  --weld-axis x (default, CFD axes): longitudinal = sigma_xx, transverse = sigma_yy
  --weld-axis y (legacy meshes):     longitudinal = sigma_yy, transverse = sigma_xx

Reads the last mechanics VTU of a run, takes the top layer of cells and extracts:
  line A: across the weld at a longitudinal station -> longitudinal and transverse stress
  line B: along the weld at a transverse offset from the centreline
Cell-mean fields (sigma_xx..xz, von_mises) are used when present, otherwise the
per-quadrature-point arrays are averaged.

Usage: analyze_weld_stress.py RUN_DIR [--weld-axis x] [--station 0.022] [--centre 0.0]
                              [--line-b-offset 0.005] [--snapshot step_000136_cooling.vtu]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import meshio
import numpy as np


def cell_mean_stress(cd, nq=8):
    comps = {}
    if "sigma_xx" in cd and "von_mises" in cd:
        for c in ("xx", "yy", "zz", "xy", "yz", "xz"):
            comps[c] = cd[f"sigma_{c}"]
        comps["vm"] = cd["von_mises"]
        return comps
    for c in ("xx", "yy", "zz", "xy", "yz", "xz"):
        comps[c] = np.mean([cd[f"stress_quad{q}_{c}"] for q in range(nq)], axis=0)
    comps["vm"] = np.mean([cd[f"vm_quad{q}"] for q in range(nq)], axis=0)
    return comps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--weld-axis", choices=("x", "y"), default="x")
    p.add_argument("--station", type=float, default=None, help="longitudinal coordinate of line A [m]; default mid-length")
    p.add_argument("--centre", type=float, default=None, help="transverse coordinate of the weld centreline [m]; default the min face")
    p.add_argument("--line-b-offset", type=float, default=0.005, help="transverse offset of line B from the centreline [m]")
    p.add_argument("--snapshot", default=None, help="VTU file name; default = last step_*.vtu")
    a = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.run_dir, "step_*.vtu")),
                   key=lambda f: int(re.search(r"step_(\d+)_", os.path.basename(f)).group(1)))
    f = os.path.join(a.run_dir, a.snapshot) if a.snapshot else files[-1]
    m = meshio.read(f)
    cd = {k: np.asarray(v[0]).ravel() for k, v in m.cell_data.items()}
    if "stress_quad0_xx" not in cd and "sigma_xx" not in cd:
        raise SystemExit(f"{f} has no mechanics stresses")
    cells = m.cells[0].data
    verts = m.points[cells]
    cen = verts.mean(axis=1)
    s = cell_mean_stress(cd)

    LA = 0 if a.weld_axis == "x" else 1          # longitudinal axis index
    TA = 1 - LA                                  # transverse axis index
    lon = "xx" if a.weld_axis == "x" else "yy"   # longitudinal stress component
    tra = "yy" if a.weld_axis == "x" else "xx"

    top = float(m.points[:, 2].max())
    top_cells = verts[:, :, 2].max(axis=1) >= top - 1e-12
    centre = a.centre if a.centre is not None else float(m.points[:, TA].min())
    station = a.station if a.station is not None else 0.5 * (m.points[:, LA].min() + m.points[:, LA].max())

    lo, hi = verts[:, :, LA].min(axis=1), verts[:, :, LA].max(axis=1)
    la = top_cells & (lo <= station) & (hi >= station)
    order = np.argsort(cen[la, TA])
    line_a = np.column_stack([cen[la, TA][order], s[lon][la][order], s[tra][la][order],
                              s["vm"][la][order], cd["material_state"][la][order]])

    tb = centre + a.line_b_offset
    lo, hi = verts[:, :, TA].min(axis=1), verts[:, :, TA].max(axis=1)
    lb = top_cells & (lo <= tb) & (hi >= tb)
    order = np.argsort(cen[lb, LA])
    line_b = np.column_stack([cen[lb, LA][order], s[lon][lb][order], s[tra][lb][order],
                              s["vm"][lb][order], cd["material_state"][lb][order]])

    hdr = "coord_m,sigma_longitudinal_Pa,sigma_transverse_Pa,vm_Pa,material_state"
    np.savetxt(os.path.join(a.run_dir, "line_A_top.csv"), line_a, delimiter=",", header=hdr, comments="")
    np.savetxt(os.path.join(a.run_dir, "line_B_top.csv"), line_b, delimiter=",", header=hdr, comments="")

    def peaks(arr, col):
        return dict(max_MPa=float(arr[:, col].max()) / 1e6, min_MPa=float(arr[:, col].min()) / 1e6,
                    at_max_m=float(arr[np.argmax(arr[:, col]), 0]), at_min_m=float(arr[np.argmin(arr[:, col]), 0]))

    summary = dict(snapshot=os.path.basename(f), weld_axis=a.weld_axis,
                   longitudinal_component=f"sigma_{lon}", transverse_component=f"sigma_{tra}",
                   station_m=station, centre_m=centre, line_b_transverse_m=tb,
                   top_cells=int(top_cells.sum()), vm_max_MPa=float(s["vm"].max()) / 1e6,
                   vm_max_top_MPa=float(s["vm"][top_cells].max()) / 1e6,
                   line_A=dict(points=int(la.sum()), longitudinal=peaks(line_a, 1), transverse=peaks(line_a, 2)),
                   line_B=dict(points=int(lb.sum()), longitudinal=peaks(line_b, 1), transverse=peaks(line_b, 2)),
                   eq_plastic_strain_max=float(cd["eq_plastic_strain"].max()) if "eq_plastic_strain" in cd else None)
    with open(os.path.join(a.run_dir, "stress_lines_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
