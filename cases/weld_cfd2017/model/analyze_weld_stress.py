#!/usr/bin/env python3
"""Top-surface residual-stress lines for the weld case (Lu 2020 Fig. 7 style; copied from cases/weld_a7n01 for the rongchi branch).

Reads the last mechanics VTU of a run, averages the eight quadrature stresses per
cell, keeps the top layer of cells, and extracts:
  line A: across the weld (x) at a y-station           -> longitudinal (yy) and transverse (xx)
  line B: along the weld (y) at an x offset from centre -> longitudinal and transverse
Writes CSVs and a JSON summary with peak tensile / compressive values.

Usage: analyze_weld_stress.py RUN_DIR [--y-station 0.03] [--x-centre 0.15] [--line-b-offset 0.02]
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
    for c in ("xx", "yy", "zz", "xy", "yz", "xz"):
        comps[c] = np.mean([cd[f"stress_quad{q}_{c}"] for q in range(nq)], axis=0)
    comps["vm"] = np.mean([cd[f"vm_quad{q}"] for q in range(nq)], axis=0)
    return comps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--y-station", type=float, default=None)
    p.add_argument("--x-centre", type=float, default=None)
    p.add_argument("--line-b-offset", type=float, default=0.02, help="x offset of line B from the weld centre [m]")
    p.add_argument("--snapshot", default=None, help="VTU file name; default = last step_*.vtu")
    a = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.run_dir, "step_*.vtu")),
                   key=lambda f: int(re.search(r"step_(\d+)_", os.path.basename(f)).group(1)))
    f = os.path.join(a.run_dir, a.snapshot) if a.snapshot else files[-1]
    m = meshio.read(f)
    cd = {k: np.asarray(v[0]).ravel() for k, v in m.cell_data.items()}
    if "stress_quad0_xx" not in cd:
        raise SystemExit(f"{f} has no mechanics stresses")
    cells = m.cells[0].data
    verts = m.points[cells]
    cen = verts.mean(axis=1)
    s = cell_mean_stress(cd)
    top = float(m.points[:, 2].max())
    top_cells = verts[:, :, 2].max(axis=1) >= top - 1e-12
    xc = a.x_centre if a.x_centre is not None else 0.5 * (m.points[:, 0].min() + m.points[:, 0].max())
    ys = a.y_station if a.y_station is not None else 0.5 * (m.points[:, 1].min() + m.points[:, 1].max())

    # line A: top cells whose y-extent covers the station
    ymin, ymax = verts[:, :, 1].min(axis=1), verts[:, :, 1].max(axis=1)
    la = top_cells & (ymin <= ys) & (ymax >= ys)
    order = np.argsort(cen[la, 0])
    line_a = np.column_stack([cen[la, 0][order], s["yy"][la][order], s["xx"][la][order], s["vm"][la][order],
                              cd["material_state"][la][order]])
    # line B: top cells whose x-extent covers xc + offset
    xb = xc + a.line_b_offset
    xmin, xmax = verts[:, :, 0].min(axis=1), verts[:, :, 0].max(axis=1)
    lb = top_cells & (xmin <= xb) & (xmax >= xb)
    order = np.argsort(cen[lb, 1])
    line_b = np.column_stack([cen[lb, 1][order], s["yy"][lb][order], s["xx"][lb][order], s["vm"][lb][order],
                              cd["material_state"][lb][order]])
    hdr = "coord_m,sigma_long_yy_Pa,sigma_trans_xx_Pa,vm_Pa,material_state"
    np.savetxt(os.path.join(a.run_dir, "line_A_top.csv"), line_a, delimiter=",", header=hdr, comments="")
    np.savetxt(os.path.join(a.run_dir, "line_B_top.csv"), line_b, delimiter=",", header=hdr, comments="")

    def peaks(arr, col):
        return dict(max_MPa=float(arr[:, col].max()) / 1e6, min_MPa=float(arr[:, col].min()) / 1e6,
                    at_max_m=float(arr[np.argmax(arr[:, col]), 0]), at_min_m=float(arr[np.argmin(arr[:, col]), 0]))

    summary = dict(snapshot=os.path.basename(f), y_station_m=ys, x_centre_m=xc, line_b_x_m=xb,
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
