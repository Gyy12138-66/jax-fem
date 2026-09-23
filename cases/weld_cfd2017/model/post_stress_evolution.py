# -*- coding: utf-8 -*-
"""Post-process the 40-frame stress-evolution run: validate VTU fields, write stress.pvd (time series),
and a CSV of the evolution of key quantities. Usage: post_40f.py <run_dir>"""
import os, sys, glob, re, csv
import numpy as np
import meshio

O = sys.argv[1]
files = sorted(glob.glob(os.path.join(O, "step_*.vtu")), key=lambda f: int(re.search(r"step_(\d+)_", f).group(1)))
steps = [int(re.search(r"step_(\d+)_", f).group(1)) for f in files]
# step -> end time from path_used.csv (row k = step k)
rows = list(csv.DictReader(open(os.path.join(O, "path_used.csv"))))
t_of_step = {k: float(r["time"]) for k, r in enumerate(rows)}
times = [t_of_step[s] for s in steps]

need = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_yz", "sigma_xz", "von_mises", "eq_plastic_strain", "stress_free_temperature", "max_temperature_history"]
evo = []
xc = 0.010
for f, s, t in zip(files, steps, times):
    m = meshio.read(f)
    cd = {k: np.asarray(v[0]).ravel() for k, v in m.cell_data.items()}
    missing = [k for k in need if k not in cd]
    quad_left = [k for k in cd if re.search(r"(^|_)quad\d+(_|$)", k)]
    if missing or quad_left:
        raise SystemExit("frame %s: missing %s / per-quad arrays left %s" % (f, missing, quad_left[:3]))
    u = m.point_data["u"]
    cells = m.cells[0].data; cen = m.points[cells].mean(axis=1)
    top = m.points[cells][:, :, 2].max(axis=1) >= 0.006 - 1e-9
    # weld-centre top cell at mid-length (x=xc, y=21.5 mm)
    sel = top & (np.abs(cen[:, 0] - xc) < 0.0003) & (np.abs(cen[:, 1] - 0.0215) < 0.0003)
    evo.append(dict(step=s, time=t, T_max=float(np.asarray(m.point_data["T"]).max()),
                    vm_max_MPa=cd["von_mises"].max() / 1e6,
                    n_cells_vm_gt_200MPa=int((cd["von_mises"] > 200e6).sum()),
                    sigma_long_weld_centre_MPa=float(cd["sigma_yy"][sel].mean()) / 1e6 if sel.any() else float("nan"),
                    sigma_trans_weld_centre_MPa=float(cd["sigma_xx"][sel].mean()) / 1e6 if sel.any() else float("nan"),
                    sigma_long_min_MPa=cd["sigma_yy"].min() / 1e6, sigma_long_max_MPa=cd["sigma_yy"].max() / 1e6,
                    eqp_max=cd["eq_plastic_strain"].max(),
                    u_max_mm=float(np.linalg.norm(u, axis=1).max()) * 1e3,
                    uz_min_mm=float(u[:, 2].min()) * 1e3, uz_max_mm=float(u[:, 2].max()) * 1e3,
                    melted_cells=int((cd["max_temperature_history"] >= 923).sum())))
with open(os.path.join(O, "stress_evolution.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(evo[0].keys())); w.writeheader(); w.writerows(evo)
with open(os.path.join(O, "stress.pvd"), "w") as fh:
    fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
    for f, t in zip(files, times):
        fh.write('<DataSet timestep="%.6f" group="" part="0" file="%s"/>\n' % (t, os.path.basename(f)))
    fh.write('</Collection>\n</VTKFile>\n')
print("frames: %d, times %.3f .. %.3f s, VTU size %.1f MB each; fields OK, no per-quad arrays" % (len(files), times[0], times[-1], os.path.getsize(files[-1]) / 1e6))
print("%6s %7s %8s %7s %9s %9s %8s %7s" % ("step", "t[s]", "Tmax", "vm_max", "s_long_wc", "s_trans_wc", "eqp_max", "u_max"))
for e in evo:
    print("%6d %7.3f %8.1f %7.1f %9.1f %9.1f %8.4f %7.3f" % (e["step"], e["time"], e["T_max"], e["vm_max_MPa"], e["sigma_long_weld_centre_MPa"], e["sigma_trans_weld_centre_MPa"], e["eqp_max"], e["u_max_mm"]))
