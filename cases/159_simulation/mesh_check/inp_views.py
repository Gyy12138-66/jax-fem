#!/usr/bin/env python3
"""Boundary surface of a C3D4 inp: area, mean-thickness proxy 2V/A, three orthographic views."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp")
from inp_health import parse, faces_of, tet_metrics

path, png = sys.argv[1], sys.argv[2]
nodes, elems, _, _ = parse(path)
ids = np.array(sorted(nodes)); P = np.array([nodes[i] for i in ids]); idx = {int(n): k for k, n in enumerate(ids)}
conn = None
for (etype, _), rows in elems.items():
    if etype.startswith("C3D4"):
        arr = np.array([r[:5] for r in rows]); conn = np.vectorize(idx.get)(arr[:, 1:])
F = np.concatenate(faces_of(conn, "C3D4"), axis=0)
Fu, c = np.unique(F, axis=0, return_counts=True)
B = Fu[c == 1]
X = P[B]
n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]); area = 0.5 * np.linalg.norm(n, axis=1)
vol = np.abs(tet_metrics(P[conn])[0]).sum()
lo, hi = P.min(0), P.max(0)
info = {"boundary_faces": int(len(B)), "surface_area_m2": float(area.sum()), "volume_m3": float(vol),
        "mean_thickness_2V_over_A_mm": float(2 * vol / area.sum() * 1e3),
        "bbox_fill_fraction": float(vol / np.prod(hi - lo)),
        "extent_mm": ((hi - lo) * 1e3).tolist()}
# boundary-face normal orientation histogram (which axis do flat faces align with?)
nn = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
for k, ax in enumerate("xyz"):
    m = np.abs(nn[:, k]) > 0.99
    info[f"area_fraction_faces_normal_to_{ax}"] = float(area[m].sum() / area.sum())
# z-slices: cross-section area proxy by boundary-face count in 10 bands along each axis
for k, ax in enumerate("xyz"):
    zc = X[:, :, k].mean(1); h, _ = np.histogram(zc, bins=10, range=(lo[k], hi[k]), weights=area)
    info[f"surface_area_profile_along_{ax}"] = [round(float(v / area.sum()), 3) for v in h]
print(json.dumps(info, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
Pmm = (P - lo) * 1e3
views = [("top  (x-y, looking -z)", 0, 1, 2, 1), ("front (x-z, looking +y)", 0, 2, 1, -1), ("side  (y-z, looking -x)", 1, 2, 0, 1)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax_, (title, a, b, d, sgn) in zip(axes, views):
    Q = Pmm[B]; depth = sgn * Q[:, :, d].mean(1); order = np.argsort(depth)
    polys = Q[order][:, :, [a, b]]
    shade = (depth[order] - depth.min()) / max(np.ptp(depth), 1e-9)
    pc = PolyCollection(polys, array=shade, cmap="viridis", edgecolors="none", linewidths=0)
    ax_.add_collection(pc); ax_.set_xlim(0, (hi - lo)[a] * 1e3); ax_.set_ylim(0, (hi - lo)[b] * 1e3)
    ax_.set_aspect("equal"); ax_.set_title(title); ax_.set_xlabel("xyz"[a] + " (mm)"); ax_.set_ylabel("xyz"[b] + " (mm)")
fig.suptitle(f"0119: {len(B)} boundary faces, V = {vol*1e6:.1f} cm3, 2V/A = {info['mean_thickness_2V_over_A_mm']:.2f} mm, bbox fill {info['bbox_fill_fraction']*100:.1f} %")
fig.tight_layout(); fig.savefig(png, dpi=130)
print("saved", png)
