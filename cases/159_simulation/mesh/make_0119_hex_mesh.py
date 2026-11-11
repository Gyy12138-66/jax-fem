#!/usr/bin/env python3
"""Body-fitted HEX8 mesh for the 0119 bent-sheet part (+ raft), rotated to build along +Z.

Method (block-structured curvilinear voxels):
  1. slice the closed TET4 boundary surface at a thin, uniform station -> section polygon -> sheet
     mid-curve C(s) (along-wall coordinate s) and normal n(s) by ray casting across the sheet; the
     spine is extrapolated straight beyond both leg tips so the longer base legs and the lugs fit;
  2. (s, t) grid on that curve: s cells ~hs on straights (refined at bends and in the lug windows),
     t cells ht across the sheet core (|t| <= t_core, so the 0.8 mm blade and the 3.3 mm foot are
     resolved) and geometrically graded beyond it (the lugs); the t datum is snapped to the most
     common surface offset so the dominant face is represented exactly;
  3. lugs: on many stations the two long faces of each lug are detected (line fit per face), giving
     their direction (constant) and their position s1(z), s2(z) (they slide along the leg with
     height) and the lug length t_end(z). Per height the chart is (a) stretched in s so that
     s1(z), s2(z) stay on grid lines, (b) sheared so that the t-lines follow the lug faces, and
     (c) scaled in t beyond the core so that the lug end is a grid surface -> the lug is a clean
     block instead of stair steps; the wide t band is only allowed inside the lug windows;
  4. at every build station (dz apart) keep the (s, t) cells whose sub-samples fall inside the section
     polygon of that station (thickness steps, taper and holes become stair steps);
  5. add a raft below the part with the base-station section as the fixture band.

Frame: old HyperMesh (x, y, z) [m] -> new (X, Y, Z) = (y - y_min, z - z_min, x - x_min + raft),
a proper rotation (det = +1): build direction +Z (old +x), section in X-Y, raft bottom Z = 0.

    make_0119_hex_mesh.py --inp 0119_c3d4_only.inp --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mesh_check"))
from inp_health import parse, faces_of, tet_metrics, hex_metrics  # noqa: E402


# ----------------------------------------------------------------------------- slicing
def boundary_triangles(P, conn):
    F = np.concatenate(faces_of(conn, "C3D4"), axis=0)
    Fu, c = np.unique(F, axis=0, return_counts=True)
    return P[Fu[c == 1]]


def slice_loops(T, x0, tol=1e-7, merge=1e-6):
    """Intersect boundary triangles T (n,3,3) with the plane old-x = x0 -> closed loops in (y,z)."""
    d = T[:, :, 0] - x0
    d = np.where(np.abs(d) < tol, tol, d)
    cross = (d.min(1) < 0) & (d.max(1) > 0)
    if not cross.any():
        return []
    Tc, dc = T[cross], d[cross]
    ends = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        m = dc[:, a] * dc[:, b] < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = dc[:, a] / (dc[:, a] - dc[:, b])
        p = Tc[:, a, 1:] + lam[:, None] * (Tc[:, b, 1:] - Tc[:, a, 1:])
        ends.append(np.where(m[:, None], p, np.nan))
    E = np.stack(ends, axis=1)
    ok = ~np.isnan(E[:, :, 0])
    if not (ok.sum(1) == 2).all():
        raise RuntimeError("triangle with != 2 plane crossings")
    S = E[ok].reshape(-1, 2, 2)
    pts = S.reshape(-1, 2)
    tree = cKDTree(pts)
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in tree.query_pairs(merge):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    vid = np.array([find(i) for i in range(len(pts))])
    seg_v = vid.reshape(-1, 2)
    adj = {}
    for si, (u, v) in enumerate(seg_v):
        adj.setdefault(u, []).append((v, si))
        adj.setdefault(v, []).append((u, si))
    used = np.zeros(len(seg_v), bool)
    loops = []
    for start in adj:
        if all(used[si] for _, si in adj[start]):
            continue
        loop = [start]
        cur = start
        while True:
            nxt = None
            for v, si in adj[cur]:
                if not used[si]:
                    nxt = (v, si)
                    break
            if nxt is None:
                break
            used[nxt[1]] = True
            cur = nxt[0]
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(pts[loop])
    return loops


def resample_closed(Q, h):
    Qc = np.vstack([Q, Q[:1]])
    seg = np.linalg.norm(np.diff(Qc, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(np.ceil(L / h)), 8)
    ss = np.linspace(0, L, n, endpoint=False)
    return np.column_stack([np.interp(ss, s, Qc[:, 0]), np.interp(ss, s, Qc[:, 1])]), L


def polygon_area(Q):
    return abs(0.5 * np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))


def inward_normals(R):
    area = 0.5 * np.sum(R[:, 0] * np.roll(R[:, 1], -1) - np.roll(R[:, 0], -1) * R[:, 1])
    t = np.roll(R, -1, axis=0) - np.roll(R, 1, axis=0)
    t = t / np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
    left = np.column_stack([-t[:, 1], t[:, 0]])
    return (left if area > 0 else -left), t


def ray_hits(O, N, A, B, eps=1e-6):
    """Min lambda > eps along rays O + lambda N to segments A-B (inf if none)."""
    E = B - A
    den = N[:, 0][:, None] * E[:, 1][None, :] - N[:, 1][:, None] * E[:, 0][None, :]
    AO = A[None, :, :] - O[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (AO[:, :, 0] * E[None, :, 1] - AO[:, :, 1] * E[None, :, 0]) / den
        mu = (AO[:, :, 0] * N[:, 1][:, None] - AO[:, :, 1] * N[:, 0][:, None]) / den
    ok = (np.abs(den) > 1e-300) & (mu >= -1e-9) & (mu <= 1 + 1e-9) & (lam > eps)
    return np.where(ok, lam, np.inf).min(axis=1)


def medial_from_loop(Q, h=1e-4, tmax=3e-3):
    """Ordered mid-curve points of the longest sheet-like run of the loop, with local thickness."""
    R, L = resample_closed(Q, h)
    N, _ = inward_normals(R)
    A, B = R, np.roll(R, -1, axis=0)
    lam = np.concatenate([ray_hits(R[i:i + 400], N[i:i + 400], A, B) for i in range(0, len(R), 400)])
    valid = np.isfinite(lam) & (lam < tmax)
    if not valid.all():
        first_bad = int(np.argmax(~valid))
        R, N, lam, valid = (np.roll(a, -first_bad, axis=0) for a in (R, N, lam, valid))
    runs, i = [], 0
    while i < len(valid):
        if valid[i]:
            j = i
            while j < len(valid) and valid[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        raise RuntimeError("no sheet-like run found in the reference section")
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    th = lam[i0:i1]
    med = float(np.median(th))
    good = (th > 0.6 * med) & (th < 1.4 * med)
    idx = np.arange(i0, i1)[good]
    M = R[idx] + 0.5 * lam[idx, None] * N[idx]
    return M, lam[idx], med, L, (i1 - i0) * h


def smooth_open(Y, w):
    w = int(w) | 1
    Y = np.atleast_2d(Y.T).T
    if w < 3 or len(Y) < w + 2:
        return Y.copy()
    pad = w // 2
    Yp = np.vstack([2 * Y[0] - Y[pad:0:-1], Y, 2 * Y[-1] - Y[-2:-pad - 2:-1]])
    k = np.ones(w) / w
    return np.column_stack([np.convolve(Yp[:, i], k, mode="valid") for i in range(Y.shape[1])])


def build_reference(M, hs, hmin, rfac, extend=0.0, lug_span=0.0, lug_h=None):
    """Smooth the medial polyline, extrapolate both ends straight by `extend`, resample it
    curvature-adaptively (and finer within `lug_span` of both ends)."""
    Ms = smooth_open(M, 11)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(Ms, axis=0), axis=1) > 1e-7])
    Ms = Ms[keep]
    if extend > 0:
        m = min(20, len(Ms) - 1)
        t0 = Ms[0] - Ms[m]
        t1 = Ms[-1] - Ms[-1 - m]
        t0 = t0 / np.linalg.norm(t0)
        t1 = t1 / np.linalg.norm(t1)
        Ms = np.vstack([Ms[0] + t0 * extend, Ms, Ms[-1] + t1 * extend])
    seg = np.linalg.norm(np.diff(Ms, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.gradient(Ms, s, axis=0)
    t = t / np.linalg.norm(t, axis=1)[:, None]
    ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    kappa = np.abs(np.gradient(ang, s))
    kappa = smooth_open(kappa[:, None], 21)[:, 0]
    hloc = np.clip(rfac / np.maximum(kappa, 1e-9), hmin, hs)
    if lug_span > 0 and lug_h:
        win = (s <= lug_span) | (s >= s[-1] - lug_span)
        hloc[win] = np.minimum(hloc[win], lug_h)
    dens = 1.0 / hloc
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * seg)])
    n = max(int(round(cum[-1])), 2)
    s_nodes = np.interp(np.linspace(0, cum[-1], n + 1), cum, s)
    dense_n = np.column_stack([-t[:, 1], t[:, 0]])
    return dict(s=s_nodes, dense_M=Ms, dense_s=s, dense_t=t, dense_n=dense_n, kappa=kappa, L=float(s[-1]))


def curve_at(ref, s):
    """Position, normal and tangent of the dense reference curve at arclength s (clamped)."""
    ds, dm, dn, dt = ref["dense_s"], ref["dense_M"], ref["dense_n"], ref["dense_t"]
    C = np.column_stack([np.interp(s, ds, dm[:, 0]), np.interp(s, ds, dm[:, 1])])
    n = np.column_stack([np.interp(s, ds, dn[:, 0]), np.interp(s, ds, dn[:, 1])])
    t = np.column_stack([np.interp(s, ds, dt[:, 0]), np.interp(s, ds, dt[:, 1])])
    return C, n / np.linalg.norm(n, axis=1)[:, None], t / np.linalg.norm(t, axis=1)[:, None]


def inside_by_nearest(loops, pts, h=1e-4):
    """Inside test via the inward normal of the nearest boundary sample (exact away from corners
    by more than the sample spacing h; O(N log N) instead of O(N x vertices))."""
    inside = np.zeros(len(pts), int)
    for Q in loops:
        R, _ = resample_closed(Q, h)
        N, _ = inward_normals(R)
        _, j = cKDTree(R).query(pts)
        inside += (np.einsum("ij,ij->i", pts - R[j], N[j]) > 0)
    return (inside % 2) == 1


def insert_nodes(s_nodes, targets, min_gap):
    """Insert target arclengths as grid nodes; drop existing nodes closer than min_gap to them."""
    first, last = s_nodes[0], s_nodes[-1]
    s = list(s_nodes)
    for tgt in targets:
        if tgt <= first or tgt >= last:
            continue
        s = [v for v in s if abs(v - tgt) > min_gap or v in (first, last)]
        s.append(tgt)
    return np.array(sorted(set(s)))


def graded_t_nodes(datum, ht, n_inner, n_outer, t_lo, t_hi, growth=1.4, hmax=1.0e-3):
    """Asymmetric grading across the thickness: `n_inner` fine cells of `ht` on the +t side of the
    datum (the 0.8 mm sheet, datum = its outer face), `n_outer` fine cells on the -t side (the foot's
    outward bulge), then geometrically growing cells (foot extra thickness, lugs) up to `hmax`."""
    inner = datum + ht * np.arange(0, n_inner + 1)
    outer = datum - ht * np.arange(1, n_outer + 1)[::-1]

    def grow(start, direction, limit):
        out, pos, h = [], start, ht
        while direction * (limit - pos) > 0:
            h = min(h * growth, hmax)
            pos = pos + direction * h
            out.append(pos)
        return out

    neg = grow(outer[0] if n_outer else inner[0], -1.0, t_lo - ht)[::-1]
    pos = grow(inner[-1], +1.0, t_hi + ht)
    return np.array(neg + list(outer) + list(inner) + pos), (n_inner, n_outer)


# ----------------------------------------------------------------------------- lugs
def detect_lug(Q, ref, s_lo, s_hi, off_min=2.5e-3, min_len=4e-3, far=4e-3):
    """Two long faces of the lug attached to the spine between s_lo and s_hi, from one section polygon.
    Returns dict(s1, d1, s2, d2, side, len1, len2, t_end) with s1 < s2, or None. d1/d2 are chart
    directions (unit vectors such that C(s) + t*d moves along the face for t of sign `side`);
    t_end is the lug's extent along its axis measured from the spine."""
    R, _ = resample_closed(Q, 2e-4)
    A, B = R, np.roll(R, -1, axis=0)
    mids = 0.5 * (A + B)
    e = B - A
    L = np.linalg.norm(e, axis=1)
    e = e / np.maximum(L, 1e-300)[:, None]
    tree = cKDTree(ref["dense_M"])
    dist, j = tree.query(mids)
    s_m = ref["dense_s"][j]
    nm = ref["dense_n"][j]
    tm = ref["dense_t"][j]
    Cm = ref["dense_M"][j]
    toff = np.einsum("ij,ij->i", mids - Cm, nm)
    transverse = np.abs(np.einsum("ij,ij->i", e, tm)) < np.cos(np.radians(55))
    sel = (s_m >= s_lo) & (s_m <= s_hi) & (np.abs(toff) > off_min) & transverse & (dist < 0.05)
    if L[sel].sum() < 2 * min_len:
        return None
    side = float(np.sign(np.sum(toff[sel] * L[sel])))
    sel &= np.sign(toff) == side
    away = np.einsum("ij,ij->i", e, nm) * side
    e_or = e * np.where(away < 0, -1.0, 1.0)[:, None]
    w = mids - Cm
    den = tm[:, 0] * e_or[:, 1] - tm[:, 1] * e_or[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (w[:, 0] * e_or[:, 1] - w[:, 1] * e_or[:, 0]) / den
    s_star = s_m + mu
    ok = sel & np.isfinite(s_star) & (np.abs(mu) < 0.1)
    if L[ok].sum() < 2 * min_len:
        return None
    idx_ok = np.nonzero(ok)[0]
    ss, ww = s_star[ok], L[ok]
    order = np.argsort(ss)
    cw = np.cumsum(ww[order]) / ww.sum()
    c = np.array([ss[order][np.searchsorted(cw, 0.25)], ss[order][np.searchsorted(cw, 0.75)]])
    for _ in range(30):
        lab = np.argmin(np.abs(ss[:, None] - c[None, :]), axis=1)
        for k in range(2):
            if ww[lab == k].sum() > 0:
                c[k] = np.average(ss[lab == k], weights=ww[lab == k])
    faces = []
    for k in range(2):
        m = idx_ok[lab == k]
        if L[m].sum() < min_len:
            return None
        d = np.average(e_or[m], axis=0, weights=L[m])
        d = d / np.linalg.norm(d)
        ctr = np.average(mids[m], axis=0, weights=L[m])
        _, jc = tree.query(ctr)
        Cc, tc = ref["dense_M"][jc], ref["dense_t"][jc]
        wc = ctr - Cc
        denc = tc[0] * d[1] - tc[1] * d[0]
        if abs(denc) < 1e-6:
            return None
        s_face = ref["dense_s"][jc] + (wc[0] * d[1] - wc[1] * d[0]) / denc
        faces.append((float(s_face), side * d, float(L[m].sum())))
    faces.sort(key=lambda f: f[0])
    (s1, d1, l1), (s2, d2, l2) = faces
    if s2 - s1 < 1.5e-3:
        return None
    dbar = d1 + d2
    dbar = dbar / np.linalg.norm(dbar)
    near = (s_m >= s1 - 8e-3) & (s_m <= s2 + 8e-3) & (np.abs(toff) > far) & (np.sign(toff) == side) & (dist < 0.05)
    t_end = None
    if near.any():
        C_mid, _, _ = curve_at(ref, np.array([0.5 * (s1 + s2)]))
        t_end = float(np.max((mids[near] - C_mid[0]) @ (side * dbar)))
    return dict(s1=s1, d1=d1, s2=s2, d2=d2, side=side, len1=l1, len2=l2, t_end=t_end)


def fit_lug(dets, min_len=4e-3, smooth=9):
    """Smoothed tracks s1(z), s2(z), t_end(z) and mean face directions from per-station detections."""
    good = [(z, lk) for z, lk in dets if lk is not None and min(lk["len1"], lk["len2"]) >= min_len]
    if len(good) < 3:
        return None
    zz = np.array([g[0] for g in good])
    w = np.array([min(g[1]["len1"], g[1]["len2"]) for g in good])
    s1 = np.array([g[1]["s1"] for g in good])
    s2 = np.array([g[1]["s2"] for g in good])
    te = np.array([g[1]["t_end"] if g[1]["t_end"] is not None else np.nan for g in good])
    b1, a1 = np.polyfit(zz, s1, 1, w=w)
    b2, a2 = np.polyfit(zz, s2, 1, w=w)
    d1 = np.average(np.array([g[1]["d1"] for g in good]), axis=0, weights=w)
    d2 = np.average(np.array([g[1]["d2"] for g in good]), axis=0, weights=w)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    side = float(np.sign(np.sum([g[1]["side"] * wi for g, wi in zip(good, w)])))
    dev = [float(np.degrees(np.arccos(np.clip(np.dot(g[1][k], dd), -1, 1)))) for g in good for k, dd in (("d1", d1), ("d2", d2))]
    order = np.argsort(zz)
    zz, s1, s2, te = zz[order], s1[order], s2[order], te[order]
    s1z = smooth_open(s1[:, None], smooth)[:, 0]
    s2z = smooth_open(s2[:, None], smooth)[:, 0]
    s2z = np.maximum(s2z, s1z + 1.5e-3)
    fin = ~np.isnan(te)
    tez = None
    if fin.sum() >= 2:
        te = np.interp(zz, zz[fin], te[fin])
        tez = smooth_open(te[:, None], smooth)[:, 0]
    r1 = s1 - (a1 + b1 * zz)
    r2 = s2 - (a2 + b2 * zz)
    return dict(a1=float(a1), b1=float(b1), a2=float(a2), b2=float(b2), d1=d1, d2=d2, side=side,
                zz=zz, s1z=s1z, s2z=s2z, tez=tez,
                z_min=float(zz.min()), z_max=float(zz.max()), n_detections=int(len(good)),
                fit_residual_rms_m=float(np.sqrt(np.mean(np.r_[r1, r2] ** 2))), fit_residual_max_m=float(np.abs(np.r_[r1, r2]).max()),
                smooth_residual_max_m=float(max(np.abs(s1 - s1z).max(), np.abs(s2 - s2z).max())),
                t_end_base_m=(float(tez[0]) if tez is not None else None),
                t_end_min_max_m=([float(tez.min()), float(tez.max())] if tez is not None else None),
                direction_deviation_max_deg=float(max(dev)))


def lug_at(lug, z):
    return dict(s1=float(np.interp(z, lug["zz"], lug["s1z"])), s2=float(np.interp(z, lug["zz"], lug["s2z"])),
                t_end=(float(np.interp(z, lug["zz"], lug["tez"])) if lug["tez"] is not None else None),
                d1=lug["d1"], d2=lug["d2"], side=lug["side"],
                s1_base=float(lug["s1z"][0]), s2_base=float(lug["s2z"][0]), t_end_base=lug["t_end_base_m"])


def s_map(s_base, lugs_z, L, span):
    """Piecewise-linear map of base arclength -> arclength at height z: lug faces slide to s1(z), s2(z);
    the lug windows [0, span] and [L - span, L] absorb the stretch, everything else is identity."""
    xp, fp = [0.0], [0.0]
    for lug in sorted(lugs_z, key=lambda l: l["s1_base"]):
        if lug["s2_base"] < span:
            xp += [lug["s1_base"], lug["s2_base"], span]
            fp += [lug["s1"], lug["s2"], span]
        else:
            xp += [L - span, lug["s1_base"], lug["s2_base"]]
            fp += [L - span, lug["s1"], lug["s2"]]
    xp += [L]
    fp += [L]
    xp, fp = np.array(xp), np.array(fp)
    if not (np.all(np.diff(xp) > 0) and np.all(np.diff(fp) > 0)):
        raise RuntimeError(f"non-monotonic lug s-map: xp={xp*1e3}, fp={fp*1e3} (mm)")
    return np.interp(s_base, xp, fp)


def lug_weight(s, lug, blend):
    """1 inside [s1, s2], linear to 0 over `blend` on both sides."""
    w = np.zeros_like(s)
    s1, s2 = lug["s1"], lug["s2"]
    w[(s >= s1) & (s <= s2)] = 1.0
    lo = (s < s1) & (s >= s1 - blend)
    w[lo] = (s[lo] - (s1 - blend)) / blend
    hi = (s > s2) & (s <= s2 + blend)
    w[hi] = (s2 + blend - s[hi]) / blend
    return w


def snap_boundary(pts, conn8, T, lo, hi, nraft, dz, frac=0.45, max_iter=4):
    """Move the lateral boundary nodes of the hex mesh onto the true TET4 surface of their own
    z-level (in-plane only), clipping each move to `frac` of the smallest incident in-plane edge
    and reverting nodes whose cells would fold. Returns (pts, stats)."""
    qf = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    F = np.concatenate([np.sort(conn8[:, list(t)], axis=1) for t in qf], axis=0)
    Fraw = np.concatenate([conn8[:, list(t)] for t in qf], axis=0)
    _, inv, c = np.unique(F, axis=0, return_inverse=True, return_counts=True)
    bnd = Fraw[c[inv] == 1]
    Qb = pts[bnd]
    nrm = np.cross(Qb[:, 1] - Qb[:, 0], Qb[:, 2] - Qb[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    nodes = np.unique(bnd[np.abs(nrm[:, 2]) < 0.5])
    hloc = np.full(len(pts), np.inf)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)):
        L = np.linalg.norm(pts[conn8[:, a], :2] - pts[conn8[:, b], :2], axis=1)
        np.minimum.at(hloc, conn8[:, a], L)
        np.minimum.at(hloc, conn8[:, b], L)
    K = np.round(pts[:, 2] / dz).astype(int)
    target = np.zeros((len(pts), 2))
    has = np.zeros(len(pts), bool)
    off = np.array([lo[1], lo[2]])
    for k in np.unique(K[nodes]):
        sel = nodes[K[nodes] == k]
        zpart = max((k - nraft) * dz, 0.0)
        x0 = lo[0] + min(max(zpart, 1e-5), (hi[0] - lo[0]) - 1e-5)
        loops = slice_loops(T, x0)
        if not loops:
            continue
        Rn = np.vstack([resample_closed(Q_, 1e-4)[0] for Q_ in loops]) - off
        _, j = cKDTree(Rn).query(pts[sel, :2])
        target[sel] = Rn[j]
        has[sel] = True
    orig = pts.copy()
    dall = target - orig[:, :2]
    Lall = np.linalg.norm(dall, axis=1)
    # a node is a true surface node of its level only if the surface is within reach; ledge / terrace
    # nodes whose nearest surface point is far away (feature ends, holes) are left where they are
    reach = has & (Lall <= frac * hloc)
    fixed = np.zeros(len(pts), bool)
    stats = {"lateral_boundary_nodes": int(has.sum()), "requested_move_median_m": float(np.median(Lall[has])),
             "requested_move_p95_m": float(np.percentile(Lall[has], 95)), "out_of_reach_nodes": int((has & ~reach).sum())}

    def corner_angles(p):
        q = p[conn8][:, :4, :2]
        am = np.full(len(q), 180.0)
        for a in range(4):
            u = q[:, (a + 1) % 4] - q[:, a]
            v = q[:, (a - 1) % 4] - q[:, a]
            cs = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
            am = np.minimum(am, np.degrees(np.arccos(np.clip(cs, -1, 1))))
        return am

    ang0 = corner_angles(orig)
    for it in range(max_iter):
        pts = orig.copy()
        mv = reach & ~fixed
        pts[mv, :2] = target[mv]
        Jc, _ = hex_metrics(pts[conn8])
        ang = corner_angles(pts)
        bad = (Jc <= 0) | ((ang < 25.0) & (ang < ang0 - 5.0))
        if not bad.any():
            break
        fixed[np.unique(conn8[bad])] = True
    resid = np.linalg.norm(target[reach] - pts[reach, :2], axis=1)
    stats.update({"snapped_nodes": int((reach & ~fixed).sum()), "reverted_nodes": int((reach & fixed).sum()), "iterations": it + 1,
                  "residual_to_surface_median_m": float(np.median(resid)), "residual_to_surface_p95_m": float(np.percentile(resid, 95)),
                  "residual_to_surface_max_m": float(resid.max())})
    return pts, stats


def chart_points(ref, s_mapped, t_base, lugs_z, blend, datum, core):
    """Physical points of chart coordinates (mapped s, base t): sheared direction inside the lugs and
    t scaled beyond the core so that the lug end stays a grid surface."""
    C, n, _ = curve_at(ref, s_mapped)
    d = n.copy()
    t = t_base.copy()
    for lug in lugs_z:
        w = lug_weight(s_mapped, lug, blend)
        s1, s2, d1, d2 = lug["s1"], lug["s2"], lug["d1"], lug["d2"]
        f = np.clip((s_mapped - s1) / max(s2 - s1, 1e-9), 0, 1)[:, None]
        d_l = (1 - f) * d1[None, :] + f * d2[None, :]
        d = d * (1 - w)[:, None] + d_l * w[:, None]
        if lug["t_end"] is not None and lug["t_end_base"] is not None and lug["t_end_base"] > core + 2e-3:
            fac = max(lug["t_end"] - core, 1e-3) / (lug["t_end_base"] - core)
            side = lug["side"]
            u = side * t_base - core                      # distance beyond the core on the lug side (spine-relative, like t_end)
            m = u > 0
            scale = 1 + w * (fac - 1)
            t[m] = side * (core + u[m] * scale[m])
    d = d / np.linalg.norm(d, axis=1)[:, None]
    return C + t[:, None] * d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dz", type=float, default=2e-4, help="cell height along the build axis [m]")
    ap.add_argument("--ht", type=float, default=0.2e-3, help="fine cell size through the sheet thickness [m] (0.8 mm sheet / n_inner)")
    ap.add_argument("--n-inner", type=int, default=4, help="fine cells across the sheet (+t side of the datum = the sheet)")
    ap.add_argument("--n-outer", type=int, default=2, help="fine cells on the -t side of the datum (foot bulge, lug roots)")
    ap.add_argument("--hs", type=float, default=2.0e-3, help="along-wall cell length on straights [m]")
    ap.add_argument("--hmin", type=float, default=8e-4, help="along-wall cell length floor at bends [m]")
    ap.add_argument("--bend-factor", type=float, default=0.35, help="along-wall length = factor x bend radius (capped by hs)")
    ap.add_argument("--raft", type=float, default=2e-3, help="raft height below the part [m] (0 = none)")
    ap.add_argument("--ref-z-mm", type=float, default=30.0, help="build height [mm] of the station whose section defines the spine")
    ap.add_argument("--ref-tmax", type=float, default=3e-3, help="ray length cap when finding the sheet mid-curve at the reference station [m]")
    ap.add_argument("--extend", type=float, default=32e-3, help="straight extrapolation of the spine beyond both leg tips [m]")
    ap.add_argument("--t-margin", type=float, default=30e-3, help="half-width of the t band searched around the mid-curve [m]")
    ap.add_argument("--t-core", type=float, default=4e-3, help="|t - datum| resolved with ht everywhere (sheet + foot) [m]")
    ap.add_argument("--t-growth", type=float, default=1.4, help="geometric growth of t cells beyond the fine band (foot, lugs)")
    ap.add_argument("--t-max", type=float, default=1.0e-3, help="largest t cell beyond the fine band [m]")
    ap.add_argument("--lug-window", type=float, default=10e-3, help="distance from each leg tip (towards the bend) where the full t band is allowed [m]")
    ap.add_argument("--lug-hs", type=float, default=1.5e-3, help="along-wall cell length inside the lug windows [m]")
    ap.add_argument("--lug-blend", type=float, default=10e-3, help="s-distance over which the chart turns from the normal to the lug face [m]")
    ap.add_argument("--lug-detect-every", type=int, default=5, help="detect lug faces every N stations for the z-tracks")
    ap.add_argument("--no-lug-shear", action="store_true", help="disable the lug alignment (plain normal chart)")
    ap.add_argument("--no-snap", action="store_true", help="do not project the lateral boundary nodes onto the true surface")
    ap.add_argument("--snap-frac", type=float, default=0.45, help="max node move as a fraction of the smallest incident in-plane edge")
    ap.add_argument("--sub", type=int, default=3, help="sub-samples per cell side for the inside test (sub^2 points)")
    ap.add_argument("--keep-fraction", type=float, default=0.5, help="keep a cell if >= this fraction of sub-samples is inside")
    ap.add_argument("--tag", default="0119_hex_raft")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"inp": str(args.inp), "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}

    nodes, elems, _, _ = parse(args.inp)
    ids = np.array(sorted(nodes))
    P = np.array([nodes[i] for i in ids])
    idx = {int(n): k for k, n in enumerate(ids)}
    conn = None
    for (etype, _), rows in elems.items():
        if etype.startswith("C3D4"):
            conn = np.vectorize(idx.get)(np.array([r[:5] for r in rows])[:, 1:])
    T = boundary_triangles(P, conn)
    tet_vol = float(np.abs(tet_metrics(P[conn])[0]).sum())
    lo, hi = P.min(0), P.max(0)
    rep["tet"] = {"cells": int(len(conn)), "volume_m3": tet_vol, "old_bbox_min": lo.tolist(), "old_bbox_max": hi.tolist()}
    Nz = int(np.ceil((hi[0] - lo[0]) / args.dz - 1e-9))
    stations = lo[0] + (np.arange(Nz) + 0.5) * args.dz
    z_c = (np.arange(Nz) + 0.5) * args.dz

    # ---- all station sections
    sections = [slice_loops(T, x0) for x0 in stations]
    areas = np.array([sum(polygon_area(q) for q in lps) for lps in sections])
    rep["sections"] = {"stations": Nz, "loops_per_station_max": int(max(len(l) for l in sections)),
                       "stations_with_holes": int(sum(len(l) > 1 for l in sections)),
                       "empty_stations": int(sum(len(l) == 0 for l in sections)),
                       "area_mm2_first_max_min_last": [float(areas[0] * 1e6), float(areas.max() * 1e6), float(areas[areas > 0].min() * 1e6), float(areas[-1] * 1e6)],
                       "area_integral_m3": float(areas.sum() * args.dz), "area_integral_over_tet_volume": float(areas.sum() * args.dz / tet_vol)}

    # ---- reference spine from a thin, uniform station, extended straight at both tips
    k_ref = int(np.clip(round(args.ref_z_mm * 1e-3 / args.dz - 0.5), 0, Nz - 1))
    Q0 = max(sections[k_ref], key=len)
    M, th, med, Lloop, run_len = medial_from_loop(Q0, tmax=args.ref_tmax)
    lug_span = args.extend + args.lug_window
    ref = build_reference(M, args.hs, args.hmin, args.bend_factor, extend=args.extend, lug_span=lug_span, lug_h=args.lug_hs)
    s_nodes = ref["s"]
    rep["reference_station"] = {"slab": k_ref, "z_mm": float((k_ref + 0.5) * args.dz * 1e3)}
    rep["sheet"] = {"ref_thickness_median_m": med, "ref_thickness_min_max_m": [float(th.min()), float(th.max())],
                    "spine_length_incl_extensions_m": ref["L"], "ref_loop_length_m": float(Lloop), "sheet_run_length_m": float(run_len),
                    "min_bend_radius_m": float(1.0 / max(ref["kappa"].max(), 1e-9))}
    print(f"spine from slab {k_ref} (z {(k_ref+0.5)*args.dz*1e3:.1f} mm): thickness {med*1e3:.2f} mm, spine {ref['L']*1e3:.1f} mm incl. 2x{args.extend*1e3:.0f} mm extensions, "
          f"min bend radius {rep['sheet']['min_bend_radius_m']*1e3:.2f} mm")

    # ---- lugs: detect the two long faces on many stations, track their position / length vs height
    lugs = []
    if not args.no_lug_shear:
        for name, s_lo, s_hi in (("A", 0.0, lug_span), ("B", ref["L"] - lug_span, ref["L"])):
            dets = []
            for kk in range(0, Nz, args.lug_detect_every):
                if sections[kk]:
                    dets.append((z_c[kk], detect_lug(max(sections[kk], key=len), ref, s_lo, s_hi)))
            lug = fit_lug(dets)
            if lug is None:
                print(f"lug {name}: not detected")
                continue
            lug["name"] = name
            _, nA, _ = curve_at(ref, np.array([0.5 * (lug["a1"] + lug["a2"])]))
            lug["face_angles_to_normal_deg"] = [float(np.degrees(np.arccos(np.clip(abs(np.dot(lug[k], nA[0])), -1, 1)))) for k in ("d1", "d2")]
            lugs.append(lug)
            te = lug["t_end_min_max_m"]
            msg = (f"lug {name}: {lug['n_detections']} detections z {lug['z_min']*1e3:.1f}..{lug['z_max']*1e3:.1f} mm; faces s1 = {lug['a1']*1e3:.1f} + {lug['b1']:.3f} z, "
                   f"s2 = {lug['a2']*1e3:.1f} + {lug['b2']:.3f} z (mm); linear-fit residual rms {lug['fit_residual_rms_m']*1e3:.2f} max {lug['fit_residual_max_m']*1e3:.2f} mm, "
                   f"track residual max {lug['smooth_residual_max_m']*1e3:.2f} mm; face angles to normal {lug['face_angles_to_normal_deg'][0]:.1f}/{lug['face_angles_to_normal_deg'][1]:.1f} deg, "
                   f"direction scatter max {lug['direction_deviation_max_deg']:.1f} deg")
            if te:
                msg += f"; lug length {te[1]*1e3:.1f} -> {te[0]*1e3:.1f} mm"
            print(msg)
        s_nodes = insert_nodes(s_nodes, [v for lug in lugs for v in (lug["s1z"][0], lug["s2z"][0])], 0.35 * args.lug_hs)
    rep["lugs"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in lug.items()} for lug in lugs]
    Ns = len(s_nodes) - 1
    s_c = 0.5 * (s_nodes[1:] + s_nodes[:-1])
    rep["sheet"].update({"s_cells": Ns, "s_cell_len_min_med_max_m": [float(np.diff(s_nodes).min()), float(np.median(np.diff(s_nodes))), float(np.diff(s_nodes).max())]})

    # ---- t band: signed offsets of all surface samples from the reference curve; datum = most common offset
    tree = cKDTree(ref["dense_M"])
    toff = []
    for lps in sections[::5]:
        for Q in lps:
            R, _ = resample_closed(Q, 3e-4)
            dist, j = tree.query(R)
            sgn = np.sign(np.einsum("ij,ij->i", R - ref["dense_M"][j], ref["dense_n"][j]))
            tt = sgn * dist
            toff.append(tt[dist < args.t_margin])
    toff = np.concatenate(toff)
    hbins = np.arange(-args.t_margin, args.t_margin + 1e-9, 5e-5)
    hist, edges = np.histogram(toff, bins=hbins)
    t_peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    t_datum = float(np.mean(toff[np.abs(toff - t_peak) < 5e-5]))
    t_lo, t_hi = float(np.percentile(toff, 0.1)), float(np.percentile(toff, 99.9))
    t_nodes, ncore = graded_t_nodes(t_datum, args.ht, args.n_inner, args.n_outer, t_lo, t_hi, args.t_growth, args.t_max)
    # the lug end faces (spine-relative offsets side * t_end_base) become grid nodes
    for lug in lugs:
        if lug["t_end_base_m"] is not None and lug["t_end_base_m"] > args.t_core + 2e-3:
            tgt = lug["side"] * lug["t_end_base_m"]
            gap = 0.35 * np.diff(t_nodes)[np.clip(np.searchsorted(t_nodes, tgt) - 1, 0, len(t_nodes) - 2)]
            t_nodes = insert_nodes(t_nodes, [tgt], gap)
    Nt = len(t_nodes) - 1
    t_c = 0.5 * (t_nodes[1:] + t_nodes[:-1])
    rep["t_band"] = {"datum_offset_m": t_datum, "offset_p0.1_p99.9_m": [t_lo, t_hi], "t_cells": Nt, "fine_cells_inner_outer": list(ncore),
                     "t_nodes_mm": (t_nodes * 1e3).round(3).tolist(), "datum_share_of_surface_samples": float(hist.max() / len(toff))}
    print(f"t band: datum {t_datum*1e3:+.3f} mm (share {hist.max()/len(toff):.2f}), offsets {t_lo*1e3:+.2f}..{t_hi*1e3:+.2f} mm, {Nt} t-cells "
          f"({ncore[0]} fine sheet cells + {ncore[1]} fine outer cells of {args.ht*1e3:.3f} mm, then graded to {args.t_max*1e3:.2f} mm), {Ns} s-cells")

    # ---- height-dependent chart
    blend = args.lug_blend

    def chart(s_base, t_base, z):
        lz = [lug_at(l, z) for l in lugs]
        sm = s_map(s_base, lz, ref["L"], lug_span) if lz else s_base
        return chart_points(ref, sm, t_base, lz, blend, t_datum, args.t_core)

    fr = (np.arange(args.sub) + 0.5) / args.sub
    ss = (s_nodes[:-1][:, None] + np.diff(s_nodes)[:, None] * fr[None, :])          # (Ns, sub)
    tt = (t_nodes[:-1][:, None] + np.diff(t_nodes)[:, None] * fr[None, :])          # (Nt, sub)
    S = np.broadcast_to(np.repeat(ss[:, None, :, None], Nt, axis=1), (Ns, Nt, args.sub, args.sub)).ravel()
    Tt = np.broadcast_to(np.repeat(tt[None, :, None, :], Ns, axis=0), (Ns, Nt, args.sub, args.sub)).ravel()
    nsub = args.sub * args.sub
    s_tipA, s_tipB = args.extend, ref["L"] - args.extend
    in_window = (s_c <= s_tipA + args.lug_window) | (s_c >= s_tipB - args.lug_window)
    allowed = np.ones((Ns, Nt), bool)
    allowed[~in_window] = (np.abs(t_c - t_datum) <= args.t_core + 0.5 * args.ht)[None, :]
    rep["t_band"]["lug_window_s_cells"] = int(in_window.sum())
    mask3 = np.zeros((Ns, Nt, Nz), bool)
    frac_area = np.zeros(Nz)
    cell_area = np.outer(np.diff(s_nodes), np.diff(t_nodes))
    for k, lps in enumerate(sections):
        if not lps:
            continue
        sub_pts = chart(S, Tt, z_c[k])
        inside = inside_by_nearest(lps, sub_pts).reshape(Ns, Nt, nsub).mean(axis=2)
        mask3[:, :, k] = (inside >= args.keep_fraction - 1e-12) & allowed
        frac_area[k] = (mask3[:, :, k] * cell_area).sum() / areas[k] if areas[k] > 0 else 0.0
    rep["fit"] = {"hex_section_area_over_polygon_area_min_mean_max": [float(frac_area[areas > 0].min()), float(frac_area[areas > 0].mean()), float(frac_area[areas > 0].max())],
                  "note": "proxy: base cell areas (lug windows are stretched per height)"}
    print(f"fit: hex/polygon section area (base-cell proxy) min {frac_area[areas>0].min():.3f} mean {frac_area[areas>0].mean():.3f} max {frac_area[areas>0].max():.3f}")

    # ---- grid -> nodes / cells (raft = base-station section and base chart)
    nraft = int(round(args.raft / args.dz)) if args.raft > 0 else 0
    Kz = Nz + nraft
    full = np.zeros((Ns, Nt, Kz), bool)
    full[:, :, nraft:] = mask3
    if nraft:
        full[:, :, :nraft] = mask3[:, :, :1]
    XY = np.empty((Ns + 1, Nt + 1, Kz + 1, 2))
    Sg = np.repeat(s_nodes[:, None], Nt + 1, axis=1).ravel()
    Tg = np.repeat(t_nodes[None, :], Ns + 1, axis=0).ravel()
    for k in range(Kz + 1):
        zk = max((k - nraft) * args.dz, 0.0)
        XY[:, :, k, :] = chart(Sg, Tg, zk).reshape(Ns + 1, Nt + 1, 2) - np.array([lo[1], lo[2]])
    Z = np.arange(Kz + 1) * args.dz
    ci, cj, ck = np.nonzero(full)
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    lin = np.arange((Ns + 1) * (Nt + 1) * (Kz + 1)).reshape(Ns + 1, Nt + 1, Kz + 1)
    node_used = np.zeros(lin.shape, bool)
    conn8 = np.empty((len(ci), 8), np.int64)
    for c, (di, dj, dk) in enumerate(corners):
        node_used[ci + di, cj + dj, ck + dk] = True
        conn8[:, c] = lin[ci + di, cj + dj, ck + dk]
    used_lin = np.nonzero(node_used.ravel())[0]
    new_id = -np.ones(lin.size, np.int64)
    new_id[used_lin] = np.arange(len(used_lin))
    conn8 = new_id[conn8]
    I, J, K = np.unravel_index(used_lin, lin.shape)
    pts = np.column_stack([XY[I, J, K, 0], XY[I, J, K, 1], Z[K]])
    if not args.no_snap:
        pts, snap = snap_boundary(pts, conn8, T, lo, hi, nraft, args.dz, frac=args.snap_frac)
        rep["snap"] = snap
        print(f"snap: {snap['lateral_boundary_nodes']} lateral boundary nodes, requested move median {snap['requested_move_median_m']*1e3:.3f} p95 {snap['requested_move_p95_m']*1e3:.3f} mm, "
              f"out of reach {snap['out_of_reach_nodes']}, snapped {snap['snapped_nodes']}, reverted {snap['reverted_nodes']}, "
              f"residual to surface median {snap['residual_to_surface_median_m']*1e3:.3f} p95 {snap['residual_to_surface_p95_m']*1e3:.3f} max {snap['residual_to_surface_max_m']*1e3:.2f} mm")
    region = np.where(ck < nraft, 0, 1)
    slab = np.where(ck < nraft, -1, ck - nraft)
    Jc, Lh = hex_metrics(pts[conn8])
    part_vol = float((Jc[region == 1] * 8).sum())
    dup_pairs = cKDTree(pts).query_pairs(1e-6)
    k8 = int(round(8e-3 / args.dz))
    q = pts[conn8][:, :4, :2]
    ang_min = np.full(len(q), 180.0)
    for a in range(4):
        u = q[:, (a + 1) % 4] - q[:, a]
        v = q[:, (a - 1) % 4] - q[:, a]
        cosang = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        ang_min = np.minimum(ang_min, np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    rep["checks"] = {"coincident_node_pairs": int(len(dup_pairs)),
                     "sheet_volume_above_8mm_over_tet": float((Jc[(region == 1) & (slab >= k8)] * 8).sum() / (areas[k8:].sum() * args.dz)),
                     "section_corner_angle_min_p01_deg": [float(ang_min.min()), float(np.percentile(ang_min, 1))],
                     "cells_with_corner_angle_below_30deg": int((ang_min < 30).sum())}
    per_slab = mask3.sum(axis=(0, 1))
    rep["hex"] = {"nodes": int(len(pts)), "cells": int(len(conn8)), "part_cells": int((region == 1).sum()), "raft_cells": int((region == 0).sum()),
                  "centre_jacobian_nonpositive": int((Jc <= 0).sum()), "part_volume_m3": part_vol, "part_volume_over_tet": part_vol / tet_vol,
                  "edge_len_min_med_max_m": [float(Lh.min()), float(np.median(Lh)), float(Lh.max())],
                  "edge_ratio_max_over_min_med_max": [float(np.median(Lh.max(1) / Lh.min(1))), float((Lh.max(1) / Lh.min(1)).max())],
                  "new_bbox_min_m": pts.min(0).tolist(), "new_bbox_max_m": pts.max(0).tolist(),
                  "slabs": int(Nz), "raft_layers": nraft, "part_base_z_m": float(nraft * args.dz),
                  "mechanics_dof": int(3 * len(pts)), "thermal_dof": int(len(pts)),
                  "part_cells_per_slab_first_max_min_last": [int(per_slab[0]), int(per_slab.max()), int(per_slab[per_slab > 0].min()), int(per_slab[-1])]}
    print(f"hex: {len(pts)} nodes, {len(conn8)} cells ({(region==1).sum()} part + {(region==0).sum()} raft), J<=0: {(Jc<=0).sum()}, "
          f"part volume / tet volume = {part_vol/tet_vol:.4f} (above 8 mm: {rep['checks']['sheet_volume_above_8mm_over_tet']:.4f}), "
          f"coincident node pairs {len(dup_pairs)}, min corner angle {ang_min.min():.1f} deg ({(ang_min < 30).sum()} cells < 30 deg), "
          f"cells/slab first {per_slab[0]} max {per_slab.max()} last {per_slab[-1]}, mechanics DOF {3*len(pts)}")

    # ---- inp
    inp_path = args.out_dir / f"{args.tag}.inp"
    with inp_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("*HEADING\n0119 bent-sheet part: body-fitted HEX8 (block-structured curvilinear voxels) + raft, build axis +Z, units m\n")
        f.write(f"** generated by make_0119_hex_mesh.py from {args.inp.name}; dz={args.dz} ht={args.ht:.6g} hs={args.hs} lug_hs={args.lug_hs} t_core={args.t_core} raft={args.raft}\n")
        f.write("*NODE\n")
        for n, (x, y, z) in enumerate(pts, 1):
            f.write(f"{n}, {x:.9e}, {y:.9e}, {z:.9e}\n")
        for name, sel in (("PART", region == 1), ("RAFT", region == 0)):
            if not sel.any():
                continue
            f.write(f"*ELEMENT, TYPE=C3D8, ELSET={name}\n")
            for e, row in zip(np.nonzero(sel)[0] + 1, conn8[sel] + 1):
                f.write(f"{e}, " + ", ".join(str(v) for v in row) + "\n")
        base = np.nonzero(pts[:, 2] < 0.5 * args.dz)[0] + 1
        f.write("*NSET, NSET=BASE\n")
        for i in range(0, len(base), 16):
            f.write(", ".join(str(v) for v in base[i:i + 16]) + "\n")
    # ---- vtu
    try:
        import meshio
        meshio.Mesh(pts, [("hexahedron", conn8)],
                    cell_data={"region": [region.astype(np.int32)], "slab": [slab.astype(np.int32)],
                               "s_cell": [ci.astype(np.int32)], "t_cell": [cj.astype(np.int32)],
                               "jacobian": [Jc.astype(np.float32)], "min_corner_angle_deg": [ang_min.astype(np.float32)]}).write(args.out_dir / f"{args.tag}.vtu")
        rep["vtu"] = str(args.out_dir / f"{args.tag}.vtu")
    except Exception as exc:  # noqa: BLE001
        rep["vtu"] = f"skipped: {exc}"
    np.save(args.out_dir / f"{args.tag}_mask.npy", full)
    np.savez(args.out_dir / f"{args.tag}_grid.npz", s_nodes=s_nodes, t_nodes=t_nodes, areas=areas, stations=stations)
    (args.out_dir / f"{args.tag}_report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("wrote", inp_path, rep.get("vtu"))

    # ---- pictures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        qf = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        F = np.concatenate([np.sort(conn8[:, list(t)], axis=1) for t in qf], axis=0)
        Fraw = np.concatenate([conn8[:, list(t)] for t in qf], axis=0)
        Freg = np.tile(region, 6)
        Fu, inv, c = np.unique(F, axis=0, return_inverse=True, return_counts=True)
        bnd = np.nonzero(c[inv] == 1)[0]
        Q = pts[Fraw[bnd]] * 1e3
        Rq = Freg[bnd]

        def render(ax, view, up, title):
            v = np.array(view, float)
            v /= np.linalg.norm(v)
            u = np.cross(up, v)
            u /= np.linalg.norm(u)
            w = np.cross(v, u)
            ctr = Q.mean(axis=(0, 1))
            X = (Q - ctr) @ np.stack([u, w, v], axis=1)
            nrm = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
            shade = 0.35 + 0.65 * np.abs(nrm @ v)
            depth = X[:, :, 2].mean(1)
            order = np.argsort(depth)
            cols = np.where(Rq[order][:, None] == 0, np.array([[0.85, 0.2, 0.2]]), np.array([[0.55, 0.65, 0.85]])) * shade[order][:, None]
            ax.add_collection(PolyCollection(X[order][:, :, :2], facecolors=np.clip(cols, 0, 1), edgecolors="none"))
            ax.set_xlim(X[:, :, 0].min() - 2, X[:, :, 0].max() + 2)
            ax.set_ylim(X[:, :, 1].min() - 2, X[:, :, 1].max() + 2)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.axis("off")

        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        render(axes[0], (1, -1.2, 0.8), (0, 0, 1), "oblique from +X -Y +Z")
        render(axes[1], (-1, 1.2, 0.8), (0, 0, 1), "oblique from -X +Y +Z")
        render(axes[2], (0.3, 0.2, -1), (0, 1, 0), "from below (-Z)")
        fig.suptitle(f"{args.tag}: {len(conn8)} HEX8 ({(region==1).sum()} part + {(region==0).sum()} raft, red), {len(pts)} nodes, part volume/tet = {part_vol/tet_vol:.3f}")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{args.tag}_views.png", dpi=90)
        picks = [0, int(Nz * 0.06), int(Nz * 0.11), int(Nz * 0.33), int(Nz * 0.55), int(Nz * 0.68), int(Nz * 0.72), Nz - 1]
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 11))
        ext = (pts.max(0) - pts.min(0)) * 1e3
        for ax_, k in zip(axes2.ravel(), picks):
            sel = np.nonzero((region == 1) & (slab == k))[0]
            quads = pts[conn8[sel][:, :4]][:, :, :2] * 1e3
            ax_.add_collection(PolyCollection(quads, facecolors="#9ecae1", edgecolors="tab:blue", linewidths=0.3))
            for Qs in sections[k]:
                Qs2 = (Qs - np.array([lo[1], lo[2]])) * 1e3
                ax_.plot(np.r_[Qs2[:, 0], Qs2[0, 0]], np.r_[Qs2[:, 1], Qs2[0, 1]], "r-", lw=0.7)
            ax_.set_aspect("equal")
            ax_.set_xlim(0, ext[0])
            ax_.set_ylim(0, ext[1])
            ax_.set_title(f"slab {k} (Z = {(k+0.5)*args.dz*1e3:.1f} mm above base): {len(sel)} cells")
        fig2.suptitle("HEX8 section (blue) vs TET4 surface slice (red)")
        fig2.tight_layout()
        fig2.savefig(args.out_dir / f"{args.tag}_sections.png", dpi=110)
        fig3, ax3 = plt.subplots(1, 3, figsize=(24, 8))
        sel = np.nonzero((region == 1) & (slab == 0))[0]
        quads = pts[conn8[sel][:, :4]][:, :, :2] * 1e3
        Qs2 = (max(sections[0], key=len) - np.array([lo[1], lo[2]])) * 1e3
        for a_, (x0, x1, y0, y1) in zip(ax3, [(0, 30, 0, 30), (55, 90, 20, 55), (20, 55, 60, 98)]):
            a_.add_collection(PolyCollection(quads, facecolors="#9ecae1", edgecolors="tab:blue", linewidths=0.4))
            a_.plot(np.r_[Qs2[:, 0], Qs2[0, 0]], np.r_[Qs2[:, 1], Qs2[0, 1]], "r-", lw=1.0)
            a_.set_aspect("equal")
            a_.set_xlim(x0, x1)
            a_.set_ylim(y0, y1)
            a_.grid(alpha=0.3)
        fig3.suptitle("base slab zoom: bend (left), lower lug (middle), upper lug (right)")
        fig3.tight_layout()
        fig3.savefig(args.out_dir / f"{args.tag}_base_zoom.png", dpi=120)
        print("pictures written")
    except Exception as exc:  # noqa: BLE001
        print("pictures skipped:", exc)


if __name__ == "__main__":
    main()
