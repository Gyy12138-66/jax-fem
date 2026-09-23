#!/usr/bin/env python3
"""Abaqus inp mesh health report: types, sizes, quality, layering, conformity, components."""
import sys, json
import numpy as np
from collections import defaultdict


def parse(path):
    nodes = {}
    elems = defaultdict(list)   # (type, elset) -> [[id, n1, n2, ...]]
    nsets = {}
    elsets = {}
    mode = None
    cur = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                parts = s.split(",")
                kw = parts[0].strip().upper()
                opts = {}
                for tok in parts[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        opts[k.strip().upper()] = v.strip()
                    else:
                        opts[tok.strip().upper()] = True
                if kw == "*NODE":
                    mode = "node"
                elif kw == "*ELEMENT":
                    mode = "elem"
                    cur = (opts.get("TYPE", "?").upper(), opts.get("ELSET", ""))
                elif kw in ("*NSET", "*ELSET"):
                    mode = kw[1:].lower()
                    name = opts.get(kw[1:], "?")
                    cur = (name, "GENERATE" in opts)
                    (nsets if mode == "nset" else elsets).setdefault(name, [])
                else:
                    mode = None
                continue
            toks = s.replace(",", " ").split()
            if mode == "node":
                nodes[int(toks[0])] = (float(toks[1]), float(toks[2]), float(toks[3]) if len(toks) > 3 else 0.0)
            elif mode == "elem":
                elems[cur].append([int(t) for t in toks])
            elif mode in ("nset", "elset"):
                p = [int(t) for t in toks]
                target = nsets if mode == "nset" else elsets
                if cur[1] and len(p) >= 2:
                    target[cur[0]].extend(range(p[0], p[1] + 1, p[2] if len(p) > 2 else 1))
                else:
                    target[cur[0]].extend(p)
    return nodes, elems, nsets, elsets


def tet_metrics(X):
    a, b, c, d = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    edges = np.stack([b - a, c - a, d - a, c - b, d - b, d - c], axis=1)
    L = np.linalg.norm(edges, axis=2)
    faces = [(1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3)]
    hmin = np.full(len(X), np.inf)
    fn = []
    for i, j, k, o in faces:
        n = np.cross(X[:, j] - X[:, i], X[:, k] - X[:, i])
        area = 0.5 * np.linalg.norm(n, axis=1)
        h = 3 * np.abs(vol) / np.maximum(area, 1e-300)
        hmin = np.minimum(hmin, h)
        n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        sgn = np.sign(np.einsum("ij,ij->i", n, X[:, i] - X[:, o]))
        n = n * np.where(sgn == 0, 1, sgn)[:, None]
        fn.append(n)
    ar = L.max(axis=1) / np.maximum(hmin, 1e-300)
    dmin = np.full(len(X), 180.0)
    for p in range(4):
        for q in range(p + 1, 4):
            cosang = np.clip(-np.einsum("ij,ij->i", fn[p], fn[q]), -1, 1)
            dmin = np.minimum(dmin, np.degrees(np.arccos(cosang)))
    return vol, L, ar, dmin


def hex_metrics(X):
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    L = np.stack([np.linalg.norm(X[:, j] - X[:, i], axis=1) for i, j in e], axis=1)
    dxi = 0.125 * ((X[:, 1] + X[:, 2] + X[:, 5] + X[:, 6]) - (X[:, 0] + X[:, 3] + X[:, 4] + X[:, 7]))
    det = 0.125 * ((X[:, 2] + X[:, 3] + X[:, 6] + X[:, 7]) - (X[:, 0] + X[:, 1] + X[:, 4] + X[:, 5]))
    dze = 0.125 * ((X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7]) - (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]))
    J = np.einsum("ij,ij->i", np.cross(dxi, det), dze)
    return J, L


def faces_of(conn, etype):
    if etype.startswith("C3D4") or etype.startswith("C3D10"):
        c = conn[:, :4]
        f = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
        return [np.sort(c[:, list(t)], axis=1) for t in f]
    if etype.startswith("C3D8") or etype.startswith("C3D20"):
        c = conn[:, :8]
        f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return [np.sort(c[:, list(t)], axis=1) for t in f]
    return []


def main(path):
    nodes, elems, nsets, elsets = parse(path)
    ids = np.array(sorted(nodes))
    P = np.array([nodes[i] for i in ids])
    idx = {int(n): k for k, n in enumerate(ids)}
    out = {"file": path, "nodes": int(len(ids)), "node_id_range": [int(ids.min()), int(ids.max())]}
    lo, hi = P.min(axis=0), P.max(axis=0)
    out["bbox_min"] = lo.tolist()
    out["bbox_max"] = hi.tolist()
    out["extent"] = (hi - lo).tolist()
    q = np.round(P, 9)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    out["duplicate_coordinate_nodes"] = int((cnt - 1).sum())
    used = np.zeros(len(ids), bool)
    blocks = []
    all_faces = []
    conn_all = []
    for (etype, elset), rows in elems.items():
        width = min(len(r) for r in rows)
        arr = np.array([r[:width] for r in rows])
        eid = arr[:, 0]
        conn = np.vectorize(idx.get)(arr[:, 1:])
        used[conn.ravel()] = True
        blk = {"type": etype, "elset": elset, "count": int(len(arr)), "nodes_per_elem": int(width - 1),
               "elem_id_range": [int(eid.min()), int(eid.max())]}
        X = P[conn]
        if etype.startswith("C3D4") or etype.startswith("C3D10"):
            vol, L, ar, dmin = tet_metrics(X[:, :4])
            blk.update({
                "volume_total": float(np.abs(vol).sum()),
                "inverted_or_zero_volume": int((vol <= 0).sum()),
                "edge_len_min_med_max": [float(L.min()), float(np.median(L)), float(L.max())],
                "aspect_ratio_Lmax_over_hmin_med_p95_max": [float(np.median(ar)), float(np.percentile(ar, 95)), float(ar.max())],
                "min_dihedral_deg_min_p05_med": [float(dmin.min()), float(np.percentile(dmin, 5)), float(np.median(dmin))],
                "elements_min_dihedral_below_10deg": int((dmin < 10).sum()),
                "volume_min_med_max": [float(np.abs(vol).min()), float(np.median(np.abs(vol))), float(np.abs(vol).max())],
            })
            all_faces.append(np.concatenate(faces_of(conn, etype), axis=0))
            conn_all.append(conn[:, :4])
        elif etype.startswith("C3D8"):
            J, L = hex_metrics(X[:, :8])
            blk.update({
                "centre_jacobian_nonpositive": int((J <= 0).sum()),
                "volume_total_approx": float(np.abs(J).sum() * 8),
                "edge_len_min_med_max": [float(L.min()), float(np.median(L)), float(L.max())],
                "edge_ratio_max_over_min_med_max": [float(np.median(L.max(1) / L.min(1))), float((L.max(1) / L.min(1)).max())],
            })
            all_faces.append(np.concatenate(faces_of(conn, etype), axis=0))
            conn_all.append(conn[:, :8])
        else:
            L = np.linalg.norm(X[:, 1] - X[:, 0], axis=1) if width - 1 >= 2 else np.zeros(1)
            blk.update({"note": "non-solid element block (ignored for solid checks)",
                        "length_min_med_max": [float(L.min()), float(np.median(L)), float(L.max())]})
            conn_all.append(conn)
        zc = X[:, :, 2].mean(axis=1)
        blk["unique_centroid_z_levels"] = int(len(np.unique(np.round(zc, 7))))
        blocks.append(blk)
    out["element_blocks"] = blocks
    out["unused_nodes"] = int((~used).sum())
    out["nsets"] = {k: len(v) for k, v in nsets.items()}
    out["elsets"] = {k: len(v) for k, v in elsets.items()}
    uz = np.unique(np.round(P[:, 2], 7))
    dz = np.diff(uz)
    out["node_z_levels"] = int(len(uz))
    if len(dz):
        out["node_z_spacing_min_med_max"] = [float(dz.min()), float(np.median(dz)), float(dz.max())]
    if all_faces and len(set(f.shape[1] for f in all_faces)) == 1:
        F = np.concatenate(all_faces, axis=0)
        Fu, c = np.unique(F, axis=0, return_counts=True)
        out["faces"] = {"boundary(1)": int((c == 1).sum()), "interior(2)": int((c == 2).sum()), "nonmanifold(>2)": int((c > 2).sum())}
        Fb = Fu[c == 1]
        k = Fb.shape[1]
        E = np.concatenate([np.sort(Fb[:, [i, (i + 1) % k]], axis=1) for i in range(k)], axis=0)
        _, ec = np.unique(E, axis=0, return_counts=True)
        out["boundary_surface"] = {"edges_in_1_face(open)": int((ec == 1).sum()), "edges_in_2_faces": int((ec == 2).sum()), "edges_in_>2_faces": int((ec > 2).sum())}
    elif all_faces:
        out["faces"] = "mixed element families; conformity across families not checked"
    parent = np.arange(len(ids))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for conn in conn_all:
        for row in conn:
            r0 = find(row[0])
            for n in row[1:]:
                r = find(n)
                if r != r0:
                    parent[r] = r0
    roots = np.array([find(i) for i in range(len(ids))])[used]
    _, cc = np.unique(roots, return_counts=True)
    out["connected_components"] = {"count": int(len(cc)), "node_counts_sorted": sorted(cc.tolist(), reverse=True)[:8]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
