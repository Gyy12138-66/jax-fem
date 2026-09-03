#!/usr/bin/env python
"""Group-A diagnostics on the v159 dumped linear systems (V159_DUMP_* probe).

Per thermal dump (7 heights): pinned-row structure, asymmetry, lambda_max /
lambda_min of the free block (raw and Jacobi-scaled -> condition numbers),
Jacobi-BiCGSTAB iteration counts at rtol 1e-6 / 1e-10 (the production
pathology reproduced offline), and the slowest eigenmodes written to VTU for
ParaView.  Per mechanics dump (3 heights): structure, asymmetry, lambda_max,
approximate lambda_min, capped Jacobi-BiCGSTAB attempt.

Outputs (JSON, CSV, PNG, VTU) go to <dump-dir>/analysis by default.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(path):
    d = np.load(path)
    n = int(d["n"])
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(n, n))
    A.sum_duplicates()
    return A, np.asarray(d["b"], float), np.asarray(d["x0"], float), int(d["layer"]), int(d["step"]), str(d["mode"])


def pinned_rows(A):
    """Dirichlet rows in the dumps: diag == 1 with all stored off-diagonals == 0."""
    diag = A.diagonal()
    offabs = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
    return np.isclose(diag, 1.0) & (offabs == 0.0)


def sym(A):
    return ((A + A.T) * 0.5).tocsr()


def asym(A):
    return float(spla.norm(A - A.T) / spla.norm(A))


def lam_max(S):
    v = spla.eigsh(S, k=1, which="LA", tol=1e-6, maxiter=5000, return_eigenvectors=False)
    return float(v[0])


def amg_precond(S, max_coarse=500):
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(S, max_coarse=max_coarse)
    return ml.aspreconditioner(cycle="V"), ml


def lam_min_lobpcg(S, k, M, maxiter, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S.shape[0], k))
    vals, vecs, hist = spla.lobpcg(S, X, M=M, largest=False, tol=tol, maxiter=maxiter, retResidualNormsHistory=True)
    order = np.argsort(vals)
    last = np.max(hist[-1]) if len(hist) else np.nan
    return vals[order], vecs[:, order], len(hist), float(last)


def jacobi_operator(A):
    d = A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda x: x / d, dtype=float)


def bicgstab_run(A, b, x0, rtol, maxiter):
    count = [0]

    def cb(_xk):
        count[0] += 1

    t0 = time.time()
    x, info = spla.bicgstab(A, b, x0=x0, M=jacobi_operator(A), rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
    wall = time.time() - t0
    rel = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    return {"iters": count[0], "info": int(info), "final_rel_res": rel, "wall_s": wall}


def write_modes_vtu(mesh_vtu, out_path, fields):
    import meshio
    m = meshio.read(mesh_vtu)
    n = len(m.points)
    for k, v in fields.items():
        assert len(v) == n, f"{k}: {len(v)} != mesh points {n}"
    meshio.write(out_path, meshio.Mesh(points=m.points, cells=m.cells,
                                       point_data={k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}))


def split_blocks(A):
    pinned = pinned_rows(A)
    free = np.flatnonzero(~pinned)
    pin = np.flatnonzero(pinned)
    Aff = A[free][:, free].tocsr()
    Afp = A[free][:, pin].tocsr()
    return pinned, free, pin, Aff, Afp


def structure_record(scope, path, A, b, layer, step, mode, pinned, free, pin, Aff, Afp):
    return {
        "scope": scope, "layer": layer, "step": step, "mode": mode, "file": os.path.basename(path),
        "n": int(A.shape[0]), "nnz": int(A.nnz), "n_pinned": int(pinned.sum()), "n_free": int(len(free)),
        "free_to_pinned_coupling_nnz": int(Afp.nnz), "asym_full": asym(A), "asym_free": asym(Aff),
        "diag_free_min": float(Aff.diagonal().min()), "diag_free_max": float(Aff.diagonal().max()),
        "b_norm": float(np.linalg.norm(b)),
    }


def reduced_rhs(b, x0, free, pin, Afp):
    # Move the known pinned values to the right-hand side of the free block.
    return b[free] - Afp @ x0[pin] if Afp.nnz else b[free]


def analyse_thermal(path, args, mesh_vtu):
    A, b, x0, layer, step, mode = load(path)
    t_start = time.time()
    pinned, free, pin, Aff, Afp = split_blocks(A)
    rec = structure_record("thermal", path, A, b, layer, step, mode, pinned, free, pin, Aff, Afp)
    log(f"thermal L{layer}: free={len(free)} pinned={len(pin)} coupling_nnz={Afp.nnz} asym_free={rec['asym_free']:.2e}")

    S = sym(Aff)
    d = S.diagonal()
    Dm = sp.diags(1.0 / np.sqrt(d))
    Sj = (Dm @ S @ Dm).tocsr()

    rec["lambda_max_raw"] = lam_max(S)
    rec["lambda_max_jacobi"] = lam_max(Sj)
    log(f"  lambda_max raw={rec['lambda_max_raw']:.3e} jacobi={rec['lambda_max_jacobi']:.3e}")

    modes = {}
    for tag, Sx in (("raw", S), ("jacobi", Sj)):
        t0 = time.time()
        M, _ml = amg_precond(Sx)
        vals, vecs, its, res = lam_min_lobpcg(Sx, k=args.n_modes, M=M, maxiter=args.lobpcg_maxiter)
        rec[f"lambda_min_{tag}"] = float(vals[0])
        rec[f"lambda_min_{tag}_list"] = [float(v) for v in vals]
        rec[f"lobpcg_{tag}_iters"] = its
        rec[f"lobpcg_{tag}_final_res"] = res
        rec[f"kappa_{tag}"] = rec[f"lambda_max_{tag}"] / float(vals[0])
        log(f"  lambda_min {tag}={vals[0]:.3e} (lobpcg {its} it, res {res:.1e}, {time.time() - t0:.0f}s)"
            f" -> kappa_{tag}={rec[f'kappa_{tag}']:.3e}")
        modes[tag] = vecs

    bf = reduced_rhs(b, x0, free, pin, Afp)
    for rtol in args.rtols:
        r = bicgstab_run(Aff, bf, x0[free], rtol, args.bicgstab_maxiter)
        rec[f"bicgstab_jacobi_rtol{rtol:g}"] = r
        log(f"  Jacobi-BiCGSTAB rtol={rtol:g}: iters={r['iters']} info={r['info']}"
            f" rel_res={r['final_rel_res']:.1e} ({r['wall_s']:.0f}s)")

    if mesh_vtu and not args.no_vtu:
        fields = {"pinned": pinned.astype(float)}
        for tag, vecs in modes.items():
            for i in range(min(args.n_modes, vecs.shape[1])):
                full = np.zeros(A.shape[0])
                v = vecs[:, i]
                if tag == "jacobi":
                    v = v / np.sqrt(d)  # error mode of the Jacobi-preconditioned operator
                full[free] = v / np.max(np.abs(v))
                fields[f"mode{i}_{tag}"] = full
        out = os.path.join(args.out_dir, f"slow_modes_L{layer:03d}.vtu")
        write_modes_vtu(mesh_vtu, out, fields)
        rec["modes_vtu"] = out
        log(f"  wrote {out}")
    rec["analysis_wall_s"] = time.time() - t_start
    return rec


def analyse_mechanics(path, args):
    A, b, x0, layer, step, mode = load(path)
    t_start = time.time()
    pinned, free, pin, Aff, Afp = split_blocks(A)
    rec = structure_record("mechanics", path, A, b, layer, step, mode, pinned, free, pin, Aff, Afp)
    log(f"mechanics L{layer}: free={len(free)} pinned={len(pin)} asym_free={rec['asym_free']:.2e}"
        f" diag range {rec['diag_free_min']:.2e}..{rec['diag_free_max']:.2e}")
    S = sym(Aff)
    d = S.diagonal()
    Dm = sp.diags(1.0 / np.sqrt(d))
    Sj = (Dm @ S @ Dm).tocsr()
    rec["lambda_max_raw"] = lam_max(S)
    rec["lambda_max_jacobi"] = lam_max(Sj)
    log(f"  lambda_max raw={rec['lambda_max_raw']:.3e} jacobi={rec['lambda_max_jacobi']:.3e}")
    try:
        t0 = time.time()
        M, _ml = amg_precond(Sj, max_coarse=1000)
        vals, _vecs, its, res = lam_min_lobpcg(Sj, k=2, M=M, maxiter=args.lobpcg_maxiter_mech)
        rec["lambda_min_jacobi"] = float(vals[0])
        rec["lobpcg_jacobi_iters"] = its
        rec["lobpcg_jacobi_final_res"] = res
        rec["lambda_min_jacobi_converged"] = bool(res < 1e-4)
        rec["kappa_jacobi"] = rec["lambda_max_jacobi"] / float(vals[0])
        log(f"  lambda_min jacobi~{vals[0]:.3e} (lobpcg {its} it, res {res:.1e},"
            f" converged={rec['lambda_min_jacobi_converged']}, {time.time() - t0:.0f}s)"
            f" -> kappa_jacobi~{rec['kappa_jacobi']:.3e}")
    except Exception as exc:  # AMG on elasticity without near-nullspace may fail; record and move on
        rec["lambda_min_error"] = f"{type(exc).__name__}: {exc}"
        log(f"  lambda_min estimate failed: {rec['lambda_min_error']}")
    bf = reduced_rhs(b, x0, free, pin, Afp)
    r = bicgstab_run(Aff, bf, x0[free], 1e-6, args.bicgstab_maxiter_mech)
    rec["bicgstab_jacobi_rtol1e-06"] = r
    log(f"  Jacobi-BiCGSTAB rtol=1e-6 (cap {args.bicgstab_maxiter_mech}): iters={r['iters']} info={r['info']}"
        f" rel_res={r['final_rel_res']:.1e} ({r['wall_s']:.0f}s)")
    rec["analysis_wall_s"] = time.time() - t_start
    return rec


def plot(records, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = sorted((r for r in records if r["scope"] == "thermal"), key=lambda r: r["layer"])
    if not th:
        return
    L = [r["layer"] for r in th]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(L, [r["kappa_raw"] for r in th], "o-", label="kappa raw")
    ax[0].semilogy(L, [r["kappa_jacobi"] for r in th], "s-", label="kappa Jacobi-scaled")
    ax[0].set_xlabel("voxel layer")
    ax[0].set_ylabel("condition number (free block, sym part)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    for key, lab in (("bicgstab_jacobi_rtol1e-06", "rtol 1e-6"), ("bicgstab_jacobi_rtol1e-10", "rtol 1e-10")):
        if all(key in r for r in th):
            ax[1].plot(L, [r[key]["iters"] for r in th], "o-", label=f"Jacobi-BiCGSTAB {lab}")
    ax[1].set_xlabel("voxel layer")
    ax[1].set_ylabel("iterations")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.suptitle("v159 thermal systems: conditioning and Jacobi-BiCGSTAB cost vs build height")
    fig.tight_layout()
    out = os.path.join(out_dir, "A_thermal_kappa_iters.png")
    fig.savefig(out, dpi=130)
    log(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="/home/user/work/159/output/v159_voxel_fast2/matrix_dumps")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mesh-vtu", default="/home/user/work/159/output/v159_voxel_fast2/production/step_000000_scan.vtu")
    ap.add_argument("--layers", default=None, help="comma list to restrict thermal layers")
    ap.add_argument("--skip-mechanics", action="store_true")
    ap.add_argument("--no-vtu", action="store_true")
    ap.add_argument("--n-modes", type=int, default=3)
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-6, 1e-10])
    ap.add_argument("--bicgstab-maxiter", type=int, default=10000)
    ap.add_argument("--bicgstab-maxiter-mech", type=int, default=2000)
    ap.add_argument("--lobpcg-maxiter", type=int, default=300)
    ap.add_argument("--lobpcg-maxiter-mech", type=int, default=100)
    args = ap.parse_args()
    args.out_dir = args.out_dir or os.path.join(args.dump_dir, "analysis")
    os.makedirs(args.out_dir, exist_ok=True)
    want = {int(v) for v in args.layers.split(",")} if args.layers else None
    results_path = os.path.join(args.out_dir, "A_results.json")

    records = []
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "thermal_L*.npz"))):
        layer = int(os.path.basename(path).split("_L")[1][:3])
        if want and layer not in want:
            continue
        records.append(analyse_thermal(path, args, args.mesh_vtu))
        json.dump(records, open(results_path, "w"), indent=2)
    if not args.skip_mechanics:
        for path in sorted(glob.glob(os.path.join(args.dump_dir, "mechanics_L*.npz"))):
            records.append(analyse_mechanics(path, args))
            json.dump(records, open(results_path, "w"), indent=2)

    keys = ["scope", "layer", "n_free", "n_pinned", "free_to_pinned_coupling_nnz", "asym_free",
            "lambda_max_raw", "lambda_min_raw", "kappa_raw", "lambda_max_jacobi", "lambda_min_jacobi", "kappa_jacobi"]
    with open(os.path.join(args.out_dir, "A_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["bicgstab_iters_rtol1e-6", "bicgstab_iters_rtol1e-10", "bicgstab_info_rtol1e-10"])
        for r in records:
            w.writerow([r.get(k, "") for k in keys]
                       + [r.get("bicgstab_jacobi_rtol1e-06", {}).get("iters", ""),
                          r.get("bicgstab_jacobi_rtol1e-10", {}).get("iters", ""),
                          r.get("bicgstab_jacobi_rtol1e-10", {}).get("info", "")])
    plot(records, args.out_dir)
    log("A diagnostics done")


if __name__ == "__main__":
    main()
