#!/usr/bin/env python
"""Group-B preconditioner benchmark on the dumped v159 mechanics systems.

For each mechanics dump (layers 30 / 90 / 150 by default) the free block
(Dirichlet rows removed, pinned values moved to the right-hand side) is solved
with a ladder of candidates, all on CPU:

  pardiso_direct        MKL PARDISO factorise + solve (the production path; time to beat)
  jacobi_cg             Jacobi-preconditioned CG (free block is symmetric)
  jacobi_bicgstab       Jacobi-preconditioned BiCGSTAB (group-A baseline, same cap)
  blockjacobi_cg        3x3 node-block Jacobi + CG
  sa_amg_default_cg     pyamg smoothed aggregation, constant near-nullspace, + CG
  sa_amg_rbm_cg         pyamg smoothed aggregation with 6 rigid-body modes + CG
  sa_amg_rbm_scaled_cg  same on the Jacobi-scaled matrix D^-1/2 A D^-1/2
  rs_amg_cg             pyamg Ruge-Stueben classical AMG + CG

Per candidate: setup time, Krylov iterations, solve time, true relative
residual, AMG operator complexity. Outputs JSON/CSV/PNG under
<dump-dir>/analysis. Node coordinates for the rigid-body modes come from a
production VTU (mesh point order == matrix node order, dof = 3*node + comp).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import traceback

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
    return A, np.asarray(d["b"], float), np.asarray(d["x0"], float), int(d["layer"]), int(d["step"])


def pinned_rows(A):
    diag = A.diagonal()
    offabs = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
    return np.isclose(diag, 1.0) & (offabs == 0.0)


def split_blocks(A):
    pinned = pinned_rows(A)
    free = np.flatnonzero(~pinned)
    pin = np.flatnonzero(pinned)
    Aff = A[free][:, free].tocsr()
    Afp = A[free][:, pin].tocsr()
    return pinned, free, pin, Aff, Afp


def rigid_body_modes(coords):
    """6 rigid-body modes for node-major (x,y,z) interleaved dofs; coords (nn,3)."""
    c = coords - coords.mean(axis=0)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    nn = len(coords)
    B = np.zeros((3 * nn, 6))
    B[0::3, 0] = 1.0
    B[1::3, 1] = 1.0
    B[2::3, 2] = 1.0
    B[1::3, 3] = -z
    B[2::3, 3] = y
    B[0::3, 4] = z
    B[2::3, 4] = -x
    B[0::3, 5] = -y
    B[1::3, 5] = x
    return B


def jacobi_operator(A):
    d = A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda v: v / d, dtype=float)


def block_jacobi_operator(A):
    n = A.shape[0]
    nb = n // 3
    coo = A.tocoo()
    same = (coo.row // 3) == (coo.col // 3)
    blocks = np.zeros((nb, 3, 3))
    np.add.at(blocks, (coo.row[same] // 3, coo.row[same] % 3, coo.col[same] % 3), coo.data[same])
    inv = np.linalg.inv(blocks)

    def mv(v):
        return np.einsum("nij,nj->ni", inv, v.reshape(nb, 3)).ravel()

    return spla.LinearOperator(A.shape, matvec=mv, dtype=float)


class Counter:
    def __init__(self):
        self.k = 0

    def __call__(self, _):
        self.k += 1


def krylov(method, A, b, x0, M, rtol, maxiter):
    cb = Counter()
    t0 = time.time()
    if method == "cg":
        x, info = spla.cg(A, b, x0=x0, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
    else:
        x, info = spla.bicgstab(A, b, x0=x0, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
    solve_s = time.time() - t0
    rel = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    return {"iters": cb.k, "info": int(info), "solve_s": solve_s, "rel_res": rel, "converged": bool(rel <= rtol * 1.01)}


def amg_info(ml):
    return {
        "levels": len(ml.levels),
        "operator_complexity": float(ml.operator_complexity()),
        "grid_complexity": float(ml.grid_complexity()),
        "coarsest_n": int(ml.levels[-1].A.shape[0]),
    }


def run_candidate(name, fn):
    t0 = time.time()
    try:
        rec = fn()
        rec["total_s"] = time.time() - t0
        rec["status"] = "ok"
    except Exception as exc:
        rec = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "total_s": time.time() - t0}
        log(f"  {name}: ERROR {rec['error']}")
        traceback.print_exc()
    rec["name"] = name
    return rec


def bench_layer(path, coords_all, args):
    import pyamg

    A, b, x0, layer, step = load(path)
    pinned, free, pin, Aff, Afp = split_blocks(A)
    bf = b[free] - Afp @ x0[pin] if Afp.nnz else b[free]
    xf0 = x0[free]
    nf = len(free)
    free_nodes = free[0::3] // 3
    coords = coords_all[free_nodes]
    log(f"mechanics L{layer}: free dofs={nf} ({nf // 3} nodes) nnz={Aff.nnz}")
    d = Aff.diagonal()
    results = []

    def pardiso():
        import pypardiso
        solver = pypardiso.PyPardisoSolver()
        t0 = time.time()
        x = solver.solve(Aff, bf)
        first = time.time() - t0
        t0 = time.time()
        x2 = solver.solve(Aff, bf + 1e-3 * bf)  # same matrix: factorisation reused
        second = time.time() - t0
        rel = float(np.linalg.norm(bf - Aff @ x) / np.linalg.norm(bf))
        solver.free_memory(everything=True)
        return {"setup_s": first - second, "solve_s": second, "first_solve_s": first, "rel_res": rel, "iters": 1, "converged": True}

    def jac_cg():
        t0 = time.time()
        M = jacobi_operator(Aff)
        setup = time.time() - t0
        return {"setup_s": setup, **krylov("cg", Aff, bf, xf0, M, args.rtol, args.maxiter)}

    def jac_bicg():
        t0 = time.time()
        M = jacobi_operator(Aff)
        setup = time.time() - t0
        return {"setup_s": setup, **krylov("bicgstab", Aff, bf, xf0, M, args.rtol, args.maxiter)}

    def bj_cg():
        t0 = time.time()
        M = block_jacobi_operator(Aff)
        setup = time.time() - t0
        return {"setup_s": setup, **krylov("cg", Aff, bf, xf0, M, args.rtol, args.maxiter)}

    def sa_default():
        t0 = time.time()
        ml = pyamg.smoothed_aggregation_solver(Aff, max_coarse=args.max_coarse)
        setup = time.time() - t0
        return {"setup_s": setup, **amg_info(ml), **krylov("cg", Aff, bf, xf0, ml.aspreconditioner(cycle="V"), args.rtol, args.maxiter)}

    def sa_rbm():
        t0 = time.time()
        B = rigid_body_modes(coords)
        ml = pyamg.smoothed_aggregation_solver(Aff, B=B, max_coarse=args.max_coarse)
        setup = time.time() - t0
        return {"setup_s": setup, **amg_info(ml), **krylov("cg", Aff, bf, xf0, ml.aspreconditioner(cycle="V"), args.rtol, args.maxiter)}

    def sa_rbm_scaled():
        t0 = time.time()
        Dm = sp.diags(1.0 / np.sqrt(d))
        Dp = sp.diags(np.sqrt(d))
        S = (Dm @ Aff @ Dm).tocsr()
        B = Dp @ rigid_body_modes(coords)
        ml = pyamg.smoothed_aggregation_solver(S, B=B, max_coarse=args.max_coarse)
        setup = time.time() - t0
        bs = Dm @ bf
        ys0 = Dp @ xf0
        r = krylov("cg", S, bs, ys0, ml.aspreconditioner(cycle="V"), args.rtol, args.maxiter)
        return {"setup_s": setup, **amg_info(ml), **r}

    def rs_amg():
        t0 = time.time()
        ml = pyamg.ruge_stuben_solver(Aff, max_coarse=args.max_coarse)
        setup = time.time() - t0
        return {"setup_s": setup, **amg_info(ml), **krylov("cg", Aff, bf, xf0, ml.aspreconditioner(cycle="V"), args.rtol, args.maxiter)}

    ladder = [
        ("pardiso_direct", pardiso),
        ("jacobi_cg", jac_cg),
        ("jacobi_bicgstab", jac_bicg),
        ("blockjacobi_cg", bj_cg),
        ("sa_amg_default_cg", sa_default),
        ("sa_amg_rbm_cg", sa_rbm),
        ("sa_amg_rbm_scaled_cg", sa_rbm_scaled),
    ]
    if not args.skip_rs:
        ladder.append(("rs_amg_cg", rs_amg))
    if args.only:
        ladder = [(n, f) for n, f in ladder if n in args.only]

    for name, fn in ladder:
        rec = run_candidate(name, fn)
        rec.update({"layer": layer, "step": step, "n_free": nf})
        results.append(rec)
        if rec["status"] == "ok":
            log(f"  {name}: setup {rec.get('setup_s', 0):.1f}s iters {rec.get('iters')} solve {rec.get('solve_s', 0):.1f}s"
                f" total {rec['total_s']:.1f}s rel_res {rec.get('rel_res', float('nan')):.1e} converged={rec.get('converged')}"
                + (f" opc {rec['operator_complexity']:.2f} levels {rec['levels']}" if "operator_complexity" in rec else ""))
    return results


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = sorted({r["layer"] for r in results})
    names = []
    for r in results:
        if r["name"] not in names:
            names.append(r["name"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.8 / max(len(names), 1)
    for j, name in enumerate(names):
        tot = [next((r["total_s"] for r in results if r["layer"] == L and r["name"] == name and r["status"] == "ok"), np.nan) for L in layers]
        its = [next((r.get("iters", np.nan) for r in results if r["layer"] == L and r["name"] == name and r["status"] == "ok"), np.nan) for L in layers]
        xs = np.arange(len(layers)) + (j - len(names) / 2 + 0.5) * width
        ax[0].bar(xs, tot, width, label=name)
        ax[1].bar(xs, its, width, label=name)
    for a, ylab in ((ax[0], "setup + solve wall time [s]"), (ax[1], "Krylov iterations (cap)")):
        a.set_xticks(np.arange(len(layers)))
        a.set_xticklabels([f"L{L}" for L in layers])
        a.set_yscale("log")
        a.set_ylabel(ylab)
        a.grid(True, which="both", axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("v159 mechanics free block: preconditioner ladder vs build height (CPU)")
    fig.tight_layout()
    out = os.path.join(out_dir, "B_mechanics_ladder.png")
    fig.savefig(out, dpi=130)
    log(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="/home/user/work/159/output/v159_voxel_fast2/matrix_dumps")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mesh-vtu", default="/home/user/work/159/output/v159_voxel_fast2/production/step_000000_scan.vtu")
    ap.add_argument("--layers", default=None, help="comma list of mechanics layers to run")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these candidate names")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--max-coarse", type=int, default=1000)
    ap.add_argument("--skip-rs", action="store_true")
    args = ap.parse_args()
    args.out_dir = args.out_dir or os.path.join(args.dump_dir, "analysis")
    os.makedirs(args.out_dir, exist_ok=True)
    want = {int(v) for v in args.layers.split(",")} if args.layers else None

    import meshio
    coords_all = np.asarray(meshio.read(args.mesh_vtu).points, dtype=float)
    log(f"mesh points {len(coords_all)}")

    results = []
    results_path = os.path.join(args.out_dir, "B_results.json")
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "mechanics_L*.npz"))):
        layer = int(os.path.basename(path).split("_L")[1][:3])
        if want and layer not in want:
            continue
        results.extend(bench_layer(path, coords_all, args))
        json.dump(results, open(results_path, "w"), indent=2)

    keys = ["layer", "n_free", "name", "status", "setup_s", "iters", "solve_s", "total_s", "rel_res", "converged", "levels", "operator_complexity"]
    with open(os.path.join(args.out_dir, "B_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, "") for k in keys])
    if results:
        plot(results, args.out_dir)
    log("B mechanics benchmark done")


if __name__ == "__main__":
    main()
