# coding: utf-8
"""pyamg-backed Krylov ``custom_solver`` -- CPU reference implementation.

Algebraic-multigrid preconditioned CG/BiCGSTAB on the *free* block of a
jax-fem tangent. jax-fem stores Dirichlet rows after row elimination as
``diag = 1, off-diagonal = 0`` with the prescribed value in ``b``; those rows
are detected, stripped, and their coupling moved to the right-hand side, so
the Krylov iteration only sees the (symmetric, positive definite) free block.

The preconditioner is pyamg smoothed aggregation with the near-nullspace the
group-B benchmark showed to be decisive for the mechanics tangent: the six
rigid-body modes built from node coordinates (``dof = vec * node + comp``),
on the Jacobi-scaled matrix ``D^-1/2 A D^-1/2``. With it, CG needs 46-56
iterations at every v159 build height (L30/L90/L150), while plain Jacobi,
3x3 block Jacobi, constant-near-nullspace SA and Ruge-Stueben all hit the
1000-iteration cap (cases/159_simulation/analysis/results_B, 2026-09-03).

The hierarchy is cached while the sparsity pattern (activation state) is
unchanged (``rebuild="pattern"``) and rebuilt once if the Krylov solve does
not converge; a second failure hands the full system to PARDISO
(``fallback="pardiso"``) so a run never dies on a bad preconditioner.

This is single-threaded pyamg on the CPU: on the v159 mechanics systems the
wall time is 2.5-4x that of 16-thread PARDISO, so it is the *reference* for
a GPU V-cycle (same hierarchy, jax SpMV smoothers), not the production
lane. Select it with ``--mechanics-linear-solver pyamg`` or a JSON spec.

Protocol (see ``jax_fem.solver.linear_solver``)::

    x = solver(A_petsc, b, x0, linear_options)

plus the optional ``bind_problem(problem)`` hook the acceleration wrapper
calls before every solve (node coordinates and ``vec`` for the rigid-body
modes) and ``stats_snapshot()`` for the profile report.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def pinned_rows(A: sp.csr_matrix) -> onp.ndarray:
    """Rows stored as ``diag == 1`` with no off-diagonal entries (Dirichlet rows)."""
    diag = A.diagonal()
    offabs = onp.asarray(abs(A).sum(axis=1)).ravel() - onp.abs(diag)
    return onp.isclose(diag, 1.0) & (offabs == 0.0)


def rigid_body_modes(coords: onp.ndarray) -> onp.ndarray:
    """Six rigid-body modes for node-major (x, y, z) interleaved dofs."""
    c = onp.asarray(coords, dtype=onp.float64)
    c = c - c.mean(axis=0)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    nn = len(c)
    B = onp.zeros((3 * nn, 6))
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


class _IterationCounter:
    def __init__(self) -> None:
        self.k = 0

    def __call__(self, _xk) -> None:
        self.k += 1


def _krylov(method, S, rhs, y0, M, tol, maxiter):
    counter = _IterationCounter()
    fn = spla.cg if method == "cg" else spla.bicgstab
    kwargs = dict(x0=y0, M=M, maxiter=maxiter, callback=counter)
    try:
        y, info = fn(S, rhs, rtol=tol, atol=0.0, **kwargs)
    except TypeError:  # scipy < 1.12 spelling
        y, info = fn(S, rhs, tol=tol, atol=0.0, **kwargs)
    return y, int(info), counter.k


class _PatternCache:
    """Free/pinned split and the AMG hierarchy for one sparsity pattern."""

    __slots__ = (
        "signature", "n", "free", "pin", "keep", "ff_indptr", "ff_indices",
        "hierarchy", "hierarchy_info",
    )

    def __init__(self, signature, n, free, pin, keep, ff_indptr, ff_indices):
        self.signature = signature
        self.n = n
        self.free = free
        self.pin = pin
        self.keep = keep
        self.ff_indptr = ff_indptr
        self.ff_indices = ff_indices
        self.hierarchy = None
        self.hierarchy_info: Dict[str, Any] = {}


class PyamgKrylovSolver:
    """``custom_solver`` adapter: PETSc AIJ -> free block -> SA-AMG(RBM) + CG."""

    def __init__(
        self,
        *,
        method: str = "cg",
        near_nullspace: str = "rigid_body",
        scaled: bool = True,
        tol: float = 1e-6,
        maxiter: int = 500,
        max_coarse: int = 1000,
        rebuild: str = "pattern",
        fallback: str = "pardiso",
        pardiso_mode: str = "phase23",
        verbose: bool = False,
    ) -> None:
        if method not in ("cg", "bicgstab"):
            raise ValueError(f"pyamg method must be cg or bicgstab, got {method!r}")
        if near_nullspace not in ("rigid_body", "constant"):
            raise ValueError(f"near_nullspace must be rigid_body or constant, got {near_nullspace!r}")
        if rebuild not in ("pattern", "always"):
            raise ValueError(f"rebuild must be pattern or always, got {rebuild!r}")
        if fallback not in ("pardiso", "none"):
            raise ValueError(f"fallback must be pardiso or none, got {fallback!r}")
        self.method = method
        self.near_nullspace = near_nullspace
        self.scaled = bool(scaled)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.max_coarse = int(max_coarse)
        self.rebuild = rebuild
        self.fallback = fallback
        self.pardiso_mode = pardiso_mode
        self.verbose = bool(verbose)
        self.label = (
            f"pyamg_solver(sa+{near_nullspace}{'+scaled' if self.scaled else ''}, "
            f"{method}, tol={self.tol:g}, maxiter={self.maxiter}, fallback={fallback})"
        )
        self._points: Optional[onp.ndarray] = None
        self._vec: Optional[int] = None
        self._cache: Optional[_PatternCache] = None
        self._rbm_warned = False
        self.stats: Dict[str, Any] = {
            "calls": 0,
            "iterations": 0,
            "setup_s": 0.0,
            "solve_s": 0.0,
            "pattern_rebuilds": 0,
            "hierarchy_rebuilds": 0,
            "fallbacks": 0,
            "last_iterations": 0,
            "last_rel_res": None,
            "last_levels": None,
            "last_operator_complexity": None,
        }

    def __deepcopy__(self, memo):
        # The option-rewrite deep copies must not clone the cached hierarchy.
        return self

    # -- hooks used by the acceleration wrapper ---------------------------------
    def bind_problem(self, problem) -> None:
        fe = problem.fes[0]
        points = onp.asarray(fe.points, dtype=onp.float64)
        vec = int(getattr(fe, "vec", 1))
        if self._points is None or self._points.shape != points.shape or self._vec != vec:
            self._points = points
            self._vec = vec
            self._cache = None

    def stats_snapshot(self) -> Dict[str, Any]:
        return dict(self.stats)

    # -- internals --------------------------------------------------------------
    @staticmethod
    def _signature(n, indptr, indices, pinned):
        pinned_idx = onp.flatnonzero(pinned)
        return (
            int(n),
            int(indptr[-1]),
            int(pinned_idx.size),
            hash(onp.ascontiguousarray(indptr[::4093]).tobytes()),
            hash(onp.ascontiguousarray(indices[::9973]).tobytes()),
            hash(onp.ascontiguousarray(pinned_idx[::997]).tobytes()),
        )

    def _build_pattern(self, signature, n, indptr, indices, pinned) -> _PatternCache:
        free = onp.flatnonzero(~pinned)
        pin = onp.flatnonzero(pinned)
        free_map = onp.full(n, -1, dtype=onp.int64)
        free_map[free] = onp.arange(free.size, dtype=onp.int64)
        rows = onp.repeat(onp.arange(n, dtype=onp.int64), onp.diff(indptr))
        keep = (~pinned[rows]) & (~pinned[indices])
        ff_indices = free_map[indices[keep]].astype(onp.int32, copy=False)
        counts = onp.bincount(free_map[rows[keep]], minlength=free.size)
        ff_indptr = onp.concatenate([[0], onp.cumsum(counts)]).astype(onp.int64)
        return _PatternCache(signature, n, free, pin, keep, ff_indptr, ff_indices)

    def _near_nullspace(self, free: onp.ndarray) -> Optional[onp.ndarray]:
        if self.near_nullspace == "constant":
            return None
        if self._points is None or self._vec != 3:
            if not self._rbm_warned:
                warnings.warn(
                    "pyamg rigid_body near-nullspace needs a bound 3-component problem; "
                    "falling back to the constant near-nullspace",
                    RuntimeWarning,
                )
                self._rbm_warned = True
            return None
        if free.size % 3 != 0:
            return None
        nodes = free[0::3] // 3
        expected = (3 * nodes[:, None] + onp.arange(3)[None, :]).ravel()
        if not onp.array_equal(expected, free):
            if not self._rbm_warned:
                warnings.warn(
                    "pyamg rigid_body near-nullspace: free dofs are not whole nodes; "
                    "falling back to the constant near-nullspace",
                    RuntimeWarning,
                )
                self._rbm_warned = True
            return None
        return rigid_body_modes(self._points[nodes])

    def _build_hierarchy(self, cache: _PatternCache, S, scale: Optional[onp.ndarray]):
        import pyamg

        t0 = time.perf_counter()
        B = self._near_nullspace(cache.free)
        if B is not None and scale is not None:
            # S = D^-1/2 A D^-1/2  =>  near-nullspace of S is D^1/2 B.
            B = B / scale[:, None]
        ml = pyamg.smoothed_aggregation_solver(S, B=B, max_coarse=self.max_coarse)
        cache.hierarchy = ml
        cache.hierarchy_info = {
            "levels": len(ml.levels),
            "operator_complexity": float(ml.operator_complexity()),
            "near_nullspace": "rigid_body" if B is not None else "constant",
        }
        setup = time.perf_counter() - t0
        self.stats["setup_s"] += setup
        self.stats["last_levels"] = cache.hierarchy_info["levels"]
        self.stats["last_operator_complexity"] = cache.hierarchy_info["operator_complexity"]
        if self.verbose:
            print(
                f"pyamg: hierarchy built in {setup:.2f}s -- levels {len(ml.levels)}, "
                f"operator complexity {ml.operator_complexity():.3f}, "
                f"near-nullspace {cache.hierarchy_info['near_nullspace']}",
                flush=True,
            )
        return ml

    def _direct_fallback(self, A, b, x0, linear_options, reason: str):
        self.stats["fallbacks"] += 1
        if self.fallback != "pardiso":
            raise RuntimeError(f"pyamg solver failed: {reason}")
        print(f"WARNING: pyamg solver {reason}; solving this system with PARDISO", flush=True)
        from jax_fem_am.solvers.linear import shared_pardiso_solver

        return shared_pardiso_solver(self.pardiso_mode)(A, b, x0, linear_options)

    # -- the solve --------------------------------------------------------------
    def __call__(self, A, b, x0, linear_options):
        self.stats["calls"] += 1
        t_call = time.perf_counter()
        indptr, indices, data = A.getValuesCSR()
        indptr = onp.asarray(indptr)
        indices = onp.asarray(indices)
        data = onp.asarray(data, dtype=onp.float64)
        b = onp.asarray(b, dtype=onp.float64).ravel()
        n = b.size
        x0 = onp.zeros(n) if x0 is None else onp.asarray(x0, dtype=onp.float64).ravel()
        Afull = sp.csr_matrix((data, indices, indptr), shape=(n, n))
        pinned = pinned_rows(Afull)
        signature = self._signature(n, indptr, indices, pinned)

        cache = self._cache
        if cache is None or cache.signature != signature:
            cache = self._build_pattern(signature, n, indptr, indices, pinned)
            self._cache = cache
            self.stats["pattern_rebuilds"] += 1
        elif self.rebuild == "always":
            cache.hierarchy = None
        free, pin = cache.free, cache.pin
        nf = free.size
        if nf == 0:
            return b.copy()

        Aff = sp.csr_matrix(
            (data[cache.keep], cache.ff_indices, cache.ff_indptr), shape=(nf, nf)
        )
        x_full = onp.zeros(n)
        x_full[pin] = b[pin]
        bf = b[free] - (Afull @ x_full)[free]
        d = Aff.diagonal()
        if not onp.all(onp.isfinite(d)) or onp.any(d <= 0.0):
            return self._direct_fallback(
                A, b, x0, linear_options, "free block has a non-positive diagonal"
            )
        if self.scaled:
            scale = 1.0 / onp.sqrt(d)
            D = sp.diags(scale)
            S = (D @ Aff @ D).tocsr()
            rhs = scale * bf
            y0 = x0[free] / scale
        else:
            scale = None
            S = Aff
            rhs = bf
            y0 = x0[free]
        rhs_norm = float(onp.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return x_full

        reused = cache.hierarchy is not None
        ml = cache.hierarchy or self._build_hierarchy(cache, S, scale)
        converged = False
        y = y0
        for attempt in range(2):
            t0 = time.perf_counter()
            y, info, iters = _krylov(
                self.method, S, rhs, y0, ml.aspreconditioner(cycle="V"), self.tol, self.maxiter
            )
            self.stats["solve_s"] += time.perf_counter() - t0
            self.stats["iterations"] += iters
            self.stats["last_iterations"] = iters
            rel = float(onp.linalg.norm(rhs - S @ y) / rhs_norm)
            self.stats["last_rel_res"] = rel
            converged = onp.isfinite(rel) and rel <= self.tol * 1.01
            if self.verbose:
                print(
                    f"pyamg: {self.method} iters {iters} rel_res {rel:.2e} "
                    f"({'converged' if converged else 'NOT converged'}, info={info}, "
                    f"hierarchy {'reused' if reused and attempt == 0 else 'fresh'})",
                    flush=True,
                )
            if converged or not reused or attempt == 1:
                break
            # The cached hierarchy no longer matches the tangent: rebuild once.
            self.stats["hierarchy_rebuilds"] += 1
            ml = self._build_hierarchy(cache, S, scale)
        if not converged:
            return self._direct_fallback(
                A, b, x0, linear_options,
                f"did not converge (rel_res {self.stats['last_rel_res']:.2e} after "
                f"{self.stats['last_iterations']} {self.method} iterations)",
            )
        x_full[free] = y * scale if scale is not None else y
        if self.verbose:
            print(f"pyamg: solve wall {time.perf_counter() - t_call:.2f}s", flush=True)
        return x_full
