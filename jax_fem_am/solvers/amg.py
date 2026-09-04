# coding: utf-8
"""AMG-preconditioned Krylov ``custom_solver`` -- pyamg hierarchy, CPU or GPU V-cycle.

Algebraic-multigrid preconditioned CG (or BiCGSTAB) on the *free* block of a
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

Two execution paths share the hierarchy (always built by pyamg on the CPU,
once per sparsity pattern = activation state):

``device="cpu"``
    pyamg's own V-cycle (``ml.aspreconditioner``) with SciPy CG. Single
    threaded; on the v159 mechanics systems 2.5-4x slower in wall time than
    16-thread PARDISO. Smoother ``jacobi`` (default, comparable with the GPU
    path) or ``block_gauss_seidel`` (the group-B configuration).

``device="gpu"`` (alias ``"jax"``)
    The level operators are converted to ``jax.experimental.sparse.CSR``
    (cuSPARSE matvec on a GPU platform), the V-cycle uses damped Jacobi
    smoothing (``omega = smoother_omega / rho(D^-1 A)``, the same formula as
    pyamg's ``('jacobi', {'withrho': True})``) and a pseudo-inverse coarse
    solve, and a preconditioned CG with an iteration counter runs the whole
    thing inside one ``jax.jit``. The fine-level matrix is refreshed with the
    current values on every call; coarser levels keep the Galerkin operators
    from the hierarchy build (as pyamg does when the hierarchy is reused).

The hierarchy is reused while the sparsity pattern is unchanged
(``rebuild="pattern"``) and rebuilt once if the Krylov solve does not
converge; a second failure hands the full system to PARDISO
(``fallback="pardiso"``) so a run never dies on a bad preconditioner.

Protocol (see ``jax_fem.solver.linear_solver``)::

    x = solver(A_petsc, b, x0, linear_options)

plus ``__deepcopy__`` returning ``self`` (the wrapper deep-copies the Newton
options every solve), the ``bind_problem(problem)`` hook the acceleration
wrapper calls before every solve (node coordinates and ``vec`` for the
rigid-body modes), ``stats_snapshot()`` for the profile report, and
``iterative = True`` so a Newton stall under this solver is retried on the
direct fallback.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SMOOTHERS = ("jacobi", "block_gauss_seidel")
DEVICES = ("cpu", "gpu", "jax")


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


def rigid_body_modes_for_dofs(coords: onp.ndarray, dofs: onp.ndarray) -> onp.ndarray:
    """Rows of the six rigid-body modes for the given global dofs
    (``dof = 3 * node + comp``), so partially pinned nodes are handled."""
    dofs = onp.asarray(dofs, dtype=onp.int64)
    nodes, local = onp.unique(dofs // 3, return_inverse=True)
    B_nodes = rigid_body_modes(coords[nodes])
    return B_nodes[3 * local + (dofs % 3)]


class _IterationCounter:
    def __init__(self) -> None:
        self.k = 0

    def __call__(self, _xk) -> None:
        self.k += 1


def _krylov_cpu(method, S, rhs, y0, M, tol, maxiter):
    counter = _IterationCounter()
    fn = spla.cg if method == "cg" else spla.bicgstab
    kwargs = dict(x0=y0, M=M, maxiter=maxiter, callback=counter)
    try:
        y, info = fn(S, rhs, rtol=tol, atol=0.0, **kwargs)
    except TypeError:  # scipy < 1.12 spelling
        y, info = fn(S, rhs, tol=tol, atol=0.0, **kwargs)
    return y, int(info), counter.k


# ---------------------------------------------------------------------------
# jax V-cycle + PCG (built lazily so the CPU path never imports jax)
# ---------------------------------------------------------------------------
_JAX = None  # module-level cache: dict with jit-compiled kernels and helpers


def _jax_kernels():
    global _JAX
    if _JAX is not None:
        return _JAX
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import sparse as jsparse

    if not jax.config.jax_enable_x64:
        warnings.warn("enabling jax_enable_x64 for the pyamg GPU path", RuntimeWarning)
        jax.config.update("jax_enable_x64", True)

    def csr(A: sp.csr_matrix):
        A = A.tocsr()
        A.sort_indices()
        return jsparse.CSR(
            (
                jnp.asarray(A.data, dtype=jnp.float64),
                jnp.asarray(A.indices.astype(onp.int32, copy=False)),
                jnp.asarray(A.indptr.astype(onp.int32, copy=False)),
            ),
            shape=A.shape,
        )

    def vcycle(levels, coarse_pinv, sweeps, r0):
        # levels[l] = (A, dinv, omega, P, R); unrolled recursion (2-5 levels)
        def descend(l, r):
            if l == len(levels):
                return coarse_pinv @ r
            A, dinv, omega, P, R = levels[l]
            e = omega * dinv * r  # first damped-Jacobi sweep from a zero guess
            for _ in range(sweeps - 1):
                e = e + omega * dinv * (r - A @ e)
            rc = R @ (r - A @ e)
            e = e + P @ descend(l + 1, rc)
            for _ in range(sweeps):
                e = e + omega * dinv * (r - A @ e)
            return e

        return descend(0, r0)

    def pcg(S, b, x0, levels, coarse_pinv, tol, maxiter, sweeps):
        def M(r):
            return vcycle(levels, coarse_pinv, sweeps, r)

        b_norm2 = jnp.vdot(b, b)
        x = x0
        r = b - S @ x
        z = M(r)
        p = z
        rz = jnp.vdot(r, z)

        def cond(state):
            _x, _r, _z, _p, _rz, k, rr = state
            return (rr > tol * tol * b_norm2) & (k < maxiter)

        def body(state):
            x, r, z, p, rz, k, _rr = state
            q = S @ p
            alpha = rz / jnp.vdot(p, q)
            x = x + alpha * p
            r = r - alpha * q
            z = M(r)
            rz_new = jnp.vdot(r, z)
            beta = rz_new / rz
            p = z + beta * p
            return x, r, z, p, rz_new, k + 1, jnp.vdot(r, r)

        state = (x, r, z, p, rz, jnp.int32(0), jnp.vdot(r, r))
        x, r, z, p, rz, k, rr = lax.while_loop(cond, body, state)
        return x, k, jnp.sqrt(rr)

    pcg_jit = jax.jit(pcg, static_argnames=("maxiter", "sweeps"))

    @jax.jit
    def scale_data(data, scale, rows, cols):
        return data * scale[rows] * scale[cols]

    @jax.jit
    def residual_norm(S, y, rhs):
        return jnp.linalg.norm(rhs - S @ y)

    _JAX = {
        "jax": jax, "jnp": jnp, "csr": csr, "CSR": jsparse.CSR,
        "pcg": pcg_jit, "scale_data": scale_data, "residual_norm": residual_norm,
    }
    return _JAX


class _PatternCache:
    """Free/pinned split and the AMG hierarchy for one sparsity pattern."""

    __slots__ = (
        "signature", "n", "free", "pin", "keep", "ff_indptr", "ff_indices", "ff_rows",
        "ff_diag_pos", "hierarchy", "hierarchy_info", "jax_levels", "jax_coarse_pinv",
        "jax_omega0", "dev_rows", "dev_cols", "dev_indptr", "dev_diag_pos",
    )

    def __init__(self, signature, n, free, pin, keep, ff_indptr, ff_indices, ff_rows, ff_diag_pos):
        self.signature = signature
        self.n = n
        self.free = free
        self.pin = pin
        self.keep = keep
        self.ff_indptr = ff_indptr
        self.ff_indices = ff_indices
        self.ff_rows = ff_rows
        self.ff_diag_pos = ff_diag_pos
        self.hierarchy = None
        self.hierarchy_info: Dict[str, Any] = {}
        self.jax_levels = None
        self.jax_coarse_pinv = None
        self.jax_omega0 = None
        self.dev_rows = None
        self.dev_cols = None
        self.dev_indptr = None
        self.dev_diag_pos = None


class PyamgKrylovSolver:
    """``custom_solver`` adapter: PETSc AIJ -> free block -> SA-AMG(RBM) + CG."""

    iterative = True

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
        device: str = "cpu",
        smoother: str = "jacobi",
        smoother_sweeps: int = 2,
        smoother_omega: float = 4.0 / 3.0,
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
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        if smoother not in SMOOTHERS:
            raise ValueError(f"smoother must be one of {SMOOTHERS}, got {smoother!r}")
        self.use_jax = device in ("gpu", "jax")
        if self.use_jax and smoother != "jacobi":
            raise ValueError("the gpu/jax V-cycle supports the jacobi smoother only")
        if self.use_jax and method != "cg":
            raise ValueError("the gpu/jax path implements cg only (the free block is symmetric)")
        if int(smoother_sweeps) < 1:
            raise ValueError("smoother_sweeps must be >= 1")
        self.method = method
        self.near_nullspace = near_nullspace
        self.scaled = bool(scaled)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.max_coarse = int(max_coarse)
        self.rebuild = rebuild
        self.fallback = fallback
        self.pardiso_mode = pardiso_mode
        self.device = "jax" if self.use_jax else "cpu"
        self.smoother = smoother
        self.smoother_sweeps = int(smoother_sweeps)
        self.smoother_omega = float(smoother_omega)
        self.verbose = bool(verbose)
        smoother_label = (
            f"{smoother}x{self.smoother_sweeps}" if smoother == "jacobi" else "block_gs(sym)"
        )
        self.label = (
            f"pyamg_solver(sa+{near_nullspace}{'+scaled' if self.scaled else ''}, "
            f"{method}, {smoother_label}, {self.device}, tol={self.tol:g}, "
            f"maxiter={self.maxiter}, fallback={fallback})"
        )
        self._points: Optional[onp.ndarray] = None
        self._vec: Optional[int] = None
        self._cache: Optional[_PatternCache] = None
        self._rbm_warned = False
        self.stats: Dict[str, Any] = {
            "device": self.device,
            "calls": 0,
            "iterations": 0,
            "setup_s": 0.0,
            "solve_s": 0.0,
            "pattern_rebuilds": 0,
            "hierarchy_rebuilds": 0,
            "fallbacks": 0,
            "last_iterations": 0,
            "last_rel_res": None,
            "last_solve_s": None,
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

    # -- pattern bookkeeping ------------------------------------------------------
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
        ff_rows = free_map[rows[keep]].astype(onp.int32, copy=False)
        ff_indices = free_map[indices[keep]].astype(onp.int32, copy=False)
        counts = onp.bincount(ff_rows, minlength=free.size)
        ff_indptr = onp.concatenate([[0], onp.cumsum(counts)]).astype(onp.int64)
        diag_pos = onp.flatnonzero(ff_rows == ff_indices)
        if diag_pos.size != free.size:
            diag_pos = None  # a free row without a stored diagonal: fall back to scipy
        return _PatternCache(signature, n, free, pin, keep, ff_indptr, ff_indices, ff_rows, diag_pos)

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
        if free.size == 0 or int(free[-1]) // 3 >= self._points.shape[0]:
            return None
        return rigid_body_modes_for_dofs(self._points, free)

    # -- hierarchy ------------------------------------------------------------------
    def _smoother_spec(self):
        if self.smoother == "jacobi":
            return ("jacobi", {"omega": self.smoother_omega, "withrho": True,
                               "iterations": self.smoother_sweeps})
        return ("block_gauss_seidel", {"sweep": "symmetric"})

    def _build_hierarchy(self, cache: _PatternCache, S, scale: Optional[onp.ndarray]):
        import pyamg

        t0 = time.perf_counter()
        B = self._near_nullspace(cache.free)
        if B is not None and scale is not None:
            # S = D^-1/2 A D^-1/2  =>  near-nullspace of S is D^1/2 B.
            B = B / scale[:, None]
        sm = self._smoother_spec()
        ml = pyamg.smoothed_aggregation_solver(
            S, B=B, max_coarse=self.max_coarse, presmoother=sm, postsmoother=sm
        )
        cache.hierarchy = ml
        cache.hierarchy_info = {
            "levels": len(ml.levels),
            "operator_complexity": float(ml.operator_complexity()),
            "near_nullspace": "rigid_body" if B is not None else "constant",
        }
        if self.use_jax:
            self._to_jax_hierarchy(cache, ml)
        setup = time.perf_counter() - t0
        self.stats["setup_s"] += setup
        self.stats["last_levels"] = cache.hierarchy_info["levels"]
        self.stats["last_operator_complexity"] = cache.hierarchy_info["operator_complexity"]
        if self.verbose:
            print(
                f"pyamg[{self.device}]: hierarchy built in {setup:.2f}s -- levels {len(ml.levels)}, "
                f"operator complexity {ml.operator_complexity():.3f}, "
                f"near-nullspace {cache.hierarchy_info['near_nullspace']}, smoother {sm[0]}",
                flush=True,
            )
        return ml

    def _to_jax_hierarchy(self, cache: _PatternCache, ml) -> None:
        from pyamg.relaxation.smoothing import rho_D_inv_A

        K = _jax_kernels()
        jnp = K["jnp"]
        levels = []
        for l, lvl in enumerate(ml.levels[:-1]):
            A = lvl.A.tocsr()
            rho = float(rho_D_inv_A(A))
            omega = jnp.float64(self.smoother_omega / rho)
            if l == 0:
                # fine level: matrix and 1/diag are refreshed from the current
                # values on every call (see __call__); only omega is kept.
                cache.jax_omega0 = omega
                levels.append(None)
                continue
            diag = A.diagonal()
            safe = onp.where(diag != 0.0, diag, 1.0)
            dinv = jnp.asarray(onp.where(diag != 0.0, 1.0 / safe, 0.0))
            levels.append((K["csr"](A), dinv, omega, K["csr"](lvl.P.tocsr()), K["csr"](lvl.R.tocsr())))
        # level-0 transfer operators
        lvl0 = ml.levels[0]
        cache.jax_levels = levels
        cache.jax_levels[0] = (None, None, cache.jax_omega0, K["csr"](lvl0.P.tocsr()), K["csr"](lvl0.R.tocsr()))
        Ac = ml.levels[-1].A.toarray()
        cache.jax_coarse_pinv = jnp.asarray(onp.linalg.pinv(Ac))
        if cache.dev_cols is None:
            cache.dev_rows = jnp.asarray(cache.ff_rows)
            cache.dev_cols = jnp.asarray(cache.ff_indices)
            cache.dev_indptr = jnp.asarray(cache.ff_indptr.astype(onp.int32, copy=False))
            cache.dev_diag_pos = (
                jnp.asarray(cache.ff_diag_pos.astype(onp.int32, copy=False))
                if cache.ff_diag_pos is not None else None
            )

    # -- fallback -------------------------------------------------------------------
    def _direct_fallback(self, A, b, x0, linear_options, reason: str):
        self.stats["fallbacks"] += 1
        if self.fallback != "pardiso":
            raise RuntimeError(f"pyamg solver failed: {reason}")
        print(f"WARNING: pyamg solver {reason}; solving this system with PARDISO", flush=True)
        from jax_fem_am.solvers.linear import shared_pardiso_solver

        return shared_pardiso_solver(self.pardiso_mode)(A, b, x0, linear_options)

    # -- krylov drivers ---------------------------------------------------------------
    def _solve_cpu(self, cache, S, rhs, y0, rhs_norm):
        ml = cache.hierarchy
        y, info, iters = _krylov_cpu(
            self.method, S, rhs, y0, ml.aspreconditioner(cycle="V"), self.tol, self.maxiter
        )
        rel = float(onp.linalg.norm(rhs - S @ y) / rhs_norm)
        return y, iters, rel, info

    def _solve_jax(self, cache, data_ff, scale, d, rhs, y0, rhs_norm):
        K = _jax_kernels()
        jnp = K["jnp"]
        nf = cache.free.size
        data_dev = jnp.asarray(data_ff)
        if scale is not None:
            data_dev = K["scale_data"](data_dev, jnp.asarray(scale), cache.dev_rows, cache.dev_cols)
        S_dev = K["CSR"]((data_dev, cache.dev_cols, cache.dev_indptr), shape=(nf, nf))
        if cache.dev_diag_pos is not None:
            dinv0 = 1.0 / data_dev[cache.dev_diag_pos]
        else:
            diag = d * (scale * scale) if scale is not None else d
            dinv0 = jnp.asarray(1.0 / diag)
        levels = list(cache.jax_levels)
        _, _, omega0, P0, R0 = levels[0]
        levels[0] = (S_dev, dinv0, omega0, P0, R0)
        rhs_dev = jnp.asarray(rhs)
        y0_dev = jnp.asarray(y0)
        y, k, _rr = K["pcg"](
            S_dev, rhs_dev, y0_dev, levels, cache.jax_coarse_pinv, self.tol,
            maxiter=self.maxiter, sweeps=self.smoother_sweeps,
        )
        rel = float(K["residual_norm"](S_dev, y, rhs_dev)) / rhs_norm
        return onp.asarray(y), int(k), rel, 0

    # -- the solve --------------------------------------------------------------------
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

        data_ff = data[cache.keep]
        Aff = sp.csr_matrix((data_ff, cache.ff_indices, cache.ff_indptr), shape=(nf, nf))
        x_full = onp.zeros(n)
        x_full[pin] = b[pin]
        bf = b[free] - (Afull @ x_full)[free]
        d = data_ff[cache.ff_diag_pos] if cache.ff_diag_pos is not None else Aff.diagonal()
        if not onp.all(onp.isfinite(d)) or onp.any(d <= 0.0):
            return self._direct_fallback(
                A, b, x0, linear_options, "free block has a non-positive diagonal"
            )
        if self.scaled:
            scale = 1.0 / onp.sqrt(d)
            D = sp.diags(scale)
            S = (D @ Aff @ D).tocsr() if (not self.use_jax or cache.hierarchy is None) else None
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
        if not reused:
            self._build_hierarchy(cache, S, scale)
        converged = False
        y = y0
        for attempt in range(2):
            t0 = time.perf_counter()
            if self.use_jax:
                y, iters, rel, info = self._solve_jax(cache, data_ff, scale, d, rhs, y0, rhs_norm)
            else:
                y, iters, rel, info = self._solve_cpu(cache, S, rhs, y0, rhs_norm)
            solve_s = time.perf_counter() - t0
            self.stats["solve_s"] += solve_s
            self.stats["last_solve_s"] = solve_s
            self.stats["iterations"] += iters
            self.stats["last_iterations"] = iters
            self.stats["last_rel_res"] = rel
            converged = bool(onp.isfinite(rel) and rel <= self.tol * 1.01)
            if self.verbose:
                print(
                    f"pyamg[{self.device}]: {self.method} iters {iters} rel_res {rel:.2e} "
                    f"in {solve_s:.2f}s ({'converged' if converged else 'NOT converged'}, "
                    f"hierarchy {'reused' if reused and attempt == 0 else 'fresh'})",
                    flush=True,
                )
            if converged or not reused or attempt == 1:
                break
            # The cached hierarchy no longer matches the tangent: rebuild once.
            self.stats["hierarchy_rebuilds"] += 1
            if S is None:
                S = (sp.diags(scale) @ Aff @ sp.diags(scale)).tocsr()
            self._build_hierarchy(cache, S, scale)
        if not converged:
            return self._direct_fallback(
                A, b, x0, linear_options,
                f"stalled (rel_res {self.stats['last_rel_res']:.2e} after "
                f"{self.stats['last_iterations']} {self.method} iterations)",
            )
        x_full[free] = y * scale if scale is not None else y
        if self.verbose:
            print(f"pyamg[{self.device}]: solve wall {time.perf_counter() - t_call:.2f}s", flush=True)
        return x_full
