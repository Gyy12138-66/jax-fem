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

Two execution paths share the hierarchy (always built by pyamg on the CPU on
the free block, once per sparsity pattern = activation state):

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

Shape modes of the GPU path (``shape_mode``)
--------------------------------------------
``"free"``
    Legacy: the jit kernels see the compacted free block, so every activation
    layer is a new array shape. Measured on the v159 MODE=5 production run
    (2026-09-07/08): 153 distinct free-block sizes over 182 slabs, and every
    new shape left ~155-195 MB of host memory resident (jax/XLA per-shape
    state that neither ``gc``, ``jax.clear_caches`` nor ``malloc_trim``
    reclaims) -- 153 x 186 MB ~ 28 GB, which is where the run froze against the
    40 GB WSL cap. See BUG_FIX.md (2026-09-14).
``"full"`` (default on the GPU path)
    The whole-mesh shape is fixed for the entire build: the Krylov system is
    the full ``n x n`` tangent with pinned rows/columns decoupled to identity
    (mathematically identical to the reduced solve -- the free rows are the
    same equations, the pinned rows are exact and carry zero residual), and
    the coarse levels of the pyamg hierarchy (still built on the free block,
    which is cheaper and yields the very same coarse operators) are embedded
    into fixed-capacity padded operators with a fixed level count. Result:
    one jit executable per mesh, host memory flat after warm-up. Cost: the
    fine-level SpMV runs over the structural zeros of the inactive region,
    +24% CG time over a full v159 build; the hierarchy build is unchanged.
``"bucket"``
    Template only (raises ``NotImplementedError``): pad the free block to a
    geometric ladder of capacities instead of the full mesh, trading a
    handful of recompilations for less padding. Needs an adaptive policy;
    see ``bucket_capacity`` and the TODO list there.
``"auto"``
    ``"full"`` on the GPU path, ``"free"`` on the CPU path (no jit there).

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
SCALE_POLICIES = ("current", "frozen")
SHAPE_MODES = ("auto", "free", "full", "bucket")


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


def _dense_inverse(Ac: onp.ndarray) -> onp.ndarray:
    """Inverse of the coarsest-level operator: Cholesky when SPD, else a
    symmetric pseudo-inverse (pyamg's default coarse solver is pinv)."""
    import scipy.linalg as sla

    Ac = onp.asarray(Ac, dtype=onp.float64)
    Ac = 0.5 * (Ac + Ac.T)
    try:
        c, lower = sla.cho_factor(Ac, check_finite=False)
        inv = sla.cho_solve((c, lower), onp.eye(Ac.shape[0]), check_finite=False)
        if onp.all(onp.isfinite(inv)):
            return inv
    except sla.LinAlgError:
        pass
    return sla.pinvh(Ac, check_finite=False)


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
# fixed-shape padding (shape_mode="full")
# ---------------------------------------------------------------------------
class _CapacityError(Exception):
    """A padded operator does not fit its capacity; carries what is needed."""

    def __init__(self, kind: str, needed: int):
        super().__init__(f"{kind}: needs {needed}")
        self.kind = kind
        self.needed = int(needed)


def _round_up(x: float, q: int) -> int:
    x = int(-(-x // 1))
    return int(-(-x // q) * q)


def default_capacities(n: int, fixed_levels: int, max_coarse: int = 1000) -> Dict[str, int]:
    """Padded capacities for a mesh with ``n`` dofs (rows per coarse level, nnz
    per operator). Calibrated on the v159 mechanics hierarchy at full
    activation (818k free dofs -> levels [818394, 60654, 1734, 78]; P0 22.6
    nnz/row, A1 170 nnz/row, P1 19.4 nnz/row) with ~1.5-2x margin, so the 159
    build runs in ONE shape. The last level must also hold the coarsest level
    of a *shallower* hierarchy (duplicated there, see
    ``_to_jax_hierarchy_full``), which pyamg bounds by ``max_coarse`` dofs. A
    capacity that turns out too small grows once (x1.5, rounded) and costs one
    recompilation -- never a failure."""
    caps: Dict[str, int] = {}
    rows_prev = int(n)
    for level in range(1, fixed_levels):
        rows = _round_up(max(512, rows_prev // (8 if level == 1 else 24)), 256)
        caps[f"rows{level}"] = rows
        rows_prev = rows
    last = f"rows{fixed_levels - 1}"
    caps[last] = max(caps[last], _round_up(1.25 * int(max_coarse), 256))
    caps["P0"] = 24 * int(n)
    for level in range(1, fixed_levels - 1):
        caps[f"A{level}"] = 200 * caps[f"rows{level}"]
        caps[f"P{level}"] = 24 * caps[f"rows{level}"]
    return caps


def _pad_csr(M: sp.csr_matrix, rows_cap: int, cols_cap: int, nnz_cap: int,
             kind: str, identity_pad: bool = False) -> sp.csr_matrix:
    """Embed ``M`` (r x c) top-left into a (rows_cap x cols_cap) CSR with exactly
    ``nnz_cap`` stored entries. Padding rows are empty (or identity when
    ``identity_pad``); the remaining entries are explicit zeros on the last
    row, so every array shape is a function of the capacities only."""
    M = M.tocsr()
    M.sort_indices()
    r, c = M.shape
    if r > rows_cap:
        raise _CapacityError(kind + ".rows", r)
    if c > cols_cap:
        raise _CapacityError(kind + ".cols", c)
    extra_rows = rows_cap - r
    ident = extra_rows if identity_pad else 0
    if M.nnz + ident > nnz_cap:
        raise _CapacityError(kind + ".nnz", M.nnz + ident)
    data = onp.asarray(M.data, dtype=onp.float64)
    indices = onp.asarray(M.indices, dtype=onp.int32)
    indptr = onp.asarray(M.indptr, dtype=onp.int64)
    if extra_rows > 0:
        if identity_pad:
            data = onp.concatenate([data, onp.ones(extra_rows)])
            indices = onp.concatenate([indices, onp.arange(r, rows_cap, dtype=onp.int32)])
            indptr = onp.concatenate([indptr, indptr[-1] + onp.arange(1, extra_rows + 1, dtype=onp.int64)])
        else:
            indptr = onp.concatenate([indptr, onp.full(extra_rows, indptr[-1], dtype=onp.int64)])
    pad = nnz_cap - data.size
    if pad > 0:
        data = onp.concatenate([data, onp.zeros(pad)])
        indices = onp.concatenate([indices, onp.zeros(pad, dtype=onp.int32)])
        indptr = indptr.copy()
        indptr[-1] = nnz_cap
    out = sp.csr_matrix((data, indices, indptr), shape=(rows_cap, cols_cap))
    return out


def _pad_dense_inverse(inv: onp.ndarray, cap: int) -> onp.ndarray:
    c = inv.shape[0]
    if c > cap:
        raise _CapacityError("coarse.rows", c)
    out = onp.eye(cap, dtype=onp.float64)
    out[:c, :c] = inv
    return out


def bucket_capacity(nf: int, ratio: float = 1.25, base: int = 4096) -> int:
    """Geometric ladder for ``shape_mode="bucket"``: the smallest capacity
    ``base * ratio**k`` that holds ``nf`` free dofs, rounded to 256.

    TODO (adaptive bucketing, not implemented -- template only):
      * choose ``ratio`` from a host-memory budget: each distinct capacity
        costs one jit shape (~186 MB resident on jax 0.10 / CUDA 12, see
        BUG_FIX.md); ratio 1.25 -> ~15 shapes / ~2.7 GB, 1.5 -> 9 / 1.6 GB
        on the v159 mesh, CG penalty +8% / +11%;
      * anchor the ladder at the first observed ``nf`` and never step down;
      * learn the coarse-level capacities per bucket from the hierarchy
        actually built (record ``hierarchy_info["level_sizes"]``), instead
        of the static ``default_capacities`` ratios;
      * when a bucket overflows, jump straight to the bucket the activation
        rate predicts for the remaining slabs (path CSV knows the future).
    """
    cap = float(base)
    while cap < nf:
        cap *= float(ratio)
    return _round_up(cap, 256)


# ---------------------------------------------------------------------------
# jax V-cycle + PCG (built lazily so the CPU path never imports jax)
# ---------------------------------------------------------------------------
_JAX = None  # module-level default kernel set (compat only; the solver uses its own sets)


def _make_jax_kernels():
    """Build a FRESH set of jit-compiled kernels.

    ``jax.jit`` caches one compiled executable per input-shape signature and
    never evicts, and jax/XLA keep further per-shape state on the host that
    survives ``gc``. Under ``shape_mode="free"`` every activation layer is a
    new shape, so the kernel set is owned by the pattern slot and dropped with
    it (bounds the executables, not the per-shape host residue -- see
    BUG_FIX.md); under ``shape_mode="full"`` the shapes never change and one
    set, owned by the structure slot, lives for the whole build.
    """
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

    def scale_data(data, scale, rows, cols):
        # ``data`` is donated: the scaled operator is written into the buffer
        # of the transferred (free mode) / decoupled (full mode) data, so the
        # prepare stage never holds two nnz-length copies (57.5 M x 8 B = 460 MB
        # each on v159). A fused decouple+scale kernel that regenerated the row
        # index from indptr on the device was measured 560 MB WORSE at peak
        # (three nnz-length int32 temporaries of jnp.repeat), see BUG_FIX 9.2.
        return data * scale[rows] * scale[cols]

    scale_data_jit = jax.jit(scale_data, donate_argnums=(0,))

    @jax.jit
    def residual_norm(S, y, rhs):
        return jnp.linalg.norm(rhs - S @ y)

    def decouple(data, diag_pos, kill, pin_mask):
        # Full-shape system: zero every entry in a pinned row or column, then
        # put 1 on the diagonal of the pinned rows. All shapes fixed per mesh.
        # ``data`` (the fresh host->device transfer) is donated: the where and
        # the diagonal scatter run in place, no second nnz-length buffer.
        data = jnp.where(kill, 0.0, data)
        diag_vals = jnp.where(pin_mask, 1.0, data[diag_pos])
        return data.at[diag_pos].set(diag_vals)

    decouple_jit = jax.jit(decouple, donate_argnums=(0,))

    return {
        "jax": jax, "jnp": jnp, "csr": csr, "CSR": jsparse.CSR,
        "pcg": pcg_jit, "scale_data": scale_data_jit, "residual_norm": residual_norm,
        "decouple": decouple_jit,
    }


def _device_memory_mb() -> Optional[Dict[str, float]]:
    """JAX device allocator figures (MB) for the run log: bytes in use, pool
    size and peak. None when jax/GPU are unavailable."""
    try:
        import jax

        ms = jax.devices()[0].memory_stats() or {}
        return {
            "in_use": ms.get("bytes_in_use", 0) / 1048576.0,
            "pool": ms.get("pool_bytes", 0) / 1048576.0,
            "peak": ms.get("peak_bytes_in_use", 0) / 1048576.0,
            "limit": ms.get("bytes_limit", 0) / 1048576.0,
        }
    except Exception:  # pragma: no cover - CPU-only environments
        return None


def _jax_kernels():
    """Module-level default kernel set (backward compatibility for callers
    outside the solver). The solver itself never uses it."""
    global _JAX
    if _JAX is None:
        _JAX = _make_jax_kernels()
    return _JAX


def _jax_base_module():
    import jax

    return jax


def _process_rss_mb() -> Optional[float]:
    """Resident set size of this process in MB (Linux /proc), else None."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        import os

        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except Exception:  # pragma: no cover - non-Linux or restricted /proc
        return None


class _StructureCache:
    """Whole-mesh sparsity structure for ``shape_mode="full"``: device index
    arrays, the padded capacities and the ONE jit kernel set of the build."""

    __slots__ = (
        "signature", "n", "nnz", "rows", "diag_pos", "caps",
        "dev_rows", "dev_cols", "dev_indptr", "dev_diag_pos", "kernels",
    )

    def __init__(self, signature, n, nnz, rows, diag_pos, caps):
        self.signature = signature
        self.n = int(n)
        self.nnz = int(nnz)
        self.rows = rows
        self.diag_pos = diag_pos
        self.caps = caps
        self.dev_rows = None
        self.dev_cols = None
        self.dev_indptr = None
        self.dev_diag_pos = None
        self.kernels = None

    def release(self) -> None:
        self.kernels = None
        self.dev_rows = None
        self.dev_cols = None
        self.dev_indptr = None
        self.dev_diag_pos = None


class _PatternCache:
    """Free/pinned split and the AMG hierarchy for one sparsity pattern."""

    __slots__ = (
        "signature", "n", "free", "pin", "keep", "ff_indptr", "ff_indices", "ff_rows",
        "ff_diag_pos", "hierarchy", "hierarchy_info", "jax_levels", "jax_coarse_pinv",
        "jax_omega0", "dev_rows", "dev_cols", "dev_indptr", "dev_diag_pos", "fresh_iterations",
        "kernels", "pin_mask", "dev_kill", "dev_pin_mask", "scale_build",
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
        self.fresh_iterations = None
        # Per-pattern jit kernel set (free mode only); released with the pattern.
        self.kernels = None
        # Full mode: pinned-row mask (n) and per-entry kill mask (nnz) on the device.
        self.pin_mask = None
        self.dev_kill = None
        self.dev_pin_mask = None
        # Jacobi scaling the current hierarchy was built with (scale_policy="frozen").
        self.scale_build = None

    def release(self) -> None:
        """Drop device buffers and the jit kernel set so their executables can be freed."""
        self.kernels = None
        self.jax_levels = None
        self.jax_coarse_pinv = None
        self.jax_omega0 = None
        self.dev_rows = None
        self.dev_cols = None
        self.dev_indptr = None
        self.dev_diag_pos = None
        self.dev_kill = None
        self.dev_pin_mask = None
        self.hierarchy = None


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
        max_coarse: int = 3000,
        rebuild: str = "pattern",
        fallback: str = "pardiso",
        pardiso_mode: str = "phase23",
        device: str = "cpu",
        smoother: str = "jacobi",
        smoother_sweeps: int = 2,
        smoother_omega: float = 4.0 / 3.0,
        rebuild_iter_factor: float = 3.0,
        verbose: bool = False,
        clear_jax_caches_on_pattern: bool = False,
        shape_mode: str = "auto",
        fixed_levels: int = 4,
        bucket_ratio: float = 1.25,
        scale_policy: str = "current",
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
        if shape_mode not in SHAPE_MODES:
            raise ValueError(f"shape_mode must be one of {SHAPE_MODES}, got {shape_mode!r}")
        self.use_jax = device in ("gpu", "jax")
        if self.use_jax and smoother != "jacobi":
            raise ValueError("the gpu/jax V-cycle supports the jacobi smoother only")
        if self.use_jax and method != "cg":
            raise ValueError("the gpu/jax path implements cg only (the free block is symmetric)")
        if int(smoother_sweeps) < 1:
            raise ValueError("smoother_sweeps must be >= 1")
        if int(fixed_levels) < 2:
            raise ValueError("fixed_levels must be >= 2")
        if float(bucket_ratio) <= 1.0:
            raise ValueError("bucket_ratio must be > 1")
        if scale_policy not in SCALE_POLICIES:
            raise ValueError(f"scale_policy must be one of {SCALE_POLICIES}, got {scale_policy!r}")
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
        # Adaptive staleness rule: a reused hierarchy whose solve needs more than
        # rebuild_iter_factor x the iterations of the fresh-hierarchy solve is
        # dropped, so the next call rebuilds it (0 disables). In the 40-slab run
        # the tangent drifts within a slab (elastic predictor -> plastic, then
        # cooling): reused/fresh median 1.4, 90% 2.4, a few solves hit maxiter.
        self.rebuild_iter_factor = float(rebuild_iter_factor)
        self.verbose = bool(verbose)
        # Last-resort switch: also wipe JAX's global compilation caches on every
        # pattern change. Frees everything (including the assembly kernels of the
        # acceleration wrapper, which then recompile), so it is off by default.
        self.clear_jax_caches_on_pattern = bool(clear_jax_caches_on_pattern)
        # Shape policy of the jit path; the CPU path has no shapes to fix.
        if shape_mode == "auto":
            shape_mode = "full" if self.use_jax else "free"
        self.shape_mode = shape_mode if self.use_jax else "free"
        self.fixed_levels = int(fixed_levels)
        self.bucket_ratio = float(bucket_ratio)
        # Jacobi scaling while a hierarchy is REUSED. The prolongator interpolates
        # the near-nullspace in the scaling it was built with (B / s_build); with
        # "current" the system is rescaled by the diagonal of every new tangent, so
        # after plastic softening the coarse correction no longer carries the
        # rigid-body modes of the softened region. "frozen" keeps s_build until the
        # hierarchy is rebuilt (equivalent to scaling the rows of P by
        # s_build / s_current). Measured on the v159 tangent-drift series: layer 30
        # reused solves 128 -> 77 iterations for 0.025 s, a full rebuild gives 73-84
        # (BUG_FIX.md section 11).
        self.scale_policy = scale_policy
        smoother_label = (
            f"{smoother}x{self.smoother_sweeps}" if smoother == "jacobi" else "block_gs(sym)"
        )
        shape_label = f", shape={self.shape_mode}" if self.use_jax else ""
        self.label = (
            f"pyamg_solver(sa+{near_nullspace}{'+scaled' if self.scaled else ''}"
            f"{'(frozen)' if self.scaled and scale_policy == 'frozen' else ''}, "
            f"{method}, {smoother_label}, {self.device}{shape_label}, tol={self.tol:g}, "
            f"maxiter={self.maxiter}, fallback={fallback})"
        )
        self._points: Optional[onp.ndarray] = None
        self._vec: Optional[int] = None
        self._cache: Optional[_PatternCache] = None
        self._struct: Optional[_StructureCache] = None
        self._rbm_warned = False
        self.stats: Dict[str, Any] = {
            "device": self.device,
            "shape_mode": self.shape_mode,
            "calls": 0,
            "iterations": 0,
            "setup_s": 0.0,
            "solve_s": 0.0,
            "pattern_rebuilds": 0,
            "structure_rebuilds": 0,
            "hierarchy_rebuilds": 0,
            "adaptive_rebuilds": 0,
            "frozen_scale_solves": 0,
            "capacity_growths": 0,
            "fallbacks": 0,
            "last_iterations": 0,
            "last_rel_res": None,
            "last_solve_s": None,
            "last_levels": None,
            "last_operator_complexity": None,
            "compiled_variants": 0,
            "rss_mb": _process_rss_mb(),
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

    def release_device(self) -> None:
        """Drop every device buffer and jit kernel set (pattern and structure
        slots). The next solve rebuilds them -- pattern, hierarchy, one
        recompilation -- so call this only before a solve that is routed away
        from this solver (the raft release goes to PARDISO). Until then the
        full-shape structure, the padded hierarchy and the scaled data
        (~2.4 GB on v159) would sit on the device next to the second
        mechanics problem's assembly, which is what ran the 2026-09-15
        shakedown out of device memory at the release step."""
        if self._cache is not None:
            self._cache.release()
            self._cache = None
        if self._struct is not None:
            self._struct.release()
            self._struct = None
        import gc

        gc.collect()
        self.stats["device_releases"] = int(self.stats.get("device_releases", 0)) + 1
        self.stats["rss_mb"] = _process_rss_mb()

    # -- kernel lifetime ---------------------------------------------------------------
    @property
    def _full(self) -> bool:
        return self.use_jax and self.shape_mode == "full"

    def _kernels_for(self, owner):
        """Jit kernel set owned by ``owner`` (the pattern slot in free mode, the
        structure slot in full mode); created on first use."""
        if owner.kernels is None:
            owner.kernels = _make_jax_kernels()
            self.stats["compiled_variants"] += 1
        return owner.kernels

    def _replace_cache(self, new_cache: _PatternCache) -> None:
        """Install a new pattern slot and free the previous one's device buffers
        (and, in free mode, its executables)."""
        old = self._cache
        self._cache = new_cache
        if old is not None:
            old.release()
            del old
            import gc

            gc.collect()
            if self.clear_jax_caches_on_pattern and self.use_jax:
                _jax_base_module().clear_caches()
        self.stats["rss_mb"] = _process_rss_mb()

    def _replace_struct(self, new_struct: _StructureCache) -> None:
        old = self._struct
        self._struct = new_struct
        if old is not None:
            old.release()
            del old
            import gc

            gc.collect()
        self.stats["structure_rebuilds"] += 1

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

    @staticmethod
    def _structure_signature(n, indptr, indices):
        return (
            int(n),
            int(indptr[-1]),
            hash(onp.ascontiguousarray(indptr[::4093]).tobytes()),
            hash(onp.ascontiguousarray(indices[::9973]).tobytes()),
        )

    def _build_structure(self, signature, n, indptr, indices) -> Optional[_StructureCache]:
        rows = onp.repeat(onp.arange(n, dtype=onp.int64), onp.diff(indptr))
        diag_pos = onp.flatnonzero(rows == indices)
        if diag_pos.size != n:
            return None  # a row without a stored diagonal: full mode cannot decouple it
        caps = default_capacities(n, self.fixed_levels, self.max_coarse)
        return _StructureCache(
            signature, n, int(indptr[-1]), rows.astype(onp.int32, copy=False),
            diag_pos.astype(onp.int32, copy=False), caps,
        )

    def _build_pattern(self, signature, n, indptr, indices, pinned) -> _PatternCache:
        free = onp.flatnonzero(~pinned)
        pin = onp.flatnonzero(pinned)
        free_map = onp.full(n, -1, dtype=onp.int64)
        free_map[free] = onp.arange(free.size, dtype=onp.int64)
        if self._struct is not None and self._struct.n == n:
            rows = self._struct.rows
        else:
            rows = onp.repeat(onp.arange(n, dtype=onp.int64), onp.diff(indptr))
        keep = (~pinned[rows]) & (~pinned[indices])
        ff_rows = free_map[rows[keep]].astype(onp.int32, copy=False)
        ff_indices = free_map[indices[keep]].astype(onp.int32, copy=False)
        counts = onp.bincount(ff_rows, minlength=free.size)
        ff_indptr = onp.concatenate([[0], onp.cumsum(counts)]).astype(onp.int64)
        diag_pos = onp.flatnonzero(ff_rows == ff_indices)
        if diag_pos.size != free.size:
            diag_pos = None  # a free row without a stored diagonal: fall back to scipy
        cache = _PatternCache(signature, n, free, pin, keep, ff_indptr, ff_indices, ff_rows, diag_pos)
        cache.pin_mask = pinned
        return cache

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
        # pyamg stops coarsening when dofs / blocksize <= max_coarse, and coarse
        # levels carry blocksize = number of near-nullspace candidates; divide so
        # that self.max_coarse is (approximately) a bound in dofs.
        n_candidates = int(B.shape[1]) if B is not None else 1
        max_coarse_blocks = max(10, self.max_coarse // n_candidates)
        extra = {"max_levels": self.fixed_levels} if self._full else {}
        ml = pyamg.smoothed_aggregation_solver(
            S, B=B, max_coarse=max_coarse_blocks, presmoother=sm, postsmoother=sm, **extra
        )
        t_pyamg = time.perf_counter() - t0
        cache.hierarchy = ml
        cache.hierarchy_info = {
            "levels": len(ml.levels),
            "operator_complexity": float(ml.operator_complexity()),
            "near_nullspace": "rigid_body" if B is not None else "constant",
        }
        if self.use_jax:
            if self._full:
                self._to_jax_hierarchy_full(cache, ml)
            else:
                self._to_jax_hierarchy(cache, ml)
        setup = time.perf_counter() - t0
        self.stats["setup_s"] += setup
        self.stats["last_levels"] = cache.hierarchy_info["levels"]
        self.stats["last_operator_complexity"] = cache.hierarchy_info["operator_complexity"]
        cache.hierarchy_info["level_sizes"] = [int(lvl.A.shape[0]) for lvl in ml.levels]
        cache.hierarchy_info["setup_pyamg_s"] = t_pyamg
        cache.hierarchy_info["setup_device_s"] = setup - t_pyamg
        dev = _device_memory_mb() if self.use_jax else None
        if dev is not None:
            cache.hierarchy_info["device_mb"] = dev
        if self.verbose:
            padded = cache.hierarchy_info.get("padded_levels")
            print(
                f"pyamg[{self.device}]: hierarchy built in {setup:.2f}s "
                f"(pyamg {t_pyamg:.2f}s, device/coarse {setup - t_pyamg:.2f}s) -- "
                f"levels {cache.hierarchy_info['level_sizes']}"
                + (f" padded to {padded}" if padded else "")
                + f", operator complexity {ml.operator_complexity():.3f}, "
                f"near-nullspace {cache.hierarchy_info['near_nullspace']}, smoother {sm[0]}"
                + (f", device in_use {dev['in_use']:.0f} / pool {dev['pool']:.0f} / peak {dev['peak']:.0f} "
                   f"/ limit {dev['limit']:.0f} MB" if dev else ""),
                flush=True,
            )
        return ml

    def _to_jax_hierarchy(self, cache: _PatternCache, ml) -> None:
        """Free mode: device operators in the compacted free-block shapes."""
        from pyamg.relaxation.smoothing import rho_D_inv_A

        K = self._kernels_for(cache)
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
        cache.jax_coarse_pinv = jnp.asarray(_dense_inverse(ml.levels[-1].A.toarray()))
        if cache.dev_cols is None:
            cache.dev_rows = jnp.asarray(cache.ff_rows)
            cache.dev_cols = jnp.asarray(cache.ff_indices)
            cache.dev_indptr = jnp.asarray(cache.ff_indptr.astype(onp.int32, copy=False))
            cache.dev_diag_pos = (
                jnp.asarray(cache.ff_diag_pos.astype(onp.int32, copy=False))
                if cache.ff_diag_pos is not None else None
            )

    def _to_jax_hierarchy_full(self, cache: _PatternCache, ml) -> None:
        """Full mode: the free-block hierarchy embedded into fixed-capacity
        operators on the whole-mesh shape, with a fixed level count.

        Levels pyamg did not produce (a shallow hierarchy early in the build)
        are padded by duplicating the coarsest level with ``P = R = I``: the
        exact coarse solve then happens one level down on the same operator,
        which is the same preconditioner. Capacities that turn out too small
        grow once (x1.5) and the kernel set is rebuilt (one recompilation).
        """
        from pyamg.relaxation.smoothing import rho_D_inv_A

        st = self._struct
        n = st.n
        L = self.fixed_levels
        free = cache.free
        # Drop the previous device hierarchy BEFORE building the new one: the
        # padded operators are ~1 GB on v159 and a hierarchy is rebuilt every
        # few solves, so holding old and new at once doubles the transient
        # device demand. The 3rd full-height run (2026-09-15) sat at the JAX
        # device limit from layer ~80 on and froze in the WSL dxg allocation
        # path during exactly such a rebuild (BUG_FIX.md section 9.2).
        cache.jax_levels = None
        cache.jax_coarse_pinv = None
        cache.jax_omega0 = None
        As: List[sp.csr_matrix] = [lvl.A.tocsr() for lvl in ml.levels]
        Ps: List[sp.csr_matrix] = [lvl.P.tocsr() for lvl in ml.levels[:-1]]
        Rs: List[sp.csr_matrix] = [lvl.R.tocsr() for lvl in ml.levels[:-1]]
        if len(As) > L:
            raise RuntimeError(
                f"pyamg produced {len(As)} levels but fixed_levels={L}; raise fixed_levels"
            )
        while len(As) < L:
            c = As[-1].shape[0]
            Ps.append(sp.identity(c, format="csr"))
            Rs.append(sp.identity(c, format="csr"))
            As.append(As[-1])
        # level-0 transfer operators: rows/cols of the free block -> global dofs
        P0 = Ps[0].tocoo()
        P0_full = sp.csr_matrix((P0.data, (free[P0.row], P0.col)), shape=(n, Ps[0].shape[1]))
        R0 = Rs[0].tocoo()
        R0_full = sp.csr_matrix((R0.data, (R0.row, free[R0.col])), shape=(Rs[0].shape[0], n))
        omegas = []
        for l in range(L - 1):
            rho = float(rho_D_inv_A(As[l]))
            omegas.append(self.smoother_omega / rho)
        coarse_inv = _dense_inverse(As[-1].toarray())

        while True:
            caps = st.caps
            try:
                K = self._kernels_for(st)
                jnp = K["jnp"]
                levels = [None] * (L - 1)
                levels[0] = (
                    None, None, jnp.float64(omegas[0]),
                    K["csr"](_pad_csr(P0_full, n, caps["rows1"], caps["P0"], "P0")),
                    K["csr"](_pad_csr(R0_full, caps["rows1"], n, caps["P0"], "R0")),
                )
                for l in range(1, L - 1):
                    rc, cc = caps[f"rows{l}"], caps[f"rows{l + 1}"]
                    A_pad = _pad_csr(As[l], rc, rc, caps[f"A{l}"], f"A{l}", identity_pad=True)
                    diag = As[l].diagonal()
                    safe = onp.where(diag != 0.0, diag, 1.0)
                    dinv = onp.ones(rc)
                    dinv[: diag.size] = onp.where(diag != 0.0, 1.0 / safe, 0.0)
                    levels[l] = (
                        K["csr"](A_pad), jnp.asarray(dinv), jnp.float64(omegas[l]),
                        K["csr"](_pad_csr(Ps[l], rc, cc, caps[f"P{l}"], f"P{l}")),
                        K["csr"](_pad_csr(Rs[l], cc, rc, caps[f"P{l}"], f"R{l}")),
                    )
                coarse = jnp.asarray(_pad_dense_inverse(coarse_inv, caps[f"rows{L - 1}"]))
                break
            except _CapacityError as err:
                kind, needed = err.kind, err.needed
                key = {"P0.nnz": "P0", "R0.nnz": "P0", "coarse.rows": f"rows{L - 1}"}.get(kind)
                if key is None:
                    base, what = kind.split(".")
                    if what == "nnz":
                        key = base if base.startswith(("A", "P")) else base
                        key = "P" + base[1:] if base.startswith("R") else key
                    elif base in ("P0", "R0"):
                        key = "rows1"
                    else:
                        level = int(base[1:])
                        key = f"rows{level + 1}" if (base.startswith("P") and what == "cols") or (base.startswith("R") and what == "rows") else f"rows{level}"
                new = _round_up(1.5 * needed, 4096 if key.startswith(("A", "P")) else 256)
                warnings.warn(
                    f"pyamg full-shape capacity {key} too small ({caps.get(key)} < {needed}); "
                    f"growing to {new} (one recompilation)",
                    RuntimeWarning,
                )
                caps[key] = max(int(caps.get(key, 0)), new)
                self.stats["capacity_growths"] += 1
                st.kernels = None  # fresh jit objects for the new shapes
        cache.jax_omega0 = levels[0][2]
        cache.jax_levels = levels
        cache.jax_coarse_pinv = coarse
        cache.hierarchy_info["padded_levels"] = [n] + [caps[f"rows{l}"] for l in range(1, L)]
        if st.dev_cols is None:
            st.dev_rows = jnp.asarray(st.rows)
            st.dev_cols = jnp.asarray(onp.asarray(self._last_indices, dtype=onp.int32))
            st.dev_indptr = jnp.asarray(onp.asarray(self._last_indptr, dtype=onp.int32))
            st.dev_diag_pos = jnp.asarray(st.diag_pos)

    # -- fallback -------------------------------------------------------------------
    def _direct_fallback(self, A, b, x0, linear_options, reason: str):
        """Hand one system to PARDISO and keep nothing.

        The shared phase23 adapter (``shared_pardiso_solver``) retains the
        symbolic+numeric factorisation per sparsity pattern for reuse. On the
        0.94 M-dof v159 mechanics system that is ~8 GB of host memory, and two
        consecutive fallbacks at layer 120 of the 2026-09-14 full-height run
        pushed the WSL guest over its 40 GB cap and froze the process
        (BUG_FIX.md section 8). The fallback here is a one-shot phase-13
        solve followed by phase -1 (release), so the only cost is time.
        """
        self.stats["fallbacks"] += 1
        if self.fallback != "pardiso":
            raise RuntimeError(f"pyamg solver failed: {reason}")
        print(
            f"WARNING: pyamg solver {reason}; solving this system with PARDISO "
            "(one-shot, factorisation released afterwards)",
            flush=True,
        )
        from jax_fem_am.solvers.pardiso import pardiso_solve_once

        t0 = time.perf_counter()
        x = pardiso_solve_once(A, b)
        self.stats["fallback_s"] = self.stats.get("fallback_s", 0.0) + (time.perf_counter() - t0)
        self.stats["rss_mb"] = _process_rss_mb()
        return x

    # -- krylov drivers ---------------------------------------------------------------
    def _solve_cpu(self, cache, S, rhs, y0, rhs_norm):
        ml = cache.hierarchy
        y, info, iters = _krylov_cpu(
            self.method, S, rhs, y0, ml.aspreconditioner(cycle="V"), self.tol, self.maxiter
        )
        rel = float(onp.linalg.norm(rhs - S @ y) / rhs_norm)
        return y, iters, rel, info

    def _solve_jax(self, cache, data_ff, scale, d, rhs, y0, rhs_norm):
        K = self._kernels_for(cache)
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

    def _solve_jax_full(self, cache, data, scale_full, rhs_full, y0_full, rhs_norm):
        """Full mode: one Krylov solve on the decoupled whole-mesh system."""
        st = self._struct
        K = self._kernels_for(st)
        jnp = K["jnp"]
        n = st.n
        if cache.dev_kill is None:
            cache.dev_kill = jnp.asarray(~cache.keep)
            cache.dev_pin_mask = jnp.asarray(cache.pin_mask)
        # both kernels donate their data argument: decouple runs in place on
        # the transfer buffer, scale_data in place on the decoupled buffer
        data_dev = K["decouple"](jnp.asarray(data), st.dev_diag_pos, cache.dev_kill, cache.dev_pin_mask)
        if scale_full is not None:
            data_dev = K["scale_data"](data_dev, jnp.asarray(scale_full), st.dev_rows, st.dev_cols)
        S_dev = K["CSR"]((data_dev, st.dev_cols, st.dev_indptr), shape=(n, n))
        dinv0 = 1.0 / data_dev[st.dev_diag_pos]
        levels = list(cache.jax_levels)
        _, _, omega0, P0, R0 = levels[0]
        levels[0] = (S_dev, dinv0, omega0, P0, R0)
        rhs_dev = jnp.asarray(rhs_full)
        y0_dev = jnp.asarray(y0_full)
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
        if self.shape_mode == "bucket":
            raise NotImplementedError(
                "pyamg shape_mode='bucket' is a template (see amg.bucket_capacity); "
                "use shape_mode='full' or 'free'"
            )
        Afull = sp.csr_matrix((data, indices, indptr), shape=(n, n))
        pinned = pinned_rows(Afull)

        full = self._full
        if full:
            sig_s = self._structure_signature(n, indptr, indices)
            if self._struct is None or self._struct.signature != sig_s:
                st = self._build_structure(sig_s, n, indptr, indices)
                if st is None:
                    warnings.warn(
                        "pyamg full shape mode needs a stored diagonal in every row; "
                        "falling back to shape_mode='free' for this solver",
                        RuntimeWarning,
                    )
                    self.shape_mode = "free"
                    self.stats["shape_mode"] = "free"
                    full = False
                else:
                    self._replace_struct(st)
                    self._cache = None
            self._last_indptr = indptr
            self._last_indices = indices

        signature = self._signature(n, indptr, indices, pinned)
        cache = self._cache
        if cache is None or cache.signature != signature:
            cache = self._build_pattern(signature, n, indptr, indices, pinned)
            self._replace_cache(cache)
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
        def scaled_vectors(scale):
            rhs = scale * bf if scale is not None else bf
            y0 = x0[free] / scale if scale is not None else x0[free]
            if not full:
                return rhs, y0, None, None, None
            # whole-mesh vectors: pinned rows carry zero residual and a zero
            # unknown (their prescribed values are added back at the end).
            scale_full = onp.ones(n) if scale is not None else None
            if scale_full is not None:
                scale_full[free] = scale
            rhs_full = onp.zeros(n)
            rhs_full[free] = rhs
            y0_full = onp.zeros(n)
            y0_full[free] = y0
            return rhs, y0, scale_full, rhs_full, y0_full

        reused = cache.hierarchy is not None
        if self.scaled:
            scale = 1.0 / onp.sqrt(d)
            if self.scale_policy == "frozen" and reused and cache.scale_build is not None:
                # the hierarchy interpolates B / scale_build: keep that scaling
                scale = cache.scale_build
                self.stats["frozen_scale_solves"] += 1
            D = sp.diags(scale)
            S = (D @ Aff @ D).tocsr() if (not self.use_jax or not reused) else None
        else:
            scale = None
            S = Aff
        rhs, y0, scale_full, rhs_full, y0_full = scaled_vectors(scale)
        rhs_norm = float(onp.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return x_full

        if not reused:
            self._build_hierarchy(cache, S, scale)
            cache.scale_build = scale
        converged = False
        y = y0
        for attempt in range(2):
            t0 = time.perf_counter()
            if full:
                y, iters, rel, info = self._solve_jax_full(cache, data, scale_full, rhs_full, y0_full, rhs_norm)
            elif self.use_jax:
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
            if converged:
                if not reused or attempt == 1:
                    cache.fresh_iterations = iters
                elif (
                    self.rebuild_iter_factor > 0
                    and cache.fresh_iterations
                    and iters > self.rebuild_iter_factor * cache.fresh_iterations
                ):
                    # stale hierarchy: drop it so the next call rebuilds
                    cache.hierarchy = None
                    self.stats["adaptive_rebuilds"] += 1
                    if self.verbose:
                        print(
                            f"pyamg[{self.device}]: {iters} iterations > "
                            f"{self.rebuild_iter_factor:g} x fresh {cache.fresh_iterations}; "
                            "hierarchy marked for rebuild",
                            flush=True,
                        )
                break
            if not reused or attempt == 1:
                break
            # The cached hierarchy no longer matches the tangent: rebuild once.
            self.stats["hierarchy_rebuilds"] += 1
            if self.scaled and self.scale_policy == "frozen":
                # a rebuild re-freezes the scaling on the current diagonal
                scale = 1.0 / onp.sqrt(d)
                rhs, y0, scale_full, rhs_full, y0_full = scaled_vectors(scale)
                rhs_norm = float(onp.linalg.norm(rhs))
                S = None
            if S is None:
                S = (sp.diags(scale) @ Aff @ sp.diags(scale)).tocsr()
            self._build_hierarchy(cache, S, scale)
            cache.scale_build = scale
        if not converged:
            return self._direct_fallback(
                A, b, x0, linear_options,
                f"stalled (rel_res {self.stats['last_rel_res']:.2e} after "
                f"{self.stats['last_iterations']} {self.method} iterations)",
            )
        if full:
            x_full = y * scale_full if scale_full is not None else y
            x_full[pin] = b[pin]
        else:
            x_full[free] = y * scale if scale is not None else y
        if self.verbose:
            print(f"pyamg[{self.device}]: solve wall {time.perf_counter() - t_call:.2f}s", flush=True)
        return x_full
