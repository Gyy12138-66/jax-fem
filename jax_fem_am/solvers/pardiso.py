# coding: utf-8
"""V07 experimental PARDISO solver variants for the ablation study.

Selected via the ``V07_PARDISO_MODE`` environment variable read by the v04
wrapper's ``_PardisoCustomSolver``. Modes form a ladder where each level adds
exactly one optimization on top of the previous, so bench deltas attribute
cleanly:

    base       -- (env unset) untouched v04 path: pypardiso.spsolve per call
                  (matrix copy + phase 12 + full compare + phase 33 + double
                  ia/ja conversion)
    nocmp      -- bypass pypardiso bookkeeping: one phase-13 call per solve,
                  fresh 1-based ia/ja each call
    cache-idx  -- nocmp + reuse the 1-based int32 ia/ja across calls while the
                  sparsity pattern is unchanged (exact indptr/indices check)
    phase23    -- cache-idx + reuse the symbolic factorization: phase 13 on
                  the first solve of a pattern, phase 23 afterwards
    fp32ir     -- phase23 with single-precision factorization (iparm(28)=1)
                  plus manual float64 iterative refinement to recover
                  direct-solver accuracy

Multiple PARDISO internal handles are kept per sparsity pattern (thermal and
mechanics systems have different sizes), keyed by (n, nnz).

Outside the ladder, ``free-spd`` (``FreeBlockSPDSolver``) solves only the free
block of the mechanics system with the symmetric positive-definite type
(mtype=2, Cholesky on the upper triangle); see its docstring.
"""

from __future__ import annotations

import atexit
import ctypes
import sys
import time

import numpy as np
import scipy.sparse

_LADDER = ("nocmp", "cache-idx", "phase23", "fp32ir")

# fp32ir manual iterative refinement controls.
_IR_MAX_STEPS = 10
_IR_RTOL = 1e-13

# Keep at most this many per-pattern PARDISO handles (thermal + mechanics
# plus headroom for activation-driven pattern changes).
_MAX_STATES = 4


class _RawPardiso:
    """Direct MKL PARDISO caller reusing pypardiso's library discovery.

    pypardiso's ``_call_pardiso`` rebuilds the 1-based index arrays on every
    call and its ``solve``/``factorize`` bookkeeping copies and compares the
    full matrix; this class calls the raw entry point with caller-owned
    arrays instead.
    """

    def __init__(self, single_precision: bool = False):
        from pypardiso import PyPardisoSolver

        # Borrow library handle + ctypes signature from a throwaway instance.
        inner = PyPardisoSolver(mtype=11)
        self._mkl_pardiso = inner._mkl_pardiso
        self._pt_type = inner._pt_type
        self.pt = np.zeros(64, dtype=inner._pt_type[1])
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)
        self.mtype = 11
        self.single = bool(single_precision)
        if self.single:
            # iparm(1)=0 would make MKL ignore all inputs, so switching on
            # iparm(28) (single precision) requires spelling out the defaults
            # for mtype=11 explicitly (MKL PARDISO iparm reference).
            ip = self.iparm
            ip[0] = 1    # iparm(1): user-supplied iparm
            ip[1] = 2    # iparm(2): METIS nested dissection
            ip[7] = 0    # iparm(8): no internal refinement (done manually)
            ip[9] = 13   # iparm(10): pivot perturbation 1e-13
            ip[10] = 1   # iparm(11): scaling (default for mtype=11)
            ip[12] = 1   # iparm(13): weighted matching (default for mtype=11)
            ip[17] = -1  # iparm(18): report nnz(L+U)
            ip[27] = 1   # iparm(28): single-precision factorization
            ip[34] = 0   # iparm(35): 1-based indexing

    def call(self, n, a_data, ia, ja, b, phase, transpose=False):
        """One raw pardiso call; b must be float64/float32 to match `single`.

        With ``transpose=True`` the call solves A^T x = b using the SAME
        factorization of A (MKL iparm(12)=2), so an adjoint solve costs one
        backsolve instead of a fresh factorization of the transposed matrix.
        """
        if transpose:
            self.iparm[11] = 2  # iparm(12): solve transposed system
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        void_p = ctypes.c_void_p
        self._mkl_pardiso(
            self.pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),
            ctypes.byref(ctypes.c_int32(1)),            # maxfct
            ctypes.byref(ctypes.c_int32(1)),            # mnum
            ctypes.byref(ctypes.c_int32(self.mtype)),   # mtype
            ctypes.byref(ctypes.c_int32(phase)),        # phase
            ctypes.byref(ctypes.c_int32(n)),            # n
            void_p(a_data.ctypes.data),                 # a
            ia.ctypes.data_as(c_int32_p),               # ia (1-based)
            ja.ctypes.data_as(c_int32_p),               # ja (1-based)
            self.perm.ctypes.data_as(c_int32_p),        # perm
            ctypes.byref(ctypes.c_int32(1)),            # nrhs
            self.iparm.ctypes.data_as(c_int32_p),       # iparm
            ctypes.byref(ctypes.c_int32(0)),            # msglvl
            void_p(b.ctypes.data),                      # b
            void_p(x.ctypes.data),                      # x
            ctypes.byref(err),
        )
        if transpose:
            self.iparm[11] = 0
        if err.value != 0:
            raise RuntimeError(
                f"MKL PARDISO failed with error {err.value} (phase {phase})"
            )
        return x

    def release(self):
        try:
            dummy = np.zeros(0)
            ia = np.ones(1, dtype=np.int32)
            ja = np.zeros(0, dtype=np.int32)
            self.call(0, dummy, ia, ja, dummy, -1)
        except Exception:
            pass


def pardiso_solve_once(A, b, single_precision: bool = False):
    """Factorise, solve and release in one go (phase 13, then phase -1).

    For fallback paths that must not keep a factorisation alive: the shared
    phase23 adapters retain ~8 GB on the v159 mechanics system, which is what
    froze the 2026-09-14 MODE=5 run (BUG_FIX.md section 8). Nothing is cached
    across calls, so the only cost is the analysis time of one solve.
    """
    indptr, indices, data = A.getValuesCSR()
    n = int(indptr.shape[0] - 1)
    ia = np.asarray(indptr).astype(np.int32)
    ia += 1
    ja = np.asarray(indices).astype(np.int32)
    ja += 1
    values = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    rhs = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    raw = _RawPardiso(single_precision=single_precision)
    try:
        return raw.call(n, values, ia, ja, rhs, phase=13)
    finally:
        raw.release()


class _PatternState:
    """Cached per-sparsity-pattern data: index arrays + PARDISO handle."""

    __slots__ = ("indptr", "indices", "ia", "ja", "raw", "analyzed", "data")

    def __init__(self, indptr, indices, single):
        self.indptr = indptr
        self.indices = indices
        self.ia = indptr.astype(np.int32)
        self.ia += 1
        self.ja = indices.astype(np.int32)
        self.ja += 1
        self.raw = _RawPardiso(single_precision=single)
        self.analyzed = False
        self.data = None  # values of the currently factorized matrix


class VariantSolver:
    """Callable used by the v04 wrapper when V07_PARDISO_MODE is set."""

    def __init__(self, mode: str):
        mode = mode.strip()
        if mode not in _LADDER:
            raise ValueError(
                f"V07_PARDISO_MODE={mode!r}; expected one of {_LADDER}"
            )
        self.mode = mode
        self.level = _LADDER.index(mode)
        self.label = f"pardiso_v07({mode})"
        self._states = {}  # (n, nnz) -> _PatternState
        self._stats = {
            "solves": 0,
            "analyze_calls": 0,
            "pattern_rebuilds": 0,
            "backsolve_hits": 0,
            "transpose_backsolves": 0,
            "ir_steps_total": 0,
            "ir_steps_max": 0,
        }
        for phase in (12, 13, 22, 23, 33):
            self._stats[f"phase{phase}_calls"] = 0
            self._stats[f"phase{phase}_seconds"] = 0.0
        atexit.register(self._report)

    def _report(self):
        print(f"[v07-pardiso {self.mode}] {self._stats}", file=sys.stderr)

    def stats_snapshot(self):
        """Return phase counters/timings without waiting for the atexit log."""
        out = dict(self._stats)
        out["numeric_factorizations"] = sum(
            int(out[f"phase{phase}_calls"])
            for phase in (12, 13, 22, 23)
        )
        # These phases include a factorization, but phase 13 also includes
        # analysis+solve and phase 23 includes solve. Do not label their wall
        # sum as pure numeric-factorization time.
        out["factorization_phase_seconds"] = sum(
            float(out[f"phase{phase}_seconds"])
            for phase in (12, 13, 22, 23)
        )
        out["phase33_backsolve_seconds"] = float(out["phase33_seconds"])
        return out

    def _call(self, raw, n, data, ia, ja, rhs, *, phase,
              transpose=False):
        t0 = time.perf_counter()
        try:
            return raw.call(
                n, data, ia, ja, rhs, phase=phase, transpose=transpose
            )
        finally:
            self._stats[f"phase{phase}_calls"] += 1
            self._stats[f"phase{phase}_seconds"] += time.perf_counter() - t0

    def _get_state(self, n, indptr, indices):
        key = (n, indices.shape[0])
        st = self._states.get(key)
        if st is not None and (
            np.array_equal(st.indptr, indptr)
            and np.array_equal(st.indices, indices)
        ):
            return st
        if st is not None:
            st.raw.release()
            self._stats["pattern_rebuilds"] += 1
        if len(self._states) >= _MAX_STATES and key not in self._states:
            _, old = self._states.popitem()
            old.raw.release()
        st = _PatternState(indptr, indices, single=(self.mode == "fp32ir"))
        self._states[key] = st
        return st

    def __call__(self, A, b, x0=None, linear_options=None):
        indptr, indices, data = A.getValuesCSR()
        rhs = np.asarray(b, dtype=np.float64)
        n = indptr.shape[0] - 1
        self._stats["solves"] += 1

        if self.mode == "nocmp":
            # Fresh index conversion every call, single phase-13 solve.
            ia = indptr.astype(np.int32)
            ia += 1
            ja = indices.astype(np.int32)
            ja += 1
            raw = self._raw_singleton()
            self._stats["analyze_calls"] += 1
            return self._call(raw, n, data, ia, ja, rhs, phase=13)

        st = self._get_state(n, indptr, indices)

        if self.mode == "cache-idx":
            self._stats["analyze_calls"] += 1
            return self._call(
                st.raw, n, data, st.ia, st.ja, rhs, phase=13
            )

        if self.mode == "phase23":
            if not st.analyzed:
                self._stats["analyze_calls"] += 1
                x = self._call(
                    st.raw, n, data, st.ia, st.ja, rhs, phase=13
                )
                st.analyzed = True
                st.data = data.copy()
                return x
            # Unchanged values (e.g. constant-matrix time stepping): the
            # existing factorization is still valid, backsolve only.
            if st.data is not None and np.array_equal(st.data, data):
                self._stats["backsolve_hits"] += 1
                return self._call(
                    st.raw, n, data, st.ia, st.ja, rhs, phase=33
                )
            x = self._call(
                st.raw, n, data, st.ia, st.ja, rhs, phase=23
            )
            st.data = data.copy()
            return x

        # fp32ir: single-precision factorize+solve, float64 refinement.
        data32 = data.astype(np.float32)
        rhs32 = rhs.astype(np.float32)
        if not st.analyzed:
            self._stats["analyze_calls"] += 1
            x32 = self._call(
                st.raw, n, data32, st.ia, st.ja, rhs32, phase=13
            )
            st.analyzed = True
        else:
            x32 = self._call(
                st.raw, n, data32, st.ia, st.ja, rhs32, phase=23
            )

        A64 = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(n, n), copy=False
        )
        x = x32.astype(np.float64)
        bnorm = np.linalg.norm(rhs)
        tol = _IR_RTOL * bnorm if bnorm > 0 else _IR_RTOL
        steps = 0
        for _ in range(_IR_MAX_STEPS):
            r = rhs - A64 @ x
            if np.linalg.norm(r) <= tol:
                break
            dx32 = self._call(
                st.raw, n, data32, st.ia, st.ja,
                r.astype(np.float32), phase=33
            )
            x += dx32.astype(np.float64)
            steps += 1
        self._stats["ir_steps_total"] += steps
        self._stats["ir_steps_max"] = max(self._stats["ir_steps_max"], steps)
        return x

    def solve_transposed(self, A, b):
        """Solve A^T x = b reusing A's factorization (phase23 mode only).

        Intended for adjoint solves: after the forward solve factorized A,
        this is a single transposed backsolve (iparm(12)) — no explicit
        transpose matrix, no second factorization. If A was not factorized
        yet (or its values changed), it is factorized here first, which also
        warms the cache for subsequent forward solves.
        """
        if self.mode != "phase23":
            raise RuntimeError("solve_transposed requires mode=phase23")
        indptr, indices, data = A.getValuesCSR()
        rhs = np.asarray(b, dtype=np.float64)
        n = indptr.shape[0] - 1
        st = self._get_state(n, indptr, indices)
        if (not st.analyzed or st.data is None
                or not np.array_equal(st.data, data)):
            if not st.analyzed:
                self._stats["analyze_calls"] += 1
                self._call(
                    st.raw, n, data, st.ia, st.ja,
                    np.zeros_like(rhs), phase=12
                )
                st.analyzed = True
            else:
                self._call(
                    st.raw, n, data, st.ia, st.ja,
                    np.zeros_like(rhs), phase=22
                )
            st.data = data.copy()
        self._stats["transpose_backsolves"] += 1
        return self._call(
            st.raw, n, data, st.ia, st.ja, rhs,
            phase=33, transpose=True
        )

    def _raw_singleton(self):
        st = self._states.get("nocmp")
        if st is None:
            st = _RawPardiso(single_precision=False)
            self._states["nocmp"] = st
        return st

    def release_states(self) -> int:
        """Release every cached PARDISO handle (phase -1) and forget the
        factorisations; the next solve of any pattern re-analyses. Returns the
        number of handles released."""
        count = 0
        for st in list(self._states.values()):
            raw = st.raw if isinstance(st, _PatternState) else st
            raw.release()
            count += 1
        self._states.clear()
        return count


def _pinned_rows(A):
    """Rows stored as ``diag == 1`` with no off-diagonal entries (Dirichlet rows).

    Same rule as ``jax_fem_am.solvers.amg.pinned_rows`` (kept local so this
    module does not import jax)."""
    diag = A.diagonal()
    offabs = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
    return np.isclose(diag, 1.0) & (offabs == 0.0)


class _FreeBlockPattern:
    """One activation pattern: free/pinned split, upper-triangular K_FF
    gather map, 1-based ia/ja and the PARDISO handle holding its factor."""

    __slots__ = ("signature", "free", "pin", "keep", "ia", "ja", "raw",
                 "analyzed", "data")

    def __init__(self, signature, n, indptr, indices, pinned):
        self.signature = signature
        self.free = np.flatnonzero(~pinned)
        self.pin = np.flatnonzero(pinned)
        free_map = np.full(n, -1, dtype=np.int64)
        free_map[self.free] = np.arange(self.free.size)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        fr = free_map[rows]
        fc = free_map[np.asarray(indices, dtype=np.int64)]
        keep = np.flatnonzero((fr >= 0) & (fc >= fr))
        fr, fc = fr[keep], fc[keep]
        # PARDISO needs increasing columns within each row; free_map is
        # monotone, so this only reorders if the input rows were unsorted.
        order = np.lexsort((fc, fr))
        if not np.all(order[1:] > order[:-1]):
            keep, fr, fc = keep[order], fr[order], fc[order]
        del rows, free_map
        nf = self.free.size
        self.keep = keep
        self.ia = np.zeros(nf + 1, dtype=np.int32)
        np.cumsum(np.bincount(fr, minlength=nf), out=self.ia[1:])
        self.ia += 1
        self.ja = (fc + 1).astype(np.int32)
        self.raw = _RawPardiso()
        self.raw.mtype = 2
        self.analyzed = False
        self.data = None  # K_FF values of the current factorization


class FreeBlockSPDSolver:
    """PARDISO on the free block only, as a real SPD system (mtype=2).

    The mechanics system arrives as the whole mesh: pinned dofs (inactive
    nodes and Dirichlet supports) are unit rows with their off-diagonal
    entries stored as zeros, so the full-system modes factorize ~0.9 M rows
    from the first layer on, and mtype=11 ignores the symmetry. This mode
    splits off the pinned rows (same rule as the AMG lane), solves
    ``K_FF x_F = b_F - A_FP b_P`` with a Cholesky factor of the upper
    triangle, and writes ``x_P = b_P``. The result equals the full-system
    solution (the unit rows decouple exactly).

    One activation pattern is kept at a time: when the free set changes
    (a new layer), the old factor is released (phase -1) BEFORE the new
    pattern is analysed, so at most one factor is resident. Within a pattern
    the symbolic analysis is reused (phase 13 once, then 23; phase 33 when
    the values are unchanged), as in ``phase23``.

    If PARDISO rejects the block (e.g. not positive definite, error -4), the
    call falls back to a one-off full-system mtype=11 solve
    (``pardiso_solve_once``) and the pattern is dropped; ``fallbacks``
    counts these.
    """

    def __init__(self):
        self.mode = "free-spd"
        self.label = "pardiso_free_spd(mtype=2)"
        self._pattern = None
        self._stats = {
            "solves": 0,
            "analyze_calls": 0,
            "pattern_rebuilds": 0,
            "backsolve_hits": 0,
            "fallbacks": 0,
            "pinned_zero_rhs": 0,
            "pattern_seconds": 0.0,
            "split_seconds": 0.0,
            "n_free_last": 0,
            "nnz_factor_max": 0,
        }
        for phase in (13, 23, 33):
            self._stats[f"phase{phase}_calls"] = 0
            self._stats[f"phase{phase}_seconds"] = 0.0
        atexit.register(self._report)

    def _report(self):
        print(f"[pardiso {self.mode}] {self._stats}", file=sys.stderr)

    def stats_snapshot(self):
        out = dict(self._stats)
        out["numeric_factorizations"] = sum(
            int(out[f"phase{phase}_calls"]) for phase in (13, 23)
        )
        out["factorization_phase_seconds"] = sum(
            float(out[f"phase{phase}_seconds"]) for phase in (13, 23)
        )
        out["phase33_backsolve_seconds"] = float(out["phase33_seconds"])
        return out

    @staticmethod
    def _signature(n, indptr, indices, pinned):
        # same sampled signature as PyamgKrylovSolver._signature
        pinned_idx = np.flatnonzero(pinned)
        return (
            int(n),
            int(indptr[-1]),
            int(pinned_idx.size),
            hash(np.ascontiguousarray(indptr[::4093]).tobytes()),
            hash(np.ascontiguousarray(indices[::9973]).tobytes()),
            hash(np.ascontiguousarray(pinned_idx[::997]).tobytes()),
        )

    def _call(self, raw, n, data, ia, ja, rhs, *, phase):
        t0 = time.perf_counter()
        try:
            return raw.call(n, data, ia, ja, rhs, phase=phase)
        finally:
            self._stats[f"phase{phase}_calls"] += 1
            self._stats[f"phase{phase}_seconds"] += time.perf_counter() - t0

    def _drop_pattern(self):
        pat, self._pattern = self._pattern, None
        if pat is not None:
            pat.raw.release()

    def __call__(self, A, b, x0=None, linear_options=None):
        indptr, indices, data = A.getValuesCSR()
        indptr = np.asarray(indptr)
        indices = np.asarray(indices)
        data = np.asarray(data, dtype=np.float64)
        rhs = np.asarray(b, dtype=np.float64).ravel()
        n = indptr.shape[0] - 1
        self._stats["solves"] += 1

        t0 = time.perf_counter()
        Afull = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
        pinned = _pinned_rows(Afull)
        signature = self._signature(n, indptr, indices, pinned)
        pat = self._pattern
        if pat is None or pat.signature != signature:
            if pat is not None:
                self._stats["pattern_rebuilds"] += 1
            # release first: never two factors resident at once
            self._drop_pattern()
            t1 = time.perf_counter()
            pat = _FreeBlockPattern(signature, n, indptr, indices, pinned)
            self._pattern = pat
            self._stats["pattern_seconds"] += time.perf_counter() - t1
        nf = pat.free.size
        self._stats["n_free_last"] = int(nf)
        x = np.zeros(n)
        x[pat.pin] = rhs[pat.pin]
        if nf == 0:
            return x
        if np.any(x[pat.pin]):
            f = rhs[pat.free] - (Afull @ x)[pat.free]
        else:
            self._stats["pinned_zero_rhs"] += 1
            f = rhs[pat.free]
        a = np.ascontiguousarray(data[pat.keep])
        self._stats["split_seconds"] += time.perf_counter() - t0

        try:
            if not pat.analyzed:
                self._stats["analyze_calls"] += 1
                xf = self._call(pat.raw, nf, a, pat.ia, pat.ja, f, phase=13)
                pat.analyzed = True
            elif pat.data is not None and np.array_equal(pat.data, a):
                self._stats["backsolve_hits"] += 1
                xf = self._call(pat.raw, nf, a, pat.ia, pat.ja, f, phase=33)
            else:
                xf = self._call(pat.raw, nf, a, pat.ia, pat.ja, f, phase=23)
        except RuntimeError as exc:
            self._stats["fallbacks"] += 1
            print(f"WARNING: PARDISO free-spd failed ({exc}); "
                  "full-system mtype=11 solve for this call", file=sys.stderr, flush=True)
            self._drop_pattern()
            return pardiso_solve_once(A, rhs)
        pat.data = a
        self._stats["nnz_factor_max"] = max(
            self._stats["nnz_factor_max"], int(pat.raw.iparm[17])
        )
        x[pat.free] = xf
        return x

    def release_states(self) -> int:
        """Release the retained factor (phase -1); returns 1 if one was held."""
        held = self._pattern is not None
        self._drop_pattern()
        return int(held)


class PardisoCustomSolver:
    """jax-fem ``custom_solver`` adapter: PETSc AIJ -> SciPy CSR -> MKL PARDISO.

    Direct solve like spsolve (same accuracy class), but factorization runs
    multithreaded via MKL. pypardiso is imported lazily so the option
    plumbing stays importable without it. ``mode`` selects a level of the
    V07 ladder above (``None``/``"base"`` keeps the plain pypardiso path or
    the legacy ``V07_PARDISO_MODE`` environment override).

    Moved here from ``jax_fem_am.simulation.acceleration`` (which keeps the
    ``_PardisoCustomSolver`` alias) so the per-physics linear-solver registry
    in ``jax_fem_am.solvers.linear`` can build it without importing the
    wrapper.
    """

    label = "pardiso_solver(mkl multithreaded direct)"
    iterative = False

    def __init__(self, mode=None):
        valid_modes = {None, "base", "nocmp", "cache-idx", "phase23", "fp32ir",
                       "free-spd", "free-spd-cudss"}
        if mode not in valid_modes:
            raise ValueError(f"unsupported PARDISO mode: {mode!r}")
        self._solver = None
        self._v07_variant = None
        self._requested_mode = mode
        if mode not in (None, "base"):
            self.label = f"pardiso_v07({mode})"

    def __deepcopy__(self, memo):
        # Shared instance keeps the PARDISO handle alive across the
        # option-rewrite deep copies done for every solve.
        return self

    def _maybe_v07_variant(self):
        # V07 ablation hook: V07_PARDISO_MODE selects an experimental
        # solver ladder from this module. Unset (or "base") keeps the plain
        # pypardiso behaviour untouched.
        if self._v07_variant is None:
            import os

            mode = self._requested_mode
            if mode is None:
                mode = os.environ.get("V07_PARDISO_MODE", "").strip()
            if not mode or mode == "base":
                self._v07_variant = False
            elif mode == "free-spd":
                self._v07_variant = FreeBlockSPDSolver()
                self.label = self._v07_variant.label
            elif mode == "free-spd-cudss":
                # same free-block split, Cholesky of K_FF on the GPU (cuDSS); a pattern cuDSS fails on is
                # solved by FreeBlockSPDSolver (cudss-plus, 2026-11-05: in tree, was an out-of-tree hook)
                from jax_fem_am.solvers.cudss import CudssFreeBlockSPDSolver

                self._v07_variant = CudssFreeBlockSPDSolver(FreeBlockSPDSolver, _pinned_rows)
                self.label = self._v07_variant.label
            else:
                self._v07_variant = VariantSolver(mode)
                self.label = self._v07_variant.label
        return self._v07_variant

    def stats_snapshot(self):
        variant = self._maybe_v07_variant()
        if variant is False:
            return None
        return dict(variant.stats_snapshot())

    def release(self) -> int:
        """Drop every retained factorisation (handles released with phase -1).
        The adapter stays usable; the next solve pays a fresh analysis."""
        variant = self._maybe_v07_variant()
        if variant is not False:
            return variant.release_states()
        solver, self._solver = self._solver, None
        if solver is not None:
            free = getattr(solver, "free_memory", None)
            if free is not None:
                try:
                    free(everything=True)
                except TypeError:
                    free()
            return 1
        return 0

    def __call__(self, A, b, x0, linear_options):
        import pypardiso

        variant = self._maybe_v07_variant()
        if variant is not False:
            return variant(A, b, x0, linear_options)

        if self._solver is None:
            self._solver = pypardiso.PyPardisoSolver()
        indptr, indices, data = A.getValuesCSR()
        Asp = scipy.sparse.csr_matrix(
            (
                data,
                indices.astype(np.int32, copy=False),
                indptr.astype(np.int32, copy=False),
            )
        )
        rhs = np.asarray(b, dtype=np.float64)
        return pypardiso.spsolve(Asp, rhs, solver=self._solver)
