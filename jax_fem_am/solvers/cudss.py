# -*- coding: utf-8 -*-
"""cuDSS free-block SPD solver: PARDISO mode ``free-spd-cudss`` (cudss-plus, in tree since 2026-11-05).

Moved in from the out-of-tree lane (cudss_bench/lane/cudss_free_spd.py, 2026-10-29), which swapped
``jax_fem_am.solvers.pardiso.FreeBlockSPDSolver`` for this class from a sitecustomize hook; the solver logic is
unchanged, only the packaging differs (library discovery, variant env, CPU fallback class taken from
``jax_fem_am.solvers.pardiso`` at construction). Selected with ``{"backend": "pardiso", "mode": "free-spd-cudss"}``.
The split is the free-spd one: pinned unit rows, K_FF upper triangle, x_P = b_P, one activation pattern
resident at a time, analysis once per
pattern, numeric factorization per call, back-substitution only when the values are unchanged. Only the Cholesky
factor/solve moves to the GPU (cuDSS 1.0, MTYPE_SPD, MVIEW_UPPER).

Variant from env ``JAX_FEM_CUDSS_VARIANT`` (or the lane's ``CUDSS_LANE``; default ``emu53+dc+ir2``):
``emuNN`` FP64 emulation with NN mantissa bits in the
factorization, ``dc`` factors stored in FP32 with the diagonal blocks kept in FP64, ``irN`` N refinement steps in FP64.

Any cuDSS failure (allocation, non-positive minor, non-finite solution) -> the current activation pattern is solved by
the original CPU FreeBlockSPDSolver (which itself falls back to a full-system mtype=11 solve); the next pattern tries
cuDSS again. Stats keep the free-spd keys (phase13 = analysis+factor+solve, phase23 = factor+solve, phase33 = solve
only; each phase time includes its H2D/D2H copies).
"""
import atexit
import ctypes
import os
import sys
import time
from ctypes import POINTER, byref, c_int, c_int64, c_size_t, c_void_p

import numpy as np
import scipy.sparse



def _nvidia_dir():
    """site-packages/nvidia of this interpreter (the CUDA runtime / cuBLAS wheels JAX uses)."""
    try:
        import nvidia
        return list(nvidia.__path__)[0]
    except Exception:
        return os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia")


NV = os.environ.get("JAX_FEM_NVIDIA_DIR") or _nvidia_dir()
# nvidia-cudss-cu12 1.0.0.35 lives in the cudss_bench venv on this machine, not in the jax-fem-gpu env
_LIBCUDSS_DEFAULT = "/home/user/work/159/cudss_bench/venv/lib/python3.13/site-packages/nvidia/cu12/lib/libcudss.so.1"
LIBCUDSS = None

# enums (cudss_data_types.h 1.0.0, library_types.h)
R_64F, R_32I = 1, 10
MTYPE_SPD, MVIEW_UPPER, BASE_ZERO, COL_MAJOR = 3, 2, 0, 0
PH_ANALYSIS, PH_FACTOR, PH_SOLVE = 3, 4, 1008
CFG_IR_N_STEPS, CFG_DETERMINISTIC, CFG_EMULATION, CFG_DOWNCAST = 5, 24, 26, 28
CFG_HYBRID_MEMORY_MODE, CFG_HYBRID_DEVICE_MEMORY_LIMIT = 11, 12
DATA_INFO, DATA_LU_NNZ = 0, 1

_rt = None
_ds = None


class CudssError(RuntimeError):
    pass


def find_libcudss():
    """JAX_FEM_CUDSS_LIB, else nvidia/cu12/lib of this interpreter, else the cudss_bench venv."""
    env = os.environ.get("JAX_FEM_CUDSS_LIB", "").strip()
    for cand in ([env] if env else []) + [os.path.join(NV, "cu12", "lib", "libcudss.so.1"), _LIBCUDSS_DEFAULT]:
        if cand and os.path.exists(cand):
            return cand
    raise CudssError("libcudss.so.1 not found (set JAX_FEM_CUDSS_LIB)")


def _ck(status, what):
    if status != 0:
        raise CudssError("%s -> status %d" % (what, status))


def _libs():
    global _rt, _ds, LIBCUDSS
    if _ds is None:
        for lib in ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12",
                    "cublas/lib/libcublas.so.12", "cuda_nvrtc/lib/libnvrtc.so.12"):
            ctypes.CDLL(os.path.join(NV, lib), mode=ctypes.RTLD_GLOBAL)
        rt = ctypes.CDLL(os.path.join(NV, "cuda_runtime/lib/libcudart.so.12"))
        LIBCUDSS = find_libcudss()
        ds = ctypes.CDLL(LIBCUDSS)
        print("[cudss] free-spd-cudss: %s" % LIBCUDSS, file=sys.stderr, flush=True)
        rt.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        rt.cudaFree.argtypes = [c_void_p]
        rt.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        rt.cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
        ds.cudssCreate.argtypes = [POINTER(c_void_p)]
        ds.cudssConfigCreate.argtypes = [POINTER(c_void_p)]
        ds.cudssDataCreate.argtypes = [c_void_p, POINTER(c_void_p)]
        ds.cudssConfigSet.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
        ds.cudssMatrixCreateCsr.argtypes = [POINTER(c_void_p), c_int64, c_int64, c_int64, c_void_p, c_void_p,
                                            c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int]
        ds.cudssMatrixCreateDn.argtypes = [POINTER(c_void_p), c_int64, c_int64, c_int64, c_void_p, c_int, c_int]
        ds.cudssExecute.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        ds.cudssDataGet.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_size_t, POINTER(c_size_t)]
        ds.cudssMatrixDestroy.argtypes = [c_void_p]
        ds.cudssDataDestroy.argtypes = [c_void_p, c_void_p]
        ds.cudssConfigDestroy.argtypes = [c_void_p]
        ds.cudssMatrixSetValues.argtypes = [c_void_p, c_void_p]
        _rt, _ds = rt, ds
    return _rt, _ds


class _Dev:
    """one device buffer (cudaMalloc, outside the JAX allocator)"""

    def __init__(self, nbytes):
        rt, _ = _libs()
        self.p = c_void_p()
        _ck(rt.cudaMalloc(byref(self.p), max(int(nbytes), 8)), "cudaMalloc(%d)" % nbytes)

    def put(self, a):
        _ck(_rt.cudaMemcpy(self.p, a.ctypes.data, a.nbytes, 1), "cudaMemcpy H2D")

    def get(self, a):
        _ck(_rt.cudaMemcpy(a.ctypes.data, self.p, a.nbytes, 2), "cudaMemcpy D2H")

    def free(self):
        if self.p:
            _rt.cudaFree(self.p)
            self.p = c_void_p()


class _GpuPattern:
    """one activation pattern: free/pinned split, upper-triangular K_FF gather map and its cuDSS objects
    (same construction as jax_fem_am.solvers.pardiso._FreeBlockPattern, 0-based)"""

    def __init__(self, handle, variant, signature, n, indptr, indices, pinned):
        self.signature = signature
        self.free = np.flatnonzero(~pinned)
        self.pin = np.flatnonzero(pinned)
        self.analyzed = False
        self.data = None
        self.bufs, self.mats = [], []
        self.cfg = self.cdata = None
        self.handle = handle
        free_map = np.full(n, -1, dtype=np.int64)
        free_map[self.free] = np.arange(self.free.size)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        fr = free_map[rows]
        fc = free_map[np.asarray(indices, dtype=np.int64)]
        keep = np.flatnonzero((fr >= 0) & (fc >= fr))
        fr, fc = fr[keep], fc[keep]
        order = np.lexsort((fc, fr))
        if not np.all(order[1:] > order[:-1]):
            keep, fr, fc = keep[order], fr[order], fc[order]
        del rows, free_map
        self.keep = keep
        self.nf = nf = int(self.free.size)
        nz = int(keep.size)
        ptr = np.zeros(nf + 1, dtype=np.int32)
        np.cumsum(np.bincount(fr, minlength=nf), out=ptr[1:])
        col = fc.astype(np.int32)
        if nf == 0:
            return
        try:
            self.d_ptr, self.d_col = _Dev(ptr.nbytes), _Dev(col.nbytes)
            self.bufs += [self.d_ptr, self.d_col]
            self.d_val, self.d_x, self.d_b = _Dev(nz * 8), _Dev(nf * 8), _Dev(nf * 8)
            self.bufs += [self.d_val, self.d_x, self.d_b]
            self.d_ptr.put(ptr)
            self.d_col.put(col)
            self.cfg = self._config(variant)
            self.cdata = c_void_p()
            _ck(_ds.cudssDataCreate(handle, byref(self.cdata)), "cudssDataCreate")
            self.A, self.X, self.B = c_void_p(), c_void_p(), c_void_p()
            _ck(_ds.cudssMatrixCreateCsr(byref(self.A), nf, nf, nz, self.d_ptr.p, None, self.d_col.p, self.d_val.p,
                                         R_32I, R_32I, R_64F, MTYPE_SPD, MVIEW_UPPER, BASE_ZERO), "MatrixCreateCsr")
            self.mats.append(self.A)
            _ck(_ds.cudssMatrixCreateDn(byref(self.X), nf, 1, nf, self.d_x.p, R_64F, COL_MAJOR), "MatrixCreateDn x")
            self.mats.append(self.X)
            _ck(_ds.cudssMatrixCreateDn(byref(self.B), nf, 1, nf, self.d_b.p, R_64F, COL_MAJOR), "MatrixCreateDn b")
            self.mats.append(self.B)
        except CudssError:
            self.destroy()
            raise

    @staticmethod
    def _config(variant):
        cfg = c_void_p()
        _ck(_ds.cudssConfigCreate(byref(cfg)), "cudssConfigCreate")

        def set_int(param, value):
            v = c_int(value)
            _ck(_ds.cudssConfigSet(cfg, param, byref(v), ctypes.sizeof(v)), "ConfigSet(%d)" % param)
        for tok in variant.split("+"):
            if tok.startswith("emu"):
                set_int(CFG_EMULATION, int(tok[3:]))
            elif tok == "dc":
                set_int(CFG_DOWNCAST, 1)                       # KEEP_BLOCK_DIAGONAL
            elif tok.startswith("ir"):
                set_int(CFG_IR_N_STEPS, int(tok[2:]))
            elif tok == "det":
                set_int(CFG_DETERMINISTIC, 1)
            elif tok.startswith("hyb"):                        # hybN: factors in host memory, N GiB device buffer
                set_int(CFG_HYBRID_MEMORY_MODE, 1)
                lim = c_int64(int(float(tok[3:]) * 2**30))
                _ck(_ds.cudssConfigSet(cfg, CFG_HYBRID_DEVICE_MEMORY_LIMIT, byref(lim), ctypes.sizeof(lim)),
                    "ConfigSet(hybrid limit)")
            elif tok != "fp64":
                raise ValueError("unknown CUDSS_LANE token %r" % tok)
        return cfg

    def execute(self, phase):
        _ck(_ds.cudssExecute(self.handle, phase, self.cfg, self.cdata, self.A, self.X, self.B), "Execute(%d)" % phase)

    def get(self, param, ctype=c_int64):
        v, w = ctype(0), c_size_t(0)
        _ck(_ds.cudssDataGet(self.handle, self.cdata, param, byref(v), ctypes.sizeof(v), byref(w)),
            "DataGet(%d)" % param)
        return int(v.value)

    def destroy(self):
        for m in self.mats:
            _ds.cudssMatrixDestroy(m)
        self.mats = []
        if self.cdata:
            _ds.cudssDataDestroy(self.handle, self.cdata)
            self.cdata = None
        if self.cfg:
            _ds.cudssConfigDestroy(self.cfg)
            self.cfg = None
        for d in self.bufs:
            d.free()
        self.bufs = []


class _GpuPatternFF(_GpuPattern):
    """Pattern of a device-assembled K_FF (jax_fem_am.solvers.device_assembly.FreeBlockPattern, M2): structure
    uploaded once, the values stay in the JAX buffer (cudssMatrixSetValues before every factorization), so there
    is no cuDSS value buffer and no host split."""

    def __init__(self, handle, variant, signature, fbp, values_ptr):
        self.signature = signature
        self.free = fbp.free_idx
        self.pin = fbp.pin_idx
        self.analyzed = False
        self.data = None
        self.bufs, self.mats = [], []
        self.cfg = self.cdata = None
        self.handle = handle
        self.keep = None
        self.nf = nf = int(fbp.nf)
        nz = int(fbp.nnz)
        if nf == 0:
            return
        try:
            ptr = np.ascontiguousarray(fbp.ptr, dtype=np.int32)
            col = np.ascontiguousarray(fbp.col, dtype=np.int32)
            self.d_ptr, self.d_col = _Dev(ptr.nbytes), _Dev(col.nbytes)
            self.bufs += [self.d_ptr, self.d_col]
            self.d_x, self.d_b = _Dev(nf * 8), _Dev(nf * 8)
            self.bufs += [self.d_x, self.d_b]
            self.d_ptr.put(ptr)
            self.d_col.put(col)
            self.cfg = self._config(variant)
            self.cdata = c_void_p()
            _ck(_ds.cudssDataCreate(handle, byref(self.cdata)), "cudssDataCreate")
            self.A, self.X, self.B = c_void_p(), c_void_p(), c_void_p()
            _ck(_ds.cudssMatrixCreateCsr(byref(self.A), nf, nf, nz, self.d_ptr.p, None, self.d_col.p, values_ptr,
                                         R_32I, R_32I, R_64F, MTYPE_SPD, MVIEW_UPPER, BASE_ZERO), "MatrixCreateCsr")
            self.mats.append(self.A)
            _ck(_ds.cudssMatrixCreateDn(byref(self.X), nf, 1, nf, self.d_x.p, R_64F, COL_MAJOR), "MatrixCreateDn x")
            self.mats.append(self.X)
            _ck(_ds.cudssMatrixCreateDn(byref(self.B), nf, 1, nf, self.d_b.p, R_64F, COL_MAJOR), "MatrixCreateDn b")
            self.mats.append(self.B)
        except CudssError:
            self.destroy()
            raise


class CudssFreeBlockSPDSolver:
    """Drop-in for FreeBlockSPDSolver: same call/stats/release protocol, Cholesky on the GPU through cuDSS."""

    def __init__(self, cpu_class=None, pinned_rows=None, variant=None):
        if cpu_class is None or pinned_rows is None:
            from jax_fem_am.solvers import pardiso
            cpu_class = cpu_class or pardiso.FreeBlockSPDSolver
            pinned_rows = pinned_rows or pardiso._pinned_rows
        self.cpu_class = cpu_class          # CPU PARDISO free-spd, used for a pattern cuDSS failed on
        self.pinned_rows = pinned_rows      # jax_fem_am.solvers.pardiso._pinned_rows
        self.mode = "free-spd"
        self.variant = (variant or os.environ.get("JAX_FEM_CUDSS_VARIANT", "").strip()
                        or os.environ.get("CUDSS_LANE", "").strip() or "emu53+dc+ir2")
        self.label = "cudss_free_spd(%s)" % self.variant
        self._handle = None
        self._pattern = None
        self._cpu = None
        self._cpu_signature = None
        self._stats = {
            "solves": 0, "analyze_calls": 0, "pattern_rebuilds": 0, "backsolve_hits": 0, "fallbacks": 0,
            "pinned_zero_rhs": 0, "pattern_seconds": 0.0, "split_seconds": 0.0, "n_free_last": 0,
            "nnz_factor_max": 0,
            "gpu_pattern_failures": 0, "cpu_pattern_solves": 0, "gpu_mem_used_gb_max": 0.0, "last_error": "",
        }
        for phase in (13, 23, 33):
            self._stats[f"phase{phase}_calls"] = 0
            self._stats[f"phase{phase}_seconds"] = 0.0
        atexit.register(self._report)

    def _report(self):
        print(f"[cudss {self.label}] {self.stats_snapshot()}", file=sys.stderr)

    def stats_snapshot(self):
        out = dict(self._stats)
        out["numeric_factorizations"] = sum(int(out[f"phase{p}_calls"]) for p in (13, 23))
        out["factorization_phase_seconds"] = sum(float(out[f"phase{p}_seconds"]) for p in (13, 23))
        out["phase33_backsolve_seconds"] = float(out["phase33_seconds"])
        out["backend"] = self.label
        if self._cpu is not None:
            cpu = self._cpu.stats_snapshot()
            out["cpu_stats"] = cpu
            out["fallbacks"] = int(cpu.get("fallbacks", 0))
        return out

    def _jax_memory(self):
        """JAX allocator peak in use / pool size (what JAX really needs vs what its pool grew to)"""
        jax = sys.modules.get("jax")
        if jax is None:
            return
        try:
            ms = jax.devices()[0].memory_stats() or {}
        except Exception:
            return
        for src, dst in (("peak_bytes_in_use", "jax_peak_in_use_gb"), ("pool_bytes", "jax_pool_gb"),
                         ("peak_pool_bytes", "jax_peak_pool_gb"), ("bytes_limit", "jax_limit_gb")):
            if src in ms:
                self._stats[dst] = max(self._stats.get(dst, 0.0), ms[src] / 1024 ** 3)

    def _drop_pattern(self):
        pat, self._pattern = self._pattern, None
        if pat is not None:
            pat.destroy()

    def _release_cpu(self):
        held = 0
        if self._cpu is not None:
            held = self._cpu.release_states()
        self._cpu_signature = None
        return held

    def release_states(self) -> int:
        """Release the retained factor(s); returns the number released."""
        held = int(self._pattern is not None)
        self._drop_pattern()
        return held + self._release_cpu()

    def _cpu_solve(self, A, b, x0, linear_options):
        if self._cpu is None:
            self._cpu = self.cpu_class()
        self._stats["cpu_pattern_solves"] += 1
        return self._cpu(A, b, x0, linear_options)

    def _gpu_failed(self, exc, signature, A, b, x0, linear_options):
        self._stats["gpu_pattern_failures"] += 1
        self._stats["last_error"] = str(exc)[:200]
        print(f"WARNING: cuDSS free-spd failed ({exc}); CPU PARDISO free-spd for this activation pattern",
              file=sys.stderr, flush=True)
        self._drop_pattern()
        if _rt is not None:
            _rt.cudaGetLastError()                 # clear a non-sticky error such as a failed allocation
        self._cpu_signature = signature
        return self._cpu_solve(A, b, x0, linear_options)

    def _solve_device(self, A, b, x0, linear_options):
        """K_FF assembled on the device (MechDeviceCSR): factor/solve straight from its buffer."""
        fbp, values = A.device_free_spd()
        rhs = np.asarray(b, dtype=np.float64).ravel()
        n = int(fbp.n)
        self._stats["solves"] += 1
        self._stats["device_solves"] = self._stats.get("device_solves", 0) + 1
        t0 = time.perf_counter()
        signature = ("device",) + tuple(fbp.signature)
        if signature == self._cpu_signature:       # this pattern already failed on the GPU
            return self._cpu_solve(A, b, x0, linear_options)
        x = np.zeros(n)
        x[fbp.pin_idx] = rhs[fbp.pin_idx]
        if fbp.nf == 0:
            return x
        if np.any(x[fbp.pin_idx]):
            f = rhs[fbp.free_idx] - A.matvec_raw(x)[fbp.free_idx]
        else:
            self._stats["pinned_zero_rhs"] += 1
            f = rhs[fbp.free_idx]
        f = np.ascontiguousarray(f)
        values.block_until_ready()               # cuDSS runs on its own stream: the values must be complete
        vptr = c_void_p(values.unsafe_buffer_pointer())
        pat = self._pattern
        if pat is None or pat.signature != signature:
            if pat is not None:
                self._stats["pattern_rebuilds"] += 1
            self._drop_pattern()
            self._release_cpu()
            t1 = time.perf_counter()
            try:
                if self._handle is None:
                    _libs()
                    h = c_void_p()
                    _ck(_ds.cudssCreate(byref(h)), "cudssCreate")
                    self._handle = h
                pat = _GpuPatternFF(self._handle, self.variant, signature, fbp, vptr)
            except CudssError as exc:
                return self._gpu_failed(exc, signature, A, b, x0, linear_options)
            self._pattern = pat
            self._stats["pattern_seconds"] += time.perf_counter() - t1
        self._stats["n_free_last"] = int(fbp.nf)
        self._stats["split_seconds"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            _ck(_ds.cudssMatrixSetValues(pat.A, vptr), "MatrixSetValues")
            if not pat.analyzed:
                phase = 13
                self._stats["analyze_calls"] += 1
                pat.execute(PH_ANALYSIS)
                pat.analyzed = True
            elif pat.data is A:
                phase = 33
                self._stats["backsolve_hits"] += 1
            else:
                phase = 23
            pat.d_b.put(f)
            if phase != 33:
                pat.execute(PH_FACTOR)
                info = pat.get(DATA_INFO, c_int)
                if info != 0:
                    raise CudssError("Cholesky: non-positive minor at %d" % info)
            pat.execute(PH_SOLVE)
            xf = np.empty(int(fbp.nf))
            pat.d_x.get(xf)
            if not np.all(np.isfinite(xf)):
                raise CudssError("non-finite solution")
            if phase != 33:
                self._stats["nnz_factor_max"] = max(self._stats["nnz_factor_max"], pat.get(DATA_LU_NNZ))
                fr_, tot_ = c_size_t(), c_size_t()
                if _rt.cudaMemGetInfo(byref(fr_), byref(tot_)) == 0:
                    self._stats["gpu_mem_used_gb_max"] = max(self._stats["gpu_mem_used_gb_max"],
                                                             (tot_.value - fr_.value) / 1024 ** 3)
                self._jax_memory()
        except CudssError as exc:
            return self._gpu_failed(exc, signature, A, b, x0, linear_options)
        self._stats[f"phase{phase}_calls"] += 1
        self._stats[f"phase{phase}_seconds"] += time.perf_counter() - t1
        pat.data = A          # keeps the device values alive for back-substitution / refinement reuse
        x[fbp.free_idx] = xf
        return x

    def __call__(self, A, b, x0=None, linear_options=None):
        if getattr(A, "device_free_spd", None) is not None:
            return self._solve_device(A, b, x0, linear_options)
        indptr, indices, data = A.getValuesCSR()
        indptr = np.asarray(indptr)
        indices = np.asarray(indices)
        data = np.asarray(data, dtype=np.float64)
        rhs = np.asarray(b, dtype=np.float64).ravel()
        n = indptr.shape[0] - 1
        self._stats["solves"] += 1

        t0 = time.perf_counter()
        Afull = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
        pinned = self.pinned_rows(Afull)
        signature = self.cpu_class._signature(n, indptr, indices, pinned)
        if signature == self._cpu_signature:       # this pattern already failed on the GPU
            return self._cpu_solve(A, b, x0, linear_options)
        pat = self._pattern
        if pat is None or pat.signature != signature:
            if pat is not None:
                self._stats["pattern_rebuilds"] += 1
            # release first: never two factors resident at once
            self._drop_pattern()
            self._release_cpu()
            t1 = time.perf_counter()
            try:
                if self._handle is None:
                    _libs()
                    h = c_void_p()
                    _ck(_ds.cudssCreate(byref(h)), "cudssCreate")
                    self._handle = h
                pat = _GpuPattern(self._handle, self.variant, signature, n, indptr, indices, pinned)
            except CudssError as exc:
                return self._gpu_failed(exc, signature, A, b, x0, linear_options)
            self._pattern = pat
            self._stats["pattern_seconds"] += time.perf_counter() - t1
        nf = pat.nf
        self._stats["n_free_last"] = nf
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
        f = np.ascontiguousarray(f)
        self._stats["split_seconds"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            if not pat.analyzed:
                phase = 13
                self._stats["analyze_calls"] += 1
                pat.d_val.put(a)
                pat.execute(PH_ANALYSIS)
                pat.analyzed = True
            elif pat.data is not None and np.array_equal(pat.data, a):
                phase = 33
                self._stats["backsolve_hits"] += 1
            else:
                phase = 23
                pat.d_val.put(a)
            pat.d_b.put(f)
            if phase != 33:
                pat.execute(PH_FACTOR)
                info = pat.get(DATA_INFO, c_int)
                if info != 0:
                    raise CudssError("Cholesky: non-positive minor at %d" % info)
            pat.execute(PH_SOLVE)
            xf = np.empty(nf)
            pat.d_x.get(xf)
            if not np.all(np.isfinite(xf)):
                raise CudssError("non-finite solution")
            if phase != 33:
                self._stats["nnz_factor_max"] = max(self._stats["nnz_factor_max"], pat.get(DATA_LU_NNZ))
                fr_, tot_ = c_size_t(), c_size_t()
                if _rt.cudaMemGetInfo(byref(fr_), byref(tot_)) == 0:
                    self._stats["gpu_mem_used_gb_max"] = max(self._stats["gpu_mem_used_gb_max"],
                                                             (tot_.value - fr_.value) / 1024 ** 3)
                self._jax_memory()
        except CudssError as exc:
            return self._gpu_failed(exc, signature, A, b, x0, linear_options)
        self._stats[f"phase{phase}_calls"] += 1
        self._stats[f"phase{phase}_seconds"] += time.perf_counter() - t1
        pat.data = a
        x[pat.free] = xf
        return x
