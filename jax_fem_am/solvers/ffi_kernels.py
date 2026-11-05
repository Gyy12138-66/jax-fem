"""Hand-written CUDA kernels for device-side assembly, called from jitted JAX code (cudss-plus, 2026-11-05).

No nvcc is needed (the jax-fem-gpu environment ships ptxas and NVRTC only):

* the kernels below are compiled in-process by NVRTC straight to an sm_XX CUBIN (no ptxas process);
* a small XLA FFI handler library (C++, host code only) launches them with the CUDA driver API on XLA's stream.
  It is built once with g++ into ``$JAX_FEM_FFI_CACHE`` (default ``~/.cache/jax_fem_am/ffi``), keyed by a hash of
  its source and the jax/jaxlib versions. Build it BEFORE a long run (``python -m jax_fem_am.solvers.ffi_kernels
  --build``): a big runner process should not fork a compiler (the ptxas D-state freezes under WSL).

Kernels (all sums run over the contributors in a fixed order, so the results are bitwise repeatable):

* ``k3_ragged(v, src, off)``: ``out[i] = sum_{k=off[i]}^{off[i+1]-1} v[src[k]]`` (thermal tangent, any COO);
* ``k1_scalar(v, table)``: ``out[i] = sum_k v[table[i, k]]`` over the entries >= 0 of a padded table;
* ``k1_block(v, contrib, pos, diag)``: node-pair 3x3 blocks of a (n_cells, 24, 24) tangent (mechanics);
* ``k1_block_acc(v_chunk, contrib, pos, diag, out, cell_start, cell_end)``: the same for the cells
  [cell_start, cell_end) of one tangent chunk, ADDED into ``out`` (aliased): chunk after chunk in cell order this
  sums every entry in exactly the element order of ``k1_block`` (M2, so the whole tangent never sits on the device).

E feasibility priors Q1/Q4/Q5 (output/cplus_priors3): NVRTC 0.05 s, K1 block 3.3 ms, K3 1.2 ms on 0119.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import threading
from ctypes import byref, c_char_p, c_size_t, c_void_p

KERNEL_SRC = r"""
extern "C" __global__ void k3_ragged(const double* __restrict__ v, const int* __restrict__ src,
                                     const int* __restrict__ off, double* __restrict__ out, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int a = off[i], b = off[i + 1];
  double s = 0.0;
  for (int k = a; k < b; ++k) s += v[src[k]];
  out[i] = s;
}
extern "C" __global__ void k1_scalar(const double* __restrict__ v, const int* __restrict__ t,
                                     double* __restrict__ out, long long n, int m) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int* r = t + i * m;
  double s = 0.0;
  for (int k = 0; k < m; ++k) { int j = r[k]; if (j < 0) break; s += v[j]; }
  out[i] = s;
}
extern "C" __global__ void k1_block(const double* __restrict__ v, const int* __restrict__ c,
                                    const int* __restrict__ pos, const int* __restrict__ diag,
                                    double* __restrict__ out, long long nb, int m) {
  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nb * 9) return;
  long long b = w / 9;
  int t = (int)(w - b * 9), i = t / 3, j = t - 3 * (t / 3);
  int d = diag[b];
  if (d && j < i) return;
  const int* r = c + b * m;
  double s = 0.0;
  for (int k = 0; k < m; ++k) {
    int x = r[k];
    if (x < 0) break;
    long long e = x >> 6;
    int p = (x >> 3) & 7, q = x & 7;
    s += v[e * 576 + (3 * p + i) * 24 + 3 * q + j];
  }
  out[pos[b * 3 + i] + (d ? j - i : j)] = s;
}
extern "C" __global__ void k1_block_acc(const double* __restrict__ v, const int* __restrict__ c,
                                        const int* __restrict__ pos, const int* __restrict__ diag,
                                        double* __restrict__ out, long long nb, int m, long long cs, long long ce) {
  long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nb * 9) return;
  long long b = w / 9;
  int t = (int)(w - b * 9), i = t / 3, j = t - 3 * (t / 3);
  int d = diag[b];
  if (d && j < i) return;
  const int* r = c + b * m;
  long long p = pos[b * 3 + i] + (d ? j - i : j);
  double s = 0.0;
  bool any = false;
  for (int k = 0; k < m; ++k) {
    int x = r[k];
    if (x < 0) break;
    long long e = x >> 6;
    if (e < cs) continue;          // contributors are sorted by cell: earlier chunks
    if (e >= ce) break;            // later chunks
    if (!any) { s = out[p]; any = true; }
    int pp = (x >> 3) & 7, q = x & 7;
    s += v[(e - cs) * 576 + (3 * pp + i) * 24 + 3 * q + j];
  }
  if (any) out[p] = s;
}
"""

HANDLER_SRC = r"""
// XLA FFI handlers for jax_fem_am.solvers.ffi_kernels: launch NVRTC-compiled kernels on XLA's stream.
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static std::vector<char> g_image;
static std::mutex g_mu;
static CUmodule g_mod = nullptr;
static CUfunction g_k3 = nullptr, g_scalar = nullptr, g_block = nullptr, g_block_acc = nullptr;

extern "C" int jfa_set_image(const void* data, size_t size) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_image.assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
  return 0;
}

static std::string cu_err(CUresult r, const char* what) {
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return std::string(what) + ": " + (s ? s : "?");
}

static ffi::Error ensure_loaded() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_mod) return ffi::Error::Success();
  if (g_image.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "jfa: no CUBIN image set");
  CUcontext ctx = nullptr;
  cuCtxGetCurrent(&ctx);
  if (!ctx) {
    CUdevice dev;
    CUresult r = cuDeviceGet(&dev, 0);
    if (r == CUDA_SUCCESS) r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r == CUDA_SUCCESS) r = cuCtxSetCurrent(ctx);
    if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err(r, "jfa: context"));
  }
  CUresult r = cuModuleLoadData(&g_mod, g_image.data());
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err(r, "jfa: cuModuleLoadData"));
  r = cuModuleGetFunction(&g_k3, g_mod, "k3_ragged");
  if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&g_scalar, g_mod, "k1_scalar");
  if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&g_block, g_mod, "k1_block");
  if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&g_block_acc, g_mod, "k1_block_acc");
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err(r, "jfa: cuModuleGetFunction"));
  return ffi::Error::Success();
}

static ffi::Error launch(CUfunction f, long long work, cudaStream_t stream, void** args, const char* what) {
  if (work <= 0) return ffi::Error::Success();
  unsigned threads = 256, grid = static_cast<unsigned>((work + threads - 1) / threads);
  CUresult r = cuLaunchKernel(f, grid, 1, 1, threads, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr);
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err(r, what));
  return ffi::Error::Success();
}

static ffi::Error K3RaggedImpl(cudaStream_t stream, ffi::Buffer<ffi::F64> v, ffi::Buffer<ffi::S32> src,
                               ffi::Buffer<ffi::S32> off, ffi::ResultBuffer<ffi::F64> out) {
  ffi::Error e = ensure_loaded();
  if (e.failure()) return e;
  long long n = static_cast<long long>(off.element_count()) - 1;
  const double* pv = v.typed_data();
  const int* ps = src.typed_data();
  const int* po = off.typed_data();
  double* pr = out->typed_data();
  void* args[] = {&pv, &ps, &po, &pr, &n};
  return launch(g_k3, n, stream, args, "k3_ragged launch");
}

static ffi::Error K1ScalarImpl(cudaStream_t stream, ffi::Buffer<ffi::F64> v, ffi::Buffer<ffi::S32> table,
                               ffi::ResultBuffer<ffi::F64> out) {
  ffi::Error e = ensure_loaded();
  if (e.failure()) return e;
  auto dims = table.dimensions();
  long long n = dims[0];
  int m = static_cast<int>(dims[1]);
  const double* pv = v.typed_data();
  const int* pt = table.typed_data();
  double* po = out->typed_data();
  void* args[] = {&pv, &pt, &po, &n, &m};
  return launch(g_scalar, n, stream, args, "k1_scalar launch");
}

static ffi::Error K1BlockImpl(cudaStream_t stream, ffi::Buffer<ffi::F64> v, ffi::Buffer<ffi::S32> contrib,
                              ffi::Buffer<ffi::S32> pos, ffi::Buffer<ffi::S32> diag,
                              ffi::ResultBuffer<ffi::F64> out) {
  ffi::Error e = ensure_loaded();
  if (e.failure()) return e;
  auto dims = contrib.dimensions();
  long long nb = dims[0];
  int m = static_cast<int>(dims[1]);
  const double* pv = v.typed_data();
  const int* pc = contrib.typed_data();
  const int* pp = pos.typed_data();
  const int* pd = diag.typed_data();
  double* po = out->typed_data();
  void* args[] = {&pv, &pc, &pp, &pd, &po, &nb, &m};
  return launch(g_block, nb * 9, stream, args, "k1_block launch");
}

// out is aliased to the result (input_output_aliases): the result buffer holds the running sums
static ffi::Error K1BlockAccImpl(cudaStream_t stream, ffi::Buffer<ffi::F64> v, ffi::Buffer<ffi::S32> contrib,
                                 ffi::Buffer<ffi::S32> pos, ffi::Buffer<ffi::S32> diag, ffi::Buffer<ffi::F64> out_in,
                                 ffi::ResultBuffer<ffi::F64> out, int64_t cell_start, int64_t cell_end) {
  ffi::Error e = ensure_loaded();
  if (e.failure()) return e;
  if (out->typed_data() != out_in.typed_data())
    return ffi::Error(ffi::ErrorCode::kInternal, "k1_block_acc: output not aliased to the accumulator input");
  auto dims = contrib.dimensions();
  long long nb = dims[0];
  int m = static_cast<int>(dims[1]);
  long long cs = cell_start, ce = cell_end;
  const double* pv = v.typed_data();
  const int* pc = contrib.typed_data();
  const int* pp = pos.typed_data();
  const int* pd = diag.typed_data();
  double* po = out->typed_data();
  void* args[] = {&pv, &pc, &pp, &pd, &po, &nb, &m, &cs, &ce};
  return launch(g_block_acc, nb * 9, stream, args, "k1_block_acc launch");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(JfaK1BlockAcc, K1BlockAccImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("cell_start")
                                  .Attr<int64_t>("cell_end"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(JfaK3Ragged, K3RaggedImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(JfaK1Scalar, K1ScalarImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(JfaK1Block, K1BlockImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());
"""

TARGETS = {"jfa_k3_ragged": "JfaK3Ragged", "jfa_k1_scalar": "JfaK1Scalar", "jfa_k1_block": "JfaK1Block",
           "jfa_k1_block_acc": "JfaK1BlockAcc"}
_lock = threading.Lock()
_state = {"registered": False, "lib": None, "image": None, "info": {}}


class FfiUnavailable(RuntimeError):
    pass


def _nvidia_dir():
    try:
        import nvidia
        return list(nvidia.__path__)[0]
    except Exception:
        return os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia")


def _libcuda_dir():
    for d in (os.environ.get("JAX_FEM_LIBCUDA_DIR", ""), "/usr/lib/wsl/lib", "/usr/lib/x86_64-linux-gnu",
              "/usr/lib64", "/usr/local/cuda/lib64/stubs"):
        if d and os.path.exists(os.path.join(d, "libcuda.so")):
            return d
    raise FfiUnavailable("libcuda.so not found (set JAX_FEM_LIBCUDA_DIR)")


def _cache_dir():
    return os.environ.get("JAX_FEM_FFI_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "jax_fem_am", "ffi")


def handler_library_path():
    import jax
    import jaxlib

    key = hashlib.sha256((HANDLER_SRC + jax.__version__ + jaxlib.__version__).encode()).hexdigest()[:16]
    return os.path.join(_cache_dir(), "libjfa_ffi_%s.so" % key)


def build_handler_library(force=False):
    """Compile the FFI handler library with g++ unless the cached copy exists; returns its path."""
    import jax

    so = handler_library_path()
    if os.path.exists(so) and not force:
        return so
    os.makedirs(os.path.dirname(so), exist_ok=True)
    nv = _nvidia_dir()
    cc = so[:-3] + ".cc"
    with open(cc, "w") as f:
        f.write(HANDLER_SRC)
    libdir = _libcuda_dir()
    tmp = "%s.tmp.%d" % (so, os.getpid())
    cmd = [os.environ.get("CXX", "g++"), "-O2", "-shared", "-fPIC", "-std=c++17",
           "-I" + jax.ffi.include_dir(), "-I" + os.path.join(nv, "cuda_runtime", "include"),
           "-I" + os.path.join(nv, "cuda_nvcc", "include"), cc, "-o", tmp,
           "-L" + libdir, "-lcuda", "-Wl,-rpath," + libdir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FfiUnavailable("g++ failed (%d): %s" % (r.returncode, (r.stderr or r.stdout)[-2000:]))
    os.replace(tmp, so)
    return so


def _gpu_arch():
    import jax

    cc = getattr(jax.devices()[0], "compute_capability", None)
    if cc:
        major, minor = str(cc).split(".")[:2]
        return "sm_%s%s" % (major, minor)
    return os.environ.get("JAX_FEM_FFI_ARCH", "sm_120")


def compile_kernels():
    """NVRTC: KERNEL_SRC -> CUBIN bytes for this GPU (in process, no ptxas)."""
    nv = ctypes.CDLL(os.path.join(_nvidia_dir(), "cuda_nvrtc", "lib", "libnvrtc.so.12"))
    prog = c_void_p()
    if nv.nvrtcCreateProgram(byref(prog), KERNEL_SRC.encode(), b"jfa_kernels.cu", 0, None, None) != 0:
        raise FfiUnavailable("nvrtcCreateProgram failed")
    arch = _gpu_arch()
    opts = (c_char_p * 2)(("--gpu-architecture=" + arch).encode(), b"-std=c++17")
    rc = nv.nvrtcCompileProgram(prog, 2, opts)
    if rc != 0:
        n = c_size_t()
        nv.nvrtcGetProgramLogSize(prog, byref(n))
        log = ctypes.create_string_buffer(n.value)
        nv.nvrtcGetProgramLog(prog, log)
        raise FfiUnavailable("NVRTC failed for %s: %s" % (arch, log.value.decode(errors="replace")[-2000:]))
    n = c_size_t()
    nv.nvrtcGetCUBINSize(prog, byref(n))
    buf = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetCUBIN(prog, buf)
    nv.nvrtcDestroyProgram(byref(prog))
    return buf, n.value, arch


def ensure_registered(build=False):
    """Load (optionally build) the handler library, hand it the CUBIN and register the FFI targets once."""
    with _lock:
        if _state["registered"]:
            return _state["info"]
        import jax

        if jax.default_backend() != "gpu":
            raise FfiUnavailable("device-assembly kernels need the CUDA backend (got %r)" % jax.default_backend())
        so = handler_library_path()
        if not os.path.exists(so):
            if not build and os.environ.get("JAX_FEM_FFI_BUILD", "1").strip() in ("0", "false", "no"):
                raise FfiUnavailable("FFI handler library not built: %s" % so)
            so = build_handler_library()
        lib = ctypes.cdll.LoadLibrary(so)
        lib.jfa_set_image.argtypes = [c_void_p, c_size_t]
        buf, size, arch = compile_kernels()
        lib.jfa_set_image(ctypes.cast(buf, c_void_p), size)
        for target, symbol in TARGETS.items():
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA")
        _state.update(registered=True, lib=lib, image=buf,
                      info=dict(library=so, arch=arch, cubin_bytes=int(size)))
        return _state["info"]


def available():
    try:
        ensure_registered()
        return True
    except Exception:
        return False


def k3_ragged(v, src, off):
    import jax

    n = off.shape[0] - 1
    return jax.ffi.ffi_call("jfa_k3_ragged", jax.ShapeDtypeStruct((n,), v.dtype))(v, src, off)


def k1_scalar(v, table):
    import jax

    return jax.ffi.ffi_call("jfa_k1_scalar", jax.ShapeDtypeStruct((table.shape[0],), v.dtype))(v, table)


def k1_block(v, contrib, pos, diag, nnz):
    import jax

    return jax.ffi.ffi_call("jfa_k1_block", jax.ShapeDtypeStruct((int(nnz),), v.dtype))(v, contrib, pos, diag)


def k1_block_acc(v_chunk, contrib, pos, diag, out, cell_start, cell_end):
    """Add the tangent entries of cells [cell_start, cell_end) (``v_chunk``: their flat (cells, 24, 24) tangents)
    into ``out`` (full upper CSR values); returns the updated ``out`` (same buffer when donated)."""
    import jax
    import numpy as np

    call = jax.ffi.ffi_call("jfa_k1_block_acc", jax.ShapeDtypeStruct(out.shape, out.dtype),
                            input_output_aliases={4: 0})
    return call(v_chunk, contrib, pos, diag, out, cell_start=np.int64(cell_start), cell_end=np.int64(cell_end))


if __name__ == "__main__":
    if "--build" in sys.argv:
        print(build_handler_library(force="--force" in sys.argv))
    else:
        print(ensure_registered(build=True))
