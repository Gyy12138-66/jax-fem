"""Device-side assembly of the tangent: thermal (K3; cudss-plus M1c) and mechanics (K1; M2), 2026-11-05.

Before: element tangents -> host (pinned buffer) -> PETSc ``setValuesCOO`` + ``zeroRows`` -> ``getValuesCSR`` ->
BCOO -> JAX CG on the GPU: ~2,100 s of copies and CPU work on the 0119 full height (abd ledger). Now, with
``JAX_FEM_THERMAL_DEVICE_ASSEMBLY=1``:

* ``Problem.compute_newton_vars`` keeps the cell + face tangents on the device (``problem.V_device``, same order
  as ``problem.I`` / ``problem.J``);
* ``jax_fem.solver.get_A`` returns a :class:`DeviceCSR`: the CSR values summed on the device by
  ``ffi_kernels.k3_ragged`` (every CSR slot sums its contributors in COO order: fixed, so bitwise repeatable),
  with the Dirichlet rows as PETSc ``zeroRows`` leaves them (row zero, diagonal 1);
* ``jax_fem.solver.jax_solve`` takes the values, the cached BCOO structure and the Jacobi diagonal from it.

The structure is the unique (I, J) pairs in row-major order, which is PETSc's AIJ order for this COO pattern
(prior Q5: identical on 0119). Values differ from PETSc's by the summation order only (<= 1 ulp of the entry
scale on 0119). A CPU fallback solver still gets a matrix: ``DeviceCSR.getValuesCSR()`` copies it to the host.
"""
from __future__ import annotations

import os
import time

import numpy as onp


def thermal_enabled() -> bool:
    return os.environ.get("JAX_FEM_THERMAL_DEVICE_ASSEMBLY", "0").strip().lower() in ("1", "true", "yes", "on")


class DeviceCSR:
    """An assembled tangent on the device, in CSR order, duck-typing what the solvers read from a PETSc Mat."""

    def __init__(self, data, assembler):
        self.data = data
        self.assembler = assembler
        self.shape = assembler.structure.shape

    def getSize(self):
        return self.shape

    def getValuesCSR(self):
        """Host copy (indptr, indices, data) for CPU solvers / fallbacks / dumps."""
        st = self.assembler.structure
        return (st.indptr.astype(onp.int32), st.csr_indices.astype(onp.int32), onp.asarray(self.data))

    def device_bcoo(self):
        """(data, bcoo_indices, shape, jacobi diagonal), all on the device."""
        a = self.assembler
        return self.data, a.structure.bcoo_indices, self.shape, a.diagonal(self.data)


class ThermalDeviceAssembler:
    """CSR structure + contributor map of a problem's COO pattern; assembles ``problem.V_device`` on the device."""

    def __init__(self, problem):
        import jax
        import jax.numpy as jnp
        from jax_fem.solver import _BcooStructureCache, _remember_bcoo_structure_pattern
        from jax_fem_am.solvers import ffi_kernels

        self.ffi = ffi_kernels.ensure_registered()
        t0 = time.perf_counter()
        n = int(problem.num_total_dofs_all_vars)
        I = onp.asarray(problem.I, dtype=onp.int64)
        J = onp.asarray(problem.J, dtype=onp.int64)
        ukey, inv = onp.unique(I * n + J, return_inverse=True)
        rows, cols = ukey // n, ukey % n
        indptr = onp.zeros(n + 1, dtype=onp.int64)
        onp.cumsum(onp.bincount(rows, minlength=n), out=indptr[1:])
        self.structure = _BcooStructureCache((n, n), indptr, cols)
        _remember_bcoo_structure_pattern(self.structure)
        counts = onp.bincount(inv, minlength=len(ukey))
        off = onp.zeros(len(ukey) + 1, dtype=onp.int32)
        onp.cumsum(counts, out=off[1:])
        src = onp.argsort(inv, kind="stable").astype(onp.int32)       # contributors of a slot in COO order
        diag_pos = self.structure.diag_positions
        self.n, self.nnz, self.len_V = n, int(len(ukey)), int(len(I))
        self.max_contributors = int(counts.max()) if len(counts) else 0
        self._src = jnp.asarray(src)
        self._off = jnp.asarray(off)
        self._rows = jnp.asarray(rows.astype(onp.int32))
        self._is_diag = jnp.asarray(rows == cols)
        self._diag_pos = jnp.asarray(onp.maximum(diag_pos, 0))
        self._diag_valid = jnp.asarray(diag_pos >= 0)
        self.map_bytes = int(src.nbytes + off.nbytes + rows.size * 4 + rows.size + diag_pos.size * 9)

        def assemble(V, bc_mask, src, off, rows, is_diag):
            vals = ffi_kernels.k3_ragged(V, src, off)
            unit = jnp.where(is_diag, jnp.ones((), vals.dtype), jnp.zeros((), vals.dtype))
            return jnp.where(bc_mask[rows], unit, vals)

        def diagonal(data, pos, valid):
            return jnp.where(valid, data[pos], jnp.zeros((), data.dtype))

        self._assemble = jax.jit(assemble)
        self._diagonal = jax.jit(diagonal)
        self._bc_versions = None
        self._bc_mask = None
        self._last = None
        self.calls = 0
        self.build_seconds = time.perf_counter() - t0

    def _bc_mask_for(self, problem):
        """Dirichlet rows, exactly as jax_fem.solver._PetscTangentCache._refresh_bc_rows builds them."""
        import jax.numpy as jnp

        versions = tuple(getattr(fe, "_dirichlet_bc_version", 0) for fe in problem.fes)
        if self._bc_mask is None or versions != self._bc_versions:
            mask = onp.zeros(self.n, dtype=bool)
            for ind, fe in enumerate(problem.fes):
                for i in range(len(fe.node_inds_list)):
                    rows = onp.asarray(fe.node_inds_list[i]) * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
                    mask[onp.asarray(rows, dtype=onp.int64)] = True
            self._bc_mask = jnp.asarray(mask)
            self._bc_versions = versions
        return self._bc_mask

    def diagonal(self, data):
        return self._diagonal(data, self._diag_pos, self._diag_valid)

    def matrix(self, problem):
        """DeviceCSR of the tangent last computed by compute_newton_vars (the device tangent is then dropped)."""
        V = getattr(problem, "V_device", None)
        if V is None:
            if self._last is None:
                raise RuntimeError("device tangent assembly: no tangent computed yet (compute_newton_vars first)")
            return self._last
        if int(V.shape[0]) != self.len_V:
            raise ValueError("device tangent has %d values, COO pattern %d" % (int(V.shape[0]), self.len_V))
        data = self._assemble(V, self._bc_mask_for(problem), self._src, self._off, self._rows, self._is_diag)
        problem.V_device = None
        self._last = DeviceCSR(data, self)
        self.calls += 1
        return self._last


def attach_thermal_device_assembly(problem):
    """Route ``problem``'s tangent through the device (compute_newton_vars, get_A, jax_solve)."""
    asm = ThermalDeviceAssembler(problem)
    problem.device_tangent_assembler = asm
    problem.device_tangent = True
    problem.V_device = None
    return asm


# ---------------------------------------------------------------------------------------------------------------
# Mechanics (K1; cudss-plus M2, 2026-11-05)
# ---------------------------------------------------------------------------------------------------------------

def mechanics_enabled() -> bool:
    return os.environ.get("JAX_FEM_MECH_DEVICE_ASSEMBLY", "0").strip().lower() in ("1", "true", "yes", "on")


def dirichlet_rows(problem):
    """Boolean mask of the Dirichlet rows, exactly as jax_fem.solver._PetscTangentCache._refresh_bc_rows."""
    n = int(problem.num_total_dofs_all_vars)
    mask = onp.zeros(n, dtype=bool)
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = onp.asarray(fe.node_inds_list[i]) * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            mask[onp.asarray(rows, dtype=onp.int64)] = True
    return mask


class FreeBlockPattern:
    """One activation pattern: the free / pinned split and the K_FF upper CSR (production free-spd ordering:
    rows ascending, columns ascending, both in free numbering), plus the device compaction index."""

    def __init__(self, signature, n, free, ptr, col, kidx, kidx_device):
        self.signature = signature
        self.n = n
        self.free_idx = onp.flatnonzero(free)
        self.pin_idx = onp.flatnonzero(~free)
        self.nf = int(self.free_idx.size)
        self.ptr, self.col = ptr, col
        self.nnz = int(col.size)
        self.kidx = kidx                    # full-upper slots of the K_FF entries (host)
        self.kidx_device = kidx_device      # same, padded to the full-upper length (static shape)


class MechDeviceCSR:
    """Assembled mechanics tangent on the device.

    ``values_ff[:pattern.nnz]`` are the K_FF upper values in the free-spd ordering (what cuDSS factors);
    ``upper`` are the raw (no Dirichlet rows) full-upper values. ``getValuesCSR()`` rebuilds on the host the full
    matrix PETSc would hold (lower from the upper by symmetry, Dirichlet rows zeroed with a unit diagonal) for CPU
    solvers and fallbacks."""

    def __init__(self, upper, values_ff, pattern, assembler):
        self.upper = upper
        self.values_ff = values_ff
        self.pattern = pattern
        self.assembler = assembler
        self.shape = (assembler.n, assembler.n)

    def getSize(self):
        return self.shape

    def device_free_spd(self):
        if self.values_ff is None:
            raise RuntimeError("mechanics device matrix used after its assembler was released")
        return self.pattern, self.values_ff

    def upper_host(self):
        import scipy.sparse as sp

        a = self.assembler
        return sp.csr_matrix((onp.asarray(self.upper), a.ucol, a.uptr), shape=self.shape)

    def matvec_raw(self, x):
        """K x with the raw symmetric tangent (no Dirichlet rows), on the host."""
        U = self.upper_host()
        d = U.diagonal()
        return U @ x + U.T @ x - d * x

    def getValuesCSR(self):
        import scipy.sparse as sp

        U = self.upper_host()
        K = (U + sp.triu(U, k=1, format="csr").T).tocsr()
        K.sort_indices()
        pinned = onp.zeros(self.shape[0], dtype=bool)
        pinned[self.pattern.pin_idx] = True
        rows = onp.repeat(onp.arange(self.shape[0]), onp.diff(K.indptr))
        on = pinned[rows]
        K.data[on] = onp.where(rows[on] == K.indices[on], 1.0, 0.0)
        return K.indptr.astype(onp.int32), K.indices.astype(onp.int32), K.data


class MechDeviceAssembler:
    """K1: (n_cells, 24, 24) HEX8 tangents -> K_FF upper values on the device, chunk by chunk (fixed order)."""

    def __init__(self, problem):
        import jax
        import jax.numpy as jnp
        from jax_fem_am.solvers import ffi_kernels

        self.ffi = ffi_kernels.ensure_registered()
        t0 = time.perf_counter()
        fe = problem.fes[0]
        if len(problem.fes) != 1 or fe.vec != 3 or onp.asarray(problem.cells_list[0]).shape[1] != 8:
            raise NotImplementedError("mechanics device assembly: one HEX8 vector variable only")
        C = onp.asarray(problem.cells_list[0], dtype=onp.int64)
        ne, nn = C.shape[0], int(fe.num_total_nodes)
        n = int(problem.num_total_dofs_all_vars)
        I, J = onp.asarray(problem.I), onp.asarray(problem.J)
        if len(I) != ne * 576:
            raise NotImplementedError("mechanics device assembly: face tangents are not supported")
        dofs = (3 * C[:, :, None] + onp.arange(3)).reshape(ne, 24)
        rows = onp.repeat(dofs[:, :, None], 24, 2).reshape(-1)
        cols = onp.repeat(dofs[:, None, :], 24, 1).reshape(-1)
        if not (onp.array_equal(I, rows) and onp.array_equal(J, cols)):
            raise NotImplementedError("mechanics device assembly: Problem.I/J are not the HEX8 cell layout")
        del I, J
        up = rows <= cols
        ukey = onp.unique(rows[up] * n + cols[up])
        del rows, cols, up
        self.n, self.nnz_upper, self.n_cells = n, int(ukey.size), ne
        ur, uc = ukey // n, ukey % n
        self.uptr = onp.zeros(n + 1, dtype=onp.int64)
        onp.cumsum(onp.bincount(ur, minlength=n), out=self.uptr[1:])
        self.ucol = uc.astype(onp.int32)
        self._ur, self._uc = ur.astype(onp.int32), uc.astype(onp.int32)
        # node-pair 3x3 block map: contributors e*64 + p*8 + q in cell order, CSR slot of each block row
        A = onp.repeat(C[:, :, None], 8, 2).reshape(-1)
        B = onp.repeat(C[:, None, :], 8, 1).reshape(-1)
        code = (onp.arange(ne, dtype=onp.int64)[:, None] * 64 + onp.arange(64)[None, :]).reshape(-1)
        keep = onp.flatnonzero(A <= B)
        bkey = A[keep] * nn + B[keep]
        ub, binv = onp.unique(bkey, return_inverse=True)
        cnt = onp.bincount(binv, minlength=ub.size)
        m = int(cnt.max())
        perm = onp.argsort(binv, kind="stable")
        st = onp.concatenate([[0], onp.cumsum(cnt)[:-1]])
        rk = onp.arange(perm.size) - onp.repeat(st, cnt)
        contrib = onp.full((ub.size, m), -1, dtype=onp.int32)
        contrib[binv[perm], rk] = code[keep][perm]
        del A, B, code, keep, bkey, binv, perm, st, rk
        a, b = ub // nn, ub % nn
        diag = (a == b).astype(onp.int32)
        pos = onp.empty((ub.size, 3), dtype=onp.int32)
        for i in range(3):
            target = (3 * a + i) * n + 3 * b + onp.where(diag == 1, i, 0)
            p = onp.searchsorted(ukey, target)
            if not onp.all(ukey[onp.minimum(p, ukey.size - 1)] == target):
                raise RuntimeError("mechanics device assembly: block map does not match the upper pattern")
            pos[:, i] = p
        del ukey
        self.n_blocks, self.max_contributors = int(ub.size), m
        self._contrib, self._pos, self._diag = jnp.asarray(contrib), jnp.asarray(pos), jnp.asarray(diag)
        self.map_bytes = int(contrib.nbytes + pos.nbytes + diag.nbytes)

        def acc(v, contrib, pos, diag, out, cell_start, cell_end):
            return ffi_kernels.k1_block_acc(v.reshape(-1), contrib, pos, diag, out, cell_start, cell_end)

        self._acc = jax.jit(acc, static_argnums=(5, 6), donate_argnums=(4,))
        self._take = jax.jit(lambda v, i: v.at[i].get(mode="promise_in_bounds", indices_are_sorted=True))
        self._zeros = jax.jit(lambda: jnp.zeros(self.nnz_upper, dtype=jnp.float64))
        self._bc_versions = None
        self._pattern = None
        self._last = None
        self.calls = 0
        self.pattern_builds = 0
        self.pattern_seconds = 0.0
        self.build_seconds = time.perf_counter() - t0

    def pattern_for(self, problem):
        import hashlib

        import jax.numpy as jnp

        versions = tuple(getattr(fe, "_dirichlet_bc_version", 0) for fe in problem.fes)
        if self._pattern is not None and versions == self._bc_versions:
            return self._pattern
        t0 = time.perf_counter()
        pinned = dirichlet_rows(problem)
        sig = (self.n, hashlib.sha1(onp.packbits(pinned).tobytes()).hexdigest())
        if self._pattern is None or sig != self._pattern.signature:
            free = ~pinned
            kidx = onp.flatnonzero(free[self._ur] & free[self._uc]).astype(onp.int32)
            free_map = onp.full(self.n, -1, dtype=onp.int64)
            free_map[onp.flatnonzero(free)] = onp.arange(int(free.sum()))
            fr = free_map[self._ur[kidx]]
            ptr = onp.zeros(int(free.sum()) + 1, dtype=onp.int32)
            onp.cumsum(onp.bincount(fr, minlength=int(free.sum())), out=ptr[1:])
            col = free_map[self._uc[kidx]].astype(onp.int32)
            pad = onp.zeros(self.nnz_upper, dtype=onp.int32)
            pad[:kidx.size] = kidx
            self._pattern = FreeBlockPattern(sig, self.n, free, ptr, col, kidx, jnp.asarray(pad))
            self.pattern_builds += 1
        self._bc_versions = versions
        self.pattern_seconds += time.perf_counter() - t0
        return self._pattern

    def drop_last(self):
        """Free the previous matrix's device values before the next tangent is built (called by
        Problem._compute_newton_vars_device): the tangent build is the JAX pool peak, and the old matrix (~0.47 GB
        on 0119) is not needed any more (the next solve gets the new matrix; cuDSS is pointed at its buffer before
        factorizing). M2 full height 2026-11-05: JAX pool OOM at layer 29 without this (peak 6.58 of 7.0 GB)."""
        last = self._last
        if last is not None:
            last.upper = None
            last.values_ff = None
        self._last = None

    def release(self):
        """Free the device buffers (block map, compaction index, last matrix values), e.g. before the release
        solve builds its own problem on the same mesh. The matrices handed out so far become unusable."""
        last = self._last
        if last is not None:
            last.upper = None
            last.values_ff = None
        self._last = None
        self._pattern = None
        self._contrib = self._pos = self._diag = None
        self.released = True

    def matrix(self, problem):
        if getattr(self, "released", False):
            raise RuntimeError("mechanics device assembly: assembler already released")
        chunks = getattr(problem, "V_device_chunks", None)
        if chunks is None:
            if self._last is None:
                raise RuntimeError("mechanics device assembly: no tangent computed yet")
            return self._last
        pattern = self.pattern_for(problem)
        out = self._zeros()
        for start, arr in chunks:
            stop = start + int(arr.shape[0])
            out = self._acc(arr, self._contrib, self._pos, self._diag, out, int(start), int(stop))
        problem.V_device_chunks = None
        values_ff = self._take(out, pattern.kidx_device)
        self._last = MechDeviceCSR(out, values_ff, pattern, self)
        self.calls += 1
        return self._last


def attach_mechanics_device_assembly(problem):
    """Route ``problem``'s HEX8 tangent through the device: compute_newton_vars keeps the chunks, get_A runs K1."""
    asm = MechDeviceAssembler(problem)
    problem.device_tangent_assembler = asm
    problem.device_tangent = True
    problem.device_tangent_chunks = True
    problem.V_device = None
    problem.V_device_chunks = None
    return asm
