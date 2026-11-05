"""Device-side assembly of the thermal tangent (K3; cudss-plus M1c, 2026-11-05).

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
