"""cudss-plus 2026-11-05 (M2 / K1): the mechanics HEX8 tangent assembled on the device and handed to cuDSS.

A small vector problem (symmetric tangent, Dirichlet rows on whole nodes and on single components, 3 cell chunks)
is assembled both ways: PETSc + the production free-spd split, and the K1 device assembler. Checked: the free set
taken from the Dirichlet rows equals the numerical detection, K_FF structure identical and values within a few
ulps, the host rebuild of the full matrix matches PETSc, bitwise repeatable device values, the cuDSS device path
(analysis, back-substitution only, refactorization) solves like the host path, a nonzero pinned right-hand side,
and the CPU fallback on the device matrix. Needs the CUDA backend and libcudss (gpu marker).
"""
import numpy as onp
import pytest
import scipy.sparse as sp

import jax
import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.problem import Problem
import jax_fem.solver as solver_mod


def _libcudss():
    try:
        from jax_fem_am.solvers import cudss
        cudss.find_libcudss()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(jax.default_backend() != "gpu", reason="device assembly needs the CUDA backend"),
    pytest.mark.skipif(not _libcudss(), reason="libcudss.so.1 not found"),
]


class _Elastic(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 2.0 * eps + 0.8 * jnp.trace(eps) * jnp.eye(3) + 0.2 * jnp.trace(eps) ** 2 * jnp.eye(3)
        return stress


def _problem():
    m = box_mesh(6, 5, 4, 1.0, 1.0, 1.0)
    mesh = Mesh(m.points, m.cells_dict["hexahedron"])
    left = lambda x: jnp.isclose(x[0], 0.0, atol=1e-5)
    bottom = lambda x: jnp.isclose(x[2], 0.0, atol=1e-5)
    zero = lambda x: 0.0
    p = _Elastic(mesh=mesh, vec=3, dim=3, ele_type="HEX8",
                 dirichlet_bc_info=[[left, left, left, bottom], [0, 1, 2, 2], [zero, zero, zero, zero]])
    p.cell_assembly_num_cuts = 3
    return p


def _sol(p, seed=0):
    rng = onp.random.default_rng(seed)
    return [jnp.asarray(1e-2 * rng.standard_normal((p.fes[0].num_total_nodes, 3)))]


def _lane_kff(A):
    from jax_fem_am.solvers.pardiso import _pinned_rows
    indptr, indices, data = [onp.asarray(x) for x in A.getValuesCSR()]
    n = len(indptr) - 1
    pinned = _pinned_rows(sp.csr_matrix((data, indices, indptr), shape=(n, n)))
    free_map = onp.full(n, -1, onp.int64)
    free_map[onp.flatnonzero(~pinned)] = onp.arange(int((~pinned).sum()))
    rows = onp.repeat(onp.arange(n), onp.diff(indptr))
    fr, fc = free_map[rows], free_map[indices.astype(onp.int64)]
    keep = onp.flatnonzero((fr >= 0) & (fc >= fr))
    keep = keep[onp.lexsort((fc[keep], fr[keep]))]
    return pinned, fr[keep], fc[keep], data[keep], (indptr, indices, data)


def _both(seed=0):
    from jax_fem_am.solvers.device_assembly import attach_mechanics_device_assembly

    p = _problem()
    sol = _sol(p, seed)
    p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    A_ref = solver_mod.get_A(p)
    ref = _lane_kff(A_ref)
    asm = attach_mechanics_device_assembly(p)
    p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    assert p.V is None and p.V_device_chunks is not None and len(p.V_device_chunks) == 3
    A_dev = solver_mod.get_A(p)
    assert p.V_device_chunks is None
    return p, sol, asm, A_ref, ref, A_dev


def test_kff_matches_production_split():
    p, sol, asm, A_ref, (pinned, fr, fc, vals, full), A_dev = _both()
    fbp, values = A_dev.device_free_spd()
    assert onp.array_equal(fbp.pin_idx, onp.flatnonzero(pinned))
    rows_ff = onp.repeat(onp.arange(fbp.nf), onp.diff(fbp.ptr))
    assert onp.array_equal(rows_ff, fr) and onp.array_equal(fbp.col, fc)
    v = onp.asarray(values)[:fbp.nnz]
    scale = float(onp.abs(vals).max())
    assert onp.max(onp.abs(v - vals)) <= 8 * onp.finfo(float).eps * scale
    # host rebuild of the full matrix (CPU fallbacks) matches PETSc
    indptr, indices, data = A_dev.getValuesCSR()
    assert onp.array_equal(indptr, full[0]) and onp.array_equal(indices, full[1])
    assert onp.max(onp.abs(data - full[2])) <= 8 * onp.finfo(float).eps * scale
    # bitwise repeatable
    p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    again = solver_mod.get_A(p)
    assert onp.array_equal(onp.asarray(again.device_free_spd()[1]), onp.asarray(values))
    assert asm.pattern_builds == 1 and asm.calls == 2


def test_cudss_device_path_solves_like_the_host_path():
    from jax_fem_am.solvers.cudss import CudssFreeBlockSPDSolver

    p, sol, asm, A_ref, (pinned, *_), A_dev = _both(seed=1)
    n = A_ref.getSize()[0]
    rng = onp.random.default_rng(4)
    b = rng.standard_normal(n)
    b[pinned] = 0.0
    host, dev = CudssFreeBlockSPDSolver(), CudssFreeBlockSPDSolver()
    x_ref = host(A_ref, b)
    x1 = dev(A_dev, b)
    x2 = dev(A_dev, b)                            # same matrix object: back-substitution only
    assert onp.max(onp.abs(x1 - x_ref)) <= 1e-9 * onp.max(onp.abs(x_ref))
    assert onp.max(onp.abs(x2 - x1)) <= 1e-12 * onp.max(onp.abs(x1))
    p.compute_newton_vars(_sol(p, 2), p.internal_vars, p.internal_vars_surfaces)
    A2 = solver_mod.get_A(p)
    x3 = dev(A2, b)                               # new values, same pattern: refactorization
    assert onp.all(onp.isfinite(x3))
    # nonzero pinned right-hand side: x_P = b_P, f = b_F - K_FP x_P
    b2 = b.copy()
    b2[pinned] = 1e-3 * rng.standard_normal(int(pinned.sum()))
    x4_ref = host(A_ref, b2)
    x4 = dev(A_dev, b2)
    assert onp.max(onp.abs(x4 - x4_ref)) <= 1e-9 * onp.max(onp.abs(x4_ref))
    st = dev.stats_snapshot()
    assert st["device_solves"] == 4 and st["phase13_calls"] == 1 and st["phase33_calls"] == 1
    assert st["phase23_calls"] == 2 and st["gpu_pattern_failures"] == 0
    # CPU fallback on the device matrix (as after a cuDSS failure)
    x5 = dev._cpu_solve(A_dev, b, None, None)
    assert onp.max(onp.abs(x5 - x_ref)) <= 1e-9 * onp.max(onp.abs(x_ref))
    # release (before the release solve): buffers dropped, stale matrices refuse to be used
    asm.release()
    with pytest.raises(RuntimeError):
        A2.device_free_spd()
    with pytest.raises(RuntimeError):
        solver_mod.get_A(p)
