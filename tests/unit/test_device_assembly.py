"""cudss-plus 2026-11-05 (M1c / K3): the thermal tangent assembled on the device instead of through PETSc.

A small nonlinear scalar problem with a mass term, a face term and Dirichlet rows is assembled both ways:
PETSc (setValuesCOO + zeroRows, the production path) and the device assembler (ffi k3_ragged + Dirichlet mask).
Checked: identical CSR structure, values equal up to the summation order (a few ulps of the entry scale),
Dirichlet rows as zeroRows leaves them, bitwise repeatable device values, the same CG solution through jax_solve,
and a CPU fallback (scipy spsolve) still works on the device matrix. Needs the CUDA backend (gpu marker).
"""
import numpy as onp
import pytest

import jax
import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.problem import Problem
import jax_fem.solver as solver_mod

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(jax.default_backend() != "gpu", reason="device assembly needs the CUDA backend"),
]


class _Heat(Problem):
    def get_tensor_map(self):
        return lambda g: (1.0 + 0.2 * jnp.sum(g * g)) * g

    def get_mass_map(self):
        def mass(u, x):
            return 0.5 * u + 0.1 * u ** 3
        return mass

    def get_surface_maps(self):
        return [lambda u, x: 0.3 * u + 0.05 * u * u]


def _problem():
    m = box_mesh(6, 5, 4, 1.0, 1.0, 1.0)
    mesh = Mesh(m.points, m.cells_dict["hexahedron"])
    return _Heat(mesh=mesh, vec=1, dim=3, ele_type="HEX8",
                 dirichlet_bc_info=[[lambda x: jnp.isclose(x[0], 0.0, atol=1e-5)], [0], [lambda x: 0.0]],
                 location_fns=[lambda x: jnp.isclose(x[2], 1.0, atol=1e-5)])


def _sol(p, seed=0):
    rng = onp.random.default_rng(seed)
    return [jnp.asarray(0.3 * rng.standard_normal((p.fes[0].num_total_nodes, 1)))]


def _both(seed=0):
    from jax_fem_am.solvers.device_assembly import attach_thermal_device_assembly

    p = _problem()
    sol = _sol(p, seed)
    p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    A_ref = solver_mod.get_A(p)
    ref = [onp.array(x) for x in A_ref.getValuesCSR()]
    asm = attach_thermal_device_assembly(p)
    p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    assert p.V is None and p.V_device is not None
    A_dev = solver_mod.get_A(p)
    assert p.V_device is None
    return p, asm, A_ref, ref, A_dev


def test_device_csr_matches_petsc_zero_rows():
    p, asm, A_ref, (indptr, indices, data), A_dev = _both()
    d_indptr, d_indices, d_data = A_dev.getValuesCSR()
    assert onp.array_equal(d_indptr, indptr) and onp.array_equal(d_indices, indices)
    scale = float(onp.abs(data).max())
    assert onp.max(onp.abs(d_data - data)) <= 8 * onp.finfo(float).eps * scale
    # Dirichlet rows: zero except a unit diagonal
    rows = onp.repeat(onp.arange(len(indptr) - 1), onp.diff(indptr))
    bc = onp.zeros(len(indptr) - 1, bool)
    for fe in p.fes:
        for i in range(len(fe.node_inds_list)):
            bc[onp.asarray(fe.node_inds_list[i]) * fe.vec + fe.vec_inds_list[i]] = True
    assert bc.any()
    on_bc = bc[rows]
    assert onp.array_equal(d_data[on_bc], onp.where(rows[on_bc] == indices[on_bc], 1.0, 0.0))
    # bitwise repeatable
    p.compute_newton_vars(_sol(p, 0), p.internal_vars, p.internal_vars_surfaces)
    again = solver_mod.get_A(p)
    assert onp.array_equal(onp.asarray(again.data), d_data)
    assert asm.calls == 2


def test_jax_solve_and_cpu_fallback_on_device_matrix():
    p, asm, A_ref, ref, A_dev = _both(seed=1)
    n = A_ref.getSize()[0]
    rng = onp.random.default_rng(2)
    b = rng.standard_normal(n)
    bc = onp.zeros(n, bool)
    for fe in p.fes:
        for i in range(len(fe.node_inds_list)):
            bc[onp.asarray(fe.node_inds_list[i]) * fe.vec + fe.vec_inds_list[i]] = True
    b[bc] = 0.0
    b = jnp.asarray(b)
    x0 = jnp.zeros(n)
    kw = dict(method="cg", tol=1e-12, atol=1e-12, maxiter=5000, check_residual=True, jit_solve=True)
    timing = {}
    x_ref = onp.asarray(solver_mod.jax_solve(A_ref, b, x0, True, **kw))
    x_dev = onp.asarray(solver_mod.jax_solve(A_dev, b, x0, True, timing=timing, **kw))
    assert onp.max(onp.abs(x_dev - x_ref)) <= 1e-9 * onp.max(onp.abs(x_ref))
    x_cpu = onp.asarray(solver_mod.scipy_spsolve(A_dev, onp.asarray(b)))
    assert onp.max(onp.abs(x_cpu - x_ref)) <= 1e-9 * onp.max(onp.abs(x_ref))
