"""cudss-plus 2026-11-05 (M1a): the mechanics problem reuses the thermal problem's per-cell geometry arrays.

Two problems on the same mesh / element / quadrature (vec 1 and vec 3, as thermal and mechanics) have bitwise
identical shape_grads, JxW, v_grads_JxW and physical_quad_points; after share_cell_geometry the second one holds
the first one's buffers and its tangent is unchanged bit for bit (the residual too on the CPU; on a GPU the
residual scatter-add is not deterministic even between two calls of the same code, so there it is compared to a
few ulps). Arrays that differ are not shared, and JAX_FEM_SHARE_GEOMETRY=0 turns sharing off.
"""
import numpy as onp

import jax
import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.problem import Problem
from jax_fem_am.simulation import device_memory


class _Scalar(Problem):
    def get_tensor_map(self):
        return lambda g: (1.0 + 0.1 * jnp.sum(g * g)) * g


class _Vector(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 2.0 * eps + 0.5 * jnp.trace(eps) * jnp.eye(3) + 0.3 * u_grad @ u_grad.T
        return stress


def _mesh():
    m = box_mesh(5, 4, 3, 1.0, 1.0, 1.0)
    return Mesh(m.points, m.cells_dict["hexahedron"])


def _newton(p, seed=0):
    rng = onp.random.default_rng(seed)
    sol = [jnp.asarray(1e-2 * rng.standard_normal((p.fes[0].num_total_nodes, p.fes[0].vec)))]
    res = p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    return onp.array(p.V), [onp.asarray(r) for r in res]


def _same_residual(a, b):
    if jax.default_backend() == "cpu":
        return onp.array_equal(a, b)
    eps = onp.finfo(onp.asarray(b).dtype).eps
    return onp.allclose(a, b, rtol=0.0, atol=64 * eps * float(onp.abs(b).max()))


def test_mechanics_reuses_thermal_geometry_bitwise():
    mesh = _mesh()
    thermal = _Scalar(mesh=mesh, vec=1, dim=3, ele_type="HEX8")
    mech = _Vector(mesh=mesh, vec=3, dim=3, ele_type="HEX8")
    ref_V, ref_res = _newton(mech)
    shared = device_memory.share_cell_geometry(thermal, mech)
    for name in device_memory.GEOMETRY_ATTRS:
        assert name in shared
        assert getattr(mech, name) is getattr(thermal, name)
    for name in device_memory.FE_GEOMETRY_ATTRS:
        assert getattr(mech.fes[0], name) is getattr(thermal.fes[0], name)
    assert sum(shared.values()) > 0
    V, res = _newton(mech)
    assert onp.array_equal(V, ref_V)
    assert all(_same_residual(a, b) for a, b in zip(res, ref_res))


def test_different_arrays_are_not_shared():
    mesh = _mesh()
    a = _Scalar(mesh=mesh, vec=1, dim=3, ele_type="HEX8")
    b = _Scalar(mesh=mesh, vec=1, dim=3, ele_type="HEX8")
    b.JxW = b.JxW * 2.0
    shared = device_memory.share_cell_geometry(a, b)
    assert "JxW" not in shared and b.JxW is not a.JxW
    assert "shape_grads" in shared and b.shape_grads is a.shape_grads


def test_switch_off(monkeypatch):
    mesh = _mesh()
    a = _Scalar(mesh=mesh, vec=1, dim=3, ele_type="HEX8")
    b = _Vector(mesh=mesh, vec=3, dim=3, ele_type="HEX8")
    monkeypatch.setenv("JAX_FEM_SHARE_GEOMETRY", "0")
    assert device_memory.share_cell_geometry(a, b) == {}
    assert b.shape_grads is not a.shape_grads
