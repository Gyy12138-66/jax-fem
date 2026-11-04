"""cudss-plus 2026-11-04: host path of the element tangents and jitted chunk slicing (options A+B, dispatch fix).

The reused-buffer path (JAX_FEM_TANGENT_HOST_PATH=pinned, the default) and the jitted chunk slicing
(JAX_FEM_CHUNK_SLICE=jit, the default) must give exactly the values and order of the original code
(legacy / eager): same Problem.V, same residual, for several chunk counts and with a face block. On the CPU
backend the copies are pageable; under ``-m gpu`` the same checks run through pinned host memory.
"""
import numpy as onp
import pytest

import jax
import jax.numpy as jnp

import jax_fem.problem as problem_mod
from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.problem import Problem, _TangentHostWriter


class _Hyper(Problem):
    """Small nonlinear vector problem with a u-dependent face term, so cell and face tangents both vary."""

    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 2.0 * eps + 0.5 * jnp.trace(eps) * jnp.eye(3) + 0.3 * u_grad @ u_grad.T
        return stress

    def get_surface_maps(self):
        def traction(u, x):
            return jnp.array([0.0, 0.0, 1.0]) + 0.7 * u * u
        return [traction]


def _problem(num_cuts):
    m = box_mesh(7, 5, 4, 1.0, 1.0, 1.0)
    mesh = Mesh(m.points, m.cells_dict["hexahedron"])
    p = _Hyper(mesh=mesh, vec=3, dim=3, ele_type="HEX8",
               location_fns=[lambda x: jnp.isclose(x[2], 1.0, atol=1e-5)])
    p.cell_assembly_num_cuts = num_cuts
    return p


def _sol(p, seed=0):
    rng = onp.random.default_rng(seed)
    return [jnp.asarray(1e-2 * rng.standard_normal((p.fes[0].num_total_nodes, 3)))]


def _newton(p, sol):
    res = p.compute_newton_vars(sol, p.internal_vars, p.internal_vars_surfaces)
    return onp.array(p.V), [onp.asarray(r) for r in res]


def _check_paths(monkeypatch, num_cuts):
    p = _problem(num_cuts)
    monkeypatch.setattr(problem_mod, "TANGENT_HOST_PATH", "legacy")
    monkeypatch.setattr(problem_mod, "CHUNK_SLICE", "eager", raising=False)
    ref = {seed: _newton(p, _sol(p, seed)) for seed in (0, 1)}
    monkeypatch.setattr(problem_mod, "TANGENT_HOST_PATH", "pinned")
    monkeypatch.setattr(problem_mod, "CHUNK_SLICE", "jit", raising=False)
    for seed in (0, 1, 0):          # the buffer is reused and overwritten build after build
        V, res = _newton(p, _sol(p, seed))
        V_ref, res_ref = ref[seed]
        assert V.shape == p.I.shape
        assert onp.array_equal(V, V_ref)
        assert len(res) == len(res_ref)
        for a, b in zip(res, res_ref):
            assert onp.array_equal(a, b)
    assert p.V is p._tangent_host_buffer
    return p


@pytest.mark.parametrize("num_cuts", [1, 3, 4])
def test_newton_vars_match_legacy(monkeypatch, num_cuts):
    _check_paths(monkeypatch, num_cuts)


@pytest.mark.parametrize("num_cuts", [1, 4])
def test_residual_chunk_slicing_matches_eager(monkeypatch, num_cuts):
    p = _problem(num_cuts)
    sol = _sol(p, 3)
    monkeypatch.setattr(problem_mod, "CHUNK_SLICE", "eager", raising=False)
    ref = [onp.asarray(r) for r in p.compute_residual(sol)]
    monkeypatch.setattr(problem_mod, "CHUNK_SLICE", "jit", raising=False)
    out = [onp.asarray(r) for r in p.compute_residual(sol)]
    for a, b in zip(out, ref):
        assert onp.array_equal(a, b)


def test_writer_rejects_wrong_total():
    class _P:
        I = onp.zeros(10)

    w = _TangentHostWriter(_P())
    w.write([onp.ones((2, 2))])
    with pytest.raises(ValueError):
        w.finish()
    with pytest.raises(ValueError):
        w.write([onp.ones((3, 3))])


def test_writer_residual_chunks_stay_on_device():
    class _P:
        I = onp.zeros(8)

    w = _TangentHostWriter(_P())
    out = w.vstack([jnp.ones((2, 3)), jnp.zeros((1, 3))])
    assert isinstance(out, jax.Array)
    assert onp.array_equal(onp.asarray(out), onp.vstack([onp.ones((2, 3)), onp.zeros((1, 3))]))
    assert not w.wrote_cells


@pytest.mark.gpu
def test_pinned_path_on_gpu(monkeypatch):
    if jax.default_backend() == "cpu":
        pytest.skip("needs a GPU backend")
    assert problem_mod._pinned_host_sharding(jnp.zeros(3)) is not None
    _check_paths(monkeypatch, 4)
