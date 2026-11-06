"""JAX_FEM_POSTSTEP_JIT (cudss-plus 2026-11-06): the jitted per-step return maps and the device contributing-cell
mask give the eager / host results."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jax_fem.generate_mesh import Mesh  # noqa: E402
from jax_fem_am.simulation import runner as driver  # noqa: E402
from jax_fem_am.process import activation  # noqa: E402

V03_PATH = Path(__file__).resolve().parents[2] / "jax_fem_am" / "simulation" / "stepper.py"


def load_fresh_v03(name):
    spec = importlib.util.spec_from_file_location(name, V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def box_mesh(n=2, h=1.0e-3):
    g = np.linspace(0.0, n * h, n + 1)
    pts = np.array([[x, y, z] for z in g for y in g for x in g])
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)  # noqa: E731
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                              idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)])
    return pts, np.asarray(cells, dtype=np.int64)


class PostStepJitTest(unittest.TestCase):
    def setUp(self):
        driver.REGISTRY.reset()
        self._flag = driver._POSTSTEP_JIT

    def tearDown(self):
        driver._POSTSTEP_JIT = self._flag

    def _problem(self, base):
        pts, cells = box_mesh()
        mech = base.ThermoMechanical(mesh=Mesh(pts, cells, ele_type="HEX8"), vec=3, dim=3, ele_type="HEX8",
                                     quadrature_order=2, dirichlet_bc_info=None,
                                     additional_info=("j2_plastic", None, 0.0, 0.0, (), True, None))
        nc, nq = len(cells), mech.fes[0].num_quads
        rng = np.random.default_rng(0)
        full = lambda v: jnp.full((nc, nq, 1), v)  # noqa: E731
        params = [full(500.0) + 50.0 * jnp.asarray(rng.random((nc, nq, 1))),   # T
                  full(-30.0), jnp.asarray((rng.random((nc, 1, 1)) < 0.75) * np.ones((nc, nq, 1))),
                  full(190.0e9), full(1.4e-5), full(0.3), full(60.0e6), full(1.0e8),
                  jnp.asarray(1e-3 * rng.random((nc, nq, 1)))]
        strain = np.diag([0.02, -0.012, -0.008]) + 0.004 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        u = jnp.asarray(pts @ strain.T + 1e-6 * rng.standard_normal(pts.shape))
        return mech, params, u

    def _run(self, flag, base):
        driver._POSTSTEP_JIT = flag
        mech, params, u = self._problem(base)
        out = mech.compute_cell_stress(u, params)
        eqp = mech.compute_eqp_update(u, params)
        return {k: np.asarray(v) for k, v in out.items()}, np.asarray(eqp), np.asarray(mech._eps_p_state), mech

    def test_jitted_return_maps_match_the_eager_ones(self):
        base = load_fresh_v03("v03_poststep_jit_test")
        driver.install_v06_adapter(base)
        ref_out, ref_eqp, ref_epsp, _ = self._run(False, base)
        out, eqp, epsp, mech = self._run(True, base)
        self.assertGreater(float(np.max(ref_eqp - 0.0)), 1e-6)       # the case is plastic somewhere
        for k in ref_out:
            np.testing.assert_allclose(out[k], ref_out[k], rtol=1e-12, atol=1e-12 * np.abs(ref_out[k]).max(),
                                       err_msg=k)
        np.testing.assert_allclose(eqp, ref_eqp, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(epsp, ref_epsp, rtol=1e-12, atol=1e-18)
        self.assertEqual(set(mech.__dict__.get("_v06_batched_cache", {})), {"stress", "eqp"})
        # a second step reuses the compiled executables
        f = mech._v06_batched_cache["eqp"]
        mech.compute_eqp_update(mech._u_grads and None or jnp.zeros((27, 3)), self._problem(base)[1])
        self.assertIs(mech._v06_batched_cache["eqp"], f)
        self.assertEqual(f._cache_size(), 1)


class DeviceContributingMaskTest(unittest.TestCase):
    def test_same_mask_as_the_host_reduction(self):
        rng = np.random.default_rng(1)
        fields = [rng.random((50, 8, 1)) * (rng.random((50, 1, 1)) < 0.5),
                  rng.random((50, 8)) * (rng.random((50, 8)) < 0.1), np.zeros(50)]
        fields[2][7] = 3.0
        old = activation._POSTSTEP_DEVICE
        try:
            activation._POSTSTEP_DEVICE = False
            ref = activation.contributing_cell_mask(*fields)
            activation._POSTSTEP_DEVICE = True
            dev = activation.contributing_cell_mask(*[jnp.asarray(f) for f in fields])
            host_in = activation.contributing_cell_mask(*fields)          # host inputs keep the host path
        finally:
            activation._POSTSTEP_DEVICE = old
        self.assertEqual(dev.dtype, np.bool_)
        np.testing.assert_array_equal(dev, ref)
        np.testing.assert_array_equal(host_in, ref)


if __name__ == "__main__":
    unittest.main()
