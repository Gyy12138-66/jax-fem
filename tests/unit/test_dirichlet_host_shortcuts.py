"""JAX_FEM_POSTSTEP_JIT host shortcuts of FE.Dirichlet_boundary_conditions give the device-evaluated arrays."""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from jax_fem import fe as fe_mod  # noqa: E402
from jax_fem.generate_mesh import Mesh  # noqa: E402
from jax_fem_am.mesh.model import make_box_locations  # noqa: E402
from jax_fem_am.physics.release import make_full_bottom_mechanics_bc  # noqa: E402
from jax_fem_am.process.activation import make_inactive_node_dirichlet_bc  # noqa: E402
from jax_fem_am.simulation.stepper import merge_dirichlet_bcs  # noqa: E402


def box(n=3, h=1e-3):
    g = np.linspace(0.0, n * h, n + 1)
    pts = np.array([[x, y, z] for z in g for y in g for x in g])
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)  # noqa: E731
    cells = [[idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k), idx(i, j, k + 1),
              idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
             for k in range(n) for j in range(n) for i in range(n)]
    return pts, np.asarray(cells)


class DirichletHostShortcutTest(unittest.TestCase):
    def setUp(self):
        self._flag = fe_mod._HOST_BC

    def tearDown(self):
        fe_mod._HOST_BC = self._flag

    def _bcs(self, pts):
        _, _, bottom, *_ = make_box_locations(pts, build_axis="z")
        rng = np.random.default_rng(0)
        inactive = rng.random(len(pts)) < 0.3
        mech = merge_dirichlet_bcs(make_full_bottom_mechanics_bc(bottom),
                                   make_inactive_node_dirichlet_bc(inactive, vec=3, value=0.0))

        def hot(_p, _v=812.5):
            return _v
        hot._jfa_const_value = 812.5
        therm = merge_dirichlet_bcs([[bottom], [0], [hot]],
                                    make_inactive_node_dirichlet_bc(inactive, vec=1, value=293.15))
        return mech, therm

    def test_same_arrays_as_the_device_evaluation(self):
        pts, cells = box()
        for vec, which in ((3, 0), (1, 1)):
            fe = fe_mod.FiniteElement(mesh=Mesh(pts, cells, ele_type="HEX8"), vec=vec, dim=3, ele_type="HEX8",
                                      quadrature_order=2, dirichlet_bc_info=None)
            bc = self._bcs(pts)[which]
            fe_mod._HOST_BC = False
            ref = fe.Dirichlet_boundary_conditions(bc)
            fe_mod._HOST_BC = True
            for _ in range(2):                                     # second pass: cached static location
                got = fe.Dirichlet_boundary_conditions(bc)
                for r_list, g_list in zip(ref, got):
                    self.assertEqual(len(r_list), len(g_list))
                    for r, g in zip(r_list, g_list):
                        self.assertEqual(np.asarray(r).dtype, np.asarray(g).dtype)
                        np.testing.assert_array_equal(np.asarray(g), np.asarray(r))
            self.assertTrue(fe.__dict__.get("_jfa_location_cache"))
            self.assertGreater(sum(len(v) for v in ref[0]), 0)


if __name__ == "__main__":
    unittest.main()
