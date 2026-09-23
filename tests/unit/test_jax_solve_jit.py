"""jax_solve(jit_solve=True): the Krylov solve is compiled once per sparsity
pattern instead of on every call (BUG_FIX.md sections 11.13 / 11.14)."""
import unittest

import numpy as np
import scipy.sparse as sp

try:
    import pyamg  # only for the Poisson gallery
except ImportError:  # pragma: no cover - depends on the environment
    pyamg = None

from jax_fem import solver as jf_solver
from jax_fem_am.solvers import linear as registry


class FakePetscMat:
    def __init__(self, A: sp.csr_matrix):
        self.A = A.tocsr()
        self.A.sort_indices()

    def getValuesCSR(self):
        return self.A.indptr, self.A.indices, self.A.data

    def getSize(self):
        return self.A.shape


def _system(scale=1.0, seed=0, shape=(9, 9, 9)):
    P = pyamg.gallery.poisson(shape, format="csr") + 0.1 * sp.identity(int(np.prod(shape)), format="csr")
    A = (scale * P).tocsr()
    b = np.random.default_rng(seed).standard_normal(A.shape[0])
    return A, b


@unittest.skipIf(pyamg is None, "pyamg not installed")
class JaxSolveJitTest(unittest.TestCase):
    def test_jitted_solve_matches_the_eager_one(self):
        A, b = _system()
        for method in ("cg", "bicgstab"):
            for x0 in (None, np.zeros(A.shape[0])):
                kw = dict(method=method, tol=1e-10, atol=1e-12, maxiter=500)
                xe = np.asarray(jf_solver.jax_solve(FakePetscMat(A), b, x0, True, **kw))
                xj = np.asarray(jf_solver.jax_solve(FakePetscMat(A), b, x0, True, jit_solve=True, **kw))
                self.assertLess(np.linalg.norm(A @ xj - b) / np.linalg.norm(b), 1e-8, (method, x0 is None))
                np.testing.assert_allclose(xj, xe, rtol=1e-9, atol=1e-12)

    def test_one_compilation_for_a_sequence_of_tangents(self):
        # same sparsity pattern, changing values: exactly what a build produces
        A0, b0 = _system()
        jf_solver.jax_solve(FakePetscMat(A0), b0, None, True, method="cg", tol=1e-8, atol=1e-12,
                            maxiter=500, jit_solve=True)
        before = jf_solver._jitted_krylov._cache_size()
        for k in range(1, 6):
            A, b = _system(scale=1.0 + 0.3 * k, seed=k)
            x = jf_solver.jax_solve(FakePetscMat(A), b, None, True, method="cg", tol=1e-8, atol=1e-12,
                                    maxiter=500, jit_solve=True)
            self.assertLess(np.linalg.norm(A @ np.asarray(x) - b) / np.linalg.norm(b), 1e-6)
        self.assertEqual(jf_solver._jitted_krylov._cache_size(), before)

    def test_counter_and_dispatch_through_linear_solver(self):
        A, b = _system()
        timing = {}
        opts = {"jax_solver": {"method": "cg", "precond": True, "tol": 1e-8, "atol": 1e-12, "maxiter": 500, "jit": True}}
        x = jf_solver.linear_solver(FakePetscMat(A), b, None, opts, timing=timing)
        self.assertLess(np.linalg.norm(A @ np.asarray(x) - b) / np.linalg.norm(b), 1e-6)
        self.assertEqual(timing.get("counts", timing).get("jax_jit_solves", timing.get("jax_jit_solves")), 1)
        timing = {}
        opts["jax_solver"].pop("jit")
        jf_solver.linear_solver(FakePetscMat(A), b, None, opts, timing=timing)
        self.assertFalse(timing.get("counts", timing).get("jax_jit_solves", timing.get("jax_jit_solves")))

    def test_unknown_method_is_still_rejected(self):
        A, b = _system()
        for jit in (False, True):
            with self.assertRaises(ValueError):
                jf_solver.jax_solve(FakePetscMat(A), b, None, True, method="sor", jit_solve=jit)


class JaxJitRegistryTest(unittest.TestCase):
    def test_key_is_optional_validated_and_reaches_the_block_and_the_label(self):
        # default ON since 2026-09-23 (the E0j setting)
        spec = registry.normalize_linear_solver_spec({"backend": "jax", "method": "cg"})
        self.assertIs(spec["jit"], True)
        block = registry.build_linear_block({"backend": "jax", "method": "cg"})
        self.assertIs(block["jax_solver"]["jit"], True)
        self.assertIn("jit=True", registry.linear_block_label(block))
        # an explicit false still turns it off (pinned pre-2026-09-23 configs)
        off = registry.build_linear_block({"backend": "jax", "method": "cg", "jit": False})
        self.assertNotIn("jit=True", registry.linear_block_label(off))

        spec = registry.normalize_linear_solver_spec({"backend": "jax", "method": "cg", "jit": True})
        self.assertIs(spec["jit"], True)
        block = registry.build_linear_block({"backend": "jax", "method": "cg", "precond": "jacobi", "jit": True})
        self.assertIs(block["jax_solver"]["jit"], True)
        self.assertIn("jit=True", registry.linear_block_label(block))
        self.assertNotIn("jit", registry.build_linear_block({"backend": "jax", "method": "cg", "jit": False})["jax_solver"])
        with self.assertRaises(ValueError):
            registry.normalize_linear_solver_spec({"backend": "jax", "jit": "sometimes"})
        with self.assertRaises(ValueError):
            registry.normalize_linear_solver_spec({"backend": "pardiso", "jit": True})


if __name__ == "__main__":
    unittest.main()


import pytest as _pytest_lane
pytestmark = _pytest_lane.mark.solver
