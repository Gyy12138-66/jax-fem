"""pyamg-backed custom_solver: Dirichlet-row stripping, rigid-body near-nullspace,
hierarchy reuse and the direct fallback, on small synthetic systems."""
import unittest
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

try:
    import pyamg  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    pyamg = None

from jax_fem_am.solvers import amg


class FakePetscMat:
    def __init__(self, A: sp.csr_matrix):
        self.A = A.tocsr()
        self.A.sort_indices()

    def getValuesCSR(self):
        return self.A.indptr, self.A.indices, self.A.data

    def getSize(self):
        return self.A.shape


def _pin_rows(A: sp.csr_matrix, rows) -> sp.csr_matrix:
    """Row elimination the jax-fem way: diag 1, no off-diagonal entries in the row."""
    A = A.tolil()
    for r in rows:
        A.rows[r] = [r]
        A.data[r] = [1.0]
    return A.tocsr()


def _grid_coords(shape):
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


@unittest.skipIf(pyamg is None, "pyamg not installed")
class PyamgKrylovSolverTest(unittest.TestCase):
    def _vector_system(self, shape=(5, 5, 5), pinned_nodes=(0, 1, 2, 3)):
        P = pyamg.gallery.poisson(shape, format="csr")
        A = sp.kron(P, sp.identity(3), format="csr")
        rows = [3 * n + c for n in pinned_nodes for c in range(3)]
        A = _pin_rows(A, rows)
        rng = np.random.default_rng(0)
        b = rng.standard_normal(A.shape[0])
        b[rows] = 0.25  # prescribed values
        x0 = np.zeros(A.shape[0])
        x0[rows] = b[rows]
        return A, b, x0, rows, _grid_coords(shape)

    def test_vector_problem_with_rigid_body_modes(self):
        A, b, x0, rows, coords = self._vector_system()
        solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=300, max_coarse=20, verbose=False)
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-8)
        np.testing.assert_allclose(x[rows], b[rows])
        self.assertEqual(solver.stats["calls"], 1)
        self.assertEqual(solver.stats["pattern_rebuilds"], 1)
        self.assertEqual(solver.stats["fallbacks"], 0)
        self.assertEqual(solver._cache.hierarchy_info["near_nullspace"], "rigid_body")
        self.assertGreaterEqual(solver._cache.hierarchy_info["levels"], 2)
        self.assertGreater(solver.stats["last_iterations"], 0)

    def test_hierarchy_reused_for_same_pattern_and_rebuilt_on_change(self):
        A, b, x0, rows, coords = self._vector_system()
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20)
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        solver(FakePetscMat(A), b, x0, {})
        # same pattern, free rows scaled (Dirichlet rows untouched) -> hierarchy reused
        coo = A.tocoo()
        scale = np.where(np.isin(coo.row, rows), 1.0, 1.7)
        A_scaled = sp.csr_matrix((coo.data * scale, (coo.row, coo.col)), shape=A.shape)
        b_scaled = b.copy()
        free = np.setdiff1d(np.arange(A.shape[0]), rows)
        b_scaled[free] *= 1.7
        x = solver(FakePetscMat(A_scaled), b_scaled, x0, {})
        self.assertLess(np.linalg.norm(A_scaled @ x - b_scaled) / np.linalg.norm(b_scaled), 1e-7)
        self.assertEqual(solver.stats["pattern_rebuilds"], 1)
        self.assertEqual(solver.stats["calls"], 2)
        # different pinned set -> new pattern
        A2, b2, x02, rows2, _ = self._vector_system(pinned_nodes=(0, 1, 2, 3, 4, 5))
        x = solver(FakePetscMat(A2), b2, x02, {})
        self.assertEqual(solver.stats["pattern_rebuilds"], 2)
        self.assertLess(np.linalg.norm(A2 @ x - b2) / np.linalg.norm(b2), 1e-7)

    def test_scalar_problem_uses_constant_near_nullspace(self):
        P = pyamg.gallery.poisson((8, 8, 8), format="csr")
        rows = list(range(10))
        A = _pin_rows(P, rows)
        b = np.ones(A.shape[0])
        x0 = np.zeros(A.shape[0])
        x0[rows] = 1.0
        solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=200, max_coarse=20)
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=_grid_coords((8, 8, 8)), vec=1)]))
        with self.assertWarns(RuntimeWarning):
            x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-8)
        self.assertEqual(solver._cache.hierarchy_info["near_nullspace"], "constant")

    def test_non_convergence_without_fallback_raises(self):
        A, b, x0, rows, coords = self._vector_system()
        solver = amg.PyamgKrylovSolver(tol=1e-12, maxiter=1, max_coarse=20, fallback="none")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        with self.assertRaises(RuntimeError):
            solver(FakePetscMat(A), b, x0, {})
        self.assertEqual(solver.stats["fallbacks"], 1)
        self.assertEqual(solver.stats["hierarchy_rebuilds"], 0)

    def test_jax_device_path_matches_cpu_jacobi_path(self):
        A, b, x0, rows, coords = self._vector_system()
        problem = SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)])
        cpu = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, smoother="jacobi")
        gpu = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, smoother="jacobi", device="gpu")
        cpu.bind_problem(problem)
        gpu.bind_problem(problem)
        xc = cpu(FakePetscMat(A), b, x0, {})
        xg = gpu(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ xg - b) / np.linalg.norm(b), 1e-7)
        np.testing.assert_allclose(xg[rows], b[rows])
        np.testing.assert_allclose(xg, xc, rtol=1e-4, atol=1e-9)
        self.assertEqual(gpu.stats["fallbacks"], 0)
        self.assertEqual(gpu.stats["device"], "jax")
        self.assertGreaterEqual(gpu._cache.hierarchy_info["levels"], 2)
        # same hierarchy + same smoother => same Krylov path up to rounding
        self.assertLessEqual(abs(cpu.stats["last_iterations"] - gpu.stats["last_iterations"]), 2)
        # second call: hierarchy and device structure reused
        gpu(FakePetscMat(A), b, x0, {})
        self.assertEqual(gpu.stats["pattern_rebuilds"], 1)
        self.assertEqual(gpu.stats["hierarchy_rebuilds"], 0)
        self.assertEqual(gpu.stats["calls"], 2)

    def test_partially_pinned_nodes_keep_rigid_body_modes(self):
        # pin single components (like the release anchors), not whole nodes
        P = pyamg.gallery.poisson((5, 5, 5), format="csr")
        A = sp.kron(P, sp.identity(3), format="csr")
        rows = [0, 4, 8, 3 * 7 + 2, 3 * 20 + 1]
        A = _pin_rows(A, rows)
        b = np.random.default_rng(1).standard_normal(A.shape[0])
        b[rows] = 0.0
        x0 = np.zeros(A.shape[0])
        solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=300, max_coarse=20, device="jax")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=_grid_coords((5, 5, 5)), vec=3)]))
        x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-8)
        self.assertEqual(solver._cache.hierarchy_info["near_nullspace"], "rigid_body")
        B = amg.rigid_body_modes_for_dofs(_grid_coords((5, 5, 5)), np.array([0, 1, 2, 5]))
        self.assertEqual(B.shape, (4, 6))
        np.testing.assert_allclose(B[:3], amg.rigid_body_modes(_grid_coords((5, 5, 5))[[0, 1]])[:3])

    def test_jax_device_scalar_problem(self):
        P = pyamg.gallery.poisson((8, 8, 8), format="csr")
        rows = list(range(10))
        A = _pin_rows(P, rows)
        b = np.ones(A.shape[0])
        x0 = np.zeros(A.shape[0])
        x0[rows] = 1.0
        solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=200, max_coarse=20, near_nullspace="constant", device="jax")
        x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-8)

    def test_jax_device_rejects_unsupported_options(self):
        with self.assertRaises(ValueError):
            amg.PyamgKrylovSolver(device="gpu", smoother="block_gauss_seidel")
        with self.assertRaises(ValueError):
            amg.PyamgKrylovSolver(device="gpu", method="bicgstab")
        with self.assertRaises(ValueError):
            amg.PyamgKrylovSolver(device="tpu")

    def test_block_gauss_seidel_cpu_smoother(self):
        A, b, x0, rows, coords = self._vector_system()
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, smoother="block_gauss_seidel")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-7)
        self.assertIn("block_gs", solver.label)

    def test_adaptive_rebuild_drops_a_stale_hierarchy(self):
        A, b, x0, rows, coords = self._vector_system()
        problem = SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)])
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=2000, max_coarse=20, rebuild_iter_factor=1.5)
        solver.bind_problem(problem)
        solver(FakePetscMat(A), b, x0, {})
        fresh = solver._cache.fresh_iterations
        self.assertGreater(fresh, 0)
        # same pattern, strongly anisotropic free block -> hierarchy no longer fits
        coo = A.tocoo()
        w = np.where(np.isin(coo.row, rows), 1.0, np.where(coo.row % 3 == coo.col % 3, 1.0, 0.02))
        scale = np.where((coo.row // 3 == coo.col // 3) | np.isin(coo.row, rows), 1.0, 40.0 ** ((coo.row % 5) / 4.0))
        A2 = sp.csr_matrix((coo.data * w * scale, (coo.row, coo.col)), shape=A.shape)
        A2 = 0.5 * (A2 + A2.T)
        A2 = _pin_rows(A2, rows)
        x = solver(FakePetscMat(A2), b, x0, {})
        self.assertLess(np.linalg.norm(A2 @ x - b) / np.linalg.norm(b), 1e-7)
        if solver.stats['last_iterations'] > 1.5 * fresh:
            self.assertEqual(solver.stats['adaptive_rebuilds'], 1)
            self.assertIsNone(solver._cache.hierarchy)
            solver(FakePetscMat(A2), b, x0, {})
            self.assertIsNotNone(solver._cache.hierarchy)
            self.assertLessEqual(solver.stats['last_iterations'], solver.stats['iterations'])
        # rule disabled -> nothing dropped
        off = amg.PyamgKrylovSolver(tol=1e-8, maxiter=2000, max_coarse=20, rebuild_iter_factor=0)
        off.bind_problem(problem)
        off(FakePetscMat(A), b, x0, {})
        off(FakePetscMat(A2), b, x0, {})
        self.assertEqual(off.stats['adaptive_rebuilds'], 0)

    def test_all_rows_pinned_returns_rhs(self):
        A = sp.identity(6, format="csr")
        b = np.arange(6, dtype=float)
        solver = amg.PyamgKrylovSolver()
        np.testing.assert_array_equal(solver(FakePetscMat(A), b, None, {}), b)


if __name__ == "__main__":
    unittest.main()


import pytest as _pytest_lane
pytestmark = _pytest_lane.mark.solver


@unittest.skipIf(pyamg is None, "pyamg not installed")
class PyamgKernelLifetimeTest(unittest.TestCase):
    """Fix A for the v159 stall (2026-09-08): jit kernel sets are owned by the
    pattern slot and released with it, so executables do not accumulate one
    per activation layer."""

    def _system(self, shape, pinned_nodes=(0, 1, 2, 3)):
        P = pyamg.gallery.poisson(shape, format="csr")
        A = sp.kron(P, sp.identity(3), format="csr")
        rows = [3 * n + c for n in pinned_nodes for c in range(3)]
        A = _pin_rows(A, rows)
        rng = np.random.default_rng(1)
        b = rng.standard_normal(A.shape[0])
        b[rows] = 0.25
        x0 = np.zeros(A.shape[0])
        x0[rows] = b[rows]
        return A, b, x0, _grid_coords(shape)

    def test_jax_path_releases_previous_pattern_kernels(self):
        import gc
        import weakref

        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax")
        dead = []
        for k, shape in enumerate([(4, 4, 4), (5, 4, 4), (6, 4, 4)]):
            A, b, x0, coords = self._system(shape)
            solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
            x = solver(FakePetscMat(A), b, x0, {})
            self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-7)
            self.assertIsNotNone(solver._cache.kernels)
            self.assertEqual(solver.stats["compiled_variants"], k + 1)
            if dead:
                gc.collect()
                self.assertTrue(all(ref() is None for ref in dead), "previous pattern kernels still alive")
            dead.append(weakref.ref(solver._cache.kernels["pcg"]))
        self.assertEqual(solver.stats["pattern_rebuilds"], 3)
        self.assertEqual(solver.stats["fallbacks"], 0)
        self.assertTrue(solver.stats["rss_mb"] is None or solver.stats["rss_mb"] > 0)

    def test_same_pattern_keeps_one_kernel_set(self):
        A, b, x0, coords = self._system((5, 4, 4))
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        solver(FakePetscMat(A), b, x0, {})
        first = solver._cache.kernels
        solver(FakePetscMat(A), b, x0, {})
        self.assertIs(solver._cache.kernels, first)
        self.assertEqual(solver.stats["compiled_variants"], 1)

    def test_cpu_path_never_builds_jax_kernels(self):
        A, b, x0, coords = self._system((5, 4, 4))
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20)
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        solver(FakePetscMat(A), b, x0, {})
        self.assertIsNone(solver._cache.kernels)
        self.assertEqual(solver.stats["compiled_variants"], 0)

    def test_registry_accepts_clear_caches_option(self):
        from jax_fem_am.solvers import linear

        spec = linear.parse_linear_solver_spec({"backend": "pyamg", "clear_jax_caches_on_pattern": "yes"}) \
            if hasattr(linear, "parse_linear_solver_spec") else None
        if spec is not None:
            self.assertTrue(spec["clear_jax_caches_on_pattern"])
        solver = amg.PyamgKrylovSolver(clear_jax_caches_on_pattern=True)
        self.assertTrue(solver.clear_jax_caches_on_pattern)
