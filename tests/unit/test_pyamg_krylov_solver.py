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


def _pin_rows_keep_structure(A: sp.csr_matrix, rows) -> sp.csr_matrix:
    """Pinning the way the AM assembler does it: the sparsity structure of the
    whole mesh never changes, entries in pinned rows/columns become explicit
    zeros and the pinned diagonal becomes 1 (inactive cells have zero stiffness)."""
    A = A.tocsr(copy=True)
    A.sort_indices()
    pinned = np.zeros(A.shape[0], dtype=bool)
    pinned[list(rows)] = True
    r = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    kill = pinned[r] | pinned[A.indices]
    data = np.where(kill, 0.0, A.data)
    diag = np.flatnonzero((r == A.indices) & pinned[r])
    data[diag] = 1.0
    return sp.csr_matrix((data, A.indices, A.indptr), shape=A.shape)


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

        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax", shape_mode="free")
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
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax", shape_mode="free")
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


@unittest.skipIf(pyamg is None, "pyamg not installed")
class PyamgFixedShapeTest(unittest.TestCase):
    """Fix B for the v159 stall (BUG_FIX.md, 2026-09-14): ``shape_mode="full"``
    keeps every jit array shape fixed for the whole build, so a growing
    activation front costs one compiled variant and no per-shape host residue.
    The full-shape solve is the reduced solve in disguise: same free equations,
    pinned rows exact."""

    def _system(self, shape, pinned_nodes=(0, 1, 2, 3), seed=1):
        P = pyamg.gallery.poisson(shape, format="csr")
        A = sp.kron(P, sp.identity(3), format="csr")
        rows = [3 * n + c for n in pinned_nodes for c in range(3)]
        A = _pin_rows(A, rows)
        rng = np.random.default_rng(seed)
        b = rng.standard_normal(A.shape[0])
        b[rows] = 0.25
        x0 = np.zeros(A.shape[0])
        x0[rows] = b[rows]
        return A, b, x0, rows, _grid_coords(shape)

    def test_gpu_path_defaults_to_full_shape_and_cpu_to_free(self):
        self.assertEqual(amg.PyamgKrylovSolver(device="gpu").shape_mode, "full")
        self.assertEqual(amg.PyamgKrylovSolver(device="cpu").shape_mode, "free")
        self.assertEqual(amg.PyamgKrylovSolver(device="gpu", shape_mode="free").shape_mode, "free")
        with self.assertRaises(ValueError):
            amg.PyamgKrylovSolver(shape_mode="padded")
        with self.assertRaises(ValueError):
            amg.PyamgKrylovSolver(fixed_levels=1)

    def test_full_shape_matches_free_shape_and_keeps_pinned_values(self):
        A, b, x0, rows, coords = self._system((5, 5, 5))
        problem = SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)])
        free_solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=300, max_coarse=20, device="jax", shape_mode="free")
        full_solver = amg.PyamgKrylovSolver(tol=1e-9, maxiter=300, max_coarse=20, device="jax", shape_mode="full")
        free_solver.bind_problem(problem)
        full_solver.bind_problem(problem)
        xf = free_solver(FakePetscMat(A), b, x0, {})
        xF = full_solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ xF - b) / np.linalg.norm(b), 1e-8)
        np.testing.assert_array_equal(xF[rows], b[rows])
        np.testing.assert_allclose(xF, xf, rtol=1e-6, atol=1e-10)
        # same Krylov sequence up to rounding
        self.assertLessEqual(abs(full_solver.stats["last_iterations"] - free_solver.stats["last_iterations"]), 2)
        self.assertEqual(full_solver.stats["shape_mode"], "full")
        self.assertEqual(full_solver.stats["structure_rebuilds"], 1)
        self.assertEqual(full_solver.stats["fallbacks"], 0)
        # the level count is padded to fixed_levels, the padded sizes are recorded
        self.assertEqual(len(full_solver._cache.jax_levels), full_solver.fixed_levels - 1)
        self.assertEqual(len(full_solver._cache.hierarchy_info["padded_levels"]), full_solver.fixed_levels)

    def _assembler_system(self, shape, n_pinned_nodes, seed):
        """Whole-mesh structure fixed, the inactive (pinned) set shrinks: what the
        AM assembler hands the solver layer after layer."""
        P = pyamg.gallery.poisson(shape, format="csr")
        A = sp.kron(P, sp.identity(3), format="csr")
        rows = [3 * n + c for n in range(n_pinned_nodes) for c in range(3)]
        A = _pin_rows_keep_structure(A, rows)
        rng = np.random.default_rng(seed)
        b = rng.standard_normal(A.shape[0])
        b[rows] = 0.25
        x0 = np.zeros(A.shape[0])
        x0[rows] = b[rows]
        return A, b, x0, rows

    def test_growing_activation_front_compiles_once(self):
        # same mesh structure, the pinned (inactive) set shrinks layer by layer
        shape = (6, 5, 5)
        coords = _grid_coords(shape)
        problem = SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)])
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax", shape_mode="full")
        solver.bind_problem(problem)
        nnodes = coords.shape[0]
        for k, n_pinned in enumerate((nnodes // 2, nnodes // 3, nnodes // 5, 4)):
            A, b, x0, rows = self._assembler_system(shape, n_pinned, seed=k)
            x = solver(FakePetscMat(A), b, x0, {})
            self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-7)
            np.testing.assert_array_equal(x[rows], b[rows])
            self.assertEqual(solver.stats["pattern_rebuilds"], k + 1)
            self.assertEqual(solver.stats["structure_rebuilds"], 1)
            self.assertEqual(solver.stats["compiled_variants"], 1, "a new pattern must not recompile")
            self.assertIsNone(solver._cache.kernels)
            self.assertIsNotNone(solver._struct.kernels)
        self.assertEqual(solver.stats["capacity_growths"], 0)

    def test_structure_change_rebuilds_structure_and_recompiles(self):
        # a different mesh (or a different stored sparsity) is a new structure:
        # the device index arrays and the kernel set are rebuilt exactly then
        coords = _grid_coords((5, 5, 5))
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax", shape_mode="full")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        A, b, x0, rows = self._assembler_system((5, 5, 5), 4, seed=0)
        solver(FakePetscMat(A), b, x0, {})
        A2, b2, x02, rows2, _ = self._system((5, 5, 5))  # _pin_rows drops entries: new structure
        x = solver(FakePetscMat(A2), b2, x02, {})
        self.assertLess(np.linalg.norm(A2 @ x - b2) / np.linalg.norm(b2), 1e-7)
        self.assertEqual(solver.stats["structure_rebuilds"], 2)
        self.assertEqual(solver.stats["compiled_variants"], 2)

    def test_capacity_overflow_grows_once_and_still_solves(self):
        A, b, x0, rows, coords = self._system((5, 5, 5))
        solver = amg.PyamgKrylovSolver(tol=1e-8, maxiter=300, max_coarse=20, device="jax", shape_mode="full")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        # first solve builds the structure with default capacities; shrink one
        # of them so the next hierarchy build overflows
        solver(FakePetscMat(A), b, x0, {})
        solver._struct.caps["P0"] = 8
        solver._cache.hierarchy = None
        with self.assertWarns(RuntimeWarning):
            x = solver(FakePetscMat(A), b, x0, {})
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-7)
        self.assertEqual(solver.stats["capacity_growths"], 1)
        self.assertGreater(solver._struct.caps["P0"], 8)
        self.assertEqual(solver.stats["compiled_variants"], 2)

    def test_bucket_mode_is_a_template(self):
        A, b, x0, rows, coords = self._system((4, 4, 4))
        solver = amg.PyamgKrylovSolver(device="jax", shape_mode="bucket")
        solver.bind_problem(SimpleNamespace(fes=[SimpleNamespace(points=coords, vec=3)]))
        with self.assertRaises(NotImplementedError):
            solver(FakePetscMat(A), b, x0, {})
        ladder = [amg.bucket_capacity(nf, ratio=1.25, base=4096) for nf in (1000, 4096, 5000, 100000, 818394)]
        self.assertEqual(ladder[0], ladder[1])
        self.assertTrue(all(a <= b for a, b in zip(ladder, ladder[1:])))
        self.assertTrue(all(cap >= nf for cap, nf in zip(ladder, (1000, 4096, 5000, 100000, 818394))))

    def test_default_capacities_hold_the_v159_hierarchy(self):
        caps = amg.default_capacities(904368, 4, max_coarse=1000)
        # measured at full activation: levels [818394, 60654, 1734, 78],
        # P0 nnz 18.5M, A1 nnz 10.3M, P1 nnz 1.18M, A2 nnz 171k
        self.assertGreaterEqual(caps["rows1"], 60654)
        self.assertGreaterEqual(caps["rows2"], 1734)
        # early in the build the hierarchy has 3 levels and its coarsest level
        # (<= max_coarse dofs, 552 seen at nf=305k) is duplicated into level 3
        self.assertGreaterEqual(caps["rows3"], 1000)
        self.assertGreaterEqual(caps["P0"], 18497718)
        self.assertGreaterEqual(caps["A1"], 10294956)
        self.assertGreaterEqual(caps["P1"], 1178964)
        self.assertGreaterEqual(caps["A2"], 170820)

    def test_registry_accepts_shape_keys(self):
        from jax_fem_am.solvers import linear

        spec = linear.normalize_linear_solver_spec(
            {"backend": "pyamg", "device": "gpu", "shape_mode": "FULL", "fixed_levels": "4", "bucket_ratio": 1.5}
        )
        self.assertEqual(spec["shape_mode"], "full")
        self.assertEqual(spec["fixed_levels"], 4)
        self.assertEqual(spec["bucket_ratio"], 1.5)
        self.assertEqual(linear.normalize_linear_solver_spec({"backend": "pyamg"})["shape_mode"], "auto")
        with self.assertRaises(ValueError):
            linear.normalize_linear_solver_spec({"backend": "pyamg", "shape_mode": "padded"})
