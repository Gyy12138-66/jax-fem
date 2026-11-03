import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

try:
    import pypardiso  # noqa: F401
except ImportError as exc:  # pragma: no cover
    PARDISO_IMPORT_ERROR = exc
else:
    PARDISO_IMPORT_ERROR = None

from jax_fem_am.solvers import linear
from jax_fem_am.solvers import pardiso as pardiso_variant


class _Csr:
    def __init__(self, A):
        A = sp.csr_matrix(A)
        A.sort_indices()
        self.indptr, self.indices, self.data = A.indptr, A.indices, A.data.astype(np.float64)

    def getValuesCSR(self):
        return self.indptr, self.indices, self.data


def _whole_mesh_system(n, pinned, seed=0, shift=0.0):
    """SPD Laplacian-like matrix on n dofs; pinned rows become unit rows whose
    off-diagonal entries stay STORED as zeros (as in the v159 whole-mesh
    assembly), and the pattern of every row is the same full stencil."""
    rng = np.random.default_rng(seed)
    main = 4.0 + shift + rng.random(n)
    off = -rng.random(n - 1)
    off2 = -0.5 * rng.random(n - 3)
    K = sp.diags([main, off, off, off2, off2], [0, 1, -1, 3, -3], format="lil")
    K = sp.csr_matrix(K)
    K.sort_indices()
    rows = np.repeat(np.arange(n), np.diff(K.indptr))
    pin = np.zeros(n, dtype=bool)
    pin[pinned] = True
    zero = pin[rows] | pin[K.indices]
    data = K.data.copy()
    data[zero] = 0.0
    diag = rows == K.indices
    data[diag & pin[rows]] = 1.0
    return sp.csr_matrix((data, K.indices.copy(), K.indptr.copy()), shape=(n, n))


class _RecordingRaw:
    """No-MKL stand-in: records phases and solves the given CSR exactly."""

    events = []
    instances = []

    def __init__(self, single_precision=False):
        self.mtype = 11
        self.iparm = np.zeros(64, dtype=np.int32)
        self.calls = []
        self.released = False
        self.__class__.instances.append(self)
        self.__class__.events.append(("new", len(self.__class__.instances) - 1))

    def call(self, n, data, ia, ja, rhs, phase, transpose=False):
        self.calls.append({"phase": phase, "mtype": self.mtype, "n": n})
        U = sp.csr_matrix((data, ja - 1, ia - 1), shape=(n, n)).toarray()
        K = U + np.triu(U, 1).T
        return np.linalg.solve(K, rhs)

    def release(self):
        self.released = True
        self.__class__.events.append(("release", self.__class__.instances.index(self)))


class FreeSpdDispatchTest(unittest.TestCase):
    def setUp(self):
        _RecordingRaw.events.clear()
        _RecordingRaw.instances.clear()
        patcher = mock.patch.object(pardiso_variant, "_RawPardiso", _RecordingRaw)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_upper_triangle_of_free_block_and_prescribed_values(self):
        n = 40
        pinned = np.r_[0:8, 30:40]
        A = _whole_mesh_system(n, pinned)
        b = np.arange(1.0, n + 1.0)
        solver = pardiso_variant.FreeBlockSPDSolver()
        x = solver(_Csr(A), b)
        np.testing.assert_allclose(x, np.linalg.solve(A.toarray(), b), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(x[pinned], b[pinned])
        raw = _RecordingRaw.instances[0]
        self.assertEqual(raw.mtype, 2)
        self.assertEqual(raw.calls[0]["n"], n - pinned.size)
        stats = solver.stats_snapshot()
        self.assertEqual(stats["n_free_last"], n - pinned.size)
        self.assertEqual(stats["pinned_zero_rhs"], 0)

    def test_phases_within_a_pattern(self):
        n = 30
        pinned = np.r_[0:6]
        A = _whole_mesh_system(n, pinned)
        b = np.ones(n)
        b[pinned] = 0.0
        solver = pardiso_variant.FreeBlockSPDSolver()
        solver(_Csr(A), b)
        solver(_Csr(A), 2 * b)  # same values -> backsolve only
        A2 = A.copy()
        A2.data = A2.data.copy()
        A2.data[A2.data > 1.5] *= 1.1  # new values, same pattern and pinned set
        solver(_Csr(A2), b)
        raw = _RecordingRaw.instances[0]
        self.assertEqual([c["phase"] for c in raw.calls], [13, 33, 23])
        stats = solver.stats_snapshot()
        self.assertEqual(stats["analyze_calls"], 1)
        self.assertEqual(stats["backsolve_hits"], 1)
        self.assertEqual(stats["pattern_rebuilds"], 0)
        self.assertEqual(stats["pinned_zero_rhs"], 3)
        self.assertEqual(stats["numeric_factorizations"], 2)

    def test_new_pattern_releases_old_factor_first(self):
        n = 50
        b = np.linspace(1.0, 2.0, n)
        solver = pardiso_variant.FreeBlockSPDSolver()
        for k, top in enumerate((40, 30, 20)):  # activation front moves: fewer pinned rows
            pinned = np.r_[top:n]
            A = _whole_mesh_system(n, pinned, seed=k)
            x = solver(_Csr(A), b)
            np.testing.assert_allclose(x, np.linalg.solve(A.toarray(), b), rtol=1e-12, atol=1e-12)
        self.assertEqual(
            _RecordingRaw.events,
            [("new", 0), ("release", 0), ("new", 1), ("release", 1), ("new", 2)],
        )
        self.assertEqual(solver.stats_snapshot()["pattern_rebuilds"], 2)
        self.assertEqual(solver.release_states(), 1)
        self.assertTrue(_RecordingRaw.instances[2].released)
        self.assertEqual(solver.release_states(), 0)

    def test_pardiso_error_falls_back_to_full_system(self):
        n = 20
        pinned = np.r_[0:4]
        A = _whole_mesh_system(n, pinned)
        b = np.ones(n)
        solver = pardiso_variant.FreeBlockSPDSolver()
        expected = np.linalg.solve(A.toarray(), b)

        def failing_call(self, *args, **kwargs):
            raise RuntimeError("MKL PARDISO failed with error -4 (phase 13)")

        with mock.patch.object(_RecordingRaw, "call", failing_call), \
                mock.patch.object(pardiso_variant, "pardiso_solve_once",
                                  return_value=expected) as once:
            x = solver(_Csr(A), b)
        once.assert_called_once()
        np.testing.assert_array_equal(x, expected)
        self.assertEqual(solver.stats_snapshot()["fallbacks"], 1)
        self.assertTrue(_RecordingRaw.instances[0].released)
        self.assertIsNone(solver._pattern)


class FreeSpdRegistryTest(unittest.TestCase):
    def test_spec_and_adapter(self):
        spec = linear.normalize_linear_solver_spec({"backend": "pardiso", "mode": "free-spd"})
        self.assertEqual(spec["mode"], "free-spd")
        self.assertEqual(linear.parse_linear_solver_spec("pardiso:free-spd")["mode"], "free-spd")
        adapter = pardiso_variant.PardisoCustomSolver("free-spd")
        self.assertIsInstance(adapter._maybe_v07_variant(), pardiso_variant.FreeBlockSPDSolver)
        self.assertEqual(adapter.label, "pardiso_free_spd(mtype=2)")


@unittest.skipIf(PARDISO_IMPORT_ERROR is not None, f"pypardiso unavailable: {PARDISO_IMPORT_ERROR}")
class FreeSpdMklTest(unittest.TestCase):
    def test_matches_full_system_phase23(self):
        n = 600
        rng = np.random.default_rng(3)
        free_spd = pardiso_variant.PardisoCustomSolver("free-spd")
        full = pardiso_variant.PardisoCustomSolver("phase23")
        for k, top in enumerate((450, 450, 300, 150)):
            pinned = np.r_[0:10, top:n]
            A = _whole_mesh_system(n, pinned, seed=10 + k)
            b = rng.standard_normal(n)
            b[0:10] = 0.0
            b[top:n] = 0.0 if k % 2 else rng.standard_normal(n - top)
            x = free_spd(_Csr(A), b, None, {})
            ref = full(_Csr(A), b, None, {})
            np.testing.assert_allclose(x, ref, rtol=1e-10, atol=1e-12)
            self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-12)
        stats = free_spd.stats_snapshot()
        self.assertEqual(stats["solves"], 4)
        self.assertEqual(stats["pattern_rebuilds"], 2)
        self.assertEqual(stats["analyze_calls"], 3)
        self.assertEqual(stats["fallbacks"], 0)
        self.assertGreater(stats["nnz_factor_max"], 0)
        self.assertEqual(free_spd.release(), 1)
        full.release()


if __name__ == "__main__":
    unittest.main()


import pytest as _pytest_lane
pytestmark = _pytest_lane.mark.solver
