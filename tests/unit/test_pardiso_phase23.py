import unittest
from unittest import mock

import numpy as np

try:
    import pypardiso  # noqa: F401
except ImportError as exc:  # pragma: no cover
    PARDISO_IMPORT_ERROR = exc
else:
    PARDISO_IMPORT_ERROR = None

from jax_fem_am.simulation.acceleration import _PardisoCustomSolver
from jax_fem_am.solvers import pardiso as pardiso_variant


class _CsrMatrix:
    def __init__(self, data):
        self.indptr = np.array([0, 2, 4], dtype=np.int32)
        self.indices = np.array([0, 1, 0, 1], dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float64)

    def getValuesCSR(self):
        return self.indptr, self.indices, self.data


class _RecordingRawPardiso:
    """No-MKL phase recorder for VariantSolver dispatch tests."""

    instances = []

    def __init__(self, single_precision=False):
        self.single_precision = single_precision
        self.calls = []
        self.__class__.instances.append(self)

    def call(self, n, data, ia, ja, rhs, phase, transpose=False):
        self.calls.append(
            {
                "n": n,
                "data": np.asarray(data).copy(),
                "ia": np.asarray(ia).copy(),
                "ja": np.asarray(ja).copy(),
                "rhs": np.asarray(rhs).copy(),
                "phase": phase,
                "transpose": transpose,
            }
        )
        return np.asarray(rhs, dtype=np.float64).copy()

    def release(self):
        pass


class PardisoPhaseDispatchTest(unittest.TestCase):
    def setUp(self):
        _RecordingRawPardiso.instances.clear()

    def make_solver(self):
        patcher = mock.patch.object(
            pardiso_variant,
            "_RawPardiso",
            _RecordingRawPardiso,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        return pardiso_variant.VariantSolver("phase23")

    def test_same_matrix_different_rhs_uses_phase33_backsolve(self):
        solver = self.make_solver()
        matrix = _CsrMatrix([4.0, 1.0, 1.0, 3.0])

        solver(matrix, np.array([1.0, 2.0]), None, {})
        solver(matrix, np.array([3.0, 4.0]), None, {})
        raw = _RecordingRawPardiso.instances[0]

        self.assertEqual([call["phase"] for call in raw.calls], [13, 33])
        np.testing.assert_array_equal(raw.calls[0]["rhs"], [1.0, 2.0])
        np.testing.assert_array_equal(raw.calls[1]["rhs"], [3.0, 4.0])
        self.assertEqual(solver._stats["analyze_calls"], 1)
        self.assertEqual(solver._stats["backsolve_hits"], 1)
        self.assertEqual(solver._stats["solves"], 2)
        stats = solver.stats_snapshot()
        self.assertEqual(stats["phase13_calls"], 1)
        self.assertEqual(stats["phase23_calls"], 0)
        self.assertEqual(stats["phase33_calls"], 1)
        self.assertEqual(stats["numeric_factorizations"], 1)
        self.assertIn("factorization_phase_seconds", stats)
        self.assertNotIn("numeric_factorization_seconds", stats)

    def test_changed_matrix_values_use_phase23_numeric_refactor(self):
        solver = self.make_solver()

        solver(
            _CsrMatrix([4.0, 1.0, 1.0, 3.0]),
            np.array([1.0, 2.0]),
            None,
            {},
        )
        solver(
            _CsrMatrix([5.0, 1.0, 1.0, 2.0]),
            np.array([1.0, 2.0]),
            None,
            {},
        )
        raw = _RecordingRawPardiso.instances[0]

        self.assertEqual([call["phase"] for call in raw.calls], [13, 23])
        np.testing.assert_array_equal(
            raw.calls[1]["data"],
            [5.0, 1.0, 1.0, 2.0],
        )
        self.assertEqual(solver._stats["analyze_calls"], 1)
        self.assertEqual(solver._stats["backsolve_hits"], 0)
        self.assertEqual(solver._stats["solves"], 2)
        stats = solver.stats_snapshot()
        self.assertEqual(stats["phase13_calls"], 1)
        self.assertEqual(stats["phase23_calls"], 1)
        self.assertEqual(stats["phase33_calls"], 0)
        self.assertEqual(stats["numeric_factorizations"], 2)


@unittest.skipIf(
    PARDISO_IMPORT_ERROR is not None,
    f"pypardiso unavailable: {PARDISO_IMPORT_ERROR}",
)
class PardisoPhase23Test(unittest.TestCase):
    def test_reuses_symbolic_analysis_when_matrix_values_change(self):
        solver = _PardisoCustomSolver("phase23")
        rhs = np.array([1.0, 2.0])

        first_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        first = solver(
            _CsrMatrix([4.0, 1.0, 1.0, 3.0]), rhs, None, {}
        )
        np.testing.assert_allclose(first, np.linalg.solve(first_dense, rhs))

        second_dense = np.array([[5.0, 1.0], [1.0, 2.0]])
        second = solver(
            _CsrMatrix([5.0, 1.0, 1.0, 2.0]), rhs, None, {}
        )
        np.testing.assert_allclose(second, np.linalg.solve(second_dense, rhs))

        self.assertEqual(solver._v07_variant._stats["analyze_calls"], 1)
        self.assertEqual(solver._v07_variant._stats["solves"], 2)


if __name__ == "__main__":
    unittest.main()
