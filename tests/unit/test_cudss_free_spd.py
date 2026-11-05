"""cudss-plus 2026-11-05 (M1b / E0): the cuDSS free-block solver is in the tree as PARDISO mode free-spd-cudss.

CPU part: the mode parses and the adapter builds CudssFreeBlockSPDSolver with the CPU free-spd class as fallback
(no CUDA library is loaded before the first solve). GPU part (marker gpu, skipped without libcudss): the same
whole-mesh systems as test_pardiso_free_spd (pinned unit rows with stored zero off-diagonals) are solved to
round-off against a dense reference, through the analysis / refactorization / back-substitution-only phases,
and an activation-pattern change rebuilds the pattern.
"""
import os

import numpy as np
import pytest

from jax_fem_am.solvers import linear
from jax_fem_am.solvers import pardiso as pardiso_variant
from tests.unit.test_pardiso_free_spd import _Csr, _whole_mesh_system


def test_mode_parses_and_builds_the_cudss_variant():
    assert linear.parse_linear_solver_spec("pardiso:free-spd-cudss")["mode"] == "free-spd-cudss"
    spec = linear.normalize_linear_solver_spec({"backend": "pardiso", "mode": "free-spd-cudss"})
    assert spec["mode"] == "free-spd-cudss"
    adapter = pardiso_variant.PardisoCustomSolver("free-spd-cudss")
    variant = adapter._maybe_v07_variant()
    from jax_fem_am.solvers.cudss import CudssFreeBlockSPDSolver

    assert isinstance(variant, CudssFreeBlockSPDSolver)
    assert variant.cpu_class is pardiso_variant.FreeBlockSPDSolver
    assert variant.pinned_rows is pardiso_variant._pinned_rows
    assert adapter.label.startswith("cudss_free_spd(")


def _libcudss_available():
    try:
        from jax_fem_am.solvers import cudss

        cudss.find_libcudss()
        return True
    except Exception:
        return False


def _dense_solve(A, b):
    x = np.linalg.solve(A.toarray(), b)
    return x


@pytest.mark.gpu
@pytest.mark.skipif(not _libcudss_available(), reason="libcudss.so.1 not found")
def test_cudss_free_spd_solves_whole_mesh_systems():
    from jax_fem_am.solvers.cudss import CudssFreeBlockSPDSolver

    n = 400
    pinned = np.r_[0:7, 150:163, 390:400]
    solver = CudssFreeBlockSPDSolver()
    rng = np.random.default_rng(3)

    def check(A, b):
        x = solver(_Csr(A), b)
        ref = _dense_solve(A, b)
        assert np.max(np.abs(x - ref)) <= 1e-10 * np.max(np.abs(ref))
        assert np.array_equal(x[pinned], b[pinned])

    b = rng.standard_normal(n)
    b[pinned] = 0.0
    A1 = _whole_mesh_system(n, pinned, seed=0)
    check(A1, b)                                   # phase 13: analysis + factor + solve
    check(A1, b)                                   # same values: back-substitution only
    check(_whole_mesh_system(n, pinned, seed=1), b)    # new values, same pattern: phase 23
    pinned2 = np.r_[0:7, 150:170, 390:400]
    b2 = rng.standard_normal(n)
    b2[pinned2] = 0.0
    pinned = pinned2
    check(_whole_mesh_system(n, pinned2, seed=2), b2)  # activation change: new pattern
    st = solver.stats_snapshot()
    assert st["phase13_calls"] == 2 and st["phase23_calls"] == 1 and st["phase33_calls"] == 1
    assert st["pattern_rebuilds"] == 1 and st["gpu_pattern_failures"] == 0 and st["fallbacks"] == 0
    assert solver.release_states() == 1
