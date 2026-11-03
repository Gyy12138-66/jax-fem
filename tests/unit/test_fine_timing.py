"""Fine-timing instrumentation (JAX_FEM_FINE_TIMING=1) bookkeeping.

The scoped copies ("<stage>@thermal") and the cell_jacobian detail stages are
views of stages that are already counted, so they must never change the
python_overhead accounting.
"""
import time

import numpy as np
import pytest

import jax_fem_am.simulation.acceleration as acc


@pytest.fixture
def fine(monkeypatch):
    monkeypatch.setattr(acc, "FINE_TIMING", True)
    monkeypatch.setitem(acc._FINE_STATE, "scope", None)
    return acc


def test_scoped_copy_is_added_only_inside_a_scope(fine):
    report = fine.ProfilingReport("t")
    report.add(fine.STAGE_SOLVER, 1.0)
    fine._FINE_STATE["scope"] = "mechanics"
    report.add(fine.STAGE_SOLVER, 2.0)
    with report.stage(fine.STAGE_CELL_JACOBIAN):
        pass
    assert report.stage_seconds[fine.STAGE_SOLVER] == pytest.approx(3.0)
    assert report.stage_seconds["solver@mechanics"] == pytest.approx(2.0)
    assert report.stage_calls["cell_jacobian@mechanics"] == 1
    assert "solver@thermal" not in report.stage_seconds


def test_scoped_and_detail_stages_do_not_change_python_overhead(fine):
    report = fine.ProfilingReport("t")
    report.wall_start = time.perf_counter() - 10.0
    fine._FINE_STATE["scope"] = "thermal"
    report.add(fine.STAGE_LOCAL_ASSEMBLY, 4.0)
    report.add(fine.STAGE_CELL_JACOBIAN, 3.0)
    report.add(fine.STAGE_CELL_JACOBIAN_DEVICE, 2.0)
    report.add(fine.STAGE_CELL_JACOBIAN_D2H, 0.5)
    report.add(fine.STAGE_NEWTON_TANGENT_COPY, 0.3)
    report.finish()
    # only local_assembly (4 s) is accounted; the rest are sub-views
    assert report.stage_seconds[fine.STAGE_PYTHON] == pytest.approx(6.0, abs=0.05)


def test_timed_host_stack_matches_numpy_and_records_both_parts(fine):
    report = fine.ProfilingReport("t")
    shim = fine._TimedHostStack(report)
    parts = [np.arange(6.0).reshape(2, 3), np.arange(6.0, 9.0).reshape(1, 3)]
    out = shim.vstack(parts)
    np.testing.assert_array_equal(out, np.vstack(parts))
    assert report.stage_calls[fine.STAGE_CELL_JACOBIAN_DEVICE] == 1
    assert report.stage_calls[fine.STAGE_CELL_JACOBIAN_D2H] == 1
    assert shim.zeros((2,)).shape == (2,)


def test_stage_without_switch_adds_no_scoped_copy(monkeypatch):
    monkeypatch.setattr(acc, "FINE_TIMING", False)
    monkeypatch.setitem(acc._FINE_STATE, "scope", "mechanics")
    report = acc.ProfilingReport("t")
    with report.stage(acc.STAGE_CELL_JACOBIAN):
        pass
    report.add(acc.STAGE_SOLVER, 1.0)
    assert not any("@" in k for k in report.stage_seconds)
