"""Regression tests for Balbaa's 10 ms instrument-response aggregation."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2]
          / "cases/AM-Benchmark/verification/v2-cube-rs/model"
          / "summarize_online_observables.py")
SPEC = importlib.util.spec_from_file_location("summarize_online_observables", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _row(time_s, dt_s, value):
    return {
        "time_s": time_s,
        "dt_s": dt_s,
        "beam_inside_spot": True,
        "n_hot": 1,
        "avg_K": value,
        "two_colour_S1": value,
        "two_colour_S2": value,
        "full_spot_avg_K": value,
        "max_K": value,
        "probe_K": [value, value + 1.0],
    }


def test_response_bins_weight_unequal_steps_by_duration(monkeypatch):
    monkeypatch.setattr(summary.tc, "read_accumulated", lambda s1, s2, wavelengths: {
        "T_K": s1, "over_range": False})
    rows = [_row(0.002, 0.002, 100.0), _row(0.010, 0.008, 200.0)]
    result = summary.response_integrated_series(rows, 0.010, (1.0, 2.0))
    assert len(result) == 1
    assert result[0]["avg_K"] == pytest.approx(180.0)
    assert result[0]["two_colour_K"] == pytest.approx(180.0)
    assert result[0]["probe_mean_K"] == pytest.approx([180.0, 181.0])
    assert result[0]["coverage_fraction"] == pytest.approx(1.0)


def test_response_step_is_split_at_bin_boundary(monkeypatch):
    monkeypatch.setattr(summary.tc, "read_accumulated", lambda s1, s2, wavelengths: {
        "T_K": s1, "over_range": False})
    rows = [_row(0.006, 0.006, 100.0), _row(0.014, 0.008, 300.0)]
    result = summary.response_integrated_series(rows, 0.010, (1.0, 2.0))
    assert [entry["bin_index"] for entry in result] == [0, 1]
    # First bin has 6 ms at 100 K and 4 ms at 300 K.
    assert result[0]["avg_K"] == pytest.approx(180.0)
    # The remaining 4 ms of the second step belongs to the next bin.
    assert result[1]["avg_K"] == pytest.approx(300.0)
    assert result[1]["coverage_fraction"] == pytest.approx(0.4)


def test_response_rejects_nonpositive_dt():
    with pytest.raises(ValueError, match="invalid time_s/dt_s"):
        summary.response_integrated_series([_row(0.01, 0.0, 100.0)], 0.01,
                                            (1.0, 2.0))


@pytest.mark.parametrize(("second", "kind"), [
    (_row(0.012, 0.001, 200.0), "gap"),
    (_row(0.012, 0.003, 200.0), "overlap"),
])
def test_response_rejects_gaps_and_overlaps(second, kind):
    with pytest.raises(ValueError, match=kind):
        summary.response_integrated_series([_row(0.010, 0.010, 100.0), second],
                                            0.010, (1.0, 2.0))
