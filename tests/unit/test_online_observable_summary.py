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


def _meta(run_id="run-123"):
    return {
        "schema_version": "v06.online-observables/1",
        "run_id": run_id,
        "spot_center_m": [0.002, 0.002],
        "spot_diameter_m": 0.002,
        "threshold_C": 1000.0,
        "range_max_C": 3000.0,
        "window_s": [0.45, 0.90],
        "record_every_n_steps": 1,
        "depth_scope": "top layer (z >= 4.000000e-04 m)",
        "two_colour": {"wavelengths_m": [0.95e-6, 1.05e-6]},
        "probes": [{"requested_m": probe} for probe in
                   [[0.001, 0.002, 0.00042], [0.002, 0.002, 0.00042],
                    [0.003, 0.002, 0.00042]]],
        "probe_resolution": {"all_probes_contained": True},
    }


def test_protocol_validation_accepts_registered_metadata():
    assert summary.validate_protocol(_meta(), expected_run_id="run-123") == pytest.approx(
        (0.95e-6, 1.05e-6))


def test_protocol_validation_accepts_recording_window_that_contains_summary_window():
    meta = _meta()
    meta["window_s"] = [0.40, 1.00]
    assert summary.validate_protocol(
        meta, expected_run_id="run-123", summary_window_s=(0.45, 0.90)
    ) == pytest.approx((0.95e-6, 1.05e-6))


@pytest.mark.parametrize("mutation", [
    lambda meta: meta.update(run_id="wrong"),
    lambda meta: meta.update(spot_center_m=[0.001, 0.002]),
    lambda meta: meta.update(record_every_n_steps=2),
    lambda meta: meta.update(window_s=[0.46, 0.90]),
    lambda meta: meta.update(window_s=[0.45, 0.89]),
    lambda meta: meta.update(depth_scope="top layer (z >= 3.900000e-04 m)"),
    lambda meta: meta["two_colour"].update(wavelengths_m=[1.0e-6, 1.1e-6]),
    lambda meta: meta["probe_resolution"].update(all_probes_contained=False),
])
def test_protocol_validation_rejects_mismatch(mutation):
    meta = _meta()
    mutation(meta)
    with pytest.raises(ValueError, match="protocol mismatch"):
        summary.validate_protocol(meta, expected_run_id="run-123")


def test_crop_rows_to_window_clips_boundary_steps():
    rows = [_row(0.44, 0.02, 100.0), _row(0.46, 0.02, 200.0),
            _row(0.91, 0.45, 300.0)]
    cropped = summary.crop_rows_to_window(rows, (0.45, 0.90))
    assert [r["time_s"] for r in cropped] == pytest.approx([0.46, 0.90])
    assert [r["dt_s"] for r in cropped] == pytest.approx([0.01, 0.44])


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


def test_response_bins_disclose_na_samples_without_changing_the_reading(monkeypatch):
    """A2 (a): NA samples stay out of the adopted denominator but are disclosed."""
    monkeypatch.setattr(summary.tc, "read_accumulated", lambda s1, s2, wavelengths: {
        "T_K": s1, "over_range": False})
    hot = _row(0.004, 0.004, 200.0)
    cold = _row(0.010, 0.006, 0.0)
    cold["n_hot"] = 0
    cold["avg_K"] = None
    result = summary.response_integrated_series([hot, cold], 0.010, (1.0, 2.0))
    assert len(result) == 1
    entry = result[0]
    assert entry["avg_K"] == pytest.approx(200.0)      # valid-sample denominator
    assert entry["n_na_samples"] == 1
    assert entry["na_sample_fraction"] == pytest.approx(0.5)
    assert entry["na_covered_s"] == pytest.approx(0.006)
    assert entry["na_time_fraction"] == pytest.approx(0.6)
    assert entry["na_over_half"] is True


def test_response_bin_with_only_na_samples_is_na_and_flagged(monkeypatch):
    monkeypatch.setattr(summary.tc, "read_accumulated", lambda s1, s2, wavelengths: {
        "T_K": s1, "over_range": False})
    cold = _row(0.010, 0.010, 0.0)
    cold["n_hot"] = 0
    cold["avg_K"] = None
    entry = summary.response_integrated_series([cold], 0.010, (1.0, 2.0))[0]
    assert entry["avg_K"] is None
    assert entry["n_na_samples"] == 1
    assert entry["na_time_fraction"] == pytest.approx(1.0)
    assert entry["na_over_half"] is True
