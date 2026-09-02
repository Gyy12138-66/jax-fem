import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = (Path(__file__).parents[2] / "cases" / "AM-Benchmark" / "verification"
        / "v2-cube-rs" / "model" / "plot_heating_cooling_trends.py")
SPEC = importlib.util.spec_from_file_location("plot_heating_cooling_trends", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_centred_mean_requires_odd_positive_width():
    with pytest.raises(ValueError):
        MODULE.centred_contiguous_mean([1.0], 0)
    with pytest.raises(ValueError):
        MODULE.centred_contiguous_mean([1.0, 2.0], 2)


def test_centred_mean_does_not_cross_missing_values():
    result = MODULE.centred_contiguous_mean(
        [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0, 11.0], 5)
    assert result[2] == pytest.approx(3.0)
    assert np.isnan(result[4:8]).all()
    assert result[8] == pytest.approx(9.0)


def test_add_trend_preserves_raw_and_breaks_at_small_denominator():
    series = [{"avg_K": 1273.15 + i, "full_spot_avg_K": 1273.15 + i,
               "mean_n_hot": 10.0, "coverage_fraction": 1.0}
              for i in range(7)]
    series[3]["mean_n_hot"] = 1.0
    result = MODULE.add_trend(series, trend_bins=5, bin_ms=10.0,
                              min_mean_n_hot=5.0)
    assert [item["adopted_C"] for item in result] == pytest.approx(
        [1000.0 + i for i in range(7)])
    assert result[3]["adopted_C"] == pytest.approx(1003.0)
    assert result[3]["adopted_trend_C"] is None
    assert result[3]["roi_trend_C"] == pytest.approx(1003.0)
    assert result[3]["adopted_reliable"] is False
    assert result[2]["adopted_reliable"] is True


def test_roi_trend_breaks_at_partial_coverage():
    series = [{"avg_K": 1273.15 + i, "full_spot_avg_K": 1273.15 + i,
               "mean_n_hot": 10.0, "coverage_fraction": 1.0}
              for i in range(7)]
    series[3]["coverage_fraction"] = 0.5
    result = MODULE.add_trend(series, trend_bins=5, bin_ms=10.0,
                              min_mean_n_hot=5.0)
    assert result[3]["adopted_C"] == pytest.approx(1003.0)
    assert result[3]["roi_trend_C"] is None


def test_fixed_bins_rejects_window_not_aligned_to_bin_boundaries(tmp_path):
    row = {
        "time_s": 0.41, "dt_s": 0.01, "beam_inside_spot": False,
        "avg_K": None, "n_hot": 0, "two_colour_S1": 0.0,
        "two_colour_S2": 0.0, "full_spot_avg_K": 300.0, "max_K": 300.0,
    }
    path = tmp_path / "observations.jsonl"
    path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        MODULE.fixed_bins(path, (0.401, 0.409), 10.0, (0.95e-6, 1.05e-6))
