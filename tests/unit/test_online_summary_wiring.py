"""Contract tests for selecting solver-step observations in the final scorer."""

from pathlib import Path


ANALYZER = (Path(__file__).resolve().parents[2]
            / "cases/AM-Benchmark/verification/v2-cube-rs/model"
            / "analyze_pyrometer.py")


def test_analyzer_promotes_online_response_series_to_production_series():
    text = ANALYZER.read_text(encoding="utf-8")
    assert 'ap.add_argument("--online-summary"' in text
    assert 'online_summary.get("response_integrated_series")' in text
    assert "series = online_series" in text
    assert '"online_solver_steps"' in text
