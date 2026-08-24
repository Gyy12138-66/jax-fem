"""Static contract tests for the Balbaa V2 production thermal-gate runner."""

from pathlib import Path


RUNNER = (Path(__file__).resolve().parents[2]
          / "cases/AM-Benchmark/verification/v2-cube-rs/model/runs"
          / "v_thermal_gate.sh")


def test_production_runner_wires_preregistered_observables_and_source_band():
    text = RUNNER.read_text(encoding="utf-8")
    required = (
        "--source-depth-cutoff 4.0e-5",
        "--source-cutoff-renormalize",
        "--online-observables",
        "--online-observables-window",
        "--online-observables-probes",
        "online_observables_summary.json",
        "--online-summary",
        "--observation-window",
    )
    for token in required:
        assert token in text


def test_production_runner_is_fail_closed():
    text = RUNNER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'run_audit 失败，停止' in text
    assert '缺少在线观测产物' in text
    assert 'WARNING: run_audit 失败' not in text
