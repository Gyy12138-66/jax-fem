"""Fail-closed contract for the Balbaa V1 production runner."""

from pathlib import Path


RUNNER = (Path(__file__).resolve().parents[2]
          / "cases/AM-Benchmark/verification/v1-single-track/model"
          / "run_v1_cbmB.sh")


def test_v1_runner_preserves_extra_argument_boundaries_and_validates_numbers():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'read -r -a EXTRA_ARGV' in text
    assert 'SOLVER_CMD+=("${EXTRA_ARGV[@]}")' in text
    assert 'SOLVER_CMD+=(${EXTRA_ARGS})' not in text
    assert 'math.isfinite(value)' in text
    assert '--source-depth-cutoff 0' in text
    assert '--no-source-cutoff-renormalize' in text
    assert '--source-depth-cutoff 2.0e-5' not in text
    assert '--fixture-thermal-phase follow-temperature' in text


def test_v1_runner_only_marks_done_after_successful_audit():
    text = RUNNER.read_text(encoding="utf-8")
    audit = text.index('-m jax_fem_am.verification.run_audit')
    done = text.index('> "${OUT_ROOT}/DONE"')
    assert audit < done
    assert '|| echo "v1 WARNING: run_audit failed"' not in text
    assert 'set -euo pipefail' in text
    assert '--thermal-only' in text
