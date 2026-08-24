"""Static contract tests for the Balbaa V2 production thermal-gate runner."""

from pathlib import Path


RUNNER = (Path(__file__).resolve().parents[2]
          / "cases/AM-Benchmark/verification/v2-cube-rs/model/runs"
          / "v_thermal_gate.sh")


def test_production_runner_wires_preregistered_observables_and_source_band():
    text = RUNNER.read_text(encoding="utf-8")
    required = (
        'REPO="${REPO:-/home/user/work/159/jax-fem}"',
        'OUTROOT="${OUTROOT:-/home/user/work/159/output}"',
        'VT="${VT:-/home/user/work/159/vtmp}"',
        'cd "$REPO"',
        "--source-depth-cutoff 0",
        "--no-source-cutoff-renormalize",
        "--online-observables",
        "--online-observables-window",
        "--online-observables-probes",
        "online_observables_summary.json",
        "--online-summary",
        "--observation-window",
        "--fixture-thermal-phase follow-temperature",
        'online_observables_summary.json',
        "run_manifest.json",
        "--online-observables-run-id",
        "--expected-run-id",
        'manifest.get("status") == "complete"',
        'meta.get("run_id") == run_id',
        'summary.get("meta", {}).get("run_id") == run_id',
        "build_run_manifest.py",
        'manifest.get("run_id") == run_id',
        'summary.get("source_jsonl_sha256") == sha256(rows)',
        'summary.get("n_rows") == sum(',
        'summary.get("coverage_s", [None, None])[1] >= 0.90',
        'KEFF_DIR=$VT/derived',
        'set +e',
        'manifest["status"] = "failed"',
        'manifest["exit_code"] = rc',
        'PARITY_CFG=$KEFF_DIR/v2_material_config_thermal_keff.json',
        'config["k_table_liquid"] = keff',
        'run_arm parity "$PARITY_CFG"',
        '--output "$OUT/run_audit.json" --thermal-only',
        'audit = json.load(open(os.path.join(out, "run_audit.json")',
        'audit.get("thermal_only") is True',
        'audit.get("transient", {}).get("all_steps_valid") is True',
        'OUT_EVERY=10; COOL_STEPS=90; COOL_DT=dynamic-to-window-end',
        'path_end = float(rows[-1]["time"])',
        'target = 0.90 - 1.0e-12',
        'remaining = target - path_end',
        'print(f"{remaining / steps:.17g}")',
        '--cooling-steps "$COOL_STEPS" --cooling-dt "$ARM_COOL_DT"',
        'is_complete "$NAME" "$CFG" "$DIR"',
        '当前产物未通过 manifest/观测/audit 完整性检查',
    )
    for token in required:
        assert token in text
    assert '"source_depth_cutoff_m": 0.0' in text
    assert '"source_cutoff_renormalize": False' in text
    assert "--source-depth-cutoff 4.0e-5" not in text


def test_production_runner_is_fail_closed():
    text = RUNNER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'run_audit 失败，停止' in text
    assert '缺少在线观测产物' in text
    assert 'WARNING: run_audit 失败' not in text
    assert '$M/tables/k_liquid_keff.csv' not in text
    assert '$M/derived/keff_' not in text
    stage4 = text.index('# ---- 阶段 4:对比 ----')
    compare = text.index('python "$M/compare_thermal_gate.py"', stage4)
    completeness = text.index('is_complete "$NAME" "$CFG" "$DIR"', stage4)
    assert stage4 < completeness < compare
