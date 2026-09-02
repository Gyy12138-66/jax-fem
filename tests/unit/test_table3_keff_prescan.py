import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "cases/AM-Benchmark/verification/v2-cube-rs/model/table3_keff_prescan.py"
CONFIG = ROOT / "cases/AM-Benchmark/verification/v2-cube-rs/inputs/table3-keff-prescan.json"
SPEC = importlib.util.spec_from_file_location("table3_keff_prescan", MODEL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_table3_full_factorial_is_explicit_and_unique():
    config = MODULE.load_config(CONFIG)
    cases = MODULE.expand_cases(config)
    assert len(cases) == 27
    assert len({case["case_id"] for case in cases}) == 27
    assert cases[0] == {
        "case_id": "P140_V500_H0p08", "power_W": 140.0,
        "speed_mm_s": 500.0, "speed_m_s": 0.5,
        "hatch_mm": 0.08, "hatch_m": 8e-05,
    }
    assert cases[-1]["case_id"] == "P270_V800_H0p12"


def test_prescan_requires_following_track_and_every_step_output(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["prescan"]["tracks"] = raw["prescan"]["measurement_track"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="one following track"):
        MODULE.load_config(bad)

    raw["prescan"]["tracks"] = 4
    raw["prescan"]["thermal_output_every"] = 2
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="must be 1"):
        MODULE.load_config(bad)


def test_plan_contains_short_no_keff_commands_and_fingerprints(tmp_path):
    config = MODULE.load_config(CONFIG)
    plan = MODULE.write_plan(CONFIG, config, ROOT, tmp_path)
    assert plan["case_count"] == 27
    assert len(plan["source_config_sha256"]) == 64
    assert len(plan["keff_deriver_sha256"]) == 64
    generated_material = pathlib.Path(plan["material_config"])
    generated = json.loads(generated_material.read_text(encoding="utf-8"))
    assert str(ROOT) in generated["k_table_solid"]
    assert "/home/user/work/159/jax-fem" not in generated["k_table_solid"]
    first = plan["cases"][0]
    assert "--tracks 4" in first["path_command"]
    assert "--nth 3" in first["measurement_command"]
    assert "--thermal-output-every 1" in first["run_command"]
    assert "--cooling-steps 0" in first["run_command"]
    assert "--laser-power 140.0" in first["run_command"]
    assert "--dt 0.0001" in first["run_command"]
    assert "--from-run" in first["keff_command_template"]
    assert "--allow-incomplete-run" not in first["keff_command_template"]
    assert "k_table_liquid" not in first["run_command"]
    saved = json.loads((tmp_path / "prescan_plan.json").read_text(encoding="utf-8"))
    assert saved["cases"][0]["case_id"] == first["case_id"]


def test_material_config_relocation_only_changes_repo_paths(tmp_path):
    source = ROOT / "cases/AM-Benchmark/verification/v2-cube-rs/model/v2_material_config_thermal_asis.json"
    destination = tmp_path / "material.json"
    result = MODULE.material_config_for_run(source, destination, ROOT)
    original = json.loads(source.read_text(encoding="utf-8"))
    relocated = json.loads(result.read_text(encoding="utf-8"))
    assert relocated["conductivity_liquid"] == original["conductivity_liquid"]
    assert pathlib.Path(relocated["k_table_powder"]).is_absolute()
    assert str(ROOT) in relocated["k_table_powder"]


def test_collect_fingerprints_fail_closed_and_partial_summary_merges(tmp_path):
    config = MODULE.load_config(CONFIG)
    plan = MODULE.write_plan(CONFIG, config, ROOT, tmp_path)
    entry = plan["cases"][0]
    pathlib.Path(entry["run_dir"]).mkdir()
    entry["status"] = "no_keff_complete"
    MODULE.save_case_manifest(entry, plan)
    state = json.loads((pathlib.Path(entry["run_dir"]) / "prescan_case.json").read_text())
    state["code_inputs"]["keff_deriver_sha256"] = "stale"
    with pytest.raises(SystemExit, match="keff_deriver_sha256"):
        MODULE.verify_case_inputs(entry, plan, state)

    first = {"case_id": plan["cases"][0]["case_id"], "value": 1}
    second = {"case_id": plan["cases"][1]["case_id"], "value": 2}
    summary = MODULE.merge_summary(None, [first], plan)
    assert summary["input_fingerprints"] == MODULE.plan_fingerprints(plan)
    merged = MODULE.merge_summary(summary, [second], plan)
    assert [item["case_id"] for item in merged["cases"]] == [
        first["case_id"], second["case_id"]]
    assert merged["case_count"] == 2
    assert merged["expected_case_count"] == 27
    assert merged["complete"] is False
    stale = dict(summary)
    stale["input_fingerprints"] = {**summary["input_fingerprints"], "mesh_sha256": "stale"}
    with pytest.raises(SystemExit, match="different plan/input fingerprint"):
        MODULE.merge_summary(stale, [second], plan)


def test_collect_requires_successful_prescan_marker(tmp_path):
    config = MODULE.load_config(CONFIG)
    plan = MODULE.write_plan(CONFIG, config, ROOT, tmp_path)
    with pytest.raises(SystemExit, match="run no-keff first"):
        MODULE.collect(plan, ROOT, {plan["cases"][0]["case_id"]})


def test_run_requires_complete_solver_ledger(tmp_path, monkeypatch):
    config = MODULE.load_config(CONFIG)
    plan = MODULE.write_plan(CONFIG, config, ROOT, tmp_path)
    entry = plan["cases"][0]

    def fake_run(command, **_kwargs):
        directory = pathlib.Path(entry["run_dir"])
        if "jax_fem_am.simulation.runner" in " ".join(map(str, command)):
            (directory / "thermal_energy_ledger_summary.json").write_text(
                json.dumps({"complete": False, "solver_completed": False}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="ledger is not complete"):
        MODULE.run_plan(plan, ROOT, {entry["case_id"]})


def test_unknown_case_fails_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        str(MODEL), "--config", str(CONFIG), "--repo", str(ROOT),
        "--output-root", str(tmp_path), "--mode", "plan", "--case", "not-a-case",
    ])
    with pytest.raises(SystemExit, match="unknown case IDs"):
        MODULE.main()
