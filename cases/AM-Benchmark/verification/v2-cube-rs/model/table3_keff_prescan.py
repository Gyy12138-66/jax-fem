#!/usr/bin/env python3
"""Build and collect short no-keff precursors for the V2 Table-3 parameter grid.

This module is orchestration only: it generates four-track serpentine paths, emits
reproducible shell commands for the existing thermal runner, and collects each
completed no-keff run through the existing melt-width / Balbaa Eq.19--24 code.
It never launches parity runs and never uses experimental temperatures for fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

SCHEMA = "v2.table3-keff-prescan/1"
SUMMARY_SCHEMA = "v2.table3-keff-summary/1"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA:
        raise SystemExit(f"unsupported prescan schema: {config.get('schema_version')!r}")
    axes = config.get("axes", {})
    expected = {"power_W", "speed_mm_s", "hatch_mm"}
    if set(axes) != expected or any(not axes[name] for name in expected):
        raise SystemExit(f"axes must contain non-empty {sorted(expected)}")
    prescan = config.get("prescan", {})
    if int(prescan.get("tracks", 0)) < int(prescan.get("measurement_track", 0)) + 1:
        raise SystemExit("tracks must include the measurement track and one following track")
    if int(prescan["thermal_output_every"]) != 1:
        raise SystemExit("thermal_output_every must be 1 so the requested track-end frame exists")
    return config


def case_id(power_W: float, speed_mm_s: float, hatch_mm: float) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p")
    return f"P{token(power_W)}_V{token(speed_mm_s)}_H{token(hatch_mm)}"


def expand_cases(config: dict) -> list[dict]:
    axes = config["axes"]
    cases = []
    for power, speed, hatch in itertools.product(
            axes["power_W"], axes["speed_mm_s"], axes["hatch_mm"]):
        values = (float(power), float(speed), float(hatch))
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise SystemExit(f"invalid non-positive/non-finite condition: {values}")
        cases.append({
            "case_id": case_id(*values),
            "power_W": values[0],
            "speed_mm_s": values[1],
            "speed_m_s": values[1] * 1.0e-3,
            "hatch_mm": values[2],
            "hatch_m": values[2] * 1.0e-3,
        })
    if len({item["case_id"] for item in cases}) != len(cases):
        raise SystemExit("case IDs are not unique")
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(config_path: Path, value: str, repo: Path) -> Path:
    candidates = [(config_path.parent / value).resolve(), (repo / value).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"configured input does not exist: {value}")


def command_line(parts) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def material_config_for_run(source: Path, destination: Path, repo: Path) -> Path:
    """Write a checkout-local config; committed configs contain box-159 absolute paths."""
    config = json.loads(source.read_text(encoding="utf-8"))
    for key, value in list(config.items()):
        if not (isinstance(value, str) and key.endswith(("_table", "_solid", "_powder"))):
            continue
        marker = "/cases/AM-Benchmark/"
        if marker in value:
            config[key] = str(repo / ("cases/AM-Benchmark/" + value.split(marker, 1)[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def write_plan(config_path: Path, config: dict, repo: Path, output_root: Path) -> dict:
    path_script = resolve_input(config_path, config["path_generator"], repo)
    picker = resolve_input(config_path, config["measurement_picker"], repo)
    keff_script = resolve_input(config_path, config["keff_deriver"], repo)
    material_source = resolve_input(config_path, config["material_config"], repo)
    material = material_config_for_run(
        material_source, output_root.resolve() / "material_config_no_keff.json", repo)
    mesh = resolve_input(config_path, config["mesh"], repo)
    settings = config["prescan"]
    entries = []
    for case in expand_cases(config):
        directory = output_root / case["case_id"]
        path_csv = directory / "path.csv"
        path_cmd = [sys.executable, path_script, "--power", case["power_W"],
                    "--speed", case["speed_m_s"], "--hatch", case["hatch_m"],
                    "--tracks", settings["tracks"], "--sample-step",
                    settings["sample_step_m"], "--beam-radius", settings["beam_radius_m"],
                    "--output", path_csv, "--ledger-json", directory / "path_ledger.json"]
        run_cmd = [sys.executable, "-m", "jax_fem_am.simulation.runner",
                   "--config", material, "--inp", mesh, "--output-dir", directory,
                   "--profile-json", directory / "profile.json", "--profile-label",
                   f"table3-keff-prescan-{case['case_id']}", "--xla-platform", "cpu",
                   "--xla-preallocate", "off", "--xla-linear-solver", "pardiso",
                   "--xla-pardiso-mode", "phase23", "--build-axis", "z", "--base-side",
                   "min", "--layer-thickness", settings["layer_thickness_m"], "--layers", 1,
                   "--support-thickness", 4.0e-4, "--path-file", path_csv,
                   "--path-length-scale", 1.0, "--source-model", "legacy", "--beam-radius",
                   settings["beam_radius_m"], "--source-depth", settings["source_depth_m"],
                   "--source-depth-cutoff", settings["source_depth_cutoff_m"],
                   "--no-source-cutoff-renormalize", "--laser-power", case["power_W"],
                   "--dt", settings["sample_step_m"] / case["speed_m_s"],
                   "--layer-activation-mode", "layer_on_scan", "--layer-activation-geometry",
                   "intersection", "--future-layer-mode", "void", "--active-window-below-layers",
                   0, "--inactive-mass-factor", 1.0, "--powder-mode", "powder",
                   "--surface-selection", "exterior", "--boundary-tol", 1.0e-6,
                   "--quadrature-order", 2, "--ambient", settings["ambient_K"],
                   "--preheat-temperature", settings["preheat_K"], "--bottom-thermal-bc",
                   "fixed", "--bottom-temperature", settings["bottom_temperature_K"],
                   "--cooling-steps", settings["cooling_steps"], "--cooling-dt", 0.01,
                   "--mechanics-every", 0,
                   "--thermal-mass-lumping", "--thermal-output-every",
                   settings["thermal_output_every"], "--summary-every", 200,
                   "--phase-history-model", "paper_irreversible", "--fixture-thermal-phase",
                   settings["fixture_thermal_phase"]]
        entries.append({**case, "run_dir": str(directory), "path_csv": str(path_csv),
                        "path_command": command_line(path_cmd),
                        "run_command": command_line(run_cmd),
                        "measurement_command": command_line([
                            sys.executable, picker, path_csv, "--nth",
                            settings["measurement_track"], "--domain", settings["domain_m"],
                            "--hatch", case["hatch_m"]]),
                        "keff_command_template": command_line([
                            sys.executable, keff_script, "--power", case["power_W"], "--speed",
                            case["speed_m_s"], "--from-run", directory,
                            "--tag", case["case_id"], "--output",
                            directory / "k_liquid_keff.csv", "--json",
                            directory / "keff_derivation.json"]),
                        "status": "planned"})
    plan = {
        "schema_version": SCHEMA,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256(config_path),
        "material_config_source": str(material_source),
        "material_config_source_sha256": sha256(material_source),
        "material_config": str(material), "material_config_sha256": sha256(material),
        "mesh": str(mesh), "mesh_sha256": sha256(mesh),
        "path_generator": str(path_script), "path_generator_sha256": sha256(path_script),
        "measurement_picker": str(picker), "measurement_picker_sha256": sha256(picker),
        "keff_deriver": str(keff_script), "keff_deriver_sha256": sha256(keff_script),
        "case_count": len(entries), "cases": entries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "prescan_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
    return plan


def save_case_manifest(entry: dict, plan: dict) -> None:
    directory = Path(entry["run_dir"])
    manifest = {
        "schema_version": SCHEMA,
        "case": {key: entry[key] for key in (
            "case_id", "power_W", "speed_mm_s", "speed_m_s", "hatch_mm", "hatch_m")},
        "source_config": plan["source_config"],
        "source_config_sha256": plan["source_config_sha256"],
        "material_config_source": plan["material_config_source"],
        "material_config_source_sha256": plan["material_config_source_sha256"],
        "material_config": plan["material_config"],
        "material_config_sha256": plan["material_config_sha256"],
        "mesh": plan["mesh"], "mesh_sha256": plan["mesh_sha256"],
        "code_inputs": {key: plan[key] for key in (
            "path_generator", "path_generator_sha256", "measurement_picker",
            "measurement_picker_sha256", "keff_deriver", "keff_deriver_sha256")},
        "commands": {key: entry[key] for key in (
            "path_command", "run_command", "measurement_command", "keff_command_template")},
        "status": entry["status"],
    }
    (directory / "prescan_case.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_plan(plan: dict, repo: Path, selected: set[str] | None) -> None:
    for entry in plan["cases"]:
        if selected and entry["case_id"] not in selected:
            continue
        directory = Path(entry["run_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        save_case_manifest(entry, plan)
        for key, log_name in (("path_command", "path.log"), ("run_command", "run.log")):
            command = shlex.split(entry[key])
            with (directory / log_name).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=repo, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise SystemExit(f"{entry['case_id']} failed ({key}, rc={result.returncode})")
        ledger_path = directory / "thermal_energy_ledger_summary.json"
        if not ledger_path.is_file():
            raise SystemExit(f"{entry['case_id']} runner exited 0 but has no ledger summary")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("complete") is not True or ledger.get("solver_completed") is not True:
            raise SystemExit(f"{entry['case_id']} runner ledger is not complete")
        entry["status"] = "no_keff_complete"
        save_case_manifest(entry, plan)


def verify_case_inputs(entry: dict, plan: dict, state: dict) -> None:
    expected_fingerprints = {
        "source_config_sha256": plan["source_config_sha256"],
        "material_config_source_sha256": plan["material_config_source_sha256"],
        "material_config_sha256": plan["material_config_sha256"],
        "mesh_sha256": plan["mesh_sha256"],
    }
    actual_fingerprints = {key: state.get(key) for key in expected_fingerprints}
    actual_fingerprints.update(state.get("code_inputs", {}))
    expected_fingerprints.update({key: plan[key] for key in (
        "path_generator_sha256", "measurement_picker_sha256", "keff_deriver_sha256")})
    mismatches = [key for key, value in expected_fingerprints.items()
                  if actual_fingerprints.get(key) != value]
    if mismatches:
        raise SystemExit(f"{entry['case_id']} input fingerprint mismatch: {', '.join(mismatches)}")
    expected_case = {key: entry[key] for key in (
        "case_id", "power_W", "speed_mm_s", "speed_m_s", "hatch_mm", "hatch_m")}
    if state.get("case") != expected_case:
        raise SystemExit(f"{entry['case_id']} case parameters do not match its manifest")


def plan_fingerprints(plan: dict) -> dict:
    return {key: plan[key] for key in (
        "source_config_sha256", "material_config_source_sha256", "material_config_sha256",
        "mesh_sha256", "path_generator_sha256", "measurement_picker_sha256",
        "keff_deriver_sha256")}


def merge_summary(existing: dict | None, records: list[dict], plan: dict) -> dict:
    if existing and existing.get("schema_version") != SUMMARY_SCHEMA:
        raise SystemExit("existing summary has an incompatible schema")
    fingerprints = plan_fingerprints(plan)
    if existing and existing.get("input_fingerprints") != fingerprints:
        raise SystemExit("existing summary belongs to a different plan/input fingerprint")
    merged = {item["case_id"]: item for item in (existing or {}).get("cases", [])}
    merged.update({item["case_id"]: item for item in records})
    ordered = [merged[item["case_id"]] for item in plan["cases"] if item["case_id"] in merged]
    return {"schema_version": SUMMARY_SCHEMA, "input_fingerprints": fingerprints,
            "case_count": len(ordered), "expected_case_count": plan["case_count"],
            "complete": len(ordered) == plan["case_count"], "cases": ordered}


def collect(plan: dict, repo: Path, selected: set[str] | None) -> dict:
    records = []
    for entry in plan["cases"]:
        if selected and entry["case_id"] not in selected:
            continue
        directory = Path(entry["run_dir"])
        case_manifest = directory / "prescan_case.json"
        if not case_manifest.is_file():
            raise SystemExit(f"{entry['case_id']} has no prescan_case.json; run no-keff first")
        state = json.loads(case_manifest.read_text(encoding="utf-8"))
        if state.get("status") not in {"no_keff_complete", "keff_derived"}:
            raise SystemExit(
                f"{entry['case_id']} no-keff precursor is not complete: {state.get('status')!r}")
        verify_case_inputs(entry, plan, state)
        picker = subprocess.run(shlex.split(entry["measurement_command"]), cwd=repo,
                                text=True, capture_output=True, check=False)
        if picker.returncode:
            raise SystemExit(f"{entry['case_id']} measurement selection failed: {picker.stderr}")
        values = picker.stdout.split()
        if len(values) != 6:
            raise SystemExit(f"{entry['case_id']} invalid measurement output: {picker.stdout!r}")
        track_y, at_time, *window = values
        command = shlex.split(entry["keff_command_template"]) + [
            "--from-run-window", ",".join(window), "--from-run-time", at_time,
            "--from-run-track-y", track_y]
        result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
        (directory / "keff.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"{entry['case_id']} keff derivation failed; see {directory/'keff.log'}")
        record = json.loads((directory / "keff_derivation.json").read_text(encoding="utf-8"))
        entry["status"] = "keff_derived"
        save_case_manifest(entry, plan)
        records.append({
            "case_id": entry["case_id"], "power_W": entry["power_W"],
            "speed_mm_s": entry["speed_mm_s"], "hatch_mm": entry["hatch_mm"],
            "L_um": record["characteristic_length"]["L_m"] * 1.0e6,
            "keff_W_mK": record["derivation"]["keff_W_mK"],
            "Marangoni": record["derivation"]["Marangoni"],
            "Nusselt": record["derivation"]["Nusselt"],
            "frame": record["characteristic_length"]["measurement"]["frame"],
            "frame_time_s": record["characteristic_length"]["measurement"]["frame_time_s"],
            "run_dir": str(directory), "keff_json_sha256": sha256(directory / "keff_derivation.json"),
        })
    root = Path(plan["cases"][0]["run_dir"]).parent
    summary_path = root / "keff_table3_summary.json"
    existing = (json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file() else None)
    summary = merge_summary(existing, records, plan)
    ordered = summary["cases"]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "keff_table3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]) if ordered else ["case_id"])
        writer.writeheader(); writer.writerows(ordered)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "run", "collect", "all"), default="plan")
    parser.add_argument("--case", action="append", dest="cases",
                        help="limit to one or more case IDs (repeatable)")
    args = parser.parse_args()
    repo, config_path = args.repo.resolve(), args.config.resolve()
    config = load_config(config_path)
    plan = write_plan(config_path, config, repo, args.output_root.resolve())
    selected = set(args.cases) if args.cases else None
    known = {item["case_id"] for item in plan["cases"]}
    if selected and not selected <= known:
        raise SystemExit(f"unknown case IDs: {sorted(selected-known)}")
    if args.mode in {"run", "all"}:
        run_plan(plan, repo, selected)
    if args.mode in {"collect", "all"}:
        collect(plan, repo, selected)
    print(f"{args.mode}: {len(selected) if selected else len(plan['cases'])} case(s); "
          f"plan={args.output_root.resolve()/'prescan_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
