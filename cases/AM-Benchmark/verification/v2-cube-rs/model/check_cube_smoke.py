#!/usr/bin/env python3
"""Stage-2 gate checks for the V2 cube smoke (STRESS-REPRODUCTION-PLAN.md).

Reads a finished runner output directory plus the stage-1 preflight ledger and
writes ``cube_smoke_gate.json``. Nothing here has a tunable threshold toward a
target: the checks are identities (step counts, activation steps, energy
closure) and health facts (NaN/Inf, Newton non-convergence, release effect).

Usage:
    check_cube_smoke.py --run <dir> --preflight <dir> [--capture-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np


def cell_field(mesh, name):
    data = mesh.cell_data_dict.get(name)
    return None if data is None else np.asarray(list(data.values())[0]).reshape(-1)


def quad_average(mesh, pattern):
    parts = [cell_field(mesh, pattern.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def hex_centroids(mesh):
    pts = np.asarray(mesh.points)
    for block in mesh.cells:
        conn = np.asarray(block.data)
        if conn.shape[1] == 8:
            return pts[conn].mean(axis=1)
    raise SystemExit("no HEX8 block in VTU")


def read_ledger(run: Path) -> tuple[list[dict], dict]:
    rows = []
    with (run / "thermal_energy_ledger.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    summary = json.loads((run / "thermal_energy_ledger_summary.json").read_text(encoding="utf-8"))
    return rows, summary


def scan_log(run: Path) -> dict:
    text = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    summaries = re.findall(
        r"global_step=(\d+) mode=(\w+) layer=(\d+)/(\d+) highest_printed_layer=(\d+).*?"
        r"printed_cells=(\d+)/(\d+).*?mechanics_current=(\d) mechanics_source_step=(-?\d+) "
        r"T_min=([-\d.eE+]+) T_max=([-\d.eE+]+) u_max=([-\d.eE+]+) vm_max=([-\d.eE+]+)", text)
    nonconv = len(re.findall(r"did not converge", text))
    cutbacks = len(re.findall(r"cutback|sub-?step", text, flags=re.IGNORECASE))
    nan_hits = len(re.findall(r"\bnan\b|\binf\b", text, flags=re.IGNORECASE))
    release = re.search(r"release_vtk=(\S+) release_u_max=([-\d.eE+]+)", text)
    cut = re.search(r"release cut box: deactivating (\d+) cells", text)
    return {
        "summary_lines": len(summaries),
        "newton_nonconvergence_count": nonconv,
        "cutback_mentions": cutbacks,
        "nan_or_inf_mentions": nan_hits,
        "last_summary": summaries[-1] if summaries else None,
        "mechanics_steps_seen": sorted({int(s[8]) for s in summaries if int(s[7]) == 1}),
        "highest_printed_layer_first_step": {
            int(s[4]): int(s[0]) for s in reversed(summaries) if int(s[4]) > 0
        },
        "release_u_max": float(release.group(2)) if release else None,
        "release_cut_cells": int(cut.group(1)) if cut else None,
    }


def energy_closure(rows: list[dict]) -> dict:
    commanded = sum(float(r.get("laser_commanded_j", 0.0)) for r in rows)
    nominal = sum(float(r.get("laser_absorbed_nominal_j", 0.0)) for r in rows)
    deposited = sum(float(r.get("laser_deposited_j", 0.0)) for r in rows)
    on = [r for r in rows if float(r.get("laser_commanded_j", 0.0)) > 0.0]
    per_step = [float(r["laser_deposited_j"]) / float(r["laser_absorbed_nominal_j"])
                for r in on if float(r["laser_absorbed_nominal_j"]) > 0.0]
    return {
        "laser_commanded_J": commanded,
        "laser_absorbed_nominal_J": nominal,
        "laser_deposited_J": deposited,
        "capture_fraction": deposited / nominal if nominal > 0 else None,
        "per_step_capture_min": min(per_step) if per_step else None,
        "per_step_capture_max": max(per_step) if per_step else None,
        "per_step_capture_std": float(np.std(per_step)) if per_step else None,
        "laser_on_steps": len(on),
        "max_relative_balance_error": max(float(r.get("relative_balance_error", 0.0) or 0.0) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--capture-only", action="store_true",
                    help="thermal-only capture trial: skip mechanics/release/activation checks")
    ap.add_argument("--max-slabs", type=int, default=None,
                    help="shakedown run truncated with --max-print-layers N: check activation for slabs <= N only "
                         "and take the expected step count from the run's own ledger summary")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run = args.run
    ledger_pre = json.loads((args.preflight / "v2_cube_smoke_ledger.json").read_text(encoding="utf-8"))
    rows, summary = read_ledger(run)
    log = scan_log(run)
    energy = energy_closure(rows)
    flash = ledger_pre.get("flash")
    if flash:
        # reading A: the commanded power is P/capture, so the ledger's own capture
        # fraction should read ~capture_analytic; the PHYSICAL closure is
        # deposited / (Ac x P x t_scan) per layer, which is what matters.
        absorptivity = float(ledger_pre.get("absorptivity", 0.62))
        on_steps = energy["laser_on_steps"]
        layers_in_run = on_steps / float(flash["substeps"]) if flash["substeps"] else None
        physical_absorbed = absorptivity * flash["physical_energy_per_layer_J"] * (layers_in_run or 0.0)
        energy["flash"] = {
            "capture_fraction_analytic": flash["capture_fraction_analytic"],
            "capture_fraction_ledger_over_analytic": (energy["capture_fraction"] / flash["capture_fraction_analytic"])
            if energy["capture_fraction"] else None,
            "layers_in_run": layers_in_run,
            "physical_absorbed_J": physical_absorbed,
            "deposited_over_physical_absorbed": (energy["laser_deposited_J"] / physical_absorbed) if physical_absorbed else None,
        }
    profile = {}
    if (run / "profile.json").is_file():
        prof = json.loads((run / "profile.json").read_text(encoding="utf-8"))
        profile = {"wall_seconds": prof.get("wall_seconds"), "steps": prof.get("steps"),
                   "newton_wall_seconds": (prof.get("meta") or {}).get("newton_wall_seconds")}
        if profile["wall_seconds"] and profile["steps"]:
            profile["seconds_per_step"] = profile["wall_seconds"] / profile["steps"]
    gate = {
        "schema": "v2.cube-smoke-gate/1",
        "run_dir": str(run.resolve()),
        "ledger_complete": summary.get("complete") is True,
        "solver_completed": summary.get("solver_completed"),
        "recorded_step_count": summary.get("recorded_step_count"),
        "expected_step_count": summary.get("expected_step_count"),
        "max_relative_balance_error": summary.get("maximum_relative_balance_error"),
        "all_balance_steps_within_tolerance": summary.get("all_balance_steps_within_tolerance"),
        "energy": energy,
        "log": log,
        "profile": profile,
        "checks": {},
    }
    checks = gate["checks"]
    checks["ledger_complete"] = gate["ledger_complete"]
    checks["no_nan_or_inf_in_log"] = log["nan_or_inf_mentions"] == 0
    checks["newton_converged_everywhere"] = log["newton_nonconvergence_count"] == 0
    if not args.capture_only:
        if args.max_slabs:
            checks["step_count_matches_run_ledger"] = (
                summary.get("recorded_step_count") == summary.get("expected_step_count"))
        else:
            checks["step_count_matches_preflight"] = (
                summary.get("recorded_step_count") == ledger_pre["expected_runner_steps"])
        # activation: cell field activation_step vs preflight events (exact)
        vtus = sorted(glob.glob(str(run / "step_*.vtu")))
        activation = {"checked": False}
        if vtus:
            import meshio
            last = meshio.read(vtus[-1])
            act = cell_field(last, "activation_step")
            layer = cell_field(last, "layer_id")
            state = cell_field(last, "material_state")
            tref = cell_field(last, "stress_free_temperature")
            z = hex_centroids(last)[:, 2]
            sub_z = ledger_pre["substrate_z_m"]
            slab_dz = ledger_pre["layers"][0]["deposition_z_m"] - sub_z
            events = []
            all_match = True
            slab_limit = args.max_slabs or len(ledger_pre["activation_events"])
            for event in ledger_pre["activation_events"]:
                k = event["slab"]
                if k > slab_limit:
                    continue
                in_slab = (z > event["z_bottom_m"]) & (z < event["z_top_m"])
                steps = np.unique(act[in_slab])
                ok = steps.size == 1 and int(steps[0]) == int(event["global_step"])
                all_match &= bool(ok)
                events.append({"slab": k, "expected_global_step": event["global_step"],
                               "observed_activation_steps": [int(s) for s in steps],
                               "cells": int(in_slab.sum()), "ok": bool(ok),
                               "material_state_values": [float(v) for v in np.unique(state[in_slab])],
                               "stress_free_temperature_values": [float(v) for v in np.unique(tref[in_slab])]})
            part = (z > sub_z) & (z < sub_z + slab_limit * slab_dz + 1e-9)   # printed slabs only
            activation = {
                "checked": True, "vtu": os.path.basename(vtus[-1]),
                "events": events, "all_events_match": bool(all_match),
                "slab_limit": int(slab_limit),
                "part_cells": int(part.sum()),
                "part_all_solid_at_end": bool(np.all(state[part] == 2.0)),
                "part_stress_free_temperature_unique": [float(v) for v in np.unique(tref[part])],
                "slab_thickness_m": slab_dz,
            }
        gate["activation"] = activation
        checks["activation_steps_match_preflight"] = activation.get("all_events_match", False)
        checks["part_consolidated_on_activation"] = activation.get("part_all_solid_at_end", False)
        contract = json.loads((args.preflight / "runner_contract.json").read_text(encoding="utf-8"))
        t_ref = float(contract["temperature_mapping"]["stress_free_reference_K"])
        observed = activation.get("part_stress_free_temperature_unique") or []
        # VTU cell data is stored in float32: 1273.15 reads back as 1273.1500244,
        # so the tolerance is a float32 ulp at ~1e3 K (1.2e-4), not 1e-6.
        checks["stress_free_reference_applied"] = (
            len(observed) == 1 and math.isclose(observed[0], t_ref, rel_tol=0.0, abs_tol=1e-3))
        # release effect: last cooling frame vs release.vtu (part cells only)
        release = {"checked": False}
        rel_path = run / "release.vtu"
        if rel_path.is_file() and vtus:
            import meshio
            before = meshio.read(vtus[-1])
            after = meshio.read(rel_path)
            zb = hex_centroids(before)[:, 2]
            sub_z = ledger_pre["substrate_z_m"]
            slab_dz_r = ledger_pre["layers"][0]["deposition_z_m"] - sub_z
            slab_limit_r = args.max_slabs or len(ledger_pre["activation_events"])
            part = (zb > sub_z) & (zb < sub_z + slab_limit_r * slab_dz_r + 1e-9)
            removed = cell_field(after, "release_removed")
            vm_b = quad_average(before, "vm_quadQ")
            vm_a = quad_average(after, "vm_quadQ")
            eqp_b = cell_field(before, "eq_plastic_strain")
            u_b = np.asarray(before.point_data["u"])
            u_a = np.asarray(after.point_data["u"])
            release = {
                "checked": True,
                "release_removed_cells": int(np.sum(removed > 0.5)) if removed is not None else None,
                "substrate_cells": int(np.sum(zb <= sub_z)),
                "removed_equals_substrate": bool(removed is not None and int(np.sum(removed > 0.5)) == int(np.sum(zb <= sub_z))),
                "part_vm_MPa_before": {"mean": float(vm_b[part].mean() / 1e6), "max": float(vm_b[part].max() / 1e6)},
                "part_vm_MPa_after": {"mean": float(vm_a[part].mean() / 1e6), "max": float(vm_a[part].max() / 1e6)},
                "part_vm_changed_by_release": bool(not np.allclose(vm_b[part], vm_a[part])),
                "part_eqp_max_before": float(eqp_b[part].max()),
                "u_max_before_m": float(np.abs(u_b).max()),
                "u_max_after_m": float(np.abs(u_a).max()),
                "removed_cells_vm_MPa_max_after": float(vm_a[removed > 0.5].max() / 1e6) if removed is not None and np.any(removed > 0.5) else None,
            }
        gate["release"] = release
        checks["release_solve_present"] = release.get("checked", False)
        checks["release_removed_substrate"] = release.get("removed_equals_substrate", False)
        checks["release_changed_part_stress"] = release.get("part_vm_changed_by_release", False)
    gate["all_checks_passed"] = all(bool(v) for v in checks.values())
    out = args.output or (run / "cube_smoke_gate.json")
    out.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "energy": {k: energy[k] for k in ("capture_fraction", "per_step_capture_min", "per_step_capture_max", "max_relative_balance_error")},
                      "profile": profile, "all_checks_passed": gate["all_checks_passed"]}, indent=2))
    raise SystemExit(0 if gate["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
