#!/usr/bin/env python3
"""Generate the V2 cube stage-1 mesh/path preflight without running a solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{name} must be a positive integer")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise SystemExit(f"{name} must be finite and positive")
    return result


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "v2.cube-stress-smoke/1":
        raise SystemExit("unsupported cube smoke schema")
    g, layers, scan = cfg["geometry"], cfg["layer_schedule"], cfg["scan"]
    physical = _positive_finite(layers["physical_layer_thickness_m"],
                                "layer_schedule.physical_layer_thickness_m")
    slab = _positive_finite(layers["activation_slab_thickness_m"],
                            "layer_schedule.activation_slab_thickness_m")
    per_slab = _positive_integer(layers["physical_layers_per_slab"],
                                 "layer_schedule.physical_layers_per_slab")
    if not math.isclose(slab, physical * per_slab, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("activation slab must equal physical layer thickness * layers per slab")
    n_physical = _positive_integer(layers["smoke_physical_layers"],
                                   "layer_schedule.smoke_physical_layers")
    n_production = _positive_integer(layers["production_physical_layers"],
                                     "layer_schedule.production_physical_layers")
    if n_physical % per_slab:
        raise SystemExit("smoke physical-layer count must be a positive multiple of layers per slab")
    if n_production % per_slab:
        raise SystemExit("production physical-layer count must be a multiple of layers per slab")
    if not math.isclose(float(g["smoke_part_z_m"]), n_physical * physical,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("smoke part height does not match physical-layer schedule")
    if not math.isclose(float(g["production_part_z_m"]), n_production * physical,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("production part height does not match physical-layer schedule")
    for key in ("part_xy_m", "smoke_part_z_m", "production_part_z_m",
                "substrate_xy_m", "substrate_z_m", "cell_size_m"):
        _positive_finite(g[key], f"geometry.{key}")
    for key in ("power_W", "speed_m_s", "hatch_m", "sample_step_m", "jump_speed_m_s"):
        _positive_finite(scan[key], f"scan.{key}")
    for key in ("start_angle_deg", "rotation_per_physical_layer_deg"):
        if isinstance(scan[key], bool) or not math.isfinite(float(scan[key])):
            raise SystemExit(f"scan.{key} must be finite")
    _positive_finite(layers["recoat_time_s"], "layer_schedule.recoat_time_s")
    if not isinstance(layers["recoat_after_final_layer"], bool):
        raise SystemExit("layer_schedule.recoat_after_final_layer must be boolean")
    if not isinstance(scan["serpentine"], bool) or scan["serpentine"] is not True:
        raise SystemExit("scan.serpentine must be true for this preflight")
    margin = scan["margin_m"]
    if isinstance(margin, bool) or not math.isfinite(float(margin)) or float(margin) < 0:
        raise SystemExit("scan.margin_m must be finite and non-negative")
    if 2 * float(margin) >= float(g["part_xy_m"]):
        raise SystemExit("scan margin leaves no exposure area")
    return cfg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_row(rows: list[dict], state: dict, *, dt: float, x: float, y: float,
            z: float, power: float, laser_on: int, layer: int, hatch: int,
            mode: str, physical_layer: int, scan_id: int) -> None:
    if dt <= 0:
        raise ValueError("path dt must be positive")
    state["time"] += dt
    rows.append({
        "time": state["time"], "x": x, "y": y, "z": z,
        "power": power, "laser_on": laser_on, "layer": layer,
        "hatch": hatch, "mode": mode, "front_coord": z,
        "physical_layer": physical_layer, "scan_id": scan_id,
    })


def generate_schedule(cfg: dict) -> tuple[list[dict], dict]:
    g, layers, scan = cfg["geometry"], cfg["layer_schedule"], cfg["scan"]
    side = float(g["part_xy_m"])
    sub_z = float(g["substrate_z_m"])
    physical_dz = float(layers["physical_layer_thickness_m"])
    per_slab = int(layers["physical_layers_per_slab"])
    n_physical = int(layers["smoke_physical_layers"])
    recoat = float(layers["recoat_time_s"])
    margin = float(scan["margin_m"])
    hatch_space = float(scan["hatch_m"])
    sample_step = float(scan["sample_step_m"])
    speed = float(scan["speed_m_s"])
    jump_speed = float(scan["jump_speed_m_s"])
    power = float(scan["power_W"])
    start_angle = float(scan["start_angle_deg"])
    rotation = float(scan["rotation_per_physical_layer_deg"])
    rows: list[dict] = []
    state = {"time": 0.0}
    previous = None
    scan_id = 0
    energy_J = 0.0
    scan_time_s = 0.0
    jump_time_s = 0.0
    recoat_time_s = 0.0
    layer_summaries = []

    exposure_width = side - 2 * margin
    n_tracks = int(math.ceil(exposure_width / hatch_space))
    lo, hi = margin, side - margin
    for physical_layer in range(1, n_physical + 1):
        slab = (physical_layer - 1) // per_slab + 1
        z = sub_z + physical_layer * physical_dz
        angle = (start_angle + (physical_layer - 1) * rotation) % 180.0
        axis = "x" if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else "y"
        if axis == "y" and not math.isclose(angle, 90.0, abs_tol=1e-9):
            raise SystemExit("stage-1 generator currently supports only 0/90 degree layers")
        layer_start = state["time"]
        for track in range(n_tracks):
            cross = (side - (n_tracks - 1) * hatch_space) / 2 + track * hatch_space
            forward = track % 2 == 0
            a, b = (lo, hi) if forward else (hi, lo)
            start = (a, cross) if axis == "x" else (cross, a)
            end = (b, cross) if axis == "x" else (cross, b)
            scan_id += 1
            if previous is not None:
                distance = math.dist(previous, (start[0], start[1], z))
                dt = max(distance / jump_speed, 1e-8)
                jump_time_s += dt
                add_row(rows, state, dt=dt, x=start[0], y=start[1], z=z,
                        power=0.0, laser_on=0, layer=slab, hatch=track + 1,
                        mode="jump", physical_layer=physical_layer, scan_id=scan_id)
            length = math.dist(start, end)
            segments = max(int(math.ceil(length / sample_step)), 1)
            dt = (length / segments) / speed
            for segment in range(1, segments + 1):
                fraction = segment / segments
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                add_row(rows, state, dt=dt, x=x, y=y, z=z, power=power,
                        laser_on=1, layer=slab, hatch=track + 1, mode="scan",
                        physical_layer=physical_layer, scan_id=scan_id)
                energy_J += power * dt
                scan_time_s += dt
            previous = (end[0], end[1], z)
        layer_summaries.append({
            "physical_layer": physical_layer, "activation_slab": slab,
            "z_m": z, "angle_deg": angle, "axis": axis,
            "tracks": n_tracks, "scan_end_time_s": state["time"],
            "scan_duration_s": state["time"] - layer_start,
        })
        if physical_layer < n_physical or bool(layers["recoat_after_final_layer"]):
            add_row(rows, state, dt=recoat, x=previous[0], y=previous[1], z=z,
                    power=0.0, laser_on=0, layer=slab, hatch=0,
                    mode="recoat", physical_layer=physical_layer, scan_id=scan_id)
            recoat_time_s += recoat

    expected_slabs = n_physical // per_slab
    ledger = {
        "schema": "v2.cube-preflight-ledger/1",
        "complete": True,
        "completion_scope": "mesh/path preflight artifacts only",
        "solver_started": False,
        "solver_compatible": False,
        "solver_compatibility_note": "stage-1 audit path includes explicit recoat rows and physical_layer metadata; a dedicated runner contract is required before any solve",
        "physical_layers": n_physical,
        "activation_slabs": expected_slabs,
        "physical_layers_per_slab": per_slab,
        "tracks_per_physical_layer": n_tracks,
        "track_cross_min_m": (side - (n_tracks - 1) * hatch_space) / 2,
        "track_cross_max_m": (side + (n_tracks - 1) * hatch_space) / 2,
        "exposure_bounds_m": [margin, side - margin],
        "scan_rows": sum(row["laser_on"] == 1 for row in rows),
        "jump_rows": sum(row["mode"] == "jump" for row in rows),
        "recoat_rows": sum(row["mode"] == "recoat" for row in rows),
        "scan_time_s": scan_time_s,
        "jump_time_s": jump_time_s,
        "recoat_time_s": recoat_time_s,
        "total_time_s": state["time"],
        "nominal_laser_energy_J": energy_J,
        "activation_rule": layers["activation_rule"],
        "angles_deg": [item["angle_deg"] for item in layer_summaries],
        "layers": layer_summaries,
    }
    validate_schedule(rows, ledger, cfg)
    return rows, ledger


def validate_schedule(rows: list[dict], ledger: dict, cfg: dict) -> None:
    layers = cfg["layer_schedule"]
    n_physical = int(layers["smoke_physical_layers"])
    per_slab = int(layers["physical_layers_per_slab"])
    expected_recoat = n_physical if layers["recoat_after_final_layer"] else n_physical - 1
    if ledger["recoat_rows"] != expected_recoat:
        raise ValueError("recoat event count mismatch")
    if not math.isclose(ledger["recoat_time_s"], expected_recoat * float(layers["recoat_time_s"]),
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("recoat duration mismatch")
    previous = -math.inf
    by_physical: dict[int, set[int]] = {}
    for row in rows:
        if row["time"] <= previous:
            raise ValueError("path times are not strictly increasing")
        previous = row["time"]
        if row["laser_on"]:
            by_physical.setdefault(row["physical_layer"], set()).add(row["layer"])
    if set(by_physical) != set(range(1, n_physical + 1)):
        raise ValueError("missing physical layer scan")
    for physical, slab_ids in by_physical.items():
        if slab_ids != {(physical - 1) // per_slab + 1}:
            raise ValueError("physical-layer to activation-slab mapping mismatch")
    expected_angles = [float(cfg["scan"]["start_angle_deg"] + i *
                             cfg["scan"]["rotation_per_physical_layer_deg"]) % 180.0
                       for i in range(n_physical)]
    if ledger["angles_deg"] != expected_angles:
        raise ValueError("layer rotation mismatch")


def write_schedule(path: Path, rows: list[dict]) -> None:
    fields = ["time", "x", "y", "z", "power", "laser_on", "layer", "hatch",
              "mode", "front_coord", "physical_layer", "scan_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in row.items()})


def build_preflight(config_path: Path, repo: Path, output: Path) -> dict:
    cfg = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / "v2_cube_smoke_c3d8.inp"
    path = output / "v2_cube_smoke_path.csv"
    ledger_path = output / "v2_cube_smoke_ledger.json"
    mesh_script = repo / "cases/AM-Benchmark/verification/v2-cube-rs/model/make_v2_mesh_cube.py"
    g = cfg["geometry"]
    command = [sys.executable, str(mesh_script), "--res", str(g["cell_size_m"]),
               "--part-xy", str(g["part_xy_m"]), "--part-z", str(g["smoke_part_z_m"]),
               "--sub-xy", str(g["substrate_xy_m"]), "--sub-z", str(g["substrate_z_m"]),
               "--output", str(mesh)]
    subprocess.run(command, check=True)
    rows, ledger = generate_schedule(cfg)
    write_schedule(path, rows)
    ledger.update({
        "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "mesh": str(mesh.resolve()), "mesh_sha256": sha256(mesh),
        "path": str(path.resolve()), "path_sha256": sha256(path),
        "mesh_command": command,
    })
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_preflight(args.config.resolve(), args.repo.resolve(), args.output.resolve())
    print(json.dumps({key: ledger[key] for key in (
        "physical_layers", "activation_slabs", "tracks_per_physical_layer",
        "recoat_time_s", "total_time_s", "nominal_laser_energy_J")}, indent=2))


if __name__ == "__main__":
    main()
