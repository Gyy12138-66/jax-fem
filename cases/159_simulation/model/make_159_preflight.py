#!/usr/bin/env python3
"""159 (0119 part) preflight: per-slab areas from the HEX8 mesh, flash schedule with per-layer
scan time / energy / commanded power, activation mapping, time and energy ledgers and the
runner contract -- the v2-cube-rs stage-1 contract transferred to an arbitrary layered mesh.

Runner contract (identical to make_v2_cube_preflight.py): one CSV row = one implicit step;
`layer` = 1-based activation slab; explicit recoat rows with --recoat-time 0; --dt = first row
dt; deposition z = slab top; `centroid` activation geometry; every cell of the mesh spans one
slab (checked). Geometry-derived quantities come from the mesh, so a new mesh = re-run this.

    make_159_preflight.py --config inputs/0119-flash.json --repo <repo> --output <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(CASE / "mesh_check"))
CUBE_MODEL = CASE.parent / "AM-Benchmark" / "verification" / "v2-cube-rs" / "model"
sys.path.insert(0, str(CUBE_MODEL))
from inp_health import parse, hex_metrics  # noqa: E402
from make_v2_cube_preflight import (  # noqa: E402
    add_row, recoat_substep_durations, relocate_material_config, sha256,
    vertex_rule_band_integral, write_schedule, PATH_FIELDS,
)

SCHEMA = "v159.flash-preflight/1"
LEDGER_SCHEMA = "v159.preflight-ledger/1"
CONTRACT_SCHEMA = "v2.cube-runner-contract/1"


# ----------------------------------------------------------------------------- mesh
def read_layered_mesh(inp: Path, mesh_cfg: dict) -> dict:
    nodes, elems, _, _ = parse(inp)
    ids = np.array(sorted(nodes))
    P = np.array([nodes[i] for i in ids])
    idx = {int(n): k for k, n in enumerate(ids)}
    blocks = {}
    for (etype, elset), rows in elems.items():
        if not etype.startswith("C3D8"):
            raise SystemExit(f"mesh must contain C3D8 only, found {etype}")
        blocks[elset] = np.vectorize(idx.get)(np.array([r[:9] for r in rows])[:, 1:])
    raft_h = float(mesh_cfg["raft_height_m"])
    dz = float(mesh_cfg["cell_height_m"])
    if mesh_cfg["part_elset"] in blocks:
        conn_part = blocks[mesh_cfg["part_elset"]]
        conn_raft = blocks.get(mesh_cfg["raft_elset"])
    else:
        conn_all = np.vstack(list(blocks.values()))
        zc_all = P[conn_all][:, :, 2].mean(1)
        conn_part = conn_all[zc_all > raft_h]
        conn_raft = conn_all[zc_all <= raft_h]
    if conn_raft is None or len(conn_raft) == 0:
        raise SystemExit("mesh has no raft cells (ELSET RAFT or cells below raft_height_m)")
    Jp, _ = hex_metrics(P[conn_part])
    Jr, _ = hex_metrics(P[conn_raft])
    if (Jp <= 0).any() or (Jr <= 0).any():
        raise SystemExit("mesh has non-positive centre Jacobians")
    vol = Jp * 8.0
    zc = P[conn_part][:, :, 2].mean(1)
    zp = P[conn_part][:, :, 2]
    base = float(zp.min())
    if not math.isclose(base, raft_h, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"part base z = {base} does not equal raft_height_m = {raft_h}")
    span = (zp.max(1) - zp.min(1)) / dz
    if np.abs(span - 1.0).max() > 1e-6:
        raise SystemExit("every part cell must span exactly one layer (cell_height_m)")
    levels = np.unique(np.round(P[:, 2] / dz))
    if np.abs(P[:, 2] / dz - np.round(P[:, 2] / dz)).max() * dz > 1e-9:
        raise SystemExit("node z-levels are not on the layer grid")
    k = np.floor((zc - base) / dz + 1e-9).astype(int)
    n_slabs = int(k.max()) + 1
    area = np.bincount(k, weights=vol, minlength=n_slabs) / dz
    cells = np.bincount(k, minlength=n_slabs)
    if (cells == 0).any():
        raise SystemExit(f"empty slabs in the mesh: {np.nonzero(cells == 0)[0].tolist()}")
    xy = P[conn_part][:, :, :2]
    part_min, part_max = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
    raft_xyz = P[conn_raft].reshape(-1, 3)
    return {
        "P": P, "conn_part": conn_part, "conn_raft": conn_raft, "vol": vol, "zc": zc, "slab": k,
        "n_slabs": n_slabs, "area_m2": area, "cells_per_slab": cells, "dz": dz, "base_z": base,
        "part_xy_min": part_min, "part_xy_max": part_max, "part_top_z": float(zp.max()),
        "raft_box": [float(raft_xyz[:, 0].min()), float(raft_xyz[:, 0].max()), float(raft_xyz[:, 1].min()),
                     float(raft_xyz[:, 1].max()), float(raft_xyz[:, 2].min()), float(raft_xyz[:, 2].max())],
        "z_levels": (levels * dz).tolist(), "part_cells": int(len(conn_part)), "raft_cells": int(len(conn_raft)),
        "part_volume_m3": float(vol.sum()), "nodes": int(len(P)),
    }


def slab_capture(mesh: dict, centre: np.ndarray, r: float) -> np.ndarray:
    """Fraction of the legacy Gaussian exp(-2 rho^2/r^2) (integral pi r^2/2) landing on each slab's
    cells: sum of top areas x intensity, evaluated at cell centroids."""
    xyc = mesh["P"][mesh["conn_part"]][:, :, :2].mean(1)
    rho2 = ((xyc - centre) ** 2).sum(1)
    w = (mesh["vol"] / mesh["dz"]) * np.exp(-2.0 * rho2 / r ** 2) * 2.0 / (math.pi * r ** 2)
    return np.bincount(mesh["slab"], weights=w, minlength=mesh["n_slabs"])


# ----------------------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise SystemExit("unsupported 159 preflight schema")
    layers, scan, runner, tm = cfg["layer_schedule"], cfg["scan"], cfg["runner"], cfg["thermal_mechanical"]
    physical = float(layers["physical_layer_thickness_m"])
    slab = float(layers["activation_slab_thickness_m"])
    per_slab = int(layers["physical_layers_per_slab"])
    # `physical_layers_per_slab` = flash events per slab; each flash carries `real_layers_per_flash`
    # real (40 um) layers of energy and clock so that the energy per flash per cell mass equals the
    # validated 0.2 mm-cell reading (slab 0.5 mm -> 12.5 real layers -> 5 flashes x 2.5 layers)
    real_per_slab = slab / physical
    layers["real_layers_per_slab"] = real_per_slab
    layers["real_layers_per_flash"] = real_per_slab / per_slab
    if layers.get("flash_grouping") != "energy per flash per cell mass = validated 0.2 mm reading" and not math.isclose(slab, physical * per_slab, abs_tol=1e-12):
        raise SystemExit("activation slab must equal physical layer thickness x layers per slab, or declare layer_schedule.flash_grouping")
    if not math.isclose(slab, float(cfg["mesh"]["cell_height_m"]), abs_tol=1e-12):
        raise SystemExit("activation slab thickness must equal the mesh cell height")
    if layers["deposition_mode"] != "flash" or layers["deposition_z_rule"] != "slab_top":
        raise SystemExit("this preflight implements the flash / slab_top reading only")
    fl = scan["flash"]
    if fl.get("power_rule") != "commanded(layer) = P x t_scan(layer) / (t_flash x capture(slab))":
        raise SystemExit("scan.flash.power_rule must be the registered energy-conserving rule")
    if float(layers["recoat_substep_ratio"]) < 1.0:
        raise SystemExit("recoat_substep_ratio must be >= 1")
    cutoff = float(runner["source_depth_cutoff_m"])
    if not (0.0 < cutoff < slab):
        raise SystemExit("runner.source_depth_cutoff_m must lie strictly inside one slab")
    band = vertex_rule_band_integral(slab, float(runner["source_depth_m"]), cutoff, bool(runner["source_cutoff_renormalize"]))
    if abs(band - 1.0) > 1e-3:
        raise SystemExit(f"discrete depth-band integral under the vertex rule is {band:.4f}, not 1")
    cons = runner["consolidation"]
    if float(cons["solidus_K"]) < 5000.0:
        raise SystemExit("consolidation.solidus_K must be a sentinel (>= 5000 K): melt detection off for the lumped reading")
    if float(cons["stress_relaxation_temperature_K"]) != float(tm["stress_free_reference_K"]):
        raise SystemExit("runner stress relaxation temperature must equal the stress-free reference")
    if runner["layer_activation_geometry"] != "centroid":
        raise SystemExit("layer_activation_geometry must be 'centroid' (intersection double-activates face-aligned slabs)")
    return cfg


# ----------------------------------------------------------------------------- schedule
def generate_schedule(cfg: dict, mesh: dict) -> tuple[list[dict], dict]:
    layers, scan, tm = cfg["layer_schedule"], cfg["scan"], cfg["thermal_mechanical"]
    fl = scan["flash"]
    dz = mesh["dz"]
    base = mesh["base_z"]
    per_slab = int(layers["physical_layers_per_slab"])
    rpf = float(layers["real_layers_per_flash"])
    physical_dz = dz / per_slab                       # flash-layer height (= real layer height when rpf == 1)
    n_slabs = mesh["n_slabs"]
    n_physical = n_slabs * per_slab
    power = float(scan["power_W"])
    speed = float(scan["speed_m_s"])
    hatch = float(scan["hatch_m"])
    absorptivity = float(cfg["runner"]["absorptivity"])
    r_flash = float(fl["beam_radius_m"])
    n_sub = int(fl["substeps"])
    t_flash = float(fl["duration_s"])
    recoat = float(layers["recoat_time_s"])
    recoat_steps = recoat_substep_durations(rpf * recoat, int(layers["recoat_substeps"]), float(layers["recoat_substep_ratio"]))
    extent = float(np.max(mesh["part_xy_max"] - mesh["part_xy_min"]))
    if r_flash < 5.0 * extent:
        raise SystemExit(f"scan.flash.beam_radius_m must be >= 5 x the part extent ({5*extent:.3f} m)")
    centre = 0.5 * (mesh["part_xy_min"] + mesh["part_xy_max"])
    xyc = mesh["P"][mesh["conn_part"]][:, :, :2].mean(1)
    rho_max = float(np.sqrt(((xyc - centre) ** 2).sum(1)).max())
    uniformity = math.exp(-2.0 * rho_max ** 2 / r_flash ** 2)
    capture = slab_capture(mesh, centre, r_flash)
    area = mesh["area_m2"]
    t_scan_slab = area / (hatch * speed)
    if (rpf * t_scan_slab <= t_flash).any():
        raise SystemExit("some slab's scan time is shorter than the flash duration")
    rows: list[dict] = []
    state = {"time": 0.0}
    scan_id = 0
    energy_J = 0.0
    scan_time_s = hold_time_s = recoat_time_s = 0.0
    layer_summaries, activation_events = [], []
    slab_energy: dict[int, float] = {}
    for physical_layer in range(1, n_physical + 1):
        slab = (physical_layer - 1) // per_slab + 1
        k = slab - 1
        physical_z = base + physical_layer * physical_dz
        deposition_z = base + slab * dz
        t_scan = float(t_scan_slab[k])
        t_hold = rpf * t_scan - t_flash
        hold_steps = recoat_substep_durations(t_hold, int(fl["hold_substeps"]), float(fl["hold_substep_ratio"]))
        p_cmd = rpf * power * t_scan / (t_flash * float(capture[k]))
        layer_start = state["time"]
        scan_id += 1
        first_row = None
        for _ in range(n_sub):
            add_row(rows, state, dt=t_flash / n_sub, x=float(centre[0]), y=float(centre[1]), z=deposition_z,
                    power=p_cmd, laser_on=1, layer=slab, hatch=1, mode="scan",
                    physical_layer=physical_layer, scan_id=scan_id, physical_z=physical_z)
            if first_row is None:
                first_row = len(rows) - 1
            energy_J += rpf * power * t_scan / n_sub
            scan_time_s += t_flash / n_sub
        for hold_dt in hold_steps:
            add_row(rows, state, dt=hold_dt, x=float(centre[0]), y=float(centre[1]), z=deposition_z, power=0.0,
                    laser_on=0, layer=slab, hatch=0, mode="hold",
                    physical_layer=physical_layer, scan_id=scan_id, physical_z=physical_z)
            hold_time_s += hold_dt
        slab_energy[slab] = slab_energy.get(slab, 0.0) + rpf * power * t_scan
        if (physical_layer - 1) % per_slab == 0:
            activation_events.append({"slab": slab, "physical_layer": physical_layer, "row_index": first_row,
                                      "global_step": first_row, "time_s": rows[first_row]["time"],
                                      "z_bottom_m": base + (slab - 1) * dz, "z_top_m": base + slab * dz})
        layer_summaries.append({"physical_layer": physical_layer, "activation_slab": slab, "z_m": physical_z,
                                "deposition_z_m": deposition_z, "area_m2": float(area[k]), "scan_time_s": t_scan,
                                "hold_time_s": t_hold, "capture_fraction": float(capture[k]),
                                "commanded_power_W": p_cmd, "nominal_laser_energy_J": rpf * power * t_scan, "real_layers": rpf,
                                "scan_start_time_s": layer_start, "scan_end_time_s": state["time"]})
        if physical_layer < n_physical or bool(layers["recoat_after_final_layer"]):
            for sub_dt in recoat_steps:
                add_row(rows, state, dt=sub_dt, x=float(centre[0]), y=float(centre[1]), z=deposition_z, power=0.0,
                        laser_on=0, layer=slab, hatch=0, mode="recoat",
                        physical_layer=physical_layer, scan_id=scan_id, physical_z=physical_z)
            recoat_time_s += rpf * recoat
    n_recoat = n_physical if layers["recoat_after_final_layer"] else n_physical - 1
    cooldown_s = float(tm["final_cooldown_s"])
    cooling_steps = int(tm["final_cooldown_steps"])
    ledger = {
        "schema": LEDGER_SCHEMA, "complete": True, "solver_started": False, "solver_compatible": True,
        "completion_scope": "path/ledger/runner-contract preflight artifacts only",
        "solver_compatibility_note": ("explicit recoat rows require --recoat-time 0; --dt equals the first row dt; "
                                      "`layer` = 1-based activation slab; --layers = slab count; --layer-thickness = "
                                      "slab thickness; per-row `power` is the commanded flash power of that layer"),
        "coordinate_frame": "mesh coordinates (raft bottom at z = 0, part base at z = raft height)",
        "substrate_z_m": base, "substrate_box_m": mesh["raft_box"],
        "part_bounds_xy_m": [[float(mesh["part_xy_min"][0]), float(mesh["part_xy_max"][0])],
                             [float(mesh["part_xy_min"][1]), float(mesh["part_xy_max"][1])]],
        "part_top_z_m": mesh["part_top_z"], "part_volume_m3": mesh["part_volume_m3"],
        "part_cells": mesh["part_cells"], "raft_cells": mesh["raft_cells"], "nodes": mesh["nodes"],
        "physical_layers": n_physical, "activation_slabs": n_slabs, "physical_layers_per_slab": per_slab,
        "real_layers_per_flash": rpf, "real_layers_total": rpf * n_physical, "real_layer_thickness_m": float(layers["physical_layer_thickness_m"]),
        "slab_area_m2": area.tolist(), "slab_cells": mesh["cells_per_slab"].tolist(),
        "slab_scan_time_s": t_scan_slab.tolist(), "slab_capture_fraction": capture.tolist(),
        "deposition_mode": "flash", "deposition_z_rule": layers["deposition_z_rule"],
        "flash": {"beam_radius_m": r_flash, "substeps": n_sub, "flash_duration_s": t_flash, "substep_dt_s": t_flash / n_sub,
                  "hold_substeps": int(fl["hold_substeps"]), "hold_substep_ratio": float(fl["hold_substep_ratio"]),
                  "centre_xy_m": [float(centre[0]), float(centre[1])], "uniformity_min_over_max": uniformity,
                  "rho_max_m": rho_max, "physical_power_W": power,
                  "capture_fraction_analytic_uniform": float(2.0 * area.mean() / (math.pi * r_flash ** 2)),
                  "capture_fraction_slab_min_max": [float(capture.min()), float(capture.max())],
                  "commanded_power_W_min_max": [float(min(l["commanded_power_W"] for l in layer_summaries)),
                                                float(max(l["commanded_power_W"] for l in layer_summaries))],
                  "physical_energy_per_layer_J_min_max": [float(rpf * power * t_scan_slab.min()), float(rpf * power * t_scan_slab.max())],
                  "physical_energy_per_layer_J": [l["nominal_laser_energy_J"] for l in layer_summaries],
                  "layer_scan_time_s_min_max": [float(t_scan_slab.min()), float(t_scan_slab.max())],
                  "power_rule": fl["power_rule"]},
        "scan_rows": sum(r["laser_on"] == 1 for r in rows), "jump_rows": 0,
        "hold_rows": sum(r["mode"] == "hold" for r in rows), "recoat_rows": n_recoat,
        "recoat_substep_rows": sum(r["mode"] == "recoat" for r in rows), "recoat_substeps_per_event": len(recoat_steps),
        "recoat_substep_durations_s": recoat_steps, "path_rows": len(rows), "first_row_dt_s": rows[0]["time"],
        "scan_time_s": scan_time_s, "hold_time_s": hold_time_s, "jump_time_s": 0.0, "recoat_time_s": recoat_time_s,
        "total_time_s": state["time"], "cooldown_time_s": cooldown_s, "cooling_steps": cooling_steps,
        "cooling_dt_s": cooldown_s / cooling_steps, "build_clock_s": state["time"] + cooldown_s,
        "expected_runner_steps": len(rows) + cooling_steps,
        "nominal_laser_energy_J": energy_J, "absorptivity": absorptivity,
        "absorbed_laser_energy_nominal_J": energy_J * absorptivity,
        "nominal_energy_per_slab_J": {str(k): v for k, v in sorted(slab_energy.items())},
        "activation_rule": layers["activation_rule"], "activation_events": activation_events, "layers": layer_summaries,
    }
    validate_schedule(rows, ledger, cfg, mesh)
    return rows, ledger


def validate_schedule(rows: list[dict], ledger: dict, cfg: dict, mesh: dict) -> None:
    layers, scan = cfg["layer_schedule"], cfg["scan"]
    per_slab = int(layers["physical_layers_per_slab"])
    n_physical = ledger["physical_layers"]
    expected_recoat = n_physical if layers["recoat_after_final_layer"] else n_physical - 1
    rpf = float(layers["real_layers_per_flash"])
    if ledger["recoat_rows"] != expected_recoat or ledger["recoat_substep_rows"] != expected_recoat * int(layers["recoat_substeps"]):
        raise ValueError("recoat bookkeeping mismatch")
    if ledger["scan_rows"] + ledger["hold_rows"] + ledger["recoat_substep_rows"] != ledger["path_rows"]:
        raise ValueError("row bookkeeping mismatch")
    if ledger["scan_rows"] != n_physical * ledger["flash"]["substeps"]:
        raise ValueError("flash row count mismatch")
    last, prev = 0.0, -math.inf
    scan_t = hold_t = recoat_t = 0.0
    by_physical: dict[int, set[int]] = {}
    bx, by = ledger["part_bounds_xy_m"]
    for row in rows:
        if row["time"] <= prev:
            raise ValueError("path times are not strictly increasing")
        dt = row["time"] - last
        last = prev = row["time"]
        if row["laser_on"]:
            if row["mode"] != "scan":
                raise ValueError("laser-on rows must carry mode 'scan'")
            by_physical.setdefault(row["physical_layer"], set()).add(row["layer"])
            scan_t += dt
            if not (bx[0] <= row["x"] <= bx[1] and by[0] <= row["y"] <= by[1]):
                raise ValueError("flash centre outside the part footprint box")
        elif row["mode"] == "hold":
            hold_t += dt
        elif row["mode"] == "recoat":
            recoat_t += dt
        else:
            raise ValueError(f"unexpected row mode {row['mode']!r}")
        if row["front_coord"] != row["z"]:
            raise ValueError("front_coord must equal the deposition z")
    for name, value in (("scan_time_s", scan_t), ("hold_time_s", hold_t), ("recoat_time_s", recoat_t)):
        # 50k rows of ~1e-4 s on a 2e4 s clock: double rounding of the row times is ~1e-8 s in total
        if not math.isclose(ledger[name], value, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"time ledger mismatch on {name}")
    if not math.isclose(ledger["total_time_s"], scan_t + hold_t + recoat_t, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("time ledger identity failed")
    if set(by_physical) != set(range(1, n_physical + 1)):
        raise ValueError("missing physical layer flash")
    for physical, slabs in by_physical.items():
        if slabs != {(physical - 1) // per_slab + 1}:
            raise ValueError("physical-layer to slab mapping mismatch")
    # per-layer energy identities: P x t_scan, flash + hold = t_scan, commanded x capture x t_flash = P x t_scan
    power, speed, hatch = float(scan["power_W"]), float(scan["speed_m_s"]), float(scan["hatch_m"])
    t_flash = ledger["flash"]["flash_duration_s"]
    for l in ledger["layers"]:
        k = l["activation_slab"] - 1
        if not math.isclose(l["scan_time_s"], mesh["area_m2"][k] / (hatch * speed), rel_tol=1e-12):
            raise ValueError("scan time rule broken")
        if not math.isclose(l["nominal_laser_energy_J"], rpf * power * l["scan_time_s"], rel_tol=1e-12):
            raise ValueError("layer energy identity failed")
        if not math.isclose(l["commanded_power_W"] * t_flash * l["capture_fraction"], rpf * power * l["scan_time_s"], rel_tol=1e-9):
            raise ValueError("flash power rule broken: commanded x capture x t_flash != rpf x P x t_scan")
        if not math.isclose(l["hold_time_s"] + t_flash, rpf * l["scan_time_s"], rel_tol=1e-12):
            raise ValueError("flash + hold must span the grouped layer scan time")
    if not math.isclose(sum(l["nominal_laser_energy_J"] for l in ledger["layers"]), ledger["nominal_laser_energy_J"], rel_tol=1e-9):
        raise ValueError("energy sum identity failed")
    if not math.isclose(sum(float(v) for v in ledger["nominal_energy_per_slab_J"].values()), ledger["nominal_laser_energy_J"], rel_tol=1e-9):
        raise ValueError("per-slab energy does not sum to the total")
    if ledger["flash"]["uniformity_min_over_max"] < 0.96:
        raise ValueError("flash is not uniform enough over the footprint")
    events = ledger["activation_events"]
    if len(events) != ledger["activation_slabs"]:
        raise ValueError("activation event count mismatch")
    dz = mesh["dz"]
    levels = np.array(mesh["z_levels"])
    for k, event in enumerate(events, start=1):
        if event["slab"] != k or event["physical_layer"] != (k - 1) * per_slab + 1:
            raise ValueError("activation event slab/physical-layer mismatch")
        row = rows[event["row_index"]]
        if not (row["laser_on"] == 1 and row["mode"] == "scan" and row["layer"] == k):
            raise ValueError("activation event does not point at a laser-on row of its slab")
        if any(r["layer"] == k and r["laser_on"] == 1 for r in rows[:event["row_index"]]):
            raise ValueError("activation event is not the first laser-on row of its slab")
        for key in ("z_bottom_m", "z_top_m"):
            if np.abs(levels - event[key]).min() > 1e-9:
                raise ValueError(f"slab boundary {event[key]} is not a mesh z-level")
        if not math.isclose(event["z_top_m"], mesh["base_z"] + k * dz, abs_tol=1e-12):
            raise ValueError("activation height mismatch")


def write_schedule_precise(path: Path, rows: list[dict]) -> None:
    """Like the cube writer but with 15 significant digits: the clock reaches ~2e4 s while the flash
    sub-steps are 1.5e-4 s, so 12 digits would round each dt by up to 1e-4 relative."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PATH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.15g}" if isinstance(value, float) else value for key, value in row.items()})


# ----------------------------------------------------------------------------- contract
def runner_contract(cfg: dict, *, mesh_path: Path, path: Path, material: Path, ledger: dict) -> dict:
    layers, tm, runner = cfg["layer_schedule"], cfg["thermal_mechanical"], cfg["runner"]
    cons, mech, out, lin = runner["consolidation"], runner["mechanics"], runner["output"], runner["linear_solver"]
    fl = ledger["flash"]
    raft = ledger["substrate_z_m"]
    box = ledger["substrate_box_m"]
    argv = [
        "--config", str(material), "--inp", str(mesh_path),
        "--path-file", str(path), "--path-length-scale", "1.0",
        "--build-axis", "z", "--base-side", "min",
        "--layer-thickness", f"{float(layers['activation_slab_thickness_m']):.12g}",
        "--layers", str(ledger["activation_slabs"]),
        "--support-thickness", f"{raft:.12g}",
        "--layer-activation-mode", "layer_on_scan",
        "--layer-activation-geometry", runner["layer_activation_geometry"],
        "--future-layer-mode", "void", "--active-window-below-layers", "0",
        "--inactive-mass-factor", "1.0", "--powder-mode", "powder",
        "--surface-selection", "exterior", "--boundary-tol", "1.0e-6",
        "--quadrature-order", "2", "--thermal-mass-lumping",
        "--source-model", "legacy",
        "--beam-radius", f"{fl['beam_radius_m']:.12g}",
        "--source-depth", f"{float(runner['source_depth_m']):.12g}",
        "--source-depth-cutoff", f"{float(runner['source_depth_cutoff_m']):.12g}",
        "--source-cutoff-renormalize" if runner["source_cutoff_renormalize"] else "--no-source-cutoff-renormalize",
        "--laser-power", f"{ledger['layers'][0]['commanded_power_W']:.12g}",
        "--absorptivity", f"{float(runner['absorptivity']):.12g}",
        "--dt", f"{ledger['first_row_dt_s']:.12g}",
        "--recoat-time", "0",
        "--solidus-temperature", f"{float(cons['solidus_K']):.12g}",
        "--liquidus-temperature", f"{float(cons['liquidus_K']):.12g}",
        "--latent-heat", f"{float(cons['latent_heat_J_kg']):.12g}",
        "--phase-history-model", cons["phase_history_model"],
        "--stress-relaxation-temperature", f"{float(cons['stress_relaxation_temperature_K']):.12g}",
        "--reset-activation-temperature",
        "--activation-reset-temperature", f"{float(cons['activation_reset_temperature_K']):.12g}",
        "--ambient", f"{float(tm['ambient_K']):.12g}",
        "--preheat-temperature", f"{float(tm['initial_temperature_K']):.12g}",
        "--bottom-thermal-bc", "fixed",
        "--bottom-temperature", f"{float(tm['initial_temperature_K']):.12g}",
        "--cooling-steps", str(int(tm["final_cooldown_steps"])),
        "--cooling-dt", f"{ledger['cooling_dt_s']:.12g}",
        "--mechanics-model", mech["model"], "--bottom-mechanics-bc", "fixed",
        "--mechanics-every", str(int(mech["every_steps"])),
        "--mechanics-rel-tol", f"{float(mech['rel_tol']):.12g}",
        "--mechanics-acceptance", mech["acceptance"],
        "--mechanics-max-iter", str(int(mech["max_iter"])),
        "--mechanics-max-cuts", str(int(mech["max_cuts"])),
        "--mechanics-temperature-floor", f"{float(mech['temperature_floor_K']):.12g}",
        "--thermal-output-every", str(int(out["thermal_output_every"])),
        "--mechanics-output-every", str(int(out["mechanics_output_every"])),
        "--summary-every", str(int(out["summary_every"])),
        "--xla-platform", lin["platform"], "--xla-preallocate", "off",
        "--xla-linear-solver", lin["solver"], "--xla-pardiso-mode", lin["pardiso_mode"],
    ]
    if mech["line_search"]:
        argv.append("--mechanics-line-search")
    if lin.get("cell_target_batch_size"):
        argv += ["--xla-cell-target-batch-size", str(int(lin["cell_target_batch_size"]))]
    if lin["solver"] == "jax":
        # jax_solver tuning: without --xla-jax-precond the acceleration wrapper
        # disables Jacobi preconditioning (its argparse default is False, the
        # opposite of upstream jax_fem's default True).
        if lin.get("jax_precond"):
            argv.append("--xla-jax-precond")
        if lin.get("jax_method"):
            argv += ["--xla-jax-method", str(lin["jax_method"])]
        if lin.get("jax_tol") is not None:
            argv += ["--xla-jax-tol", f"{float(lin['jax_tol']):.12g}"]
        if lin.get("jax_atol") is not None:
            argv += ["--xla-jax-atol", f"{float(lin['jax_atol']):.12g}"]
        if lin.get("jax_maxiter") is not None:
            argv += ["--xla-jax-maxiter", str(int(lin["jax_maxiter"]))]
        if lin.get("jax_skip_residual_check"):
            # The vendored jax_solve asserts an ABSOLUTE post-solve residual
            # (err < 0.1) regardless of tol; loosening jax_tol/jax_atol only
            # works with that check skipped. Newton-level convergence checks
            # (thermal tol, mechanics acceptance + cutback) remain in force.
            argv.append("--xla-jax-skip-residual-check")
    if tm["release_after_cooling"]:
        argv += ["--release-after-cooling", "--release-anchor-mode", "rigid_body",
                 "--release-cut-box", f"{box[0]:.12g}", f"{box[1]:.12g}", f"{box[2]:.12g}", f"{box[3]:.12g}",
                 f"{box[4]:.12g}", f"{raft:.12g}"]
    return {
        "schema": CONTRACT_SCHEMA, "argv": argv, "python_bin": lin.get("python_bin"), "platform": lin["platform"],
        "io_flags_added_by_launcher": ["--output-dir", "--profile-json", "--profile-label"],
        "deposition": {"mode": "flash", "flash": fl,
                       "note": ("reading A' (D-V2-11) on a real part: each physical layer's energy P x t_scan(layer), "
                                "t_scan = A(slab)/(hatch x v), is flashed in the cell dwell time; the per-row `power` "
                                "column carries that layer's commanded power P x t_scan/(t_flash x capture(slab)); "
                                "the ledger's laser_absorbed_nominal_j reads Ac x commanded x dt, so deposited / nominal "
                                "should equal capture(slab) and deposited per layer should equal Ac x P x t_scan(layer)")},
        "activation": {"mode": f"layer_on_scan/{runner['layer_activation_geometry']}",
                       "slab_thickness_m": float(layers["activation_slab_thickness_m"]), "slabs": ledger["activation_slabs"],
                       "trigger": "first laser-on row carrying layer == k activates slab k (cells whose centroid layer id <= k)",
                       "events": ledger["activation_events"],
                       "consolidation": "solidus = liquidus = sentinel: activated cells are solid immediately; T_ref once at activation"},
        "temperature_mapping": {"coupling": "staggered in one time loop; mechanics at global_step % mechanics_every == 0 and the final step; release after cooldown",
                                "stress_free_reference_K": float(tm["stress_free_reference_K"]),
                                "activation_reset_temperature_K": float(cons["activation_reset_temperature_K"]),
                                "mechanics_temperature_floor_K": float(mech["temperature_floor_K"]),
                                "mechanics_every_steps": int(mech["every_steps"])},
        "time": {"path_rows": ledger["path_rows"], "cooling_steps": ledger["cooling_steps"],
                 "expected_runner_steps": ledger["expected_runner_steps"], "build_clock_s": ledger["build_clock_s"],
                 "recoat": "explicit geometric sub-step rows; --recoat-time 0 so the runner inserts none"},
        "energy": {"nominal_laser_energy_J": ledger["nominal_laser_energy_J"],
                   "absorbed_laser_energy_nominal_J": ledger["absorbed_laser_energy_nominal_J"],
                   "deposition_band": f"legacy source, depth cut at {float(runner['source_depth_cutoff_m'])*1e6:.0f} um, renormalised; d = {float(runner['source_depth_m'])*1e6:.3f} um",
                   "vertex_rule_band_integral": vertex_rule_band_integral(float(layers["activation_slab_thickness_m"]), float(runner["source_depth_m"]),
                                                                          float(runner["source_depth_cutoff_m"]), bool(runner["source_cutoff_renormalize"])),
                   "capture_fraction": "per slab on the mesh (ledger slab_capture_fraction); measured by the stage-2 energy trial"},
        "release": {"mode": "rigid_body anchors on printed nodes", "cut_box_m": [box[0], box[1], box[2], box[3], box[4], raft],
                    "cut_semantics": "cells with centroid inside the box (the raft) lose stiffness and locked-in stress; the raft is the --support-thickness fixture band"},
    }


def build_preflight(config_path: Path, repo: Path, output: Path) -> dict:
    cfg = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    src_mesh = repo / cfg["mesh"]["inp"]
    if not src_mesh.is_file():
        raise SystemExit(f"mesh not found: {src_mesh}")
    mesh_path = output / src_mesh.name
    shutil.copyfile(src_mesh, mesh_path)
    mesh = read_layered_mesh(mesh_path, cfg["mesh"])
    rows, ledger = generate_schedule(cfg, mesh)
    path = output / "0119_path.csv"
    write_schedule_precise(path, rows)
    material, material_info = relocate_material_config(cfg, repo, output)
    contract = runner_contract(cfg, mesh_path=mesh_path, path=path, material=material, ledger=ledger)
    ledger.update({"config": str(config_path.resolve()), "config_sha256": sha256(config_path),
                   "mesh": str(mesh_path.resolve()), "mesh_source": str(src_mesh), "mesh_sha256": sha256(mesh_path),
                   "path": str(path.resolve()), "path_sha256": sha256(path),
                   "material_config": str(material.resolve()), "material_config_sha256": sha256(material),
                   "material_config_provenance": material_info,
                   "mesh_check": {"ok": True, "z_levels": len(mesh["z_levels"]), "part_base_z_m": mesh["base_z"],
                                  "part_top_z_m": mesh["part_top_z"], "raft_box_m": mesh["raft_box"]},
                   "runner_contract": str((output / "runner_contract.json").resolve())})
    contract["inputs"] = {k: ledger[k] for k in ("config_sha256", "mesh_sha256", "path_sha256", "material_config_sha256")}
    (output / "runner_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ledger["runner_contract_sha256"] = sha256(output / "runner_contract.json")
    slim = {k: v for k, v in ledger.items() if k != "layers"}
    slim["layers_first_last"] = [ledger["layers"][0], ledger["layers"][-1]]
    (output / "v2_cube_smoke_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "0119_ledger_summary.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    led = build_preflight(args.config.resolve(), args.repo.resolve(), args.output.resolve())
    fl = led["flash"]
    print(json.dumps({
        "part_cells": led["part_cells"], "raft_cells": led["raft_cells"], "nodes": led["nodes"],
        "activation_slabs": led["activation_slabs"], "physical_layers": led["physical_layers"],
        "slab_area_mm2_first_min_max_last": [led["slab_area_m2"][0] * 1e6, min(led["slab_area_m2"]) * 1e6, max(led["slab_area_m2"]) * 1e6, led["slab_area_m2"][-1] * 1e6],
        "layer_scan_time_s_min_max": fl["layer_scan_time_s_min_max"],
        "capture_fraction_slab_min_max": fl["capture_fraction_slab_min_max"], "uniformity_min_over_max": fl["uniformity_min_over_max"],
        "commanded_power_W_min_max": fl["commanded_power_W_min_max"],
        "path_rows": led["path_rows"], "expected_runner_steps": led["expected_runner_steps"],
        "build_clock_s": led["build_clock_s"], "nominal_laser_energy_J": led["nominal_laser_energy_J"],
        "absorbed_laser_energy_nominal_J": led["absorbed_laser_energy_nominal_J"],
        "release_cut_box_m": led["substrate_box_m"],
    }, indent=2))
    ev = led["activation_events"]
    for event in ev[:3] + ev[-2:]:
        print(f"  slab {event['slab']}: physical layer {event['physical_layer']} row {event['row_index']} "
              f"t={event['time_s']:.4f} s z=[{event['z_bottom_m']*1e3:.2f}, {event['z_top_m']*1e3:.2f}] mm")


if __name__ == "__main__":
    main()
