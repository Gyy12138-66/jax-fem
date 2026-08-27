#!/usr/bin/env python3
"""V2 cube stage-1 preflight: mesh, scan schedule, slab activation mapping,
time/energy ledgers and the runner contract -- without running a solver.

STRESS-REPRODUCTION-PLAN.md stage 1 (geometry / activation / scan events) plus
the contract half of stage 2. Every artefact is fingerprinted so the stage-2
smoke binds to exactly these inputs.

Runner contract (verified against jax_fem_am/process/scan_path.py::
generate_path_file_step_states and jax_fem_am/process/activation.py on this
branch; nothing in the shared solver is modified):

* one CSV row = one implicit step; dt = time difference to the previous row,
  the first row uses --dt, so --dt is emitted equal to the first row's dt;
* `layer` is the 1-based ACTIVATION SLAB index (runner: layer_idx = layer-1).
  With --layer-activation-mode layer_on_scan --layer-activation-geometry
  centroid, slab k = cells whose centroid layer id ceil(z_c/lt) == k above
  the part base, activated by the FIRST laser-on "scan" row carrying
  layer == k (should_activate_layer_for_state). That is exactly the config's
  activation_rule; the event list below records the rows that trigger it.
  The `intersection` geometry is refused for this model: its band test is
  boundary-inclusive, and with slab faces on cell faces it prints slab k+1
  together with slab k (seen on the first attempt: 13600 = 12800 + 2 x 400
  printed cells on the very first scan row);
* recoats are EXPLICIT rows (mode "recoat", laser_on 0, geometric sub-steps),
  therefore the runner is given --recoat-time 0 -- otherwise it would insert
  its own recoat at every SLAB transition (4 of them) and could never see the
  24 physical-layer recoats;
* extra columns (physical_layer, physical_z) are ignored by the runner's
  csv.DictReader; scan_id is consumed;
* deposition z (`z`/`front_coord`) follows layer_schedule.deposition_z_rule.
  "slab_top" places every physical-layer scan of a slab on the slab's top
  face so the exponential depth profile, cut at one slab thickness and
  renormalised, deposits into the active slab cell: the lumped-slab reading
  of Balbaa's point source (D-V2-10 / D-V2-11). "physical" keeps the real
  layer height (audit only: with a 200 um cell the upper Gauss point then
  sits above the laser for layers 1-4 of each slab and receives nothing).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

SCHEMA = "v2.cube-stress-smoke/2"
LEDGER_SCHEMA = "v2.cube-preflight-ledger/2"
CONTRACT_SCHEMA = "v2.cube-runner-contract/1"
PATH_FIELDS = ["time", "x", "y", "z", "power", "laser_on", "layer", "hatch",
               "mode", "front_coord", "physical_layer", "scan_id", "physical_z"]
DEPOSITION_MODES = ("serpentine", "flash")


def geom(g: dict, key: str):
    """Modelled geometry: `model_<key>` wins over the legacy `smoke_<key>` name.

    The smoke config named the modelled (reduced) geometry `smoke_*`; the
    production config models Balbaa's real geometry and uses `model_*`. Both
    families are accepted so the two configs share one generator.
    """
    if f"model_{key}" in g:
        return g[f"model_{key}"]
    return g[f"smoke_{key}"]


def flash_capture_fraction(half_side: float, beam_radius: float) -> float:
    """Fraction of the legacy in-plane Gaussian exp(-2 rho^2 / r^2) (integral
    pi r^2 / 2) that falls inside a centred square of half side a:
    erf(sqrt(2) a / r)^2. Closed form, no quadrature."""
    return math.erf(math.sqrt(2.0) * half_side / beam_radius) ** 2


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"{name} must be finite")
    return result


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise SystemExit("unsupported cube smoke schema")
    g, layers, scan = cfg["geometry"], cfg["layer_schedule"], cfg["scan"]
    # normalise the geometry family: expose model_* as smoke_* so the rest of
    # the validation (written for the smoke keys) applies to both configs
    for key in ("part_xy_m", "part_z_m", "substrate_xy_m", "substrate_z_m", "substrate_grading"):
        if f"model_{key}" in g:
            g[f"smoke_{key}"] = g[f"model_{key}"]
    if "model_physical_layers" in layers:
        layers["smoke_physical_layers"] = layers["model_physical_layers"]
    mode = layers.get("deposition_mode", "serpentine")
    if mode not in DEPOSITION_MODES:
        raise SystemExit(f"layer_schedule.deposition_mode must be one of {DEPOSITION_MODES}")
    layers["deposition_mode"] = mode
    if mode == "flash":
        fl = scan.get("flash")
        if not isinstance(fl, dict):
            raise SystemExit("flash deposition needs a scan.flash block")
        _positive_integer(fl["substeps"], "scan.flash.substeps")
        r = _positive_finite(fl["beam_radius_m"], "scan.flash.beam_radius_m")
        if r < 5.0 * float(g["smoke_part_xy_m"]):
            raise SystemExit("scan.flash.beam_radius_m must be >= 5 x the part side for a near-uniform flash "
                             "(centre-to-corner drop < 4 %)")
        if fl.get("power_rule") != "commanded = P / capture(footprint)":
            raise SystemExit("scan.flash.power_rule must be the registered energy-conserving rule")
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
    for key in ("part_xy_m", "smoke_part_xy_m", "smoke_part_z_m", "production_part_z_m",
                "substrate_xy_m", "substrate_z_m", "smoke_substrate_xy_m",
                "smoke_substrate_z_m", "cell_size_m"):
        _positive_finite(g[key], f"geometry.{key}")
    cell = float(g["cell_size_m"])
    if not math.isclose(slab, cell, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("activation slab thickness must equal the cell size (one slab = one cell row)")
    for key in ("smoke_part_xy_m", "smoke_substrate_xy_m", "smoke_substrate_z_m"):
        ratio = float(g[key]) / cell
        if abs(ratio - round(ratio)) > 1e-9:
            raise SystemExit(f"geometry.{key} must be an integer number of cells")
    if float(g["smoke_part_xy_m"]) > float(g["part_xy_m"]):
        raise SystemExit("smoke footprint cannot exceed the production footprint")
    if float(g["smoke_substrate_xy_m"]) < float(g["smoke_part_xy_m"]):
        raise SystemExit("smoke substrate footprint must contain the part")
    for key in ("substrate_grading", "smoke_substrate_grading"):
        if g.get(key) is not None:
            grading = _positive_finite(g[key], f"geometry.{key}")
            if grading <= 1.0:
                raise SystemExit(f"geometry.{key} must exceed 1")
    for key in ("power_W", "speed_m_s", "hatch_m", "sample_step_m", "jump_speed_m_s"):
        _positive_finite(scan[key], f"scan.{key}")
    for key in ("start_angle_deg", "rotation_per_physical_layer_deg"):
        _finite(scan[key], f"scan.{key}")
    _positive_finite(layers["recoat_time_s"], "layer_schedule.recoat_time_s")
    _positive_integer(layers["recoat_substeps"], "layer_schedule.recoat_substeps")
    ratio = _positive_finite(layers["recoat_substep_ratio"], "layer_schedule.recoat_substep_ratio")
    if ratio < 1.0:
        raise SystemExit("layer_schedule.recoat_substep_ratio must be >= 1")
    if layers.get("deposition_z_rule") not in ("slab_top", "physical"):
        raise SystemExit("layer_schedule.deposition_z_rule must be 'slab_top' or 'physical'")
    if not isinstance(layers["recoat_after_final_layer"], bool):
        raise SystemExit("layer_schedule.recoat_after_final_layer must be boolean")
    if not isinstance(scan["serpentine"], bool) or scan["serpentine"] is not True:
        raise SystemExit("scan.serpentine must be true for this preflight")
    margin = scan["margin_m"]
    if isinstance(margin, bool) or not math.isfinite(float(margin)) or float(margin) < 0:
        raise SystemExit("scan.margin_m must be finite and non-negative")
    if 2 * float(margin) >= float(g["smoke_part_xy_m"]):
        raise SystemExit("scan margin leaves no exposure area")
    tm = cfg["thermal_mechanical"]
    _positive_finite(tm["final_cooldown_s"], "thermal_mechanical.final_cooldown_s")
    _positive_integer(tm["final_cooldown_steps"], "thermal_mechanical.final_cooldown_steps")
    for key in ("initial_temperature_K", "ambient_K", "stress_free_reference_K"):
        _positive_finite(tm[key], f"thermal_mechanical.{key}")
    runner = cfg["runner"]
    for key in ("absorptivity", "beam_radius_m", "source_depth_m"):
        _positive_finite(runner[key], f"runner.{key}")
    cutoff = _finite(runner["source_depth_cutoff_m"], "runner.source_depth_cutoff_m")
    if cutoff < 0:
        raise SystemExit("runner.source_depth_cutoff_m must be non-negative")
    if runner["source_cutoff_renormalize"] and cutoff <= 0:
        raise SystemExit("source_cutoff_renormalize requires a positive cutoff")
    if not (0.0 < cutoff < slab):
        # vertex-collocated source (--thermal-mass-lumping): the slab-bottom
        # vertices sit at depth == slab and would be sampled twice (slab cell
        # + substrate cell). The band must stop short of them.
        raise SystemExit("runner.source_depth_cutoff_m must lie strictly inside one slab (0 < cutoff < slab)")
    band = vertex_rule_band_integral(slab, float(runner["source_depth_m"]), cutoff,
                                     bool(runner["source_cutoff_renormalize"]))
    if abs(band - 1.0) > 1e-3:
        raise SystemExit(f"discrete depth-band integral under the vertex rule is {band:.4f}, not 1: "
                         "choose source_depth_m so that d(1 - exp(-cutoff/d)) = slab/2")
    cons = runner["consolidation"]
    if float(cons["liquidus_K"]) < float(cons["solidus_K"]):
        raise SystemExit("liquidus must not be below solidus")
    if float(cons["solidus_K"]) < 5000.0:
        # melt detection must be OFF for the lumped-slab reading: with a
        # physical solidus the v06 lifecycle raises a reference-reset event on
        # nearly every scan step (peaks ~1.7-2.0e3 K on 200 um cells), each
        # forcing an unplanned mechanics solve and erasing eqp/T_ref on every
        # re-scan of a slab (measured 2026-08-26: 2.27 s/step, ~10 h).
        raise SystemExit("consolidation.solidus_K must be a sentinel above any reachable temperature (>= 5000 K)")
    if float(cons["latent_heat_J_kg"]) > 0 and float(cons["liquidus_K"]) <= float(cons["solidus_K"]):
        raise SystemExit("positive latent heat needs liquidus > solidus")
    if float(cons["stress_relaxation_temperature_K"]) != float(tm["stress_free_reference_K"]):
        raise SystemExit("runner stress relaxation temperature must equal the stress-free reference")
    _positive_integer(runner["mechanics"]["every_steps"], "runner.mechanics.every_steps")
    geometry_rule = runner.get("layer_activation_geometry")
    if geometry_rule not in ("centroid", "intersection"):
        raise SystemExit("runner.layer_activation_geometry must be 'centroid' or 'intersection'")
    if geometry_rule == "intersection":
        # cells_intersect_distance_band() is boundary-inclusive; with slab faces
        # on cell faces the band [0, k*lt] also captures slab k+1's touching
        # cells, so slab k+1 would be printed together with slab k.
        raise SystemExit("layer_activation_geometry 'intersection' double-activates face-aligned slabs; use 'centroid'")
    return cfg


def vertex_rule_band_integral(slab: float, depth: float, cutoff: float, renormalize: bool) -> float:
    """Discrete depth-band integral of the legacy exponential source under the
    HEX8 vertex-collocation rule installed by --thermal-mass-lumping.

    Samples: slab-top vertices at depth 0 (weight slab/2) and slab-bottom
    vertices at depth == slab (weight slab/2, counted by the slab cell AND by
    the substrate cell below). Exact continuum value of the band is
    d(1 - exp(-cutoff/d)); the kernel divides by that when renormalising.
    """
    top = 1.0 if 0.0 <= cutoff else 0.0
    bottom = 2.0 * math.exp(-slab / depth) if slab <= cutoff else 0.0
    discrete = 0.5 * slab * (top + bottom)
    norm = depth * (1.0 - math.exp(-cutoff / depth)) if renormalize else depth
    return discrete / norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_row(rows: list[dict], state: dict, *, dt: float, x: float, y: float,
            z: float, power: float, laser_on: int, layer: int, hatch: int,
            mode: str, physical_layer: int, scan_id: int, physical_z: float) -> None:
    if dt <= 0:
        raise ValueError("path dt must be positive")
    state["time"] += dt
    rows.append({
        "time": state["time"], "x": x, "y": y, "z": z,
        "power": power, "laser_on": laser_on, "layer": layer,
        "hatch": hatch, "mode": mode, "front_coord": z,
        "physical_layer": physical_layer, "scan_id": scan_id,
        "physical_z": physical_z,
    })


def recoat_substep_durations(total: float, count: int, ratio: float) -> list[float]:
    """Geometric sub-steps summing exactly to ``total`` (short first: the
    fast initial cooling after a scan is where the thermal history bends)."""
    if count == 1 or math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return [total / count] * count
    first = total * (ratio - 1.0) / (ratio ** count - 1.0)
    durations = [first * ratio ** i for i in range(count)]
    durations[-1] += total - sum(durations)
    return durations


def generate_schedule(cfg: dict, *, footprint_m: float | None = None,
                      substrate_xy_m: float | None = None) -> tuple[list[dict], dict]:
    g, layers, scan = cfg["geometry"], cfg["layer_schedule"], cfg["scan"]
    tm = cfg["thermal_mechanical"]
    side = float(g["smoke_part_xy_m"]) if footprint_m is None else float(footprint_m)
    sub_xy = float(g["smoke_substrate_xy_m"]) if substrate_xy_m is None else float(substrate_xy_m)
    if sub_xy < side:
        raise SystemExit("substrate footprint must contain the part")
    # MESH coordinates: make_v2_mesh_cube.py centres the part column on the
    # substrate footprint, so the part occupies [origin, origin + side] in x
    # and y. The runner takes path x/y verbatim (--path-length-scale 1), hence
    # the schedule is emitted in mesh coordinates. (First attempt wrote
    # part-local coordinates and put half of every layer's scan over bare
    # substrate -- caught by the capture ledger, 2026-08-26.)
    origin = 0.5 * (sub_xy - side)
    sub_z = float(g["smoke_substrate_z_m"])
    physical_dz = float(layers["physical_layer_thickness_m"])
    slab_dz = float(layers["activation_slab_thickness_m"])
    per_slab = int(layers["physical_layers_per_slab"])
    n_physical = int(layers["smoke_physical_layers"])
    recoat = float(layers["recoat_time_s"])
    recoat_steps = recoat_substep_durations(recoat, int(layers["recoat_substeps"]),
                                            float(layers["recoat_substep_ratio"]))
    z_rule = layers["deposition_z_rule"]
    margin = float(scan["margin_m"])
    hatch_space = float(scan["hatch_m"])
    sample_step = float(scan["sample_step_m"])
    speed = float(scan["speed_m_s"])
    jump_speed = float(scan["jump_speed_m_s"])
    power = float(scan["power_W"])
    absorptivity = float(cfg["runner"]["absorptivity"])
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
    activation_events = []
    slab_energy: dict[int, float] = {}

    exposure_width = side - 2 * margin
    n_tracks = int(math.ceil(exposure_width / hatch_space))
    lo, hi = origin + margin, origin + side - margin
    mode = layers.get("deposition_mode", "serpentine")
    flash = None
    if mode == "flash":
        # Reading A (D-V2-11): every physical layer's scan is replaced by a
        # uniform "flash" over the whole layer for the REAL scan duration.
        # Implemented without solver changes as a legacy Gaussian of a radius
        # much larger than the part (centre-to-corner drop erf-small) whose
        # commanded power is scaled by 1/capture so that the energy captured
        # inside the footprint equals the physical layer energy P * t_scan.
        fl = scan["flash"]
        r_flash = float(fl["beam_radius_m"])
        n_sub = int(fl["substeps"])
        capture = flash_capture_fraction(0.5 * side, r_flash)
        t_scan_layer = n_tracks * exposure_width / speed          # real serpentine time, jumps excluded
        p_flash = power / capture
        flash = {"beam_radius_m": r_flash, "substeps": n_sub, "capture_fraction_analytic": capture,
                 "uniformity_min_over_max": math.exp(-2.0 * 2.0 * (0.5 * side) ** 2 / r_flash ** 2),
                 "layer_scan_time_s": t_scan_layer, "substep_dt_s": t_scan_layer / n_sub,
                 "physical_power_W": power, "commanded_power_W": p_flash,
                 "physical_energy_per_layer_J": power * t_scan_layer,
                 "commanded_energy_per_layer_J": p_flash * t_scan_layer,
                 "centre_xy_m": [origin + 0.5 * side, origin + 0.5 * side]}
    for physical_layer in range(1, n_physical + 1):
        slab = (physical_layer - 1) // per_slab + 1
        physical_z = sub_z + physical_layer * physical_dz
        deposition_z = sub_z + slab * slab_dz if z_rule == "slab_top" else physical_z
        angle = (start_angle + (physical_layer - 1) * rotation) % 180.0
        axis = "x" if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else "y"
        if axis == "y" and not math.isclose(angle, 90.0, abs_tol=1e-9):
            raise SystemExit("stage-1 generator currently supports only 0/90 degree layers")
        layer_start = state["time"]
        layer_energy = 0.0
        first_scan_row = None
        if flash is not None:
            scan_id += 1
            cx, cy = flash["centre_xy_m"]
            for _ in range(flash["substeps"]):
                add_row(rows, state, dt=flash["substep_dt_s"], x=cx, y=cy, z=deposition_z,
                        power=flash["commanded_power_W"], laser_on=1, layer=slab, hatch=1,
                        mode="scan", physical_layer=physical_layer, scan_id=scan_id,
                        physical_z=physical_z)
                if first_scan_row is None:
                    first_scan_row = len(rows) - 1
                # PHYSICAL energy: commanded x captured fraction == P x dt
                energy_J += power * flash["substep_dt_s"]
                layer_energy += power * flash["substep_dt_s"]
                scan_time_s += flash["substep_dt_s"]
            previous = (cx, cy, deposition_z)
        for track in (range(n_tracks) if flash is None else ()):
            cross = origin + (side - (n_tracks - 1) * hatch_space) / 2 + track * hatch_space
            forward = track % 2 == 0
            a, b = (lo, hi) if forward else (hi, lo)
            start = (a, cross) if axis == "x" else (cross, a)
            end = (b, cross) if axis == "x" else (cross, b)
            scan_id += 1
            if previous is not None:
                distance = math.dist(previous, (start[0], start[1], deposition_z))
                dt = max(distance / jump_speed, 1e-8)
                jump_time_s += dt
                add_row(rows, state, dt=dt, x=start[0], y=start[1], z=deposition_z,
                        power=0.0, laser_on=0, layer=slab, hatch=track + 1,
                        mode="jump", physical_layer=physical_layer, scan_id=scan_id,
                        physical_z=physical_z)
            length = math.dist(start, end)
            segments = max(int(math.ceil(length / sample_step)), 1)
            dt = (length / segments) / speed
            for segment in range(1, segments + 1):
                fraction = segment / segments
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                add_row(rows, state, dt=dt, x=x, y=y, z=deposition_z, power=power,
                        laser_on=1, layer=slab, hatch=track + 1, mode="scan",
                        physical_layer=physical_layer, scan_id=scan_id,
                        physical_z=physical_z)
                if first_scan_row is None:
                    first_scan_row = len(rows) - 1
                energy_J += power * dt
                layer_energy += power * dt
                scan_time_s += dt
            previous = (end[0], end[1], deposition_z)
        slab_energy[slab] = slab_energy.get(slab, 0.0) + layer_energy
        if (physical_layer - 1) % per_slab == 0:
            # the runner activates slab k on this very row (first laser-on
            # scan row whose layer column reads k); row index == global_step
            # because no runner-side recoat rows are inserted (--recoat-time 0)
            activation_events.append({
                "slab": slab, "physical_layer": physical_layer,
                "row_index": first_scan_row, "global_step": first_scan_row,
                "time_s": rows[first_scan_row]["time"],
                "z_bottom_m": sub_z + (slab - 1) * slab_dz,
                "z_top_m": sub_z + slab * slab_dz,
            })
        layer_summaries.append({
            "physical_layer": physical_layer, "activation_slab": slab,
            "z_m": physical_z, "deposition_z_m": deposition_z,
            "angle_deg": angle, "axis": axis,
            "tracks": n_tracks, "scan_start_time_s": layer_start,
            "scan_end_time_s": state["time"],
            "scan_duration_s": state["time"] - layer_start,
            "nominal_laser_energy_J": layer_energy,
        })
        if physical_layer < n_physical or bool(layers["recoat_after_final_layer"]):
            for sub_dt in recoat_steps:
                add_row(rows, state, dt=sub_dt, x=previous[0], y=previous[1],
                        z=deposition_z, power=0.0, laser_on=0, layer=slab, hatch=0,
                        mode="recoat", physical_layer=physical_layer, scan_id=scan_id,
                        physical_z=physical_z)
            recoat_time_s += recoat

    expected_slabs = n_physical // per_slab
    n_recoat_events = n_physical if layers["recoat_after_final_layer"] else n_physical - 1
    cooldown_s = float(tm["final_cooldown_s"])
    cooling_steps = int(tm["final_cooldown_steps"])
    first_dt = rows[0]["time"]
    ledger = {
        "schema": LEDGER_SCHEMA,
        "complete": True,
        "completion_scope": "mesh/path/ledger/runner-contract preflight artifacts only",
        "solver_started": False,
        "solver_compatible": True,
        "solver_compatibility_note": (
            "explicit recoat rows require --recoat-time 0; --dt equals the first "
            "row dt; `layer` = 1-based activation slab; --layers = slab count; "
            "--layer-thickness = slab thickness; extra columns are ignored by "
            "the runner's csv.DictReader"),
        "footprint_m": side,
        "substrate_xy_m": sub_xy,
        "substrate_z_m": sub_z,
        "coordinate_frame": "mesh coordinates (part column centred on the substrate footprint)",
        "part_origin_xy_m": origin,
        "part_bounds_xy_m": [origin, origin + side],
        "physical_layers": n_physical,
        "activation_slabs": expected_slabs,
        "physical_layers_per_slab": per_slab,
        "tracks_per_physical_layer": n_tracks,
        "track_cross_min_m": origin + (side - (n_tracks - 1) * hatch_space) / 2,
        "track_cross_max_m": origin + (side + (n_tracks - 1) * hatch_space) / 2,
        "exposure_bounds_m": [origin + margin, origin + side - margin],
        "deposition_mode": mode,
        "flash": flash,
        "deposition_z_rule": z_rule,
        "scan_rows": sum(row["laser_on"] == 1 for row in rows),
        "jump_rows": sum(row["mode"] == "jump" for row in rows),
        "recoat_rows": n_recoat_events,
        "recoat_substep_rows": sum(row["mode"] == "recoat" for row in rows),
        "recoat_substeps_per_event": len(recoat_steps),
        "recoat_substep_durations_s": recoat_steps,
        "path_rows": len(rows),
        "first_row_dt_s": first_dt,
        "scan_time_s": scan_time_s,
        "jump_time_s": jump_time_s,
        "recoat_time_s": recoat_time_s,
        "total_time_s": state["time"],
        "cooldown_time_s": cooldown_s,
        "cooling_steps": cooling_steps,
        "cooling_dt_s": cooldown_s / cooling_steps,
        "build_clock_s": state["time"] + cooldown_s,
        "expected_runner_steps": len(rows) + cooling_steps,
        "nominal_laser_energy_J": energy_J,
        "absorptivity": absorptivity,
        "absorbed_laser_energy_nominal_J": energy_J * absorptivity,
        "nominal_energy_per_slab_J": {str(k): v for k, v in sorted(slab_energy.items())},
        "activation_rule": layers["activation_rule"],
        "activation_events": activation_events,
        "angles_deg": [item["angle_deg"] for item in layer_summaries],
        "layers": layer_summaries,
    }
    validate_schedule(rows, ledger, cfg)
    return rows, ledger


def validate_schedule(rows: list[dict], ledger: dict, cfg: dict) -> None:
    layers = cfg["layer_schedule"]
    g = cfg["geometry"]
    n_physical = int(layers["smoke_physical_layers"])
    per_slab = int(layers["physical_layers_per_slab"])
    slab_dz = float(layers["activation_slab_thickness_m"])
    sub_z = float(g["smoke_substrate_z_m"])
    expected_recoat = n_physical if layers["recoat_after_final_layer"] else n_physical - 1
    if ledger["recoat_rows"] != expected_recoat:
        raise ValueError("recoat event count mismatch")
    if ledger["recoat_substep_rows"] != expected_recoat * int(layers["recoat_substeps"]):
        raise ValueError("recoat sub-step row count mismatch")
    if not math.isclose(ledger["recoat_time_s"], expected_recoat * float(layers["recoat_time_s"]),
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("recoat duration mismatch")
    if not math.isclose(sum(ledger["recoat_substep_durations_s"]), float(layers["recoat_time_s"]),
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("recoat sub-steps do not sum to the recoat time")
    previous = -math.inf
    by_physical: dict[int, set[int]] = {}
    scan_time = 0.0
    jump_time = 0.0
    recoat_time = 0.0
    last_time = 0.0
    for row in rows:
        if row["time"] <= previous:
            raise ValueError("path times are not strictly increasing")
        dt = row["time"] - last_time
        last_time = row["time"]
        previous = row["time"]
        if row["laser_on"]:
            by_physical.setdefault(row["physical_layer"], set()).add(row["layer"])
            scan_time += dt
            if row["mode"] != "scan":
                raise ValueError("laser-on rows must carry mode 'scan'")
        elif row["mode"] == "jump":
            jump_time += dt
        elif row["mode"] == "recoat":
            recoat_time += dt
        else:
            raise ValueError(f"unexpected row mode {row['mode']!r}")
        if row["front_coord"] != row["z"]:
            raise ValueError("front_coord must equal the deposition z")
        if row["laser_on"] and not (ledger["part_bounds_xy_m"][0] <= row["x"] <= ledger["part_bounds_xy_m"][1]
                                    and ledger["part_bounds_xy_m"][0] <= row["y"] <= ledger["part_bounds_xy_m"][1]):
            raise ValueError("laser-on row outside the part footprint (coordinate frame error)")
    for name, value in (("scan_time_s", scan_time), ("jump_time_s", jump_time),
                        ("recoat_time_s", recoat_time)):
        if not math.isclose(ledger[name], value, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"time ledger mismatch on {name}")
    if not math.isclose(ledger["total_time_s"], scan_time + jump_time + recoat_time,
                        rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("time ledger identity failed")
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
    events = ledger["activation_events"]
    if len(events) != ledger["activation_slabs"]:
        raise ValueError("activation event count mismatch")
    seen_scan_before = False
    for k, event in enumerate(events, start=1):
        if event["slab"] != k or event["physical_layer"] != (k - 1) * per_slab + 1:
            raise ValueError("activation event slab/physical-layer mismatch")
        row = rows[event["row_index"]]
        if not (row["laser_on"] == 1 and row["mode"] == "scan" and row["layer"] == k):
            raise ValueError("activation event does not point at a laser-on scan row of its slab")
        if any(r["layer"] == k and r["laser_on"] == 1 for r in rows[:event["row_index"]]):
            raise ValueError("activation event is not the first laser-on row of its slab")
        if not math.isclose(event["z_top_m"], sub_z + k * slab_dz, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("activation height mismatch")
        if k > 1 and event["time_s"] <= events[k - 2]["time_s"]:
            raise ValueError("activation events must be time ordered")
        seen_scan_before = True
    if not seen_scan_before:
        raise ValueError("no activation events")
    # scan direction alternates track by track (serpentine) within each layer
    for layer in (ledger["layers"] if ledger["deposition_mode"] == "serpentine" else ()):
        scan_rows = [r for r in rows if r["laser_on"] == 1 and r["physical_layer"] == layer["physical_layer"]]
        by_track: dict[int, list[dict]] = {}
        for r in scan_rows:
            by_track.setdefault(r["hatch"], []).append(r)
        coord = "x" if layer["axis"] == "x" else "y"
        directions = []
        for track in sorted(by_track):
            pts = by_track[track]
            directions.append(1 if pts[-1][coord] > pts[0][coord] else -1)
        if any(directions[i] == directions[i + 1] for i in range(len(directions) - 1)):
            raise ValueError("serpentine direction does not alternate")
    # energy identity: nominal = P * exposed length / v per layer, summed
    # (flash mode deposits the same physical energy: commanded x capture == P x t_scan)
    scan_cfg = cfg["scan"]
    per_layer = (float(scan_cfg["power_W"]) * (ledger["footprint_m"] - 2 * float(scan_cfg["margin_m"]))
                 / float(scan_cfg["speed_m_s"]) * ledger["tracks_per_physical_layer"])
    if not math.isclose(ledger["nominal_laser_energy_J"], per_layer * n_physical, rel_tol=1e-9):
        raise ValueError("energy identity failed")
    if ledger["deposition_mode"] == "flash":
        fl = ledger["flash"]
        if not math.isclose(fl["commanded_energy_per_layer_J"] * fl["capture_fraction_analytic"],
                            fl["physical_energy_per_layer_J"], rel_tol=1e-12):
            raise ValueError("flash power rule broken: commanded x capture != physical")
        if fl["uniformity_min_over_max"] < 0.96:
            raise ValueError("flash is not uniform enough over the footprint")
        on_rows = [r for r in rows if r["laser_on"] == 1]
        if any(not (math.isclose(r["x"], fl["centre_xy_m"][0]) and math.isclose(r["y"], fl["centre_xy_m"][1]))
               for r in on_rows):
            raise ValueError("flash rows must sit at the footprint centre")
        if len(on_rows) != n_physical * fl["substeps"]:
            raise ValueError("flash row count mismatch")
    if not math.isclose(sum(float(v) for v in ledger["nominal_energy_per_slab_J"].values()),
                        ledger["nominal_laser_energy_J"], rel_tol=1e-9):
        raise ValueError("per-slab energy does not sum to the total")


def write_schedule(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PATH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in row.items()})


def mesh_command(cfg: dict, repo: Path, mesh: Path) -> list[str]:
    g = cfg["geometry"]
    mesh_script = repo / "cases/AM-Benchmark/verification/v2-cube-rs/model/make_v2_mesh_cube.py"
    command = [sys.executable, str(mesh_script), "--res", str(g["cell_size_m"]),
               "--part-xy", str(g["smoke_part_xy_m"]), "--part-z", str(g["smoke_part_z_m"]),
               "--sub-xy", str(g["smoke_substrate_xy_m"]), "--sub-z", str(g["smoke_substrate_z_m"]),
               "--output", str(mesh)]
    if g.get("smoke_substrate_grading") is not None:
        command += ["--sub-grading", str(g["smoke_substrate_grading"])]
    return command


def mesh_nodes(mesh: Path) -> list[tuple[float, float, float]]:
    nodes = []
    with mesh.open(encoding="utf-8") as stream:
        in_nodes = False
        for line in stream:
            if line.startswith("*"):
                in_nodes = line.upper().startswith("*NODE")
                continue
            if in_nodes and line.strip():
                _, x, y, z = line.split(",")[:4]
                nodes.append((float(x), float(y), float(z)))
    return nodes


def mesh_z_levels(mesh: Path) -> list[float]:
    return sorted({round(n[2], 12) for n in mesh_nodes(mesh)})


def check_mesh_against_schedule(mesh: Path, ledger: dict, cfg: dict) -> dict:
    nodes = mesh_nodes(mesh)
    levels = sorted({round(n[2], 12) for n in nodes})
    sub_z = float(cfg["geometry"]["smoke_substrate_z_m"])
    missing = []
    for event in ledger["activation_events"]:
        for key in ("z_bottom_m", "z_top_m"):
            if not any(math.isclose(event[key], z, rel_tol=0.0, abs_tol=1e-9) for z in levels):
                missing.append({key: event[key], "slab": event["slab"]})
    part_top = sub_z + float(cfg["geometry"]["smoke_part_z_m"])
    # part footprint in MESH coordinates = nodes strictly above the substrate top
    part_nodes = [n for n in nodes if n[2] > sub_z + 1e-9]
    if not part_nodes:
        raise SystemExit("mesh has no nodes above the substrate: no part column")
    xs = [n[0] for n in part_nodes]
    ys = [n[1] for n in part_nodes]
    part_bounds = [[min(xs), max(xs)], [min(ys), max(ys)]]
    expected = ledger["part_bounds_xy_m"]
    footprint_ok = all(math.isclose(b[i], expected[i], rel_tol=0.0, abs_tol=1e-9)
                       for b in part_bounds for i in (0, 1))
    lo, hi = ledger["exposure_bounds_m"]
    exposure_inside = (part_bounds[0][0] - 1e-9 <= lo and hi <= part_bounds[0][1] + 1e-9
                       and part_bounds[1][0] - 1e-9 <= ledger["track_cross_min_m"]
                       and ledger["track_cross_max_m"] <= part_bounds[1][1] + 1e-9)
    ok = (not missing
          and any(math.isclose(sub_z, z, rel_tol=0.0, abs_tol=1e-9) for z in levels)
          and math.isclose(levels[-1], part_top, rel_tol=0.0, abs_tol=1e-9)
          and footprint_ok and exposure_inside)
    return {"slab_boundaries_on_mesh_levels": not missing, "missing": missing,
            "mesh_z_levels_m": levels, "part_base_z_m": sub_z, "part_top_z_m": part_top,
            "mesh_part_bounds_xy_m": part_bounds, "schedule_part_bounds_xy_m": expected,
            "footprint_matches_schedule": bool(footprint_ok),
            "exposure_inside_part_footprint": bool(exposure_inside),
            "ok": bool(ok)}


def relocate_material_config(cfg: dict, repo: Path, output: Path) -> tuple[Path, dict]:
    """Copy the adopted V2 material config with its box-159 absolute table paths
    re-rooted onto ``repo`` (values untouched); fail closed on missing tables."""
    runner = cfg["runner"]
    src = repo / runner["material_config"]
    prefix = runner["material_config_relocate_prefix"]
    material = json.loads(src.read_text(encoding="utf-8"))
    relocated = {}
    for key, value in material.items():
        if isinstance(value, str) and value.startswith(prefix):
            new_value = str(repo / value[len(prefix):])
            if not Path(new_value).is_file():
                raise SystemExit(f"material table missing after relocation: {new_value}")
            relocated[key] = {"from": value, "to": new_value}
            material[key] = new_value
    if not math.isclose(float(material.get("absorptivity", -1.0)), float(runner["absorptivity"]),
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("runner.absorptivity must equal the material config absorptivity")
    dst = output / "material_config.json"
    dst.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst, {"source": str(src), "source_sha256": sha256(src), "relocated": relocated}


def runner_contract(cfg: dict, *, mesh: Path, path: Path, material: Path, ledger: dict) -> dict:
    g, layers, tm, runner = cfg["geometry"], cfg["layer_schedule"], cfg["thermal_mechanical"], cfg["runner"]
    cons, mech, out, lin = runner["consolidation"], runner["mechanics"], runner["output"], runner["linear_solver"]
    sub_xy = float(g["smoke_substrate_xy_m"])
    sub_z = float(g["smoke_substrate_z_m"])
    flash = ledger.get("flash")
    beam_radius = float(flash["beam_radius_m"]) if flash else float(runner["beam_radius_m"])
    laser_power = float(flash["commanded_power_W"]) if flash else float(cfg["scan"]["power_W"])
    argv = [
        "--config", str(material), "--inp", str(mesh),
        "--path-file", str(path), "--path-length-scale", "1.0",
        "--build-axis", "z", "--base-side", "min",
        "--layer-thickness", f"{float(layers['activation_slab_thickness_m']):.12g}",
        "--layers", str(ledger["activation_slabs"]),
        "--support-thickness", f"{sub_z:.12g}",
        "--layer-activation-mode", "layer_on_scan",
        "--layer-activation-geometry", runner["layer_activation_geometry"],
        "--future-layer-mode", "void", "--active-window-below-layers", "0",
        "--inactive-mass-factor", "1.0", "--powder-mode", "powder",
        "--surface-selection", "exterior", "--boundary-tol", "1.0e-6",
        "--quadrature-order", "2", "--thermal-mass-lumping",
        "--source-model", "legacy",
        "--beam-radius", f"{beam_radius:.12g}",
        "--source-depth", f"{float(runner['source_depth_m']):.12g}",
        "--source-depth-cutoff", f"{float(runner['source_depth_cutoff_m']):.12g}",
        "--source-cutoff-renormalize" if runner["source_cutoff_renormalize"] else "--no-source-cutoff-renormalize",
        "--laser-power", f"{laser_power:.12g}",
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
    if tm["release_after_cooling"]:
        argv += ["--release-after-cooling", "--release-anchor-mode", "rigid_body",
                 "--release-cut-box", "0", f"{sub_xy:.12g}", "0", f"{sub_xy:.12g}",
                 "0", f"{sub_z:.12g}"]
    return {
        "schema": CONTRACT_SCHEMA,
        "argv": argv,
        "python_bin": lin.get("python_bin"),
        "platform": lin["platform"],
        "io_flags_added_by_launcher": ["--output-dir", "--profile-json", "--profile-label"],
        "deposition": {
            "mode": ledger["deposition_mode"],
            "flash": flash,
            "note": ("reading A (D-V2-11): per-physical-layer uniform flash for the real scan time; "
                     "--laser-power is the COMMANDED power P/capture, the ledger's laser_absorbed_nominal_j "
                     "therefore reads Ac x P/capture x dt and its capture fraction must come out ~ "
                     f"{flash['capture_fraction_analytic']:.5f}; deposited energy per layer is expected to equal "
                     f"Ac x {flash['physical_energy_per_layer_J']:.3f} J") if flash else
                    "serpentine per-track scan (one row per cell of travel)",
        },
        "activation": {
            "mode": f"layer_on_scan/{runner['layer_activation_geometry']}",
            "geometry_note": runner.get("layer_activation_note"),
            "slab_thickness_m": float(layers["activation_slab_thickness_m"]),
            "slabs": ledger["activation_slabs"],
            "trigger": "first laser-on scan row carrying layer == k activates slab k "
                       "(cells whose centroid layer id ceil(z_c/lt) <= k become printed)",
            "events": ledger["activation_events"],
            "consolidation": "solidus = liquidus = sentinel above any reachable temperature: activated cells are "
                             "solid immediately and never re-melt; T_ref = stress_relaxation_temperature once at "
                             "activation; no lifecycle reference-reset events, so mechanics runs on the fixed cadence",
        },
        "temperature_mapping": {
            "coupling": "staggered in one time loop: the mechanics step at global_step % mechanics_every == 0 "
                        "(and the final step) consumes the thermal field of that step; release solve after cooldown",
            "stress_free_reference_K": float(tm["stress_free_reference_K"]),
            "activation_reset_temperature_K": float(cons["activation_reset_temperature_K"]),
            "mechanics_temperature_floor_K": float(mech["temperature_floor_K"]),
            "mechanics_every_steps": int(mech["every_steps"]),
        },
        "time": {
            "path_rows": ledger["path_rows"], "cooling_steps": ledger["cooling_steps"],
            "expected_runner_steps": ledger["expected_runner_steps"],
            "build_clock_s": ledger["build_clock_s"],
            "recoat": "explicit geometric sub-step rows; --recoat-time 0 so the runner inserts none",
        },
        "energy": {
            "nominal_laser_energy_J": ledger["nominal_laser_energy_J"],
            "absorbed_laser_energy_nominal_J": ledger["absorbed_laser_energy_nominal_J"],
            "deposition_band": (f"legacy source, depth cut at {float(runner['source_depth_cutoff_m'])*1e6:.0f} um "
                                f"(< slab), renormalised; d = {float(runner['source_depth_m'])*1e6:.3f} um")
                               if runner["source_cutoff_renormalize"] else "legacy source, half-space",
            "vertex_rule_band_integral": vertex_rule_band_integral(
                float(layers["activation_slab_thickness_m"]), float(runner["source_depth_m"]),
                float(runner["source_depth_cutoff_m"]), bool(runner["source_cutoff_renormalize"])),
            "vertex_rule_note": runner.get("heat_source_note"),
            "capture_fraction": "measured by the stage-2 capture trial (check_cube_smoke.py)",
        },
        "release": {
            "mode": "rigid_body anchors on printed nodes",
            "cut_box_m": [0.0, sub_xy, 0.0, sub_xy, 0.0, sub_z] if tm["release_after_cooling"] else None,
            "cut_semantics": "cells with centroid inside the box lose stiffness and locked-in stress "
                             "(stepper release_cut_box); the substrate is a --support-thickness fixture band",
        },
    }


def build_preflight(config_path: Path, repo: Path, output: Path) -> dict:
    cfg = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / "v2_cube_smoke_c3d8.inp"
    path = output / "v2_cube_smoke_path.csv"
    ledger_path = output / "v2_cube_smoke_ledger.json"
    contract_path = output / "runner_contract.json"
    command = mesh_command(cfg, repo, mesh)
    subprocess.run(command, check=True)
    rows, ledger = generate_schedule(cfg)
    write_schedule(path, rows)
    mesh_check = check_mesh_against_schedule(mesh, ledger, cfg)
    if not mesh_check["ok"]:
        raise SystemExit(f"mesh z-levels do not match the slab schedule: {mesh_check}")
    material, material_info = relocate_material_config(cfg, repo, output)
    contract = runner_contract(cfg, mesh=mesh, path=path, material=material, ledger=ledger)
    ledger.update({
        "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "mesh": str(mesh.resolve()), "mesh_sha256": sha256(mesh),
        "path": str(path.resolve()), "path_sha256": sha256(path),
        "material_config": str(material.resolve()), "material_config_sha256": sha256(material),
        "material_config_provenance": material_info,
        "mesh_command": command,
        "mesh_check": mesh_check,
        "runner_contract": str(contract_path.resolve()),
    })
    contract["inputs"] = {
        "config_sha256": ledger["config_sha256"], "mesh_sha256": ledger["mesh_sha256"],
        "path_sha256": ledger["path_sha256"], "material_config_sha256": ledger["material_config_sha256"],
    }
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ledger["runner_contract_sha256"] = sha256(contract_path)
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
        "footprint_m", "physical_layers", "activation_slabs", "tracks_per_physical_layer",
        "path_rows", "recoat_time_s", "total_time_s", "build_clock_s",
        "expected_runner_steps", "nominal_laser_energy_J",
        "absorbed_laser_energy_nominal_J")}, indent=2))
    print("activation events:")
    for event in ledger["activation_events"]:
        print(f"  slab {event['slab']}: physical layer {event['physical_layer']} "
              f"row {event['row_index']} t={event['time_s']:.6f} s "
              f"z=[{event['z_bottom_m']*1e3:.2f}, {event['z_top_m']*1e3:.2f}] mm")


if __name__ == "__main__":
    main()
