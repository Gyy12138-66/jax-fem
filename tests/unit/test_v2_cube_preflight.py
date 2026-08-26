import csv
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "cases/AM-Benchmark/verification/v2-cube-rs/model/make_v2_cube_preflight.py"
CONFIG = ROOT / "cases/AM-Benchmark/verification/v2-cube-rs/inputs/cube-stress-smoke.json"
SPEC = importlib.util.spec_from_file_location("make_v2_cube_preflight", MODEL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_smoke_config_freezes_25_real_layers_into_5_slabs():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    assert ledger["complete"] is True
    assert ledger["solver_started"] is False
    assert ledger["solver_compatible"] is True
    assert ledger["physical_layers"] == 25
    assert ledger["activation_slabs"] == 5
    assert ledger["physical_layers_per_slab"] == 5
    assert ledger["recoat_rows"] == 24
    assert ledger["recoat_substep_rows"] == 240
    assert ledger["recoat_time_s"] == pytest.approx(120.0)
    assert ledger["angles_deg"][:6] == [0.0, 90.0, 0.0, 90.0, 0.0, 90.0]
    # smoke footprint 4 mm: (4 - 0.2) / 0.12 -> 32 tracks
    assert ledger["footprint_m"] == pytest.approx(0.004)
    assert ledger["tracks_per_physical_layer"] == 32
    assert ledger["track_cross_min_m"] >= ledger["exposure_bounds_m"][0]
    assert ledger["track_cross_max_m"] <= ledger["exposure_bounds_m"][1]
    assert {row["layer"] for row in rows if row["physical_layer"] <= 5} == {1}
    assert {row["layer"] for row in rows if 6 <= row["physical_layer"] <= 10} == {2}


def test_production_footprint_keeps_82_tracks():
    cfg = MODULE.load_config(CONFIG)
    _, ledger = MODULE.generate_schedule(cfg, footprint_m=cfg["geometry"]["part_xy_m"],
                                         substrate_xy_m=cfg["geometry"]["substrate_xy_m"])
    assert ledger["tracks_per_physical_layer"] == 82
    assert ledger["part_origin_xy_m"] == pytest.approx(0.010)


def test_schedule_is_in_mesh_coordinates_inside_the_centred_part_column():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    # 4 mm part centred on the 8 mm substrate -> part column at [2, 6] mm
    assert ledger["part_origin_xy_m"] == pytest.approx(0.002)
    assert ledger["part_bounds_xy_m"] == pytest.approx([0.002, 0.006])
    assert ledger["exposure_bounds_m"] == pytest.approx([0.0021, 0.0059])
    on = [r for r in rows if r["laser_on"] == 1]
    assert min(r["x"] for r in on) >= 0.0021 - 1e-12 and max(r["x"] for r in on) <= 0.0059 + 1e-12
    assert min(r["y"] for r in on) >= 0.0021 - 1e-12 and max(r["y"] for r in on) <= 0.0059 + 1e-12


def test_schedule_is_strictly_increasing_and_energy_is_analytic():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    times = [row["time"] for row in rows]
    assert all(right > left for left, right in zip(times, times[1:]))
    side = cfg["geometry"]["smoke_part_xy_m"] - 2 * cfg["scan"]["margin_m"]
    expected = (cfg["scan"]["power_W"] * side / cfg["scan"]["speed_m_s"]
                * ledger["tracks_per_physical_layer"] * ledger["physical_layers"])
    assert ledger["nominal_laser_energy_J"] == pytest.approx(expected)
    assert ledger["absorbed_laser_energy_nominal_J"] == pytest.approx(expected * 0.62)
    assert ledger["total_time_s"] == pytest.approx(
        ledger["scan_time_s"] + ledger["jump_time_s"] + ledger["recoat_time_s"])
    assert ledger["build_clock_s"] == pytest.approx(ledger["total_time_s"] + 600.0)
    assert ledger["expected_runner_steps"] == len(rows) + 60
    assert sum(float(v) for v in ledger["nominal_energy_per_slab_J"].values()) == pytest.approx(expected)


def test_activation_events_are_first_scan_rows_of_each_slab():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    events = ledger["activation_events"]
    assert [e["slab"] for e in events] == [1, 2, 3, 4, 5]
    assert [e["physical_layer"] for e in events] == [1, 6, 11, 16, 21]
    sub_z = cfg["geometry"]["smoke_substrate_z_m"]
    for k, event in enumerate(events, start=1):
        row = rows[event["row_index"]]
        assert row["laser_on"] == 1 and row["mode"] == "scan" and row["layer"] == k
        assert event["global_step"] == event["row_index"]
        assert event["z_top_m"] == pytest.approx(sub_z + k * 0.0002)
        assert all(r["laser_on"] == 0 or r["layer"] < k for r in rows[:event["row_index"]])
    assert events[0]["row_index"] == 0  # very first row is a scan row of slab 1


def test_slab_top_deposition_and_recoat_substeps():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    sub_z = cfg["geometry"]["smoke_substrate_z_m"]
    for row in rows:
        slab = (row["physical_layer"] - 1) // 5 + 1
        assert row["z"] == pytest.approx(sub_z + slab * 0.0002)
        assert row["front_coord"] == row["z"]
        assert row["physical_z"] == pytest.approx(sub_z + row["physical_layer"] * 0.00004)
    durations = ledger["recoat_substep_durations_s"]
    assert len(durations) == 10
    assert sum(durations) == pytest.approx(5.0)
    assert durations[0] < durations[-1]
    assert durations[1] / durations[0] == pytest.approx(1.5)
    assert ledger["first_row_dt_s"] == pytest.approx(0.0002 / 0.65)


def test_bad_layer_lumping_fails_closed(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["layer_schedule"]["physical_layers_per_slab"] = 4
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="activation slab"):
        MODULE.load_config(bad)


def test_non_integer_and_non_positive_schedule_values_fail_closed(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["layer_schedule"]["physical_layers_per_slab"] = 5.5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="positive integer"):
        MODULE.load_config(bad)

    raw["layer_schedule"]["physical_layers_per_slab"] = 5
    raw["layer_schedule"]["recoat_time_s"] = 0
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="finite and positive"):
        MODULE.load_config(bad)

    raw["layer_schedule"]["recoat_time_s"] = 5.0
    raw["geometry"]["production_part_z_m"] = True
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="production part height|finite and positive"):
        MODULE.load_config(bad)


@pytest.mark.parametrize("key", [
    "margin_m", "start_angle_deg", "rotation_per_physical_layer_deg",
])
def test_numeric_scan_fields_reject_boolean_values(tmp_path, key):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["scan"][key] = False
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"scan.{key}"):
        MODULE.load_config(bad)


def test_production_schedule_must_match_height_and_slab_count(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["layer_schedule"]["production_physical_layers"] = 249
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="multiple of layers per slab"):
        MODULE.load_config(bad)

    raw["layer_schedule"]["production_physical_layers"] = 250
    raw["geometry"]["production_part_z_m"] = 0.009
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="production part height"):
        MODULE.load_config(bad)


def test_vertex_rule_band_integral_is_unity_for_the_contract_and_1_47_for_the_naive_pair():
    cfg = MODULE.load_config(CONFIG)
    r = cfg["runner"]
    assert MODULE.vertex_rule_band_integral(0.0002, r["source_depth_m"], r["source_depth_cutoff_m"], True) == pytest.approx(1.0, abs=1e-4)
    # the first attempt: d = 100 um, cutoff = 200 um -> measured 1.447-1.47 on 2026-08-26
    assert MODULE.vertex_rule_band_integral(0.0002, 1.0e-4, 2.0e-4, True) == pytest.approx(1.4696, abs=1e-3)


def test_runner_contract_guards_fail_closed(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["source_depth_cutoff_m"] = 0.0002
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="strictly inside one slab"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["source_depth_m"] = 0.0001
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="vertex rule"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["consolidation"]["solidus_K"] = 1563.0
    raw["runner"]["consolidation"]["liquidus_K"] = 1563.0
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="sentinel"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["consolidation"]["stress_relaxation_temperature_K"] = 1000.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="stress-free reference"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["source_depth_cutoff_m"] = 0.0
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="renormalize"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["source_depth_cutoff_m"] = 0.0
    raw["runner"]["source_cutoff_renormalize"] = False
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="strictly inside one slab"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["geometry"]["smoke_substrate_xy_m"] = 0.0081
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="integer number of cells"):
        MODULE.load_config(bad)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["runner"]["layer_activation_geometry"] = "intersection"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="double-activates"):
        MODULE.load_config(bad)


def test_build_preflight_writes_mesh_path_contract_and_fingerprinted_ledger(tmp_path):
    ledger = MODULE.build_preflight(CONFIG, ROOT, tmp_path)
    mesh = tmp_path / "v2_cube_smoke_c3d8.inp"
    path = tmp_path / "v2_cube_smoke_path.csv"
    contract_path = tmp_path / "runner_contract.json"
    material = tmp_path / "material_config.json"
    saved = json.loads((tmp_path / "v2_cube_smoke_ledger.json").read_text(encoding="utf-8"))
    assert mesh.exists() and path.exists() and contract_path.exists() and material.exists()
    assert len(saved["mesh_sha256"]) == 64
    assert len(saved["path_sha256"]) == 64
    assert saved["complete"] is True
    assert saved["mesh_check"]["ok"] is True
    assert saved["mesh_check"]["footprint_matches_schedule"] is True
    assert saved["mesh_check"]["exposure_inside_part_footprint"] is True
    bounds = saved["mesh_check"]["mesh_part_bounds_xy_m"]
    assert bounds[0] == pytest.approx([0.002, 0.006]) and bounds[1] == pytest.approx([0.002, 0.006])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["mode"] for row in rows} == {"scan", "jump", "recoat"}
    assert rows[0]["physical_layer"] == "1"
    assert rows[-1]["physical_layer"] == "25"
    assert ledger["activation_slabs"] == 5
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    argv = contract["argv"]
    assert "--recoat-time" in argv and argv[argv.index("--recoat-time") + 1] == "0"
    assert argv[argv.index("--layers") + 1] == "5"
    assert argv[argv.index("--layer-activation-geometry") + 1] == "centroid"
    assert argv[argv.index("--layer-thickness") + 1] == "0.0002"
    assert argv[argv.index("--support-thickness") + 1] == "0.0016"
    assert argv[argv.index("--liquidus-temperature") + 1] == argv[argv.index("--solidus-temperature") + 1]
    assert float(argv[argv.index("--solidus-temperature") + 1]) >= 5000.0
    assert argv[argv.index("--stress-relaxation-temperature") + 1] == "1273.15"
    assert "--release-cut-box" in argv
    assert float(argv[argv.index("--dt") + 1]) == pytest.approx(0.0002 / 0.65)
    assert contract["inputs"]["path_sha256"] == saved["path_sha256"]
    relocated = json.loads(material.read_text(encoding="utf-8"))
    for key in ("E_table", "alpha_table", "poisson_table", "flow_curve_table", "k_table_solid"):
        assert pathlib.Path(relocated[key]).is_file()
        assert str(ROOT) in relocated[key]
