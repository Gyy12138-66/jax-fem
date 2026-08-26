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
    assert ledger["solver_compatible"] is False
    assert ledger["physical_layers"] == 25
    assert ledger["activation_slabs"] == 5
    assert ledger["physical_layers_per_slab"] == 5
    assert ledger["recoat_rows"] == 24
    assert ledger["recoat_time_s"] == pytest.approx(120.0)
    assert ledger["angles_deg"][:6] == [0.0, 90.0, 0.0, 90.0, 0.0, 90.0]
    assert ledger["tracks_per_physical_layer"] == 82
    assert ledger["track_cross_min_m"] >= ledger["exposure_bounds_m"][0]
    assert ledger["track_cross_max_m"] <= ledger["exposure_bounds_m"][1]
    assert {row["layer"] for row in rows if row["physical_layer"] <= 5} == {1}
    assert {row["layer"] for row in rows if 6 <= row["physical_layer"] <= 10} == {2}


def test_schedule_is_strictly_increasing_and_energy_is_analytic():
    cfg = MODULE.load_config(CONFIG)
    rows, ledger = MODULE.generate_schedule(cfg)
    times = [row["time"] for row in rows]
    assert all(right > left for left, right in zip(times, times[1:]))
    side = cfg["geometry"]["part_xy_m"] - 2 * cfg["scan"]["margin_m"]
    expected = (cfg["scan"]["power_W"] * side / cfg["scan"]["speed_m_s"]
                * ledger["tracks_per_physical_layer"] * ledger["physical_layers"])
    assert ledger["nominal_laser_energy_J"] == pytest.approx(expected)
    assert ledger["total_time_s"] == pytest.approx(
        ledger["scan_time_s"] + ledger["jump_time_s"] + ledger["recoat_time_s"])


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


def test_build_preflight_writes_mesh_path_and_fingerprinted_ledger(tmp_path):
    ledger = MODULE.build_preflight(CONFIG, ROOT, tmp_path)
    mesh = tmp_path / "v2_cube_smoke_c3d8.inp"
    path = tmp_path / "v2_cube_smoke_path.csv"
    saved = json.loads((tmp_path / "v2_cube_smoke_ledger.json").read_text(encoding="utf-8"))
    assert mesh.exists() and path.exists()
    assert len(saved["mesh_sha256"]) == 64
    assert len(saved["path_sha256"]) == 64
    assert saved["complete"] is True
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["mode"] for row in rows} == {"scan", "jump", "recoat"}
    assert rows[0]["physical_layer"] == "1"
    assert rows[-1]["physical_layer"] == "25"
    assert ledger["activation_slabs"] == 5
