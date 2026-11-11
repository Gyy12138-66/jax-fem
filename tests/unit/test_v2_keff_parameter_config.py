import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
MODEL = ROOT / "cases" / "AM-Benchmark" / "verification" / "v2-cube-rs" / "model"
CONFIG = MODEL.parent / "inputs" / "keff-v2-220W-650mms.json"


def load_module():
    spec = importlib.util.spec_from_file_location("make_keff_table", MODEL / "make_keff_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keff_config_separates_nonkeff_measured_input():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["independent_inputs"]["process"]["power_W"] == 220.0
    assert config["nonkeff_derived_inputs"]["characteristic_half_width_L_m"] == 80e-6
    assert config["nonkeff_derived_inputs"]["measurement_protocol"]["source_run_complete_required"] is True
    assert config["derived_outputs"]["keff_W_mK"] == 87.00130152764835
    assert "Fig.14" in config["sensitivity_and_flags"]["zero_calibration"]


def test_keff_config_reproduces_frozen_derivation():
    module = load_module()
    config, defaults = module.load_parameter_config(CONFIG)
    props = module.load_table1()
    L = config["nonkeff_derived_inputs"]["characteristic_half_width_L_m"]
    qv = module.qv_peak_exponential(
        defaults["power"], defaults["absorptivity"],
        defaults["beam_radius"], defaults["opd"])
    result = module.keff_at(L, qv, defaults["speed"], props, defaults["rho_liquid"])
    assert result["keff_W_mK"] == pytest.approx(
        config["derived_outputs"]["keff_W_mK"], rel=1e-14)
