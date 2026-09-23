"""采纳口径 + 4 个角点的起飞前体检(标准动作,Fable5 2026-08-05 保留)。"""
import csv
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/home/user/work/159/jax-fem")
from jax_fem_am.materials.material_validation import validate_material_inputs

M = Path("/home/user/work/159/jax-fem/cases/AM-Benchmark/verification"
         "/v2-cube-rs/model")

TARGETS = [("采纳口径", M / "v2_material_config.json")]
for tag in ("f0005_c07", "f0005_c09", "f002_c07", "f002_c09"):
    TARGETS.append((f"角点 {tag}", M / f"v2_material_config_fc_ecol_{tag}.json"))


class Table:
    def __init__(self, T, v):
        self.T = np.asarray(T); self.values = np.asarray(v)


class FlowCurve:
    def __init__(self, T, e, s):
        self.temperatures = np.asarray(T)
        self.plastic_strains = np.asarray(e)
        self.stresses = np.asarray(s)


def load_pair(p):
    T, v = [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            T.append(float(r["T"])); v.append(float(r["value"]))
    return Table(T, v)


def load_flow(p):
    g = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            g.setdefault(float(r["temperature_K"]), []).append(
                (float(r["equivalent_plastic_strain"]), float(r["flow_stress_Pa"])))
    Ts = sorted(g)
    eps = [e for e, _ in sorted(g[Ts[0]])]
    return FlowCurve(Ts, eps, [[s for _, s in sorted(g[T])] for T in Ts])


def args():
    return SimpleNamespace(
        mechanics_model="j2_plastic", young=1.71e11, alpha=1.2e-5, poisson=0.3,
        yield_saturation_stress=None, rho=8453.0, rho_solid=None, rho_liquid=None,
        rho_powder=5071.8, cp=500.0, cp_solid=None, cp_liquid=None, cp_powder=500.0,
        conductivity=20.0, conductivity_solid=None, conductivity_liquid=None,
        conductivity_powder=1.0)


rc = 0
for label, p in TARGETS:
    cfg = json.loads(p.read_text(encoding="utf-8"))
    e = load_pair(cfg["E_table"])
    mono = all(b > a for a, b in zip(e.T, e.T[1:]))
    tables = {"E": e, "alpha": load_pair(cfg["alpha_table"]),
              "poisson": load_pair(cfg["poisson_table"]),
              "yield": None, "hardening": None,
              "flow_curve": load_flow(cfg["flow_curve_table"]),
              "k_solid": None, "cp_solid": None, "k_powder": None,
              "cp_powder": None, "k_liquid": None, "cp_liquid": None}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_material_inputs(args(), tables)
        w = [str(x.message)[:100] for x in caught
             if issubclass(x.category, RuntimeWarning)]
        status = "OK" if not w else "OK(有警告)"
    except Exception as exc:
        status, w, rc = f"REJECTED: {exc}", [], 1
    print(f"{label:>14}  E单调={'是' if mono else '否'}  "
          f"E({Path(cfg['E_table']).name}) 行数={len(e.T)}  "
          f"flow={Path(cfg['flow_curve_table']).name}  -> {status}")
    for msg in w:
        print(f"                 ! {msg}")
print("\n全部通过" if rc == 0 else "\n有配置会被拒")
sys.exit(rc)
