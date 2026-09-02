#!/usr/bin/env python3
"""Compare solver/platform arms of the single-layer cube run (v_cube_solver_compare.sh).

Speed from profile.json (wall, per-step, stage seconds, Newton wall, fallbacks),
health from run.log / ledger, accuracy as field differences of the LAST frame
against the baseline arm: T (nodal), u (nodal), von Mises / eq. plastic strain
(cell). Writes solver_compare.json and prints a table. No thresholds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import meshio
import numpy as np


def cell_field(mesh, name):
    d = mesh.cell_data_dict.get(name)
    return None if d is None else np.asarray(list(d.values())[0]).reshape(-1)


def quad_average(mesh, pattern):
    parts = [cell_field(mesh, pattern.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def load_arm(root: Path, name: str) -> dict:
    d = root / name
    out = {"arm": name, "present": d.is_dir()}
    if not out["present"]:
        return out
    prof_p = d / "profile.json"
    if prof_p.is_file():
        prof = json.loads(prof_p.read_text(encoding="utf-8"))
        ss = prof.get("stage_seconds", {})
        meta = prof.get("meta", {})
        out.update({
            "wall_s": prof.get("wall_seconds"), "steps": prof.get("steps"),
            "s_per_step": (prof["wall_seconds"] / prof["steps"]) if prof.get("steps") else None,
            "assembly_s": ss.get("assembly"), "solver_s": ss.get("solver"),
            "newton_wall_s": meta.get("newton_wall_seconds"),
            "nonlinear_solve_calls": prof.get("stage_calls", {}).get("nonlinear_solve"),
            "solver_calls": prof.get("stage_calls", {}).get("solver"),
            "solver_fallbacks": meta.get("solver_fallbacks", 0),
            "last_solver_fallback": meta.get("last_solver_fallback"),
        })
    log_p = d / "run.log"
    if log_p.is_file():
        text = log_p.read_text(encoding="utf-8", errors="replace")
        out["newton_nonconvergence"] = len(re.findall(r"did not converge", text))
        out["fallback_warnings"] = len(re.findall(r"retrying this solve with SciPy spsolve", text))
        out["nan_mentions"] = len(re.findall(r"\bnan\b", text, flags=re.IGNORECASE))
        m = re.search(r"linear_solver_override\s*=\s*(.+)", text)
        out["linear_solver"] = m.group(1).strip()[:80] if m else None
        m = re.search(r"xla_platform\s*=\s*(\w+)", text)
        out["platform"] = m.group(1) if m else None
        last = re.findall(r"^global_step=(\d+).*T_max=([-\d.eE+]+) u_max=([-\d.eE+]+) vm_max=([-\d.eE+]+)", text, flags=re.M)
        if last:
            out["last_summary"] = {"step": int(last[-1][0]), "T_max": float(last[-1][1]),
                                   "u_max": float(last[-1][2]), "vm_max": float(last[-1][3])}
    summ = d / "thermal_energy_ledger_summary.json"
    if summ.is_file():
        s = json.loads(summ.read_text(encoding="utf-8"))
        out["ledger_complete"] = s.get("complete")
        out["max_relative_balance_error"] = s.get("maximum_relative_balance_error")
    vtus = sorted(glob.glob(str(d / "step_*.vtu")))
    out["last_vtu"] = os.path.basename(vtus[-1]) if vtus else None
    return out


def field_diffs(root: Path, base: str, arm: str) -> dict:
    vb = sorted(glob.glob(str(root / base / "step_*.vtu")))
    va = sorted(glob.glob(str(root / arm / "step_*.vtu")))
    if not vb or not va:
        return {"compared": False}
    mb, ma = meshio.read(vb[-1]), meshio.read(va[-1])
    if os.path.basename(vb[-1]) != os.path.basename(va[-1]):
        return {"compared": False, "reason": f"frames differ: {os.path.basename(vb[-1])} vs {os.path.basename(va[-1])}"}
    res = {"compared": True, "frame": os.path.basename(vb[-1])}
    Tb, Ta = np.asarray(mb.point_data["T"]), np.asarray(ma.point_data["T"])
    res["T_max_abs_diff_K"] = float(np.abs(Ta - Tb).max())
    res["T_rms_diff_K"] = float(np.sqrt(np.mean((Ta - Tb) ** 2)))
    ub, ua = np.asarray(mb.point_data["u"]), np.asarray(ma.point_data["u"])
    res["u_max_abs_diff_m"] = float(np.abs(ua - ub).max())
    res["u_max_abs_baseline_m"] = float(np.abs(ub).max())
    for name, pat in (("vm", "vm_quadQ"), ("sxx", "stress_quadQ_xx")):
        fb, fa = quad_average(mb, pat), quad_average(ma, pat)
        if fb is not None and fa is not None:
            res[f"{name}_max_abs_diff_MPa"] = float(np.abs(fa - fb).max() / 1e6)
            res[f"{name}_rms_diff_MPa"] = float(np.sqrt(np.mean((fa - fb) ** 2)) / 1e6)
            res[f"{name}_rms_baseline_MPa"] = float(np.sqrt(np.mean(fb ** 2)) / 1e6)
    eb, ea = cell_field(mb, "eq_plastic_strain"), cell_field(ma, "eq_plastic_strain")
    if eb is not None and ea is not None:
        res["eqp_max_abs_diff"] = float(np.abs(ea - eb).max())
        res["eqp_max_baseline"] = float(eb.max())
    sb, sa = cell_field(mb, "stress_free_temperature"), cell_field(ma, "stress_free_temperature")
    if sb is not None and sa is not None:
        res["stress_free_T_cells_differing"] = int(np.sum(np.abs(sa - sb) > 1e-3))
    res["bitwise_identical_T"] = bool(np.array_equal(Ta, Tb))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--baseline", default="cpu_pardiso")
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    arms = {name: load_arm(args.root, name) for name in args.arms}
    for name in args.arms:
        if name != args.baseline and arms[name].get("present"):
            arms[name]["vs_baseline"] = field_diffs(args.root, args.baseline, name)
    out = {"schema": "v2.cube-solver-compare/1", "baseline": args.baseline, "arms": arms}
    (args.root / "solver_compare.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    base = arms[args.baseline]
    print(f"{'arm':20s} {'platform':8s} {'wall s':>8s} {'s/step':>7s} {'assembly':>9s} {'solver':>8s} {'newton':>8s} {'fallb':>5s} {'nonconv':>7s} {'T diff K':>9s} {'u diff':>9s} {'vm rms MPa':>10s} {'eqp diff':>9s}")
    for name in args.arms:
        a = arms[name]
        if not a.get("present"):
            print(f"{name:20s} (missing)"); continue
        v = a.get("vs_baseline", {})
        def f(x, w, fmt="{:.3g}"):
            return (fmt.format(x) if isinstance(x, (int, float)) else "-").rjust(w)
        print(f"{name:20s} {str(a.get('platform')):8s} {f(a.get('wall_s'),8,'{:.0f}')} {f(a.get('s_per_step'),7,'{:.3f}')} "
              f"{f(a.get('assembly_s'),9,'{:.0f}')} {f(a.get('solver_s'),8,'{:.0f}')} {f(a.get('newton_wall_s'),8,'{:.0f}')} "
              f"{f(a.get('solver_fallbacks'),5,'{:d}')} {f(a.get('newton_nonconvergence'),7,'{:d}')} "
              f"{f(v.get('T_max_abs_diff_K'),9)} {f(v.get('u_max_abs_diff_m'),9)} {f(v.get('vm_rms_diff_MPa'),10)} {f(v.get('eqp_max_abs_diff'),9)}")
    if base.get("wall_s"):
        for name in args.arms:
            a = arms[name]
            if a.get("wall_s"):
                print(f"  speedup vs {args.baseline}: {name} = {base['wall_s'] / a['wall_s']:.2f}x")


if __name__ == "__main__":
    main()
