#!/usr/bin/env python3
"""Three-way in-depth residual-stress comparison: ours vs Balbaa ABAQUS vs XRD (stage 5).

Inputs: in_depth_rs.json from extract_in_depth_rs.py (our released cube) and
inputs/balbaa-fig43-44-xrd-in-depth-rs.json (digitized Fig 43/44).
Outputs: a JSON with per-depth residuals and RMSE (ours vs XRD, Balbaa's two
ABAQUS curves vs XRD as the yardstick -- same framing as the thermal gate:
no pass line, his own gap is the ruler), a CSV table and a PNG plot.

The XRD depths (100, 200, 300, 450, 600, 800, 1000 um) are compared against
our profile linearly interpolated between cell-row centres (0.1, 0.3, ... mm);
both our sigma_xx and sigma_yy are scored because the flash reading has no scan
direction and the paper does not say which component Fig 43/44 plots.

    compare_in_depth_rs.py --ours <run>/in_depth_rs.json --figure fig44 --output-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIGITIZED = HERE.parent / "inputs" / "balbaa-fig43-44-xrd-in-depth-rs.json"


def series_xy(series):
    pts = [p for p in series if p.get("stress_MPa") is not None]
    return np.array([p["depth_um"] for p in pts]), np.array([p["stress_MPa"] for p in pts]), pts


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--figure", choices=("fig43", "fig44"), default="fig44")
    ap.add_argument("--digitized", type=Path, default=DIGITIZED)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--label", default="jax-fem (flash reading A', GPU+PARDISO)")
    args = ap.parse_args()
    ours = json.loads(args.ours.read_text(encoding="utf-8"))
    dig = json.loads(args.digitized.read_text(encoding="utf-8"))
    fig = dig["figures"][args.figure]
    out_dir = args.output_dir or args.ours.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = ours["profile_cell_rows"]
    d_ours = np.array([r["depth_m"] for r in rows]) * 1e6
    sxx = np.array([r["sxx_MPa"] for r in rows]); syy = np.array([r["syy_MPa"] for r in rows])
    x_xrd, y_xrd, pts_xrd = series_xy(fig["series"]["experimental_xrd"])
    err_p = np.array([p.get("err_plus_MPa", 0.0) for p in pts_xrd]); err_m = np.array([p.get("err_minus_MPa", 0.0) for p in pts_xrd])
    x_lit, y_lit, _ = series_xy(fig["series"]["predicted_jc_literature"])
    x_mod, y_mod, _ = series_xy(fig["series"]["predicted_modified_jc"])

    ours_xx_at = np.interp(x_xrd, d_ours, sxx); ours_yy_at = np.interp(x_xrd, d_ours, syy)
    lit_at = np.interp(x_xrd, x_lit, y_lit); mod_at = np.interp(x_xrd, x_mod, y_mod)
    table = []
    for i, d in enumerate(x_xrd):
        table.append({"depth_um": float(d), "xrd_MPa": float(y_xrd[i]), "xrd_err_plus": float(err_p[i]), "xrd_err_minus": float(err_m[i]),
                      "ours_sxx_MPa": float(ours_xx_at[i]), "ours_syy_MPa": float(ours_yy_at[i]),
                      "balbaa_jc_literature_MPa": float(lit_at[i]), "balbaa_modified_jc_MPa": float(mod_at[i])})
    metrics = {
        "ours_sxx_vs_xrd": {"rmse_MPa": rmse(ours_xx_at, y_xrd), "mean_signed_MPa": float(np.mean(ours_xx_at - y_xrd)),
                            "max_abs_MPa": float(np.max(np.abs(ours_xx_at - y_xrd)))},
        "ours_syy_vs_xrd": {"rmse_MPa": rmse(ours_yy_at, y_xrd), "mean_signed_MPa": float(np.mean(ours_yy_at - y_xrd)),
                            "max_abs_MPa": float(np.max(np.abs(ours_yy_at - y_xrd)))},
        "balbaa_jc_literature_vs_xrd": {"rmse_MPa": rmse(lit_at, y_xrd), "mean_signed_MPa": float(np.mean(lit_at - y_xrd))},
        "balbaa_modified_jc_vs_xrd": {"rmse_MPa": rmse(mod_at, y_xrd), "mean_signed_MPa": float(np.mean(mod_at - y_xrd))},
        "ours_sxx_vs_balbaa_modified_jc": {"rmse_MPa": rmse(ours_xx_at, mod_at)},
        "xrd_read_off_MPa": fig["read_off_uncertainty"]["stress_MPa"],
        "yardstick": "Balbaa's own ABAQUS-vs-XRD RMSE (both J-C sets); the paper calls this 'same trend'",
    }
    out = {"schema": "v2.cube-in-depth-rs-compare/1", "figure": args.figure, "condition": fig["condition"],
           "ours": str(args.ours), "ours_frame": ours.get("frame"), "table": table, "metrics": metrics,
           "component_note": ours.get("in_plane_anisotropy_note")}
    (out_dir / f"in_depth_rs_compare_{args.figure}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"in_depth_rs_compare_{args.figure}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    print(f"{'depth um':>9s} {'XRD':>7s} {'ours sxx':>9s} {'ours syy':>9s} {'Balbaa JC-lit':>13s} {'Balbaa mod-JC':>13s}")
    for t in table:
        print(f"{t['depth_um']:9.0f} {t['xrd_MPa']:7.0f} {t['ours_sxx_MPa']:9.0f} {t['ours_syy_MPa']:9.0f} {t['balbaa_jc_literature_MPa']:13.0f} {t['balbaa_modified_jc_MPa']:13.0f}")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k:32s} RMSE {v['rmse_MPa']:6.1f} MPa" + (f"  mean signed {v['mean_signed_MPa']:+6.1f}" if 'mean_signed_MPa' in v else ""))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(x_xrd, y_xrd, yerr=[err_m, err_p], fmt="ko", capsize=3, label="XRD (Balbaa 2022, digitized)")
        ax.plot(x_lit, y_lit, "r^-", label="Balbaa ABAQUS, literature J-C")
        ax.plot(x_mod, y_mod, "s-", color="tab:blue", label="Balbaa ABAQUS, adjusted J-C")
        ax.plot(d_ours, sxx, "D-", color="tab:green", label=f"{args.label}: sigma_xx")
        ax.plot(d_ours, syy, "v--", color="tab:olive", label=f"{args.label}: sigma_yy")
        ax.set_xlabel("depth below top face (um)"); ax.set_ylabel("residual stress (MPa)")
        ax.set_title(f"IN625 cube, {fig['condition']}: in-depth residual stress at the centre (2 mm spot)")
        ax.set_xlim(0, 1100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig_.tight_layout(); fig_.savefig(out_dir / f"in_depth_rs_compare_{args.figure}.png", dpi=160)
        print("plot", out_dir / f"in_depth_rs_compare_{args.figure}.png")
    except Exception as exc:  # plotting is a convenience, never a gate
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
