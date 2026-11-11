#!/usr/bin/env python3
"""Plot heating/cooling trends from paired V2 online observables.

The registered 10 ms adopted reading remains untouched.  A centred 50 ms trend
is added only inside contiguous valid runs, together with its slope and the
hot-cell denominator, so conditional-average spikes are not mistaken for bulk
heating.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_online_observables import (  # noqa: E402
    crop_rows_to_window, response_integrated_series,
)

C2K = 273.15


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty observations: {path}")
    return rows


def centred_contiguous_mean(values, width):
    """Centred mean requiring a complete odd window of finite values."""
    if width < 1 or width % 2 != 1:
        raise ValueError("trend width must be a positive odd number of bins")
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half:index + half + 1]
        if np.isfinite(window).all():
            result[index] = float(window.mean())
    return result


def fixed_bins(path, window_s, bin_ms, wavelengths):
    rows = crop_rows_to_window(load_rows(path), window_s)
    series = response_integrated_series(rows, bin_ms * 1.0e-3, wavelengths)
    by_index = {item["bin_index"]: item for item in series}
    scaled_start = window_s[0] * 1000.0 / bin_ms
    scaled_end = window_s[1] * 1000.0 / bin_ms
    first = int(round(scaled_start))
    last = int(round(scaled_end))
    if not (np.isclose(first, scaled_start, rtol=0.0, atol=1.0e-9)
            and np.isclose(last, scaled_end, rtol=0.0, atol=1.0e-9)):
        raise ValueError("window boundaries must align with fixed bin boundaries")
    output = []
    for index in range(first, last):
        item = by_index.get(index, {})
        output.append({
            "bin_index": index,
            "t_center_s": (index + 0.5) * bin_ms * 1.0e-3,
            "coverage_fraction": item.get("coverage_fraction", 0.0),
            "avg_K": item.get("avg_K"),
            "mean_n_hot": item.get("mean_n_hot"),
            "two_colour_K": item.get("two_colour_K"),
            "full_spot_avg_K": item.get("full_spot_avg_K"),
            "max_K": item.get("max_K"),
        })
    return output


def add_trend(series, trend_bins, bin_ms, min_mean_n_hot):
    raw = np.asarray([np.nan if item["avg_K"] is None else item["avg_K"] - C2K
                      for item in series])
    roi_raw = np.asarray([
        np.nan if item["full_spot_avg_K"] is None else item["full_spot_avg_K"] - C2K
        for item in series
    ])
    reliable = np.asarray([
        (np.isfinite(raw[index]) and item["coverage_fraction"] >= 1.0 - 1e-6
         and item["mean_n_hot"] is not None
         and item["mean_n_hot"] >= min_mean_n_hot)
        for index, item in enumerate(series)
    ])
    adopted_trend = centred_contiguous_mean(np.where(reliable, raw, np.nan), trend_bins)
    complete = np.asarray([
        item["coverage_fraction"] >= 1.0 - 1e-6 for item in series
    ])
    roi_trend = centred_contiguous_mean(np.where(complete, roi_raw, np.nan), trend_bins)
    slope = np.full(roi_trend.shape, np.nan)
    step_s = bin_ms * 1.0e-3
    for index in range(1, len(roi_trend) - 1):
        if np.isfinite(roi_trend[index - 1:index + 2]).all():
            slope[index] = (roi_trend[index + 1] - roi_trend[index - 1]) / (2.0 * step_s)
    for index, item in enumerate(series):
        item.update({
            "adopted_C": None if not np.isfinite(raw[index]) else float(raw[index]),
            "adopted_trend_C": (None if not np.isfinite(adopted_trend[index])
                                  else float(adopted_trend[index])),
            "roi_trend_C": None if not np.isfinite(roi_trend[index]) else float(roi_trend[index]),
            "roi_trend_rate_C_per_s": None if not np.isfinite(slope[index]) else float(slope[index]),
            "adopted_reliable": bool(reliable[index]),
        })
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-keff", type=Path, required=True)
    parser.add_argument("--keff", type=Path, required=True)
    parser.add_argument("--window", default="0.40,1.00")
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--trend-ms", type=float, default=50.0)
    parser.add_argument("--min-mean-n-hot", type=float, default=5.0,
                        help="visual reliability marker only; does not change readings")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    window_s = tuple(float(value) for value in args.window.split(","))
    trend_bins_float = args.trend_ms / args.bin_ms
    trend_bins = int(round(trend_bins_float))
    if not np.isclose(trend_bins, trend_bins_float) or trend_bins % 2 != 1:
        raise SystemExit("trend-ms/bin-ms must be an odd integer")
    wavelengths = (0.95e-6, 1.05e-6)
    arms = {
        "no-keff": add_trend(fixed_bins(args.no_keff, window_s, args.bin_ms, wavelengths),
                             trend_bins, args.bin_ms, args.min_mean_n_hot),
        "keff": add_trend(fixed_bins(args.keff, window_s, args.bin_ms, wavelengths),
                          trend_bins, args.bin_ms, args.min_mean_n_hot),
    }
    rows = []
    for index in range(len(next(iter(arms.values())))):
        row = {"t_center_s": arms["no-keff"][index]["t_center_s"]}
        for arm, series in arms.items():
            for key in ("coverage_fraction", "adopted_C", "adopted_trend_C",
                        "roi_trend_C", "roi_trend_rate_C_per_s", "mean_n_hot", "adopted_reliable",
                        "two_colour_K", "full_spot_avg_K", "max_K"):
                value = series[index][key]
                if key.endswith("_K") and value is not None:
                    value -= C2K
                row[f"{arm}_{key.replace('_K', '_C')}"] = value
        rows.append(row)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps({
        "window_s": window_s, "bin_ms": args.bin_ms,
        "trend_ms": args.trend_ms,
        "trend_semantics": "fixed-ROI centred mean over complete contiguous bins; adopted trend additionally requires reliable bins",
        "min_mean_n_hot_visual_marker": args.min_mean_n_hot,
        "note": "trend and reliability markers are diagnostic only; raw adopted is unchanged",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    time = np.asarray([row["t_center_s"] for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    colors = {"no-keff": "tab:orange", "keff": "tab:blue"}
    for arm in arms:
        raw = np.asarray([np.nan if row[f"{arm}_adopted_C"] is None else row[f"{arm}_adopted_C"] for row in rows])
        roi_raw = np.asarray([row[f"{arm}_full_spot_avg_C"] for row in rows])
        roi_trend = np.asarray([np.nan if row[f"{arm}_roi_trend_C"] is None else row[f"{arm}_roi_trend_C"] for row in rows])
        reliable = np.asarray([row[f"{arm}_adopted_reliable"] for row in rows])
        axes[0].plot(time, raw, "o-", ms=3, alpha=.45, color=colors[arm], label=f"{arm} adopted {args.bin_ms:g} ms")
        axes[0].plot(time[~reliable & np.isfinite(raw)], raw[~reliable & np.isfinite(raw)], "x", ms=6, color=colors[arm])
        axes[1].plot(time, roi_raw, alpha=.35, color=colors[arm], label=f"{arm} fixed ROI {args.bin_ms:g} ms")
        axes[1].plot(time, roi_trend, lw=2.4, color=colors[arm], label=f"{arm} {args.trend_ms:g} ms trend")
        rate = [np.nan if row[f"{arm}_roi_trend_rate_C_per_s"] is None else row[f"{arm}_roi_trend_rate_C_per_s"] for row in rows]
        axes[2].plot(time, np.asarray(rate)/1000.0, color=colors[arm], label=arm)
        axes[3].plot(time, [row[f"{arm}_mean_n_hot"] for row in rows], color=colors[arm], label=arm)
    axes[0].set_ylabel("Adopted (degC)")
    axes[1].set_ylabel("Fixed ROI avg (degC)")
    axes[2].axhline(0, color="black", lw=.8); axes[2].set_ylabel("ROI trend rate\n(degC/ms)")
    axes[3].axhline(args.min_mean_n_hot, color="grey", ls=":", label="visual reliability threshold")
    axes[3].set_ylabel("Mean n_hot"); axes[3].set_yscale("symlog", linthresh=1); axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(ncol=2, fontsize=8)
    fig.suptitle(f"V2 heating/cooling diagnostic: fixed {args.bin_ms:g} ms bins, "
                 f"{args.trend_ms:g} ms fixed-ROI trend")
    fig.tight_layout(); fig.savefig(prefix.with_suffix(".png"), dpi=200)
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
