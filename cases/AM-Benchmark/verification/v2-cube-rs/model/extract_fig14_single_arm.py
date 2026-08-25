#!/usr/bin/env python3
"""Extract the five registered Fig. 14 values from one thermal-gate arm."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyrometer", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arm", default="no-keff")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    pyrometer = json.loads(args.pyrometer.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    series = pyrometer["series"]
    points = []
    for experiment in reference["experimental"]:
        time_s = experiment["time_s"]
        row = min(series, key=lambda value: abs(value["t_center_s"] - time_s))
        if row.get("avg_K") is None:
            raise SystemExit(f"Fig.14 target t={time_s:.9f}s has no valid adopted reading")
        points.append({
            "experiment_time_s": time_s,
            "model_bin_center_s": row["t_center_s"],
            "experiment_C": experiment["temperature_C"],
            "conditional_average_C": row["avg_K"] - 273.15,
            "two_colour_C": row["two_colour_K"] - 273.15,
            "full_spot_average_C": row["full_spot_avg_K"] - 273.15,
            "max_C": row["max_K"] - 273.15,
            "n_samples": row["n_samples"],
            "mean_n_hot": row["mean_n_hot"],
        })

    protocol = pyrometer["protocol"]
    output = {
        "arm": args.arm,
        "recording_window_s": protocol["recorded_window_s"],
        "summary_window_s": protocol["summary_window_s"],
        "matching": "nearest registered 10 ms response bin center",
        "points": points,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    times = [point["experiment_time_s"] for point in points]
    ax.plot(times, [point["experiment_C"] for point in points], "ko-", label="Balbaa experiment")
    ax.plot(times, [point["conditional_average_C"] for point in points], "o-",
            label=f"{args.arm}: Balbaa conditional avg")
    ax.plot(times, [point["two_colour_C"] for point in points], "s--",
            label=f"{args.arm}: synthetic two-colour")
    ax.plot(times, [point["full_spot_average_C"] for point in points], "^--",
            label=f"{args.arm}: full-spot avg")
    ax.set(xlabel="Time (s)", ylabel="Temperature (degC)",
           title=f"Fig. 14 check: {args.arm}, registered 10 ms bins")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180)

    for point in points:
        print(json.dumps(point, ensure_ascii=False))


if __name__ == "__main__":
    main()
