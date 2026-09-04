#!/usr/bin/env python3
"""Cross-arm comparison for V1-P campaigns (source-band arms, mesh ladder).

Input: name=run_dir pairs. Each run_dir must contain
v1_meltpool_metrics.json (analyze_v1.py output); energy-ledger /audit JSONs
are scanned opportunistically for the source capture fraction.

Output: <output>.json (machine-readable) and <output>.md (review table).
The first arm on the command line is the reference column for deltas.
Zero-calibration: this script only tabulates; adoption rules live in
V1PE-RERUN-PLAN.md and are frozen before production reads.
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+", metavar="name=run_dir")
ap.add_argument("--output", required=True,
                help="output path stem (.json / .md appended)")
args = ap.parse_args()


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        out[prefix[:-1]] = float(obj)
    return out


def find_capture_fraction(run_dir):
    """Search run JSONs for a source capture fraction, wherever it nests."""
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "capture_fraction" in key and isinstance(value, (int, float)):
                    yield key, float(value)
                else:
                    yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    hits = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if path.endswith("v1_meltpool_metrics.json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        for key, value in walk(data):
            hits.setdefault(f"{os.path.basename(path)}:{key}", value)
    return hits


arms = []
for spec in args.arms:
    name, _, run_dir = spec.partition("=")
    if not run_dir:
        raise SystemExit(f"expected name=run_dir, got: {spec}")
    metrics_path = os.path.join(run_dir, "v1_meltpool_metrics.json")
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = flatten(json.load(fh))
    metrics.update(
        {f"capture[{k}]": v for k, v in find_capture_fraction(run_dir).items()}
    )
    arms.append({"name": name, "run_dir": run_dir, "metrics": metrics})

reference = arms[0]
common = sorted(
    set.intersection(*(set(arm["metrics"]) for arm in arms))
    if len(arms) > 1 else set(reference["metrics"])
)
headline = [key for key in common if any(
    token in key for token in (
        "width", "depth", "pool_length", "cooling_rate", "Tmax_overall",
        "peak_T_probe", "capture",
    )
)]

result = {
    "reference_arm": reference["name"],
    "arms": [{"name": a["name"], "run_dir": a["run_dir"]} for a in arms],
    "metrics": {
        key: {a["name"]: a["metrics"].get(key) for a in arms}
        for key in common
    },
}
with open(f"{args.output}.json", "w", encoding="utf-8", newline="\n") as fh:
    json.dump(result, fh, indent=1)

lines = [
    "# V1-P cross-arm comparison",
    "",
    f"Reference arm: **{reference['name']}** "
    "(deltas are arm - reference, % of reference).",
    "",
    "| metric | " + " | ".join(a["name"] for a in arms) + " | max |delta| % |",
    "|---|" + "---|" * (len(arms) + 1),
]
for key in headline:
    ref_value = reference["metrics"].get(key)
    cells, max_delta = [], None
    for arm in arms:
        value = arm["metrics"].get(key)
        cells.append("-" if value is None else f"{value:.6g}")
        if (
            value is not None and ref_value not in (None, 0.0)
            and arm is not reference
        ):
            delta = 100.0 * (value - ref_value) / abs(ref_value)
            max_delta = delta if max_delta is None else max(
                max_delta, delta, key=abs
            )
    delta_cell = "-" if max_delta is None else f"{max_delta:+.2f}%"
    lines.append(f"| {key} | " + " | ".join(cells) + f" | {delta_cell} |")
lines += [
    "",
    "Full metric set (including per-file capture fractions): see the "
    "companion .json.",
]
with open(f"{args.output}.md", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"wrote {args.output}.json / .md "
      f"({len(common)} common metrics, {len(headline)} headline rows)")
