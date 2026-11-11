#!/usr/bin/env python3
"""Compare repeated full-Newton and modified-Newton CPU/GPU arms.

The field-difference implementation is deliberately shared with
compare_solver_arms.py so that the frozen solver comparison and this research
lane use the same observation operators. No physics or speed threshold is
encoded here: the script reports evidence and activation counters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_ARMS = (
    "cpu_full_phase23",
    "cpu_modified_phase33",
    "gpu_full_phase23",
    "gpu_modified_phase33",
)
PROFILE_COUNTERS = (
    "jacobian_builds",
    "jacobian_reuse_hits",
    "jacobian_refreshes",
    "thermal_jacobian_builds",
    "mechanics_jacobian_builds",
    "mechanics_jacobian_reuse_hits",
    "mechanics_jacobian_refreshes",
)
TIMING_KEYS = (
    "wall_s",
    "s_per_step",
    "assembly_s",
    "solver_s",
    "newton_wall_s",
)


def load_shared_compare():
    path = HERE / "compare_solver_arms.py"
    spec = importlib.util.spec_from_file_location("solver_compare_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared comparison helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = load_shared_compare()


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def median(values: list[Any]) -> float | None:
    usable = [float(value) for value in values if numeric(value)]
    return statistics.median(usable) if usable else None


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        raise SystemExit(f"missing experiment identity: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_repeat(repeat_dir: Path, name: str) -> dict[str, Any]:
    arm = SHARED.load_arm(repeat_dir, name)
    profile_path = repeat_dir / name / "profile.json"
    arm["profile_present"] = profile_path.is_file()
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        meta = profile.get("meta", {})
        for key in PROFILE_COUNTERS:
            arm[key] = meta.get(key, 0)
        stats = meta.get("pardiso_stats", {})
        arm["pardiso_stats"] = stats if isinstance(stats, dict) else {"raw": stats}
        scoped = meta.get("pardiso_stats_by_scope", {})
        mechanics_stats = (
            scoped.get("mechanics", {}) if isinstance(scoped, dict) else {}
        )
        arm["mechanics_pardiso_stats"] = (
            mechanics_stats
            if isinstance(mechanics_stats, dict)
            else {"raw": mechanics_stats}
        )
    return arm


def safe_field_diffs(repeat_dir: Path, baseline: str, arm: str) -> dict[str, Any]:
    try:
        return SHARED.field_diffs(repeat_dir, baseline, arm)
    except Exception as exc:  # keep timing evidence even when a frame is absent
        return {"compared": False, "reason": f"field comparison failed: {exc}"}


def aggregate_pardiso(
    repeats: list[dict[str, Any]], key: str = "pardiso_stats"
) -> dict[str, Any]:
    keys = sorted(
        {
            item
            for repeat in repeats
            for item, value in repeat.get(key, {}).items()
            if numeric(value)
        }
    )
    return {
        f"{item}_median": median(
            [repeat.get(key, {}).get(item) for repeat in repeats]
        )
        for item in keys
    }


def aggregate_diffs(repeats: list[dict[str, Any]], key: str) -> dict[str, Any]:
    records = [repeat.get(key, {}) for repeat in repeats]
    diff_keys = sorted(
        {
            field
            for record in records
            for field, value in record.items()
            if numeric(value) and ("diff" in field or "differing" in field)
        }
    )
    return {
        "compared_repeats": sum(record.get("compared") is True for record in records),
        "max_over_repeats": {
            field: max(
                float(record[field]) for record in records if numeric(record.get(field))
            )
            for field in diff_keys
        },
        "records": records,
    }


def aggregate_arm(
    repeats: list[dict[str, Any]], expected_steps: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "repeat_count": len(repeats),
        "complete_count": sum(repeat.get("ledger_complete") is True for repeat in repeats),
        "repeats": repeats,
    }
    for key in TIMING_KEYS + PROFILE_COUNTERS:
        summary[f"{key}_median"] = median([repeat.get(key) for repeat in repeats])
    for key in ("solver_fallbacks", "newton_nonconvergence", "fallback_warnings", "nan_mentions"):
        values = [repeat.get(key) for repeat in repeats]
        summary[f"{key}_total"] = sum(int(value) for value in values if numeric(value))
    summary["pardiso_stats"] = aggregate_pardiso(repeats)
    summary["mechanics_pardiso_stats"] = aggregate_pardiso(
        repeats, "mechanics_pardiso_stats"
    )
    summary["vs_accuracy_baseline"] = aggregate_diffs(repeats, "vs_accuracy_baseline")
    summary["vs_platform_baseline"] = aggregate_diffs(repeats, "vs_platform_baseline")
    reasons = []
    if not all(repeat.get("present") is True for repeat in repeats):
        reasons.append("one or more arm directories are missing")
    if summary["complete_count"] != len(repeats):
        reasons.append("one or more ledgers are incomplete")
    if not all(repeat.get("profile_present") is True for repeat in repeats):
        reasons.append("one or more profiles are missing")
    if not all(repeat.get("steps") == expected_steps for repeat in repeats):
        reasons.append(f"profile steps do not all equal {expected_steps}")
    for key in (
        "solver_fallbacks_total",
        "newton_nonconvergence_total",
        "fallback_warnings_total",
        "nan_mentions_total",
    ):
        if summary[key] != 0:
            reasons.append(f"{key}={summary[key]}")
    for key in ("vs_accuracy_baseline", "vs_platform_baseline"):
        if summary[key]["compared_repeats"] != len(repeats):
            reasons.append(f"{key} field comparison is incomplete")
    summary["evidence_valid"] = not reasons
    summary["invalid_reasons"] = reasons
    return summary


def paired_speedups(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[float]:
    reference_by_repeat = {item.get("repeat"): item for item in reference}
    values = []
    for item in candidate:
        baseline = reference_by_repeat.get(item.get("repeat"), {})
        base_wall = baseline.get("wall_s")
        wall = item.get("wall_s")
        if numeric(base_wall) and numeric(wall) and wall:
            values.append(float(base_wall) / float(wall))
    return values


def format_number(value: Any, width: int, fmt: str = ".3g") -> str:
    if not numeric(value):
        return "-".rjust(width)
    return format(value, fmt).rjust(width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", default="cpu_full_phase23")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.root)
    manifest_arms = list(manifest.get("arms", []))
    if list(args.arms) != manifest_arms:
        raise SystemExit(
            f"requested arms {args.arms} differ from manifest arms {manifest_arms}"
        )
    expected_repeat_names = [
        f"repeat-{index:02d}"
        for index in range(1, int(manifest["repeats"]) + 1)
    ]
    actual_repeat_names = sorted(
        path.name for path in args.root.glob("repeat-*") if path.is_dir()
    )
    identity_errors = []
    if actual_repeat_names != expected_repeat_names:
        identity_errors.append(
            "repeat directories differ from manifest: "
            f"expected={expected_repeat_names}, actual={actual_repeat_names}"
        )
    repeat_dirs = [args.root / name for name in expected_repeat_names]
    specs = manifest["arm_specs"]
    by_arm: dict[str, list[dict[str, Any]]] = {name: [] for name in args.arms}
    for repeat_dir in repeat_dirs:
        for name in args.arms:
            record = load_repeat(repeat_dir, name)
            record["repeat"] = repeat_dir.name
            record["platform"] = record.get("platform") or specs[name].get("platform")
            record["vs_accuracy_baseline"] = safe_field_diffs(
                repeat_dir, specs[name].get("accuracy_baseline", args.baseline), name
            )
            record["vs_platform_baseline"] = safe_field_diffs(
                repeat_dir, specs[name].get("comparison_baseline", args.baseline), name
            )
            by_arm[name].append(record)

    expected_steps = int(manifest["expected_steps"])
    arms = {
        name: aggregate_arm(records, expected_steps)
        for name, records in by_arm.items()
    }
    accuracy_wall = arms.get(args.baseline, {}).get("wall_s_median")
    for name, summary in arms.items():
        reference = specs[name].get("comparison_baseline", args.baseline)
        ref_wall = arms.get(reference, {}).get("wall_s_median")
        wall = summary.get("wall_s_median")
        summary["comparison_baseline"] = reference
        paired = paired_speedups(by_arm.get(reference, []), by_arm[name])
        summary["paired_speedups_vs_platform_baseline"] = paired
        reference_valid = arms.get(reference, {}).get("evidence_valid") is True
        summary["ratio_of_medians_vs_platform_baseline"] = (
            float(ref_wall) / float(wall)
            if numeric(ref_wall) and numeric(wall) and wall
            else None
        )
        summary["speedup_vs_platform_baseline"] = (
            median(paired)
            if summary["evidence_valid"] and reference_valid
            else None
        )
        summary["speedup_vs_accuracy_baseline"] = (
            float(accuracy_wall) / float(wall)
            if (
                summary["evidence_valid"]
                and arms.get(args.baseline, {}).get("evidence_valid") is True
                and numeric(accuracy_wall)
                and numeric(wall)
                and wall
            )
            else None
        )
        extra_argv = specs[name].get("extra_argv", [])
        reuse_requested = "--mechanics-jacobian-reuse" in extra_argv
        reuse_hits = summary.get("mechanics_jacobian_reuse_hits_median") or 0
        phase33_calls = summary.get("mechanics_pardiso_stats", {}).get(
            "phase33_calls_median"
        ) or 0
        backsolve_hits = summary.get("mechanics_pardiso_stats", {}).get(
            "backsolve_hits_median"
        ) or 0
        summary["reuse_requested"] = reuse_requested
        summary["reuse_activated"] = (
            bool(reuse_hits > 0 and backsolve_hits > 0 and phase33_calls > 0)
            if reuse_requested
            else None
        )
        summary["reuse_interpretable"] = (
            summary["reuse_activated"] if reuse_requested else None
        )
        if reuse_requested and not summary["reuse_activated"]:
            summary["speedup_vs_platform_baseline"] = None

    result = {
        "schema": "v2.cube-decomposition-compare/1",
        "root": str(args.root.resolve()),
        "accuracy_baseline": args.baseline,
        "repeat_directories": [path.name for path in repeat_dirs],
        "experiment_manifest": manifest,
        "identity_errors": identity_errors,
        "arms": arms,
        "interpretation": {
            "gpu_scope": "JAX element assembly on GPU; MKL PARDISO factorization/phase33 remain on CPU",
            "performance": "median wall time over measured repeats; warmups are excluded",
            "correctness": "true solver fields are compared with the shared solver-compare observation operators",
            "activation": "jacobian and pardiso counters must show that the requested reuse path actually ran",
        },
    }
    result["evidence_valid"] = not identity_errors and all(
        arm["evidence_valid"] for arm in arms.values()
    )
    output = args.output or (args.root / "decomposition_compare.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"{'arm':25s} {'ok':>3s} {'n':>3s} {'done':>4s} {'wall med':>9s} {'assembly':>9s} "
        f"{'solver':>9s} {'newton':>9s} {'J build':>8s} {'J reuse':>8s} "
        f"{'refresh':>7s} {'speedup':>8s} {'u diff':>10s} {'vm rms MPa':>11s}"
    )
    for name in args.arms:
        summary = arms[name]
        diffs = summary["vs_accuracy_baseline"]["max_over_repeats"]
        print(
            f"{name:25s} {('yes' if summary['evidence_valid'] else 'no'):>3s} "
            f"{summary['repeat_count']:3d} {summary['complete_count']:4d} "
            f"{format_number(summary.get('wall_s_median'), 9, '.3f')} "
            f"{format_number(summary.get('assembly_s_median'), 9, '.3f')} "
            f"{format_number(summary.get('solver_s_median'), 9, '.3f')} "
            f"{format_number(summary.get('newton_wall_s_median'), 9, '.3f')} "
            f"{format_number(summary.get('mechanics_jacobian_builds_median'), 8, '.0f')} "
            f"{format_number(summary.get('mechanics_jacobian_reuse_hits_median'), 8, '.0f')} "
            f"{format_number(summary.get('mechanics_jacobian_refreshes_median'), 7, '.0f')} "
            f"{format_number(summary.get('speedup_vs_platform_baseline'), 8, '.3f')} "
            f"{format_number(diffs.get('u_max_abs_diff_m'), 10, '.3g')} "
            f"{format_number(diffs.get('vm_rms_diff_MPa'), 11, '.3g')}"
        )
    print(f"wrote {output}")
    if not result["evidence_valid"]:
        raise SystemExit(
            "comparison evidence is invalid; see identity_errors and "
            "arms.*.invalid_reasons in the JSON"
        )


if __name__ == "__main__":
    main()
