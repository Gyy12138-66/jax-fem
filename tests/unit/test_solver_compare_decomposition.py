"""Unit contracts for the solver-comparison research helpers."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = (
    REPO_ROOT
    / "cases"
    / "AM-Benchmark"
    / "verification"
    / "v2-cube-rs"
    / "solver-compare"
)


def load_helper(name):
    path = HELPER_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUICK_PATH = load_helper("make_quick_path")
MANIFEST = load_helper("prepare_experiment_manifest")
COMPARE = load_helper("compare_decomposition_arms")


class QuickPathContractTest(unittest.TestCase):
    def write_source(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["time", "x", "y", "z", "power"])
            writer.writerow(["0.0", "0", "0", "0", "140"])
            writer.writerow(["0.1", "1", "0", "0", "140"])
            writer.writerow(["0.2", "2", "0", "0", "140"])

    def test_copy_prefix_is_idempotent_but_never_replaces_different_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            destination = root / "out" / "quick.csv"
            self.write_source(source)

            first = QUICK_PATH.copy_prefix(source, destination, 2)
            original_bytes = destination.read_bytes()
            second = QUICK_PATH.copy_prefix(source, destination, 2)

            self.assertEqual(first["destination_sha256"], second["destination_sha256"])
            self.assertEqual(destination.read_bytes(), original_bytes)
            self.assertEqual(first["steps"], 2)
            self.assertEqual(first["last_time"], "0.1")

            destination.write_text("old failed-run evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "choose a new OUT"):
                QUICK_PATH.copy_prefix(source, destination, 2)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "old failed-run evidence\n",
            )

    def test_copy_prefix_rejects_alias_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            self.write_source(source)
            with self.assertRaisesRegex(ValueError, "different files"):
                QUICK_PATH.copy_prefix(source, source, 1)
            with self.assertRaisesRegex(ValueError, "contains only 3"):
                QUICK_PATH.copy_prefix(source, Path(tmp) / "too-long.csv", 4)


class ExperimentManifestContractTest(unittest.TestCase):
    def test_manifest_is_idempotent_and_rejects_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"schema": "test/1", "steps": 8, "arm": "full"}
            first = MANIFEST.write_or_validate_manifest(root, payload)
            second = MANIFEST.write_or_validate_manifest(root, dict(payload))
            self.assertEqual(first, second)

            with self.assertRaisesRegex(RuntimeError, "choose a new OUT"):
                MANIFEST.write_or_validate_manifest(
                    root, {**payload, "steps": 16}
                )

    def test_manifest_rejects_legacy_outputs_without_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "repeat-01").mkdir()
            with self.assertRaisesRegex(RuntimeError, "legacy outputs"):
                MANIFEST.write_or_validate_manifest(
                    root, {"schema": "test/1"}
                )


class ComparisonHealthGateTest(unittest.TestCase):
    def test_missing_profile_invalidates_arm_even_with_complete_ledger(self):
        record = {
            "present": True,
            "profile_present": False,
            "ledger_complete": True,
            "steps": None,
            "solver_fallbacks": 0,
            "newton_nonconvergence": 0,
            "fallback_warnings": 0,
            "nan_mentions": 0,
            "vs_accuracy_baseline": {"compared": True},
            "vs_platform_baseline": {"compared": True},
        }
        summary = COMPARE.aggregate_arm([record], expected_steps=8)
        self.assertIs(summary["evidence_valid"], False)
        self.assertIn("one or more profiles are missing", summary["invalid_reasons"])
        self.assertIn(
            "profile steps do not all equal 8", summary["invalid_reasons"]
        )


if __name__ == "__main__":
    unittest.main()
