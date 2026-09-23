#!/usr/bin/env python3
"""Create or validate a fail-closed identity for one decomposition run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


RUNTIME_SUFFIXES = {".py", ".sh", ".json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_data_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def runtime_files(repo: Path, solver_compare: Path) -> Iterable[Path]:
    for package in (repo / "jax_fem", repo / "jax_fem_am"):
        for path in package.rglob("*.py"):
            if path.is_file() and "__pycache__" not in path.parts:
                yield path
    for path in solver_compare.rglob("*"):
        if (
            path.is_file()
            and path.suffix in RUNTIME_SUFFIXES
            and "results" not in path.relative_to(solver_compare).parts
            and "decomposition-results" not in path.relative_to(solver_compare).parts
        ):
            yield path


def runtime_tree_identity(repo: Path, solver_compare: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(set(runtime_files(repo, solver_compare)))
    for path in files:
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "git_head": head,
        "runtime_tree_sha256": digest.hexdigest(),
        "runtime_file_count": len(files),
    }


def environment_identity(python_bin: str) -> dict[str, str]:
    script = (
        "import json,platform,jax,jaxlib,numpy,scipy;"
        "print(json.dumps({'python':platform.python_version(),"
        "'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__},sort_keys=True))"
    )
    completed = subprocess.run(
        [python_bin, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def write_or_validate_manifest(root: Path, payload: dict[str, Any]) -> str:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment_manifest.json"
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != canonical:
            raise RuntimeError(
                f"experiment identity differs from {path}; choose a new OUT"
            )
        return identity

    stale = sorted(root.glob("warmup-*")) + sorted(root.glob("repeat-*"))
    if stale or (root / "decomposition_compare.json").exists():
        raise RuntimeError(
            f"legacy outputs exist without an experiment manifest under {root}; "
            "choose a new OUT"
        )

    fd, temporary = tempfile.mkstemp(
        prefix=".experiment_manifest.", suffix=".tmp", dir=root
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return identity


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    solver_compare = Path(__file__).resolve().parent
    arm_specs: dict[str, Any] = {}
    environments: dict[str, Any] = {}
    for name in args.arms:
        spec_path = solver_compare / "decomposition-arms" / f"{name}.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        arm_specs[name] = spec
        python_bin = str(spec["python_bin"])
        environments.setdefault(python_bin, environment_identity(python_bin))
    return {
        "schema": "v2.cube-decomposition-experiment/1",
        "mode": args.mode,
        "path_file": str(args.path_file.resolve()),
        "path_sha256": sha256_file(args.path_file),
        "expected_steps": csv_data_rows(args.path_file),
        "contract_file": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "mechanics_every": args.mechanics_every,
        "mkl_omp_threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "order_policy": "warmup-forward; measured odd-forward/even-reverse",
        "arms": args.arms,
        "arm_specs": arm_specs,
        "environments": environments,
        "repo": str(repo),
        "code": runtime_tree_identity(repo, solver_compare),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--path-file", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--mechanics-every", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    identity = write_or_validate_manifest(args.root, build_payload(args))
    print(f"experiment_manifest_sha256={identity}")


if __name__ == "__main__":
    main()
