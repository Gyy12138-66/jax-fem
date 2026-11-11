#!/usr/bin/env python3
"""Build a content-addressed identity for a Balbaa V2 thermal-gate arm."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(repo):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "."], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo).split(b"\0")
    digest = hashlib.sha256()
    digest.update(diff)
    untracked_hashes = {}
    for raw in sorted(item for item in untracked if item):
        relative = raw.decode("utf-8", errors="surrogateescape")
        path = Path(repo) / relative
        if path.is_file():
            value = sha256_file(path)
            untracked_hashes[relative] = value
            digest.update(raw + b"\0" + value.encode() + b"\0")
    return {
        "commit": commit,
        "worktree_sha256": digest.hexdigest(),
        "dirty": bool(diff or untracked_hashes),
        "untracked_sha256": untracked_hashes,
    }


def referenced_inputs(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = {}
    for key, value in sorted(config.items()):
        # Metadata keys such as ``_k_table_liquid_note`` describe a table but
        # are not themselves paths. Only non-private config keys participate.
        is_file_reference = (not key.startswith("_")
                             and (key.endswith(("_table", "_csv"))
                                  or "_table_" in key))
        if not isinstance(value, str) or not is_file_reference:
            continue
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"config reference {key} does not exist: {value}")
        result[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def build_identity(args):
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    mesh = Path(args.mesh).resolve()
    scan_path = Path(args.path).resolve()
    identity = {
        "schema_version": "balbaa-v2-run/2",
        "arm": args.arm,
        "git": git_snapshot(repo),
        "inputs_sha256": {
            "config": sha256_file(config),
            "mesh": sha256_file(mesh),
            "path": sha256_file(scan_path),
        },
        "config_references": referenced_inputs(config),
        "parameters": json.loads(args.parameters_json),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["run_id"] = hashlib.sha256(canonical).hexdigest()
    identity["status"] = "running"
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = build_identity(args)
    args.output.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(identity["run_id"])


if __name__ == "__main__":
    main()
