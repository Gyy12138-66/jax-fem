#!/usr/bin/env python3
"""Create a short, non-destructive prefix of the layer-1 scan path."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_prefix(source: Path, destination: Path, steps: int) -> dict[str, object]:
    if steps < 1:
        raise ValueError("--steps must be >= 1")
    if not source.is_file():
        raise FileNotFoundError(f"source path does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("input and output must be different files")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"source path is empty: {source}") from exc
        rows = []
        for _ in range(steps):
            try:
                rows.append(next(reader))
            except StopIteration:
                break

    if len(rows) != steps:
        raise ValueError(
            f"requested {steps} rows, but {source} contains only {len(rows)} data rows"
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        if destination.exists():
            if not destination.is_file():
                raise FileExistsError(
                    f"refusing to replace non-file destination: {destination}"
                )
            if sha256(Path(tmp_name)) != sha256(destination):
                raise FileExistsError(
                    "refusing to overwrite an existing quick-path input with "
                    f"different content: {destination}; choose a new OUT"
                )
            os.unlink(tmp_name)
        else:
            os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "destination": str(destination.resolve()),
        "destination_sha256": sha256(destination),
        "steps": len(rows),
        "header": header,
        "last_time": rows[-1][header.index("time")] if "time" in header else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the header and first N data rows of a layer-1 path CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    result = copy_prefix(args.input, args.output, args.steps)
    print(
        "quick path: "
        f"steps={result['steps']} last_time={result['last_time']} "
        f"source_sha256={result['source_sha256']} "
        f"output={result['destination']}"
    )


if __name__ == "__main__":
    main()
