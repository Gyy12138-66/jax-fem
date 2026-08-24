"""Content-addressed identity tests for Balbaa V2 thermal-gate runs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (Path(__file__).resolve().parents[2]
          / "cases/AM-Benchmark/verification/v2-cube-rs/model/build_run_manifest.py")
SPEC = importlib.util.spec_from_file_location("build_run_manifest", SCRIPT)
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def test_referenced_material_tables_are_hashed(tmp_path):
    table = tmp_path / "k.csv"
    table.write_text("temperature,value\n300,10\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"k_table_solid": str(table), "name": "IN625"}),
                      encoding="utf-8")
    refs = manifest.referenced_inputs(config)
    assert refs["k_table_solid"]["sha256"] == manifest.sha256_file(table)


def test_run_id_changes_with_dirty_content_and_referenced_table(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    mesh = tmp_path / "mesh.inp"
    path = tmp_path / "path.csv"
    table = tmp_path / "k.csv"
    for file, content in ((mesh, "mesh"), (path, "path"), (table, "table-a")):
        file.write_text(content, encoding="utf-8")
    config.write_text(json.dumps({"k_table_liquid": str(table)}), encoding="utf-8")
    args = SimpleNamespace(repo=tmp_path, arm="asis", config=config, mesh=mesh,
                           path=path, parameters_json='{"dt":1}')
    snapshot = {"commit": "abc", "worktree_sha256": "dirty-a", "dirty": True,
                "untracked_sha256": {}}
    monkeypatch.setattr(manifest, "git_snapshot", lambda repo: dict(snapshot))
    first = manifest.build_identity(args)["run_id"]
    snapshot["worktree_sha256"] = "dirty-b"
    second = manifest.build_identity(args)["run_id"]
    assert first != second
    snapshot["worktree_sha256"] = "dirty-a"
    table.write_text("table-b", encoding="utf-8")
    third = manifest.build_identity(args)["run_id"]
    assert first != third
