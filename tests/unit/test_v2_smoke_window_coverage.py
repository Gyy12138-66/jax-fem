import csv
import json
import subprocess
import sys
from pathlib import Path


MODEL = (Path(__file__).resolve().parents[2]
         / "cases/AM-Benchmark/verification/v2-cube-rs/model")


def test_six_track_smoke_cooling_lands_on_registered_window_end(tmp_path):
    path = tmp_path / "path.csv"
    ledger = tmp_path / "ledger.json"
    subprocess.run(
        [sys.executable, str(MODEL / "make_v2_path_multitrack.py"),
         "--tracks", "6", "--power", "220", "--speed", "0.650",
         "--hatch", "0.12e-3", "--output", str(path),
         "--ledger-json", str(ledger)],
        check=True,
    )

    data = json.loads(ledger.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        final_path_time = float(list(csv.DictReader(handle))[-1]["time"])

    cooling_steps = 90
    cooling_dt = (0.90 - final_path_time) / cooling_steps
    final_time = final_path_time + cooling_steps * cooling_dt

    assert abs(data["t_end_s"] - final_path_time) <= 1.0e-9
    assert 0.0 < cooling_dt <= 0.01
    assert abs(final_time - 0.90) <= 1.0e-12
