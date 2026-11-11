from pathlib import Path


def test_analyzer_accepts_wider_recording_window_for_narrow_summary():
    path = (Path(__file__).parents[2] / "cases" / "AM-Benchmark" / "verification"
            / "v2-cube-rs" / "model" / "analyze_pyrometer.py")
    text = path.read_text(encoding="utf-8")
    assert 'summary_window_in_file = online_summary.get("summary_window_s")' in text
    assert 'recorded_window[0] > summary_window[0]' in text
    assert 'recorded_window[1] < summary_window[1]' in text
    assert '"window_s": [float(args.observation_window' not in text
