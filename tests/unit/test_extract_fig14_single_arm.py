from pathlib import Path


def test_single_arm_extractor_uses_registered_five_points_and_three_readings():
    path = (Path(__file__).parents[2] / "cases" / "AM-Benchmark" / "verification"
            / "v2-cube-rs" / "model" / "extract_fig14_single_arm.py")
    text = path.read_text(encoding="utf-8")
    for token in (
        'reference["experimental"]',
        '"conditional_average_C"',
        '"two_colour_C"',
        '"full_spot_average_C"',
        '"summary_window_s"',
        'args.output_figure',
    ):
        assert token in text
