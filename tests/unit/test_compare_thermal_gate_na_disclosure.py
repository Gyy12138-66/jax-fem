"""Static contract: the Fig 14 comparer discloses NA fractions per spec v2.1 §3.2 (A2 (a))."""

from pathlib import Path


MODEL = (Path(__file__).resolve().parents[2]
         / "cases/AM-Benchmark/verification/v2-cube-rs/model")
INPUTS = MODEL.parent / "inputs"


def test_comparer_carries_na_disclosure_per_target_bin():
    text = (MODEL / "compare_thermal_gate.py").read_text(encoding="utf-8")
    for token in (
        'row[f"adopted_{tag}_na_time_fraction"] = frac',
        'row[f"adopted_{tag}_na_over_half"] = (frac > 0.5) if frac is not None else None',
        'row[f"adopted_{tag}_containing_bin_index"]',
        'three["_na_disclosure"]',
        "scoring-spec-thermal-gate-v2.1.md",
    ):
        assert token in text
    # the containing bin is looked up by index, never interpolated
    assert "bins_.get(int(t // bin_s_doc))" in text


def test_summarizer_discloses_na_per_bin():
    text = (MODEL / "summarize_online_observables.py").read_text(encoding="utf-8")
    for token in ('"n_na_samples": len(na_pieces)', '"na_time_fraction"',
                  '"na_over_half"', '"na_disclosure": {',
                  'int(row["n_hot"]) == 0'):
        assert token in text


def test_spec_v2_1_exists_and_v2_0_is_untouched():
    v20 = (INPUTS / "scoring-spec-thermal-gate-v2.md").read_text(encoding="utf-8")
    v21 = (INPUTS / "scoring-spec-thermal-gate-v2.1.md").read_text(encoding="utf-8")
    # v2.0 keeps the frozen sample-level clause verbatim
    assert "目标箱内出现任何 `NA` 即该点及整轮 Fig 14 评分 INVALID" in v20
    assert "## 修订记录" not in v20
    # v2.1 carries the adopted (a) wording and points back to A2
    assert "分母只取有效时样" in v21
    assert "na_time_fraction" in v21
    assert "scoring-spec-amendment-A2.md" in v21
    assert "3b9c220" in v21
    assert (INPUTS / "scoring-spec-amendment-A2.md").is_file()
