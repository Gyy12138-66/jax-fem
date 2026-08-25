#!/usr/bin/env python3
"""评分规范 §3.2 严格口径体检:Fig 14 五个目标 10 ms 箱内有没有 `NA` 时样。

冻结规范(scoring-spec-thermal-gate-v2.md @ 3b9c220)§3.2 写的是:条件平均在
**目标箱内出现任何 NA 时样**(该瞬时圆内没有单元 > 1000 degC)→ 该点及整轮
Fig 14 评分 INVALID;§4.1 对峰时刻的要求同样是窗内任一箱不得有 NA。

这条判据的分母是**时样**,不是箱 —— 与 `summarize_online_observables.py`
的实现(按有效时样加权的区间平均,即 Balbaa 原文的分箱读法)不是一回事。
本脚本不评分、不改任何读数,只把两种口径的差异摆到台面上:对每个目标箱数
一数 n_hot == 0 的求解步有多少。判断"规范原样用还是另立版本"需要这个数。

用法:
    python check_fig14_bin_na.py <online_observables.jsonl> \
        [--experiment ../inputs/balbaa-fig14-pyrometer.json] \
        [--bin-ms 10] [--window 0.45,0.90] [--output out.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FIG14 = HERE.parent / "inputs" / "balbaa-fig14-pyrometer.json"


def target_bins(rows, times, bin_s):
    out = []
    for t in times:
        k = int(math.floor(t / bin_s))
        lo, hi = k * bin_s, (k + 1) * bin_s
        inb = [r for r in rows if lo <= float(r["time_s"]) < hi]
        na = [r for r in inb if int(r["n_hot"]) == 0]
        valid = [float(r["avg_K"]) for r in inb if r.get("avg_K") is not None]
        out.append({
            "time_s": t, "bin_index": k, "bin_lo_s": lo, "bin_hi_s": hi,
            "n_samples": len(inb), "n_na_samples": len(na),
            "na_fraction": (len(na) / len(inb)) if inb else None,
            "min_n_hot": min((int(r["n_hot"]) for r in inb), default=None),
            "max_n_hot": max((int(r["n_hot"]) for r in inb), default=None),
            "avg_K_valid_only": (sum(valid) / len(valid)) if valid else None,
            "strict_3_2_verdict": ("INVALID" if na else "OK") if inb else "NO_SAMPLES",
        })
    return out


def window_bins(rows, window, bin_s):
    lo_w, hi_w = window
    bins = {}
    for r in rows:
        t = float(r["time_s"])
        if lo_w <= t < hi_w:
            bins.setdefault(int(math.floor(t / bin_s)), []).append(r)
    with_na = sorted(k for k, v in bins.items() if any(int(r["n_hot"]) == 0 for r in v))
    return {"n_bins": len(bins), "n_bins_with_na": len(with_na),
            "bins_with_na": with_na,
            "strict_4_1_verdict": "INVALID" if with_na else "OK"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--experiment", type=Path, default=DEFAULT_FIG14)
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--window", default="0.45,0.90")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{args.jsonl} 是空的")
    fig14 = json.loads(args.experiment.read_text(encoding="utf-8"))
    # 规范 §3.1:评分时刻固定为 experimental[].time_s 的全精度值
    times = [float(p["time_s"]) for p in fig14["experimental"]]
    window = tuple(float(v) for v in args.window.split(","))
    if len(window) != 2 or not window[0] < window[1]:
        raise SystemExit("--window 需要 t0,t1 且 t0 < t1")
    bin_s = args.bin_ms * 1.0e-3

    targets = target_bins(rows, times, bin_s)
    scan = window_bins(rows, window, bin_s)
    doc = {
        "id": "fig14-bin-na-check",
        "what": "规范 §3.2/§4.1 严格口径(时样级 NA)的体检,不是评分",
        "source": str(args.jsonl), "n_rows": len(rows),
        "bin_ms": args.bin_ms, "window_s": list(window),
        "target_bins": targets, "window_scan": scan,
        "strict_3_2_fig14_verdict": ("INVALID" if any(t["n_na_samples"] for t in targets)
                                     else "OK"),
        "note": "NA 时样 = 该求解步圆内顶层没有单元 >= 1000 degC(n_hot == 0)。"
                "summarize_online_observables.py 的区间平均只对有效时样加权,"
                "与本判据不同;两者差异见 scoring-spec-amendment-A2-PROPOSAL.md",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    print(f"=== 规范 §3.2 严格口径体检:{args.jsonl.name},{len(rows)} 行,"
          f"箱 {args.bin_ms:.0f} ms ===")
    print(f"  {'t_i [s]':>8} {'目标箱 [s]':>16} {'时样':>5} {'NA时样':>6} {'NA占比':>7}"
          f" {'n_hot':>9} {'有效均值 K':>10}  判定")
    for t in targets:
        f = lambda v: f"{v:10.1f}" if v is not None else "       n/a"
        print(f"  {t['time_s']:8.4f} [{t['bin_lo_s']:.3f},{t['bin_hi_s']:.3f}) "
              f"{t['n_samples']:5d} {t['n_na_samples']:6d} "
              f"{(t['na_fraction'] or 0)*100:6.1f}% "
              f"{t['min_n_hot']!s:>3}..{t['max_n_hot']!s:<4} {f(t['avg_K_valid_only'])}"
              f"  {t['strict_3_2_verdict']}")
    print(f"  Fig 14 整轮(§3.2 严格):{doc['strict_3_2_fig14_verdict']}")
    print(f"  窗 {window[0]:.2f}-{window[1]:.2f} s:{scan['n_bins']} 箱,"
          f"{scan['n_bins_with_na']} 箱含 NA 时样 -> §4.1 峰时刻 {scan['strict_4_1_verdict']}")
    if args.output:
        print(f"  写出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
