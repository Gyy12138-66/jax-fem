#!/usr/bin/env python3
"""逐字节比对两个运行目录的求解产物。

沿用仓库既有的金标验法(GOLDEN_EQUIVALENCE.txt,2026-07-22 裁决):行为保持性
用**逐字节** cmp 判,不用容差判 —— 容差会把"几乎一样"读成"一样",而这里要
证明的恰恰是"一样"。

比什么:
  *.vtu                                求解出来的温度场
  thermal_energy_ledger.jsonl          逐步能量台账
  thermal_energy_ledger_summary.json   台账汇总
  path_used.csv                        求解器实际走的路径与时间

**不**比 used_config.json:新旗标本来就该出现在配置回显里(那是溯源,不是
行为),把它藏起来才是问题。它的键差异单独列出来,给判读的人看。
也不比 profile.json / run.log —— 里面有墙钟时间和路径,天然逐次不同。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SOLUTION_GLOBS = ("*.vtu", "thermal_energy_ledger.jsonl",
                  "thermal_energy_ledger_summary.json", "path_used.csv")

# used_config 里这几个键**必然**逐次不同:每一遍写进它自己的目录,路径就带着
# 那一遍的标签。它们是运行身份,不是配置差异,所以不进判定。其余任何已有键
# 取值变了都算失败。
RUN_IDENTITY_KEYS = frozenset({
    "output_dir", "path_file", "profile_json", "profile_label",
    "config", "inp", "calibration_dir",
})


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(run_dir):
    out = {}
    for pattern in SOLUTION_GLOBS:
        for p in sorted(Path(run_dir).glob(pattern)):
            out[p.name] = digest(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    base, cand = collect(args.baseline), collect(args.candidate)
    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    differ = sorted(k for k in set(base) & set(cand) if base[k] != cand[k])
    same = sorted(k for k in set(base) & set(cand) if base[k] == cand[k])

    cfg_delta = {}
    try:
        b = json.loads((Path(args.baseline) / "used_config.json").read_text())
        c = json.loads((Path(args.candidate) / "used_config.json").read_text())
        cfg_delta = {
            "keys_added": sorted(set(c) - set(b)),
            "keys_removed": sorted(set(b) - set(c)),
            "values_changed": sorted(k for k in set(b) & set(c)
                                     if b[k] != c[k] and k not in RUN_IDENTITY_KEYS),
            "run_identity_keys_differing_by_construction": sorted(
                k for k in set(b) & set(c)
                if b[k] != c[k] and k in RUN_IDENTITY_KEYS),
            "added_values": {k: c[k] for k in sorted(set(c) - set(b))},
        }
    except Exception as error:
        cfg_delta = {"error": f"{type(error).__name__}: {error}"}

    ok = not (differ or only_base or only_cand)
    tag = f"[{args.label}] " if args.label else ""
    print(f"{tag}逐字节比对 {args.baseline} vs {args.candidate}")
    print(f"  求解产物 {len(same)} 个逐字节相同" + (f",{len(differ)} 个不同" if differ else ""))
    for name in differ:
        print(f"    DIFFERS  {name}")
    for name in only_base:
        print(f"    只在基线  {name}")
    for name in only_cand:
        print(f"    只在候选  {name}")
    if cfg_delta.get("keys_added"):
        print(f"  used_config 新增键(溯源,不参与逐字节判定):")
        for k in cfg_delta["keys_added"]:
            print(f"    + {k} = {cfg_delta['added_values'][k]!r}")
    if cfg_delta.get("run_identity_keys_differing_by_construction"):
        print("  used_config 运行身份键(每遍各写各的目录,不进判定):"
              f"{cfg_delta['run_identity_keys_differing_by_construction']}")
    if cfg_delta.get("values_changed"):
        print(f"  !! used_config 已有键取值变了:{cfg_delta['values_changed']}")
        ok = False
    if cfg_delta.get("keys_removed"):
        print(f"  !! used_config 少了键:{cfg_delta['keys_removed']}")
        ok = False
    print(f"  判定:{'通过 —— 行为保持' if ok else '失败'}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "label": args.label, "passed": ok,
            "baseline": str(args.baseline), "candidate": str(args.candidate),
            "method": "sha256 per file, zero tolerance (GOLDEN_EQUIVALENCE.txt "
                      "discipline: behaviour preservation is a byte question)",
            "solution_artifacts_compared": SOLUTION_GLOBS,
            "identical": same, "differing": differ,
            "only_in_baseline": only_base, "only_in_candidate": only_cand,
            "used_config_delta": cfg_delta,
            "used_config_note": "provenance, deliberately excluded from the "
                                "byte verdict -- new flags SHOULD show up here",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  写出 {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
