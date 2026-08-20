#!/usr/bin/env python3
"""热闸门对比:keff 双臂(parity / as-is) x 高温计实测 x Balbaa 的 ABAQUS。

**一等输出是两臂之差**(Fable5 2026-08-18 指定):Marangoni 升级决策
(本 issue metadata,DEFERRED)的输入就是"加不加 keff 到底差多少",
所以两臂差不是副产品,是本脚本的第一张表 —— 即使两臂各自都对不上实测,
两臂之差依然是有效结论。

其余两条腿沿用 V1 的三角形口径:
  code-to-experiment  各臂 vs 高温计实测点(带数字化读数不确定度)
  code-to-code        各臂 vs Balbaa 自己的数值点
并且把 **Balbaa 自己的数值-实测差**也算出来当尺子 —— 我们的差要和他的差比,
而不是和零比(SPEC "Acceptance framing":没有可调的阈值)。

## 三路读数(2026-08-20 起,IET-20)

`analyze_pyrometer.py` 现在每箱给三个读数,本脚本把三路**并列**评分:

  adopted      采纳口径(Balbaa Sec 3.3 条件平均)—— code-to-code 的主尺
  two_colour   双色辐射合成(仪器物理,D-V2-24)—— code-to-experiment 的主尺
  full_spot    全光斑无阈值平均 —— 诊断下界

**不同的腿该用不同的尺子**:复现他的数值曲线要用他的协议(adopted),
逼近实验要用仪器的物理(two_colour)。混用会得到 IET-19 昨夜那个怪象 ——
晚窗两点"看着对上了"其实是 n_hot 涨到 140 稀释出来的伪迹,同一时刻的双色
合成读数是 3000 K。三路一起出表,就没法再只挑一路说话。

输入:两个 analyze_pyrometer.py 的 JSON + digitize_fig14_pyrometer.py 的 JSON。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

C2K = 273.15

# (键, 短名, 中文标签, 这一路是为哪条腿服务的)
READINGS = (
    ("avg_K", "adopted", "采纳(条件平均)", "code-to-code:复现他的协议与数值曲线"),
    ("two_colour_K", "two_colour", "双色合成", "code-to-experiment:仪器物理"),
    ("full_spot_avg_K", "full_spot", "全光斑无阈值", "诊断下界:分母取满"),
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def series_arrays(doc, key="avg_K", beam_bins_only=False):
    """有效箱的 (t, T);无效箱(圆内无单元过 1000 degC)不参与。

    `beam_bins_only` 是三路并列时的**可比性**开关,不是过滤器,理由如下。

    采纳口径在"光束出圆"的箱里天然是 None(圆内没有单元过 1000 degC),所以
    它的时间插值只跨热箱。双色合成不一样:仪器在任何时刻都读得到东西,每个箱
    都有值,于是热箱(约 3000-3700 K)和出圆箱(约 650-950 K)交替出现 ——
    帧距 7.69 ms 与道周期 15.10 ms 的拍频(D-V2-25)。**在这样一条锯齿序列上
    做线性插值是没有意义的**:t = 0.5923 会插在 907 K 和 3703 K 中间给出
    2939 K,那个数既不是场,也不是仪器会读到的值。

    所以三路在 Fig 14 的五个时刻打分时,统一取**采纳口径也有效的那些箱**
    (即光束在圆内的箱)。这不是给双色路加阈值 —— 单元级仍然无阈值 ——
    而是让三路踩在同一组采样时刻上。拍频造成的曝光量另由
    `aliasing_envelope` 如实上报,不藏进插值里。
    """
    t, v = [], []
    for s in doc["series"]:
        if s.get(key) is None:
            continue
        if beam_bins_only and s.get("avg_K") is None:
            continue
        t.append(s["t_center_s"])
        v.append(s[key])
    return np.asarray(t), np.asarray(v)


def aliasing_envelope(doc, key, t_query, half_window_s):
    """t_query 附近 ±half_window 内该读数的包络 [min, max]。

    拍频曝光量的直接度量:包络越宽,说明"这一刻读到多少"越依赖采样相位。
    帧距压到 <= 0.5 ms 之后(IET-20 交付件 3)这个包络应当塌缩。
    """
    vals = [s[key] for s in doc["series"]
            if s.get(key) is not None
            and abs(s["t_center_s"] - t_query) <= half_window_s]
    if not vals:
        return None
    return {"min_K": float(min(vals)), "max_K": float(max(vals)),
            "span_K": float(max(vals) - min(vals)), "n_bins": len(vals)}


def sample_at(t_arr, v_arr, t_query, max_gap_s):
    """把一条箱序列取到 t_query。最近的有效箱超过 max_gap_s 就返回 None ——
    宁可报缺,不外推(外推出来的点会被当成模型预测读,那是假证据)。"""
    if t_arr.size == 0:
        return None, None
    i = int(np.argmin(np.abs(t_arr - t_query)))
    gap = float(abs(t_arr[i] - t_query))
    if gap > max_gap_s:
        return None, gap
    if t_arr.size == 1:
        return float(v_arr[0]), gap
    return float(np.interp(t_query, t_arr, v_arr)), gap


def stats(diff):
    d = np.asarray([x for x in diff if x is not None], float)
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "mean": float(d.mean()),
            "rms": float(np.sqrt(np.mean(d ** 2))),
            "max_abs": float(np.abs(d).max()),
            "min": float(d.min()), "max": float(d.max())}


def _plot_three_readings(plt, path, parity, asis, legs, arm_diff, u_exp, u_num):
    """三路并列 + 两臂差 + 分母轨迹,四格一张图。

    A 采纳口径(code-to-code 腿)  B 双色合成(code-to-experiment 腿)
    C 两臂差三路分列(一等输出)  D 分母 n_hot 与场峰温 —— 形状从哪来
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
    # 横轴锁在 Fig 14 的观测窗上下各留 60 ms:全程 0-1.65 s 里有意义的只有这一段,
    # 画全程等于把要判读的东西压成一根竖线。
    t_lo = min(r["time_s"] for r in legs) - 0.06
    t_hi = max(r["time_s"] for r in legs) + 0.06
    axes[0][0].set_xlim(t_lo, t_hi)
    ex_t = [r["time_s"] for r in legs]
    ex_v = [r["experimental_K"] - C2K for r in legs]
    ba_v = [r["balbaa_numerical_K"] - C2K for r in legs]

    def curves(ax, key, title, note):
        for doc_, colour, lab in ((asis, "tab:blue", "ours: as-is"),
                                  (parity, "tab:red", "ours: keff parity")):
            # 淡线 = 全部箱(含光束出圆的箱)。它的锯齿就是拍频本身,画出来
            # 而不是插值掉 —— 判读的人必须看见采样相位在做什么。
            t_all, v_all = series_arrays(doc_, key)
            if t_all.size:
                ax.plot(t_all, v_all - C2K, "-", lw=0.7, alpha=0.35, color=colour)
            t, v = series_arrays(doc_, key, beam_bins_only=True)
            if t.size:
                ax.plot(t, v - C2K, "-o", ms=3.5, lw=1.3, color=colour, label=lab)
        ax.errorbar(ex_t, ex_v, yerr=u_exp or 0.0, fmt="s", ms=8, mfc="none",
                    color="k", label="Balbaa Fig 14 experiment (digitized)")
        ax.errorbar(ex_t, ba_v, yerr=u_num or 0.0, fmt="^", ms=8, mfc="none",
                    color="tab:orange", label="Balbaa Fig 14 ABAQUS (digitized)")
        ax.axhline(1000.0, ls=":", color="gray", lw=1)
        ax.axhline(3000.0, ls="--", color="tab:green", lw=1)
        ax.set_ylabel("Temperature (degC)")
        ax.set_title(title, fontsize=11)
        ax.text(0.015, 0.97, note, transform=ax.transAxes, fontsize=7.5,
                va="top", color="0.25")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)

    curves(axes[0][0], "avg_K",
           "A  ADOPTED reading (Balbaa Sec 3.3 conditional average)",
           "his protocol, kept verbatim -> this is the code-to-code leg\n"
           "dotted 1000 degC = pyrometer floor; dashed 3000 degC = ceiling")
    curves(axes[0][1], "two_colour_K",
           "B  TWO-COLOUR synthetic instrument (radiance ratio, grey body)",
           "instrument physics -> this is the code-to-experiment leg\n"
           "faint sawtooth = beat aliasing (7.69 ms frames vs 15.10 ms track)\n"
           "above the dashed ceiling the real gauge could not have reported")

    ax = axes[1][0]
    for (_key, short, label, _use), colour in zip(
            READINGS, ("tab:purple", "tab:brown", "tab:olive")):
        rows = arm_diff["by_reading"][short]["per_bin"]
        t = [b["t_center_s"] for b in rows if b["parity_minus_asis_K"] is not None]
        v = [b["parity_minus_asis_K"] for b in rows
             if b["parity_minus_asis_K"] is not None]
        if t:
            ax.plot(t, v, "-", lw=1.3, color=colour, label=f"{short}")
    ax.axhline(0.0, ls="-", color="k", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("parity - as-is (K)")
    ax.set_title("C  FIRST-CLASS OUTPUT: arm-to-arm difference, per reading",
                 fontsize=11)
    ax.text(0.015, 0.03, "keff changes how hot the core is; the three readings\n"
                         "weigh that core very differently", transform=ax.transAxes,
            fontsize=8, va="bottom", color="0.25")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    s = [b for b in asis["series"] if b["avg_K"] is not None]
    t = [b["t_center_s"] for b in s]
    ax.plot(t, [b["mean_n_hot"] for b in s], "-o", ms=3, color="tab:blue",
            label="n_hot (adopted denominator), as-is")
    ax.axhline(asis.get("gauge_cells") or 0, ls="--", color="tab:blue", lw=1,
               label=f"cells in circle = {asis.get('gauge_cells')}")
    ax.set_yscale("log")
    ax.set_ylabel("cells above 1000 degC")
    ax.set_xlabel("Time (s)")
    ax2 = ax.twinx()
    ax2.plot(t, [b["max_K"] - C2K for b in s], "-", lw=1.3, color="tab:red",
             label="peak cell temperature, as-is")
    ax2.set_ylabel("peak cell T (degC)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("D  Where the shape comes from: denominator vs field",
                 fontsize=11)
    ax.text(0.015, 0.03, "the field is flat; the reading falls because the\n"
                         "denominator grows -- see A against D",
            transform=ax.transAxes, fontsize=8, va="bottom", color="0.25")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("V2 thermal gate re-read: THREE PARALLEL READINGS of the same "
                 "production fields (zero GPU, no re-run)\n"
                 "220 W / 650 mm/s / hatch 0.12 mm / layer 0.04 mm", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", type=Path, required=True, help="keff parity 臂 JSON")
    ap.add_argument("--asis", type=Path, required=True, help="as-is 臂 JSON")
    ap.add_argument("--experiment", type=Path, required=True, help="Fig 14 数字化 JSON")
    ap.add_argument("--max-gap-ms", type=float, default=10.0,
                    help="取样时允许的最近有效箱时间间隔上限")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--table", type=Path, default=None,
                    help="三路并列 + 两臂差的 CSV 表(逐箱一行)")
    args = ap.parse_args()

    parity, asis, fig14 = load(args.parity), load(args.asis), load(args.experiment)
    max_gap = args.max_gap_ms * 1e-3

    tp, vp = series_arrays(parity)
    ta, va = series_arrays(asis)
    has_three = any("two_colour_K" in s for s in parity["series"])

    # ---------------- 一等输出:两臂之差 ----------------
    pbins = {s["bin_index"]: s for s in parity["series"]}
    abins = {s["bin_index"]: s for s in asis["series"]}
    shared = sorted(set(pbins) & set(abins))
    # 逐箱相减的前提是两臂箱号同锚(analyze_pyrometer.py 锚在 t=0)。
    # 万一以后有人改了锚点,这里当场炸,而不是安静地错位一格。
    for idx in shared:
        if abs(pbins[idx]["t_center_s"] - abins[idx]["t_center_s"]) > 1e-9:
            raise SystemExit(
                f"两臂箱 {idx} 的中心时刻不一致 "
                f"({pbins[idx]['t_center_s']} vs {abins[idx]['t_center_s']}):"
                "箱号锚点不同,逐箱相减无效")
    per_bin, both_valid = [], []
    for idx in shared:
        p, a = pbins[idx]["avg_K"], abins[idx]["avg_K"]
        d = (p - a) if (p is not None and a is not None) else None
        per_bin.append({"bin_index": idx, "t_center_s": pbins[idx]["t_center_s"],
                        "parity_K": p, "asis_K": a, "parity_minus_asis_K": d})
        both_valid.append(d)
    arm_diff = {
        "_why_first": "Marangoni 升级决策(issue metadata,DEFERRED)的直接输入",
        "shared_bins": len(shared),
        "bins_both_valid": sum(1 for d in both_valid if d is not None),
        "parity_minus_asis_K": stats(both_valid),
        "peak_bin_avg_K": {"parity": parity.get("peak_avg_K"),
                           "asis": asis.get("peak_avg_K")},
        "peak_cell_K": {"parity": parity.get("peak_K"), "asis": asis.get("peak_K")},
        "per_bin": per_bin,
    }
    for field, node in (("peak_bin_avg_K", arm_diff["peak_bin_avg_K"]),
                        ("peak_cell_K", arm_diff["peak_cell_K"])):
        if node["parity"] is not None and node["asis"] is not None:
            node["parity_minus_asis"] = node["parity"] - node["asis"]

    # 两臂差按三路分别给:keff 改变的是"核心有多热",而三路对核心的敏感度
    # 完全不同(采纳口径被 1/n_hot 稀释,双色由最热的那一瞬主导)。只报一路
    # 的两臂差,等于把 Marangoni 决策的输入压成一个数。
    if has_three:
        by_reading = {}
        for key, short, label, _use in READINGS:
            diffs, rows = [], []
            for idx in shared:
                p, a = pbins[idx].get(key), abins[idx].get(key)
                d = (p - a) if (p is not None and a is not None) else None
                diffs.append(d)
                rows.append({"bin_index": idx, "t_center_s": pbins[idx]["t_center_s"],
                             "parity_K": p, "asis_K": a, "parity_minus_asis_K": d})
            by_reading[short] = {"label": label, "field": key,
                                 "parity_minus_asis_K": stats(diffs),
                                 "per_bin": rows}
        arm_diff["by_reading"] = by_reading
        arm_diff["_by_reading_why"] = (
            "keff 改变的是核心有多热;三路对核心的敏感度完全不同(采纳口径被 "
            "1/n_hot 稀释,双色由最热的那一瞬主导),所以两臂差必须三路都给")

    # ---------------- 三角形的另外两条腿 ----------------
    legs = []
    for pair in fig14["pairs"]:
        t = pair["time_s"]
        exp_K = pair["experimental_C"] + C2K
        bal_K = pair["balbaa_numerical_C"] + C2K
        row = {"time_s": t, "experimental_K": exp_K, "balbaa_numerical_K": bal_K,
               "balbaa_minus_experiment_K": bal_K - exp_K}
        for tag, (t_arr, v_arr) in (("parity", (tp, vp)), ("asis", (ta, va))):
            val, gap = sample_at(t_arr, v_arr, t, max_gap)
            row[f"{tag}_K"] = val
            row[f"{tag}_gap_s"] = gap
            row[f"{tag}_minus_experiment_K"] = (val - exp_K) if val is not None else None
            row[f"{tag}_minus_balbaa_K"] = (val - bal_K) if val is not None else None
        # 三路并列:同样的 5 个时刻、同样的取样规则,只换读数定义
        if has_three:
            for key, short, _label, _use in READINGS:
                for tag, doc_ in (("parity", parity), ("asis", asis)):
                    t_arr, v_arr = series_arrays(doc_, key, beam_bins_only=True)
                    val, gap = sample_at(t_arr, v_arr, t, max_gap)
                    row[f"{short}_{tag}_K"] = val
                    row[f"{short}_{tag}_minus_experiment_K"] = (
                        (val - exp_K) if val is not None else None)
                    row[f"{short}_{tag}_minus_balbaa_K"] = (
                        (val - bal_K) if val is not None else None)
                    row[f"{short}_{tag}_aliasing_envelope"] = aliasing_envelope(
                        doc_, key, t, max_gap)
        legs.append(row)

    # 数字化读数不确定度:实测腿的残差要和它比
    u_exp = max((p["temperature_C_uncertainty"] for p in fig14["experimental"]),
                default=None)
    u_num = max((p["temperature_C_uncertainty"] for p in fig14["balbaa_numerical"]),
                default=None)

    summary = {
        "code_to_experiment": {
            "parity": stats([r["parity_minus_experiment_K"] for r in legs]),
            "asis": stats([r["asis_minus_experiment_K"] for r in legs]),
            "balbaa_own_abaqus": stats([r["balbaa_minus_experiment_K"] for r in legs]),
            "_yardstick": "我们的残差与 Balbaa 自己的 ABAQUS 残差比,不与零比"
                          "(SPEC Acceptance framing:没有可调阈值)",
        },
        "code_to_code": {
            "parity": stats([r["parity_minus_balbaa_K"] for r in legs]),
            "asis": stats([r["asis_minus_balbaa_K"] for r in legs]),
        },
        "digitization_read_off_uncertainty_C": {
            "experimental_max": u_exp, "balbaa_numerical_max": u_num,
            "_note": "小于该值的残差差异没有判读意义",
        },
    }

    if has_three:
        three = {}
        for key, short, label, use in READINGS:
            three[short] = {
                "label": label, "field": key, "serves": use,
                "code_to_experiment": {
                    tag: stats([r[f"{short}_{tag}_minus_experiment_K"] for r in legs])
                    for tag in ("parity", "asis")},
                "code_to_code": {
                    tag: stats([r[f"{short}_{tag}_minus_balbaa_K"] for r in legs])
                    for tag in ("parity", "asis")},
            }
        three["_sampling"] = (
            "三路在 Fig 14 五时刻统一取采纳口径也有效的箱(光束在圆内的箱)。"
            "双色路每个箱都有值,若跨出圆箱插值会得到既非场也非仪器读数的中间量"
            "(例:t=0.5923 在 907 K 与 3703 K 之间插出 2939 K)。"
            "单元级仍然无阈值;拍频曝光量由 aliasing_envelope 单独上报")
        three["_aliasing"] = {
            "_what": "±10 ms 内该读数的包络。拍频(帧距 7.69 ms vs 道周期 "
                     "15.10 ms)使'这一刻读到多少'依赖采样相位",
            "per_time": [
                {"time_s": r["time_s"],
                 **{f"{s}_{a}": r.get(f"{s}_{a}_aliasing_envelope")
                    for _k, s, _l, _u in READINGS for a in ("asis", "parity")}}
                for r in legs],
            "_fix": "IET-20 交付件 3 的在线观测量记录器(dt 级,76.9 us)让这个包络塌缩",
        }
        three["_yardsticks"] = {
            "code_to_experiment": (
                "Balbaa 自己的 ABAQUS-vs-实测 RMS 142 K / 最差 232 K,而正文称之为"
                " good agreement。这是 code-to-experiment 腿的尺子"),
            "code_to_code": "他自己的 5 个数值点;数字化读数不确定度 ±2.5 degC 是下限",
            "_which_reading_for_which_leg": (
                "code-to-code 只该看 adopted(那是他的协议);code-to-experiment "
                "只该看 two_colour(那是仪器的物理)。交叉着看会把稀释伪迹读成一致"),
        }
        summary["three_readings"] = three
        summary["denominator_trace"] = {
            "_why": "采纳口径的形状主要由分母决定;把分母本身列出来,判读时"
                    "才看得见哪一段读数是被稀释出来的",
            "asis": [{"t_center_s": s["t_center_s"], "mean_n_hot": s["mean_n_hot"],
                      "avg_K": s["avg_K"], "two_colour_K": s.get("two_colour_K"),
                      "full_spot_avg_K": s.get("full_spot_avg_K"),
                      "max_K": s["max_K"]}
                     for s in asis["series"] if s["avg_K"] is not None],
            "gauge_cells_in_top_layer_circle": asis.get("gauge_cells"),
        }

    doc = {
        "id": "V2-thermal-gate-comparison",
        "condition": fig14.get("condition"),
        "arms": {"parity": {"json": str(args.parity), "run": parity.get("run_dir"),
                            "protocol": parity.get("protocol")},
                 "asis": {"json": str(args.asis), "run": asis.get("run_dir"),
                          "protocol": asis.get("protocol")}},
        "experiment_source": fig14.get("provenance"),
        "scope_finding": fig14.get("SCOPE_FINDING"),
        "ARM_DIFFERENCE_FIRST_CLASS": arm_diff,
        "triangle_at_figure_sample_times": legs,
        "summary": summary,
        "warnings": (parity.get("warnings", []) + asis.get("warnings", [])),
        "no_threshold_note": "本脚本不判 pass/fail:零标定,差异如实上报",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print("=== 热闸门对比 ===")
    d = arm_diff["parity_minus_asis_K"]
    print(f"[一等输出] 两臂差 parity - asis:共享箱 {arm_diff['shared_bins']},"
          f"双方有效 {arm_diff['bins_both_valid']}")
    if d.get("n"):
        print(f"  均值 {d['mean']:+.1f} K  RMS {d['rms']:.1f} K  "
              f"最大绝对 {d['max_abs']:.1f} K  区间 [{d['min']:+.1f}, {d['max']:+.1f}] K")
    for k in ("peak_bin_avg_K", "peak_cell_K"):
        n = arm_diff[k]
        if n.get("parity_minus_asis") is not None:
            print(f"  {k}: parity {n['parity']:.1f} - asis {n['asis']:.1f} "
                  f"= {n['parity_minus_asis']:+.1f} K")
    print(f"\n[三角形] 在 Fig 14 的 {len(legs)} 个采样时刻(K):")
    print(f"  {'t [s]':>7} {'实测':>8} {'Balbaa':>8} {'parity':>8} {'asis':>8}"
          f" {'par-exp':>9} {'asis-exp':>9} {'Bal-exp':>9}")
    for r in legs:
        fmt = lambda v: f"{v:8.1f}" if v is not None else "     n/a"
        fmt9 = lambda v: f"{v:+9.1f}" if v is not None else "      n/a"
        print(f"  {r['time_s']:7.4f} {r['experimental_K']:8.1f} "
              f"{r['balbaa_numerical_K']:8.1f} {fmt(r['parity_K'])} {fmt(r['asis_K'])}"
              f" {fmt9(r['parity_minus_experiment_K'])}"
              f" {fmt9(r['asis_minus_experiment_K'])}"
              f" {fmt9(r['balbaa_minus_experiment_K'])}")
    print("\n[汇总] 对实测的残差 RMS / 最大绝对 (K):")
    for tag, s in summary["code_to_experiment"].items():
        if isinstance(s, dict) and s.get("n"):
            print(f"  {tag:18s} n={s['n']}  RMS {s['rms']:8.1f}  max|.| {s['max_abs']:8.1f}")
    # code-to-code 腿以前只进 JSON,判读时容易漏(Kimi 2026-08-19)
    print("[汇总] 对 Balbaa 数值点的残差(code-to-code)RMS / 最大绝对 (K):")
    for tag, s in summary["code_to_code"].items():
        if isinstance(s, dict) and s.get("n"):
            print(f"  {tag:18s} n={s['n']}  RMS {s['rms']:8.1f}  max|.| {s['max_abs']:8.1f}")
    print(f"  数字化读数不确定度 +/-{u_exp:.1f} degC(实测点):小于此的差异无判读意义")

    if has_three:
        print("\n[三路并列] Fig 14 五时刻的读数 (K):")
        print(f"  {'t [s]':>7} {'实测':>8} {'Balbaa':>8}"
              f" {'采纳-as':>8} {'采纳-par':>8} {'双色-as':>8} {'双色-par':>8}"
              f" {'全斑-as':>8} {'全斑-par':>8}")
        f8 = lambda v: f"{v:8.1f}" if v is not None else "     n/a"
        for r in legs:
            print(f"  {r['time_s']:7.4f} {r['experimental_K']:8.1f} "
                  f"{r['balbaa_numerical_K']:8.1f} "
                  + " ".join(f8(r.get(f"{s}_{a}_K"))
                             for _k, s, _l, _u in READINGS
                             for a in ("asis", "parity")))
        print("\n[三路并列] 残差 RMS (K) —— 每条腿只该看它自己的那一路:")
        print(f"  {'读数':<14} {'服务于':<34} {'vs实测 as':>10} {'vs实测 par':>11}"
              f" {'vs他数值 as':>12} {'vs他数值 par':>13}")
        for _key, short, label, use in READINGS:
            n = summary["three_readings"][short]
            g = lambda leg, tag: (f"{n[leg][tag]['rms']:.1f}"
                                  if n[leg][tag].get("n") else "n/a")
            print(f"  {label:<14} {use:<34} {g('code_to_experiment','asis'):>10}"
                  f" {g('code_to_experiment','parity'):>11}"
                  f" {g('code_to_code','asis'):>12}"
                  f" {g('code_to_code','parity'):>13}")
        print(f"  {'(尺子)':<14} {'Balbaa 自己的 ABAQUS vs 实测':<34}"
              f" {summary['code_to_experiment']['balbaa_own_abaqus']['rms']:>10.1f}")
        print("\n[两臂差] parity - asis,按读数分列 (K):")
        for _key, short, label, _use in READINGS:
            d = arm_diff["by_reading"][short]["parity_minus_asis_K"]
            if d.get("n"):
                print(f"  {label:<14} n={d['n']:<4d} 均值 {d['mean']:+8.1f}  "
                      f"RMS {d['rms']:7.1f}  最大绝对 {d['max_abs']:7.1f}")

    for w in doc["warnings"]:
        print(f"  [WARN] {w}")
    print(f"\n写出 {args.output}")

    if args.table:
        # 逐箱一行:三路 x 两臂 + 两臂差 + 分母。判读时用表,不用从 JSON 里挖。
        cols = ["bin_index", "t_center_s", "n_frames", "any_laser_on",
                "mean_n_hot_asis", "mean_n_hot_parity",
                "max_K_asis", "max_K_parity"]
        for _key, short, _label, _use in READINGS:
            cols += [f"{short}_asis_K", f"{short}_parity_K",
                     f"{short}_parity_minus_asis_K"]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        with open(args.table, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for idx in shared:
                pb, ab = pbins[idx], abins[idx]
                row = {"bin_index": idx, "t_center_s": ab["t_center_s"],
                       "n_frames": ab["n_frames"],
                       "any_laser_on": ab["any_laser_on"],
                       "mean_n_hot_asis": ab["mean_n_hot"],
                       "mean_n_hot_parity": pb["mean_n_hot"],
                       "max_K_asis": ab["max_K"], "max_K_parity": pb["max_K"]}
                for key, short, _label, _use in READINGS:
                    a, p = ab.get(key), pb.get(key)
                    row[f"{short}_asis_K"] = a
                    row[f"{short}_parity_K"] = p
                    row[f"{short}_parity_minus_asis_K"] = (
                        (p - a) if (a is not None and p is not None) else None)
                w.writerow(row)
        print(f"写出 {args.table}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] 没装 matplotlib,跳过出图(JSON 已写出)")
            return 0
        if has_three:
            _plot_three_readings(plt, args.plot, parity, asis, legs, arm_diff,
                                 u_exp, u_num)
            print(f"写出 {args.plot}")
            return 0
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        if tp.size:
            ax.plot(tp, vp - C2K, "-", lw=1.2, color="tab:red", label="ours: keff parity")
        if ta.size:
            ax.plot(ta, va - C2K, "-", lw=1.2, color="tab:blue", label="ours: as-is")
        ex_t = [r["time_s"] for r in legs]
        ex_v = [r["experimental_K"] - C2K for r in legs]
        ba_v = [r["balbaa_numerical_K"] - C2K for r in legs]
        ax.errorbar(ex_t, ex_v, yerr=u_exp or 0.0, fmt="s", ms=8, mfc="none",
                    color="k", label="Balbaa Fig 14 experiment (digitized)")
        ax.errorbar(ex_t, ba_v, yerr=u_num or 0.0, fmt="^", ms=8, mfc="none",
                    color="tab:orange", label="Balbaa Fig 14 ABAQUS (digitized)")
        ax.axhline(1000.0, ls=":", color="gray", lw=1)
        ax.set_ylabel("Temperature (degC)")
        ax.set_title("V2 thermal gate: keff two-arm vs two-colour pyrometer\n"
                     "220 W / 650 mm/s / hatch 0.12 mm / layer 0.04 mm")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        dt_ = [b["t_center_s"] for b in per_bin if b["parity_minus_asis_K"] is not None]
        dv_ = [b["parity_minus_asis_K"] for b in per_bin
               if b["parity_minus_asis_K"] is not None]
        ax.plot(dt_, dv_, "-", color="tab:purple", lw=1.2)
        ax.axhline(0.0, ls="-", color="k", lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("parity - as-is (K)")
        ax.set_title("FIRST-CLASS OUTPUT: arm-to-arm difference", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"写出 {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
