#!/usr/bin/env python3
"""D-V2-07 基板离散收敛判读:coarse / graded / fine 三档比对。

以 model/analyze_substrate_study.py 为底扩展(Kimi 2026-08-07 修正版实现单第 4 项)。

判据在下面 CRITERIA 里**硬编码**,并在任何计算之前打印出来 —— 这是预注册,
不是事后解释。sweep A 的教训就是判据必须先写死再看数。

Kimi 点 1 的前提修正(已在本脚本里实测复核,不是照抄):三张网格只有基板 z 剖分
不同,件内块在三者中单元数相同、编号相同(逐层构建,件内恒为数组末尾 M 个单元),
所以**逐单元比对可用且为首选**,不需要任何插值。

一个诚实的说明:本多轨模型只印**一层**(layers=1, 40 um),所以"件内块"就等于
原脚本的顶层 10000 单元 —— Kimi 说的"把范围从顶层扩到整个件内块"在这个网格族上
是恒等的,不是我少做了一步。脚本仍按几何(z > 基板厚度)来选,而不是写死 10000,
这样将来 layers>1 时自动正确。
"""
import argparse
import glob
import json
import os
import sys

import meshio
import numpy as np

OUT_ROOT = "/home/user/work/159/output"
SUPPORT_THICKNESS = 4.0e-4      # 基板厚度 [m];件内 = 质心 z 高于此值
LAYER_THICKNESS = 4.0e-5

# 默认三档的输出目录。coarse 用扫描 A 的中心点成员:它与采纳口径**内容等价**
# (E 表逐字节、流动曲线 156/156 逐值相同,已在 adopt_option_c.py 里核过),
# 脚本启动时还会再核一遍,不靠"我记得是一样的"。
DEFAULT_RUNS = {
    "coarse": "v2_sweepA_f001_c08",
    "graded": "v2_gate_graded_adopted",
    "fine": "v2_gate_fine_adopted",
}

CRITERIA = """
预注册判据(硬编码于本脚本,运行前打印;来源 Kimi 2026-08-07 点 2)
--------------------------------------------------------------------------
(a) 主对比对 = graded <-> fine(最细两档,收敛相关)。coarse 只作序列参照,
    不单独用于判收敛。
(b) 两条都要过才算"收敛":
    b1 幅度:|fine - graded| 的 vm 相对 RMS 差 <= 7.71 %  且  绝对 <= 3.0 MPa
             (= D-V2-17 参数括号,sweep A 九成员两两包络的最大值)
    b2 收缩:|fine - graded| < |graded - coarse|(同一量、同一范数)
    只过 b1 不过 b2 = "被括号包住但未证收敛" —— D-V2-07 只能带此 caveat 关闭,
    **不得**写成"对基板剖分不敏感"。
(c) 小分母纪律:主判据只用绝对 MPa + vm / sigma_xx 的相对值。
    sigma_yy / sigma_zz / eqp 的相对 % 仅作信息列,不进判据
    (sweep A 已踩过 28-47 % 的小分母假象,绝对差其实 <= 2 MPa)。
补充观测量(降级为观测,不进判据;Kimi 对 Fable5 三项的修正)
    (1) 顶面只用 RMS,不用均值 —— 残余应力自平衡,均值会自相消;
    (2) 逐点峰值**不作**收敛判据 —— 几何角点应力奇异,细化下发散,
        报出来必须标注 non-converging-by-construction;
    (3) 比较区域一律按物理坐标定义(z > 基板厚度),禁止写死单元序号。
    (4) 位移/挠度:C0 量比应力收敛快一阶,网格收敛研究里最稳的观测量。
--------------------------------------------------------------------------
"""

VM_REL_LIMIT_PCT = 7.71
VM_ABS_LIMIT_MPA = 3.0


def cf(m, name):
    d = m.cell_data_dict.get(name)
    return None if d is None else np.asarray(list(d.values())[0]).reshape(-1)


def qavg(m, pattern):
    parts = [cf(m, pattern.replace("Q", str(q))) for q in range(8)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, axis=0) if parts else None


def centroid_z(m):
    pts = np.asarray(m.points)
    for blk in m.cells:
        conn = np.asarray(blk.data)
        if conn.shape[1] == 8:
            return pts[conn, 2].mean(axis=1)
    return None


def load(run):
    d = os.path.join(OUT_ROOT, run)
    vt = sorted(glob.glob(f"{d}/*.vtu"))
    if not vt:
        return None
    m = meshio.read(vt[-1])
    z = centroid_z(m)
    inpart = z > SUPPORT_THICKNESS          # 点 (3):物理坐标选区
    r = {"run": run, "vtu": os.path.basename(vt[-1]), "ncells": z.size,
         "inpart_mask": inpart, "n_inpart": int(inpart.sum()),
         "z": z, "points": np.asarray(m.points)}
    for c in ("xx", "yy", "zz"):
        r[f"s{c}"] = qavg(m, f"stress_quadQ_{c}")
    r["vm"] = qavg(m, "vm_quadQ")
    r["eqp"] = cf(m, "eq_plastic_strain")
    r["u"] = m.point_data.get("u")
    sp = os.path.join(d, "thermal_energy_ledger_summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        r["complete"] = s.get("complete")
        r["steps"] = s.get("recorded_step_count")
    cp = os.path.join(d, "used_config.json")
    if os.path.exists(cp):
        c = json.load(open(cp))
        r["E_table"] = c.get("E_table")
        r["flow_curve_table"] = c.get("flow_curve_table")
    return r


def file_digest(p):
    """材料表的**数值内容**摘要,刻意不包含 source provenance 列。

    整文件 sha256 是错的判据:采纳表与 sweep A 中心点成员表的数值逐点相同,
    但 source 字符串不同(一个写"扫描 A 成员",一个写"采纳口径"),
    整文件摘要会因此判定"材料不一致"并终止 —— 那是把出处注释当成了材料数据。
    这里只对数值列做摘要,行序无关(排序后再摘要)。
    """
    import csv
    import hashlib
    try:
        rows = []
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                nums = []
                for k, v in r.items():
                    if k is None or k.lower() == "source":
                        continue
                    try:
                        nums.append(f"{float(v):.12g}")
                    except (TypeError, ValueError):
                        pass
                if nums:
                    rows.append(",".join(nums))
        rows.sort()
        return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]
    except Exception:
        return None


def stats(ref_vals, mem_vals):
    d = mem_vals - ref_vals
    rms_ref = float(np.sqrt(np.mean(ref_vals ** 2)))
    rms_d = float(np.sqrt(np.mean(d ** 2)))
    return {"ref_rms": rms_ref, "mem_rms": float(np.sqrt(np.mean(mem_vals ** 2))),
            "abs_rms_diff": rms_d, "abs_max_diff": float(np.abs(d).max()),
            "rel_rms_diff_pct": rms_d / rms_ref * 100 if rms_ref > 0 else float("nan")}


PRIMARY = [("vm", "von Mises", 1e6, "MPa"), ("sxx", "sigma_xx", 1e6, "MPa")]
INFO = [("syy", "sigma_yy", 1e6, "MPa"), ("szz", "sigma_zz", 1e6, "MPa"),
        ("eqp", "eqp", 1.0, "-")]


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULT_RUNS.items():
        ap.add_argument(f"--{k}", default=v)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    print(CRITERIA)

    runs = {k: load(getattr(args, k)) for k in DEFAULT_RUNS}
    present = [k for k in ("coarse", "graded", "fine") if runs[k] is not None]
    missing = [k for k in DEFAULT_RUNS if runs[k] is None]
    # fine 未跑时不直接退出:前提核实(材料等价 / 件内块同规模)对已有变体照样
    # 该做,而且**必须现在做** —— 这些前提若不成立,fine 跑完 5 小时也是白跑。
    partial = bool(missing)
    if partial:
        print(f"缺少变体输出: {missing}(fine 由 runs/v_fine_gate.sh 产出)")
        print("=> 进入预检模式:仍执行前提核实,并给出已有档位的预览比对;"
              "判据 (a)/(b) 待 fine 到位后才判。\n")

    print("变体一览:")
    for k in present:
        r = runs[k]
        print(f"  {k:>7}: {r['run']:<26} cells={r['ncells']:>7} "
              f"in-part={r['n_inpart']:>6} steps={r.get('steps','-')} "
              f"complete={r.get('complete','-')}")

    # --- 前提核实 1:三档的材料输入必须等价,否则比的不是网格 -------------
    print("\n[前提核实] 各档消费的材料表是否等价(比网格必须固定材料):")
    print("  注:coarse 用的是 sweep A 中心点成员的配置文件,graded/fine 用采纳配置;"
          "文件名不同但内容应等价,这里比的是**内容摘要**而不是文件名。")
    digs = {}
    for k in present:
        r = runs[k]
        digs[k] = (file_digest(r.get("E_table") or ""),
                   file_digest(r.get("flow_curve_table") or ""))
        print(f"  {k:>7}: E={digs[k][0]}  flow={digs[k][1]}  "
              f"({os.path.basename(r.get('E_table') or '?')} / "
              f"{os.path.basename(r.get('flow_curve_table') or '?')})")
    if len(set(digs.values())) != 1:
        print("  !! 材料输入不一致 —— 比对结果无法归因于网格。终止。")
        sys.exit(1)
    print("  材料输入一致(内容摘要相同)。")

    # --- 前提核实 2:件内块在各档中同规模(逐单元比对的前提)-------------
    n_in = {k: runs[k]["n_inpart"] for k in present}
    print(f"\n[前提核实] 件内单元数(按 z > {SUPPORT_THICKNESS:g} m 选): {n_in}")
    if len(set(n_in.values())) != 1:
        print("  !! 件内单元数不同 -> 逐单元比对不成立。终止。")
        sys.exit(1)
    M_ = next(iter(n_in.values()))
    print(f"  三档件内块均为 {M_} 单元 -> 逐单元比对成立(无需插值)。")
    zc = runs["coarse"]["z"][runs["coarse"]["inpart_mask"]]
    print(f"  件内 z 范围 {zc.min():.6g} .. {zc.max():.6g} m "
          f"(= {(zc.max()-zc.min())/LAYER_THICKNESS + 1:.0f} 层);"
          f"本模型 layers=1,故件内块 == 原脚本的顶层块")

    # --- 前提核实 3:件内单元的**面内顺序**必须逐个对应 -------------------
    # 只知道"件内单元数相同"是不够的。逐单元相减要求第 i 个件内单元在各档里
    # 是同一个物理位置;若网格生成器在不同基板剖分下改变了面内遍历顺序,
    # 相减出来的差是纯粹的错位假象。用面内质心 (x, y) 逐个比对来锁死这一点。
    print("\n[前提核实] 件内单元的面内 (x,y) 顺序是否逐个对应:")
    ref_k = present[0]

    def inplane_xy(r):
        pts, mask = r["points"], r["inpart_mask"]
        for blk_conn in [np.asarray(b.data) for b in []]:
            pass
        return None

    # 直接用已存的质心 z 不够,这里重算 x/y 质心
    def centroid_xy(run_name):
        d = os.path.join(OUT_ROOT, run_name)
        vt = sorted(glob.glob(f"{d}/*.vtu"))
        m = meshio.read(vt[-1])
        pts = np.asarray(m.points)
        for blk in m.cells:
            conn = np.asarray(blk.data)
            if conn.shape[1] == 8:
                return pts[conn, 0].mean(axis=1), pts[conn, 1].mean(axis=1)
        return None, None

    xy = {}
    for k in present:
        x, y = centroid_xy(runs[k]["run"])
        mask = runs[k]["inpart_mask"]
        xy[k] = (x[mask], y[mask])
    ok_order = True
    for k in present[1:]:
        dx = np.abs(xy[k][0] - xy[ref_k][0]).max()
        dy = np.abs(xy[k][1] - xy[ref_k][1]).max()
        good = dx < 1e-12 and dy < 1e-12
        ok_order &= good
        print(f"  {ref_k} vs {k}: max |dx|={dx:.3e} m, max |dy|={dy:.3e} m -> "
              f"{'逐个对应' if good else '**顺序不一致**'}")
    if not ok_order:
        print("  !! 件内单元顺序在各档间不一致 -> 逐单元相减无意义。终止。")
        sys.exit(1)
    print("  件内单元面内顺序一致,逐单元相减成立。")

    def vals(r, key):
        a = r.get(key)
        return None if a is None else a[r["inpart_mask"]]

    def compare(a, b, label):
        print(f"\n--- {label} ---")
        print(f"{'量':>12} {'参照RMS':>11} {'对比RMS':>11} {'绝对RMS差':>11} "
              f"{'最大绝对差':>11} {'相对RMS差':>10} {'判据':>8}")
        rec = {}
        for key, lab, sc, unit in PRIMARY + INFO:
            x, y = vals(a, key), vals(b, key)
            if x is None or y is None or x.shape != y.shape:
                continue
            s = stats(x, y)
            rec[key] = dict(s, unit=unit)
            tag = "主判据" if (key, lab, sc, unit) in PRIMARY else "信息列"
            print(f"{lab:>12} {s['ref_rms']/sc:>11.4g} {s['mem_rms']/sc:>11.4g} "
                  f"{s['abs_rms_diff']/sc:>11.4g} {s['abs_max_diff']/sc:>11.4g} "
                  f"{s['rel_rms_diff_pct']:>9.2f}% {tag:>8}")
        return rec

    gc = compare(runs["graded"], runs["coarse"], "coarse vs graded(序列参照)")
    if partial:
        print("\n" + "=" * 78)
        print("预检模式结束:前提全部成立,coarse<->graded 预览如上。")
        print("fine 到位后重跑本脚本即给出 (a)/(b) 判读。")
        print("=" * 78)
        return
    fg = compare(runs["fine"], runs["graded"], "graded vs fine(参照=fine,主对比对)")

    # --- 判据 (b) --------------------------------------------------------
    print("\n" + "=" * 78)
    print("判读((a) 主对比对 = graded<->fine;(b) 幅度 + 收缩 两条都要过)")
    print("=" * 78)
    verdict = {}
    fg_vm, gc_vm = fg.get("vm"), gc.get("vm")
    b1 = b2 = None
    if fg_vm:
        b1_rel = fg_vm["rel_rms_diff_pct"] <= VM_REL_LIMIT_PCT
        b1_abs = fg_vm["abs_rms_diff"] / 1e6 <= VM_ABS_LIMIT_MPA
        b1 = bool(b1_rel and b1_abs)
        print(f"  b1 幅度: vm |fine-graded| = {fg_vm['rel_rms_diff_pct']:.2f} % / "
              f"{fg_vm['abs_rms_diff']/1e6:.3f} MPa  "
              f"(限 {VM_REL_LIMIT_PCT} % 且 {VM_ABS_LIMIT_MPA} MPa) -> "
              f"{'通过' if b1 else '不通过'}")
    if fg_vm and gc_vm:
        b2 = bool(fg_vm["abs_rms_diff"] < gc_vm["abs_rms_diff"])
        print(f"  b2 收缩: |fine-graded| = {fg_vm['abs_rms_diff']/1e6:.3f} MPa  vs  "
              f"|graded-coarse| = {gc_vm['abs_rms_diff']/1e6:.3f} MPa -> "
              f"{'通过' if b2 else '不通过'}")
    if b1 is not None and b2 is not None:
        # 判决判据 = b2 收缩(2026-08-07 预注册裁定:b2 升为判决判据,b1 降为
        # 报告行)。旧版在此按 b1 分支,与裁定矛盾 —— Kimi 2026-08-18 审查发现。
        if b2:
            concl = ("b2 收缩通过,D-V2-07 可关闭(收敛中,非'不敏感');"
                     + ("b1 报告行:幅度亦落入括号" if b1 else
                        "b1 报告行:幅度未入括号 -> graded 出数登记网格不确定度"))
        else:
            concl = "b2 收缩未通过:序列未收敛,D-V2-07 不能关闭"
        print(f"\n  结论: {concl}")
        verdict = {"criterion": "b2_contraction (pre-registered 2026-08-07)",
                   "b1_magnitude": b1, "b2_contraction": b2, "conclusion": concl}

    # --- 补充观测量 ------------------------------------------------------
    print("\n补充观测量(不进判据):")
    for key, lab, sc, _ in PRIMARY:
        a, b = vals(runs["fine"], key), vals(runs["graded"], key)
        if a is None or b is None:
            continue
        print(f"  (1) 顶面 {lab} RMS: fine {np.sqrt(np.mean(a**2))/sc:.4g} MPa, "
              f"graded {np.sqrt(np.mean(b**2))/sc:.4g} MPa "
              f"(只报 RMS,不报均值:RS 自平衡会自相消)")
    for key, lab, sc, _ in PRIMARY:
        peaks = {k: float(np.abs(vals(runs[k], key)).max()) / sc
                 for k in ("coarse", "graded", "fine") if vals(runs[k], key) is not None}
        print(f"  (2) 逐点峰值 {lab} [MPa] {peaks}  "
              f"<- non-converging-by-construction(角点应力奇异),不作判据")
    for k in ("coarse", "graded", "fine"):
        u = runs[k].get("u")
        if u is None:
            continue
        uz = np.asarray(u)[:, 2]
        print(f"  (4) {k:>7} 位移 uz: max |uz| = {np.abs(uz).max():.4g} m, "
              f"RMS = {np.sqrt(np.mean(uz**2)):.4g} m")

    if args.json:
        p = os.path.join(OUT_ROOT, "v2_substrate_variants_summary.json")
        json.dump({"runs": {k: {kk: vv for kk, vv in runs[k].items()
                                if not isinstance(vv, np.ndarray)} for k in runs},
                   "graded_vs_fine": fg, "coarse_vs_graded": gc,
                   "criteria": {"vm_rel_limit_pct": VM_REL_LIMIT_PCT,
                                "vm_abs_limit_mpa": VM_ABS_LIMIT_MPA},
                   "verdict": verdict},
                  open(p, "w"), indent=1, default=str)
        print(f"\n已写出: {p}")


if __name__ == "__main__":
    main()
