#!/usr/bin/env python3
"""D-V2-21 parity 臂:Balbaa 熔池等效导热 keff 的预计算(Eq 19-24)。

Balbaa 用 FE 跑不了 Marangoni 流动,于是把静止液相导热率人为抬高:

    keff = kl + h L                                   (19)
    Nu   = h L / kl                                   (20)  => keff = kl (1 + Nu)
    Nu   = 1.6129 ln(Ma) - 10.183                     (21)
    Ma   = -(dsigma/dT) L dT rho Cpl / (mu kl)        (22)
    dT   = [-Cpl + sqrt(Cpl^2 + 2 a^2 dt Qv / rho)] / a^2,  a = (dsigma/dT)/mu   (23)
    dt   = L / v                                      (24)

L = 熔池半宽,本身是模型的**输出**(D-V1-10 登记的鸡生蛋问题)。本脚本
因此不猜 L:要么由 as-is 臂的实际熔池量出来(--from-run,一次 Picard),
要么显式给定(--half-width),并且总是连同 L / Qv 的敏感度括号一起登记 ——
keff 对 L 是 log 敏感但**不是**弱敏感(dNu/dlnL = 2 x 1.6129 = 3.226)。

产物是一张液相导热表(T,value,source),经 config 的 k_table_liquid 喂给
求解器 —— 不动共享求解器代码(D-V2-21 方案 (c))。

零标定:所有材料常数从 ../../v1-single-track/inputs/balbaa-model.json
(Balbaa Table 1 的逐值转写)读出,没有一个数向实测回调。
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
TABLE1 = V2.parent / "v1-single-track" / "inputs" / "balbaa-model.json"

# Eq 21 的定义域:Nu <= 0 处相关式失效(等效导热不该低于静止液相导热)
MA_CRIT = math.exp(10.183 / 1.6129)


def load_table1():
    """Balbaa Table 1 逐值溯源。缺一个就报错,不用默认值兜底。"""
    with open(TABLE1, encoding="utf-8") as f:
        doc = json.load(f)
    t1 = doc["table1_thermophysical"]

    def need(key, field="value"):
        node = t1[key]
        if field not in node:
            raise SystemExit(f"balbaa-model.json: {key} 缺 {field}")
        return float(node[field])

    return {
        "k_liquid": need("liquid_thermal_conductivity_W_mK"),
        "cp_liquid": need("liquid_specific_heat_J_kgK"),
        "mu": need("dynamic_viscosity_Pa_s"),
        "dsigma_dT": need("surface_tension_temperature_gradient"),
        "solidus_K": need("solidus_K"),
        "liquidus_K": need("liquidus_K"),
        "boiling_K": need("boiling_K"),
        "_provenance": t1["provenance"],
        # 仓库相对路径:绝对路径会把生成机器的目录结构化石进登记产物里
        "_source_file": "cases/AM-Benchmark/verification/v1-single-track/"
                        "inputs/balbaa-model.json",
    }


def qv_peak_exponential(power, absorptivity, beam_radius, opd):
    """Eq 18 指数衰减源在束心表面 (x=vt, y=h, z=0) 的峰值体功率密度。

    Qv = 2 Ac P / (pi r^2 d) * exp(-2 rho^2 / r^2) * exp(-|z| / d)
    Eq 23 的 dT 按 Balbaa 的定义是熔池**中心**峰值温度减固相线,故与之
    自洽的是峰值 Qv,不是某个平均。该选择登记在 deviations D-V2-21。
    """
    return 2.0 * absorptivity * power / (math.pi * beam_radius ** 2 * opd)


def delta_T(qv, dt, rho_liquid, cp_liquid, dsigma_dT, mu):
    """Eq 23。a = (dsigma/dT)/mu,量纲 m/(K s);a^2 dT^2/2 是 Marangoni 项。

    a -> 0 时退化为绝热升温 Qv dt / (rho Cpl),这里用该极限做数值兜底,
    避免小 a 下 (-Cpl + sqrt(...)) 的灾难性抵消。
    """
    a2 = (dsigma_dT / mu) ** 2
    energy = qv * dt / rho_liquid                 # J/kg
    if a2 * energy < 1e-9 * cp_liquid ** 2:
        return energy / cp_liquid
    disc = cp_liquid ** 2 + 2.0 * a2 * energy
    return (-cp_liquid + math.sqrt(disc)) / a2


def marangoni(dT, L, rho_liquid, cp_liquid, dsigma_dT, mu, k_liquid):
    """Eq 22。dsigma/dT 为负,Ma 取正。"""
    return abs(dsigma_dT) * L * dT * rho_liquid * cp_liquid / (mu * k_liquid)


def keff_at(L, qv, speed, props, rho_liquid):
    """一个 (L, Qv) 点上的完整推导链,返回中间量便于登记。"""
    if not (L > 0.0):
        # Eq 20 的 h = Nu kl / L 在 L=0 处除零。上游护栏应该先拦住,
        # 这里再兜一层,免得以后有人绕过 melt_half_width_from_run 直接调。
        raise SystemExit(f"特征长度 L 必须为正,收到 {L}")
    dt = L / speed                                                      # Eq 24
    dT = delta_T(qv, dt, rho_liquid, props["cp_liquid"],
                 props["dsigma_dT"], props["mu"])                       # Eq 23
    ma = marangoni(dT, L, rho_liquid, props["cp_liquid"],
                   props["dsigma_dT"], props["mu"], props["k_liquid"])  # Eq 22
    nu_raw = 1.6129 * math.log(ma) - 10.183 if ma > 0.0 else float("-inf")  # Eq 21
    nu, clamped = (nu_raw, False) if nu_raw > 0.0 else (0.0, True)
    h = nu * props["k_liquid"] / L                                      # Eq 20
    keff = props["k_liquid"] + h * L                                    # Eq 19
    return {
        "L_m": L, "interaction_time_s": dt, "Qv_W_m3": qv,
        "delta_T_K": dT, "peak_T_estimate_K": props["solidus_K"] + dT,
        "Marangoni": ma, "Nusselt_raw": nu_raw, "Nusselt": nu,
        "h_W_m2K": h, "keff_W_mK": keff,
        "keff_over_kl": keff / props["k_liquid"],
        "nusselt_clamped_to_zero": clamped,
    }


def source_run_fingerprint(run_dir):
    """as-is 臂运行的指纹,写进 keff 台账。

    F2(Fable5 2026-08-19):运行器的"文件已存在就跳过"会把一次**半途死掉**的
    as-is 臂推出的 keff 固化下来 —— 人修好问题重跑阶段 1,阶段 2 却因文件已在
    而跳过,parity 臂带着陈旧 keff 跑完全程。有了指纹,运行器可以逐字段比对,
    不匹配就重算。
    """
    if not run_dir:
        return None
    import hashlib

    summary = os.path.join(run_dir, "thermal_energy_ledger_summary.json")
    ledger = os.path.join(run_dir, "thermal_energy_ledger.jsonl")
    fp = {"run_dir": os.path.abspath(run_dir), "complete": None,
          "ledger_lines": None, "summary_sha256": None}
    if os.path.isfile(summary):
        raw = open(summary, "rb").read()
        fp["summary_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            fp["complete"] = json.loads(raw.decode("utf-8")).get("complete")
        except Exception:
            fp["complete"] = "unparseable"
    if os.path.isfile(ledger):
        with open(ledger, "rb") as f:
            fp["ledger_lines"] = sum(1 for _ in f)
    return fp


def _select_frame(run_dir, at_time):
    """选帧。缺省用末帧(曾熔区是累积量);给了 at_time 就取 **<= at_time 的最后
    一帧** —— 多道运行必须能在"第 N 道跑完、第 N+1 道还没到"的那一刻取样,
    否则曾熔场已经是相邻道的并集。时间由求解器自己的 path_used.csv 给。"""
    import csv as _csv

    vtus = sorted(glob.glob(os.path.join(run_dir, "*.vtu")))
    if not vtus:
        raise SystemExit(f"{run_dir} 内没有 VTU")
    if at_time is None:
        return vtus[-1], None
    times = {}
    pu = os.path.join(run_dir, "path_used.csv")
    if not os.path.isfile(pu):
        raise SystemExit(f"--from-run-time 需要 {pu} 来给帧配时间")
    with open(pu, newline="") as f:
        for row in _csv.DictReader(f):
            times[int(row["step"])] = float(row["time"])
    cands = []
    for v in vtus:
        m = re.findall(r"(\d+)", os.path.basename(v))
        t = times.get(int(m[-1])) if m else None
        if t is not None and t <= at_time + 1e-12:
            cands.append((t, v))
    if not cands:
        raise SystemExit(
            f"{run_dir}: 没有时间 <= {at_time:.6f} s 的输出帧。"
            "热输出太稀疏(调小 --thermal-output-every),或 at_time 早于首帧")
    t, v = max(cands)
    return v, t


def melt_half_width_from_run(run_dir, solidus, window=None,
                             at_time=None, track_y=None):
    """从 as-is 臂量熔池半宽:曾熔区(max_temperature_history >= 固相线)的
    横向半跨度。与 V1 analyze_v1.py 同一个固相线口径,不引入新约定。

    多道扫描下 y 跨度是整层而不是单道,所以多道运行**必须**用 window 把
    量测限制到一条道上,否则半宽被严重高估 —— 未给 window 时本函数拒绝返回。

    track_y 给定时改用**单侧(上半)**量法:L = 曾熔区上边缘 - 道中心线。
    这是 Fable5 2026-08-19 复审 F1 的修法。双侧量法在真实 M1 网格上有两个
    结构性缺陷:(1) 窗界 ±hatch/2 落在单元中心上,浮点决定端排取舍,量测
    不对称;(2) 熔池全宽(V1 自己预测 141 um)大于 hatch 120 um 时,双侧
    跨度被窗口截断,**上限恒为 hatch/2 = 60 um**,而且是静默的。单侧量法
    从道中心往未被相邻道污染的那一侧量,没有这个天花板,也不受域边裁剪。
    """
    import meshio
    import numpy as np

    vtu, frame_time = _select_frame(run_dir, at_time)
    m = meshio.read(vtu)
    conn = None
    for block in m.cells:
        if block.data.shape[1] == 8:
            conn = np.asarray(block.data)
            break
    if conn is None:
        raise SystemExit(f"{vtu}: 找不到 HEX8 单元块")
    data = m.cell_data_dict.get("max_temperature_history")
    if not data:
        raise SystemExit(f"{vtu}: 缺 max_temperature_history,"
                         "无法量熔池(需要 as-is 臂的热输出)")
    tmax = np.asarray(list(data.values())[0]).reshape(-1)
    centers = np.asarray(m.points)[conn].mean(axis=1)

    # 单元 y 向节距:从**全网格**取,不从曾熔子集取(子集可能只有一排)
    all_y = np.unique(np.round(centers[:, 1], 12))
    dy = float(np.diff(all_y).min()) if all_y.size > 1 else 0.0
    if dy <= 0.0:
        raise SystemExit(f"{vtu}: 网格 y 向只有一排单元,量不出熔池宽度")

    molten = tmax >= solidus
    if window is not None:
        x0, x1, y0, y1 = window
        # 窗界容差半个单元:窗界压在单元中心上时不让浮点噪声决定端排取舍
        tol = 0.5 * dy
        molten &= ((centers[:, 0] >= x0 - tol) & (centers[:, 0] <= x1 + tol)
                   & (centers[:, 1] >= y0 - tol) & (centers[:, 1] <= y1 + tol))
    if not molten.any():
        raise SystemExit(f"{run_dir}: 窗内没有单元达到固相线 {solidus} K,量不出熔池"
                         + (f"(取样时刻 {frame_time:.6f} s)" if frame_time else ""))
    ys = centers[molten, 1]
    xs = centers[molten, 0]
    uniq_y = np.unique(np.round(ys, 12))

    if track_y is not None:
        # 单侧(上半):道中心线到曾熔区上边缘
        half = float(ys.max() + 0.5 * dy - track_y)
        mode = f"单侧(上半),道中心线 y = {track_y*1e6:.1f} um"
        if half <= 0.0:
            raise SystemExit(
                f"单侧量法得到非正半宽 {half*1e6:.1f} um:曾熔区上边缘 "
                f"{(ys.max()+0.5*dy)*1e6:.1f} um 不高于道中心线 {track_y*1e6:.1f} um。"
                "取样时刻或道中心线给错了")
        if uniq_y.size < 2:
            raise SystemExit(
                f"曾熔区只有 1 排单元(节距 {dy*1e6:.0f} um):熔池没有被网格解析,"
                "量出来的半宽纯粹是离散尺度,不是物理量。加密网格或改用 --half-width")
    else:
        y_span = float(ys.max() - ys.min()) + dy   # 单元中心跨度补一个单元宽
        half = 0.5 * y_span
        mode = "双侧(全跨度的一半)"
        if uniq_y.size < 2:
            raise SystemExit(
                f"曾熔区只有 1 排单元(节距 {dy*1e6:.0f} um):熔池没有被网格解析,"
                "量出来的半宽纯粹是离散尺度,不是物理量。加密网格或改用 --half-width")
        if window is None and uniq_y.size > 8:
            raise SystemExit(
                f"曾熔区横跨 {uniq_y.size} 排单元(y 跨度 {y_span*1e6:.0f} um),"
                "这是多道扫描的并集而不是单道熔池半宽。请用 --from-run-window "
                "x0,x1,y0,y1 把量测限制到单独一条道上。")

    # 饱和护栏(F1):曾熔区顶到窗界 => 量测被窗口截断,不是熔池真实边界。
    # 静默返回一个被 hatch/2 压住的 L 是最坏的失效模式 —— 台账里看不出异常,
    # keff 被系统性压低,parity 臂朝 as-is 方向偏移。宁可停,不可静默。
    if window is not None:
        x0, x1, y0, y1 = window
        touch = []
        if ys.max() >= y1 - 0.5 * dy:
            touch.append(f"上界 y1={y1*1e6:.1f} um")
        if track_y is None and ys.min() <= y0 + 0.5 * dy:
            touch.append(f"下界 y0={y0*1e6:.1f} um")
        if touch:
            raise SystemExit(
                f"曾熔区顶到量测窗的 {' 与 '.join(touch)}(曾熔 y 范围 "
                f"{ys.min()*1e6:.1f}..{ys.max()*1e6:.1f} um,节距 {dy*1e6:.0f} um):"
                "半宽被窗口截断,不是熔池边界。放宽 --from-run-window 的 y 界;"
                "若放宽后会吃进相邻道,说明熔池全宽已超过 hatch,单道半宽在整层"
                "累积场里原理上量不出来 —— 改用 --from-run-time 取"
                "'某道跑完、下道未到'的帧配 --from-run-track-y 单侧量,"
                "或退回 --half-width 显式给值并登记来源")

    return {
        "half_width_m": half,
        "measurement_mode": mode,
        "molten_cells": int(molten.sum()),
        "y_extent_m": [float(ys.min() - 0.5 * dy), float(ys.max() + 0.5 * dy)],
        "x_span_m": float(xs.max() - xs.min()),
        "y_rows": int(uniq_y.size),
        "cell_pitch_y_m": dy,
        "frame": os.path.basename(vtu),
        "frame_time_s": frame_time,
        "requested_time_s": at_time,
        "track_y_m": track_y,
        "solidus_K": solidus,
        "window": list(window) if window else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", type=float, default=220.0, help="W(热闸门工况)")
    ap.add_argument("--speed", type=float, default=0.650, help="m/s")
    ap.add_argument("--absorptivity", type=float, default=0.62,
                    help="Balbaa Sec 3.1 DRS 实测")
    ap.add_argument("--beam-radius", type=float, default=50.0e-6,
                    help="m,与 runner --beam-radius 一致")
    ap.add_argument("--opd", type=float, default=100.0e-6,
                    help="m,Table 2 插值到 -45 um PSD")
    ap.add_argument("--rho-liquid", type=float, default=7925.0,
                    help="kg/m3,Table 1 密度区间的液相端(config rho_liquid)")
    ap.add_argument("--half-width", type=float, default=None, help="L,熔池半宽 [m]")
    ap.add_argument("--from-run", default=None, help="从 as-is 臂运行目录量 L")
    ap.add_argument("--from-run-window", default=None,
                    help="量 L 时的空间窗 x0,x1,y0,y1 [m],多道运行必填")
    ap.add_argument("--from-run-time", type=float, default=None,
                    help="取 <= 该时刻的最后一帧来量 L [s]。多道运行必须用它取"
                         "'某道跑完、下道未到'的那一帧,否则曾熔场是相邻道的并集")
    ap.add_argument("--from-run-track-y", type=float, default=None,
                    help="被量那条道的中心线 y [m]。给了就改用单侧(上半)量法,"
                         "避开双侧量法在 hatch/2 处的结构性天花板(F1)")
    ap.add_argument("--allow-incomplete-run", action="store_true",
                    help="允许从未跑完的 as-is 臂推 keff(默认拒绝,见 F2)")
    ap.add_argument("--qv", type=float, default=None, help="覆盖 Qv [W/m3](敏感度用)")
    ap.add_argument("--sweep", default="0.5,0.75,1.0,1.5,2.0",
                    help="L 与 Qv 的倍数括号,逗号分隔;空串关闭")
    ap.add_argument("--tag", default="220W", help="产物标签")
    ap.add_argument("--output", type=Path, default=None, help="液相导热表 CSV")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="推导台账 JSON")
    args = ap.parse_args()

    props = load_table1()
    rho_l = args.rho_liquid

    measured = None
    if args.from_run:
        # F2 的第一道闸:不完整的 as-is 臂不许推 keff。半途死掉的运行也会留下
        # 一堆 VTU,量出来的 L 看不出异常,却会被固化进 parity 臂。
        fp = source_run_fingerprint(args.from_run)
        if fp and fp.get("complete") is not True and not args.allow_incomplete_run:
            raise SystemExit(
                f"as-is 臂未完成(thermal_energy_ledger_summary.json complete="
                f"{fp.get('complete')!r},台账 {fp.get('ledger_lines')} 行):"
                "不完整的运行推出来的 keff 会被 parity 臂当成真值用完全程。"
                "先把 as-is 臂跑完;确实要用半成品请显式加 --allow-incomplete-run")
        window = None
        if args.from_run_window:
            window = [float(v) for v in args.from_run_window.split(",")]
            if len(window) != 4:
                raise SystemExit("--from-run-window 需要 x0,x1,y0,y1 四个值")
        measured = melt_half_width_from_run(
            args.from_run, props["solidus_K"], window,
            at_time=args.from_run_time, track_y=args.from_run_track_y)
        L = measured["half_width_m"]
        if args.half_width is not None and abs(L - args.half_width) > 1e-12:
            print(f"注意:--from-run 量得 L={L:.4e} m,"
                  f"覆盖 --half-width={args.half_width:.4e} m")
    elif args.half_width is not None:
        L = args.half_width
    else:
        raise SystemExit(
            "L(熔池半宽)是模型输出,本脚本不替你猜(D-V1-10 鸡生蛋)。\n"
            "  --from-run <as-is 臂运行目录>  由我们自己的 as-is 结果量出"
            "(推荐,自洽一次 Picard)\n"
            "  --half-width <m>              显式给定并自行登记来源")

    qv = args.qv if args.qv is not None else qv_peak_exponential(
        args.power, args.absorptivity, args.beam_radius, args.opd)

    base = keff_at(L, qv, args.speed, props, rho_l)

    # 括号:L 与 Qv 各自缩放。keff 对二者都只是 log 敏感,但 dNu/dlnL = 3.226,
    # 因子 2 的 L 误差就搬动 keff 约 2.2 kl —— 必须登记,不能说"弱敏感"。
    brackets = {"L": [], "Qv": []}
    mults = [float(v) for v in args.sweep.split(",") if v.strip()] if args.sweep else []
    for mult in mults:
        brackets["L"].append({"multiplier": mult,
                              **keff_at(L * mult, qv, args.speed, props, rho_l)})
        brackets["Qv"].append({"multiplier": mult,
                               **keff_at(L, qv * mult, args.speed, props, rho_l)})

    # 量出来的 L 只能落在单元边界上,分辨率 = 半个单元。这个离散化偏差是**单边
    # 向上**的(熔池边缘所在的那个单元整格计入),不是随机噪声,所以必须单独
    # 报,不能混进上面那个乘数括号里。
    if measured:
        dy = measured["cell_pitch_y_m"]
        brackets["L_discretization"] = {
            "cell_pitch_y_m": dy,
            "L_resolution_m": 0.5 * dy,
            "note": f"L 量测分辨率 +/-{dy*0.5*1e6:.0f} um(半个单元)。熔池边缘"
                    "落在哪个单元里就整格计入,故该偏差偏向高估 L、进而高估 keff",
            "members": [
                {"L_m": max(L + s * 0.5 * dy, 0.5 * dy),
                 **keff_at(max(L + s * 0.5 * dy, 0.5 * dy), qv, args.speed,
                           props, rho_l)}
                for s in (-1.0, 0.0, 1.0)
            ],
        }

    flags = []
    if base["nusselt_clamped_to_zero"]:
        flags.append(
            f"Nu<=0(Ma={base['Marangoni']:.4g} < Ma_crit={MA_CRIT:.4g}):Eq 21 "
            "相关式在此工况外推失效,keff 已钳到 kl,parity 臂退化为 as-is")
    if base["peak_T_estimate_K"] > props["boiling_K"]:
        flags.append(
            f"Eq 23 的 dT 给出峰值温度估计 {base['peak_T_estimate_K']:.0f} K > 沸点 "
            f"{props['boiling_K']:.0f} K。这是 Balbaa 公式的绝热性质(不含传导损失),"
            "如实登记,不做钳制(钳制即标定)")
    if base["keff_W_mK"] < props["k_liquid"]:
        raise SystemExit("内部错误:keff < kl")

    record = {
        "id": "D-V2-21-parity",
        "what": "Balbaa Eq 19-24 熔池等效导热的预计算(方案 (c):预计算表,不改共享求解器)",
        "equations": {
            "19": "keff = kl + h L", "20": "Nu = h L / kl",
            "21": "Nu = 1.6129 ln(Ma) - 10.183",
            "22": "Ma = -(dsigma/dT) L dT rho Cpl / (mu kl)",
            "23": "dT = [-Cpl + sqrt(Cpl^2 + 2 a^2 dt Qv / rho)] / a^2, a = (dsigma/dT)/mu",
            "24": "dt = L / v",
            "_provenance": "Balbaa2022 Sec 2.5 p.7-8,references/docs 内仓库副本转写",
        },
        "material_inputs": props,
        "rho_liquid_kg_m3": rho_l,
        "process": {
            "power_W": args.power, "speed_m_s": args.speed,
            "absorptivity": args.absorptivity, "beam_radius_m": args.beam_radius,
            "opd_m": args.opd,
            "_provenance": "热闸门工况 220 W / 650 mm/s"
                           "(balbaa-v2-model.json validation_targets.pyrometer)",
        },
        "qv_mode": "override" if args.qv is not None
                   else "Eq 18 指数源束心峰值 2 Ac P/(pi r^2 d)",
        "characteristic_length": {
            "L_m": L,
            "source": "as-is 臂实测(--from-run)" if measured else "显式给定(--half-width)",
            "source_run_fingerprint": source_run_fingerprint(args.from_run),
            "measurement": measured,
            "chicken_and_egg": "L 是模型输出;这是 D-V1-10/D-V2-21 登记的固有循环,"
                               "本实现用一次 Picard(as-is -> L -> parity)而不是迭代到自洽",
        },
        "derivation": base,
        "sensitivity_brackets": brackets,
        "ma_critical_for_positive_nu": MA_CRIT,
        "flags": flags,
        "zero_calibration_note": "没有任何一个量向高温计/XRD/ABAQUS 数值回调;"
                                 "L 来自我们自己的 as-is 臂输出,不是 Balbaa 的实测熔池",
    }

    print(f"=== D-V2-21 parity 臂 keff 预计算 ({args.tag}) ===")
    print(f"  材料表来源  {TABLE1}")
    print(f"  kl={props['k_liquid']} W/mK  Cpl={props['cp_liquid']} J/kgK  "
          f"mu={props['mu']} Pa s  dsigma/dT={props['dsigma_dT']} N/(m K)  "
          f"rho_l={rho_l} kg/m3")
    print(f"  P={args.power} W  v={args.speed*1e3:.0f} mm/s  Ac={args.absorptivity}  "
          f"r={args.beam_radius*1e6:.0f} um  OPD={args.opd*1e6:.0f} um")
    print(f"  L={L*1e6:.2f} um ({record['characteristic_length']['source']})")
    if measured:
        print(f"    量法 {measured['measurement_mode']};帧 {measured['frame']}"
              + (f" @ t={measured['frame_time_s']:.6f} s"
                 if measured["frame_time_s"] is not None else " (末帧)"))
        print(f"    曾熔 {measured['molten_cells']} 单元 / {measured['y_rows']} 排,"
              f"y 范围 {measured['y_extent_m'][0]*1e6:.1f}..{measured['y_extent_m'][1]*1e6:.1f} um,"
              f"节距 {measured['cell_pitch_y_m']*1e6:.0f} um")
    print(f"  Qv={qv:.4e} W/m3   dt=L/v={base['interaction_time_s']:.4e} s")
    print(f"  dT={base['delta_T_K']:.1f} K  ->  Ma={base['Marangoni']:.4g}  "
          f"Nu={base['Nusselt']:.4f}  h={base['h_W_m2K']:.4e} W/m2K")
    print(f"  keff={base['keff_W_mK']:.4f} W/mK  =  {base['keff_over_kl']:.3f} x kl")
    if mults:
        print("  括号(L 倍数 -> keff W/mK):  " +
              "  ".join(f"{b['multiplier']}x:{b['keff_W_mK']:.1f}" for b in brackets["L"]))
        print("  括号(Qv 倍数 -> keff W/mK): " +
              "  ".join(f"{b['multiplier']}x:{b['keff_W_mK']:.1f}" for b in brackets["Qv"]))
    if "L_discretization" in brackets:
        d = brackets["L_discretization"]
        print(f"  括号(L 离散化 +/-{d['L_resolution_m']*1e6:.0f} um -> keff W/mK): " +
              "  ".join(f"L={b['L_m']*1e6:.0f}um:{b['keff_W_mK']:.1f}"
                        for b in d["members"]))
        print("    ^ 单边偏高:熔池边缘所在单元整格计入,L 与 keff 都偏向高估")
    for flag in flags:
        print(f"  [FLAG] {flag}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        src = (f"D-V2-21 parity: Balbaa Eq19-24, L={L*1e6:.2f}um, "
               f"P={args.power}W v={args.speed*1e3:.0f}mm/s, Nu={base['Nusselt']:.4f}")
        # 常值表:Balbaa 的 keff 是每个工况一个数(dT 是"熔池最大温差"这一个量),
        # 不是 T 的函数。铺满全温域使 np.interp 的端点钳制不引入隐含斜率。
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["T", "value", "source"])
            for T in (273.15, props["solidus_K"], props["liquidus_K"],
                      props["boiling_K"], 6000.0):
                w.writerow([f"{T:.2f}", f"{base['keff_W_mK']:.6f}", src])
        print(f"  写出液相导热表 {args.output}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  写出推导台账 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
