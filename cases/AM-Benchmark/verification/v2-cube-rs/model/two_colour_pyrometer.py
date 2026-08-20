#!/usr/bin/env python3
"""双色(比色)高温计的合成读数 —— 仪器物理,不是场统计量。

**为什么需要这一路**(Opus5 2026-08-20 场级分析,IET-19):采纳口径
(`analyze_pyrometer.py` 的 `avg_K`,Balbaa Sec 3.3 的条件平均)是
"2 mm 圆内所有 T > 1000 degC 单元的算术平均"。它的分母 `n_hot` 在观测窗内
从 11 涨到 140,读数因此掉了 919 K,**而场的峰温在同一窗口内趋势只有 -27 K**。
也就是说采纳口径的形状几乎全部来自分母,而不是来自场。它是复现 Balbaa 协议
的主尺(必须原样保留),但它**不是仪器的模型**。

真实的双色高温计不做条件平均。它测两个波长的**光谱辐射亮度**,取比值,
按普朗克定律反演温度。本模块就是这件事的正演 + 反演:

    L_lambda(T) = e_lambda * (2 h c^2 / lambda^5) / (exp(h c /(lambda k T)) - 1)

    S_i = sum_cells A_cell * L_lambda_i(T_cell)        i = 1, 2
    R   = S_1 / S_2
    T_2c: 解 L_lambda1(T)/L_lambda2(T) = R

灰体假设(e_lambda1 = e_lambda2)让发射率在比值里**精确抵消** —— 这正是双色
高温计相对单色仪器的优点,也是它对 D-V1-19 的发射率两读法歧义完全免疫的原因。

## 口径约定(登记在 D-V2-24)

- **波长** 0.95 / 1.05 um(Si/Si,论文只写 "Si/Si ~1 um";两条 Si 探测器
  通道的分离必须由我们定,`--tc-wavelengths` 可改,敏感性见 `sensitivity()`);
- **面积权重** 单元面积相等(M1 网格顶层是 40 um 均匀六面体),故 A_cell 在
  比值里抵消,合成读数 = 亮度的**算术平均**再反演;
- **几何** 与采纳口径**完全一致**:同一个 2 mm 圆、同一顶层深度、同一单元
  温度定义(8 节点算术平均)。两列因此可直接相比;
- **无阈值** 圆内所有单元都参与积分。这不是选择,是仪器的物理:1 um 波段的
  维恩尾让冷单元自动可忽略(`cold_tail_fraction()` 把这件事量化,不是断言);
- **量程** 该仪器 1000-3000 degC。反演结果**不钳制**(钳制即标定),超量程
  单独计数并标记 —— "真机当时没有超量程,而我们的场会" 本身就是证据。

## 零标定

本模块不含任何可调到实测上的参数。波长是仪器规格,灰体是双色仪器的定义,
面积权重是网格的性质。唯一的自由度(波长分离)以敏感性括号如实上报。
"""
from __future__ import annotations

import numpy as np

# CODATA 2018,与 thermal.py 的 sigma 同源
_H = 6.62607015e-34          # J s
_C = 2.99792458e8            # m/s
_KB = 1.380649e-23           # J/K
C2 = _H * _C / _KB           # 1.4387768775e-2 m K,第二辐射常数

DEFAULT_WAVELENGTHS_M = (0.95e-6, 1.05e-6)
RANGE_MIN_C = 1000.0
RANGE_MAX_C = 3000.0
C2K = 273.15


def spectral_radiance(T_K, lambda_m):
    """普朗克光谱辐射亮度(黑体,e=1)。T 可以是标量或数组。

    用 expm1 而不是 exp(x)-1:在冷单元上 x ~ 40,exp 会先溢出成 inf 再相减,
    而 expm1 在整个量程里都稳。低于 1 K 的温度直接给 0(不参与积分)。
    """
    T = np.asarray(T_K, dtype=np.float64)
    out = np.zeros_like(T)
    ok = T > 1.0
    if not np.any(ok):
        return out
    x = C2 / (lambda_m * T[ok])
    # x 很大时 expm1(x) 溢出 -> 亮度下溢到 0,这正是物理上想要的
    with np.errstate(over="ignore"):
        denom = np.expm1(x)
    pref = 2.0 * _H * _C ** 2 / lambda_m ** 5
    out[ok] = np.where(np.isfinite(denom) & (denom > 0.0), pref / denom, 0.0)
    return out


def ratio_of(T_K, wavelengths=DEFAULT_WAVELENGTHS_M):
    """单一温度下的亮度比 L(l1)/L(l2)。反演的正演对照。"""
    l1, l2 = wavelengths
    a = spectral_radiance(np.atleast_1d(T_K), l1)
    b = spectral_radiance(np.atleast_1d(T_K), l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return r if np.ndim(T_K) else float(r[0])


def invert_ratio(ratio, wavelengths=DEFAULT_WAVELENGTHS_M,
                 t_lo=200.0, t_hi=100000.0, tol=1.0e-9):
    """由亮度比反演温度。

    R(T) 在 l1 < l2 时对 T 严格单调递增(维恩位移:越热,短波涨得越快),
    所以二分是充分且无歧义的。括号取得很宽 [200, 100000] K 是故意的:
    我们的场会冲到 4800 K,把上界卡在沸点附近等于偷偷钳制。
    """
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None
    l1, l2 = wavelengths
    if not l1 < l2:
        raise ValueError("约定 lambda1 < lambda2(短波在前),否则单调性反号")
    lo, hi = float(t_lo), float(t_hi)
    r_lo, r_hi = ratio_of(lo, wavelengths), ratio_of(hi, wavelengths)
    if not (r_lo <= ratio <= r_hi):
        return None            # 比值落在括号外:报缺,不外推
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio_of(mid, wavelengths) < ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, mid):
            break
    return 0.5 * (lo + hi)


def integrate_field(T_K, wavelengths=DEFAULT_WAVELENGTHS_M, weights=None):
    """把一批单元温度积分成两个通道的亮度信号。

    weights=None 表示单元面积相等(M1 顶层均匀网格),此时用算术平均。
    返回 (S1, S2);两者的绝对标度无意义,只有比值进反演。
    """
    T = np.asarray(T_K, dtype=np.float64).reshape(-1)
    if T.size == 0:
        return 0.0, 0.0
    l1, l2 = wavelengths
    L1 = spectral_radiance(T, l1)
    L2 = spectral_radiance(T, l2)
    if weights is None:
        return float(L1.mean()), float(L2.mean())
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != T.shape:
        raise ValueError("weights 与温度数组形状不一致")
    tot = w.sum()
    if tot <= 0.0:
        return 0.0, 0.0
    return float((w * L1).sum() / tot), float((w * L2).sum() / tot)


def read_field(T_K, wavelengths=DEFAULT_WAVELENGTHS_M, weights=None,
               range_min_C=RANGE_MIN_C, range_max_C=RANGE_MAX_C):
    """一帧的合成双色读数 + 量程判定 + 冷尾占比。

    返回 dict:
      T_K              反演温度(不钳制;比值落括号外时为 None)
      S1, S2           两通道亮度信号(可跨帧做时间平均,再一起反演)
      over_range       反演温度是否超出仪器上限
      under_range      是否低于仪器下限(真机此时什么都读不到)
      cold_tail_frac   低于下限的单元对 S1 的贡献占比 —— 用来证明"不设阈值"
                       在 1 um 波段是无害的,而不是断言它无害
      n_cells          参与积分的单元数
    """
    T = np.asarray(T_K, dtype=np.float64).reshape(-1)
    S1, S2 = integrate_field(T, wavelengths, weights)
    T_read = invert_ratio(S1 / S2, wavelengths) if S2 > 0.0 else None
    l1 = wavelengths[0]
    L1 = spectral_radiance(T, l1)
    cold = T < (range_min_C + C2K)
    tot1 = float(L1.sum())
    return {
        "T_K": T_read,
        "S1": S1, "S2": S2,
        "n_cells": int(T.size),
        "over_range": bool(T_read is not None and T_read > range_max_C + C2K),
        "under_range": bool(T_read is not None and T_read < range_min_C + C2K),
        "cold_tail_frac": float(L1[cold].sum() / tot1) if tot1 > 0.0 else None,
        "n_cells_over_range": int((T > range_max_C + C2K).sum()),
    }


def read_accumulated(S1, S2, wavelengths=DEFAULT_WAVELENGTHS_M,
                     range_min_C=RANGE_MIN_C, range_max_C=RANGE_MAX_C):
    """由已经(按时间)累加/平均过的两通道信号反演。

    这是 10 ms 响应的**正确**分箱读法:仪器积分的是亮度,不是温度。先把箱内
    各帧的 S1/S2 平均,再反演一次;而不是先逐帧反演温度再平均温度。
    """
    if not (S2 > 0.0):
        return {"T_K": None, "over_range": False, "under_range": False}
    T_read = invert_ratio(S1 / S2, wavelengths)
    return {
        "T_K": T_read,
        "over_range": bool(T_read is not None and T_read > range_max_C + C2K),
        "under_range": bool(T_read is not None and T_read < range_min_C + C2K),
    }


def uniform_field_self_check(temperatures=(1273.15, 1800.0, 2500.0, 3273.15, 4800.0),
                             wavelengths=DEFAULT_WAVELENGTHS_M, atol_K=1.0e-6):
    """自检:均匀场必须被精确还原。

    这是本模块唯一有资格叫"验证"的东西 —— 若一片单元全是 T0,双色反演按定义
    必须给回 T0。任何正演/反演的符号错、常数错、单调性错都会在这里当场暴露。
    返回 (ok, 明细表)。
    """
    rows, ok = [], True
    for T0 in temperatures:
        field = np.full(64, float(T0))
        got = read_field(field, wavelengths)["T_K"]
        err = None if got is None else abs(got - T0)
        good = err is not None and err <= atol_K
        ok = ok and good
        rows.append({"T_input_K": float(T0), "T_readback_K": got,
                     "abs_error_K": err, "pass": good})
    return ok, rows


def sensitivity(T_field, separations_um=(0.05, 0.10, 0.20), centre_um=1.00):
    """波长分离的敏感性括号:论文只写 "Si/Si ~1 um",分离是我们定的。

    对同一个场,用几组 (centre -/+ sep/2) 反演,把读数的散布如实上报。
    """
    out = []
    for sep in separations_um:
        wl = ((centre_um - 0.5 * sep) * 1e-6, (centre_um + 0.5 * sep) * 1e-6)
        r = read_field(T_field, wl)
        out.append({"separation_um": sep,
                    "wavelengths_um": [wl[0] * 1e6, wl[1] * 1e6],
                    "T_K": r["T_K"]})
    return out


PROTOCOL_NOTE = {
    "instrument": "双色(比色)高温计合成读数,Si/Si,灰体假设",
    "wavelengths_m": list(DEFAULT_WAVELENGTHS_M),
    "planck_constant_source": "CODATA 2018;C2 = hc/k = 1.4387769e-2 m K",
    "emissivity_handling": "灰体 -> e 在亮度比里精确抵消;本路对 D-V1-19 的"
                           "发射率两读法歧义完全免疫",
    "area_weighting": "顶层 M1 网格单元面积相等 -> 亮度算术平均",
    "threshold": "无。1 um 波段的维恩尾使冷单元自动可忽略,cold_tail_frac 量化之",
    "bin_reading": "箱内先平均亮度(S1,S2)再反演一次 —— 仪器积分的是亮度不是温度",
    "range_handling": "1000-3000 degC 如实模拟;反演不钳制,超量程单列计数",
    "geometry": "与采纳口径完全一致(同圆心/同直径/同顶层/同单元温度定义)",
    "registered_as": "D-V2-24",
    "zero_calibration": "无任何可向实测回调的参数",
}


if __name__ == "__main__":
    import json
    ok, rows = uniform_field_self_check()
    print("=== 双色合成读数:均匀场自检 ===")
    for r in rows:
        got = "None" if r["T_readback_K"] is None else f"{r['T_readback_K']:.9f}"
        err = "n/a" if r["abs_error_K"] is None else f"{r['abs_error_K']:.3e}"
        print(f"  输入 {r['T_input_K']:9.2f} K -> 反演 {got:>16s} K  "
              f"|err| {err}  {'PASS' if r['pass'] else 'FAIL'}")
    print(f"  自检 {'通过' if ok else '失败'}")
    print("\n=== 口径 ===")
    print(json.dumps(PROTOCOL_NOTE, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)
