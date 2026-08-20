#!/usr/bin/env python3
"""Balbaa Fig 15(三定点温度历程)与 Fig 16(冷却速率 / 温度梯度)数字化。

沿用 D-V2-23 为 Fig 14 建立的同一套工具链与同一份逐点不确定度预算:
  1. 从仓库内 PDF 抽该页内嵌位图(原始分辨率,不二次渲染);
  2. 由长直深色行/列定出绘图框 -> 一次线性标定;
  3. 用框外刻度线做**独立**的第二次标定,两者之差进入读数不确定度;
  4. 按颜色分离数据系列,图例框/插图框内的一切丢弃;
  5. 锚点歧义按系列类型计:曲线取**线宽的一半**,标记取包围盒中心与像素质心之差。

**先说证据性质**(D-V2-27,必须写在最前面):**Fig 15 和 Fig 16 都是 Balbaa
自己模型的输出,不是实验测量。** 全文有时间分辨的实验数据只有 Fig 14 那 5 个
高温计点。所以这两张图是 **code-to-code 靶子**,永远不能当作独立的实验验证。
把一条温度历程曲线误当成实测,是这类复现工作里最容易犯的错。

它们仍然值得做,理由是**它们没有分母**:定点探针不做条件平均,也就没有
n_hot,Fig 14 上主导一切的 1/n_hot 稀释(D-V2-24)碰不到它们。形状、熔化、
凝固、冷却速率都可以直接比。这是论文提供的最干净的 code-to-code 靶子。

零标定:只读图,不拟合、不平滑、不向任何模型结果靠拢。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
PDF = (V2.parent.parent / "references" / "docs"
       / "Balbaa2022_multiscale-RS-LPBF-IN625_JMMP-6-2.pdf")

EVIDENCE_NATURE = (
    "HIS MODEL OUTPUT, NOT MEASUREMENT. Verified against p.17: the only "
    "time-resolved experimental data in the paper is the five pyrometer points "
    "of Fig 14. Fig 15/16 are a code-to-code target and can never serve as "
    "independent experimental validation (D-V2-27).")


# --------------------------------------------------------------------- 抽图
def extract_page_image(pdf, page_no, out_dir, index):
    """抽该页第 index 大的内嵌位图。需要 PyMuPDF;没有就报清楚,别静默降级。"""
    try:
        import fitz
    except ImportError:
        raise SystemExit("需要 PyMuPDF 抽图:pip install pymupdf,"
                         "或先手工导出该页位图再用 --image 传入")
    doc = fitz.open(pdf)
    page = doc[page_no - 1]
    images = []
    for info in page.get_images(full=True):
        data = doc.extract_image(info[0])
        images.append(data)
    if len(images) <= index:
        raise SystemExit(f"{pdf} 第 {page_no} 页只有 {len(images)} 张内嵌位图")
    # 按在页面上出现的顺序取(Fig 15 在上、Fig 16 在下),不按大小排序
    data = images[index]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"balbaa_p{page_no}_img{index}.{data['ext']}"
    out.write_bytes(data["image"])
    return out, data["width"], data["height"]


# ----------------------------------------------------------------- 框与刻度
def find_frame(dark, row_frac=0.4, col_frac=0.4, strength=0.6):
    """绘图框:横跨图幅一定比例以上的深色行/列。返回 (中心, 半厚) 四元组。

    只按"跨度超过阈值"取首尾两组是不够的:Fig 15 的右侧有一条又细又短的深色
    杂列(x = 2842,跨度刚过 0.3 图高),它会被当成右框线,把横轴标定拉长
    3.8 %,刻度交叉校验的残差因此从 2 px 涨到 94 px —— 也就是说**交叉校验
    确实报警了**,这里是按它的报警把选法修正过来。

    修正:框线是全图最长的直线。先取所有候选组里最强的那一组,再只保留强度
    不低于它 `strength` 倍的组,然后取首尾。图中间那些近乎竖直的尖峰即使够强
    也在中间,不影响首尾。
    """
    h, w = dark.shape

    def bands(counts, threshold):
        hits = np.where(counts > threshold)[0]
        groups = []
        for i in hits:
            if groups and i - groups[-1][-1] <= 2:
                groups[-1].append(i)
            else:
                groups.append([i])
        if not groups:
            return []
        best = max(float(np.max(counts[g])) for g in groups)
        return [g for g in groups if float(np.max(counts[g])) >= strength * best]

    row_counts, col_counts = dark.sum(axis=1), dark.sum(axis=0)
    rows = bands(row_counts, row_frac * w)
    cols = bands(col_counts, col_frac * h)
    if len(rows) < 2 or len(cols) < 2:
        raise SystemExit(f"找不到绘图框(rows={len(rows)}, cols={len(cols)})")
    fr = lambda g: (float(np.mean(g)), 0.5 * len(g))
    return {"top": fr(rows[0]), "bottom": fr(rows[-1]),
            "left": fr(cols[0]), "right": fr(cols[-1])}


def find_ticks(dark, frame, axis, tick_band=25, min_len=8):
    """框外刻度线的位置,用作独立的第二次标定(与 Fig 14 同法)。"""
    if axis == "x":
        y0 = int(frame["bottom"][0] + frame["bottom"][1]) + 2
        band = dark[y0:y0 + tick_band,
                    int(frame["left"][0]):int(frame["right"][0]) + 1]
        counts = band.sum(axis=0)
        offset = int(frame["left"][0])
    else:
        x1 = int(frame["left"][0] - frame["left"][1]) - 2
        band = dark[int(frame["top"][0]):int(frame["bottom"][0]) + 1,
                    max(x1 - tick_band, 0):x1]
        counts = band.sum(axis=1)
        offset = int(frame["top"][0])
    hits = np.where(counts >= min_len)[0]
    groups, lengths = [], []
    for i in hits:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        lengths.append(float(np.max(counts[g])))
    if not groups:
        return []
    cut = 0.5 * max(lengths)
    return [float(np.mean(g)) + offset
            for g, ln in zip(groups, lengths) if ln >= cut]


def calib_residual(ticks, lo_px, hi_px):
    """刻度网格与框线标定的一致性:框跨度是否是刻度间距的整数倍,残差多大 [px]。"""
    if len(ticks) < 3:
        return None
    ticks = sorted(ticks)
    spacing = float(np.median(np.diff(ticks)))
    span = hi_px - lo_px
    n = round(span / spacing)
    if n < 1:
        return None
    predicted = [lo_px + i * span / n for i in range(n + 1)]
    resid = [min(abs(t - p) for p in predicted) for t in ticks]
    return {"n_major_ticks": len(ticks), "median_spacing_px": spacing,
            "intervals_across_frame": int(n),
            "max_residual_px": float(max(resid)),
            "rms_residual_px": float(np.sqrt(np.mean(np.square(resid))))}


def inside_frame(shape, frame, pad=3):
    """严格框内的布尔掩码,余量 = 该侧框线半厚 + pad。"""
    inner = np.zeros(shape, bool)
    top = int(np.ceil(frame["top"][0] + frame["top"][1])) + pad
    bottom = int(np.floor(frame["bottom"][0] - frame["bottom"][1])) - pad
    left = int(np.ceil(frame["left"][0] + frame["left"][1])) + pad
    right = int(np.floor(frame["right"][0] - frame["right"][1])) - pad
    inner[top:bottom + 1, left:right + 1] = True
    return inner


def clusters(mask, min_size):
    from scipy import ndimage
    lab, _n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        sel = lab[sl] == i
        size = int(sel.sum())
        if size < min_size:
            continue
        ys, xs = np.where(lab == i)
        out.append({"size": size,
                    "bbox": (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop),
                    "bbox_center": (0.5 * (sl[0].start + sl[0].stop - 1),
                                    0.5 * (sl[1].start + sl[1].stop - 1)),
                    "centroid": (float(ys.mean()), float(xs.mean())),
                    "mask": lab == i})
    return out


# -------------------------------------------------------------------- Fig 15
def digitize_fig15(im, args):
    """三条定点温度历程曲线,逐像素列追踪。"""
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 110) & (G < 110) & (B < 110)
    red = (R > 120) & (R - G > 60) & (R - B > 60)
    blue = (B > 110) & (B - R > 40) & (B - G > 25)

    # 右框线只画在插图下方(y 跨度约 36% 图高),用 Fig 14 的 0.4 阈值找不到它,
    # 所以这里放到 0.3。中间那些近乎竖直的尖峰确实也能跨到 60% 图高,但 bands()
    # 取的是**首尾**两组,尖峰在中间,不会被误当成框线。
    frame = find_frame(dark, col_frac=args.fig15_col_frac)
    x0p, y0p = frame["left"][0], frame["bottom"][0]
    x1p, y1p = frame["right"][0], frame["top"][0]
    xlo, xhi = (float(v) for v in args.xlim.split(","))
    ylo, yhi = (float(v) for v in args.ylim.split(","))
    sx = (xhi - xlo) / (x1p - x0p)
    sy = (yhi - ylo) / (y1p - y0p)

    ticks_x = find_ticks(dark, frame, "x")
    ticks_y = find_ticks(dark, frame, "y")
    cross = {"x": calib_residual(ticks_x, x0p, x1p),
             "y": calib_residual(ticks_y, y1p, y0p)}

    # 严格框内。余量必须**盖过框线自己的半厚**:左框线半厚 4.5 px,固定留
    # 4 px 的话 x = 380 那一列仍是框线,而框线跨满整幅高度,于是黑色一路会在
    # t = 0.401 s 读出 2992 degC 的"尖峰"—— 那是框线,不是曲线。
    inner = inside_frame(dark.shape, frame, pad=args.frame_pad_px)

    # 右上角的网格插图:一大块中性灰。它内部有细密深色网格线,若不排除会污染
    # 黑色曲线。按"最大的一块中性灰连通域"定位并连同边界一起排除。
    neutral = (np.abs(R - G) < 25) & (np.abs(G - B) < 25) & (R > 80) & (R < 215)
    inset_bbox = None
    blocks = sorted(clusters(neutral & inner, args.min_inset_px),
                    key=lambda c: -c["size"])
    if blocks:
        r0, r1, c0, c1 = blocks[0]["bbox"]
        pad = 8
        inset_bbox = [int(r0), int(r1), int(c0), int(c1)]
        inner[max(int(r0) - pad, 0):int(r1) + pad,
              max(int(c0) - pad, 0):int(c1) + pad] = False

    series = {}
    for name, mask, colour in (("point_1", red, "red"),
                               ("point_2", dark, "black"),
                               ("point_3", blue, "blue")):
        m = mask & inner
        # 曲线**不是**单个连通域,两个原因都实测确认过:(a) JPEG 压缩把细尖峰
        # 打断,(b) 三条曲线互相交叠,后画的把先画的截断。所以"只留最大连通域"
        # 会把红线切到 1285 列、蓝线切到 240 列(全宽应为约 2370 列)。
        #
        # 改为:先做 5x5 闭运算把 JPEG 缺口补上,再按连通域筛 —— 保留包围盒在
        # 任一方向上足够长的块(曲线段横向长,尖峰纵向长),丢掉又小又矮的块。
        # 那行 "1290 degC" 注记正是又小又矮的一批,这样就被丢掉了,而不必给
        # 深色一路开特例。最后与**原始**掩码求交再量测,量的仍是真墨迹。
        from scipy import ndimage
        closed = ndimage.binary_closing(m, structure=np.ones((5, 5)))
        keep = np.zeros_like(m)
        discarded = []
        for c in clusters(closed, args.min_curve_px):
            r0, r1, c0, c1 = c["bbox"]
            if max(r1 - r0, c1 - c0) >= args.min_piece_px:
                keep |= c["mask"]
            else:
                discarded.append({"size": c["size"],
                                  "bbox": [int(r0), int(r1), int(c0), int(c1)]})
        m = keep & mask & inner
        if not m.any():
            raise SystemExit(f"{name}:框内没有够大的 {colour} 连通域")
        cols = np.where(m.any(axis=0))[0]
        if cols.size == 0:
            raise SystemExit(f"{name}:框内没有 {colour} 像素")
        samples, thickness = [], []
        for cx in cols:
            rows = np.where(m[:, cx])[0]
            # 一列里可能有多段(尖峰的上升沿与下降沿分开)。取整体范围,
            # 并把"这一列覆盖多少行"如实记下来 —— 陡段本来就该覆盖很多行,
            # 那不是读数不确定度,是曲线真的陡。
            span = int(rows.max() - rows.min()) + 1
            thickness.append(int(rows.size))
            # 位数按数字化真正支持的分辨率给:时间 1 us、温度 0.1 degC。
            # 一个已知到 +/-14 degC 的量写 15 位有效数字是假精度,而且会把
            # 这份输入撑到 2 MB。
            samples.append({
                "time_s": round(xlo + (cx - x0p) * sx, 6),
                "temperature_C": round(ylo + (float(rows.mean()) - y0p) * sy, 1),
                "temperature_C_hi": round(ylo + (float(rows.min()) - y0p) * sy, 1),
                "temperature_C_lo": round(ylo + (float(rows.max()) - y0p) * sy, 1),
                "column_span_px": span,
                "pixel_x": int(cx),
            })
        series[name] = {"colour": colour, "samples": samples,
                        "median_trace_thickness_px": float(np.median(thickness)),
                        "kept_pixels": int(m.sum()),
                        "columns_covered": int(cols.size),
                        "discarded_components": discarded,
                        "_discarded_note": "框内同色但不属于该曲线的连通域大小;深色一路里最大的那块通常是 \"1290 degC\" 注记"}

    frame_half = max(v[1] for v in frame.values())
    resid_x = cross["x"]["max_residual_px"] if cross["x"] else 0.0
    resid_y = cross["y"]["max_residual_px"] if cross["y"] else 0.0

    for name, s in series.items():
        # 锚点歧义 = 线宽的一半。对连续曲线这是正确的锚点项,也是它比 Fig 14
        # 的离散标记读数**更粗**的原因 —— 证据质量的真实差别,不是追踪的缺陷。
        anchor = 0.5 * s["median_trace_thickness_px"]
        terms_T = {"frame_line_half_thickness": frame_half * abs(sy),
                   "tick_cross_check": resid_y * abs(sy),
                   "trace_half_thickness": anchor * abs(sy)}
        terms_t = {"frame_line_half_thickness": frame_half * abs(sx),
                   "tick_cross_check": resid_x * abs(sx),
                   "pixel_column_half_width": 0.5 * abs(sx)}
        s["temperature_C_uncertainty"] = float(
            np.sqrt(sum(v * v for v in terms_T.values())))
        s["time_s_uncertainty"] = float(
            np.sqrt(sum(v * v for v in terms_t.values())))
        s["uncertainty_terms_C"] = terms_T
        s["uncertainty_terms_s"] = terms_t
        # temperature_K 不再逐点复写:它是 temperature_C + 273.15,派生量在
        # derived 里给一次就够,逐点复制只会让这份输入白白大一倍。
        s["temperature_K_note"] = "temperature_C + 273.15"

        # 判读用的派生量。峰值取该曲线所有列的最高像素(尖峰的顶),
        # 不做任何平滑 —— 平滑会把尖峰削掉,那正是这张图的信息所在。
        peak = max(s["samples"], key=lambda p: p["temperature_C_hi"])
        above = [p for p in s["samples"] if p["temperature_C_hi"] >= args.solidus_C]
        s["derived"] = {
            "peak_temperature_C": peak["temperature_C_hi"],
            "peak_time_s": peak["time_s"],
            "n_columns_above_solidus": len(above),
            "solidus_C_used": args.solidus_C,
            "first_time_above_solidus_s": above[0]["time_s"] if above else None,
            "last_time_above_solidus_s": above[-1]["time_s"] if above else None,
            "t_range_s": [s["samples"][0]["time_s"], s["samples"][-1]["time_s"]],
            "_peak_note": "峰值取该列像素的最高点(尖峰顶),不平滑:平滑会削掉"
                          "尖峰,而尖峰正是这张图要给的东西",
        }

    return {
        "id": "balbaa2022-fig15-thermal-history",
        "what": "三个定点(间隔 1 mm)的温度历程,220 W / 650 mm/s / 0.12 mm",
        "EVIDENCE_NATURE": EVIDENCE_NATURE,
        "axes": {"x": "Time (s)", "y": "Temperature (degC)",
                 "xlim": [xlo, xhi], "ylim": [ylo, yhi]},
        "annotated_line_C": args.solidus_C,
        "annotated_line_note": "图上那条灰虚线标着 1290 degC;论文用它标示凝固线。"
                               "本脚本只把它当**读数**用于派生量,不参与标定",
        "calibration": {
            "frame_px": {k: {"center": v[0], "half_thickness": v[1]}
                         for k, v in frame.items()},
            "scale_s_per_px": sx, "scale_C_per_px": sy,
            "tick_cross_check": cross,
            "inset_bbox_excluded": inset_bbox,
        },
        "series": series,
        "series_to_point_mapping": "红=Pt 1、黑=Pt 2、蓝=Pt 3,按图中插图的编号"
                                   "与三条曲线的升温先后一致(Pt 1 最先被扫到)",
        "read_off_uncertainty_note": (
            "逐系列分项:框线半厚 + 刻度线独立标定的最大残差 + 线宽的一半,"
            "方和根合成。连续曲线的锚点项比 Fig 14 的离散标记大,这是两张图"
            "证据质量的真实差别。未含 Balbaa 制图本身的误差"),
        "zero_calibration_note": "只读图,不拟合、不平滑、不向任何模型结果靠拢",
    }


# -------------------------------------------------------------------- Fig 16
def digitize_fig16(im, args):
    """三个类别位置 x 三个系列(冷却速率 / dTx / dTy),双纵轴。"""
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 110) & (G < 110) & (B < 110)
    red = (R > 120) & (R - G > 60) & (R - B > 60)

    # Fig 16 的**右**纵轴是红的(温度梯度轴,连轴线带刻度一起用红画),所以框线
    # 检测必须用 dark|red,只用 dark 会漏掉右框线(实测:dark 下右框列跨度只有
    # 0.27 图高,ink 下是 0.87)。红色趋势线是近水平的,不会造出高列,不干扰。
    ink = dark | red
    frame = find_frame(ink)
    x0p, y0p = frame["left"][0], frame["bottom"][0]
    x1p, y1p = frame["right"][0], frame["top"][0]
    llo, lhi = (float(v) for v in args.left_ylim.split(","))
    rlo, rhi = (float(v) for v in args.right_ylim.split(","))
    s_left = (lhi - llo) / (y1p - y0p)
    s_right = (rhi - rlo) / (y1p - y0p)

    ticks_y = find_ticks(ink, frame, "y")
    cross_y = calib_residual(ticks_y, y1p, y0p)

    inner = inside_frame(dark.shape, frame, pad=args.frame_pad_px)

    # 图例框:框内那个**闭合矩形**的深色连通域(与 Fig 14 同法,但判据更硬)。
    # 不能只按"最大的深色连通域"选 —— 黑色的冷却速率曲线横跨几乎整幅图,
    # 像素数和图例框边同量级。矩形边框的判据是:它的包围盒**四条边**都被自己
    # 填满;曲线的包围盒顶边只有峰值那几个像素。
    def looks_like_box(comp, cover=0.75):
        r0, r1, c0, c1 = comp["bbox"]
        sub = comp["mask"][r0:r1, c0:c1]
        if sub.shape[0] < 20 or sub.shape[1] < 20:
            return False
        edges = (sub[0, :].mean(), sub[-1, :].mean(),
                 sub[:, 0].mean(), sub[:, -1].mean())
        return min(edges) >= cover

    legend = None
    boxes = [c for c in sorted(clusters(dark & inner, args.min_legend_px),
                               key=lambda c: -c["size"]) if looks_like_box(c)]
    if boxes:
        r0, r1, c0, c1 = boxes[0]["bbox"]
        legend = [int(r0), int(r1), int(c0), int(c1)]
        inner[max(int(r0) - 6, 0):int(r1) + 6,
              max(int(c0) - 6, 0):int(c1) + 6] = False

    # 三个类别位置:x 轴主刻度。Pt 1/2/3 均匀分布,取框内的三个刻度。
    ticks_x = find_ticks(ink, frame, "x")
    if len(ticks_x) < 3:
        # 刻度找不到就退回等分:类别轴的三个位置在框内 1/6, 3/6, 5/6
        span = x1p - x0p
        ticks_x = [x0p + span * f for f in (1 / 6, 3 / 6, 5 / 6)]
        tick_source = "fallback: category axis assumed at 1/6, 3/6, 5/6 of frame"
    else:
        ticks_x = sorted(ticks_x)[:3]
        tick_source = "detected x-axis major ticks"

    half = int(args.marker_window_px)

    def read_markers(mask, cx, label):
        """在 cx 附近的窗口里按 y 聚类,给出每个标记的中心与锚点歧义。

        连线穿过标记本身,所以窗口内像素的质心就是标记中心的无偏估计;
        质心与包围盒中心之差按 Fig 14 的口径计入锚点歧义。
        """
        lo, hi = max(int(cx) - half, 0), min(int(cx) + half + 1, mask.shape[1])
        win = mask[:, lo:hi] & inner[:, lo:hi]
        rows = np.where(win.any(axis=1))[0]
        if rows.size == 0:
            return []
        groups = []
        for r in rows:
            if groups and r - groups[-1][-1] <= args.cluster_gap_px:
                groups[-1].append(r)
            else:
                groups.append([r])
        out = []
        for g in groups:
            if len(g) < args.min_marker_height_px:
                continue                       # 只有连线穿过,没有标记
            sub = win[g[0]:g[-1] + 1, :]
            ys, xs = np.where(sub)
            cy = float(ys.mean()) + g[0]
            bbox_cy = 0.5 * (g[0] + g[-1])
            # 中心是否被填充:x 填心,方框/圆圈空心 —— 用来区分 dTx 与 dTy
            h_, w_ = sub.shape
            cr0, cr1 = int(h_ * 0.4), int(h_ * 0.6) + 1
            cc0, cc1 = int(w_ * 0.4), int(w_ * 0.6) + 1
            filled = float(sub[cr0:cr1, cc0:cc1].mean())
            out.append({"pixel_y": cy, "bbox_center_y": bbox_cy,
                        "anchor_ambiguity_px": abs(cy - bbox_cy),
                        "height_px": len(g), "n_pixels": int(sub.sum()),
                        "centre_fill_fraction": filled,
                        "shape": "filled_centre" if filled > args.fill_cut
                                 else "hollow_centre",
                        "_label": label})
        return out

    frame_half = max(v[1] for v in frame.values())
    resid_y = cross_y["max_residual_px"] if cross_y else 0.0

    def budget(anchor_px, scale):
        terms = {"frame_line_half_thickness": frame_half * abs(scale),
                 "tick_cross_check": resid_y * abs(scale),
                 "marker_anchor": anchor_px * abs(scale)}
        return float(np.sqrt(sum(v * v for v in terms.values()))), terms

    cooling_rate, dtx, dty = [], [], []
    for i, cx in enumerate(ticks_x, start=1):
        for m in read_markers(dark, cx, "black"):
            value = llo + (m["pixel_y"] - y0p) * s_left
            u, terms = budget(m["anchor_ambiguity_px"], s_left)
            cooling_rate.append({
                "point": f"Pt {i}", "value": value,
                "unit": "degC/s x 1e5", "value_degC_per_s": value * 1.0e5,
                "uncertainty": u, "uncertainty_terms": terms,
                "marker_px": m["n_pixels"], "marker_height_px": m["height_px"]})
        for m in read_markers(red, cx, "red"):
            value = rlo + (m["pixel_y"] - y0p) * s_right
            u, terms = budget(m["anchor_ambiguity_px"], s_right)
            row = {"point": f"Pt {i}", "value": value, "unit": "degC/mm",
                   "uncertainty": u, "uncertainty_terms": terms,
                   "marker_px": m["n_pixels"], "marker_height_px": m["height_px"],
                   "centre_fill_fraction": m["centre_fill_fraction"]}
            (dtx if m["shape"] == "filled_centre" else dty).append(row)

    return {
        "id": "balbaa2022-fig16-cooling-rate-and-gradients",
        "what": "三个定点的冷却速率与温度梯度,220 W / 650 mm/s / 0.12 mm",
        "EVIDENCE_NATURE": EVIDENCE_NATURE,
        "axes": {"x": "categorical: Pt 1 / Pt 2 / Pt 3",
                 "y_left": "Cooling Rate (degC/s x 1e5)",
                 "y_right": "Temperature Gradient (degC/mm)",
                 "left_ylim": [llo, lhi], "right_ylim": [rlo, rhi]},
        "calibration": {
            "frame_px": {k: {"center": v[0], "half_thickness": v[1]}
                         for k, v in frame.items()},
            "scale_left_per_px": s_left, "scale_right_per_px": s_right,
            "tick_cross_check_y": cross_y,
            "category_x_px": [float(v) for v in ticks_x],
            "category_x_source": tick_source,
            "legend_bbox_excluded": legend,
        },
        "series": {"cooling_rate": cooling_rate,
                   "delta_T_x": dtx, "delta_T_y": dty},
        "series_separation_method": (
            "红色系列里 dTx 是 x 号(笔画在中心相交 -> 中心被填充),dTy 是空心"
            "方框(中心空)。按标记中心区域的填充率区分,阈值 "
            f"{args.fill_cut};每个标记的实际填充率一并输出,便于复核"),
        "connecting_lines_note": (
            "两条红色系列各自带一条趋势线穿过标记。趋势线**穿过标记本身**,"
            "所以窗口内像素质心仍是标记中心的无偏估计;质心与包围盒中心之差"
            "按 Fig 14 的口径计入锚点歧义,而不是假装连线不存在"),
        "read_off_uncertainty_note": (
            "逐点分项:框线半厚 + 刻度线独立标定的最大残差 + 标记锚点歧义,"
            "方和根合成。未含 Balbaa 制图本身的误差"),
        "zero_calibration_note": "只读图,不拟合、不平滑、不向任何模型结果靠拢",
    }


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--page", type=int, default=18, help="Fig 15/16 所在页")
    ap.add_argument("--fig15-image", type=Path, default=None)
    ap.add_argument("--fig16-image", type=Path, default=None)
    ap.add_argument("--xlim", default="0.4,1.0", help="Fig 15 横轴框线值 [s]")
    ap.add_argument("--ylim", default="0,3000", help="Fig 15 纵轴框线值 [degC]")
    ap.add_argument("--solidus-C", type=float, default=1290.0,
                    help="图上标注的那条虚线 [degC]")
    ap.add_argument("--left-ylim", default="1,3", help="Fig 16 左轴框线值")
    ap.add_argument("--right-ylim", default="0,2000", help="Fig 16 右轴框线值")
    ap.add_argument("--fig15-col-frac", type=float, default=0.30,
                    help="Fig 15 找左右框线的列跨度阈值(右框线只画在插图下方)")
    ap.add_argument("--min-curve-px", type=int, default=60,
                    help="Fig 15 曲线连通域的最小像素数")
    ap.add_argument("--min-piece-px", type=int, default=150,
                    help="Fig 15 曲线段包围盒的最小跨度(任一方向);比这更小"
                         "又更矮的深色块是图上注记,不是曲线")
    ap.add_argument("--min-inset-px", type=int, default=100000)
    ap.add_argument("--min-legend-px", type=int, default=3000)
    ap.add_argument("--frame-pad-px", type=int, default=3,
                    help="框内余量,在框线半厚之外再留这么多像素")
    ap.add_argument("--marker-window-px", type=int, default=30)
    ap.add_argument("--cluster-gap-px", type=int, default=6)
    ap.add_argument("--min-marker-height-px", type=int, default=14)
    ap.add_argument("--fill-cut", type=float, default=0.25)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--overlay", type=Path, default=None,
                    help="把追踪结果画回原图上,供人眼复核")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from PIL import Image

    work = args.work_dir or Path(".")
    notes = {}
    if args.fig15_image:
        p15, notes["fig15"] = args.fig15_image, "外部位图"
    else:
        p15, w, h = extract_page_image(args.pdf, args.page, work, 0)
        notes["fig15"] = (f"仓库内 PDF 第 {args.page} 页第 1 张内嵌位图 {w}x{h}"
                          "(原始分辨率,未二次渲染)")
    if args.fig16_image:
        p16, notes["fig16"] = args.fig16_image, "外部位图"
    else:
        p16, w, h = extract_page_image(args.pdf, args.page, work, 1)
        notes["fig16"] = (f"仓库内 PDF 第 {args.page} 页第 2 张内嵌位图 {w}x{h}"
                          "(原始分辨率,未二次渲染)")

    im15 = np.asarray(Image.open(p15).convert("RGB")).astype(int)
    im16 = np.asarray(Image.open(p16).convert("RGB")).astype(int)
    fig15 = digitize_fig15(im15, args)
    fig16 = digitize_fig16(im16, args)
    fig15["provenance"] = f"Balbaa2022 Figure 15(Sec 4.2.1,p.{args.page}):{notes['fig15']}"
    fig16["provenance"] = f"Balbaa2022 Figure 16(Sec 4.2.1,p.{args.page}):{notes['fig16']}"

    doc = {
        "id": "balbaa2022-fig15-16",
        "EVIDENCE_NATURE": EVIDENCE_NATURE,
        "condition": {"power_W": 220, "speed_mm_s": 650, "hatch_mm": 0.12,
                      "layer_mm": 0.04, "_source": "Balbaa2022 Sec 4.2.1"},
        "registered_as": "D-V2-27",
        "toolchain": "与 D-V2-23 的 Fig 14 数字化同一套(框线标定 + 刻度线独立"
                     "交叉校验 + 逐点分项不确定度方和根)",
        "fig15": fig15,
        "fig16": fig16,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print("=== Balbaa Fig 15 数字化(定点温度历程)===")
    print(f"  源 {notes['fig15']}")
    c = fig15["calibration"]
    print(f"  标定 {c['scale_s_per_px']*1e3:.5f} ms/px, "
          f"{abs(c['scale_C_per_px']):.4f} degC/px;插图排除 {c['inset_bbox_excluded']}")
    for axis, cc in c["tick_cross_check"].items():
        print(f"  刻度交叉检查 {axis}: " + ("未检出" if not cc else
              f"{cc['n_major_ticks']} 主刻度, 框跨 {cc['intervals_across_frame']} 格, "
              f"最大残差 {cc['max_residual_px']:.2f} px"))
    print(f"  {'系列':<9} {'颜色':<7} {'列数':>6} {'峰值 degC':>10} {'峰时 s':>9} "
          f"{'过 1290 列数':>12} {'不确定度 degC':>14}")
    for name, s in fig15["series"].items():
        d = s["derived"]
        print(f"  {name:<9} {s['colour']:<7} {len(s['samples']):6d} "
              f"{d['peak_temperature_C']:10.1f} {d['peak_time_s']:9.4f} "
              f"{d['n_columns_above_solidus']:12d} "
              f"{s['temperature_C_uncertainty']:14.1f}")

    print("\n=== Balbaa Fig 16 数字化(冷却速率 / 温度梯度)===")
    print(f"  源 {notes['fig16']}")
    c = fig16["calibration"]
    print(f"  类别 x 位置 {[round(v,1) for v in c['category_x_px']]} ({c['category_x_source']})")
    print(f"  图例排除 {c['legend_bbox_excluded']}")
    for key, unit in (("cooling_rate", "degC/s x 1e5"),
                      ("delta_T_x", "degC/mm"), ("delta_T_y", "degC/mm")):
        rows = fig16["series"][key]
        txt = ", ".join(f"{r['point']} {r['value']:.3f}+/-{r['uncertainty']:.3f}"
                        for r in rows)
        print(f"  {key:<13} [{unit:>12}]  {txt if txt else '(未检出)'}")

    if args.overlay:
        # 把追踪结果画回原图上。数字化最容易出的错是"看起来合理但追错了东西"
        # (框线被当成尖峰、注记被当成曲线),这张叠图是唯一能当场看出来的证据。
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(19, 5.6))
        ax = axes[0]
        ax.imshow(im15)
        c = fig15["calibration"]
        x0 = c["frame_px"]["left"]["center"]; y0 = c["frame_px"]["bottom"]["center"]
        sx_, sy_ = c["scale_s_per_px"], c["scale_C_per_px"]
        for name, ser in fig15["series"].items():
            xs = [p_["pixel_x"] for p_ in ser["samples"]]
            ys = [y0 + (p_["temperature_C"] - 0.0) / sy_ for p_ in ser["samples"]]
            ax.plot(xs, ys, ".", ms=1.2, color="lime", zorder=3)
        fr = c["frame_px"]
        ax.add_patch(plt.Rectangle(
            (fr["left"]["center"], fr["top"]["center"]),
            fr["right"]["center"] - fr["left"]["center"],
            fr["bottom"]["center"] - fr["top"]["center"],
            fill=False, ec="magenta", lw=1.5, zorder=4))
        if c["inset_bbox_excluded"]:
            r0, r1, c0, c1 = c["inset_bbox_excluded"]
            ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                       ec="cyan", lw=1.5, ls="--", zorder=4))
        ax.set_title("Fig 15 trace overlay (green = digitized, magenta = frame, "
                     "cyan = excluded inset)", fontsize=10)
        ax.axis("off")

        ax = axes[1]
        ax.imshow(im16)
        c = fig16["calibration"]
        y0 = c["frame_px"]["bottom"]["center"]
        for key, scale, lo, colour in (
                ("cooling_rate", c["scale_left_per_px"],
                 fig16["axes"]["left_ylim"][0], "lime"),
                ("delta_T_x", c["scale_right_per_px"],
                 fig16["axes"]["right_ylim"][0], "cyan"),
                ("delta_T_y", c["scale_right_per_px"],
                 fig16["axes"]["right_ylim"][0], "yellow")):
            for row, cx in zip(fig16["series"][key], c["category_x_px"]):
                ax.plot([cx], [y0 + (row["value"] - lo) / scale], "+",
                        ms=18, mew=2.5, color=colour, zorder=3)
        fr = c["frame_px"]
        ax.add_patch(plt.Rectangle(
            (fr["left"]["center"], fr["top"]["center"]),
            fr["right"]["center"] - fr["left"]["center"],
            fr["bottom"]["center"] - fr["top"]["center"],
            fill=False, ec="magenta", lw=1.5, zorder=4))
        if c["legend_bbox_excluded"]:
            r0, r1, c0, c1 = c["legend_bbox_excluded"]
            ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                       ec="red", lw=1.5, ls="--", zorder=4))
        ax.set_title("Fig 16 marker overlay (green = cooling rate, cyan = dTx, "
                     "yellow = dTy, red = excluded legend)", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.overlay, dpi=110)
        print(f"  写出叠图 {args.overlay}")

    problems = []
    for name, s in fig15["series"].items():
        if len(s["samples"]) < 100:
            problems.append(f"Fig 15 {name} 只追到 {len(s['samples'])} 列")
    for key in ("cooling_rate", "delta_T_x", "delta_T_y"):
        if len(fig16["series"][key]) != 3:
            problems.append(f"Fig 16 {key} 读到 {len(fig16['series'][key])} 点,应为 3")
    for p in problems:
        print(f"  [WARN] {p}")
    print(f"  写出 {args.output}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
