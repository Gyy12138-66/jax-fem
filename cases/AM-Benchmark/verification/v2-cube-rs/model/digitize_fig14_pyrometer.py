#!/usr/bin/env python3
"""Balbaa Fig 14(实验 vs 数值温度)数字化 —— 沿用 V1 的数字化工作流。

**先说一个尺度上的事实**:Fig 14 不是连续的高温计**轨迹**,而是 5 组离散
点对(5 个黑方块 = 实验,5 个红三角 = Balbaa 的 ABAQUS),横跨
t = 0.548 .. 0.733 s。热闸门能对上的观测量因此是 5 个点,不是一条曲线;
这一点直接决定了闸门的判读口径(见 THERMAL-GATE.md)。

做法:
  1. 从仓库内 PDF 抽出该页的内嵌位图(2755 x 1804,原始分辨率,
     不经二次渲染,避免重采样引入的读数误差);
  2. 由长直深色行/列定出绘图框 -> 一次线性标定;
  3. 用框外的刻度线做**独立**的第二次标定,两者之差进入读数不确定度
     (而不是假设框线就是坐标轴);
  4. 红像素 / 深色像素分别连通域聚类,落在图例框内的丢弃;
  5. 标记锚点取包围盒中心(绘图软件的标记锚点),同时算像素质心,
     两者之差作为"标记锚点歧义"计入不确定度。

零标定:这里只读图,不拟合、不平滑、不向任何模型结果靠拢。
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


def extract_page_image(pdf, page_no, out_png):
    """抽该页最大的内嵌位图。需要 PyMuPDF;没有就报清楚,别静默降级。"""
    try:
        import fitz
    except ImportError:
        raise SystemExit(
            "需要 PyMuPDF 抽图:pip install pymupdf,"
            "或先手工导出该页位图再用 --image 传入")
    doc = fitz.open(pdf)
    page = doc[page_no - 1]
    best = None
    for info in page.get_images(full=True):
        data = doc.extract_image(info[0])
        if best is None or data["width"] * data["height"] > best["width"] * best["height"]:
            best = data
    if best is None:
        raise SystemExit(f"{pdf} 第 {page_no} 页没有内嵌位图")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.write_bytes(best["image"])
    return out_png, best["width"], best["height"]


def find_frame(dark):
    """绘图框:横跨图幅 40% 以上的深色行/列。返回 (中心, 半厚) 四元组。"""
    h, w = dark.shape

    def bands(counts, threshold):
        hits = np.where(counts > threshold)[0]
        groups = []
        for i in hits:
            if groups and i - groups[-1][-1] <= 2:
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    rows = bands(dark.sum(axis=1), 0.4 * w)
    cols = bands(dark.sum(axis=0), 0.4 * h)
    if len(rows) < 2 or len(cols) < 2:
        raise SystemExit(f"找不到绘图框(rows={len(rows)}, cols={len(cols)})")
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    fr = lambda g: (float(np.mean(g)), 0.5 * len(g))
    return {"top": fr(top), "bottom": fr(bottom), "left": fr(left), "right": fr(right)}


def find_ticks(dark, frame, axis, tick_band=20, min_len=8):
    """框外刻度线的位置,用作独立的第二次标定。"""
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
    # 主刻度比次刻度长:按最长的一半为界筛出主刻度
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


def clusters(mask, min_size):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        sel = lab[sl] == i
        size = int(sel.sum())
        if size < min_size:
            continue
        ys, xs = np.where(lab == i)
        out.append({
            "size": size,
            "bbox": (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop),
            "bbox_center": (0.5 * (sl[0].start + sl[0].stop - 1),
                            0.5 * (sl[1].start + sl[1].stop - 1)),
            "centroid": (float(ys.mean()), float(xs.mean())),
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--page", type=int, default=17, help="Fig 14 所在页")
    ap.add_argument("--image", type=Path, default=None, help="跳过 PDF,直接用位图")
    ap.add_argument("--xlim", default="0.5,0.75", help="横轴框线对应值 [s]")
    ap.add_argument("--ylim", default="1000,1700", help="纵轴框线对应值 [degC]")
    ap.add_argument("--min-marker-px", type=int, default=400)
    ap.add_argument("--work-dir", type=Path, default=None, help="位图落地目录")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from PIL import Image

    if args.image:
        img_path, note = args.image, "外部位图"
    else:
        work = args.work_dir or Path(".")
        img_path, w, h = extract_page_image(args.pdf, args.page,
                                            work / f"balbaa_p{args.page}_fig14.jpg")
        note = f"仓库内 PDF 第 {args.page} 页内嵌位图 {w}x{h}(原始分辨率,未二次渲染)"

    im = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = (R > 120) & (R - G > 60) & (R - B > 60)
    dark = (R < 110) & (G < 110) & (B < 110)

    frame = find_frame(dark)
    x0p, y0p = frame["left"][0], frame["bottom"][0]
    x1p, y1p = frame["right"][0], frame["top"][0]
    xlo, xhi = (float(v) for v in args.xlim.split(","))
    ylo, yhi = (float(v) for v in args.ylim.split(","))
    sx = (xhi - xlo) / (x1p - x0p)          # s per px
    sy = (yhi - ylo) / (y1p - y0p)          # degC per px (y1p < y0p -> sy < 0)

    ticks_x = find_ticks(dark, frame, "x")
    ticks_y = find_ticks(dark, frame, "y")
    cross = {"x": calib_residual(ticks_x, x0p, x1p),
             "y": calib_residual(ticks_y, y1p, y0p)}

    # 图例框:内部最大的深色连通域(带大包围盒);其内的一切都不是数据
    inner = np.zeros_like(dark)
    inner[int(y1p) + 6:int(y0p) - 5, int(x0p) + 6:int(x1p) - 5] = True
    dark_in, red_in = dark & inner, red & inner
    boxes = sorted(clusters(dark_in, 3000), key=lambda c: -c["size"])
    legend = boxes[0]["bbox"] if boxes else None

    def outside_legend(c):
        if legend is None:
            return True
        r0, r1, c0, c1 = legend
        y, x = c["bbox_center"]
        return not (r0 <= y <= r1 and c0 <= x <= c1)

    numerical = [c for c in clusters(red_in, args.min_marker_px) if outside_legend(c)]
    experimental = [c for c in clusters(dark_in, args.min_marker_px)
                    if outside_legend(c) and c["size"] < 3000]

    def to_data(c):
        py, px = c["bbox_center"]
        cy, cx = c["centroid"]
        return {
            "time_s": xlo + (px - x0p) * sx,
            "temperature_C": ylo + (py - y0p) * sy,
            "pixel_bbox_center": [py, px],
            "anchor_ambiguity_px": [abs(cy - py), abs(cx - px)],
            "marker_px": c["size"],
        }

    num = sorted((to_data(c) for c in numerical), key=lambda d: d["time_s"])
    exp = sorted((to_data(c) for c in experimental), key=lambda d: d["time_s"])

    # ---- 读数不确定度(逐点),V1 口径:分项列出,再方和根 ----
    frame_half = max(frame["top"][1], frame["bottom"][1],
                     frame["left"][1], frame["right"][1])
    resid_x = cross["x"]["max_residual_px"] if cross["x"] else 0.0
    resid_y = cross["y"]["max_residual_px"] if cross["y"] else 0.0

    def budget(points):
        for p in points:
            ay, ax = p["anchor_ambiguity_px"]
            terms_T = {"frame_line_half_thickness": frame_half * abs(sy),
                       "tick_cross_check": resid_y * abs(sy),
                       "marker_anchor": ay * abs(sy)}
            terms_t = {"frame_line_half_thickness": frame_half * abs(sx),
                       "tick_cross_check": resid_x * abs(sx),
                       "marker_anchor": ax * abs(sx)}
            p["temperature_C_uncertainty"] = float(
                np.sqrt(sum(v * v for v in terms_T.values())))
            p["time_s_uncertainty"] = float(
                np.sqrt(sum(v * v for v in terms_t.values())))
            p["uncertainty_terms_C"] = terms_T
            p["uncertainty_terms_s"] = terms_t
            p["temperature_K"] = p["temperature_C"] + 273.15
        return points

    num, exp = budget(num), budget(exp)

    # 实验/数值同一时刻成对;配对残差是第三个独立的一致性检查
    pairs = []
    for e in exp:
        m = min(num, key=lambda n: abs(n["time_s"] - e["time_s"]))
        pairs.append({
            "time_s": 0.5 * (e["time_s"] + m["time_s"]),
            "time_mismatch_s": abs(e["time_s"] - m["time_s"]),
            "experimental_C": e["temperature_C"],
            "balbaa_numerical_C": m["temperature_C"],
            "balbaa_minus_experiment_C": m["temperature_C"] - e["temperature_C"],
        })

    doc = {
        "id": "balbaa2022-fig14-pyrometer",
        "provenance": f"Balbaa2022 Figure 14(Sec 4.2.1,p.17):{note}",
        "what": "多道模型温度预测 vs 双色高温计实测",
        "condition": {"power_W": 220, "speed_mm_s": 650, "hatch_mm": 0.12,
                      "layer_mm": 0.04,
                      "_source": "Balbaa2022 Sec 3.3"},
        "instrument": ("Fluke Endurance 双色高温计,Si/Si ~1 um,slope 0.96,"
                       "量程 1000-3000 degC,响应 10 ms,85 mm 处光斑 2 mm"),
        "SCOPE_FINDING": (
            "Fig 14 是 5 组离散点对,不是连续轨迹。可比对的观测量因此是 5 个"
            f"采样时刻(t = {exp[0]['time_s']:.3f} .. {exp[-1]['time_s']:.3f} s),"
            "而不是一条 10 ms 分辨的温度曲线;Sec 3.3 的 10 ms 只是模型侧的"
            "抽样/平均增量,不代表图里有 10 ms 密度的数据"),
        "axes": {"x": "Time (s)", "y": "Temperature (degC)",
                 "xlim": [xlo, xhi], "ylim": [ylo, yhi]},
        "calibration": {
            "frame_px": {k: {"center": v[0], "half_thickness": v[1]}
                         for k, v in frame.items()},
            "scale_s_per_px": sx, "scale_C_per_px": sy,
            "tick_cross_check": cross,
            "legend_bbox_excluded": list(legend) if legend else None,
        },
        "experimental": exp,
        "balbaa_numerical": num,
        "pairs": pairs,
        "read_off_uncertainty_note": (
            "逐点分项:框线半厚 + 刻度线独立标定的最大残差 + 标记锚点歧义"
            "(包围盒中心 vs 像素质心),方和根合成。未含 Balbaa 制图本身的"
            "误差与高温计自身的测量不确定度 —— 后者论文未给"),
        "zero_calibration_note": "只读图,不拟合、不平滑、不向任何模型结果靠拢",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"=== Balbaa Fig 14 数字化 ===\n  源 {note}")
    print(f"  框 x[{x0p:.1f},{x1p:.1f}] y[{y1p:.1f},{y0p:.1f}] px  "
          f"-> {sx*1e3:.5f} ms/px, {abs(sy):.4f} degC/px")
    for axis, c in cross.items():
        print(f"  刻度交叉检查 {axis}: " + ("未检出" if not c else
              f"{c['n_major_ticks']} 主刻度, 框跨 {c['intervals_across_frame']} 格, "
              f"最大残差 {c['max_residual_px']:.2f} px"))
    print(f"  实验 {len(exp)} 点 / 数值 {len(num)} 点")
    print(f"  {'t [s]':>8} {'实验 [degC]':>14} {'Balbaa 数值 [degC]':>18} {'差 [degC]':>12}")
    for p in pairs:
        print(f"  {p['time_s']:8.4f} {p['experimental_C']:14.1f} "
              f"{p['balbaa_numerical_C']:18.1f} {p['balbaa_minus_experiment_C']:12.1f}")
    u_T = max(p["temperature_C_uncertainty"] for p in exp + num)
    u_t = max(p["time_s_uncertainty"] for p in exp + num)
    print(f"  读数不确定度(最大):温度 +/-{u_T:.1f} degC,时间 +/-{u_t*1e3:.2f} ms")
    print(f"  写出 {args.output}")
    if len(exp) != 5 or len(num) != 5:
        print("  [WARN] 点数不是 5+5,聚类阈值可能要调 (--min-marker-px)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
