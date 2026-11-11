#!/usr/bin/env python3
"""Balbaa Figs 43 / 44 (in-depth residual stress, XRD vs two ABAQUS predictions) digitized.

Same toolchain and uncertainty budget as digitize_fig14_pyrometer.py (frame-line
calibration cross-checked by the outward tick marks; marker anchor = bounding-box
centre with the centroid difference as ambiguity), applied to the page-35
EMBEDDED bitmaps (no re-render).

Evidence nature, stated first: the black circles are the ONLY experimental
residual-stress data in the paper (XRD after electropolishing, Sec 3.4); the
red triangles / blue squares are Balbaa's ABAQUS with the literature J-C set and
with his tensile-adjusted set (Sec 4.3.2). The y label is just "Residual
Stresses (MPa)" -- the paper measured scan- and hatch-direction stresses but the
figures do not say which component (or average) is plotted. Registered, not
resolved here.

Three series per figure. Hollow markers on solid connecting lines are separated
from the lines by hole-filling + erosion (markers become solid blobs, lines do
not); the experimental error bars are read as the dark vertical extent through
each circle centre.

    digitize_fig43_44.py --output ../inputs/balbaa-fig43-44-xrd-in-depth-rs.json [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
PDF = (V2.parent.parent / "references" / "docs" / "Balbaa2022_multiscale-RS-LPBF-IN625_JMMP-6-2.pdf")
SPEC = importlib.util.spec_from_file_location("fig14", HERE / "digitize_fig14_pyrometer.py")
FIG14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIG14)

FIGS = {
    # tick VALUE spacing of the detected (outward) ticks: Fig 43 draws a tick every 100 um / 100 MPa
    # (labels every 200), Fig 44 only every 200 -- verified from the tick counts (12 x 11 vs 7 x 7).
    # depths_um: the marker abscissae as read from the figures (XRD removal steps 100/200/300/450/600/800/1000 um;
    # model points every 100 um; Fig 44's literature-J-C series is plotted every 200 um from 100 to 1100).
    "fig43": {"index": 0, "condition": "220 W / 650 mm/s / hatch 0.12 mm", "x_major_um": 100.0, "y_major_MPa": 100.0,
              "depths_um": {"experimental_xrd": [100, 200, 300, 450, 600, 800, 1000],
                            "predicted_jc_literature": list(range(100, 1001, 100)),
                            "predicted_modified_jc": list(range(100, 1001, 100))}},
    "fig44": {"index": 1, "condition": "140 W / 650 mm/s / hatch 0.12 mm", "x_major_um": 200.0, "y_major_MPa": 200.0,
              "depths_um": {"experimental_xrd": [100, 200, 300, 450, 600, 800, 1000],
                            "predicted_jc_literature": [100, 300, 500, 700, 900, 1100],
                            "predicted_modified_jc": list(range(100, 1001, 100))}},
}


def column_read(mask, x_px, y_top, y_bottom, half_width=1):
    """Marker centre from the coloured pixels in the column through a known abscissa.

    All three series are drawn as centre-to-centre lines through hollow markers, so
    the column through a marker's abscissa meets: outline top, line/centre, outline
    bottom (and, for the XRD series, the error-bar caps). The mean of the run
    centres is the marker centre; the full extent is returned for error bars.
    """
    x = int(round(x_px))
    col = mask[int(y_top):int(y_bottom) + 1, max(x - half_width, 0):x + half_width + 1].any(axis=1)
    idx = np.where(col)[0]
    if idx.size == 0:
        return None
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            runs.append((start, a)); start = b
    runs.append((start, idx[-1]))
    centres = [0.5 * (a + b) for a, b in runs]
    return {"centre_px": float(np.mean(centres)) + int(y_top), "runs": len(runs),
            "extent_px": [float(idx[0]) + int(y_top), float(idx[-1]) + int(y_top)],
            "marker_half_height_px": float(0.5 * (idx[-1] - idx[0]))}


def page_images(pdf: Path, page_no: int, work: Path):
    import fitz
    doc = fitz.open(pdf)
    page = doc[page_no - 1]
    items = []
    for info in page.get_images(full=True):
        rects = page.get_image_rects(info[0])
        data = doc.extract_image(info[0])
        y = min(r.y0 for r in rects) if rects else 0.0
        items.append((y, data))
    items.sort(key=lambda t: t[0])           # top of page first: Fig 43 then Fig 44
    out = []
    for i, (_, data) in enumerate(items):
        p = work / f"balbaa_p{page_no}_img{i}.{data['ext']}"
        p.write_bytes(data["image"])
        out.append((p, data["width"], data["height"]))
    return out


def tick_calibration(ticks, major_value):
    """Ticks are at 0, major, 2*major, ... from the axis origin (first tick = 0)."""
    ticks = sorted(ticks)
    if len(ticks) < 3:
        raise SystemExit("need >= 3 major ticks for calibration")
    spacing = float(np.median(np.diff(ticks)))
    # robust linear fit value = major_value * k, k = round((t - t0)/spacing)
    k = np.array([round((t - ticks[0]) / spacing) for t in ticks], dtype=float)
    A = np.vstack([k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ticks), rcond=None)
    px_per_major, px0 = coef
    resid = np.array(ticks) - (px0 + k * px_per_major)
    return {"px_per_unit": px_per_major / major_value, "px_origin": px0,
            "max_residual_px": float(np.abs(resid).max()), "n_ticks": len(ticks), "k": k.tolist()}


def marker_blobs(mask, min_size, erode=None):
    """Hollow markers sitting on solid lines of the same colour.

    close (heals anti-aliased outline gaps) -> fill holes (markers become solid
    blobs, lines stay thin) -> erode (lines vanish, blob cores survive) ->
    keep compact blobs only (bbox 8-90 px, aspect 0.5-2, fill >= 0.3). The
    erosion radius is chosen automatically as the largest one that still
    leaves >= 5 compact blobs, so thin-outlined markers (Fig 43) and larger
    ones (Fig 44) are both resolved without per-figure tuning.
    """
    from scipy import ndimage
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    filled = ndimage.binary_fill_holes(closed)

    def compact(c):
        r0, r1, c0, c1 = c["bbox"]
        h, w = r1 - r0, c1 - c0
        if not (8 <= h <= 90 and 8 <= w <= 90):
            return False
        aspect = h / max(w, 1)
        return 0.5 <= aspect <= 2.0 and c["size"] >= 0.3 * h * w

    radii = [erode] if erode is not None else [5, 4, 3, 2]
    for r in radii:
        eroded = ndimage.binary_erosion(filled, structure=np.ones((2 * r + 1, 2 * r + 1)))
        blobs = [c for c in FIG14.clusters(eroded, min_size) if compact(c)]
        if len(blobs) >= 5:
            return blobs
    return []


def digitize(img_path: Path, spec: dict) -> dict:
    from PIL import Image
    im = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = (R > 150) & (G < 110) & (B < 110)
    blue = (B > 120) & (R < 120) & (B - R > 60)
    dark = (R < 110) & (G < 110) & (B < 110)
    frame = FIG14.find_frame(dark)
    x0p, y0p = frame["left"][0], frame["bottom"][0]
    x1p, y1p = frame["right"][0], frame["top"][0]
    ticks_x = FIG14.find_ticks(dark, frame, "x")
    ticks_y = FIG14.find_ticks(dark, frame, "y")
    cal_x = tick_calibration(ticks_x, spec["x_major_um"])
    cal_y = tick_calibration(ticks_y, spec["y_major_MPa"])   # y ticks listed top->bottom; px increases downward
    # y: value = (px_origin_bottom - px) / px_per_unit ; ticks sorted ascending px = descending value
    ys = sorted(ticks_y)
    y_px_per_unit = abs(cal_y["px_per_unit"])
    y_origin_px = max(ys)                                     # the 0 MPa tick (lowest on the page)
    x_origin_px = min(sorted(ticks_x))                        # the 0 um tick (leftmost)
    x_px_per_unit = cal_x["px_per_unit"]

    inner = np.zeros_like(dark)
    inner[int(y1p) + 6:int(y0p) - 5, int(x0p) + 6:int(x1p) - 5] = True
    boxes = sorted(FIG14.clusters(dark & inner, 3000), key=lambda c: -c["size"])
    # legend = largest dark component with a large, hollow bounding box
    legend = None
    for b in boxes:
        r0, r1, c0, c1 = b["bbox"]
        if (r1 - r0) > 0.15 * dark.shape[0] and (c1 - c0) > 0.25 * dark.shape[1]:
            legend = b["bbox"]; break

    def outside_legend(c):
        if legend is None:
            return True
        r0, r1, c0, c1 = legend
        y, x = c["bbox_center"]
        return not (r0 - 5 <= y <= r1 + 5 and c0 - 5 <= x <= c1 + 5)

    def to_data(c):
        py, px = c["bbox_center"]; cy, cx = c["centroid"]
        return {"depth_um": (px - x_origin_px) / x_px_per_unit,
                "stress_MPa": (y_origin_px - py) / y_px_per_unit,
                "pixel_bbox_center": [float(py), float(px)],
                "anchor_ambiguity_px": [abs(cy - py), abs(cx - px)]}

    # PRIMARY reading: column through each known abscissa (robust to thick lines / small markers).
    # The legend box (frame, text, sample markers) is blanked first: a column through the legend
    # would otherwise pick up its dark/coloured pixels (seen on the 800/1000 um XRD points).
    plot = inner.copy()
    if legend is not None:
        r0, r1, c0, c1 = legend
        plot[max(r0 - 8, 0):r1 + 9, max(c0 - 8, 0):c1 + 9] = False
    masks = {"predicted_jc_literature": red & plot, "predicted_modified_jc": blue & plot, "experimental_xrd": dark & plot}
    y_top_in, y_bot_in = int(y1p) + 6, int(y0p) - 6
    series = {}
    blob_cross = {}
    for name, mask in masks.items():
        pts = []
        for depth in spec["depths_um"][name]:
            x_px = x_origin_px + depth * x_px_per_unit
            r = column_read(mask, x_px, y_top_in, y_bot_in)
            if r is None:
                pts.append({"depth_um": float(depth), "stress_MPa": None, "missing": True}); continue
            p = {"depth_um": float(depth), "stress_MPa": (y_origin_px - r["centre_px"]) / y_px_per_unit,
                 "column_runs": r["runs"], "marker_half_height_MPa": r["marker_half_height_px"] / y_px_per_unit}
            if name == "experimental_xrd":
                # error bar = full dark extent of the column (caps included); asymmetric about the centre
                p["err_plus_MPa"] = (r["centre_px"] - r["extent_px"][0]) / y_px_per_unit
                p["err_minus_MPa"] = (r["extent_px"][1] - r["centre_px"]) / y_px_per_unit
            pts.append(p)
        series[name] = pts
        # CROSS-CHECK: independent blob detection; report the nearest blob within 30 um of each abscissa
        blobs = [to_data(c) for c in marker_blobs(mask, 40) if outside_legend(c) and c["size"] < 4000]
        cross = []
        for p in pts:
            near = [b for b in blobs if abs(b["depth_um"] - p["depth_um"]) < 30.0]
            if near and p.get("stress_MPa") is not None:
                b = min(near, key=lambda b: abs(b["depth_um"] - p["depth_um"]))
                cross.append({"depth_um": p["depth_um"], "blob_stress_MPa": b["stress_MPa"],
                              "delta_MPa": b["stress_MPa"] - p["stress_MPa"]})
        blob_cross[name] = cross

    px_uncertainty = max(frame["left"][1], frame["bottom"][1], cal_x["max_residual_px"], cal_y["max_residual_px"])
    return {
        "condition": spec["condition"], "image": str(img_path), "image_shape": list(dark.shape),
        "calibration": {"x": cal_x, "y": cal_y, "x_origin_px": float(x_origin_px), "y_origin_px": float(y_origin_px),
                        "frame": {k: [float(v[0]), float(v[1])] for k, v in frame.items()}, "legend_bbox": legend},
        "read_off_uncertainty": {
            "depth_um": float(px_uncertainty / x_px_per_unit),
            "stress_MPa": float(px_uncertainty / y_px_per_unit),
            "note": "worst of frame-line half thickness and tick-grid residual, mapped through the calibration; "
                    "per point, marker_half_height_MPa bounds the centre-finding ambiguity of the column read "
                    "(the column through a marker sees its outline top/bottom and the centre line; their mean is "
                    "the anchor). The XRD error bars are read as the full dark extent of the column."},
        "series": series,
        "blob_cross_check": blob_cross,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--page", type=int, default=35)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    work = args.work_dir or (args.output.parent / "digitize_work")
    work.mkdir(parents=True, exist_ok=True)
    imgs = page_images(args.pdf, args.page, work)
    out = {
        "schema": "v2.balbaa-fig43-44/1",
        "source": f"{args.pdf.name} page {args.page}, embedded bitmaps at native resolution",
        "evidence_nature": "experimental_xrd = the paper's only measured residual stresses (XRD + electropolishing, "
                           "Sec 3.4, 2 mm spot, cube centre, two samples 220 W / 140 W); predicted_* = Balbaa's ABAQUS "
                           "(literature J-C vs tensile-adjusted J-C, Sec 4.3.2). The plotted component (scan, hatch or "
                           "average) is NOT stated in the paper.",
        "figures": {},
    }
    for name, spec in FIGS.items():
        img_path, w, h = imgs[spec["index"]]
        out["figures"][name] = digitize(img_path, spec)
        f = out["figures"][name]
        print(f"{name} ({spec['condition']}): image {w}x{h}, read-off +/-{f['read_off_uncertainty']['stress_MPa']:.1f} MPa "
              f"/ +/-{f['read_off_uncertainty']['depth_um']:.1f} um; ticks x {f['calibration']['x']['n_ticks']} y {f['calibration']['y']['n_ticks']}")
        for sname, pts in f["series"].items():
            print(f"  {sname:24s} n={len(pts)}: " + ", ".join(
                (f"{p['depth_um']:.0f}:{p['stress_MPa']:.0f}" + (f"(+{p['err_plus_MPa']:.0f}/-{p['err_minus_MPa']:.0f})" if 'err_plus_MPa' in p else ""))
                if p.get("stress_MPa") is not None else f"{p['depth_um']:.0f}:MISSING"
                for p in pts))
            cross = f["blob_cross_check"][sname]
            if cross:
                print(f"    blob cross-check: {len(cross)} matched, max |delta| = {max(abs(c['delta_MPa']) for c in cross):.1f} MPa")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("written", args.output)


if __name__ == "__main__":
    main()
