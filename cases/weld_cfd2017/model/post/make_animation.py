# -*- coding: utf-8 -*-
"""
用 ParaView 的 pvpython 批量导出动画（mp4 / avi / ogv / png 序列）。

必须用 pvpython 运行，不是普通 python：
  "C:\\Program Files\\ParaView 6.1.0\\bin\\pvpython.exe" make_animation.py <参数>

例：
  # 力学：von Mises，带位移放大 30 倍
  pvpython make_animation.py paraview_stress_half_cfdaxes_025/stress.pvd out/vm.mp4 ^
      --field von_mises --assoc CELLS --range 0 3e8 --warp 30 --title "von Mises (Pa)"

  # 热学：温度，只看熔池附近
  pvpython make_animation.py paraview/field.pvd out/T.mp4 ^
      --field T --assoc POINTS --range 298 1000 --title "T (K)"

  # 纵向应力（沿焊缝 = sigma_xx）
  pvpython make_animation.py paraview_stress_half_cfdaxes_025/stress.pvd out/sxx.mp4 ^
      --field sigma_xx --assoc CELLS --vmin=-3e8 --vmax=3e8 --preset "Cool to Warm (Extended)"
  # 负数上下限必须写成 --vmin=-3e8 这种带等号的形式，否则 argparse 会当成选项名
"""
import argparse
import os
import sys

from paraview.simple import *  # noqa: F403


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pvd", help="输入的 .pvd 时间序列")
    p.add_argument("out", help="输出文件；扩展名 .mp4 / .avi / .ogv / .png（png 为序列）")
    p.add_argument("--field", default="von_mises")
    p.add_argument("--assoc", choices=("CELLS", "POINTS"), default="CELLS")
    p.add_argument("--range", nargs=2, type=float, default=None,
                   help="色标上下限；负数请用 --vmin/--vmax（argparse 会把 -3e8 当成选项）")
    p.add_argument("--vmin", type=float, default=None, help="色标下限（可为负；必须写成 --vmin=-3e8）")
    p.add_argument("--vmax", type=float, default=None, help="色标上限")
    p.add_argument("--preset", default=None, help='色标预设，例如 "Viridis (matplotlib)" / "Cool to Warm (Extended)" / "Black-Body Radiation"')
    p.add_argument("--warp", type=float, default=0.0, help="按位移矢量 u 放大变形的倍数；0 表示不变形")
    p.add_argument("--res", nargs=2, type=int, default=[1280, 720])
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--frames", nargs=2, type=int, default=None, help="帧区间 [首 末]，默认全部")
    p.add_argument("--title", default=None, help="色标标题")
    p.add_argument("--elevation", type=float, default=35.0)
    p.add_argument("--azimuth", type=float, default=-35.0)
    p.add_argument("--no-time", action="store_true", help="不显示时间标注")
    p.add_argument("--overlay", default=None, help="叠加显示的第二个 .pvd（灰色半透明参照，如焊缝隆起壳）")
    p.add_argument("--overlay-opacity", type=float, default=0.45)
    p.add_argument("--overlay-color", nargs=3, type=float, default=[0.55, 0.55, 0.55])
    a = p.parse_args(argv)

    src = OpenDataFile(a.pvd)
    if src is None:
        sys.exit("打不开 " + a.pvd)
    src.UpdatePipeline()
    ts = list(src.TimestepValues)
    # 关键：把动画时间轴绑定到数据的时间步，否则 SaveAnimation 只会重复渲染同一时刻
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    try:
        scene.PlayMode = "Snap To TimeSteps"
    except Exception:
        pass
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = a.res
    view.OrientationAxesVisibility = 1
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0

    obj = src
    if a.warp > 0.0:
        obj = WarpByVector(Input=src)
        obj.Vectors = ["POINTS", "u"]
        obj.ScaleFactor = a.warp

    disp = Show(obj, view)
    disp.SetRepresentationType("Surface")
    ColorBy(disp, (a.assoc, a.field))
    lut = GetColorTransferFunction(a.field)
    if a.preset:
        lut.ApplyPreset(a.preset, True)
    rng = a.range
    if a.vmin is not None and a.vmax is not None:
        rng = [a.vmin, a.vmax]
    if rng:
        lut.RescaleTransferFunction(rng[0], rng[1])
    else:
        disp.RescaleTransferFunctionToDataRangeOverTime()
    disp.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = a.title or a.field
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0, 0, 0]

    if not a.no_time:
        ann = AnnotateTimeFilter(Input=src)
        for fmt in ("t = {time:.3f} s", "t = %.3f s"):
            try:
                ann.Format = fmt
                break
            except Exception:
                continue
        ad = Show(ann, view)
        ad.Color = [0, 0, 0]
        ad.FontSize = max(a.res[1] // 40, 12)
        ad.WindowLocation = "Upper Left Corner"

    ov = None
    if a.overlay:
        ov = OpenDataFile(a.overlay)
        ov.UpdatePipeline()
        od = Show(ov, view)
        od.SetRepresentationType("Surface")
        ColorBy(od, ('POINTS', ''))   # 纯色，不按数据着色
        od.AmbientColor = od.DiffuseColor = a.overlay_color
        od.Opacity = a.overlay_opacity

    # 显式设置斜视相机：视线方向与 up 向量不共线，避免退化
    b = obj.GetDataInformation().GetBounds()
    cx, cy, cz = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])
    diag = max(((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2) ** 0.5, 1e-9)
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraPosition = [cx - 0.6 * diag, cy - 1.4 * diag, cz + 1.0 * diag]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    ResetCamera()
    Render()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    win = a.frames or [0, len(ts) - 1]
    SaveAnimation(a.out, view, ImageResolution=a.res, FrameWindow=win, FrameRate=a.fps)
    made = [n for n in os.listdir(os.path.dirname(os.path.abspath(a.out)))
            if n.startswith(os.path.splitext(os.path.basename(a.out))[0])]
    total = sum(os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(a.out)), n)) for n in made)
    print("已写出 %d 个文件, 共 %.1f MB；帧 %d..%d, 共 %d 帧, %d fps, 分辨率 %dx%d"
          % (len(made), total / 1e6, win[0], win[1], win[1] - win[0] + 1, a.fps, a.res[0], a.res[1]))
    print("时间范围 %.3f .. %.3f s" % (ts[win[0]], ts[win[1]]))


if __name__ == "__main__":
    main()
