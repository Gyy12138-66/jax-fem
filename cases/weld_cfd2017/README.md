# weld_cfd2017 —— 2017 年激光-MIG 复合焊 CFD 温度场驱动的残余应力

A7N01 单道堆焊（bead-on-plate），板 44 × 20 × 6 mm。温度场来自 2017 年的 Fortran CFD 熔池程序
（桌面 `熔池温度场result/`，2026-09 重算的 v1.1 版），本案例只做力学：把 CFD 的节点温度历史插值到力学网格，
逐帧求解热弹塑性，得到残余应力与变形。**不做热求解。**

## 数据来源

温度历史 v1.1（`熔池温度场result/output/v1.1/`，已冻结，带 SHA256 清单）：Release 重算的 0 到 11.5 s 连续过程，
137 帧（加热 87 帧、冷却 50 帧），加热段 15 到 20 ms 一帧，凝固期 5 到 15 ms，全固后逐步放宽到 0.5 s。
加热段能量平衡每步 1.000 到 1.010，冷却段能量闭合 2.4e-3。

原 CFD 的已知偏差（本案例不修正，全部继承）：表面无蒸发上限（峰温 4000 K 以上、Marangoni 流速 100 m/s 量级）、
无洛伦兹力与自由表面变形反馈、板三面 298 K 定值边界、热源参数未对熔池截面标定。

## 坐标系（与 CFD 一致，不做轴交换）

| 轴 | 含义 |
|---|---|
| x | 沿焊缝，0 到 44 mm，热源 8 → 35 mm，18 mm/s，1.5 s 关断 |
| y | 垂直焊缝，0 到 10 mm（半模型），y=0 为对称面/焊缝中线 |
| z | 厚度，0 到 6 mm，z=6 为上表面 |

因此 **`sigma_xx` 是纵向（沿焊缝）、`sigma_yy` 是横向**。

## 流程

```
# 1 网格（横向与厚向双向渐变，沿焊缝均匀）
python model/make_plate_mesh.py --model half --dx 0.5 \
    --dy-fine 0.25 --y-fine 6 --dy-max 1.0 --dz-fine 0.25 --z-fine 3 --dz-max 0.5 \
    --out inputs/plate_half_cfdaxes_025mm.inp

# 2 温度映射（恒等映射，三线性插值，上限 1000 K）
python model/make_prescribed_temperature.py --v01 <v1.1 目录> \
    --inp inputs/plate_half_cfdaxes_025mm.inp --mapping identity --sym-centre 0.0 --cap 1000 \
    --out-npz inputs/prescribed_cfdaxes_025.npz --out-path inputs/prescribed_cfdaxes_path.csv \
    --report inputs/prescribed_cfdaxes_025_report.json

# 3 力学（CPU PARDISO，每帧一步，逐步输出）
INP=$PWD/inputs/plate_half_cfdaxes_025mm.inp TNPZ=$PWD/inputs/prescribed_cfdaxes_025.npz \
PATHCSV=$PWD/inputs/prescribed_cfdaxes_path.csv OUT=<输出目录> THREADS=16 \
  bash model/runs/run_cfd2017_prescribed.sh --solidification-reference solidus --vtu-quad-arrays off

# 4 测线与演化
python model/analyze_weld_stress.py <输出目录> --weld-axis x --station 0.022 --centre 0.0 --line-b-offset 0.0
python model/post_stress_evolution.py <输出目录>
```

大文件（`*.inp`、`*.npz`）在 `.gitignore` 里，只提交摘要与报告 JSON。

### 焊道余高（v1.2）

```
# 网格带焊道层（母板几何不变，输出 PLATE / BEAD 两个 ELSET）
python model/make_plate_mesh.py --model half --dx 0.5     --dy-fine 0.25 --y-fine 6 --dz-fine 0.25 --z-fine 3     --bead-height-file <bead_height.npz> --bead-layers 4 --bead-min-height 0.25     --out inputs/plate_half_bead_025mm.inp

# 温度映射：焊道节点 z 钳到板面（--clamp-top）+ 生成分段沉积列
python model/make_prescribed_temperature.py --v01 <v1.1 目录>     --inp inputs/plate_half_bead_025mm.inp --mapping identity --sym-centre 0.0 --cap 1000     --clamp-top --bead-elset BEAD --bead-x-range 0.0075,0.0405 --bead-segment-length 5e-4     --out-npz inputs/prescribed_bead_025.npz --out-path inputs/prescribed_bead_path.csv     --report inputs/prescribed_bead_025_report.json

# 力学：焊道随热源分段出生
INP=... TNPZ=... PATHCSV=... OUT=... bash model/runs/run_cfd2017_prescribed.sh     --solidification-reference solidus --vtu-quad-arrays off     --layer-activation-mode along_path --bead-elsets BEAD
```

VTU 里查看焊道：`bead`（1=焊道，Threshold 直接选）、`printed`（是否已出生）、`activation_step`（出生步）、
`activation_temperature`（出生温度，焊道为 943 到 1000 K）、`stress_free_temperature`（焊道全为固相线 858 K）。

结果（0.25 mm，55936 单元 = 母板 51832 + 焊道 4104）：焊道 100% 熔化过，截面纵向合力在屈服力的 0.03% 以内；
相对无焊道，峰值 von Mises 与塑性应变不变、纵向应力 +2%、角变形 +15%（0.256 → 0.295 mm）；焊道自身纵向拉伸均值
+60 到 +70 MPa、峰值 +202 MPa。**已登记偏差**：焊道暂用母材表，实际 ER5356 稀释焊缝屈服约为母材的 0.4 到 0.6。

### 后处理脚本 `model/post/`

`read_field.py`（读热学二进制帧）、`export_stress_vtr.py`（张量积网格转 VTR）、`make_animation.py`（pvpython 出动画）、
`make_bead_shell.py`（焊道灰壳）、`add_bead_field.py`（给旧 VTU 补 bead 字段）、`compare_runs.py` / `compare_history.py`。
这些是桌面 `熔池温度场result/output/v1.1|v1.2/tools/` 里同名文件的副本，放在这里是为了版本管理，后续可择一保留。

## 求解器开关（本分支新增，全部默认关闭，不改 LPBF 行为）

| 开关 | 作用 |
|---|---|
| `--prescribed-temperature-file` | 读外置节点温度历史（npz），跳过热求解，按步末时刻线性插值 |
| `--solidification-reference {temperature,solidus}` | 凝固时的无应力参考温度；粗帧下须用 solidus，否则参考温度跟着采样走 |
| `--born-phase {powder,solid}` | 单元出生即为实体（焊接单元生死语义） |
| `--bottom-mechanics-bc symmetry_plane` + `--symmetry-plane-axis/side` | 半模型：对称面法向固定 + 最小面内锚定 |
| `--bottom-mechanics-bc free_anchor` | 自由平放：仅 3-2-1 刚体锚定 |
| `--bottom-mechanics-bc edge_minimal` + `--edge-minimal-axis` | Lu 2020 对接板约束（自 songsi 移植） |
| `--reset-plastic-on-solidify` / `--elastic-melt` | 凝固清塑性 / 熔体弹性（自 songsi 移植） |
| `--vtu-quad-arrays off` | VTU 只存单元均值场，体积约为原来的 1/10 |
| `--output-times-file` | 按指定时刻输出 VTU |

VTU 单元场新增 `sigma_xx..xz`（8 个积分点均值）与 `von_mises`。

## 网格收敛（2026-09-10，半模型，Lu 2020 材料表，自由度依次 7.3 万 / 17.1 万 / 38.7 万）

| 焊缝区 dy=dz | 单元 | 137 步耗时 |
|---|---|---|
| 0.5 mm | 21120 | 482 s |
| 0.25 mm | 51832 | 1756 s |
| 0.15 mm | 120736 | 9255 s |

已收敛（0.5 mm 即可）：von Mises 峰 298.4 / 298.9 / 299.2 MPa；板底挠度 −0.256 / −0.256 / −0.257 mm；
熔合区体积 149.5 / 151.1 / 150.8 mm³；HAZ 体积 824.5 / 822.0 / 822.7 mm³；远场纵向应力 −294.6 / −294.9 / −294.9 MPa。

未收敛（焊缝区顶面剖面）：焊缝中心纵向拉应力 202 → 184 → 178 MPa，横向压应力 −60 → −83 → −91 MPa；
0.5→0.25 mm 最大差 48.1 MPa（均方根 17.7），0.25→0.15 mm 最大差 18.3 MPa（均方根 9.2）。约一阶收敛。

**建议**：趋势与流程用 0.5 mm，定量剖面用 0.25 mm；0.15 mm 只再改约 9 MPa 却要 2.5 小时。

## 约束敏感性（全板算例，2026-09-09，旧轴向）

| 约束 | 焊缝纵向平台 | 板边纵向 | eqp 峰 | 最大位移 |
|---|---|---|---|---|
| `edge_minimal`（一条底边全固定） | 213 到 247 MPa | −251 / −277 MPa | 0.109 | 0.14 mm |
| `free_anchor`（自由平放） | 195 到 215 MPa | −291 / −295 MPa | 0.041 | 0.28 mm |

两者之差即约束不确定度；实际实验的夹持方式待确认。

## 材料

`inputs/material/a7n01_material_config.json`，Lu 2020 J. Manuf. Processes 50:380-393 表 1（⑤级转引）。
E 70→3 GPa、σ_y 295→5 MPa、α 割线 2.38e-5→2.9e-5、ν 0.33，硬化取常数 1e8 Pa（论文无数据，**占位**）。
2026-09-09 的文献核查（`熔池温度场result/output/v1.1/material_literature_20260909/`）确认 20 到 450 °C 的量级与
7020 的独立数据一致（偏差 8 到 18%），但两项缺口会改变结果分布：焊缝金属（ER5356 焊态屈服 110 到 131 MPa，
约为母材的 0.4 到 0.6）与 HAZ 软化（硬度比 0.82）目前都按母材处理。

## 待办

1. 焊缝金属与 HAZ 的屈服折减（需要稀释率或焊缝拉伸数据）。
2. 板材状态 T4/T5/T6，决定室温屈服取 264 / 295 / 315 MPa。
3. 实验夹持方式与残余应力测量时刻。
4. 温度侧 v1.2：蒸发上限、热源标定；洛伦兹力耦合已在 `熔池温度场result/.../cooling_v2/` 验证为小量。
