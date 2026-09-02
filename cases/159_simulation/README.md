# 159_simulation — 0119 薄壁件真实工况热—力应力算例

承接 `cases/AM-Benchmark/verification/v2-cube-rs` 的生产合同（闪热读法 A'、IN625 材料链、GPU 装配 + PARDISO），
迁到真实 3D 输入模型 `0119`（来源 `~/work/159/schema/0119_c3d4_only.inp`，HyperMesh TET4 抽取网格）。

## 几何（`mesh_check/`）

- 弯折薄板"发夹"件：180° 弯（外半径 ≈ 20 mm）+ 两条平行 45° 斜腿，腿端各有一个约 90° 转出的厚凸耳；
  沿原 x 拉伸 90.9 mm，凸耳随高度缩短，x ≈ 63 mm 处凸耳消失（截面 882 → 255 → 107 mm²）。
- **板厚 0.80 mm**（z > 7.5 mm 全程恒定）；底部 z < 6.5 mm 为 3.3 mm 厚的脚，2.5–7.5 mm 为过渡倒角。
- 单位：米（Parasolid 原生）；原坐标偏在 x ≈ 3.75 m，已平移。
- 原 TET4：197,266 单元 / 52,739 节点，共形、单连通、表面封闭；棱长中位 1.03 mm（壁厚方向 1 个单元），533 个薄片单元。
  只适合做热-only 试跑，不能做残余应力（壁厚方向无分辨、TET4 + J2 锁定）。

## HEX8 网格（`mesh/make_0119_hex_mesh.py` → `mesh/out/0119_hex_raft.{inp,vtu}`）

建造方向 = 原 x → 新 +Z（右手旋转 (x,y,z)→(y,z,x)，det=+1），U 截面在 X–Y 平面，底座底面 Z = 0，零件底面 Z = 2 mm。

方法：分块结构化的曲线坐标体素。以 z = 30 mm 截面的板中线为脊线 s（两端直线外延 32 mm 以覆盖底部更长的腿和凸耳），
法向 t 为厚度坐标（datum 对齐到最常见的表面偏移 = 板外表面，因此 0.8 mm 板的外表面精确、内表面落在网格线上）；
|t| ≤ 4 mm 的核心带用 0.267 mm 均匀单元（板 + 脚），核心带外按 1.3 倍几何增长到 1.5 mm（凸耳）。
**凸耳**：每 1 mm 高度在切片上检测凸耳两条长面（逐面直线拟合），得到方向（随高度不变，与法向夹 11–15°）、
位置 s1(z)/s2(z)（凸耳沿腿滑动约 15 mm）和长度 t_end(z)；每个高度把 (a) 凸耳窗内的 s 网格拉伸使 s1、s2 落在网格线上，
(b) t 线剪切到凸耳面方向，(c) 核心带外的 t 缩放使凸耳端面落在网格面上 → 凸耳是整块，不再是台阶。
每 0.2 mm 站位切一次 TET4 表面，按 3×3 子采样 ≥ 50 % 在截面内保留单元；腿端 10 mm 以外只允许核心带（防止两腿图卡重叠）。

| 项 | 值 |
|---|---|
| 单元尺寸 | 建造 z 0.2 mm × 厚度 t 0.267 mm（板厚恰 3 个单元，凸耳内渐变到 1.5 mm）× 沿壁 s 2.3 mm（弯处细化到 0.8 mm，凸耳窗 2.0 mm） |
| 单元数 | **204,899**（零件 191,899 + 底座 13,000） |
| 节点 / 自由度 | 268,740 / 力学 806,220（立方体生产算例 0.67 M —— **内存要先探针**） |
| 体积 | 零件 / TET4 = 0.986（板区 0.993，凸耳区 0.96–0.98，倒角区 0.97） |
| 质量 | 中心 Jacobian 全部 > 0；截面最小角 25°（23 个单元 < 30°，均在凸耳根部圆角）；无重合节点；`mesh_check/check_hex_layers.py` PASS |
| 表面贴合 | 侧向边界节点投影到 TET4 真实表面（83k 节点，残差 ≤ 0.12 mm）；表面距网格线 > 0.45 单元的 41k 台阶/特征端节点不动 |
| slab | 455 层 × 0.2 mm；`--layers 455`，slab 号 = ceil(z_c / 0.2 mm)（含底座偏移见下） |
| 底座 | 2 mm（10 层），截面 = 底部站位截面，ELSET=RAFT；零件 ELSET=PART；底面节点 NSET=BASE |

已知近似（都 ≤ 一个单元）：腿端端面和凸耳端面随高度倾斜 → 0.2 mm 层间的小台阶；脚的倒角（z 2.5–7.5 mm）和凸耳上的
缺口/通孔按体素处理；凸耳位置轨迹的检测残差 ≤ 2 mm 时个别高度凸耳面会跳一格（2 mm）。

重新生成：
```
python mesh/make_0119_hex_mesh.py --inp ~/work/159/schema/0119_c3d4_only.inp --out-dir mesh/out
```
参数见 `mesh/out/0119_hex_raft_report.json`；图：`*_views.png`（三视图，底座红）、`*_sections.png`（8 个站位 HEX8 vs TET4 切片）、`*_base_zoom.png`。

## 分级网格（用户要求：厚脚粗、薄壁细；2026-08-28，`mesh/out_graded/0119_hex_graded.{inp,vtu}`）

厚度方向非对称分级：板外表面为 datum，板内 **4 层 0.2 mm**（0.8 mm 板两面都落在网格线上），datum 外侧 2 层 0.2 mm（脚的外凸、凸耳根部），
再向外/向内按 1.4 倍渐变到 1.0 mm（脚的多余厚度 4 层、凸耳沿轴 ~20 层）；沿壁 2.4 mm（弯处 0.8，凸耳窗 1.8）；建造方向 0.2 mm 不变；底座 1 mm（5 层）。
结果 **223,400 HEX8**（零件 218,165 + 底座 5,235）、284,759 节点、**0.85 M 力学 DOF**，J>0，最小角 28°，体积比 0.985，`check_hex_layers.py` PASS；
表面 snapping 同前（90k 节点贴合，残差 ≤ 0.3 mm）。生成命令：
`python mesh/make_0119_hex_mesh.py --inp ~/work/159/schema/0119_c3d4_only.inp --out-dir mesh/out_graded --tag 0119_hex_graded --hs 2.4e-3 --lug-hs 1.8e-3 --raft 1e-3`
（`--ht 0.2e-3 --n-inner 4 --n-outer 2 --t-growth 1.4 --t-max 1.0e-3` 为默认）。沿壁尺寸不能随高度变（结构化网格拓扑固定），
"底部粗"只体现在厚度方向；`--hs 2.0` 版本 27.2 万单元 / 1.03 M DOF 超出显存上限，故取 2.4。

分级网格三级门槛（TAG=graded）：能量闭合 0.99968；2-slab shakedown 10/10（0.85 M DOF 释放通过）。用户复看后仍不满意，未用于生产。

## HyperMesh 体素网格（用户选定 2026-08-28，`mesh/out_voxel/0119_hm_voxel.{inp,vtu}`，config `inputs/0119-flash-voxel.json`）

`hm/voxel.tcl`：hmbatch 批处理导入 STEP（mm）→ `*voxel_lattice_hex_mesh_init 0.5` + `add_entities solids 1 3` + `create` → 导出 292,112 个 0.5 mm 体素
（mode 3 = 触到实体即保留，体积 +44%；mode 0/1/2 不出单元）。`mesh/convert_hm_voxel.py`：只保留**中心在 TET4 表面内**的体素、旋转到 +Z、换米、
底座向下复制 2 层、ELSET PART/RAFT、NSET BASE → 197,381 零件 + 7,020 底座，301,456 节点，**0.90 M DOF**，182 层 × 0.5 mm，体积比 0.973，验收 PASS。

**合同扩展（闪热分组）**：0.5 mm slab = 12.5 个 40 µm 真实层 → 每 slab 5 次闪热、每次携带 2.5 层的能量与时钟（`physical_layers_per_slab` 5 +
`flash_grouping`，预检里 `real_layers_per_flash` = 2.5），闪热的"能量/单元质量"与验证过的 0.2 mm 读法相同（峰温 4604 K vs 立方体 4437）；
深度带按 0.5 mm 单元重推：d = 314.182 µm、切带 499 µm、离散积分 1.000000。总能量/时钟不变（吸收 686 kJ，19,870 s），步数 20,070。
门槛（TAG=voxel）：能量试跑闭合 **1.000000**（体素与顶点采样规则精确吻合）；2-slab shakedown 10/10：226 步 717 s，eqp 0.023，
vm 799/908 → 释放后 193/481 MPa，释放切掉全部 7,020 底座单元，0.90 M DOF 释放通过。已知代价：0.8 mm 板只有 1–2 个体素，表面台阶 0.5 mm。

**生产运行（体素网格）**：`CFG=inputs/0119-flash-voxel.json STAGES=3 TAG=voxel` 于 2026-08-28 12:18 UTC 起跑（用户拍板），
`~/work/159/output/v159_voxel/production/`，20,070 步，预估 ≈ 20 h，结束自动跑 `check_159.py`。

## 路线 B：HyperMesh 从 `0119.stp` 重划六面体（用户执行）

目标与本网格相同：建造方向（原 x）0.2 mm 层、板厚 3 层 0.267 mm、沿壁约 2 mm、凸耳约 2 × 1.5 mm。几何按高度分 3 段 solid map，
每段源/目标面拓扑一致（HyperMesh 线性插值正好匹配凸耳的线性滑动，b ≈ 0.27–0.29 mm/mm）：

| 段（原 x） | 截面特征 | 做法 |
|---|---|---|
| 0 → 7.5 mm | 3.3 mm 脚 + 倒角 + 凸耳 | solid map：x=0 面 2D 四边形网格（脚 12 层 0.267）→ x=7.5 mm 面（板 3 层）；38 层 |
| 7.5 → 63 mm | 0.8 mm 板 + 滑动凸耳 | solid map（general）：两端面拓扑一致；凸耳上的横向通孔（x≈7、16 mm）和缺口先 defeature；278 层 |
| 63 → 90.9 mm | 0.8 mm U，无凸耳 | drag：63 mm 端面去掉凸耳后沿 x 拖 140 层 |
| 底座 | = x=0 截面 | 把 x=0 面网格向 −x 拖 2 mm（10 层），ELSET=RAFT |

导出：Abaqus Standard 3D，C3D8，单位 mm 即可（runner 用 `--mesh-length-scale 1e-3`），零件 ELSET=PART、底座 ELSET=RAFT，
段间节点必须合并（equivalence），不要 tie。建造方向可以留在 x，我用脚本旋转到 +Z。验收：
`python mesh_check/check_hex_layers.py your.inp --build-axis x --scale 1e-3 --dz 2e-4 --ref-tet ~/work/159/schema/0119_c3d4_only.inp`
（检查 C3D8 单族、节点落在 0.2 mm 层格上、每单元恰跨一层、共形、Jacobian、ELSET、体积比）。预计 20–25 万单元。

## 能量台账链（`inputs/0119-flash.json` → `model/make_159_preflight.py` → `model/runs/v_159.sh`）

cube-rs 生产合同（闪热读法 A'）迁到任意分层 HEX8 网格：几何量全部从网格取，换网格 = 重跑预检。

- 逐 slab 面积 A(k) = slab 内单元体积 / 0.2 mm；每真实层扫描时间 t_scan = A/(h·v)，能量 P·t_scan；
  闪热 3.08e-4 s（2 子步）+ 关灯保温（10 几何子步）+ 铺粉 5 s（10 子步）= 每层 22 行；
  闪热半径 1 m（96 mm 件上均匀性 0.9935），捕获率按 slab 在网格上算（Σ 单元顶面积 × 高斯 × 2/πr²），
  命令功率 = P·t_scan/(t_flash·capture)，逐行写进 CSV 的 `power` 列，因此每层沉积能量恒等于 Ac·P·t_scan。
- 合同：`--layers 455 --support-thickness 0.002`（底座 = 夹具带），释放盒 = 底座包围盒，其余与 cube 生产合同一致
  （深度带、哨兵熔点、T_ref 1273.15、力学每 22 步、GPU 装配 + PARDISO、批 32768）。
- 阶段：`STAGES=1`（预检 + 指纹）、`2`（第 1 真实层热-only 能量试跑 + 门槛）、`S`（2-slab shakedown，带力学与释放）、`3`（全程）。
  输出根 `~/work/159/output/v159_<TAG>/`。

首轮（网格 = 路线 A snapping 版，2026-08-28）：455 slab / 2,275 层，50,100 步，建造钟 19,979 s，吸收能量 695.2 kJ；
能量试跑：沉积 989.33 J / 物理 989.61 J（0.99972），捕获率 5.645e-4 vs 网格预测 5.647e-4，平衡误差 8e-6，
闪热节点峰温 4583 K（立方体 4437 K，同一量级），11 s 内回到预热温度；热-only 3.96 s/步（含 JIT）。
2-slab shakedown（10 真实层 + 600 s 冷却/6 步 + 释放底座，226 步，774 s，3.43 s/步，15 次力学求解）：**门槛 10/10**；
能量闭合 0.99973；eqp 峰 0.0164；件内 vm 均值/峰 792/862 MPa → 释放后 187/532 MPa；u_max 27 → 424 µm；
释放切掉全部 13,000 底座单元；**释放求解在 0.81 M DOF 下通过，GPU 峰 12.3 / 16.3 GB**（立方体 1.0 M DOF 时 OOM 的那一步）。
结果存 `results/a_snap/`。全程预估：50,100 步 × ~2.5 s 热 + 2,275 次力学 × ~25 s ≈ **50 h**；力学改每 slab 一次（every_steps 110）≈ 38 h。

## 生产运行（路线 A 网格，用户拍板 2026-08-28）

`STAGES=3 TAG=a_snap` 于 2026-08-28 11:19 UTC 起跑（`~/work/159/output/v159_a_snap/production/`），
**11:34 UTC 在第 180 步被用户叫停**：用户复看后判定路线 A 的网格仍不合格，生产运行作废，待网格重做后重走 1/2/S/3。
HyperMesh 路线 B 的探测结论：CAD 234 面（凸耳台阶簇、凸台、通孔），[7.4,63] 段不可映射，未采用；脚本在 `hm/`。

## 后续（未做）

- runner 合同迁移：`--layers 455 --layer-thickness 0.0002 --support-thickness 0.002`（底座 = 夹具带），
  `--release-cut-box` = 底座包围盒，闪热半径 ≥ 0.5 m，命令功率 = P·πr²/(2hv·t_flash)，每层保温时长 = A(层)/(h·v)。
- 阶段 2 能量试跑（1 slab 热-only）核对台账；2-slab shakedown 看 eqp ≠ 0；内存探针（0.94 M DOF）。
