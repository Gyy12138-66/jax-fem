# V2 立方体 — 阶段 1/2 预检与缩高顺序热—力 smoke（IET-9）

分支 `agent/gpt5-6sol/v2-stress-reproduction`，worktree `/home/user/work/d11_B_tree`。
红线遵守：零标定、未改任何共享求解器代码、未向 XRD/ABAQUS/高温计回调。
本文件只记**机制验证**；这里没有任何一个数可以拿去和 XRD 或 Balbaa 的 ABAQUS 比。

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `inputs/cube-stress-smoke.json`（schema `v2.cube-stress-smoke/2`） | 冻结输入：几何（生产值 + smoke 值）、层调度、扫描、热-力口径、runner 口径 |
| `model/make_v2_cube_preflight.py` | 阶段 1：网格 + 路径（网格坐标）+ slab 激活事件 + 时间/能量台账 + **runner 合同**（`runner_contract.json`）+ 材料配置重定位；全部 sha256 指纹；闭合检查（slab 边界落在网格 z 层、曝光矩形 ⊂ 零件足印、能量恒等式、蛇形交替、激活事件=各 slab 首个开光行） |
| `tests/unit/test_v2_cube_preflight.py` | 15 个单测（含三条 fail-closed：`intersection` 几何、切带越过 slab、顶点法则带积分≠1） |
| `model/runs/v_cube_smoke.sh` | 串行、幂等、fail-closed 的三阶段运行器（预检 → 单层捕获试跑/半径阶梯 → 25 层热-力 smoke → 门槛） |
| `model/check_cube_smoke.py` | 阶段 2 门槛：台账完整、步数恒等、激活步逐 slab 与预检事件一致、激活即固化、T_ref、能量捕获、Newton、NaN、释放（基板单元被切、件内应力被改变） |
| `<run>/preflight/` | `v2_cube_smoke_c3d8.inp`、`v2_cube_smoke_path.csv`、`v2_cube_smoke_ledger.json`、`runner_contract.json`、`material_config.json` |

## 2. 阶段 1 结果（`output/v2_cube_smoke_smoke2/preflight`）

- 网格：零件 20 × 20 × 5 = 2,000 单元（4 × 4 × 1 mm，25 个 40 µm 真实层 → 5 个 200 µm slab），基板 40 × 40 × 8 = 12,800 单元（8 × 8 × 1.6 mm 均匀 200 µm），共 14,800 单元 / 17,334 节点。零件居中：网格坐标 [2, 6] mm。
- 路径：16,239 行 = 15,200 扫描 + 799 跳转 + 240 recoat 子步（24 次 × 10 步，几何比 1.5：0.044 → 1.696 s）；每层 32 道，0°/90° 交替，蛇形；每 slab 的五个真实层沉积 z 都取 slab 顶面。
- 时间台账：扫描 4.677 s + 跳转 0.037 s + recoat 120 s = 124.71 s；冷却 600 s（60 × 10 s）；建造钟 724.71 s；runner 步数 16,299。
- 能量台账：名义 654.77 J（= 140 W × 3.8 mm/0.65 m/s × 32 道 × 25 层，恒等式断言）；吸收名义 405.96 J；每 slab 130.95 J。
- 激活事件（slab：行/global_step，时刻）：1：0，0.0003 s；2：3250，25.943 s；3：6500，51.886 s；4：9750，77.829 s；5：13000，103.772 s。五个 slab 的上下边界都落在网格 z 层上。
- 指纹（smoke2 链）：config `f8ca1be6…`，mesh `bde7f790…`，path `3bdde717…`，material `deae715a…`，contract `07d58afb…`。网格/路径/材料与 smoke1 链逐字节相同（哈希一致），config 与 contract 因哨兵固相线不同。

## 3. runner 合同（阶段 2 前半）

`runner_contract.json` 的 argv 由配置生成，运行器只补 `--output-dir/--profile-json/--profile-label`。要点：

- 激活：`layer_on_scan` + **`centroid`**；`layer` 列 = 1 基 slab 号；`--layers 5 --layer-thickness 2e-4 --support-thickness 1.6e-3`。
- recoat 显式在路径里，`--recoat-time 0`；`--dt` = 首行 dt = 3.0769e-4 s。
- 热源：legacy 高斯 × 指数，r 200 µm，d 125.935 µm，切带 199 µm 重归一（见 §4）。
- 温度映射：激活即固化且**熔检测关闭**（solidus = liquidus = 10000 K 哨兵，潜热 0，`legacy_reset`，T_ref = 1273.15 K 一次锚定），新 slab 重置到 353.15 K；力学每 100 步 + 末步强制；温度地板 293.15 K。
- 冷却与释放：600 s；底面 353.15 K 不 ramp；释放 = 已打印节点 3 点刚体锚 + `--release-cut-box` 切掉整块基板。

## 4. 阶段 2 前半：单层能量捕获（D-V2-10 量化）

同一条第 1 层路径（639 行），热-only，只改热源参数：

| 读法 | 总捕获 | 逐步 [min, max, std] | 道间 | T_max |
|---|---|---|---|---|
| V1 物理光斑 r 50 µm，d 100，半空间 | **0.0024** | [0.000, 0.006, 0.003] | — | 355 K |
| r 200/300/400 µm，d 100，切带 200（首版） | 1.417 / 1.421 / 1.400 | 内部步恒为 1.447–1.47 | — | 1738/1510/1357 K |
| **合同**：r 200 µm，d 125.935，切带 199 | **0.9616** | [0.620, 0.997, 0.067] | 0.958–0.983，边道 0.82，道端 0.74 | 1539 K |

两个机制，都已登记：

1. **物理光斑在 200 µm 单元上根本沉积不进去**（0.24 %）：高斯落在采样点之间。D-V2-10 原先"差别在亚单元尺度"的前提不成立。
2. **集总配点把热源搬到了顶点采样**：`--thermal-mass-lumping` 用顶点置换重算 `physical_quad_points`，指数深度剖面在 slab 顶（深度 0）和 slab 底/基板顶（深度 200 µm，各一次）被梯形式高估，预测 (1+2e⁻²)/2 ÷ (1−e⁻²) = 1.4696，实测 1.447–1.47，与半径无关。合同把切带收到 199 µm 并解 d(1−e^{−cut/d}) = h/2 得 d = 125.935 µm，使顶点法则的离散带积分恰为 1（高斯采样下同组参数给 1.0007）。内部步实测 0.997。剩余 3.8 % 是高斯尾在零件足印外的损失（Balbaa 的点源没有这项），如实登记不修。

顺带核过热闸门 as-is 臂：沉积 28.35 J vs 窗内名义 27.8 J（比值 1.02），40 µm 网格上顶点法则对 100 µm 剖面近似精确，热闸门不受此影响。

## 5. 阶段 1/2 抓到并修掉的三个合同缺陷

| 缺陷 | 症状 | 修法 |
|---|---|---|
| `intersection` 激活几何 | 首行 printed_cells = 13600 = 基板 12800 + **两个** slab（`cells_intersect_distance_band` 闭区间，slab 面贴单元面） | 合同改 `centroid`；预检拒绝 `intersection`；单测 |
| 路径零件局部坐标 | y < 2 mm 的道捕获 0.13–0.15，其余 ~0.8（一半扫在基板上） | 预检输出网格坐标；断言每个开光行在零件足印内且 .inp 零件节点包围盒 = 调度包围盒 |
| 顶点采样高估深度剖面 | 捕获 1.47 与半径无关 | 切带 199 µm + d 125.935 µm；预检断言顶点法则带积分 = 1 |
| 物理固相线下的熔态记账 | 第一版 smoke（`v2_cube_smoke_smoke1/smoke`，solidus = liquidus = 1563 K）：峰温 1.65–1.98e3 K，几乎每步都有单元进出"熔态"（液相软化 ×1e-4、熔化即抹 eqp）。Balbaa 的立方体模型没有熔态 | solidus = liquidus = 10000 K 哨兵（预检要求 ≥ 5000 K）；smoke1 在 step 300 处按 PID 停掉 |

"liquidus == solidus 关闭熔检测"这句 L0 约定只在峰温低于它时成立——L0 没到过 1552 K，所以那里看不出来。

**力学节拍是事件驱动的，不是每 100 步**：v06 生命周期在任一受管单元升温越过 T_cut = 1273.15 K 时抛出参考事件（`reference_event = newly_solidified | became_melted | became_relaxation_hot`），冷却穿越时抹 eqp（D-11 完整历史复位），`install_mechanics_event_wrapper` 对每个有待处理事件的步强制求一次力学。移动热源下几乎每个扫描步都有；smoke2 实测 2.11 s/步（406 步 858 s）vs 热-only 0.31，整个 smoke ≈ 9–10 h。这正是 Balbaa 引用 Denlinger 的"1000 °C 以上无热应力"语义（D-11 主线同款），所以**保留**；`mechanics.every_steps` 只是节拍下限。

## 6. 阶段 2 后半：25 层顺序热—力 smoke

运行：`output/v2_cube_smoke_smoke2/smoke`（脱离会话 `nohup setsid` 启动，2026-08-26 06:06 UTC；同一链先重跑预检和捕获试跑以绑定新指纹）。
门槛脚本 `check_cube_smoke.py`，结果写 `cube_smoke_gate.json`。

**结果：16,299 步全部跑完，rc=0，门槛 10/10 通过**（2026-08-26 06:12 → 15:59 UTC，墙钟 35,208 s = 9.78 h，2.16 s/步）。

| 门槛项 | 观测 | 判定 |
|---|---|---|
| 台账完整 / 步数恒等 | `complete=true`，16,299 = 16,239 行 + 60 冷却 | ✓ |
| 激活步逐 slab 对合同 | `activation_step` 单元场：slab 1–5 分别恰在 0 / 3250 / 6500 / 9750 / 13000，每 slab 400 单元，无其他取值 | ✓ |
| 激活即固化 / T_ref | 2,000 个零件单元末态 `material_state=2`，`stress_free_temperature` 唯一值 1273.15 K | ✓ |
| Newton | 0 次不收敛，0 次回切，无 NaN/Inf | ✓ |
| 能量捕获 | 沉积 390.362 J vs 单层预测 0.9616 × 405.957 = 390.363 J；每个 slab 78.072 J（五个逐位相同） | ✓ |
| 能量平衡 | 扫描/recoat 步 ≤ 1e-5；60 个 10 s 冷却步 2.02e-4（大 dt 下 Newton 停机残差，台账自身判据 `balance_within_solver_tolerance` 全真） | ✓（登记） |
| 释放 | `release_removed` = 12,800 = 全部基板单元；件内 vm 均值 567 → 329 MPa、峰 877 → 540 MPa；释放后 σxx/σyy 件内均值 = 0（自平衡）；u_z 峰 13 µm → 648 µm（4 × 4 × 1 mm 薄板切离后翘曲） | ✓ |

末态（冷却结束 353 K，释放前 → 后）：顶层 σxx 均值 539 → 185 MPa，σyy 393 → 72 MPa；eqp 峰 0.0178、均 0.0061；温度 352.9–353.15 K。
这些数只说明链路语义正确（有锁入应力、释放自平衡、塑性历史累积），**不是**可与 XRD 比较的量。

求解开销：`nonlinear_solve` 29,200 次 = 16,299 热 + **12,901 力学**（79 % 的步有力学求解，事件驱动），力学 Newton 占墙钟 88 %。

指纹说明：smoke2 的预检在 06:06 UTC 绑定了当时的 `cube-stress-smoke.json`；之后只改了 `runner.consolidation.note` 的说明文字（把力学节拍的成因从"熔态"更正为"T_cut 事件"），argv 与所有数值未变，故 `runner_contract.json` 的 argv 与本文 §3 逐字一致，`config_sha256` 与提交后的文件不同属预期。`check_cube_smoke.py` 的 T_ref 容差从 1e-6 放到 1e-3 K（VTU 单元数据是 float32，1273.15 读回 1273.1500244）——首轮门槛因此报 9/10，重判 10/10，求解器产物未动。

## 7. 给阶段 3/4 的阻塞项（不是本包的任务）

- **250 事件读法的算力**（D-V2-11 cost_projection）：10 × 10 生产几何每真实层约 4,100 行 × 250 层 ≈ 1e6 热步，实测 0.31 s/步（1.5 万单元）→ 生产网格上 12–24 天热-only。要么线源沉积（改共享求解器），要么逐道/逐层集总（放弃 250 事件读法）。
- D-V2-07 基板（生产用 graded 30 × 30 × 6）、D-V2-12 初温、D-V2-18 α 拟合仍是生产未冻结项。
