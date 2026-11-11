# v159（0119 件）MODE=5 pyamg-GPU 全高运行记录

> 运行编号：第四次全高（run-4）　TAG：`voxel_pyamg_fix2`　代码：`precond-optimization @ d8927e3`
> 生产段：2026-09-16 12:54 → 2026-09-17 03:02（北京时间；UTC 04:54:16Z → 19:02:34Z）
> 记录日期：2026-09-17。问题排查与修法见 `BUG_FIX.md` §8–§10。
> **缩放策略标注（2026-09-20 补）**：本次运行 `scale_policy = current`——层级复用期间每次求解都按当前切线的对角线重算 Jacobi 缩放，而层级（P、A₁）是按建层级时的缩放构造的。这是当时代码唯一的行为（该键 2026-09-20 才引入，缺省 `current`，E0 的配置文件现已显式写出）。§6.3 的 303 次不收敛后重建、485 次建层级即由此而来，机理与修法见 `BUG_FIX.md` §11；修正后的对照运行为 E0f（`scale_policy = frozen`）。

---

## 0. 摘要

| 项 | 结果 |
|---|---|
| 计算是否完成 | **是**：`production end rc=0`，11,885 / 11,885 步，182 / 182 层，冷却 60 步 + raft 释放解全部完成 |
| 生产门禁 | `PRODUCTION_GATE_RC=1`：11 项检查 10 项通过；未过的 `ledger_complete` 是 12 个步能量平衡超容差，**与 PARDISO 基线 fast2 完全同一组步**（fast2 同为 RC=1），属 fast 时间节奏固有问题，与求解器无关 |
| 与 PARDISO 基线一致性 | 11 个检查点 u_max 相对差 ≤ 7e-6、vm_max ≤ 4e-8（含冷却末） |
| 墙钟 | **14.13 h**（50,878.7 s），4.281 s/步；fast2（全 PARDISO）19.07 h，5.777 s/步 → **1.35x** |
| 力学线性解 | pyamg 调用 7,469 次，CG 190 万次；PARDISO 兜底 0；jit 变体 1；容量增长 0 |
| 显存 | JAX 分配器峰值全程 **7,810 MB**（上限 13,858 MB）；nvidia-smi 最高 11.0 GB |
| 主机内存 | 打印阶段 RSS 中位 18.7 GB、最高 21.9 GB；释放解（PARDISO）峰值 30.8 GB（WSL 上限 40 GB） |
| 输出 | `~/work/159/output/v159_voxel_pyamg_fix2/production/`，6.5 GB：46 个扫描帧 + 1 个冷却末帧 + `release.vtu` |

---

## 1. 代码与运行环境

| 项 | 值 |
|---|---|
| 仓库 / 工作树 | `Gyy12138-66/jax-fem` fork，WSL `~/work/d11_B_tree`（worktree） |
| 分支 / 提交 | `precond-optimization @ d8927e3`，启动时 `dirty=0`（运行期间只追加了文档提交 af9af3a，不影响计算） |
| 求解器相关提交 | `29d29ba` full 固定形状 → `5400d58` max_coarse 3000 + 一次性 PARDISO 兜底 + 路由前释放 → `d8927e3` 重建前先释放旧层级、解耦/缩放缓冲捐赠、日志记录显存（BUG_FIX §9.2） |
| Python 环境 | `/home/user/miniconda3/envs/jax-fem-gpu`：python 3.13.13、jax 0.10.2（`jax_enable_x64`）、pyamg 5.3.0、scipy 1.17.1、numpy 2.4.6、pypardiso 0.4.7（MKL PARDISO） |
| GPU | NVIDIA GeForce RTX 5080 16 GB，驱动 581.57（CUDA 13.0） |
| WSL | 内核 6.18.33.2-microsoft-standard-WSL2，32 逻辑核；`.wslconfig`：`memory=40GB`、`swap=16GB`、`autoMemoryReclaim=disabled`，不设 `processors` |
| JAX 环境变量 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`（启动器导出）、预分配关闭（`--xla-preallocate off`）、`JAX_PLATFORM_NAME=gpu`（v_159.sh 按 contract 设置） |

---

## 2. 配置文件

**案例配置**：`~/work/d11_B_tree/cases/159_simulation/inputs/0119-flash-voxel-fast-pyamg.json`
（继承 `0119-flash-voxel.json`；与 `0119-flash-voxel-fast-hybrid.json` 只差 `runner.linear_solver.mechanics`）

**运行时实际使用的派生文件**（stage 1 于 2026-09-16 12:54 重新生成，位于 `~/work/159/output/v159_voxel_pyamg_fix2/`）：

| 文件 | 内容 |
|---|---|
| `preflight/runner_contract.json` | runner 完整命令行（§4.3）、python/platform、各物理量映射 |
| `preflight/material_config.json` | 材料参数（源自 `cases/AM-Benchmark/verification/v2-cube-rs/model/v2_material_config.json`） |
| `preflight/0119_hm_voxel.inp` | 网格（源自 `cases/159_simulation/mesh/out_voxel/0119_hm_voxel.inp`） |
| `preflight/0119_path.csv` | 路径表 11,825 行 |
| `preflight/0119_ledger_summary.json` | 预检账本：steps 11,885、slabs 182、物理层 910、名义激光能量 1,107,105 J、建造时钟 19,870 s |
| `production/used_config.json` | runner 实际解析后的全部参数 |

### 2.1 关键物理参数

| 类别 | 参数 |
|---|---|
| 网格 | HyperMesh 0.5 mm 体素 HEX8，204,401 个单元，904,368 个力学自由度，建造方向 +Z，raft 1 mm（2 层体素） |
| 分层 | 激活 slab 0.5 mm，每 slab 5 个物理层（40 µm）/ 5 次 flash；slab 首个 flash 开始时激活（centroid 判据） |
| 扫描 | 140 W、0.65 m/s、hatch 0.12 mm；flash 读法 A'：每 flash 2 个激光步 + 6 个保温步 + 5 个铺粉步（13 行） |
| 热 | 预热/底面 353.15 K，环境 313 K，吸收率 0.62，热质量集总，冷却 600 s / 60 步 |
| 力学 | J2 塑性（简化径向回退），每 13 步解一次（每 flash 一次），应力自由参考 1273.15 K，底面固定至释放 |
| Newton | `rel_tol 5e-5`，Abaqus 判据，线搜索，最多 50 次迭代，增量最多切 3 次，温度下限 293.15 K |
| 释放 | 冷却后切除 raft（cut box `0–0.088 × 0–0.096 × 0–0.001 m`），3 点刚体锚定（`rigid_body`，秩 6 检查） |
| 输出 | 力学帧每 260 步一帧，summary 每 20 步 |

### 2.2 线性求解器块（配置原文）

```json
"linear_solver": {
  "platform": "gpu",
  "python_bin": "/home/user/miniconda3/envs/jax-fem-gpu/bin/python",
  "solver": "auto",
  "pardiso_mode": "phase23",
  "cell_target_batch_size": 32768,
  "thermal": {
    "backend": "jax", "method": "cg", "precond": "jacobi",
    "tol": 1e-06, "atol": 1e-06, "maxiter": 10000, "check_residual": true
  },
  "mechanics": {
    "backend": "pyamg", "device": "gpu", "shape_mode": "full",
    "fixed_levels": 4, "max_coarse": 3000, "maxiter": 800,
    "rebuild_iter_factor": 0, "verbose": true
  },
  "fallback": "pardiso"
}
```

---

## 3. 求解器配置（运行时生效）

run.log 回显：

```
mechanics_linear_solver = pyamg_solver(sa+rigid_body+scaled, cg, jacobix2, jax, shape=full, tol=1e-06, maxiter=800, fallback=pardiso)
fallback_solver       = pardiso_v07(phase23)
```

| 层级 | 组件 | 设置 |
|---|---|---|
| 装配 | jax-fem XLA 核（GPU） | 残差 + 切线，cell batch 32,768 → 主机侧全局 CSR（PETSc AIJ） |
| 热学线性解 | jax CG + Jacobi（GPU） | tol / atol 1e-6，maxiter 10,000，残差守卫开启 |
| 力学线性解 | `PyamgKrylovSolver`（`jax_fem_am/solvers/amg.py`） | 见下 |
| 　缩放 | 对称 Jacobi | S = D^-1/2 · A · D^-1/2 |
| 　预条件 | 光滑聚合 AMG（pyamg，CPU 上建层级） | 近零空间 = 6 个刚体模态；`max_coarse` 3000 自由度；实际全程 3 层，按固定 4 层嵌入（最粗层复制，P=R=I） |
| 　形状 | `shape_mode=full` | 细层为整体 904,368 阶，钉住行/列解耦为单位行；粗层补零容量 `[113152, 4864, 3840]`；全程 1 个 jit 变体 |
| 　V-cycle | GPU（cuSPARSE SpMV） | 每层前/后各 2 次阻尼 Jacobi，ω = (4/3)/ρ(D⁻¹A)；最粗层稠密伪逆（3840² ≈ 118 MB） |
| 　Krylov | PCG，整段在一个 `jax.jit` 内 | 缩放系统上 ‖r‖/‖b‖ ≤ 1e-6，最多 800 次 |
| 　层级复用 | `rebuild="pattern"`，`rebuild_iter_factor=0` | 激活状态不变即复用；3×fresh 过期规则关闭；复用失败 → 重建一次重解 → 仍失败则一次性 PARDISO |
| 　缩放策略 | `scale_policy = current`（2026-09-20 补注） | 每次求解按当前对角线重算缩放；层级仍是旧缩放系的量 → 复用解退化（BUG_FIX §11）。E0f 起改为 `frozen` |
| 　显存管理 | d8927e3 | 重建前先释放旧层级；解耦/缩放 kernel `donate_argnums`；`hierarchy built` 日志带 `device in_use/pool/peak/limit` |
| 释放解 | 共享 PARDISO phase23（CPU MKL） | `prefer_direct_linear_solver` 路由；路由前 `release_device()`，解完 `release_shared_solvers()` |
| Newton 停滞兜底 | PARDISO 重跑该次 Newton | 本次未触发 |

---

## 4. 启动文件

### 4.1 调用链

```
restart_run4_fixcfg.sh          （本次生产段的实际入口；STAGES="1 3"）
 └─ launch_v159_fix2_gates.sh   （MODE=5，TAG=voxel_pyamg_fix2，MEM_FRACTION 0.85，日志追加到 launch_voxel_pyamg_fix2.log）
     └─ launch_mode.sh          （MODE=5 → CFG=0119-flash-voxel-fast-pyamg.json）
         └─ cases/159_simulation/model/runs/v_159.sh
             ├─ stage 1：make_159_preflight.py → preflight/（指纹变化时重新生成）
             ├─ stage 3：python -m jax_fem_am.simulation.runner <contract argv> → production/
             └─ 门禁：check_159.py --run production --preflight preflight
```

旁路进程：`rss_sampler_run4.sh`（每 60 s 采样，写 `rss_samples.csv`）。

### 4.2 文件清单

| 文件 | 位置 | 作用 |
|---|---|---|
| `start_run4.sh` | `~/work/159/` | 先跑 2-slab shakedown（`STAGES="S"`），旧 shakedown 目录改名保留 |
| `launch_v159_fix2_prod.sh` | `~/work/159/` | `STAGES="3"` 的生产入口（第一次起跑用它，因 preflight 过期被弃，见 §5） |
| `restart_run4_fixcfg.sh` | `~/work/159/` | 杀掉过期参数的跑、删部分输出、`STAGES="1 3"` 重启 —— **会删 production，勿再执行** |
| `launch_v159_fix2_gates.sh` | `~/work/159/` | 导出 `MODE=5`、`TAG=voxel_pyamg_fix2`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`，调用 `launch_mode.sh` |
| `launch_mode.sh` | `~/work/159/` | MODE → 配置文件与默认 TAG，调用 `v_159.sh` |
| `v_159.sh` | `~/work/d11_B_tree/cases/159_simulation/model/runs/` | 分阶段执行、幂等、门禁 |
| `make_159_preflight.py` / `check_159.py` | `~/work/d11_B_tree/cases/159_simulation/model/` | 预检生成 / 门禁检查 |
| `rss_sampler_run4.sh` | `~/work/159/` | 采样：步、层、激活单元、RSS/HWM/匿名页、GPU、建层级数、过期标记、不收敛、`dev_in_use`、`dev_peak` |
| `health_run4.sh` / `cmp_bins_run34.sh` / `cost_split_run4.py` | `~/work/159/` | 运行中健康检查 / 与 run-3 逐层对比 / 每调用耗时拆分 |

从 Windows 分离启动的方式（`-e bash -c '...'` 形式会静默不启动）：

```powershell
Start-Process -FilePath wsl.exe -ArgumentList @('-e','bash','/home/user/work/159/<脚本>.sh') -WindowStyle Hidden
```

### 4.3 runner 完整命令行（`preflight/runner_contract.json`）

```bash
/home/user/miniconda3/envs/jax-fem-gpu/bin/python -m jax_fem_am.simulation.runner \
  --config /home/user/work/159/output/v159_voxel_pyamg_fix2/preflight/material_config.json \
  --inp /home/user/work/159/output/v159_voxel_pyamg_fix2/preflight/0119_hm_voxel.inp \
  --path-file /home/user/work/159/output/v159_voxel_pyamg_fix2/preflight/0119_path.csv \
  --path-length-scale 1.0 --build-axis z --base-side min \
  --layer-thickness 0.0005 --layers 182 --support-thickness 0.001 \
  --layer-activation-mode layer_on_scan --layer-activation-geometry centroid \
  --future-layer-mode void --active-window-below-layers 0 --inactive-mass-factor 1.0 \
  --powder-mode powder --surface-selection exterior --boundary-tol 1.0e-6 \
  --quadrature-order 2 --thermal-mass-lumping \
  --source-model legacy --beam-radius 1 --source-depth 0.000314181760657 \
  --source-depth-cutoff 0.000499 --source-cutoff-renormalize \
  --laser-power 22967001468.5 --absorptivity 0.62 --dt 0.000153846153846 --recoat-time 0 \
  --solidus-temperature 10000 --liquidus-temperature 10000 --latent-heat 0 \
  --phase-history-model legacy_reset --stress-relaxation-temperature 1273.15 \
  --reset-activation-temperature --activation-reset-temperature 353.15 \
  --ambient 313 --preheat-temperature 353.15 --bottom-thermal-bc fixed --bottom-temperature 353.15 \
  --cooling-steps 60 --cooling-dt 10 \
  --mechanics-model j2_plastic --bottom-mechanics-bc fixed --mechanics-every 13 \
  --mechanics-rel-tol 5e-05 --mechanics-acceptance abaqus --mechanics-max-iter 50 \
  --mechanics-max-cuts 3 --mechanics-temperature-floor 293.15 --mechanics-line-search \
  --thermal-output-every 0 --mechanics-output-every 260 --summary-every 20 \
  --xla-platform gpu --xla-preallocate off --xla-linear-solver auto --xla-pardiso-mode phase23 \
  --xla-cell-target-batch-size 32768 \
  --thermal-linear-solver '{"atol":1e-06,"backend":"jax","check_residual":true,"maxiter":10000,"method":"cg","precond":"jacobi","tol":1e-06}' \
  --mechanics-linear-solver '{"backend":"pyamg","device":"gpu","fixed_levels":4,"max_coarse":3000,"maxiter":800,"rebuild_iter_factor":0,"shape_mode":"full","verbose":true}' \
  --xla-fallback-solver pardiso \
  --release-after-cooling --release-anchor-mode rigid_body --release-cut-box 0 0.088 0 0.096 0 0.001 \
  --output-dir /home/user/work/159/output/v159_voxel_pyamg_fix2/production \
  --profile-json /home/user/work/159/output/v159_voxel_pyamg_fix2/production/profile.json \
  --profile-label v159-production
```

---

## 5. 时间线（北京时间，括号内 UTC）

| 时间 | 事件 |
|---|---|
| 09-16 09:44 | `wsl --shutdown` 清除第三次跑的 D 态进程；删除其 33 帧（run.log 与采样表改名保留） |
| 09-16 09:47–10:13 | 单测（pyamg 28、`-m solver` 95 passed）与 L150 显存基准 V14 / V14old / V15（BUG_FIX §9.2） |
| 09-16 10:15–10:20（02:15–02:20Z） | 2-slab shakedown：`SHAKEDOWN_GATE_RC=0`，136 步 2.26 s/步，max CG 45，`release_u_max 2.197e-3`，11 项检查全过 |
| 09-16 10:20（02:20:50Z） | 第一次起 stage 3（`STAGES="3"`） |
| 09-16 12:54（04:54:14Z） | 发现 stage 3 沿用 09-15 的旧 preflight contract（`maxiter=600`、无 `rebuild_iter_factor`），已跑到 step ~2660；杀掉并删除部分输出 |
| 09-16 12:54（04:54:15Z） | `STAGES="1 3"` 重启：stage 1 指纹变化 → 重新生成 preflight；run.log 回显 `maxiter=800` 确认生效 |
| 09-16 12:54（04:54:16Z） | **生产段开始** |
| 09-17 03:02（19:02:34Z） | **生产段结束** `rc=0 ledger=11885` |
| 09-17 03:02（19:02:38Z） | 门禁 `PRODUCTION_GATE_RC=1`（见 §6.1），`V159_DONE` |

---

## 6. 运行结果

### 6.1 生产门禁（`check_159.py`）

| 检查 | 结果 |
|---|---|
| `ledger_complete` | **false**（`all_balance_steps_within_tolerance=false`：12 个步超容差） |
| `no_nan_or_inf_in_log` | true |
| `newton_converged_everywhere` | true |
| `step_count_matches_preflight` | true |
| `activation_steps_match_preflight` | true |
| `part_consolidated_on_activation` | true |
| `stress_free_reference_applied` | true |
| `release_solve_present` | true |
| `release_removed_substrate` | true |
| `release_changed_part_stress` | true |
| `release_anchor_rank_6` | true |

能量账本（`production/thermal_energy_ledger_summary.json`）：记录步数 11,885 / 11,885，`solver_completed=true`；
最大相对平衡误差 4.155e-5（fast2 4.583e-5），最大绝对平衡误差 1.736e-3 J，装配恒等式误差 1.8e-12 J，温度不变量全部有效。
超容差的 12 个步与 fast2 **完全相同**（相对误差 2.7e-6 vs 2.8e-6 量级），fast2 门禁同样因此 RC=1。

### 6.2 性能（`production/profile.json`）

| 项 | 值 |
|---|---|
| 总墙钟 | 50,878.7 s = **14.13 h** |
| 步数 / 每步 | 11,885 / **4.281 s** |
| Newton 墙钟 | 44,406 s（87.3%） |
| 线性解（热学 + 力学，59,208 次） | 26,682 s（52.4%） |
| 装配 | 14,205 s（27.9%）；其中 cell_jacobian 6,351 s、cell_residual 2,774 s、global_matrix 2,360 s |
| Python 开销 | 5,836 s（11.5%） |
| IO（51 次） | 151 s |

分段统计（按采样表与 run.log；CG 求解次数含不收敛后的重解）：

| 层 | s/步 | CG 求解次数 | 平均 CG 迭代 | 不收敛（均重建后救回） | 建层级 |
|---|---|---|---|---|---|
| 1–20 | 2.26 | 532 | 36 | 0 | 21 |
| 21–40 | 2.87 | 606 | 124 | 0 | 20 |
| 41–60 | 3.10 | 491 | 254 | 12 | 32 |
| 61–80 | 3.15 | 462 | 255 | 24 | 44 |
| 81–100 | 4.90 | 828 | 352 | 55 | 75 |
| 101–120 | 5.18 | 1,026 | 293 | 46 | 66 |
| 121–140 | 6.24 | 1,291 | 293 | 73 | 93 |
| 141–160 | 5.08 | 1,204 | 200 | 40 | 60 |
| 161–180 | 5.59 | 1,198 | 264 | 46 | 66 |
| 181–182 + 冷却 + 释放 | 4.90 | 134 | 263 | 7 | 8 |

### 6.3 力学求解器统计（`profile.json` → `meta.mechanics_custom_solver_stats`）

| 项 | 值 |
|---|---|
| 调用次数 | 7,469 |
| CG 迭代总数 | 1,900,029（平均 254 / 调用） |
| 建层级耗时 `setup_s` | 4,890 s（1.36 h） |
| 求解耗时 `solve_s` | 14,071 s（3.91 h） |
| 建层级次数 | 485 = 模式变化 182 + 不收敛后重建 303；过期规则重建 0 |
| PARDISO 兜底 | **0** |
| jit 变体 / 容量增长 / 结构重建 | 1 / 0 / 1 |
| `release_device` | 1 次（释放解前） |
| 释放解 PARDISO | 5 次解（phase13 ×1 + phase23 ×4），数值分解 30.8 s；解后释放 1 个分解 |

### 6.4 内存（`rss_samples.csv`，846 个样本）

| 项 | 值 |
|---|---|
| runner RSS（打印阶段） | 中位 18.7 GB，最高 21.9 GB |
| runner RSS 峰值（释放解，PARDISO） | 30.8 GB |
| swap / oom_kill | 0 / 0 |
| nvidia-smi 显存 | 最高 11.0 GB（16.3 GB 卡） |
| JAX 分配器 in_use / peak | 最高 5.48 GB / **7.81 GB**（上限 13.86 GB） |

### 6.5 与 PARDISO 基线 fast2 对比

fast2：`~/work/159/output/01_v159_simulation/v159_voxel_fast2`，MODE=3（GPU 装配 + 全 PARDISO phase23），代码 `b3dbd91`，2026-09-02/03；runner 参数与本次只差线性求解器三项（`--xla-linear-solver`、`--thermal-linear-solver`、`--mechanics-linear-solver`/`--xla-fallback-solver`）。

| 步 | u_max 本次 / fast2（相对差） | vm_max 本次 / fast2（相对差） |
|---|---|---|
| 260 | 8.1940e-5 / 8.1940e-5（−5.1e-9） | 9.7993e8 / 9.7993e8（−2.1e-9） |
| 2600 | 3.0222e-4 / 3.0222e-4（−4.6e-8） | 1.0929e9 / 1.0929e9（−8.1e-9） |
| 5200 | 3.9348e-4 / 3.9348e-4（−1.7e-7） | 1.0946e9 / 1.0946e9（−2.5e-9） |
| 7800 | 5.5136e-4 / 5.5137e-4（−5.9e-6） | 1.0945e9 / 1.0945e9（+2.2e-9） |
| 10400 | 7.0929e-4 / 7.0929e-4（+6.8e-6） | 1.0972e9 / 1.0972e9（−1.3e-9） |
| 11884（冷却末） | 7.6312e-4 / 7.6311e-4（+6.3e-6） | 1.0984e9 / 1.0984e9（−1.6e-9） |

| 项 | 本次 | fast2 |
|---|---|---|
| 墙钟 / 每步 | 14.13 h / 4.281 s | 19.07 h / 5.777 s |
| Newton 墙钟 | 44,406 s | 62,651 s |
| 门禁 | RC=1（同一原因） | RC=1 |

### 6.6 释放解

`release_u_max = 1.81537e-3 m`（保留打印节点位移模的最大值），`release_u_mean = 6.6067e-4 m`，`release_u_max_all_nodes = 1.76170e-3 m`（位移分量绝对值的最大值）。

**不能与 fast2 的 `release_u_max = 1.05634e-3 m` 直接比较**：fast2 之后提交 `ba09c49`（release: rigid-body anchors from retained material, fail-closed rank check）改了锚点选取规则。
两次的释放解都走 PARDISO。与当前释放规则同代码的 PARDISO 车道全高对照尚未跑。

---

## 7. 输出文件位置

根目录：`/home/user/work/159/output/v159_voxel_pyamg_fix2/`
（Windows：`\\wsl$\Ubuntu\home\user\work\159\output\v159_voxel_pyamg_fix2\`）

| 路径 | 大小 | 内容 |
|---|---|---|
| `production/step_000000_scan.vtu` … `step_011700_scan.vtu` | 46 个，逐帧增大 | 打印过程帧，每 260 步一帧 |
| `production/step_011884_cooling.vtu` | 207 MB | 冷却末状态 |
| `production/release.vtu` | 203 MB | raft 切除后的释放状态 |
| `production/run.log` | 1.2 MB | runner 全日志（每次 CG 迭代数、层级构建、显存、分阶段计时表） |
| `production/profile.json` | 11 KB | 分阶段计时与求解器统计 |
| `production/thermal_energy_ledger.jsonl` / `thermal_energy_ledger_summary.json` | 20 MB / 1 KB | 逐步能量账本 / 汇总 |
| `production/cube_smoke_gate.json` | 67 KB | 门禁详细结果 |
| `production/used_config.json` / `path_used.csv` | 9 KB / 1.3 MB | 实际参数 / 实际路径表 |
| `preflight/` | 32 MB | 本次生产用的 contract、网格、路径、材料 |
| `shakedown_2slabs/` | 42 MB | 本轮 2-slab shakedown（d8927e3，RC=0） |
| `energy_layer1/` | 9 MB | 能量试跑（09-15） |
| `v159.log` | 12 KB | 各阶段起止与门禁结果（含 09-15 的历史段） |
| `rss_samples.csv` | 59 KB | 本次每 60 s 采样 |
| `shakedown_2slabs_run1_code5400d58/` | 42 MB | 上一轮代码的 shakedown（保留对照） |
| `production_run3_frozen_layer129.log`、`rss_samples_run3_frozen_layer129.csv` | 0.9 MB | 第三次跑（冻结于层 129）的证据，帧已删 |
| `rss_samples_run1_killed_layer98.csv` | 24 KB | 第二次跑（被 WSL 更新杀掉）的采样 |

`production/` 合计 6.5 GB。其他：启动器汇总日志 `~/work/159/output/launch_voxel_pyamg_fix2.log`；显存基准 `~/work/159/amgbench/`（V13 / V14 / V14old / V15 的 JSON 与日志）。

---

## 8. 查看与复现

查看（WSL 内）：

```bash
O=~/work/159/output/v159_voxel_pyamg_fix2
tail -40 $O/v159.log                                                          # 阶段与门禁
grep -m2 -E "mechanics_linear_solver|fallback_solver" $O/production/run.log   # 实际生效的求解器
grep '^global_step=' $O/production/run.log | tail -1                          # 最终状态
grep -A30 'wall=' $O/production/run.log | head -32                            # 分阶段计时表
grep release_vtk $O/production/run.log                                        # 释放结果
```

复现（**用新 TAG**）：

```bash
export MODE=5 TAG=voxel_pyamg_rerun STAGES="1 2 S 3" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
bash ~/work/159/launch_mode.sh >> ~/work/159/output/launch_voxel_pyamg_rerun.log 2>&1
```

注意事项：

1. **不要用 `TAG=voxel_pyamg_fix2` 再跑 stage 3**：v_159.sh 只有账本 `complete=true` 才跳过已完成阶段，本次 `complete=no`，重跑会先 `rm -rf production` 删掉本次结果。
2. **不要再执行 `restart_run4_fixcfg.sh`**：它会杀进程并删除 `production/` 与 `rss_samples.csv`。
3. 改过案例 JSON 的任何 runner / 求解器字段后，同 TAG 重启必须带 stage 1（`STAGES` 含 `1`），否则 v_159.sh 静默沿用旧 `runner_contract.json`；起跑后用 `grep -m1 mechanics_linear_solver production/run.log` 核对回显。
4. 长跑前先 `wsl --update`，并暂停 Windows 更新与 Microsoft Store 应用自动更新（第二次全高就是被 WSL 包自动更新重建 VM 杀掉的）。
5. `/tmp` 在 `wsl --shutdown` 后清空，脚本与数据放 `~/work/159/`。

---

## 9. 相关文档与历次全高运行

| 运行 | TAG / 代码 | 结局 | 文档 |
|---|---|---|---|
| 2026-09-07/08 | `v159_voxel_pyamg`，free 形状 | 层 153 冻结：每个新形状主机驻留 ~186 MB | BUG_FIX §1–§5 |
| 2026-09-14 | `v159_voxel_pyamg_full`，29d29ba | 层 120 冻结：4 层层级退化 + PARDISO 分解驻留 | BUG_FIX §7、§8 |
| 2026-09-15 | `v159_voxel_pyamg_fix2`，5400d58 | 层 98 被 WSL Store 自动更新杀掉 | BUG_FIX §9、§9.1 |
| 2026-09-15 | `v159_voxel_pyamg_fix2`，5400d58 | 层 129 冻结：重建风暴 + 显存顶格 → dxg D 态 | BUG_FIX §9.2 |
| **2026-09-16/17** | **`v159_voxel_pyamg_fix2`，d8927e3** | **完成（本文）** | 本文；优化方向见 BUG_FIX §10 |
