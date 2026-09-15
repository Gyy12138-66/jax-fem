# BUG FIX — MODE=5 pyamg-GPU 车道全高生产跑挂死

- **日期**：2026-09-14（挂死发生于 2026-09-07/08，运行 `v159_voxel_pyamg`）
- **分支**：`precond-optimization`（`jax_fem_am/solvers/**` 为本线独占文件）
- **改动文件**：`jax_fem_am/solvers/amg.py`、`jax_fem_am/solvers/linear.py`、
  `tests/unit/test_pyamg_krylov_solver.py`、`cases/159_simulation/inputs/0119-flash-voxel-fast-pyamg.json`
- **状态**：单测 `-m solver` 90 passed；真实矩阵微基准通过（下文 §4）；全高生产跑待重跑

---

## 1. 症状

`MODE=5 STAGES="1 2 S 3"`（`0119-flash-voxel-fast-pyamg.json`，分支 `d91905a`）全高生产跑在
**第 9907/11885 步（83%，第 153/182 层）挂死**：日志最后一行是一次 CG 300 次未收敛，
之后 6.7 小时没有任何新日志、步数不推进、进程未退出。此前有效运行 11.5 小时。

## 2. 根因（实测，不是推断）

排查链上先后被排除的两个假象：

| 假象 | 为什么不是 |
|---|---|
| CG 不收敛 | 只是终态症状；整个跑的 CG 效率（ms/迭代/M-DOF）从 13.6 一路**改善**到 6.8，没有退化迹象 |
| JIT 编译撑爆显存 / 编译耗时 | 显存在 0.5 h 内就平在 11.4 GB（= JAX 默认 75% 上限 `bytes_limit` 12,228 MB）不再涨；按 nf 匹配对照，换形状后的首次求解**不比**同形状新建层级慢，编译墙钟 ≈ 0 |

真因：**主机内存按"自由度形状"驻留**。

- 求解器把力学切线**抽取成自由块**（`_build_pattern` 里 `Aff` 为 `nf × nf`），激活每长一层 `nf` 就变，
  JIT 按新形状重新特化。该跑共 **1,042 次层级重建、153 个不同的 nf**（36,945 → 825,297）。
- 用 fast2 的真实 L150 力学矩阵（n = 904,368，nnz = 57,488,058）做 24 个"激活前沿上移"模式的微基准，
  驱动真实的 `PyamgKrylovSolver`，逐模式测得（`/tmp/amgbench/`，2026-09-14）：

| 变体 | 每换一个形状，主机 RSS 增量 | 24 模式后 RSS |
|---|---|---|
| V0 旧代码 `d91905a`（挂死那版） | **+195 MB** | 3.26 → 7.56 GB |
| V1 `437b83d`（per-pattern kernel + release） | +186 MB | 3.26 → 7.34 GB |
| V2 V1 + `clear_jax_caches_on_pattern` | +155 MB | 3.26 → 6.64 GB |
| V5 V1 + 每模式 `malloc_trim(0)` | +174 MB（trim 只回收常数 ~550 MB） | 2.75 → 6.74 GB |
| **V3 固定形状（同一最大模式只改值、每次重建层级）** | **−1 MB/次（8 次预热后完全平台化）** | 5.31 → 6.41 GB 后不动 |

  GPU 侧四个变体一致（pool ≈ 4.1 GB / in_use ≈ 1.1 GB），**不是区分因素**。唯一让曲线平掉的变量是"形状不变"。
- 外推：153 个形状 × ~186 MB ≈ **28 GB**（仅 pyamg 一处），而 `.wslconfig` 为 `memory=40GB, swap=16GB`。挂死跑本身没有
  RSS 记录（progress.log 第 5 列取值 2–37、非单调、与 GPU 显存负相关，**不是** RSS，含义未考证），所以
  "主机内存到顶后换页抖动"是与 6.7 h 静默冻结最相符的解释，不是直接观测。
- 已提交的 `437b83d` 两版方案（per-pattern kernel 释放；清 JAX 缓存）在生产尺度上分别只压掉 5% / 20%，**不够**。

## 3. 修改方法：`shape_mode="full"`（整体固定形状）

设计要点（`amg.py`）：

1. **整体矩阵结构本来就固定。** 装配器每步给出的是全网格 `n × n` 切线，未激活/受约束自由度是
   `diag = 1`、非对角为**结构零**的单位行（L030 与 L150 dump 的 nnz 完全相同）。
2. **Krylov 系统改为整体 `n × n`**：钉住行/列的值置零、对角置 1（`decouple` kernel，形状固定），
   Jacobi 缩放下钉住行的 scale = 1；右端与初值在钉住行置 0，解出后把预设值加回。
   与原来的"抽取自由块 + `bf = b[free] − (A x_pin)[free]`"**数学等价**：自由行是同一组方程，钉住行残差恒为 0。
3. **层级仍在自由块上建**（V4 实测：整体喂 pyamg 的粗层尺寸与自由块完全相同，单位行被 B = 0 剔除，无粗层膨胀，
   但构建慢 7–37%，所以不这么做），然后把每一层的 A / P / R / dinv / 粗层稠密逆**补零到固定容量**，层数固定为
   `fixed_levels`（默认 4）；pyamg 给出的层数不足时**复制最粗层并令 P = R = I**——精确粗层解只是下移一层，
   预条件子不变。容量由 `default_capacities(n, fixed_levels, max_coarse)` 按 n 推出（对 v159：
   `[904368, 113152, 4864, 1280]`，P0 nnz 24n、A1 200·rows1、P1 24·rows1……），
   **溢出则 ×1.5 增长并重编译一次，绝不失败**（`stats["capacity_growths"]`）。
4. JIT kernel 集改为**归结构所有**（`_StructureCache`），跨激活模式不重建；`stats["compiled_variants"]` 在
   整个构建过程中应保持 1。
5. `shape_mode="auto"`（默认）在 gpu/jax 路径解析为 `"full"`，cpu 路径为 `"free"`（无 JIT）。
   `"free"` 保留旧行为；`"bucket"` 为待开发模板（§6）。
6. 规范键（`linear.py`）：`shape_mode`、`fixed_levels`、`bucket_ratio`；MODE=5 配置显式写 `"shape_mode": "full"`。

## 4. 对比结果

### 4.1 真实矩阵微基准（24 模式，nf 146k → 818k，同一 RTX 5080 / jax 0.10.2 / pyamg 5.3.0）

| | V1 自由块（c3962bb） | **V6 整体固定（本次）** |
|---|---|---|
| 编译变体数 | 24 | **1**（首版默认容量在 k=3 因 `rows3` 512 < 552 增长过一次；默认改为 ≥ 1.25·max_coarse 后 V8 重跑：变体 1、增长 0） |
| 主机 RSS | 3.26 → 7.34 GB，**+186 MB/模式**（其中 ~130 MB/形状是 jax 侧按形状驻留） | 4.45 → 5.73 GB，**+51 MB/模式 = 当前模式工作集随 nf 增大，无按形状驻留**（§5） |
| GPU in_use / pool | 1.1 / 4.1 GB | 1.55 / 4.1 GB（补零后的固定容量算子） |
| 层级新建求解墙钟合计 | 223 s | 237 s（+6%） |
| 层级复用求解墙钟合计 | 19.8 s | 26.2 s（**+33%**，逐模式比 1.14–1.59，末段 1.17–1.25） |
| 24 模式总耗时 | 268 s | 296 s（+10%） |
| 原系统真实相对残差 | — | 0.9–1.2e-6（tol 1e-6，CG 判据在缩放系统上） |
| 钉住行 `x[pin] == b[pin]` | — | 24/24 精确 |

### 4.2 外推到全高生产跑（以挂死跑的 6,527 次求解、1,042 次重建为账本）

| 项 | 基线（自由块，挂死那版） | 整体固定 |
|---|---|---|
| CG | 1.70 h | 2.11 h（+24%，每迭代 5.59 → 5.98 ms；单位行的结构零参与 SpMV） |
| 层级构建 | 2.40 h | 2.40 h（不变，仍在自由块上建） |
| JIT 编译 | 153 次 ≈ 0 s | 1 次 |
| **合计** | **4.10 h** | **4.51 h（+10%）** |
| 主机 RSS 增量 | ~28 GB → 顶到 40 GB 上限挂死 | 预计 ≤ 数 GB（§5），不再随层数线性增长 |

即：每次全高多花约 0.4 小时，换来能跑完。

## 5. 残余问题与后续

- V6 仍有 +51 MB/模式的 RSS 增长，**它不是驻留，是当前模式的工作集随 nf 增大**——逐项排除后定量确认：
  - V9（full，`tol=1e30` 不迭代）+50、V10（cpu 路径，完全不碰 jax）+55、V11（V10 + `MALLOC_MMAP_THRESHOLD_=131072`）+50、
    V12（V10 + BLAS/OMP 单线程）+52 MB/模式：与 jax、glibc 动态阈值、BLAS 线程缓冲都无关；`malloc_trim` 只回收常数 ~174 MB；
  - Python 层无对象滞留：任何时刻存活的 `_PatternCache` = 1、pyamg `MultilevelSolver` = 1；
  - `tracemalloc` 追踪到的 Python/numpy 分配量与 RSS 同步增长（cpu 路径 8 个模式 +1,575 MB vs RSS +1,689 MB），
    即增长全部是**当前存活对象**：本模式的自由块索引（`keep`、`ff_rows/ff_indices` 各 ~210 MB）与层级算子
    （level-0 缩放矩阵 ~630 MB、P0/R0 ~440 MB、A1 ~120 MB……）随 nf 变大，全激活时约 1.5 GB，**在 nf 停止增长时饱和**
    （V3：同尺寸重复 24 次，斜率 −1 MB/次）。
  - 对照：free 模式的 +186 MB/模式里，扣掉同样的工作集增长后，仍有 ~130 MB/形状是 jax 侧按形状驻留、gc/clear_caches/trim 都不回收——
    这部分在 full 模式下为 **0**。全高生产跑的主机内存预期：工作集 ~1.5–2 GB + 求解器以外的常量，不再随层数线性增长。
- 逐位差异：整体求解与抽取求解运算次序不同，**MODE=5 车道的 40-slab "bit-identical build stresses" 要重立基线**；
  金标门跑的是默认 PARDISO 车道，不经过 pyamg，不受影响。
- 全高生产重跑：`MODE=5 STAGES="1 2 S 3" bash launch_mode.sh`，盯 `compiled_variants`（应为 1）、`capacity_growths`（应为 0）、`rss_mb`。

## 6. 待开发：`shape_mode="bucket"`（自适应分桶）

把自由块补零到几何比递增的桶容量而不是整体网格，用少数几次重编译换更少的补零开销。按挂死跑的账本估算：

| 分桶几何比 | 形状数 | CG 代价 | 预计 RSS 增量 |
|---|---|---|---|
| 1.10 | 28 | +3% | ~5 GB |
| 1.25 | 15 | +8% | ~2.7 GB |
| 1.50 | 9 | +11% | ~1.6 GB |
| 整体固定 | 1 | +24% | ≈ 0 |

模板已就位（`amg.bucket_capacity`，`shape_mode="bucket"` 目前抛 `NotImplementedError`），需要做的自适应：
按主机内存预算选 `ratio`；以首个 nf 锚定阶梯且只升不降；按实际建出的层级尺寸学习粗层容量；
桶溢出时按路径表预测的剩余激活量直接跳到目标桶。

---

## 7. 后续优化点：`max_coarse` 与层级深度（2026-09-14 下午，只读复盘，未改代码）

**决定**：`v159_voxel_pyamg_full` 全高跑（2026-09-14 14:57 起）**让它跑完**，先验证 §3 的内存修法；本节记录的问题在下一轮修。

### 7.1 症状（挂死那次 `v159_voxel_pyamg` 的 run.log，按 10 层分箱）

| 层段 | 层级数 | 新建层级后 CG 中位迭代 | 复用层级中位 | 触顶 300 比例 | 建层级次数 / 10 层 |
|---|---|---|---|---|---|
| 1–20 | 3 | 23–24 | 32–36 | 0 | 10–12 |
| 21–70 | 3 | 65–79 | 87–101 | ≈4% | 10–57 |
| 71–160 | 4 | 195–248 | 221–252 | 15–20% | 32–152 |

迭代数跳三倍与层级从 3 层变 4 层（`levels [519606, 37932, 1092, 54]`）**精确同步**，不是此前记的"薄壁屈服区 AMG 退化"。
同一份日志算端到端：pyamg 车道 vs PARDISO 车道（fast2）层 1–40 约 1.9x、41–93 约 1.3x、97–153 约 1.0–1.3x；到第 9880 步累计 11.36 h vs 14.93 h = **1.31x**。40-slab 的 2.4x 只在低层成立。

### 7.2 根因（离线证实）

提交 1fbac39 把 `max_coarse` 的语义从 pyamg 的"块数"改成"自由度"，`_build_hierarchy` 里 `max_coarse_blocks = max(10, 1000 // 6) = 166`。
level-2 随激活长大（61–70 层 978 → 71 层 1092 自由度），一超过 ~1000 就再粗化出 54–78 自由度的第 4 层：最粗层解得精确，但 1.1–1.7k 那一层从此只吃两遍 Jacobi，且 290 个聚集压成 13 个的粗化太激进，粗层校正失效。
B 组（`precond_B_mechanics.py --max-coarse 1000`）传的是 1000 **块** ≈ 6000 自由度，所以停在 3 层、拿到 46–56 次。

真实 `PyamgKrylovSolver`，CPU 路径，Jacobi×2（与 GPU 路径同一光滑器），tol 1e-6，fast2 倾倒矩阵，只改 `max_coarse`：

| 矩阵 | `max_coarse=1000`（4 层） | `max_coarse=6000`（3 层） |
|---|---|---|
| L090（560,607 自由块） | `[560607, 41070, 1188, 60]`，**169** 次，48.8 s，setup 6.6 s | `[560607, 41070, 1188]`，**63** 次，22.9 s，setup 6.7 s |
| L150（818,394 自由块） | `[818394, 60654, 1734, 78]`，**201** 次，83.2 s，setup 10.8 s | `[818394, 60654, 1734]`，**76** 次，39.0 s，setup 10.2 s |

与生产日志（3 层 65–79 / 4 层 195–248）和 2026-09-04 离线 GPU Jacobi 3 层 L150 = 76 次全部吻合。
脚本与原始输出：`C:\Users\user\Desktop\paper\fastscan_solver_paper\scripts\coarse_test.py`、`data\coarse_test_20260914.txt`、`data\run_stats_20260914.txt`。

### 7.3 资源账（为什么把上限调高不会过载）

`max_coarse` 只影响最粗层的规模；full 模式下最粗层补零容量 = `round_up(1.25 × max_coarse, 256)`，稠密逆显存 = 容量² × 8 B：

| `max_coarse` | 最粗层容量 | 稠密逆显存 | 本网格层数 |
|---|---|---|---|
| 1000（现值） | 1280 | 13 MB | 71 层后 4 层 |
| 2000 | 2560 | 52 MB | 全程 3 层（全高最粗层 1746） |
| 3000 | 3840 | 118 MB | 全程 3 层 |
| 6000 | 7680 | 472 MB | 全程 3 层 |

Cholesky 1746 阶 < 0.1 s，建层级时间不变；CG 迭代少三倍、每次迭代少一层、重建风暴消失 → 总资源下降。
唯一的过载情形是设得离谱：> 6 万时 pyamg 停在 61k 自由度的 level-1，稠密逆 30 GB 直接 OOM。规则：**稠密逆 = 值² × 8 B，上限别超过 ~8000**。

### 7.4 修法与自适应方案

**静态（先做，一行）**：MODE=5 配置 `mechanics` 块加 `"max_coarse": 3000`；`amg.py` 构造函数与 `linear.py` 的缺省 1000 → 3000；`pytest -m solver`（基线 90 passed）。验收：`padded to [...]` 末位 3840；`levels [...]` 全程 3 个数；71 层后 fresh 迭代 60–90；`NOT converged` ≈ 0。预条件器变了，MODE=5 40-slab 逐位基线要重立；PARDISO 金标门不受影响。

**为什么不按网格密度定**：正确的不变量是**最粗层的聚集数**，不是自由度或密度。SA 各层规模约为 n/13.5、n/470、n/10k；网格加密只会在细端多出层数，最粗层需要保留的聚集数由几何决定（薄壁件的整体低能模态需要 ≥ 200–500 个刚体片才能表示，13 个不够）。所以 `max_coarse ≈ 6 × N_agg_min`（N_agg_min 300–500 → 1.8k–3k 自由度）对同一几何随 n 不变；对 n < ~1.4 M 是 3 层，更大网格 pyamg 自动在细端加第 4 层，那一层不是问题层。

**动态（真正的自适应，下一轮实现）**：按测量选深度，而不是猜。
1. 建层级后得到 L 层；候选深度 L 与 L−1（截断 = 丢掉最后一层，对新的最粗层做稠密 Cholesky，k ≤ 3k 时 < 0.1 s，不必重建层级）。
2. 用真实右端各跑 ~10 次 PCG，比较"残差下降 / 墙钟"，取优者；按稀疏模式记住选择。
3. 触发再评估的信号就是本次的失败指纹：新建层级的迭代数超过滚动中位数的 1.5 倍（`stats` 里已有 fresh/reused 迭代计数，加一个滚动中位即可）。
4. 资源上限：候选最粗层 k 满足 k² × 8 B ≤ 显存预算（如 200 MB → k ≤ 5000）。
5. 备选正交手段：粗层（≤ 60k 自由度）在 GPU 上很便宜，改用 Chebyshev(3) 或 4–8 遍 Jacobi 光滑，可让深度选择不再敏感；`improve_candidates` 也是 pyamg 现成的开关。

**待办**：[ ] 静态修法 + 全高重跑对照（新 TAG，不覆盖 `v159_voxel_pyamg_full`）；[ ] 动态深度选择实现 + 单测；[ ] 粗层光滑器对比。

---

## 8. 全高重跑结果（`v159_voxel_pyamg_full`，2026-09-14 06:57Z 起，代码 29d29ba）

**§3 的修法达到了目标，但跑在第 120/182 层被另一条路径冻结。**

| 指标 | 结果 |
|---|---|
| `compiled_variants` / `capacity_growths` | **1 / 0**，全程（653 次层级构建、补零容量恒为 `[904368, 113152, 4864, 1280]`） |
| 主机 RSS 底线（60 s 采样，`rss_samples.csv`） | 15.5 GB（07:24Z）→ 17.2 GB（14:55Z）：+0.2–0.3 GB/h，随 nf 趋平，**无按形状驻留** |
| 进度 | step 7740 / 11885（65%），layer 120 / 182，30 帧，8.1 h |
| CG | 3 层阶段 25–80 次；≥71 层进入 4 层后 40% 复用求解触顶 300、重建后 170–260 次收敛（§7 已知问题） |
| PARDISO 兜底 | 2 次（15:01Z，连续两次重建后仍触顶） |

**冻结机理（与 §2 不同）**：两次 PARDISO 兜底把 0.94 M 自由度的整体系统做了因式分解，`shared_pardiso_solver`（phase23，复用句柄）把分解结果驻留在主机——RSS 在 2 分钟内 17.2 → 20.1 → **27.8 GB**（采样器 15:01–15:02Z，层级构建计数 639 → 645 正好夹住 run.log 第 9122/9126 行的两条 PARDISO 警告）。此时 guest 内存 = runner 29.9 GB + 页缓存 9.6 GB + shmem 1.5 GB ≈ 41 GB，撞到 `memory=40GB` 上限；15:06Z 起 runner 进入 **D 状态**（`State: D (disk sleep)`，`wchan=__vma_start_write`，`oom_kill` 计数 0，`procs_blocked` 0），日志与帧停止写入，GPU 10.5 GB 仍被该 pid 持有，`v_159.sh` 停在 `do_wait`。这与 2026-07-14 `.wslconfig` 注释里记的"求解器进程冻结为 D 状态"是同一类 WSL 病理，不是 OOM 杀死。

**结论与下一步**（按优先级）：

1. **根治触顶**：§7 的 `max_coarse` 语义修正（保持 3 层，CG 回到 60–80 次）——触顶消失则兜底不会发生。
2. **兜底不许驻留**：pyamg 车道的 PARDISO 兜底改为一次性求解（用完释放，不走 `shared_pardiso_solver` 的 phase23 复用），或 `maxiter` 300 → 600 让重建后的求解有余量。
3. **内存预算**：40 GB 上限下，runner 稳态 17–20 GB + 兜底一次 +8 GB 已无余量；兜底必须释放，或把页缓存压下去（`autoMemoryReclaim` 因 D 状态问题已禁用，不能靠它）。
4. 恢复：D 状态进程 `kill -9` 无效，只能 `wsl --shutdown`；30 帧、`run.log`、能量台账已落盘可用。

---

## 9. 第二轮修法：`max_coarse` 静态修正 + 兜底/释放不驻留（2026-09-15，提交见 git log）

针对 §8 的冻结链条，三处改动（全部只在迭代车道生效，PARDISO 车道不经过）：

| 改动 | 位置 | 验证 |
|---|---|---|
| `max_coarse` 缺省 1000 → **3000**（§7.4 静态修法）；MODE=5 配置显式 `max_coarse: 3000`、`maxiter: 600` | `amg.py`、`linear.py`、`0119-flash-voxel-fast-pyamg.json` | 真实矩阵 24 模式（`~/work/159/amgbench/V13.json`）：**全程 3 层**，迭代 15 → 76（同尺寸 4 层为 201），兜底 0，变体 1，容量增长 0，末位容量 3840，818k 复用求解 1.20 s（4 层 1.78 s） |
| pyamg 兜底改为**一次性 PARDISO**：phase 13 解完立即 phase −1 释放，不再经过 `shared_pardiso_solver` 的 phase23 复用句柄 | `amg.py::_direct_fallback`、`pardiso.py::pardiso_solve_once` | 单测：兜底后 `_SHARED_PARDISO` 为空、残差 1e-10、钉住行精确 |
| 路由到直接解（raft 释放、Newton 停滞重试）**之前**释放 pyamg 的全部设备缓冲（`release_device`），**之后**释放共享 PARDISO 的分解（`release_shared_solvers`） | `acceleration.py::accelerated_solver`、`amg.py::release_device`、`pardiso.py::release_states/release`、`linear.py::release_shared_solvers` | 2-slab shakedown 日志 214–215 行：`released the iterative solver's device buffers before the routed direct solve` / `released 1 PARDISO factorisation(s) after the routed direct solve`；`profile.meta.direct_factorisations_released = 1` |

**shakedown 验收**（`v159_voxel_pyamg_fix2`，2026-09-15 01:53–01:58Z）：`SHAKEDOWN_GATE_RC=0`，`all_checks_passed`，136 步 2.28 s/步，释放解 `release_u_max=2.20e-3`，CG 最大 45 次、`NOT converged` 0、PARDISO 警告 0，层级全程 3 个数、`padded to [904368, 113152, 4864, 3840]`。

**第一次 shakedown（01:40–01:45Z）失败的教训**：在 raft 释放解的装配阶段 `CUDA_ERROR_OUT_OF_MEMORY`（PARDISO 一次未跑）。释放解走 PARDISO，此时 pyamg 的整体固定结构 + 补零层级 + 缩放数据（~2.4 GB）是设备上的死重量，加上第二个力学 Problem 的装配临时量，贴着 JAX 默认 75%（12.2 GB）上限；`max_coarse=3000` 使粗层稠密逆多占 105 MB，正好推过线。修法即上表第三行的"路由前 `release_device`"，外加启动器导出 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`（`~/work/159/launch_v159_fix2_gates.sh`，仓库外）。开头那条 "Failed to allocate device memory of 4.00GiB" 在成功的跑里同样存在，是分配器探测，不是故障。

单测：pyamg 27 passed，`-m solver` 94 passed，`tests/unit` 342 passed，`tests/contract` 362 passed。

**待办**：[x] §7.4 静态修法；[ ] 全高重跑（新 TAG）；[ ] MODE=5 40-slab 逐位基线重立；[ ] §7 动态深度选择；[ ] `bucket` 自适应。

### 9.1 第二次全高（`v159_voxel_pyamg_fix2` stage 3，2026-09-15 02:11Z 起）

跑到 **step 6340 / 11885（layer 98 / 182，25 帧，5.6 h）** 时整个 WSL VM 被 Windows 更新重建：15:45:56–15:46:01 本地 Microsoft Store 自动安装
`WindowsSubsystemforLinux 2.7.14.0`，SCM 7040/7045 重装 WSL 服务、RestartManager 重启 `wsl.exe`，`uptime -s` = 15:46:51。
runner / `v_159.sh` / 采样器同时消失，无 OOM（`oom_kill 0`）、无 Traceback、无 `end rc=`。**与求解器无关。**

到中断为止的指标全部符合预期：层数全程 3（`{2,3}`）、`capacity_growths` 0、PARDISO 兜底 0、RSS 底线 17.9 GB 平台（峰 20.7 GB）、进程始终 R/S 态。
成本项：复用层级在 slab 内过期快——层 71–98 复用中位 ~95、P90 ~350、顶到 600 次上限累计 79（全部重建重试收敛），末段 5.2 s/步；这就是 §7 "动态过期判据 / 深度选择"要做的。

运维规则（已入记忆）：长跑前 `wsl --update` 并暂停 Windows 更新与 Store 应用更新；事后判断进程消失先查 `uptime -s` 与 System 日志。
