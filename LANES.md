# 两条主线（2026-09-04 起）

本仓库只为两条主线服务。其余目录要么是这两条线的验证资产，要么是冻结遗产。

命名（用户 2026-09-04 定）：**快速扫描支线** = `precond-optimization`；**细节扫描支线** = `meltpool-source`。

## 主线 A：快速扫描支线（`precond-optimization`）：flash 模型加速重构

- 目标：层集总（flash）模式下热-力每步成本下降，服务 v159/0119 件与 V2 立方体的生产节奏。
- 精度门：逐位金标门 5/5；fast 节奏的保真偏移（平均 vm、回弹）随报告登记。
- 分支 / worktree：`precond-optimization` @ `~/work/d11_B_tree`。
- 独占文件：`jax_fem_am/solvers/**`、`jax_fem_am/simulation/acceleration.py`、`jax_fem/solver.py`。
- 快速门：`pytest -m solver`（线性求解器注册表、pyamg/Krylov、pardiso phase23 等单测）。
- 用户主导实现；Claude 提供对照、测试、review。

## 主线 B：细节扫描支线（`meltpool-source`）：层级路径扫描 + 熔池特征高精度

- 目标 B1：事件表路径 + 扫掠线段热源 + 自适应 dt，得到按扫描矢量集总的层级模型，
  并在 V2 立方体域上量化 flash 相对逐道的误差（反哺主线 A 的 flash 标定量）。
- 目标 B2：Goldak 双椭球、蒸发热沉与沸点钳制、液相各向异性有效导热，
  以 V1 单道三角（AMB2018-02 CBM-A/B/C）为回归锚：深度保持近精确，长度与近固相线冷却速率收窄。
- 分支 / worktree：`meltpool-source` @ `~/work/meltpool_tree`（从 test 1baa904 分出）。
- 独占文件：`jax_fem_am/physics/thermal.py`、`jax_fem_am/process/scan_path.py`、`jax_fem_am/mesh/quadrature.py`。
- 快速门：`pytest -m meltpool`（待建：缩小版 V1 单道回归，粗网格少步数，秒级）。
- 外围工具链在 AM-Sim 仓库：切片/事件表生成与校验、网格验收、熔池提取、标定反演。

## 共享文件规则

- `simulation/stepper.py`、`simulation/runner.py`、`config/schema.py`：只做加法。新行为一律开关，默认值与默认行为不变。
- 改共享文件前先 merge test，改完立即跑对方主线的快速门。
- 合入 test 的条件：金标门 5/5 逐位 + `tests/contract` 全过 + 两条线的快速门都过。
- 同一时间只允许一条主线合入 test；另一条随后 merge test，保持共同基点。
- 输出目录分开：A 用 `~/work/159/output/v159_*`、`output/v2_*`；B 用 `output/mp_*`。

## 现有内容归类

| 目录 / 分支 | 归类 |
|---|---|
| `cases/AM-Benchmark/verification/v1-single-track` | B 的回归锚 |
| `cases/AM-Benchmark/verification/v2-cube-rs` | A 的生产节奏基准；B1 的 flash 对逐道对照域 |
| `cases/159_simulation` | A 的生产算例（0119 件） |
| `cases/AM-Benchmark` 主线（L0/D-11 等） | 验证资产，按需引用，不再作为独立主线推进 |
| `cases/kaess_2023` | 冻结（2026-07-26 放弃复现） |
| `legacy/` | 只读遗产 |
| `codex/r3-optimization` worktree | 冻结，可移除 |
| `tests/benchmarks` | 上游 jax-fem 基准，不动 |

## 两条线都会碰的已登记缺陷（KNOWN_ISSUES.md）

- `--dt` 在扫描相被路径行距静默覆盖（B 需按 dt 细分线段；A 的 flash 不受影响）。
- `--scan-steps-per-layer` 是每条 hatch 线语义（B 用事件表绕开；文档需改）。
- `run_audit` 不能审计热-only 运行（B 的熔池运行全是热-only）。
- T_cut 事件在移动源下几乎每步强制力学求解（B 必须解耦；A 决定每步成本）。

## 主线 B 首批任务

1. 冻结事件表合同（`docs/contracts/path-event-table.md`），AM-Sim 侧写校验器并提交样例 CSV 作为本仓库 contract 测试夹具。
2. 扫掠线段热源（一步沉积一条或一组矢量），能量台账闭合作验收。
3. 自适应 dt（扫描 / 驻留 / 铺粉三种节奏）与"力学按层或按事件触发"开关，解除 T_cut 强制力学。
4. 缩小版 V1 单道回归进 `-m meltpool`。
5. Goldak、蒸发热沉、液相各向异性 k，各自附能量台账与 V1 三角复跑。
6. 张量积渐变 HEX8 试样生成器（AM-Sim/mesh），多道多层熔池算例。

## 2026-09-23 分支布局调整（论文收尾）

- **`precond-optimization` 冻结在 82138a4**：论文主结果 E0j（d6bd04f）与 E1j（722d915）的代码，不再改动。
- **`test` 快进到 82138a4 后，缺省值改为 E0j 的设置**：pyamg `scale_policy="frozen"`、`maxiter=800`、`rebuild_iter_factor=0`；jax 热学块 `jit=true`。
  `device` 仍缺省 `cpu`，`--xla-linear-solver auto` 的力学仍是 PARDISO。依赖旧缺省值的配置已显式写死旧值
  （`0119-flash-voxel-fast-pyamg.json`、`-pyamg-frozen.json`、`-hybrid.json` 写 `"jit": false`；仓库外 `~/work/159/0911/inputs/0911-half2p5-flash-fast-pyamg.json` 另写 `"scale_policy": "current"`）。
- **消融支线 `paper-ablation`** @ `~/work/ablation_tree`，从上述 test 提交分出，基线 tag `paper-base-20260923`：
  - 用途：论文消融与扩展实验（E5c、E5a、E5d、E3、E4、E7、E6 后半、E8、E9、E10、E11）的配置、启动脚本与分析脚本。
  - **求解器代码冻结**：`jax_fem_am/solvers/**`、`jax_fem/solver.py`、`jax_fem/fe.py`、`jax_fem_am/simulation/**` 不改，保证所有消融与 E0j 同一求解器代码；只加 `cases/**` 下的配置与脚本。
  - 若确需改求解器，先在 test 上改并重跑门禁，再合入本支线，受影响的消融重跑。
  - 输出目录：`~/work/159/output/abl_*`（E10 为 `abl_0911_*`）。
- `halfmodel-0911`（a847f75，基于 185041c）：E10 前需把其提交移到 `paper-ablation` 上并重跑单测与 2-slab。
