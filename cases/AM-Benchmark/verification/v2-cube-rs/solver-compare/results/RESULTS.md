# 单层求解器/平台对比 — 结果（2026-08-27）

算例：smoke2 合同的第 1 真实层（639 步，力学开、事件驱动节拍），14,800 单元（17,334 节点），
热 17k DOF / 力学 52k DOF，MKL 8 线程，RTX 5080。三臂串行，同一台机器，同一输入指纹。
原始产物：`/home/user/work/159/output/v2_cube_smoke_smoke2/solver_compare/<arm>/`；本目录只存小文件。

## 1. 速度

| 臂 | 平台 | 线性求解 | 墙钟 s | s/步 | 装配 s | 线性求解 s | 其余 s | 牛顿墙钟 s | 加速 |
|---|---|---|---|---|---|---|---|---|---|
| `cpu_pardiso`（基线） | CPU | PARDISO phase23 | 1364 | 2.134 | 745 | 365 | 254 | 1218 | 1.00× |
| `gpu_jax_bicgstab` | GPU 装配 | 上游 `jax_solver`：BCOO + BiCGSTAB + Jacobi，tol/atol 1e-10 | 926 | 1.450 | 131 | 421 | 375 | 686 | **1.47×** |
| `gpu_pardiso` | GPU 装配 | PARDISO phase23（CPU） | 831 | 1.300 | 136 | 364 | 331 | 594 | **1.64×** |

- 三臂非线性求解次数一致：1153 = 639 热 + 514 力学（事件驱动节拍没有被任何一臂的数值差翻转）。
- `gpu_jax_bicgstab` 的 `solver` 阶段调用数是 5568（另两臂 2784）：上游 jax 路径每次解多计一次阶段调用（CSR→BCOO 转换/残差核验），不是求解次数翻倍。
- 回退（spsolve）次数：三臂都是 0；Newton 不收敛：0；NaN：0。

## 2. 精度（对基线 `cpu_pardiso`）

| 量 | `gpu_pardiso` | `gpu_jax_bicgstab` | 来源 |
|---|---|---|---|
| T_min / T_max（8 个 summary 步，12 位有效数字） | 差 0 | 差 0 | run.log |
| u_max 相对差 | 0 | 1.4e-10 | run.log |
| vm_max 相对差 | 0 | 2.0e-10 | run.log |
| 逐步储能 `storage_j` 相对差（float64） | 3.3e-12 | 1.0e-7 | 台账 |
| 逐步沉积能量相对差 | 2.2e-15 | 2.2e-15 | 台账 |
| 末帧 T / u / vm / eqp（VTU，float32 存储） | 0 / 0 / 0 / 0 | 0 / 5.7e-14 m / 6.8e-8 MPa rms / 0 | `solver_compare.json` |
| 最大相对能量平衡误差 | 7.0e-6 | 7.1e-6 | 台账摘要 |

## 3. 判读

1. **1.47× 和 1.64× 全部来自 GPU 装配**（745 → 131/136 s，5.5×），线性求解本身没有变快：迭代法 421 s 比 PARDISO 365 s **慢 15%**。上游 `jax_solver` 路径在这个规模上"精度没丢、速度略输"。
2. **精度**：在 1.5 万单元、tol/atol 1e-10 的条件下，BiCGSTAB + Jacobi 把结果收敛到了与直接法 12 位相同的温度、1e-10 相对的位移/应力；没有触发回退。这与 v07 在 197k TET4 热问题上测得的 2e-4 K 偏差不同——**问题规模和条件数决定迭代法的表现**，本结论只对本规模成立，不能外推到 30 万单元的生产网格；到那个规模上需要重测（见 `experiments/solver/V07_APPS_COMPARISON.md` 的直接法/迭代法边界）。
3. **"其余"开销**（Python、主机端组装、设备拷贝）在 GPU 臂上从 254 s 涨到 331–375 s：设备↔主机传输是固定税，网格越大越容易摊薄。
4. **对生产的含义**：装配在 smoke2 全程占 54%，GPU 装配是当下最便宜的杠杆（配置级改动）；线性求解器不必换；真正的大头仍是事件驱动的力学求解次数（smoke2 全程 12,901 次）。

## 4. 复现

```bash
bash cases/AM-Benchmark/verification/v2-cube-rs/solver-compare/run_compare.sh      # 三臂，已完成的臂自动跳过
bash cases/AM-Benchmark/verification/v2-cube-rs/solver-compare/collect_results.sh
```

两处启动坑（已修进脚本）：`jax-fem-gpu` 在 miniconda3 下，不能用 miniforge3 的 `conda activate`，按绝对解释器路径启动；空的 `extra_argv` 不能展开成一个空字符串参数（argparse 报 `unrecognized arguments: ''`）。
