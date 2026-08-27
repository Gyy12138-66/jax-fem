# 求解器/平台对比 — smoke2 合同上的单层试验

**问题**：上游 jax-fem 默认的 `jax_solver`（GPU 单元级 → 主机 CSR 组装 → GPU BCOO + BiCGSTAB + Jacobi）在我们的 AM 热-力场景里有没有提升？以及 GPU 装配 + PARDISO 值多少？

**设计**：同一张网格、同一条第 1 层路径（639 行，见 `../CUBE-SMOKE.md` §2）、同一份合同 argv（力学开、事件驱动节拍），只换平台和线性求解器。三臂串行，每臂独立输出目录。

| 臂 | 环境 | 平台 | 线性求解 |
|---|---|---|---|
| `cpu_pardiso` | jax-fem-env | CPU | MKL PARDISO phase23（smoke2 基线） |
| `gpu_pardiso` | jax-fem-gpu | GPU 装配 | MKL PARDISO phase23（CPU） |
| `gpu_jax_bicgstab` | jax-fem-gpu | GPU 装配 | 上游 `jax_solver`：BCOO + BiCGSTAB + Jacobi，tol/atol 1e-10，maxiter 1e4，spsolve 回退开（计数） |

臂定义在 `arms/<name>.json`（conda 环境、平台、额外 argv）。

## 文件

| 文件 | 作用 |
|---|---|
| `run_compare.sh` | 启动器：读 `arms/*.json`，串行跑各臂，最后调用比较脚本 |
| `compare_solver_arms.py` | 速度（墙钟、每步、装配/求解/牛顿分段、回退次数）、健康（Newton 失败、NaN）、精度（末帧对基线：T/u 最大差，vm/σxx RMS 差，eqp 最大差，T_ref 翻转单元数） |
| `collect_results.sh` | 把各臂的小文件（profile、used_config、台账摘要、日志首尾、summary 行）拷进 `results/` |
| `results/` | 对比结果（`solver_compare.json`、`RESULTS.md`、各臂子目录） |

求解器产物（VTU 等）留在 `OUT`（默认 `/home/user/work/159/output/v2_cube_smoke_smoke2/solver_compare/<arm>/`）。

## 用法

```bash
cd /home/user/work/d11_B_tree
bash cases/AM-Benchmark/verification/v2-cube-rs/solver-compare/run_compare.sh        # 三臂
ARMS="gpu_pardiso" bash .../run_compare.sh                                             # 单臂
bash cases/AM-Benchmark/verification/v2-cube-rs/solver-compare/collect_results.sh     # 收集
```

前置：smoke2 的预检产物（`PRE`，含 `runner_contract.json` 与 `v2_cube_smoke_path_layer1.csv`）。

## 判读原则

- 没有通过/失败阈值。基线是 `cpu_pardiso`；另两臂报**差多少**，由读者按用途判断。
- `gpu_pardiso` 的期望：与基线数值接近到浮点噪声级（同一直接法，装配顺序不同），墙钟看装配占比。
- `gpu_jax_bicgstab` 的期望：迭代容差带来的漂移；力学矩阵（非对称，1e-4/1e-2/1% 刚度对比）是 Jacobi-Krylov 的弱项；回退次数是关键指标。
- 事件驱动的力学节拍（T_cut 越过即强制求解）对三臂完全相同，所以求解次数应一致；若不一致，说明某臂的热场偏差已经翻转了事件。

结果见 `results/RESULTS.md`。
