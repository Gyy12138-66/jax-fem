# 热—力求解器分解提速实验

## 目的与边界

这是一条独立的求解器研究通道，用来回答一个窄问题：在不改变热—力模型、材料参数、时间路径和 Newton 接受准则的前提下，力学切线矩阵的分解能否减少装配与直接法分解成本。

当前模型已经按时间步交错求解热场和力学场。本实验不再拆分物理方程，也不把 GPU 当作 FP64 稀疏直接求解器。这里的 GPU 仅执行 JAX 单元残量/切线装配；CSR 系统仍回到主机，由 MKL PARDISO 求解。

冻结的 `results/` 只作为既有证据，脚本不会写入或覆盖它。新的运行产物默认进入：

```text
/home/user/work/159/output/v2_cube_smoke_smoke2/decomposition_compare/
```

## 被检验的公式

标准 Newton 在每次修正上重新构造并分解一致切线：

```text
J(u_k) Δu_k = -R(u_k)       [PARDISO phase 23]
u_{k+1} = u_k + α_k Δu_k
```

modified Newton 保留最近一次接受的切线分解，在有限次数内只换右端项：

```text
J(u_r) Δu_k = -R(u_k)       [PARDISO phase 33, r <= k]
u_{k+1} = u_k + α_k Δu_k
```

其中每次修正后仍重新计算真实残量 `R(u_{k+1})`，不能用旧切线推算的残量代替收敛检查。当前候选接口为：

```text
--mechanics-jacobian-reuse 2
--mechanics-jacobian-refresh-ratio 0.9
```

含义是最多复用两次，并在真实残量下降不足、达到刷新比守卫时重新构造切线。准确语义应以实现及 profile 计数器为准；实验不通过调节这两个数字去拟合场结果。

四个 arm 都启用 `--mechanics-residual-only-check`。这样 full 与 modified 都会先用真实残量判断收敛，避免把“省掉最终已收敛状态的一次无用切线”混入 modified-Newton 收益；两者的唯一非平台差异是是否允许 lagged tangent。

## 2 × 2 对照

| arm | 装配平台 | Newton/直接法路径 | 同平台基线 |
|---|---|---|---|
| `cpu_full_phase23` | CPU | 每次切线 + phase 23 | 自身 |
| `cpu_modified_phase33` | CPU | 有限复用 + phase 33/刷新 | `cpu_full_phase23` |
| `gpu_full_phase23` | GPU | 每次切线 + CPU phase 23 | 自身 |
| `gpu_modified_phase33` | GPU | 有限复用 + CPU phase 33/刷新 | `gpu_full_phase23` |

同平台成对比较用于判断公式分解是否提速；`cpu_full_phase23` 还作为统一的数值差基线，用于发现 CPU/GPU 后端漂移。

## 两级运行

默认调用是安全的，只打印实验计划：

```bash
cd /home/user/work/d11_B_tree/cases/AM-Benchmark/verification/v2-cube-rs/solver-compare
bash run_decomposition_compare.sh
```

快速通道复制 layer-1 CSV 的表头和前 16 行到独立输出目录，原始 CSV 不变。它默认强制 `--mechanics-every 1`，目的是形成每步都有力学求解的 **solver stress test**，不是冻结合同的生产墙钟：

```bash
bash run_decomposition_compare.sh --run quick
```

完整通道使用原 639 行 layer-1 路径，并默认保留 `runner_contract.json` 中冻结的力学节拍：

```bash
bash run_decomposition_compare.sh --run full
```

如需专门研究力学频率，可以显式设置 `MECHANICS_EVERY=N`；这样的结果必须标为 cadence override，不能与合同运行混报。例如，完整路径的每步力学 stress test 为：

```bash
MECHANICS_EVERY=1 bash run_decomposition_compare.sh --run full
```

两种通道默认都是每个 arm 先做 1 次独立进程 warmup，再做 4 次独立进程计时；arm 串行运行，计时重复按“奇数轮正序、偶数轮逆序”执行，使 full/modified 各有相同次数先跑与后跑。GPU 预分配关闭。warmup 和计时进程按平台共享 `$OUT/<mode>/jax-cache/<platform>` 持久编译缓存，使首次大于 JAX 默认缓存阈值的编译可以被后续独立进程复用；每个计时重复仍是完整的新进程端到端运行。缓存机制和默认 1 s 编译阈值见 [JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)。可通过 `WARMUP`、`REPEATS`、`STEPS`、`MKL_NUM_THREADS`、`MECHANICS_EVERY`、`OUT` 覆盖；低于约 5% 的候选收益应保留偶数次 AB/BA 轮换，正式合同结论应保留完整模式的冻结节拍。例如：

```bash
STEPS=32 REPEATS=5 OUT=/home/user/work/159/output/decomposition_32 \
  bash run_decomposition_compare.sh --run quick
```

每个 mode 根目录都会生成 `experiment_manifest.json`，绑定路径与合同 SHA-256、期望步数、线程、arm JSON、Python/JAX/NumPy/SciPy 环境、Git HEAD 和运行时代码树指纹。只有 manifest 完全一致，且 arm 的 profile、used-config、末帧 VTU、完整 ledger、步数和 arm JSON 都匹配时才会跳过；任何漂移都 fail-closed 并要求新的 `OUT`。已有但不完整的目录不会删除或覆盖。

## 读取什么证据

`compare_decomposition_arms.py` 复用 `compare_solver_arms.py` 的最后帧观测算子，并输出 `decomposition_compare.json`。每个 arm 至少检查：

- `wall_seconds`、`stage_seconds.assembly/solver`、`meta.newton_wall_seconds` 的四次轮换重复及配对中位数；
- `meta.mechanics_jacobian_builds`、`meta.mechanics_jacobian_reuse_hits`、`meta.mechanics_jacobian_refreshes`，与 thermal 分项隔离后证明候选路径确实被执行；
- 完整的 `meta.pardiso_stats`，以及按 nonlinear solve 做差得到的 `meta.pardiso_stats_by_scope.mechanics`；力学 phase 23、`backsolve_hits`/phase 33 才用于确认复用激活；
- ledger complete、Newton 不收敛、fallback、NaN；
- 相对 `cpu_full_phase23` 的 `T`、`u`、von Mises、`sxx`、等效塑性应变和应力自由温度差；
- modified arm 相对同平台 full arm 的字段差和 wall-time speedup。

生产状态事务边界另由
`tests/contract/test_v06_adapter.py::V06AdapterTest::test_trial_residual_failure_and_cutback_do_not_commit_tensor_history`
守住：单 HEX8 会执行 9 次真实 J2 trial residual，制造一次 Newton 失败并走真实的两半步 cutback；`eps_p/eps_ref/eqp` 在这些试探和重试中保持未提交，只有显式接受后调用 `compute_eqp_update` 才更新。cold reset 与 stress-free-reference capture 是求解前的生命周期事件，不属于这个 Newton 事务测试。

## 判读规则

1. **先确认实验被激活。** modified arm 的 `jacobian_reuse_hits` 和 PARDISO `backsolve_hits` 都为零时，本次算例没有进入可复用区间，结果只能记为“不具判别力”，不能记为“无提速”。
2. **先过数值健康再看速度。** 任一不完整 ledger、Newton 不收敛、fallback 或 NaN 都阻止性能晋级。最终场差必须与 full arm 一起报告，不能只报墙钟。
3. **看轮换后的配对中位数。** 每个 repeat 内 CPU modified 对 CPU full、GPU modified 对 GPU full，再取配对 speedup 中位数；不要把 GPU 装配收益或固定运行顺序误记为 modified Newton 收益。
4. **quick 只验证管线。** 前 16 个扫描步可能缺少足够的塑性/线搜索迭代，适合发现 CLI、计数器、收敛和输出错误，不足以证明完整 layer-1 提速。
5. **健康 gate 是 fail-closed。** 任一缺 profile、步数/身份不符、字段比较失败、不完整 ledger、fallback、Newton 不收敛或 NaN 都使整个 arm invalid，比较器不给 speedup 并以非零状态退出。
6. **完整合同通道才支持生产性能结论。** 至少四次轮换重复都健康，并且 wall、Newton、solver 分项与“少构造切线、少数值分解”的计数方向一致，才可称为候选优化；显式 cadence override 的完整路径仍只是 solver stress test，否则保留为负结果。
7. **不做性能标定。** `reuse=2` 和 `refresh_ratio=0.9` 是预先登记的候选策略。若以后扫描参数，应建立新的 arm 和独立输出，不能在看到结果后修改本组定义。

这条通道回答的是“切线与分解复用是否有净收益”。它不替代网格收敛、物理 benchmark 或 XRD/位移实验验证。

审查后的公平 8 步四臂结果见 [`decomposition-results/QUICK8-FAIR-20260828.md`](decomposition-results/QUICK8-FAIR-20260828.md)。原 [`QUICK8-20260828.md`](decomposition-results/QUICK8-20260828.md) 因 residual-only 条件不公平且顺序固定，已保留为撤回的混杂试跑。两者都不能代替完整 639 步结论。
