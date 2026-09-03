# jax-fem 原生求解器配置参考(v159 管线)

> 版本:2026-09-03,分支 `precond-optimization`。适用于 `jax_fem_am.simulation.runner` 驱动的 v159(0119 件)热-力仿真;所有数值来自本仓库 vendored 的 `jax_fem/solver.py` 与 `jax_fem_am/simulation/acceleration.py`。

## 1. 求解链路:配置怎样一层层变成一次线性求解

```
inputs/*.json  linear_solver 块
      │  make_159_preflight.py(stage 1,编译契约;指纹校验,配置变了才重编译)
      ▼
preflight/runner_contract.json  argv(冻结的 --xla-* / --mechanics-* 命令行)
      │  jax_fem_am.simulation.runner(解析 argv;acceleration.install_solver_patch 包裹 stepper.solver)
      ▼
accelerated_solver(problem, solver_options)
      │  rewrite_solver_options:把 Newton 选项里的 linear 块整体替换为配置指定的求解器
      │  若 problem.prefer_direct_linear_solver 且配置为迭代类 → 改为 PARDISO(直接法路由)
      ▼
jax_fem.solver.solver → Newton 循环 → linear_solver(A, b, x0, linear_options)
      ├── jax_solver   → jax_solve:GPU 上 Jacobi 预条件 BiCGSTAB / CG / GMRES
      ├── custom_solver → _PardisoCustomSolver:CPU MKL PARDISO(phase23 复用符号分解)
      ├── spsolve_solver → SciPy spsolve(单线程直接法;也是失败回退路径)
      ├── petsc_solver / amgx_solver → 预留钩子(AMGX 需 pyamgx,未安装)
      └── 任何异常(非 Newton 停滞)且 xla_fallback_to_spsolve → 用 spsolve 重试该次求解
```

**stage 3 从不直接读 JSON**,只读 stage 1 冻结的契约。改了任何求解器字段,启动时 `STAGES` 必须包含 `1`,否则沿用旧契约。

## 2. 配置块 `linear_solver` 字段(`make_159_preflight.py` 映射)

| 字段 | 取值 | 映射到的 CLI | 说明 |
|---|---|---|---|
| `platform` | `gpu` / `cpu` | `--xla-platform` + 契约 `platform`(设置 `JAX_PLATFORM_NAME`) | 决定装配/残差内核在哪跑;与线性求解器无关 |
| `python_bin` | 路径 | 契约 `python_bin` | 生产用 `miniconda3/envs/jax-fem-gpu/bin/python`(CUDA 12) |
| `solver` | `pardiso` / `jax` / `spsolve` / `petsc` / `amgx` / `keep` | `--xla-linear-solver` | `keep` = 不改写,沿用 stepper 里的 `spsolve_solver` |
| `pardiso_mode` | `base` / `nocmp` / `cache-idx` / `phase23` / `fp32ir` | `--xla-pardiso-mode` | 生产用 `phase23`;字段为必填(preflight 直接索引) |
| `cell_target_batch_size` | 整数 | `--xla-cell-target-batch-size` | 装配分批,控显存;生产 32768 |
| `jax_precond` | bool | `--xla-jax-precond` | **必须显式为 true**,见 §3 的默认值反转 |
| `jax_method` | `bicgstab` / `cg` / `gmres` / `spsolve` | `--xla-jax-method` | 热学活跃块精确对称,`cg` 可用 |
| `jax_tol` / `jax_atol` | float | `--xla-jax-tol` / `--xla-jax-atol` | 传给 `jax.scipy.sparse.linalg.*` 的 `tol` / `atol` |
| `jax_maxiter` | int | `--xla-jax-maxiter` | 上游默认 10000 |
| `jax_skip_residual_check` | bool | `--xla-jax-skip-residual-check` | 关闭 §4 的绝对残差硬断言;放宽 tol 时**必须**同时打开 |
| `mechanics_direct` | bool | `--mechanics-direct-solver` | 混合模式:力学全部走 PARDISO,热学保持 `solver` 指定的迭代法 |

## 3. `jax_solver` 参数:上游默认 vs 本管线 wrapper 默认

| 参数 | `jax_fem.solver.jax_solve` 默认 | wrapper(`linear_options_from_args`)默认 | 生产验证过的取值 |
|---|---|---|---|
| `precond`(Jacobi) | **True** | **False**(`--xla-jax-precond` 未给即关) | true |
| `method` | `bicgstab` | 不传 → 上游默认 | `bicgstab`、`cg` |
| `tol`(相对) | 1e-10 | 不传 → 1e-10 | 1e-10(紧)、1e-6(松,需跳断言) |
| `atol`(绝对) | 1e-10 | 不传 → 1e-10 | 1e-6(松) |
| `maxiter` | 10000 | 不传 | 10000 |
| `restart`(GMRES) | 20 | 不传 | 未用 |
| `solve_method`(GMRES) | `batched` | 不传 | 未用 |
| `check_residual` | True | True(仅 `--xla-jax-skip-residual-check` 时 False) | 两者都验证过 |

**默认值反转是最大的坑**:上游默认开 Jacobi,wrapper 默认关。配置里不写 `jax_precond: true` 等于裸 BiCGSTAB。

## 4. `jax_solve` 内部流程与硬断言

1. PETSc AIJ → CSR(`getValuesCSR`)→ JAX BCOO(按稀疏模式缓存,`bcoo_cache_hits` 计数)。
2. `precond` 为真时 Jacobi 取自对角:`M = x / diag(A)`。
3. 调 `jax.scipy.sparse.linalg.{bicgstab,cg,gmres}(A, b, x0=x0, M=M, tol=tol, atol=atol, maxiter=maxiter)`;收敛判据 `‖r‖ ≤ max(tol·‖b‖, atol)`。
4. `check_residual` 为真时:`err = ‖A x − b‖`(**绝对值**),`assert err < 0.1`,并把不满足的解置 NaN。热学 flash 步 ‖b‖ ~ 1e9,这道门等效强制相对精度 1e-10——放宽 `tol` 而不跳过它,断言必挂、逐步回退 spsolve。
5. 断言/异常被 `accelerated_solver` 捕获:若 `xla_fallback_to_spsolve`(默认 true)则改用 `spsolve_solver` 重试;Newton 停滞(`Newton solver did not converge`)不回退,直接抛给上层(力学 cutback 处理)。回退的 spsolve 在全网格矩阵上是**单线程**,90 万自由度的力学矩阵一次要 30 分钟以上。

## 5. Newton 层参数(线性求解器之外)

| 问题 | 来源 | tol | rel_tol | max_iter | 线搜索 | 其它 |
|---|---|---|---|---|---|---|
| 热学 | `make_thermal_solver_options`,未覆盖 → 上游默认 | 1e-6 | 1e-8 | 100 | 仅相变激活 | 初值 = 上一步温度;`xla_residual_only_check` 对热学注入 `residual_only_check`(收敛检查不重算切线) |
| 力学 | `run_mechanics` 基础值 → CLI 覆盖 | 1e-9 | **5e-5**(`--mechanics-rel-tol`) | **50** | **开** | acceptance `abaqus`(disp 0.01 / force 0.005 / fallback after 9 → 0.02);cutback `--mechanics-max-cuts 3`(增量最多切 2³ 段);`jacobian_reuse 0`;温度下限 293.15 K |

## 6. 直接法路由标记

`problem.prefer_direct_linear_solver = True` 时,`accelerated_solver` 在配置求解器属于迭代类(`jax_solver` / `petsc_solver` / `amgx_solver` / `cg|bicgstab|gmres_solver`)的情况下,把**该问题的**线性块改写为 `{"custom_solver": _PardisoCustomSolver("phase23")}`;配置本身是 pardiso / spsolve 时标记无效。

- **release 问题恒定打标**(stepper release 块):切 raft + 刚体锚固的系统是全流程最病态的,BiCGSTAB 必挂(实测 err≈42)。
- **主力学问题在 `--mechanics-direct-solver` 时打标**:即混合模式。

`_PardisoCustomSolver` 定义了 `__deepcopy__` 返回自身,所以 `rewrite_solver_options` 每次求解的深拷贝不会丢掉 MKL 句柄与符号分解,跨 Newton 迭代复用。

## 7. 其它 `--xla-*` 运行时开关(生产契约当前值)

| CLI | 生产值 | 作用 |
|---|---|---|
| `--xla-platform` | gpu | 导入 jax 前设 `JAX_PLATFORM_NAME` |
| `--xla-preallocate` | off | `XLA_PYTHON_CLIENT_PREALLOCATE=false`,显存按需分配 |
| `--xla-cell-target-batch-size` | 32768 | 装配分批 |
| `--xla-cell-num-cuts` | 未设 | 手动指定分批数 |
| `--xla-dof-to-quad-cache` | on | 缓存 dof→积分点插值 |
| `--xla-jit-loop-kernels` | on | 主循环内核 jit |
| `--xla-step-predicate-cache` | on | 步类型判据缓存 |
| `--xla-skip-unused-mechanics-material` | on | 跳过未用材料分支 |
| `--xla-thermal-only-mechanics-surrogate` | on | 纯热学步用力学代理 |
| `--xla-residual-only-check` | on | 热学 Newton 收敛检查不重算切线 |
| `--xla-thermal-warm-start` | off | 线性求解初值注入 |
| `--xla-lazy-output-postprocess` | off | 延迟输出后处理 |
| `--xla-quiet-jax-fem-logs` | on | 压制 jax_fem 日志 |
| `--xla-fallback-to-spsolve` | on | §4 的回退 |
| `--xla-mem-fraction` | 未设 | `XLA_PYTHON_CLIENT_MEM_FRACTION` |
| `MKL_NUM_THREADS`(环境变量,`v_159.sh` 默认 8) | 8 / 24 | PARDISO 线程数;24 线程实测仅快 5%(瓶颈在串行流水线) |

## 8. 四种模式与文件

| MODE | 名称 | 配置文件 | 热学线性解 | 力学线性解 | 装配 |
|---|---|---|---|---|---|
| 1 | jax-gpu | `inputs/0119-flash-voxel-fast-jax.json` | GPU Jacobi-BiCGSTAB(1e-6,跳断言) | 同左(release 除外) | GPU |
| 2 | cpu-pardiso | `inputs/0119-flash-voxel-fast-cpu.json` | PARDISO | PARDISO | CPU |
| 3 | gpu-assembly | `inputs/0119-flash-voxel-fast.json` | PARDISO | PARDISO | GPU |
| 4 | hybrid | `inputs/0119-flash-voxel-fast-hybrid.json` | GPU Jacobi-CG(1e-6,跳断言) | PARDISO | GPU |

启动器 `/home/user/work/159/launch_mode.sh`:`MODE=<n> [TAG=…] [MKL_NUM_THREADS=…] [SHAKEDOWN_SLABS=…] STAGES="1 2 S 3" bash launch_mode.sh`。

## 9. 实测验证矩阵(2026-09-01 ~ 09-03)

| 设置 | 能量门 | 2-slab shakedown | 生产 / 中高度 | 结论 |
|---|---|---|---|---|
| 模式 3(PARDISO) | 3.85 s/步 ✅ | 3.95–3.99 ✅ | 19.07 h 完成,均值 5.78 s/步,层 100+ 约 7.5 | 基准车道;成本随活跃自由度温和增长 |
| 模式 1,紧容差(1e-10,断言开) | 3.55 ✅ | 2.47 ✅(release 已路由 PARDISO) | 层 21–28 掉到 7.2 s/步 | 力学迭代数爆炸 |
| 模式 1,松容差(1e-6,跳断言) | — | 2.34 ✅ | 15 h 到层 83,2.86 → 18 s/步 | 数值全程干净;同样输在力学 |
| 模式 4 hybrid(CG + 力学直接法) | 3.21 ✅ | 3.52 ✅ | 40-slab 中高度测试进行中 | 打印态与模式 3 逐位一致 |

A 组诊断(倾倒矩阵离线):热学活跃块 κ(Jacobi 缩放)12–42、与高度无关,Jacobi-Krylov 12–22 次@1e-6;力学活跃块 κ 5.8e4(L30)→ 3.5e6(L150),Jacobi-BiCGSTAB 2000 次不收敛。**模式 1 的所有减速都来自力学步。**

## 10. 已知坑清单

1. `jax_precond` 不写 = 关(§3)。
2. 放宽 `jax_tol` 必须同时 `jax_skip_residual_check: true`(§4)。
3. 迭代法不要用于 release,也不要用于全尺度力学(§6、§9)。
4. 失败回退是单线程 spsolve,不是 PARDISO;生产里出现 `WARNING: experimental linear solver failed` 就等于慢 100 倍。
5. 改求解器配置后 `STAGES` 必须含 `1`(§1)。
6. 2-slab shakedown 的 `release_u_max` 无参考价值:切 raft 后剩 1 mm 薄板,系统近奇异,同一打印态四个臂给出 2–151 mm;门只查存在性不查量级。
7. `pardiso_mode` 是 preflight 必填字段,即使 `solver: jax` 也要保留。
8. 2-slab 门覆盖不到高层数病态(力学 κ 增长、能量审计的 hold 步超差都在层 13+ 才出现),速度类结论要用 ≥40 层的中高度 shakedown。
