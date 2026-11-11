## Table 3 快速 no-keff → keff 前置扫描

该工具把 `balbaa-v2-model.json` 中登记的三组参数轴展开为明确的 `3×3×3=27` 个筛选工况：

- `P = {140, 220, 270} W`
- `v = {500, 650, 800} mm/s`
- `hatch = {0.08, 0.10, 0.12} mm`

> 仓库转写只给出了三个参数集合，没有论文逐行 case matrix。这里明确采用 full-factorial expansion，不能把生成的 27 行描述成论文逐行列出的 27 个实验。

### 为什么只跑 4 道

每个工况生成 4 道居中蛇形路径，并在第 3 道刚结束、第 4 道尚未开始的时刻测量。`pick_keff_measurement.py` 从实际路径计算该时刻和空间窗；`make_keff_table.py` 从 no-keff 的 `max_temperature_history` 中沿未扫描的 `+y` 方向量单侧熔池半宽 `L`，随后按 Balbaa Eq.19–24 计算该工况的 `keff`。

这不是完整 83 道验证运行，也不输出 Fig.14 指标。它只计算给后续 parity/Table 研究使用的工况级 `L` 和 `keff`。

### 1. 只生成计划（不启动计算）

```bash
python cases/AM-Benchmark/verification/v2-cube-rs/model/table3_keff_prescan.py \
  --config cases/AM-Benchmark/verification/v2-cube-rs/inputs/table3-keff-prescan.json \
  --repo "$PWD" \
  --output-root /home/user/work/159/output/v2_table3_keff_prescan \
  --mode plan
```

检查 `prescan_plan.json`。它记录每个工况的路径、运行、量测和 keff 命令，以及 config、mesh 和相关脚本的 SHA-256。工具还会在输出根目录生成 `material_config_no_keff.json`，仅把仓库中历史遗留的 box-159 绝对表路径重定位到当前 `--repo`；材料数值不变。

### 2. 单个工况先做冒烟

```bash
python cases/AM-Benchmark/verification/v2-cube-rs/model/table3_keff_prescan.py \
  --config cases/AM-Benchmark/verification/v2-cube-rs/inputs/table3-keff-prescan.json \
  --repo "$PWD" \
  --output-root /home/user/work/159/output/v2_table3_keff_prescan \
  --mode all \
  --case P220_V650_H0p12
```

`--case` 可重复。省略时处理全部 27 个工况。运行是串行的，避免多个 JAX/PARDISO 进程争用内存。

### 3. 分离运行与汇总

```bash
# 只运行 no-keff
... --mode run

# 已完成后只测 L 并推导 keff
... --mode collect
```

每个 case 目录包含：

```text
path.csv
path_ledger.json
run.log
prescan_case.json
*.vtu
path_used.csv
thermal_energy_ledger*.json*
keff.log
keff_derivation.json
k_liquid_keff.csv
```

根目录汇总（使用 `--case` 分批 collect 时按 `case_id` 合并，不覆盖已有工况；汇总绑定全部输入指纹，不允许不同计划混合；收齐 27 例后 JSON 的 `complete` 才为 `true`）：

```text
prescan_plan.json
keff_table3_summary.csv
keff_table3_summary.json
```

### Fail-closed 约束

- 必须至少有“量测道 + 下一道”，默认 `4` 道、量第 `3` 道。
- 必须 `thermal_output_every=1`，保证道结束附近存在可选帧。
- 每个工况独立运行 no-keff；不能跨工况复用 `L`。
- `L` 只有一排网格、熔池触碰量测窗或缺少 `max_temperature_history` 时直接失败。
- 短程 precursor 仍要求求解器正常退出且 `thermal_energy_ledger_summary.json` 标记 `complete=true`；另外以 `prescan_case.json` 的 `no_keff_complete` 状态和配置指纹作为 collect 前置门。它不借用完整热闸门的 `0.90 s` 观测覆盖定义。
- 输出 `keff` 是一次 Picard：`no-keff → L → keff`，不是向实验温度回调，也不是迭代标定。
