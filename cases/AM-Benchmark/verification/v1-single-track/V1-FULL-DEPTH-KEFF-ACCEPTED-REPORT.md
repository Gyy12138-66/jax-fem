# V1 Balbaa 全深度热源 + `keff` 接受结果报告

- **报告日期**：2026-08-24
- **结果状态**：接受为当前 V1 纯热基线
- **适用范围**：IN625、NIST AMB2018-02 CBM-B 对应单道工况的纯热计算
- **不包含**：力学求解、残余应力、V2 多道/多层验证
- **代码提交**：`6181bc4cb1ff897a539b90aca0d6017c959fbbf4`
- **运行目录**：`/home/user/work/159/output/v1_two_arm_full_depth_6181bc4_20260824T065833Z/full_depth_keff`
- **运行环境**：box-159 WSL，CPU，`jax 0.10.2`，`petsc4py 3.25.1`，PARDISO linear solver

> 本报告接受的是“基于 Balbaa 模型、对论文未公开或歧义项采用已登记解释的复现”，不是 Balbaa ABAQUS 模型的逐项完全复制。尤其是热源作用域、`keff` 更新方式、边界换热、相态处理和求解离散存在明确差异。

## 1. 接受结论

当前接受第二版，即：

1. 使用 Balbaa Eq. 18 指数体热源；
2. 光学穿透深度为 `100 µm`；
3. **不把热源截断并重新归一化到 `20 µm` 粉层**，而是允许指数热源沿完整 `100 µm` 深度向下沉积；
4. 在第一版全深度、无 `keff` 结果基础上，以熔池半宽进行一次 Picard 预计算，得到 `keff = 58.8472 W/(m·K)`；
5. 纯热运行通过 52/52 输出帧审计和 103/103 步离散能量账本检查。

该版本消除了 `20 µm cutoff + renormalization` 造成的约 `5.52×` 局部热源放大。峰值温度由被否决版本的 `11953.5 K` 降至 `2886.8 K`，与 Balbaa 图示峰值约 `2835 K` 接近。

接受该结果不表示所有几何指标已与 Balbaa ABAQUS 对齐：当前深度约为 Balbaa 值的两倍，长度高约 47%。相对 NIST，深度吻合，但长度仍低约 29%。

## 2. 最终结果及参考对照

熔池边界统一取 `1563 K`（`1290 °C`）solidus 等温面。

| 指标 | 本次第二版 | Balbaa ABAQUS（指数源） | 本次 vs Balbaa | NIST CBM-B | 本次 vs NIST |
|---|---:|---:|---:|---:|---:|
| 峰值温度 [K] | 2886.82 | 约 2835 | +1.83% | 未报告 | — |
| 熔池宽度 [µm] | 142.36 | 131 | +8.67% | 133 | +7.04% |
| 熔池深度（相对基板顶面）[µm] | 90.70 | 44 | +106.15% | 91 | −0.33% |
| 熔池长度 [µm] | 555.00 | 378 | +46.83% | 780 | −28.85% |
| 冷却速率 `1290→1190 °C` [°C/s] | `1.8513×10^6` | 该工况未报告 | — | `9.35×10^5` | +98.0% |

说明：

- Balbaa 宽度 `131 µm` 为正文明确值；深度和长度由论文图 9–10 数字化，估计读图误差分别约 `±2 µm`、`±10 µm`。
- NIST CBM-B 数值来自 Lane 2020 汇总：宽/深/长为 `133/91/780 µm`。
- 当前熔池宽度、深度采用 solidus 边界插值值；长度为稳态帧中位数。
- 本次 probe peak 为 `2868.45 K`，全域最大值为 `2886.82 K`。

## 3. 本次实际运行参数

### 3.1 工艺、几何和热源

| 参数 | 实际值 | 来源分类 | 说明 |
|---|---:|---|---|
| 材料 | IN625 | Balbaa 明确 | 与论文一致 |
| Laser power | `195 W` | **历史/外部推断** | Balbaa 未写明验证图工况；因其 Exp[68] 数据条与 NIST CBM-B 完全相符，登记为 CBM-B |
| Scan speed | `0.8 m/s` | **历史/外部推断** | 同上，对应 NIST CBM-B `800 mm/s` |
| Absorptivity | `0.62` | Balbaa 明确 | 论文对 1070 nm IN625 粉末的测量值 |
| Beam radius | `50 µm` | **NIST/EOS 补足，非 Balbaa 明确值** | Balbaa Eq. 18 仅给符号 `r`；采用 NIST D4σ `100 µm` spot diameter 的半径 |
| Heat-source model | Eq. 18 exponential volumetric source | Balbaa 明确 | 实现名 `source_model=legacy` |
| Optical penetration depth | `100 µm` | Balbaa 明确/插值 | 论文由纯 Ni 数据向 IN625 PSD 插值得到 |
| Source cutoff | `0`（不截断） | **本次物理解释，非 ABAQUS 严格复刻** | 让能量按 Eq. 18 向下沉积，不锁在粉层内 |
| Cutoff renormalization | `false` | **本次物理解释** | 避免将完整吸收功率压缩至 `20 µm` 粉层 |
| Powder layer | `20 µm` | Balbaa 验证变体明确 | 用于与 NIST 验证图比较 |
| Support/substrate thickness | `280 µm` | Balbaa 几何推导 | 总高度 `300 µm` 减去粉层 `20 µm` |
| Domain | `1.0 × 0.48 × 0.30 mm` | **论文几何 + 对称解释** | 论文写 `1.0 × 0.24 × 0.30 mm` 并使用 XZ symmetry；本实现展开为全宽 `0.48 mm` |
| Scan path | `x=0.05–0.95 mm`，中心线 | **实现登记假设** | Balbaa 未公布轨迹端点；留 `50 µm` 边距 |
| Mesh | uniform `10 µm` C3D8，约 144k cells | 部分 Balbaa、部分保守解释 | 论文只明确粉层 minimum size `10 µm`，未公布基板 grading；本实现全域统一 `10 µm` |

### 3.2 材料和相变

| 参数 | 实际值 | 来源分类 | 说明 |
|---|---:|---|---|
| Solidus | `1563 K` | Balbaa 明确 | 同时作为熔池边界 |
| Liquidus | `1623 K` | Balbaa 明确 | — |
| Latent heat of fusion | `290000 J/kg` | Balbaa 明确（修正印刷单位） | 论文单位印为 `kJ/kg.K`，按量纲解释为 `kJ/kg` |
| Liquid specific heat | `709.25 J/(kg·K)` | Balbaa 明确 | — |
| Base liquid conductivity | `30.078 W/(m·K)` | Balbaa 明确 | Eq. 19 的 `k_l` |
| Effective liquid conductivity | `58.847216 W/(m·K)` | **Balbaa 方程 + 本次预计算假设** | 见第 4 节 |
| Solid density | `8453 kg/m³` | **Balbaa 范围端点/分相近似** | 论文给温变范围 `8453→7925`，本 solver 使用分相常值 |
| Liquid density | `7925 kg/m³` | Balbaa 范围端点 | — |
| Powder density | `5071.8 kg/m³` | Balbaa Eq. 15 | `0.6 × 8453`，porosity `0.4` |
| Powder specific heat | `0.6 × cp_s(T)` table | Balbaa 按印刷式复现 | Eq. 14 与 Eq. 15 同时使用会使体积热容出现 `(1−φ)^2`；保留为 code-to-code 假设 |
| Solid `k(T)`, `cp(T)` | 仓库重建表 | **不完全来源于 Balbaa** | 论文只给范围，完整曲线不可得；由图、范围和引用重建 |
| Powder `k(T)` | Balbaa Fig. 1 数字化表 | **图像数字化** | 论文印刷 Sih–Barlow 公式与自身 Fig. 1 不一致，采用其图示曲线 |
| Phase history | `paper_irreversible` | Balbaa 思路、实现不同 | Balbaa 使用 `USDFLD` binary state switch；本 solver 使用 mushy-band enthalpy interpolation 与不可逆历史 |

### 3.3 初始、边界、时间和数值设置

| 参数 | 实际值 | 来源分类 | 说明 |
|---|---:|---|---|
| Initial/preheat temperature | `353.15 K` (`80 °C`) | Balbaa 通用初值，**单道未重述** | 登记采用论文 Eq. 2 的 preheat |
| Bottom temperature | fixed `353.15 K` | **实现边界假设** | Balbaa 单道底边界细节未完整公开 |
| Ambient | `313 K` | Balbaa Table 1 的一种读法 | 正文另有 `300 K`，存在内部矛盾 |
| Emissivity | `0.5312` | **歧义参数** | 将 Table 1 的 `0.4` 解释为 bulk solid emissivity，再按 Eq. 9–11 得 powder-bed emissivity；另一合理读法是直接用 `0.4` |
| Convection coefficient | `20 W/(m²·K)` | **实现假设，未严格复现 Balbaa** | Balbaa 使用 N₂ `3 m/s` flat-plate correlation，未给完整气体性质；估算约 `12–13 W/(m²·K)` |
| Surface radiation request | enabled in command | 实现设置 | 但 `used_config.derived.front_surface_loss_enabled=false`；因此本次产物不能宣称严格执行了论文表面对流/辐射，需在后续边界专项中复核 |
| Time step | `10 µs` | **数值设置，非 Balbaa 明确值** | 实际 103 steps |
| Cooling | 30 steps × `10 µs` | **数值设置** | 仅用于当前提取窗口，并非完整室温冷却 |
| Quadrature order | `2` | **数值设置** | — |
| Thermal mass lumping | enabled | **求解器设置** | 与 ABAQUS 离散不完全相同 |
| Thermal output | every 2 steps | **输出设置** | 52 VTU frames |
| Mechanics | disabled (`mechanics_every=0`) | 本次范围 | 本报告只接受纯热结果 |
| Linear solver | PARDISO, CPU | 允许差异 | 用户已允许求解器不同 |

## 4. `keff` 的计算与非严格对齐项

本次按照 Balbaa Eq. 19–24 计算：

- `k_eff = k_l + hL`
- `Nu = hL/k_l`
- `Nu = 1.6129 ln(Ma) − 10.183`
- `Δt = L/v`

实际输入和结果：

| 项 | 值 |
|---|---:|
| 第一版全深度无 `keff` 熔池宽度 | `141.0190 µm` |
| Characteristic length `L = width/2` | `70.5095 µm` |
| Interaction time `L/v` | `88.1369 µs` |
| Marangoni number | `998.7413` |
| Nusselt number | `0.956487` |
| `h` | `408018.95 W/(m²·K)` |
| `keff` | `58.847216 W/(m·K)` |
| `keff/k_l` | `1.956487` |

必须注明以下差异：

1. Balbaa 没有公开由未知熔池宽度求 `L` 的迭代/更新算法；本次采用**一次 Picard 更新**（第一版宽度 → `L` → 第二版），没有迭代到完全自洽。
2. Balbaa 表述 Marangoni 增强只基于熔池宽度方向的 outward flow；当前实现把 `keff` 作为液相 conductivity table 的**标量增强**，方向性并不与 ABAQUS 实现严格相同。
3. `L` 来自本 solver 第一版输出，不是 Balbaa 公布的内部 ABAQUS 参数，也未针对 Balbaa/NIST 目标值拟合。

因此，第二版与 Balbaa 峰值温度的接近应视为模型行为的一致性证据，不应被描述为全参数严格复刻或校准成功。

## 5. 参数来源分级汇总

### A. Balbaa 论文明确或直接按论文方程采用

- IN625；Eq. 18 exponential source；absorptivity `0.62`；OPD `100 µm`。
- 验证模型 `20 µm` powder layer、总高度 `300 µm`、最小单元 `10 µm`。
- Solidus `1563 K`、liquidus `1623 K`、latent heat `290 kJ/kg`。
- `k_l=30.078 W/(m·K)`、`cp_l=709.25 J/(kg·K)`、`μ=0.007 Pa·s`、`dσ/dT=-1.1×10^-4 N/(m·K)`。
- Porosity `0.4` 及论文 powder-property 方程。
- 熔池边界 `1290 °C`；`keff` Eq. 19–24 的公式结构。

### B. 根据 NIST 数据、EOS 规格或历史结果推断，Balbaa 未严格给出

- `195 W / 800 mm/s`：由 Balbaa Exp[68] 柱状数据与 NIST CBM-B `133/91/780 µm` 完全匹配推断。
- Beam radius `50 µm`：由 NIST CBM 的 D4σ `100 µm` spot diameter 补足。
- Balbaa ABAQUS depth/length：由论文柱状图数字化，不是作者给出的原始数字表。
- NIST 数值：目前来自 Lane 2020 论文表格转录；原始 data.nist.gov 记录尚未再次逐项核验。
- 完整 `k_s(T)`、`cp_s(T)` 和 powder conductivity curve：论文没有提供可直接重建的完整数字表，使用范围、图像数字化及历史登记表。

### C. 当前实现/预登记假设，不应标成 Balbaa 原始设置

- Eq. 18 在完整 `100 µm` 深度作用且不做 cutoff renormalization。
- 全宽 `0.48 mm` 代替 ABAQUS half-domain symmetry。
- 全域 uniform `10 µm` mesh。
- `x=0.05–0.95 mm` 的扫描起止点。
- 固定底面 `353.15 K`。
- Emissivity `0.5312` 的歧义选择和 constant `h=20 W/(m²·K)`。
- Mushy-band enthalpy/interpolation 代替 ABAQUS `USDFLD` binary switch。
- `keff` 的一次 Picard 预计算及各向同性液相 conductivity table。
- `dt=10 µs`、mass lumping、quadrature order 2、PARDISO 和输出频率。
- 30 个冷却步不是 Balbaa 的完整冷却历程。

## 6. 完整性与审计

| 检查项 | 结果 |
|---|---:|
| Solver exit | `0` |
| Run sentinel | `DONE` |
| Thermal-only audit | 52/52 frames valid |
| Energy ledger | 103/103 steps recorded |
| Ledger complete | `true` |
| Maximum absolute balance error | `1.4058×10^-12 J` |
| Maximum relative balance error | `3.4642×10^-9` |
| Assembly identities within tolerance | `true` |
| Temperature invariants valid | `true` |

该账本只证明本 solver 离散弱形式中的能量装配与求解结果自洽，不证明热源/边界物理选择就是 Balbaa ABAQUS 的唯一正确解释。

## 7. 接受边界及后续使用规则

1. 本结果可作为后续 V1 纯热工作的当前基线，并可用于讨论熔池温度和几何趋势。
2. 不得将本报告表述为“除求解器外与 Balbaa 完全相同”。
3. 未经独立 thermal gate，不应直接把本结果扩展为 V2 多道/多层模型。
4. 在残余应力生产运行前，需确认纯热基线在目标 V2 几何和时间尺度仍成立。
5. 若后续要求严格 code-to-code ABAQUS parity，应另运行 `20 µm cutoff + no renormalization + keff` sensitivity arm，并复核 Balbaa `DFLUX/USDFLD` 的实际作用域与 `keff` 方向性。
6. `derived.front_surface_loss_enabled=false` 与命令中的 radiation flag 不一致，必须保留为已知检查项；在解决前不能声称表面对流/辐射边界与 Balbaa 严格一致。

## 8. 可追溯文件

本次生产目录内的关键证据：

- `solver_command.txt`：实际 solver 命令；末尾 override 最终将 cutoff 设为 `0` 且关闭 renormalization。
- `used_config.json`：合并并解析后的最终配置。
- `v1_meltpool_metrics.json` / 上层 `arm2_full_depth_keff_analysis.json`：结果指标。
- `v1_run_audit.json`：thermal-only 输出审计。
- `thermal_energy_ledger.jsonl`：逐步能量账本。
- `thermal_energy_ledger_summary.json`：账本汇总。
- 上层 `derived/keff.json`：Eq. 19–24 的输入、推导和 sensitivity brackets。
- 上层 `derived/k_liquid_keff.csv`：第二版实际液相 conductivity table。

参考登记文件：

- `inputs/balbaa-model.json`
- `inputs/nist-meltpool.json`
- `inputs/deviations.yaml`
- `RESULTS.md`
