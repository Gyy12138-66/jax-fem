# V1-P / V1-E 重跑计划（预注册骨架 v0.1 — 未冻结）

状态：**DRAFT**。生产运行前本文件必须冻结（含所有 `TBD`），冻结后任何口径不得再改。
背景：2026-08-21 审查草案 P0-1/P0-2/P0-3/P0-4/P0-5；WP1 求解器修复
（`--source-depth-cutoff` / `--source-cutoff-renormalize` /
`--fixture-thermal-phase`）已交付于分支 `agent/fable5/wp1-solver-fixes`。

## 目标拆分（P0-1 修正）

- **V1-P**（code-to-code 诊断）：严格复现 Balbaa 2022 验证变体（20 µm 粉层
  + 280 µm 基板 + 353.15 K 底面），只对 Balbaa 已发表数值说话，不再冒充
  NIST 裸板验证。
- **V1-E**（code-to-experiment）：NIST AMB2018-02 裸板工况重建，对 Lane
  2020 实测说话。评分必须经 NIST 合成观测算子（见依赖项），在算子落地前
  V1-E 的一切输出仅为诊断。

## 运行矩阵

### 战役 1：V1-P 源实现歧义臂（`model/runs/run_v1p_srcband_arms.sh`）

| 臂 | 含义 | 旗标 |
|---|---|---|
| p_legacy | 半空间指数源（历史基线，同 harness 重跑） | — |
| p_band_renorm | 沉积限于 20 µm 粉层，带内重归一化（论文读法 A） | `--source-depth-cutoff 2.0e-5 --source-cutoff-renormalize` |
| p_band_trunc | 沉积限于 20 µm 粉层，尾部能量不吸收（下界臂） | `--source-depth-cutoff 2.0e-5` |
| p_fixphase | 半空间源 + fixture 热属性随温度（单独隔离 P0-3） | `--fixture-thermal-phase follow-temperature` |

- QoI：宽/深/长（analyze_v1.py 全口径）、峰温、探针冷却率、账本捕获分数。
- 判读规则（冻结前 TBD）：与 Balbaa 发表值（B 工况 宽 131 / 深 44 /
  长 378 µm）的接近度**只用于歧义臂之间的相对判读**，不得回调任何参数。
  预期：若 p_band_* 显著移向 Balbaa 值，则支持"P0-2 是深度虚合的成因"。
- 预算：4 臂 × 基线单跑成本（box-159 上 CBM-B 级别，小时级）。

### 战役 2：V1-P 网格三连（`model/runs/run_v1p_mesh3.sh`）

- 20 / 10 / 5 µm 均匀六面体；路径行距 12.5 µm 与 dt 固定（隔离空间离散）。
- 承载臂（TBD：冻结时从战役 1 选定，默认基线臂）。
- 收敛门（TBD）：每个 QoI 报告 20→10、10→5 变化率；建议门为 10→5 变化
  < 实验 U(k=2) 的 50%（G2 口径），达不到则数值不确定度并入 U_val 降级申明。
- 预算：5 µm 臂 1.15 M 单元，为长腿（预计基线的 ~8×）。
- 后续独立战役（本文件不覆盖，冻结时排期）：路径步长三连、域尺寸阶梯、
  求解容差扫描（P0-5 完整矩阵）。

### 战役 3：V1-E 裸板（`model/runs/run_v1e_bareplate_draft.sh`）

fail-closed 输入（脚本拒跑直到冻结提供）：

- `ABSORPTIVITY`：裸板吸收率。**禁止沿用 0.62（粉末 DRS）**。来源候选
  （TBD，须带出处登记）：文献裸板 IN625 @1070 nm 实测区间，以区间双臂跑。
- `SOURCE_DEPTH`：致密金属源深尺度。**禁止沿用 100 µm 粉末 OPD**。候选
  （TBD）：表面/浅层沉积读法（如 1–10 µm 带 + `--source-depth-cutoff`），
  以区间双臂跑。

DRAFT 默认（冻结时确认或改为扫描）：域深 0.60 mm（阶梯 0.3/0.6/1.2 验证
不敏感）、环境/初温 293.15 K（D-V1-11 读法；Lane 未给出，保留区间）、底面
对流 BC（非定温）、发射率 0.4（固体面）、粉层材料塌缩为固体（D-V1-07）。

## 依赖项（冻结前必须闭合）

1. **NIST 合成观测算子**（P0-4）：顶面中心线长度、横截面宽/深、单帧空间
   梯度冷却率——实现前 V1-E 不出正式对比数。参考 IET-20 三路读数的先例。
2. **M31931 数据集受控下载归档**（WP0；注意汇总 XLSX 的 A/C 功率速度互换
   缺陷，导入只用逐轨迹行）。
3. Lane 宽/深 U(k=2) 补录进 `inputs/nist-meltpool.json`。
4. 本文件全部 TBD 落定 + 双人复核（Fable5 起草 / 复核人 TBD）。

## 纪律

- 零标定：任何旗标、阈值、材料值不得向 Lane 实测或 Balbaa 发表值回调；
  歧义只以预登记区间臂覆盖。
- 冻结顺序：先冻结本文件 → 战役 1/2（V1-P，不依赖观测算子）→ 观测算子
  落地并回归 → 战役 3（V1-E）。
- 所有运行走 WP1 分支（`agent/fable5/wp1-solver-fixes`）或其合并后的主
  干；运行目录保留 `solver_command.txt` 与协议输入回执。
- 与 IET-22 的关系：本计划的一切运行不得触碰热闸门分支的 checkout 与
  box-159 的 C 包窗口；排程以 C 包优先（2026-08-21 chat 决策）。
