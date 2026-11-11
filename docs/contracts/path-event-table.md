# 事件表合同：`--path-file` CSV（草案 v0.1，2026-09-04）

读取实现：`jax_fem_am/process/scan_path.py::generate_path_file_step_states`。
本文件把 V2 立方体复现时踩出的约束固定下来。机器可读版本见 `path_event_table.schema.json`（行级），
文件级约束（单调时间等）只能由校验器检查（AM-Sim 仓库 `contracts/`）。

## 列定义

| 列 | 类型 | 必需 | 语义 |
|---|---|---|---|
| time | float | 是 | 该行结束时刻（s）。严格递增。第 i 行的 dt = t_i − t_{i−1}；首行 dt 取 `--dt` |
| x, y, z | float | 是 | 激光中心，**网格坐标系**（零件已居中在基板上），单位与网格一致；求解器再乘 `--path-length-scale`（缺省继承 `--mesh-length-scale`） |
| power | float | 是 | 本行激光功率，与 `--laser-power` 同语义（求解器把它当作已含吸收率的功率使用）。待确认：是否在合同里明确"命令功率 × 吸收率" |
| laser_on | 0/1 | 是 | 开关；0 表示跳转、驻留、铺粉等无沉积行 |
| layer | int | 是 | **1 基**的数值层（slab）编号；`layer > --layers` 的行被丢弃 |
| hatch | int | 是 | 1 基的 hatch 线编号 |
| mode | str | 是 | 标签（scan / jump / recoat / dwell 等）；空串按 "path" 处理，仅用于日志与统计 |
| front_coord | float | 否 | 激活前沿在建造轴上的坐标（网格单位）。缺省用激光中心的建造轴坐标并打印 WARNING。V2 约定：沉积 z = slab 顶 |
| scan_id | int | 否 | 矢量编号；缺省用行号 |

## 文件级约束

- 时间严格递增；写 15 位有效数字（2e4 s 时钟下行差舍入约 1e-8 s，恒等式容差取 1e-6 s）。
- 层间铺粉：要么显式写 recoat 行并传 `--recoat-time 0`（V2 做法），要么让求解器按 `--recoat-time / --recoat-steps` 自动插入。两者不能同时用。
- 末尾冷却由 `--cooling-steps` 与 `--cooling-dt` 追加，不写进 CSV。
- 激活模式：`--layer-activation-mode layer_on_scan` + `--layer-activation-geometry centroid`（`intersection` 会把贴面的下一 slab 一起激活）。
- 粉末床网格必传 `--powder-elset`，否则宏观模式会把整层粉末熔成实体。

## 已知缺陷（与本合同相关）

- 扫描相 `--dt` 被行距覆盖；dt 收敛研究必须改路径离散。
- 集总质量下 HEX8 热源在顶点采样，粗网格高估沉积（V2 合同用切带 199 µm + d=125.935 µm 使离散带积分为 1）。

## 主线 B 计划扩展（未实现）

- 线段行：增加 `x_end, y_end, z_end` 或 `segment_id`，一行沉积一条矢量的能量（扫掠线段热源）。
- 分组行：`group_id`，一步沉积一组矢量（层级集总）。
- 校验器输出：总能量、每层能量、每层扫描时长，与预检能量台账对账。
