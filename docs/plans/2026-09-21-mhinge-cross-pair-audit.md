# Mhinge 跨 pair 预冻结复现与一致性审查

## 研究问题与证据单位

本轮只检验：已经在 `5→6`、`27→28` 上发现的单转折 Mhinge 形式，能否在同一次采集中、
此前没有参与 Mhinge 模型选择的其他可靠 single-lag pair 上重复出现。

pair 是最小汇总单位，block 只用于 pair 内空间外推。由于所有 pair 来自同一次采集序列并
共享设备与背景估计，pair-level bootstrap 只作为描述性敏感性区间，不解释为来自独立重复
实验的总体置信区间。本轮审查模型形式的复现性，不宣称跨设备、跨会话或物理机制泛化。

## 完全冻结的输入

- 背景：`configs/background_frozen_v1.json` 中的 `masked_hybrid_spline`；
- 背景配置 SHA-256：`313ad0dafc48102da57131f77534117bba366fb8893eed6cf845be1d65e4e50e`；
- SAM archive SHA-256：`6a2f43410c4d92023ff7f5df72e01af31f2bb20d51feb67fabd30ad7c9658ab6`；
- 先验 cohort 审计：`outputs/background_freeze_audit_v1/background_freeze_audit.json`；
- cohort 审计 SHA-256：`21b544674cb8dba66ecc0df464fd0924fe968af8b72ee3786bbe101077c487f3`；
- block：16×16；
- `X`、`Y=D-BG`、source support、评价掩膜和四折空间划分均由新的 13-pair 正式导出读取；
- 模型只比较冻结 M1 与 Mhinge，不新增模型、转折点或组合项。

新的 13-pair 导出写入 `outputs/mhinge_cross_pair_v1/`，不得覆盖两组发现数据的
`outputs/pseudo_ghost_mechanism_v1/`，因为后者的 manifest 已被早期模型结果按哈希引用。

正式复现命令为：

```bash
python scripts/analyze_pseudo_ghost_mechanism.py \
  --output-dir outputs/mhinge_cross_pair_v1 \
  --pairs single-lag \
  --analysis-blocks 16 \
  --background-modes masked_hybrid_spline \
  --primary-background masked_hybrid_spline \
  --source-mask-npz outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz \
  --skip-background-oof \
  --skip-pair-figures \
  --frozen-background-config configs/background_frozen_v1.json

python scripts/audit_mhinge_cross_pair.py
```

## 预先固定的队列

先验 nominal frozen-BG gate 在 13 个 single-lag pair 中通过 11 个；再沿用 BG 审计已经使用
的 `|alpha| >= 1e-4` 实用效应门槛，得到 9 个 eligible pair。

- 发现集，仅作参照：`5→6`、`27→28`；
- 主要扩展集：`1→2`、`3→4`、`7→8`、`8→9`、`9→10`、`22→23`、`23→24`；
- 弱信号参考：`2→3`、`25→26`；
- 未检出对照：`4→5`、`6→7`。

主要跨-pair结论只使用 7 个扩展 pair。发现集与扩展集合并的 9-pair 结果只作次要描述。
早期的 `11→12`、`19→20`、`28→29` 不进入本轮，因为其目标 dark 位于既有多帧记忆
阳性范围，不能作为冻结 single-lag 复现队列。

## 冻结模型与拟合

令 `S=max(-X,0)`，使用连续单转折模型：

```text
Y = b + alpha_low * X - delta_alpha * max(0, S - tau_S)
alpha_high = alpha_low + delta_alpha
```

`tau_S` 始终为正；如需 X 坐标只另行报告 `tau_X=-tau_S`。候选训练分位数固定为
`0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85`。每个 pair、每个外层空间折只在其余
三折中通过内层空间 CV 选择分位数，再在外层训练区拟合线性系数。外层测试 Y 不参与
修剪、阈值选择或参数拟合。

## pair 级预设检查

每个 pair 固定报告：`delta CV-R2`、RMSE 改善、残差趋势加权 RMS 比例、source-edge 与
outside-source RMSE 相对变化、`tau_S`、`tau_X`、tau 所处分位数、`tau_S/S90`、
`alpha_low`、`alpha_high`、`delta_alpha` 和方向。

同时只作上下文记录 source light 的 kV、mA、曝光毫秒、mAs、source-support 的
S-P05/P50/P90/P95 与饱和 block 比例；这些字段不参与模型选择或通过判定。

一个 pair 只有同时满足下列条件才通过：

1. `delta CV-R2 > 0` 且 RMSE 下降；
2. 残差趋势加权 RMS 不高于 M1 的 50%；
3. source-edge 与 outside-source RMSE 各自恶化不超过 5%；
4. `tau_S` 四折 CV 不超过 0.50，候选网格边界命中不超过 1 折；
5. 四折中 `0 < alpha_low <= 0.02` 且 `0 < alpha_high <= 0.02`；
6. `alpha_low`、`alpha_high` 四折 CV 均不超过 0.50，四折 `delta_alpha` 方向一致；
7. 每个外层训练折在转折两侧均至少有 50 个 source-support block 且各占至少 5%；
8. 训练折 source-support 的 P05–P95 动态范围中，tau 两侧各占至少 5%；
9. 标准化 Mhinge 设计矩阵条件数不超过 100。

第 7–9 项防止稳定 tau 由窄平台、少数高杠杆 block 或共线设计产生。门槛在新增 7 个
扩展 pair 的 Mhinge 结果产生前固定，不因结果接近边界而调整。

## pair 级汇总与敏感性

扩展集等权汇总：

- 通过全部检查的 `K/7`；
- median `delta CV-R2`；
- 10,000 次、固定种子 20260921 的 pair bootstrap median 区间；
- `delta_alpha` 上升/下降/不稳定的 pair 数；
- 对 7 个扩展 pair 逐一执行 leave-one-pair-out，报告剩余 median `delta CV-R2` 与通过数；
- 单列删除最大 `delta CV-R2` pair 后的结果。

跨-pair一致性检查预设为：至少 5/7 pair 通过、median `delta CV-R2 > 0`、bootstrap
区间下限大于 0、所有 leave-one-pair-out median 均大于 0，并且至少 6/7 pair 的
`delta_alpha` 方向一致。即使全部通过，也只称为同一采集内的预冻结复现证据，不自动把
Mhinge 命名为已验证的物理 M2。
