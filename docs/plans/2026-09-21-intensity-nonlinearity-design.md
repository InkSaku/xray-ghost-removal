# 冻结背景下的强度非线性诊断设计

## 研究边界

本轮只回答：在冻结的 `masked_hybrid_spline` 背景、`X`、可观测
`Y = raw_dark - background` 和四折空间划分不变时，低复杂度非线性模型能否改善
`E[Y | X]` 的空间外推表现。

本轮不把 `X` 称为入射曝光量，不把 `Y` 称为真实残影标签，也不把候选模型解释为已经
确认的电荷俘获机制。当前 DICOM 的 `Modality` 为 `CR`，厂商和探测器型号未知；a-Si
平板文献只用于提出可检验的函数形状。真正的“alpha 是否随剂量变化”仍需独立曝光和
`clean / previous / ghosted` 三元组确认。

## 预先固定的模型

- `M1`：沿用冻结的四折 OOF 预测，`Y = alpha X + b`。
- `Mquad`：沿用训练折标准化的二次诊断，仍为 `diagnostic_only`。
- `Msat`（主候选）：令 `S = max(-X, 0)`，拟合
  `Y = b + X[alpha0 + alpha1(1 - exp(-S/S0))]`。正 `X` 区域自动退回常数
  `alpha0`，避免把负 `S` 输入指数。
- `Mhinge`（替代候选）：令 `S = max(-X, 0)`，拟合
  `Y = b + aX - c max(0, S - tau)`。
- `Mspline`：固定四个训练折分位数内结点的低自由度三次 B 样条，仅用于形状诊断，
  不参与候选晋升。

`Msat` 的 `S0` 和 `Mhinge` 的 `tau` 都以训练区 source-support 中正 `S` 的候选
分位数表示。每个外层折只用其余三折执行内层空间交叉验证来选择分位数，再把所选分位数
映射为该外层训练区中的 DICOM 强度参数。外层测试区不参与修剪、尺度、结点、阈值或
饱和参数选择。

## 预先固定的证据检查

每个 pair 分开报告：

1. OOF CV-R2 必须高于 M1；
2. 残差对 `X` 的 24 分位箱均值加权 RMS 必须不高于 M1 的 50%；
3. source-edge RMSE 相对 M1 的恶化不得超过 5%；
4. outside-source RMSE 相对 M1 的恶化不得超过 5%；
5. 四个外层折的 `S0` 或 `tau` 变异系数不得超过 0.50，且候选网格边界命中不得超过
   一个外层折。

只有两个冻结 pair 都通过，才记录为“跨 pair 一致的探索性证据”。无论结果如何，本轮
都不直接命名 M2；如果需要选择模型类别，必须先冻结选择，再用新采集独立确认。

## 输出保留

结果稳定覆盖写入：

```text
outputs/pseudo_ghost_mechanism_v1/intensity_nonlinearity/
├── model_results.json
└── nonlinearity_diagnostics.png
```

旧的 `model_comparison/` 必须保留，因为它是 Mblur 失败与 Mquad 预设 50% 门槛的冻结
审计记录；本轮改变了科学问题和候选集合，不能用新结果覆盖旧证据边界。
