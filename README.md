# CR X 射线残影伪影去除

去除连续计算机放射成像（Computed Radiography，CR）图像中的残影（残留/余辉）伪影。
当擦除周期不完整时，计算机放射成像板会保留上一次曝光留下的微弱潜像，因此图像 `t` 中会带有图像 `t-1` 的一个缩放副本。本项目包含针对该问题的代码、设计文档以及当前研究结果。

## 当前状态 — 2026-09-28

2026-09-28 补做 `3→4` 与 `5→6` 的双向 M1 迁移：直接迁移能预测大部分冻结 `Y` 的
空间变化，但目标图 12.5% 区域校准后误差进一步下降；详情与证据边界见
`RESULTS.md` §4.17。这是同会话伪标签的探索性比较，不是独立采集的真实去除验证。

30 帧连续序列上的线性残影现象仍然成立，但现有去除器只在强残影样本上显示出明确改善，缺少 ground truth 的问题尚未解决。

62 张 dark/light 图像的前三阶段稳健性分析已经完成。拍摄顺序支持把 `light_(N-1) → dark_N` 作为因果候选配对，但 30 组中只有 5 组在 24 种分析设置下得到稳健确认，21 组对基线、饱和掩膜或分块大小敏感，4 组未检出。这批 dark 图像不能整批作为 clean reference，也不能把“未检出”直接解释为“无残影”。项目后续研究范围已固定为紧邻前一帧造成的单帧残影；早期多阶结果只作为历史记录，不继续建模或扩展。

正式 pseudo-ghost 机制实验不使用 `dark ≈ ghost` 的简化，而是定义 `Y_t = D_t - estimated_background_t`，并对 `5→6`、`27→28` 生成严格四折空间 OOF 预测。冻结 `masked_hybrid_spline` BG 和 16×16 block-mean 网格下，M1 的 OOF CV-R² 分别为 0.9603 和 0.9228；但 residual 邻域相关仍高达 0.858 和 0.963，因此剩余部分明显不是随机噪声。

背景估计使用 `masked_hybrid_spline`：它用前序 light 物体掩膜排除可能的残影污染，分解每帧平滑漂移，再用加权 Huber 均值估计固定图样 `F`，不再对 dark stack 逐位置取中位数。2026-09-21 的参数与固定图样覆盖审查支持冻结 SAM-based BG。2026-09-22 进一步对每个空间测试折重新估计 BG：核心 pair 的 alpha 最大变化 0.42%，M0/M1/Mhinge 的预测排名在 13/13 pair 中不变。不过完整真实形状的 169 次已知空气区尝试只有 58 次能放下，且没有一次位于原探测器位置；占图 19.94%–57.83% 的大物体仍未验证。因此冻结方案适合作为当前比较基线，不能当作真实 BG ground truth。

同日完成的强度非线性诊断没有产生正式 M2：预设主候选 Msat 在两个 pair 都未通过且 `S0` 四折全部命中搜索上边界。随后的 Mhinge 跨 pair 预冻结复现审查也未通过：7 个未参与候选提出的 extension pair 中 0 个同时通过预测和可辨识性门槛。其 `delta CV-R2` 中位数为 +0.00293，pair-bootstrap 95% 描述性区间为 `[-0.00012, 0.03316]`；斜率方向也未达 6/7 一致性门槛。因此当前证据只说明部分 pair 的冻结 `E[Y|X]` 有非线性迹象，不支持统一的 Mhinge 规律，更不能写成 alpha 已被证明随入射曝光变化。

M1 后的首轮候选机制比较已完成。Mblur 通过嵌套空间交叉验证在四个外层折中均选择 `σ=0` block，因此当前数据不支持高斯空间扩散作为 M2。只用于诊断的 Mquad 将 `27→28` 的 OOF CV-R² 从 0.9228 提高到 0.9647、RMSE 从 4.217 降到 2.851，但预设的两个残差趋势门槛只通过一个，所以尚未晋升为正式 M2。未拟合组合模型。

单帧曝光关联分析已在冻结的 13-pair 队列上完成。mAs 与 alpha 的 Spearman `rho=0.108`，与 OOF 预测残影对比度的 `rho=0.283`，且逐一删除样本后的方向或幅度不稳定。4 组完全相同的曝光设置中，3 组的 alpha 最大/最小比达到 32.3、50.1 和 33.7。固定 3×3 探测器区域的诊断中，10/10 个支撑充足区域保持相同强弱顺序，但严格满足局部预测可靠性的只有两组中的 4 个区域，4/4 保持顺序；第三组没有可靠共同区域。最清楚的 `1→2 / 2→3` 重复组具有几乎相同的 source 强度、面积和位置，两个可靠共同区域的 alpha 仍相差 28.5 和 33.7 倍。结果更像未记录的板/擦除/读取全局状态，但 DICOM 中缺少验证该解释的技术字段。因此当前数据不支持仅用 kV、mA、曝光时长或 mAs 预测单帧残影。全部 62 张 DICOM 均缺少 `AcquisitionTime`，所以仍不能估计余辉半衰期或真实时间衰减。最终定量验证需要按采集协议获得真实 `clean / previous / ghosted` 三元组。引用数字前请先阅读 `RESULTS.md`。

---

## 模型

观测图像是真实曝光图像与前序图像缩放副本的线性叠加。

```
I_t(x) = S_t(x) + alpha * (I_{t-1}(x) - bg_{t-1})
```

`alpha` 是残影耦合系数。在 30 帧初步序列中，对于 11 对存在可测残影的相邻图像，`alpha` 的中位数为 0.0040，在残影最强的图像中升至 0.0095。因此，残影所携带的信号远低于上一张图像信号的 1%，这使它接近逐像素噪声底，因而“如何将残影与噪声分离”成为本项目的核心困难。经过空间平均后，残影十分明显，分块 R² 最高可达到 0.584；但在逐像素层面并非如此，R² 约为 0.013。完整数据表见 `RESULTS.md`。

---

## 已实现内容

| 模块 | 方法 | 状态 |
|---|---|---|
| `src/models/physics_model.py` | 对受残影污染的空气区域进行扩散修补 | 可运行，能够避免光晕，但无法恢复物体覆盖区域下方的结构 |
| `src/models/linear_ghost.py` | 使用最小二乘估计 `alpha`，并加入置信度门控 | 逻辑上最安全，宁可跳过也不过度校正，但目前没有脚本在启用门控的情况下运行它 |
| `src/models/seamless_ghost.py` | 保留噪声纹理的低频替换 | `run_ghost_removal.py` 实际调用的方法，可见修正痕迹最小，但不同帧之间的效果差异很大 |
| `src/models/unet.py` | 在合成残影图像对上训练的 U-Net | 可以训练，但无法迁移到真实残影，详见 `RESULTS.md` |
| `src/data/synthetic.py` | 生成合成的（带残影、干净、上一帧、残影）图像对 | 可以工作，但合成残影模型与真实情况并不匹配 |
| `src/utils/dicom_utils.py` | 在保留头信息的情况下加载和保存 DICOM | 可运行 |
| `scripts/analyze_dark_light_pairs.py` | 对 62 张 dark/light 图像执行证据边界、单阶稳健性和多阶记忆分析 | 已实现并完成一次全量运行；默认只写一个机器可读 JSON，不生成逐对图片或 CSV |
| `scripts/analyze_pseudo_ghost_mechanism.py` | 对可观测残影信号生成严格空间 OOF 预测、null 对照、背景重建验证和污染感知 `F` 估计 | 已完成 13 组 single-lag 背景比较；不把 dark 当作 clean ground truth |
| `scripts/analyze_intensity_nonlinearity.py` | 在冻结的 `X/Y/BG/fold` 上比较 M1、Mquad、Msat、Mhinge 与低自由度样条 | 探索性条件均值诊断；不把 `X` 当入射剂量，不自动晋升 M2 |
| `scripts/audit_mhinge_cross_pair.py` | 在 13 个预声明 pair 上独立比较 M1/Mhinge，审查 `tau`、斜率、动态范围和 pair-level 稳健性 | 7 个 extension pair 中 0 个通过全部门槛；Mhinge 不晋升 M2 |
| `scripts/analyze_single_frame_exposure.py` | 在冻结 13-pair 输入上比较单帧 alpha、OOF 残影对比度与曝光参数，并审查完全相同设置的重复样本 | 曝光参数单独解释不了 pair 间差异；稳定覆盖一个 JSON 和一张汇总图 |
| `scripts/audit_background_freeze.py` | 固定 13 组队列的 BG 参数敏感性、pair 级伪遮挡 bootstrap 和 source-aligned residual 审计 | 9 组正式扰动全部通过，SAM-based BG v1 已冻结 |
| `scripts/audit_background_crossfit.py` | 用完整真实物体形状审查 BG 插补，并对每个空间测试折重算 BG 后复核 alpha 与 M0/M1/Mhinge 排名 | 核心 alpha 与模型排名稳定；大物体和原位置 BG 仍未验证 |
| `scripts/audit_fixed_pattern_identifiability.py` | 审查冻结 `F` 的空间覆盖、有效样本数和低覆盖排除敏感性 | 空间覆盖与低覆盖排除审查通过 |
| `scripts/analyze_frozen_candidates.py` | 只读取冻结 NPZ，以嵌套空间 CV 比较 Mblur，并执行 OOF 二次诊断 | Mblur 未获支持；Mquad 仅保留为诊断证据 |

---

## 目录结构

```
.
├── README.md      # 本文件
├── DATA.md        # DICOM 数据应该放在哪里，运行任何程序前先阅读本文件
├── RESULTS.md     # 已经确认的结论及对应数字
├── requirements.txt
├── configs/       # 已冻结的可复现分析配置
├── docs/plans/    # 设计文档与配对数据采集协议
├── outputs/       # 本地分析产物，不应提交 DICOM 或临时诊断文件
├── src/           # 库代码，不包含运行脚本
├── scripts/       # 运行入口，每个脚本都可以独立执行
└── tests/         # 冒烟测试，无需 DICOM 数据即可验证安装
```

---

## 环境配置

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**推荐使用 Python 3.12。** 这是当前实际验证 dark/light 分析与 smoke test 的版本；不要把旧记录中对 Python 3.13/3.14 或 PyTorch wheel 可用性的描述视为当前保证。三个传统去除器和分析流程主要依赖 NumPy、SciPy、pydicom、openpyxl 与 matplotlib；U-Net 路线另外需要与运行平台匹配的 PyTorch。如果要在远程机器上使用 GPU 训练，`scripts/setup_remote.sh` 会基于 Python 3.12 和 CUDA 12.4 wheel 创建 conda 环境。

已于 2026-09-21 在 Python 3.12.7 上验证 dark/light、pseudo-ghost、BG freeze audit、空间可识别性审查、冻结候选机制比较和 Mhinge 跨 pair 审查可以完整运行。最新测试数以下方的完整 `pytest` 命令为准。

在处理数据之前，先验证安装是否正确。

```bash
python -m pytest tests/ -v
```

---

## 运行方法

首先按照 `DATA.md` 的说明放置数据。随后从仓库根目录运行所有命令，因为每个脚本都会相对于当前工作目录解析路径。

```bash
# 主去除器。尽管命名如此，它实际运行的是 SEAMLESS 方法，
# 而不是带门控的线性方法，详见 RESULTS.md 第 2 节。
# 会输出清理后的 DICOM、对比图以及逐图像 JSON 日志。
python scripts/run_ghost_removal.py
python scripts/run_ghost_removal.py --indices 4,6,18

# 在真实序列上量化线性模型。
# 输出每对图像的 alpha、R^2 和抑制率，并写入 figures/final/fig_alpha_fit.png。
python scripts/analyze_linear_fit.py

# 对 62 张 dark/light 图像复现阶段 0-2 的稳健性分析。
# 默认只写 outputs/dark_light_analysis_20260913/robustness_results.json；
# 不使用 InstanceCreationTime，不生成逐对图片或 CSV。
python scripts/analyze_dark_light_pairs.py

# 正式 pseudo-ghost 建模输入。在稳定路径上覆盖旧版 5→6 / 27→28
# block16 产物；配置、SAM 哈希或 BG 参数不符时会失败即停。
python scripts/analyze_pseudo_ghost_mechanism.py \
  --analysis-blocks 16 \
  --background-modes masked_hybrid_spline \
  --primary-background masked_hybrid_spline \
  --source-mask-npz outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz \
  --skip-background-oof \
  --frozen-background-config configs/background_frozen_v1.json

# 只读取上述冻结 NPZ，覆盖更新候选机制 JSON 和唯一诊断图。
python scripts/analyze_frozen_candidates.py

# 生成新的 13-pair 冻结输入后，执行预冻结的 M1/Mhinge 复现审查。
# 完整命令与判据见 docs/plans/2026-09-21-mhinge-cross-pair-audit.md。
python scripts/audit_mhinge_cross_pair.py

# 只读取同一 13-pair 冻结输入，分析单帧残影与曝光参数的关联。
# 重跑时原地覆盖一个 JSON 和一张图，不创建新的轮次目录。
python scripts/analyze_single_frame_exposure.py

# 在冻结的 3→4、5→6 block16 数据上做双向 M1 迁移和少量目标校准。
# 稳定覆盖 outputs/pair_transfer_v1/ 中的一个 JSON 和一张诊断图。
python scripts/analyze_pair_transfer.py

# 复核冻结固定图样 F 的原始覆盖、最终权重有效覆盖，
# 以及排除低覆盖 block 后线性结论是否稳定。
python scripts/audit_fixed_pattern_identifiability.py

# 用完整真实物体形状做已知空气区遮挡，并逐空间折重算 BG；
# 中间 NPZ 写入系统临时目录，最终只覆盖现有 BG 审计目录中的两个固定文件。
python scripts/audit_background_crossfit.py

# 在整个序列上运行基于扩散修补的基线方法。
python scripts/run_physics_baseline.py --n-previous 5

# 扫描窗位/窗宽，用于观察残影在哪里出现以及是否被去除。
python scripts/generate_comparisons.py --indices 4,6,9,19,30

# U-Net 训练。除冒烟测试外，实际训练需要 GPU。
python scripts/train_unet.py --epochs 100 --batch-size 16 --device cuda
python scripts/train_unet.py --epochs 10 --samples-per-epoch 500   # 快速检查

# 诊断脚本。用于区分训练失败与“合成数据到真实数据”的迁移失败。
python scripts/diagnose_model.py --device cuda
```

`scripts/generate_proposal_figures.py` 会为项目申请材料生成 6 张图。图 1 和图 2 来自真实数据与去除代码；图 3 至图 6 是计划工作的示意流程图，并不是实验结果。

---

## 如果你刚接手这个项目，请从这里开始

1. 阅读 `DATA.md`，并按照其中说明放置 DICOM 文件。
2. 运行 `python -m pytest tests/ -v`，然后运行 `python scripts/run_ghost_removal.py --indices 4`。
   第 4 张图具有最强且最干净的残影，是参考样例。预期可以看到明显改善；按照当前指标，其相干残影抑制率约为 40%。旧文档使用不同指标时曾给出该图 97% 的数字，在引用其中任意一个数字之前，请先阅读 `RESULTS.md` 第 2 节。
3. 阅读 `RESULTS.md`，了解哪些结论已经确定，哪些仍未确定。
4. 如本地存在 62 张 dark/light 数据，运行 `python scripts/analyze_dark_light_pairs.py`，并对照 `RESULTS.md` 第 4 节核对 5/21/4 和 17/30 的汇总结果。
5. 使用上述“正式 pseudo-ghost 建模输入”命令，再运行 `python scripts/analyze_frozen_candidates.py`。BG 参数稳定性和 `F` 空间可识别性均已通过审查；当前 Mblur、Mquad 和 Mhinge 都未晋升为正式 M2，不再调整 BG 或事后放宽非线性门槛。
6. 阅读 `docs/plans/2026-05-28-real-data-acquisition-protocol.md`，准备真实三元组采集。后者需要扫描仪使用时间，仍是验证实际去除质量的必要条件。

---

## 已知缺口

- 大多数图像没有真实标签，因此无法用 PSNR 或 SSIM 评估去除质量。目前的验证方式是人工视觉判断加“相干残影抑制”指标。
- `src/evaluation/` 和 `src/training/` 是空包。指标计算与训练循环都直接写在 `scripts/train_unet.py` 中。
- 超参数通过命令行参数设置，而不是 YAML 配置文件。
- dark/light 分析只验证了单帧候选关系中的空间相关，没有证明任何 dark 图像是 clean reference，也没有产生监督训练标签。
- 62 张 dark/light DICOM 没有 `AcquisitionTime`；`InstanceCreationTime` 不足以支持物理时间衰减建模。现有曝光关联是同一会话内的描述性结果，不能建立剂量定律。
- 完整物体 BG 验证只能在已知空气区进行；现有数据无法在真实物体原位置给出 clean BG，也无法安放占图约 20%–58% 的大物体形状。
- `outputs/dark_light_analysis_20260913/` 保存本地可复现结果，并由 `.gitignore` 排除。不要用强制添加把工作簿、JSON 或临时诊断产物提交到 Git。
- dark/light 默认输出目录沿用 `20260913` 这一历史名称；真实运行时间以 JSON 的 `generated_at` 为准，不要从目录名推断。
- `RESULTS.md` 第 1 节和第 2 节记录了两处命名/标签不一致问题。它们都不会改变任何结果，但会误导代码阅读者。
- 仓库中没有包含已训练的 U-Net checkpoint。由于基于合成数据训练的模型无法迁移到真实数据，因此后续计划是在真实配对数据上重新训练。
