# CR X 射线残影伪影去除

去除连续计算机放射成像（Computed Radiography，CR）图像中的残影（残留/余辉）伪影。
当擦除周期不完整时，计算机放射成像板会保留上一次曝光留下的微弱潜像，因此图像 `t` 中会带有图像 `t-1` 的一个缩放副本。本项目包含针对该问题的代码、设计文档以及当前研究结果。

## 当前状态 — 2026-09-16

30 帧连续序列上的线性残影现象仍然成立，但现有去除器只在强残影样本上显示出明确改善，缺少 ground truth 的问题尚未解决。

62 张 dark/light 图像的前三阶段稳健性分析已经完成。拍摄顺序支持把 `light_(N-1) → dark_N` 作为因果候选配对，但 30 组中只有 5 组在 24 种分析设置下得到稳健确认，21 组对基线、饱和掩膜或分块大小敏感，4 组未检出；另有 17/30 张 dark 图像支持多帧记忆效应。因此，这批 dark 图像不能整批作为 clean reference，也不能把“未检出”直接解释为“无残影”。

第一轮 pseudo-ghost 机制实验也已完成。实验不使用 `dark ≈ ghost` 的简化，而是定义 `Y_t = D_t - median(D_t) - B_(-t)`，并对 `5→6`、`27→28` 生成严格四折空间 OOF 预测。16×16 分块下两组 OOF CV-R² 分别为 0.432 和 0.778，且在 8/16/32/64 四种尺度上均超过对应 null P95；不过 residual 仍有很强的空间结构，因此当前只能确认单帧 affine 模型解释了前序图像相关成分，不能声称剩余部分是随机噪声。

背景估计已新增 `masked_hybrid_spline`：它用前序 light 物体掩膜排除可能的残影污染，分解每帧平滑漂移，再用加权 Huber 均值估计固定图样 `F`，不再对 dark stack 逐位置取中位数。在预先固定的 13 组队列上，其中位 OOF CV-R² 从旧 hybrid 的 0.492 提高到 0.722，116 个伪遮挡样本的中位重建 MAE 从 1.282 降到 0.824。该方法仍受限于无 clean ground truth 和少数位置的低覆盖，目前是候选背景模型，不是已验证的最终去残影器。

全部 62 张 DICOM 均缺少 `AcquisitionTime`。`InstanceCreationTime` 只用于审计，未进入配对、拟合、标签或物理解释，所以当前结果不能估计余辉半衰期或真实时间衰减。当前最高价值的代码工作是只基于已记录的 kV、mA、曝光时长和 mAs 做曝光关联分析；最终定量验证和监督训练仍需要按采集协议获得真实 `clean / previous / ghosted` 三元组。引用数字前请先阅读 `RESULTS.md`。

---

## 模型

观测图像是真实曝光图像与前序图像缩放副本的线性叠加。

```
I_t(x) = S_t(x) + sum_k alpha_k * (I_{t-k}(x) - bg_{t-k})
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

---

## 目录结构

```
.
├── README.md      # 本文件
├── DATA.md        # DICOM 数据应该放在哪里，运行任何程序前先阅读本文件
├── RESULTS.md     # 已经确认的结论及对应数字
├── requirements.txt
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

已于 2026-09-14 在 Python 3.12.7 上验证 dark/light 与 pseudo-ghost 分析脚本可以完整运行。当前本机测试结果为 10 项通过、1 项跳过；跳过项是因为本机没有安装可导入的 PyTorch，不影响本次 NumPy/SciPy 分析。

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

# 第一轮 pseudo-ghost 机制实验。默认运行 5→6 和 27→28，输出严格
# 空间 OOF 六联图、四尺度指标、null 对照和可复查的 OOF 数组。
python scripts/analyze_pseudo_ghost_mechanism.py

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
5. 运行 `python scripts/analyze_pseudo_ghost_mechanism.py`，查看 `5→6` 和 `27→28` 的严格 OOF 六联图。下一步先区分 residual 中的 baseline 误差与旧帧结构，再冻结模型并回到全部 30 组验证。
6. 阅读 `docs/plans/2026-05-28-real-data-acquisition-protocol.md`，准备真实三元组采集。后者需要扫描仪使用时间，仍是验证实际去除质量的必要条件。

---

## 已知缺口

- 大多数图像没有真实标签，因此无法用 PSNR 或 SSIM 评估去除质量。目前的验证方式是人工视觉判断加“相干残影抑制”指标。
- `src/evaluation/` 和 `src/training/` 是空包。指标计算与训练循环都直接写在 `scripts/train_unet.py` 中。
- 超参数通过命令行参数设置，而不是 YAML 配置文件。
- dark/light 分析只验证了序列中的空间相关与多帧记忆，没有证明任何 dark 图像是 clean reference，也没有产生监督训练标签。
- 62 张 dark/light DICOM 没有 `AcquisitionTime`；`InstanceCreationTime` 不足以支持物理时间衰减建模。曝光关联分析尚未执行。
- 多阶模型需要在共同全图域上使用观测 DICOM 值；当 light 图像发生截断饱和时，lag 系数可能有偏。
- `outputs/dark_light_analysis_20260913/` 保存本地可复现结果，并由 `.gitignore` 排除。不要用强制添加把工作簿、JSON 或临时诊断产物提交到 Git。
- dark/light 默认输出目录沿用 `20260913` 这一历史名称；真实运行时间以 JSON 的 `generated_at` 为准，不要从目录名推断。
- `RESULTS.md` 第 1 节和第 2 节记录了两处命名/标签不一致问题。它们都不会改变任何结果，但会误导代码阅读者。
- 仓库中没有包含已训练的 U-Net checkpoint。由于基于合成数据训练的模型无法迁移到真实数据，因此后续计划是在真实配对数据上重新训练。
