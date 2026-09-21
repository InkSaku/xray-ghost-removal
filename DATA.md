# 数据

代码仓库不跟踪或分发任何 DICOM 文件。当前工作机上已经放置影像数据，但其他机器 clone 仓库后缺少 `data/` 属于正常情况；约 2.8 GB 的原始数据需要通过受控方式单独传输。本文件准确说明每个数据文件夹应该放在哪里。

所有脚本都会相对于仓库根目录解析路径，因此请在 `src/` 和 `scripts/` 的同级目录下创建 `data/`，并从仓库根目录运行所有命令。

---

## 预期目录结构

```
<repo root>/
├── src/
├── scripts/
└── data/
    ├── raw/
    │   ├── 残影图像/                 # 30 张连续 CR 图像，1.dcm ... 30.dcm
    │   └── AI修残影例图/              # 62 张 dark/light 图像，以及 拍摄参数记录.xlsx
    └── processed/                    # 由脚本自动创建，请勿手动填充
        ├── linear_cleaned/
        ├── physics_cleaned/
        ├── seamless_cleaned/
        └── unet_cleaned/
```

首次运行前先创建一次输出目录。

```bash
mkdir -p data/raw data/processed/{linear_cleaned,physics_cleaned,seamless_cleaned,unet_cleaned}
mkdir -p results/figures/{linear_cleaned,physics_baseline,seamless_cleaned,unet} figures/final
```

文件夹名称中包含中文，并且脚本中对这些名称进行了硬编码，因此请严格保持名称不变。`残影图像` 表示 ghost images，`AI修残影例图` 表示 AI ghost-repair example images。

---

## 数据集 1：连续残影序列

`data/raw/残影图像/` 中包含 30 个文件，文件名从 `1.dcm` 到 `30.dcm`，每个文件约 15.5 MB。

| 属性 | 数值 |
|---|---|
| 模态 | CR |
| 尺寸 | 3048 x 2548 |
| 位深 | 16-bit，MONOCHROME2 |
| 光度约定 | 空气区域为高像素值，吸收 X 射线的物体为低像素值 |
| DICOM 头中的检查日期 | 20260515 |
| 身份标识 | 头信息中不包含患者姓名、ID、出生日期、机构名称或就诊号 |

这些图像是在同一块成像板上连续采集的，因此图像 `n` 中包含图像 `n-1` 的残影，并且还会更微弱地包含更早图像的残影。该数据没有真实标签。`RESULTS.md` 第 1 至第 3 节的定量结论来自这组序列。

第 4 张图是参考样例。它具有最强、最干净的残影，也是确认环境和算法是否正常工作的最快方式。

## 数据集 2：dark/light 图像对

`data/raw/AI修残影例图/` 中包含 62 个 DICOM 文件，命名为 `N-dark.dcm` 和 `N-light.dcm`，其中 N 从 1 到 31，另外还有 `拍摄参数记录.xlsx`。

该电子表格包含 31 条 light 曝光记录，字段为编号、管电压（kV）、电流（mA）和曝光时长（ms）。其中“时间”是曝光持续时间，不是拍摄时刻。实测参数覆盖 50、70、100、110 kV，10–250 mA，32–200 ms，对应 1–50 mAs。

| 属性 | 已核对事实 |
|---|---|
| DICOM 数量 | 62 张：31 张 dark、31 张 light |
| 图像尺寸 | 3072 × 3072 |
| 记录顺序 | 每个编号先 `dark_N`，后 `light_N` |
| 主要因果候选 | `light_(N-1) → dark_N`，共 30 组 |
| `AcquisitionTime` | 62 张均缺失 |
| `InstanceCreationTime` | 仅用于审计；不得用于拟合、标签、真实间隔或衰减常数推算 |
| 配准 | 未配准，保留探测器坐标 |
| 隐私审计 | 常见直接标识字段（姓名、ID、出生日期、就诊号、机构、站点、设备序列号）为空，但 62 张均保留日期字段，且没有正式去标识声明；仍按敏感医学数据处理 |

`scripts/analyze_dark_light_pairs.py` 已经完成这组数据的阶段 0–2 分析。结果支持部分 H2 配对和多帧记忆，但不支持把 dark 图像整批视为 clean reference。该数据也不是采集协议定义的 ground-truth 三元组，不能直接作为监督标签。当前事实和判据见 `RESULTS.md` 第 4 节。

---

## 尚不存在的数据

数据采集协议要求采集由一张干净图、一张上一帧图像以及一张带残影图像组成的三元组，目录结构如下：

```
data/raw/paired/
├── pair_001/
│   ├── clean.dcm
│   ├── previous.dcm
│   ├── ghosted.dcm
│   └── meta.json
└── pair_002/
```

采集 10 到 30 组这样的数据，是当前唯一能够解除定量验证和监督训练瓶颈的改变。完整操作流程见 `docs/plans/2026-05-28-real-data-acquisition-protocol.md`。

---

## 输出

脚本会把清理后的 DICOM 写入 `data/processed/`，把对比图写入 `results/figures/`，把分析图写入 `figures/final/`。每次执行去残影任务还会生成一个 JSON 日志，逐张记录是否执行了修正、拟合出的系数、拟合质量，以及在跳过处理时对应的原因。在相信任何输出图像之前，请先阅读该日志，因为带门控的去除器会有意保持低置信度图像不变。

dark/light 稳健性分析默认只写一个可复现的机器记录：

```text
outputs/dark_light_analysis_20260913/robustness_results.json
```

同目录中的 `dark_light_afterglow_analysis.xlsx` 是便于人工查看的汇总副本；JSON 是程序输出和事实来源。默认脚本不会生成逐对图片或 CSV。一次性渲染、检查日志和调试图应放在系统临时目录，不应长期留在项目中。

pseudo-ghost 机制实验默认写入：

```text
outputs/pseudo_ghost_mechanism_v1/
├── analysis_results.json
├── pair_5_6/
│   ├── six_panel_oof.png
│   └── oof_maps_block16.npz
└── pair_27_28/
    ├── six_panel_oof.png
    └── oof_maps_block16.npz
```

其中 `Y_t = D_t - estimated_background_t` 是包含噪声、背景估计误差和潜在旧帧记忆的可观测残影信号，不是 clean ground truth。脚本可比较 LOO median、单帧平滑场、旧 hybrid 和污染感知 masked hybrid；后者的固定图样 `F` 不使用逐位置时间中位数。JSON 记录所用输入文件的 SHA-256；NPZ 只保存 16×16 block-mean 域的派生数组，不保存 DICOM 头。

当前该目录也是后续模型的唯一正式输入：NPZ 内 `source` 即 `X`，`observable_signal` 即冻结的 `Y`，并同时保存 `background`、`raw_dark`、背景拟合掩膜、source support、饱和比例、评价掩膜和空间 fold map。`analysis_results.json` 同时是 manifest，记录冻结配置/SAM/输入 DICOM/输出产物以及关键分析代码的哈希，不再额外生成一批独立数组文件。

冻结 M1 的候选机制比较覆盖写入同一稳定位置：

```text
outputs/pseudo_ghost_mechanism_v1/model_comparison/
├── model_results.json
└── residual_diagnostics.png
```

当前该 JSON 包含 M1 基准、嵌套空间 CV 的 Mblur 正式候选和 OOF Mquad 诊断。Mquad 被明确标记为 `diagnostic_only`；不包含组合模型、空间变化 alpha 或更后续的模型。两个文件会在每次正式运行时原地替换，不新建实验轮次目录。

强度非线性诊断使用新的稳定子目录
`outputs/pseudo_ghost_mechanism_v1/intensity_nonlinearity/`，其中只保留一个机器结果
`model_results.json` 和一张 `nonlinearity_diagnostics.png`。它不会覆盖
`model_comparison/`：旧目录是 Mblur 失败和 Mquad 50% 门槛的冻结审计记录，新目录改变了
科学问题与候选集合。两者都只读取相同的冻结 NPZ，不重新估计 BG。

Mhinge 跨 pair 审查使用独立的版本化输出边界：

```text
outputs/mhinge_cross_pair_v1/
├── analysis_results.json
├── pair_*/oof_maps_block16.npz
└── model_audit/
    ├── audit_results.json
    ├── pair_summary.csv
    └── cross_pair_summary.png
```

该版本仍复用原 SAM archive、冻结 BG 配置和空间 fold，但证据边界从两个 discovery
pair 扩展到 13 个预声明 pair，因此不覆盖 `pseudo_ghost_mechanism_v1`。每个 pair
只保留一个冻结 block16 NPZ；不再生成 13 张重复的六联图。`model_audit` 中三个
文件是当前 M1/Mhinge 复现审查的最小审计证据，同一科学定义下的重运行应原地覆盖。

BG 冻结配置位于 `configs/background_frozen_v1.json`。该配置依赖本地 `outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz`，但只通过 SHA-256 校验，不将来自受控影像的派生 mask 强制提交到仓库。冻结审计的本地机器结果位于 `outputs/background_freeze_audit_v1/`。

固定图样空间可识别性审查写入 `outputs/fixed_pattern_identifiability_v1/`。其中 JSON 是完整机器记录，CSV 是 13 个 pair 的精简覆盖表，Markdown 和 3 张 PNG 用于人工复核。所有统计都位于真正进入模型的 16×16 block-mean 网格；原始 DICOM 不会被写入该目录。

目录名中的 `20260913` 是首次分析时遗留的输出路径，不代表文件的实际生成日期。准确运行时间读取 JSON 内的 `generated_at`；当前正式结果生成于 2026-09-14。

`outputs/` 已由 `.gitignore` 排除。不要使用 `git add -f outputs/` 绕过该边界；确需提交汇总产物时，应先检查文件大小、敏感头信息和长期复用价值。原始 DICOM 必须保持只读，不得为了分析方便覆盖或改名。

后续分析使用稳定的输出路径：同一科学定义下的新运行应在核对用途后覆盖旧结果，不按日期或实验轮次无限新建目录。一次性调试、渲染检查和测试产物必须使用系统临时目录或 pytest `tmp_path`。只有科学定义或冻结证据边界发生变化时才新建版本化输出，并必须说明保留旧版本的原因。
