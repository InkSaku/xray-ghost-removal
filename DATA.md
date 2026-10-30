# 数据

设备已由用户确认是DR平板。存量两批92张DICOM头均写Modality=CR；保留原文件编码，不用它证明硬件类型或光擦除机制。新采集按DR复位与读出流程执行，不能从存量文件缺失字段推断已控制状态。

代码仓库不跟踪或分发任何 DICOM 文件。当前工作机上已经放置影像数据，但其他机器 clone 仓库后缺少 `data/` 属于正常情况；92张原始DICOM只读审查容量约1.636 GB（实际交付按文件大小核算），需要通过受控方式单独传输。本文件准确说明每个数据文件夹应该放在哪里。

所有脚本都会相对于仓库根目录解析路径，因此请在 `run.py` 同级目录下准备数据，并通过 `python run.py <分组> <命令>` 运行。相对输入/输出路径统一按项目根目录解析。

---

## 预期目录结构

```
<repo root>/
├── xray_ghost_removal/
├── run.py
└── data/
    ├── raw/
    │   ├── 残影图像/                 # 30张连续图像（文件头编码CR），1.dcm ... 30.dcm
    │   └── AI修残影例图/              # 62 张 dark/light 图像，以及 拍摄参数记录.xlsx
    └── processed/                    # 由脚本自动创建，请勿手动填充
        ├── linear_cleaned/
        ├── physics_cleaned/
        ├── seamless_cleaned/
        └── unet_cleaned/
```

原始数据目录按需准备，输出目录由相应入口创建。

```bash
mkdir -p data/raw
```

文件夹名称中包含中文，并且脚本中对这些名称进行了硬编码，因此请严格保持名称不变。`残影图像` 表示 ghost images，`AI修残影例图` 表示 AI ghost-repair example images。

---

## 数据集 1：连续残影序列

`data/raw/残影图像/` 中包含 30 个文件，文件名从 `1.dcm` 到 `30.dcm`，每个文件约 15.5 MB。

| 属性 | 数值 |
|---|---|
| DICOM模态字段 | CR（文件头编码；设备按用户确认的DR记录） |
| 尺寸 | 3048 x 2548 |
| 位深 | 16-bit，MONOCHROME2 |
| 光度约定 | 空气区域为高像素值，吸收 X 射线的物体为低像素值 |
| DICOM 头中的检查日期 | 20260515 |
| 身份标识 | 头信息中不包含患者姓名、ID、出生日期、机构名称或就诊号 |

这些图像构成连续历史序列；已有分析支持部分相邻图像的残留结构，也曾检查更早历史。设备为DR，文件缺少探测器ID与完整读出状态，不能用文件头CR或序号单独证明同一成像板及光擦除流程。该数据没有真实标签。`RESULTS.md` 第 1 至第 3 节的定量结论来自这组序列。

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

`xray_ghost_removal/research/pairing.py` 已经完成这组数据的阶段 0–2 分析。结果支持部分相邻帧候选配对，但不支持把 dark 图像整批视为 clean reference。该数据也不是采集协议定义的 ground-truth 三元组，不能直接作为监督标签。项目后续只研究紧邻前一张 light 造成的单帧残影；当前事实和判据见 `RESULTS.md` 第 4 节。

### 已确认的数据问题：相同曝光参数不等于相同实验条件

工作簿中存在 4 组 kV、mA 和曝光时长完全相同的记录，但对应的单帧拟合残影系数存在差异：

| 名义曝光设置 | 对应 pair | alpha 最大/最小比 | 现象 |
|---|---|---:|---|
| 70 kV、32 mA、32 ms | `3→4`、`4→5` | 32.3 | 一个检出、一个未检出 |
| 70 kV、63 mA、32 ms | `5→6`、`6→7` | 50.1 | 一个检出、一个未检出 |
| 70 kV、100 mA、100 ms | `1→2`、`2→3` | 33.7 | 两个均检出，但强度相差很大 |
| 110 kV、16 mA、100 ms | `23→24`、`25→26` | 1.70 | 差异相对较小 |

alpha为四折空间OOF斜率均值。前两组较小alpha未通过检出门槛，倍率只是拟合系数比，不能当作精确物理残影倍率；同设置不等于受控独立复测。四组全部展示的PNG/SVG由`xray_ghost_removal/research/exposure_plot.py`读取既有JSON并核对冻结系数生成，位于`outputs/mhinge_cross_pair_v1/exposure_association/repeated_settings_alpha.{png,svg}`。

其中 `1→2` 和 `2→3` 最能说明问题：两张 source light 的物体面积、中位信号强度和
探测器位置几乎相同，但全图 alpha 仍相差 33.7 倍；在两个双方都具有可靠预测的共同
探测器区域内，局部 alpha 也分别相差 28.5 和 33.7 倍。因此这个差异不能简单归因于
曝光量、物体大小、物体位置或某一个局部探测器区域。

现有 DICOM 头不足以追查真正原因。文件只保留尺寸、位深、光度解释等基础字段，没有保存
板标识、擦除状态、曝光至读取间隔、ExposureIndex、Sensitivity、Detector 信息、LUT、
Rescale 或读片处理描述。现在只能推测探测器状态、复位、读取时机或处理流程可能不同，
不能从现有文件确认是哪一个因素。

因此使用这批数据时必须遵守以下边界：

- 不得把“曝光参数相同”当作完整实验条件相同，也不得把这些样本直接当作受控重复实验。
- 不得仅根据当前数据建立 mAs–alpha 剂量规律；现有 13-pair 队列中没有稳定单调关系。
- 不得使用 `InstanceCreationTime` 补造曝光至读取间隔或时间衰减常数。
- 后续DR采集固定并记录探测器标识、复位、实际帧历史、积分/读出模式、曝光至暗读出的时序、处理链、source和几何；曝光档位分轮采集，包含独立重复及无曝光流程对照。

---

## 数据集 3：10.30 RAW 例图（初检，采集定义未确认）

2026-10-30 收到仓库根目录 `10.30例图/`，内有三组共142张 `.raw`，文件名日期为2026-10-29，无规格、操作记录或设备/时序侧车。原始文件保持原位、只读；这批文件不能视为下节控制采集协议已经交付。

用户已确认所见目录即全部交付内容，没有其他采集说明。低值帧是否为关束后的第一张实际暗读出、目录中1000/3000ms的事件含义、每轮复位及物体固定情况均保留为未知；没有记录不能证明这些操作已执行或未执行。

| 原目录名 | RAW数量 | 多数像素为55000的帧 | 其余低值帧 | 相邻高值→低值候选对 |
|---|---:|---:|---:|---:|
| 残影 70kv 250ma 200ms 曝光时间3000ms | 51 | 25 | 26 | 25 |
| 残影 70kv 250ma 200ms 曝光时间1000ms | 50 | 25 | 25 | 25 |
| 残影 70kv 200ma 160ms 曝光时间1000ms | 41 | 20 | 21 | 20 |

每张18,874,368字节，吻合无头3072×3072×16位布局；按小端无符号16位解码得到连贯物体结构，采样邻域差异也明显小于大端解码。因此暂按 `<u2` 读取，但厂商格式、方向及处理链仍未确认。142张文件SHA-256均不同。高低值帧按文件名排序交替出现；两个组各有一张开头低值帧，无本组内紧前高值来源。70个候选对内部的文件名十六进制序号均连续；全序列仍有9处跳号，跳过10个计数值，无法确定对应哪些未导出事件。

预览可见低值帧保留物体结构，但没有说明证明它们是关束后第一张实际暗读出。高值帧约57%–95%的像素恰好为55000，是明显的观测平台；55000并非uint16上限，不能在缺少处理链说明时断言硬件饱和原因，亦不宜直接当作未截断的线性曝光输入。目录同时写200/160ms和“曝光时间1000/3000ms”，后者含义不明，保留原名，不解释为曝光后等待时间。

候选对的**文件名时间差**分别为：250mA/3000ms组3.973–5.267秒，250mA/1000ms组1.974–3.265秒，200mA/1000ms组2.290–22.772秒。这些时间尚无采集事件定义，不能作为真实曝光至暗读出的间隔，不能据此拟合物理衰减。

现有文件可用于导出序列中的前序结构对应核验、背景估计可用性检查及M1图像域探索。分析沿用高值帧/低值帧的观测名称，不预设light/dark采集角色；两个组的开头低值帧不自动视为clean背景，70个候选对不自动视为独立重复。三目录仅作为文件分组，组间差异不归因为曝光量或读出延迟；缺少经确认的clean参考时不报告真实恢复率，也不直接构造监督训练标签。

可重复初检命令如下，稳定覆盖 `outputs/raw_intake/inventory.json` 和 `overview.png`。该目录是本批当前初检结果，没有历史前身；不改动冻结分析。JSON保存原始文件哈希、逐帧统计、序号跳变和候选配对；预览使用逐图独立显示窗，不用于比较绝对残影强度。未做分割、掩膜操作、背景拟合或去除。

```bash
.venv/bin/python run.py data inspect-raw \
  --data-dir '10.30例图' --shape 3072 3072 --dtype '<u2'
```

## 尚未交付的协议化DR控制数据

本轮不是当前物体真实曝光图的三元组，而是单次曝光后无新曝光的暗读出控制实验。已确定A有物体8组、B无遮挡3档×3次共9组、S无曝光流程对照2组，共19个独立循环、17次实际曝光、169张核心图像；可选dark_03/04全部保留时203张。两组低/高试拍合格时抵扣正式组，不重复计数。

每个A/B循环交付3张clean_before、1张light/flood、2张dark、3张clean_after；S交付3张clean_before、2张sham_dark、3张clean_after。公司交付目录为`DR_ghost_data/A_有物体/`、`B_无遮挡单次/`、`S_无曝光对照/`及逐图记录/设备与时序说明。导入时放入`data/raw/DR_ghost_data/`，不改写已有raw数据；尚未采集，不将计划写成已取得结果。

第一实际暗读出帧、积分窗口OFF、隐藏读出、背景稳定与量程检查按[现行DR执行协议](docs/plans/2026-05-28-real-data-acquisition-protocol.md)。本轮支持残留可测性、重复性、曝光响应及暗图扣除初步验证，不保证剂量规律。

需要验证当前物体结构恢复时再追加`data/raw/paired/pair_001/`，保存previous、ghosted、clean_before、clean_after及事件侧车；clean重复用于量化参考误差，dark背景不能替代物体clean曝光。规模按试采变异和目标范围决定，不把早期10–30组或50–100组设为已证明充分的数量。

---

## 输出

脚本会把清理后的 DICOM 写入 `data/processed/`，把对比图写入 `results/figures/`，把分析图写入 `figures/final/`。每次执行去残影任务还会生成一个 JSON 日志，逐张记录是否执行了修正、拟合出的系数、拟合质量，以及在跳过处理时对应的原因。在相信任何输出图像之前，请先阅读该日志，因为带门控的去除器会有意保持低置信度图像不变。

dark/light 稳健性分析默认只写一个可复现的机器记录：

```text
outputs/dark_light_analysis_20260913/robustness_results.json
```

该路径是脚本默认输出位置，当前工作目录未保留该批稳健性输出。早期 `results/dark_light_analysis/` 的17/5/8汇总使用InstanceCreationTime，已被后续5/21/4稳健性口径取代，于2026-10-30连同逐对图片清理。默认脚本不会生成逐对图片或CSV。一次性渲染、检查日志和调试图应放在系统临时目录，不应长期留在项目中。

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

该目录保留原两组冻结输入，扩展13组输入位于下述 `mhinge_cross_pair_v1/`。NPZ 内 `source` 即 `X`，`observable_signal` 即冻结的 `Y`，并同时保存 `background`、`raw_dark`、背景拟合掩膜、source support、饱和比例、评价掩膜和空间 fold map。`analysis_results.json` 同时是 manifest，记录冻结配置/SAM/输入 DICOM/输出产物以及关键分析代码的哈希，不再额外生成一批独立数组文件。

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

单帧曝光关联复用同一 13-pair 冻结证据边界，不再创建新的顶层版本目录：

```text
outputs/mhinge_cross_pair_v1/exposure_association/
├── exposure_association.json
└── exposure_association.png
```

两项产物分别是完整机器记录和唯一汇总图。重运行必须原地覆盖；临时统计、试画和渲染检查
继续写入系统临时目录，不进入仓库。

双向迁移改变了科学问题与证据边界，因此使用独立的稳定目录：

```text
outputs/pair_transfer_v1/
├── transfer_results.json
└── transfer_diagnostics.png
```

它只读取已冻结的 `3→4` 与 `5→6` block16 NPZ；`3→4` 在 13-pair manifest 所列
原路径暂缺，脚本读取 `pseudo_ghost_mechanism_v1` 中哈希完全相同的副本。JSON 记录
输入哈希、双向直接迁移和四折轮换校准的结果。重运行原地覆盖，不增加逐折图片。

BG 冻结配置位于 `configs/background_frozen_v1.json`。该配置依赖本地 `outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz`，但只通过 SHA-256 校验，不将来自受控影像的派生 mask 强制提交到仓库。冻结审计的本地机器结果位于 `outputs/background_freeze_audit_v1/`。其中原 `background_freeze_audit.*` 是已被后续审查引用的冻结记录，不能覆盖；新增的完整物体遮挡与逐折重算 BG 证据稳定覆盖写入 `background_crossfit_audit.json` 和 `background_crossfit_audit.md`，不另建按日期增长的目录。

当前数据对 BG 的验证仍有一个无法靠代码消除的空缺。13 个真实 source 形状在 13 张目标 dark 的已知空气区共尝试 169 次完整放置；只成功 58 次，111 次在不缩小物体的前提下放不进去。成功样本只覆盖占图 8.57%–12.48% 的物体，所有占图 19.94%–57.83% 的大物体均没有成功案例，而且没有任何完整形状能在原探测器位置验证；成功样本的最小可行平移中位数为 53.29 个 block。因此现有数据可以检查一部分真实面积和形状，但不能提供“大物体原位置下的真实 BG”。无法放置的案例在 JSON 中明确标为 `unverified`，不得当作通过验证。

固定图样空间可识别性审查写入 `outputs/fixed_pattern_identifiability_v1/`。其中 JSON 是完整机器记录，CSV 是 13 个 pair 的精简覆盖表，Markdown 和 3 张 PNG 用于人工复核。所有统计都位于真正进入模型的 16×16 block-mean 网格；原始 DICOM 不会被写入该目录。

目录名中的 `20260913` 是首次分析时遗留的输出路径，不代表文件的实际生成日期。准确运行时间读取 JSON 内的 `generated_at`；当前正式结果生成于 2026-09-14。

`outputs/` 已由 `.gitignore` 排除。不要使用 `git add -f outputs/` 绕过该边界；确需提交汇总产物时，应先检查文件大小、敏感头信息和长期复用价值。原始 DICOM 必须保持只读，不得为了分析方便覆盖或改名。

后续分析使用稳定的输出路径：同一科学定义下的新运行应在核对用途后覆盖旧结果，不按日期或实验轮次无限新建目录。一次性调试、渲染检查和测试产物必须使用系统临时目录或 pytest `tmp_path`。只有科学定义或冻结证据边界发生变化时才新建版本化输出，并必须说明保留旧版本的原因。

半合成入口`xray_ghost_removal/m1/semisynthetic.py`稳定写入`outputs/dicom_review/`，可生成6个案例：供体3→4/5→6的冻结实测Y，分别加到4/6/27-light，输出18个DICOM及`results.json`。使用供体原有full-fit alpha/intercept，不在合成目标上重新拟合，保持信号×1；无存储裁剪时浮点残差等于供体Y−预测，与底图无关。

2026-10-30核对：顶层仅保留results.json，其中记录的18张DICOM已不在当前目录；本次清理未修改该记录。精选presentation/的9张DICOM、3张PNG及说明仍保留。

同目录`xray_ghost_removal/m1/review.py`可重建3对冻结OOF dark阅片，包含9个DICOM、3张对照图及review_results/review.md，但运行时会替换当前顶层半合成派生产物。历史seamless检查和OOF阅片结果见RESULTS；不能将历史输出描述为当前仍存在。

`presentation/`是当前精选演示包，由`xray_ghost_removal/m1/presentation.py`复现：G0506_B08、G0304_B30、G0506_B04，共9个全尺寸DICOM、3张PNG及现有说明/selection.json。原light结构窗独立，加入残影与校正后使用完全相同WL/WW：前两组55000/180，B04为15798/250。PNG为16×16均值展示，误差图共享色标。B08/B30最大值堆积分别约93.1%/97.9%，只适合作半合成流程说明；底图不是认证clean、供体参与过拟合，不据此声明真实结构恢复或未见残影泛化。

曝光研究启动前的原始数据审查复用
`outputs/mhinge_cross_pair_v1/exposure_association/data_feasibility.{json,md,png}`。
审查结论与现行采集边界见`RESULTS.md`和`docs/plans/2026-05-28-real-data-acquisition-protocol.md`；复现命令：
`.venv/bin/python run.py research exposure-audit --diagnostic-dir /tmp/xray-exposure-feasibility`。
脚本只读解码并核验全部62张原图、参数表和13对冻结NPZ，原图接触表写入指定临时目录。
当前3个低最大值堆积且通过实用效应门槛的主样本为3→4、5→6、27→28；4→5、6→7保留
为未检出对照。其他6个有效pair存在source截断风险，不能与主样本直接混合标定曝光系数。
原始62张共用SOPInstanceUID；31张light最大值均为55000，24张堆积超过全图1%。
这些是新核验的数据质量事实，不修改原图或已冻结审计。现有数据缺少同物体、同状态、
可重复的曝光梯度，不足以归因曝光效应或建立曝光相关alpha函数。

2026-10-30第一阶段清理移除了旧 `data/horos_30series/` 查看副本和 `data/processed/seamless_cleaned/` 中30张处理图，旧运行记录removal_log.json保留。Horos副本已逐张验证与原图像素一致；`data/raw/残影图像` 是指向仓库外原始30帧目录的符号链接，链接及原图均保留。旧seamless对照PNG已清理，需要时由保留的去除入口重建。

## 第二阶段后的输出边界

新的无膨胀M1运行使用 `python run.py m1 fit`，默认输出 `outputs/m1/`；该目录仅在实际运行时创建。本次结构整理没有生成新实验结果。旧 `pseudo_ghost_mechanism_v1/`、`mhinge_cross_pair_v1/` 等保留为历史冻结输入，新拟合入口拒绝向这些目录及其子目录写入。既有候选比较和迁移命令仍读取旧冻结NPZ，不能据此声称已验证新的掩膜定义。142张RAW保持原位，未经帧角色核验不能直接套用31组DICOM序列入口。
