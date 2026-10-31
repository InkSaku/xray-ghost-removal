# 项目结构与迁移说明

本次按现有实现的职责整理，不改变算法和研究结论。`legacy` 表示已有基础；目录划分不作为逐文件作者归属的证明。M1保留为独立基线，研究候选通过导入基线模块或读取冻结输入开展比较。

## 模块边界

- `common`：DICOM、无头RAW读取（`raw.py`）、阈值分割、空间分块与fold、曝光表、路径和溯源。只依赖基础库，不导入具体方法或研究。
- `legacy`：既有物理修补、线性门控、seamless、合成数据与U-Net。依赖common。
- `m1`：`config.py`保存现有参数和验证；`segmentation.py`负责掩膜加载/SAM；`background.py`负责背景与固定图样；`fitting.py`负责仿射拟合、严格空间OOF和检出；`validation.py`负责遮挡验证；`reporting.py`负责图表和汇总；`pipeline.py`只组织运行。依赖common，不导入研究候选或legacy。
- `research`：配对核验、非线性、跨pair审查、迁移和曝光分析；10.30 RAW研究中，`raw_cohort.py`负责帧读取、配对、干净暗帧与评分，`raw_m1.py`检验M1跨配对预测，`raw_split.py`实现顶格/未顶格分区模型Msplit及其与M1的比较。依赖common和M1；候选逻辑不得反向写入基线；历史背景、逐折和覆盖审查统一放在其audits子包。
- `data`：数据接收与格式检查；`raw_dicom.py`读取初检清单并复用common的DICOM写入/读回，生成整批Horos阅片副本。依赖common。
- `cli.py`：惰性选择命令，统一相对路径基准；无参数任务的帮助不会启动导出。只有legacy训练/诊断加载PyTorch。

测试按 `tests/legacy`、`tests/m1`、`tests/research`、`tests/data` 组织；`tests/test_cli.py`检查入口、依赖方向和冻结路径保护。依赖只在pyproject.toml维护，仓库采用可编辑安装；原始影像和冻结产物不随Python包分发。

## 旧路径对应

旧结果JSON中的路径和哈希记录生成时状态，继续原样保留。下表用于定位当前实现；迁移后的文件哈希不代表旧源文件，不能用本次重构声称精确复现旧膨胀定义。两份2026-09-21冻结设计文档也原样保留，其旧命令属于历史定义。

| 旧实现 | 当前实现 | 统一入口 |
|---|---|---|
| `src/models/physics_model.py` | `xray_ghost_removal/legacy/physics.py` | — |
| `src/models/linear_ghost.py` | `xray_ghost_removal/legacy/linear.py` | — |
| `src/models/seamless_ghost.py` | `xray_ghost_removal/legacy/seamless.py` | — |
| `src/models/unet.py` | `xray_ghost_removal/legacy/unet.py` | — |
| `src/data/synthetic.py` | `xray_ghost_removal/legacy/synthetic.py` | — |
| `src/utils/dicom_utils.py` | `xray_ghost_removal/common/dicom.py` | — |
| `scripts/run_ghost_removal.py` | `xray_ghost_removal/legacy/remove.py` | `python run.py legacy remove` |
| `scripts/run_physics_baseline.py` | `xray_ghost_removal/legacy/inpaint.py` | `python run.py legacy inpaint` |
| `scripts/analyze_linear_fit.py` | `xray_ghost_removal/legacy/analysis.py` | `python run.py legacy linear-fit` |
| `scripts/train_unet.py` | `xray_ghost_removal/legacy/train.py` | `python run.py legacy train` |
| `scripts/diagnose_model.py` | `xray_ghost_removal/legacy/diagnose.py` | `python run.py legacy diagnose` |
| `scripts/inspect_raw_batch.py` | `xray_ghost_removal/data/raw_intake.py` | `python run.py data inspect-raw` |
| `scripts/analyze_dark_light_pairs.py` | `xray_ghost_removal/research/pairing.py` | `python run.py research pairing` |
| `scripts/analyze_frozen_candidates.py` | `xray_ghost_removal/research/candidates.py` | `python run.py research candidates` |
| `scripts/analyze_intensity_nonlinearity.py` | `xray_ghost_removal/research/nonlinearity.py` | `python run.py research nonlinearity` |
| `scripts/analyze_three_pair_nonlinearity.py` | `xray_ghost_removal/research/three_pair.py` | `python run.py research three-pair` |
| `scripts/analyze_pair_transfer.py` | `xray_ghost_removal/research/transfer.py` | `python run.py research transfer` |
| `scripts/audit_mhinge_cross_pair.py` | `xray_ghost_removal/research/hinge_audit.py` | `python run.py research hinge-audit` |
| `scripts/analyze_single_frame_exposure.py` | `xray_ghost_removal/research/exposure.py` | `python run.py research exposure` |
| `scripts/audit_exposure_feasibility.py` | `xray_ghost_removal/research/exposure_audit.py` | `python run.py research exposure-audit` |
| `scripts/plot_repeated_exposure_alphas.py` | `xray_ghost_removal/research/exposure_plot.py` | `python run.py research exposure-plot` |
| `scripts/audit_background_freeze.py` | `xray_ghost_removal/research/audits/freeze.py` | `python run.py research freeze-audit` |
| `scripts/audit_background_crossfit.py` | `xray_ghost_removal/research/audits/crossfit.py` | `python run.py research crossfit-audit` |
| `scripts/audit_fixed_pattern_identifiability.py` | `xray_ghost_removal/research/audits/coverage.py` | `python run.py research coverage-audit` |
| `scripts/prepare_dicom_review.py` | `xray_ghost_removal/m1/review.py` | `python run.py m1 review` |
| `scripts/run_m1_semisynthetic.py` | `xray_ghost_removal/m1/semisynthetic.py` | `python run.py m1 simulate` |
| `scripts/prepare_m1_presentation.py` | `xray_ghost_removal/m1/presentation.py` | `python run.py m1 presentation` |
| `scripts/analyze_pseudo_ghost_mechanism.py` | `xray_ghost_removal/m1/pipeline.py` | `python run.py m1 fit` |

原 `analyze_pseudo_ghost_mechanism.py` 的函数按上面的M1职责拆分。分块、fold和source居中移至 `common/spatial.py`，通用阈值掩膜移至 `common/segmentation.py`，曝光表读取移至 `common/exposure.py`。冻结M1输入读取集中在 `m1/frozen.py`，文件哈希与显示窗分别由common的provenance和dicom模块提供。原先从阅片入口借用这些工具的依赖已解除。不保留旧目录包装层，新增代码应直接导入定义函数的模块。

## 数据、结果与重构验证

原始数据、原始30帧的符号链接及其目标保持原位。`10.30例图/` 是既有根目录数据例外，新批次应放到data/raw；不为整理外观改写已有原始文件或冻结manifest。outputs下现有证据保持路径和字节内容。

新的无膨胀M1运行默认使用outputs/m1，实际执行才创建。旧冻结目录不作为新拟合输出。既有研究命令仍使用历史冻结输入，须按其原掩膜定义解释；当前还没有新的无膨胀正式结果。

本次比较了重构前后3→4、5→6、27→28的M1重拟合报告，以及预测、残差等11个数值数组（含两种背景拟合），逐项一致。原56项回归测试保留，并补充统一入口、模块依赖、冻结写入保护和数据接收检查。诊断与比较中间文件均位于系统临时目录。
