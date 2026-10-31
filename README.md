# DR X射线单帧残影研究

研究紧邻前序曝光在后续图像中的残留。项目分为已有方法基础、M1基线和后续研究，所有运行任务共用一个入口。

```text
run.py                      统一命令入口
xray_ghost_removal/
  common/                   图像读写、阈值分割、空间操作、路径与溯源
  legacy/                   既有传统去除器、合成数据和U-Net实验
  m1/                       M1配置、分割、背景、拟合、验证与展示
  research/                 候选模型、迁移、曝光关联及配对研究
    audits/                 已有背景和覆盖审计
  data/                     新数据接收与检查
tests/                     按相同职责组织的回归测试
configs/                   历史冻结配置
docs/                      结构说明、采集协议和冻结实验设计
data/                      原始数据与少量必要历史记录（不提交）
outputs/                   当前结果与必要冻结证据（不提交）
```

[数据与输出](DATA.md) · [研究结果](RESULTS.md) · [结构及迁移对应](docs/architecture.md) · [开发约束](AGENTS.md)

## 安装与运行

推荐Python 3.12。依赖统一声明在 `pyproject.toml`：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python run.py --help
python -m pytest -q
```

已有U-Net实验需要额外安装 `python -m pip install -e '.[legacy]'`。基本M1和数据检查不依赖PyTorch。可用 `python -m xray_ghost_removal` 或安装后的 `xray` 运行相同命令；所有相对数据与结果路径按项目根目录解析。

| 工作 | 命令 |
|---|---|
| 查看M1入口 | `python run.py m1 --help` |
| 当前无膨胀M1拟合参数 | `python run.py m1 fit --help` |
| 新RAW初检 | `python run.py data inspect-raw --data-dir '10.30例图' --shape 3072 3072 --dtype '<u2'` |
| RAW整批导入Horos | `python run.py data export-raw-dicom`，将生成的 `outputs/raw_intake/dicom/` 整个目录导入Horos |
| RAW上的M1跨配对检验 | `python run.py research raw-m1`，设计见 `docs/plans/2026-10-30-raw-m1-cross-pair.md` |
| RAW顶格/未顶格分区模型Msplit | `python run.py research raw-split`，设计见 `docs/plans/2026-10-30-raw-split-model.md` |
| 读取冻结M1并导出阅片 | `python run.py m1 review` |
| 半合成演示与精选展示 | `python run.py m1 simulate` / `python run.py m1 presentation` |
| 既有seamless去除样例 | `python run.py legacy remove --indices 4,6,18` |
| 候选方法及曝光研究 | `python run.py research --help` |

阅片和半合成命令会替换同一稳定路径的派生产物，运行前查阅[阅片说明](docs/plans/2026-09-30-dicom-review.md)。旧脚本路径已迁移，不保留重复实现或兼容脚本。

## 当前研究边界

- 设备已确认为DR平板；存量DICOM的 `Modality=CR` 保留为文件头事实。
- 2026-10-30新增142张RAW，有70组高值→低值配对；格式暂按3072×3072、小端uint16读取。帧角色、目录时间字段和处理链未确认，确认清单已整理、待设备方回复。以两张开头低值帧作干净暗帧，M1跨配对检验（RESULTS第9节）显示参数可迁移、仿射形式不充分；按预设门槛，顶格/未顶格分区模型Msplit已晋升为这批RAW的新基线（RESULTS第10节），仍不含时间项。M1仍是DICOM研究的基线，代码未改。
- 当前分割保持阈值/SAM掩膜边界，禁止膨胀、填孔或扩大排除区。新的M1拟合默认写入 `outputs/m1/`，并拒绝覆盖历史冻结目录。
- 历史 `background_frozen_v1.json` 使用24像素膨胀，其配置、SAM归档和审计原样保留。旧冻结NPZ可用于既有比较；当前实现不能重建其掩膜定义。历史BG审计保留作证据代码，不能直接重跑旧膨胀网格。
- M1是存储像素/分块域的经验模型。`Y = dark - estimated_background` 包含噪声和背景误差；真实恢复率、物理衰减和曝光因果结论仍需要相应参考与采集记录。数字统一见RESULTS。

## 整理记录

第一阶段清理5个一次性或重复入口、1份过期设计和2个空包，移除7个闲置函数及3项退役测试，清理122个旧派生文件（985,403,611字节）。原始数据、既有方法和冻结证据保留；原始30帧链接与旧运行记录仍在。59项原测试中退役3项，保留56项通过。

第二阶段将28个实现文件按职责迁移，拆分M1大脚本，统一命令、依赖和测试布局；具体对应见结构说明。重构前后的3组真实冻结样例拟合报告及11个数值数组逐项一致；67项测试、22个命令帮助入口检查通过，315个受保护文件哈希未变。未重跑或替换正式实验结果。
