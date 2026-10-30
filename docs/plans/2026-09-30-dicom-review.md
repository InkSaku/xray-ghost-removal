# DICOM阅片与半合成展示

现有入口的科学定义不同，复用 `outputs/dicom_review/` 顶层路径。运行前确认所需方法；顶层派生产物会被替换，`presentation/` 子目录单独管理。

| 入口 | 定义 | 生成内容 |
|---|---|---|
| `xray_ghost_removal/m1/review.py` | 冻结四折空间OOF M1；dark减去预测，保留BG | 3→4、5→6、27→28，共9个DICOM、3张对照图及阅片记录 |
| `xray_ghost_removal/m1/semisynthetic.py` | 将3→4/5→6供体实测Y按×1加到4/6/27-light，再减供体既有full-fit预测 | 6例、18个DICOM及results.json |
| `xray_ghost_removal/m1/presentation.py` | 半合成精选展示，原light结构窗独立，叠加与校正同窗 | presentation/中G0506_B08、G0304_B30、G0506_B04，共9个DICOM、3张PNG及selection.json/README |

## 当前保留文件（2026-10-30核对）

顶层保留半合成 `results.json`，其记录的18张DICOM当前已不存在；需要时按原入口重建。现有可查看文件为 `presentation/` 的9张DICOM和3张PNG及说明。本次清理未修改这些结果。

2026-09-30的seamless检查和2026-10-12的OOF阅片历史、数值及限制保留在 [RESULTS](../../RESULTS.md)。旧seamless选样、ROI及预览代码已退役；当前 `prepare_dicom_review.py` 仅导出冻结M1。

## 验证与解释

- 原始文件只读，读取时核验冻结哈希；研究DICOM使用独立标识，并通过像素读回检查。
- OOF预测按16×16块常数应用，保留块内像素差异；不新增像素级模型证据。
- 半合成供体参与过拟合，底图未认证clean，不能据此报告未见残影泛化或真实恢复率。
- 精选展示B08/B30有明显最大值堆积；共享窄窗用于显示附加信号，不能证明截断细节得到恢复。
- 设备为DR，源文件头编码CR原样保留。历史冻结掩膜定义及输出不可改写为无膨胀结果。
