# TCN-comparison 实验目录

本目录按实验链路分为两个独立工程：

- [TCN](./TCN/README.md)：五组仅 TCN 的离线功率与能耗预测，无 RLS、无置信区间、无在线更新。
- [TCN+RLS](./TCN+RLS/README.md)：在上述五组 TCN 权重上分别搜索 RLS 遗忘因子、初始协方差和反馈时间窗，并评价置信区间覆盖率。

两个工程的加工数据分别位于：

- `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/TCN/processed`
- `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/TCN+RLS/processed`

模型分别位于：

- `D:/Python-files/Energy-prediction/model/TCN`
- `D:/Python-files/Energy-prediction/model/TCN+RLS`

五组设置保持一致：A 为原始 10 维混合激活，B 为原始 10 维统一 SiLU，C 为原始 10 维加 5 个 10 步 SMA 通道的混合激活，D 为相同 15 维输入并统一 SiLU，E 为相同 15 维输入并统一 LeakyReLU(0.05)。详细参数、运行命令、日志和图表分别以两个子目录中的 README 与 `out/outreadme.md` 为准。

2026-11-01新增E的完整实验已完成：TCN任务能耗WAPE为2.9733%，RLS后为0.6949%，反馈后区间覆盖94.64%、评分4.0943Wh。D的点预测误差更低，E的反馈后区间评分更低；详细参数及局限见两个子工程README。
