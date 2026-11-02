# TCN-comparison 五组实验总览

原有DJI和新增AMOVFLY实验分别保存；下方旧实验章节仅对应TCN、TCN+RLS。

更新日期：2026-11-01。A-E均已完成实际训练与测试，数值来自两个工程当前的CSV/JSON；本次文档刷新不重新训练或重选参数。

## 目录

- [AMOVFLY新数据实验](#amovfly新数据实验)
- [实验与文档入口](#实验与文档入口)
- [五组配置](#五组配置)
- [训练与校准预算](#训练与校准预算)
- [当前结果](#当前结果)
- [复现与报告更新](#复现与报告更新)
- [存储与日志](#存储与日志)
- [结果解释](#结果解释)


## AMOVFLY新数据实验

更新：2026-11-02，按指定输入重新训练TCN五组。原TCN和TCN+RLS保留DJI实验；AMOVFLY实验入口为 [TCN-AMOVFLY](TCN-AMOVFLY/README.md)、[TCN+RLS-AMOVFLY](TCN+RLS-AMOVFLY/README.md)，当前TCN报告见 [TCN报告](TCN-AMOVFLY/out/outreadme.md)。AMOVFLY的RLS权重仍基于旧特征契约，尚未与新TCN同步重训，旧RLS报告仅作历史记录。

新数据整理为状态输入10维/加SMA15维；277个ready文件接受256个，按关联组划分183训练、36验证、37测试。验证另分18选模/调参、18校准。数据字段、拒收原因和SHA-256见 `D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/TCN-comparison/README.md`。新数据与旧DJI特征定义、划分和训练预算不同，不能直接将误差大小作为跨数据集优劣比较。

| 组别 | 输入维数 | TCN窗口(s) | 功率WAPE(%) | 1秒能耗WAPE(%) | 整航段能耗WAPE(%) |
| --- | --- | --- | --- | --- | --- |
| A | 10 | 1 | 10.1318 | 9.9306 | 5.3715 |
| B | 10 | 1 | 10.1517 | 9.9478 | 5.0734 |
| C | 15 | 10 | 8.7603 | 8.5677 | 5.7913 |
| D | 15 | 10 | 9.0605 | 8.8751 | 6.0683 |
| E | 15 | 10 | 8.9641 | 8.7729 | 5.5684 |

```powershell
python TCN-comparison/TCN-AMOVFLY/main.py all --device cuda
python TCN-comparison/TCN+RLS-AMOVFLY/main.py all --device cuda
```

加工数据：`D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/TCN-comparison`。模型使用D盘独立TCN-AMOVFLY及TCN+RLS-AMOVFLY目录。TCN候选4×8轮、正式训练最多40轮并以验证早停。当前功率和1秒能耗误差最低为C，整航段能耗误差最低为B；本次结果不支持D组最佳。旧RLS权重与新TCN输入维数/列顺序不兼容，245候选结果不作为当前结论。实测轨迹条件输入和其他限制见TCN报告。

## 实验与文档入口

| 文档 | 内容 |
| --- | --- |
| [TCN/README.md](TCN/README.md) | DJI数据仅TCN离线实验入口、统一参数及五组摘要 |
| [TCN-AMOVFLY/README.md](TCN-AMOVFLY/README.md) | AMOVFLY指定10维与扩展15维SMA/激活对照 |
| [AMOVFLY/README.md](AMOVFLY/README.md) | ready数据统计、质量诊断与可视化 |
| [TCN/SMA_SiLU_plan.md](TCN/SMA_SiLU_plan.md) | 设计依据、当前实施状态、LeakyReLU附加对照与尚未执行的复核建议 |
| [TCN/improve.md](TCN/improve.md) | 按时间追加的工程记录；历史四组条目描述当时阶段 |
| [TCN/out/outreadme.md](TCN/out/outreadme.md) | 完整算法、窗口验证表、逐flight结果、9套PNG/SVG及日志索引 |
| [TCN+RLS/README.md](TCN+RLS/README.md) | 冻结TCN后的RLS搜索、名义95%预测区间和覆盖率摘要 |
| [TCN+RLS/improve.md](TCN+RLS/improve.md) | RLS与区间实现、口径修正和E组新增记录 |
| [TCN+RLS/out/outreadme.md](TCN+RLS/out/outreadme.md) | RLS递推、校准、分层/阶段覆盖率、7套PNG/SVG及产物清单 |

## 五组配置

| 组别 | 模型输入 | 卷积后及预测头中间激活 | 残差出口激活 |
| --- | --- | --- | --- |
| A | 原始10维 | SiLU | ReLU |
| B | 原始10维 | SiLU | SiLU |
| C | 原始10维+5维SMA，共15维 | SiLU | ReLU |
| D | 与C相同，共15维 | SiLU | SiLU |
| E | 与C/D相同，共15维 | LeakyReLU(0.05) | LeakyReLU(0.05) |

SMA仅对三维规划位置和二维风分量计算，固定10个0.2s采样点，按flight独立使用当前及历史值。C/D/E均保留原始10维，最终15维。五组回归输出层均为线性；E替换四次卷积后的隐藏激活、两次残差出口激活和预测头中间激活，负半轴斜率固定0.05。

## 训练与校准预算

- TCN：8个窗口候选（0.6、1、2、3、4、5、8、10s），每候选完整20轮，正式完整80轮；种子42，AdamW、SmoothL1Loss、batch2048、通道(32,32)。五组各自选择窗口，保留验证损失最低轮次。
- RLS：冻结对应TCN，7个遗忘因子×7个初始协方差×5个反馈窗口，共245个候选/组。反馈窗候选为1、2、5、10、20s；仅完整窗口结束后使用该窗口实际能耗更新。
- 数据：124个训练flight、29个验证flight、47个测试flight，划分互不交叉。验证flight排序后按序列索引交替分为15个RLS搜索flight和14个校准flight。
- 区间：名义95%，各组采用相同残差比率分位数校准策略、分别估计倍率，测试集冻结参数后评价。

## 当前结果

| 组别 | TCN窗口(s) | TCN任务能耗WAPE(%) | RLS累计能耗WAPE(%) | 任务前覆盖(%) | 反馈后覆盖(%) | 反馈后区间评分(Wh) |
| --- | --- | --- | --- | --- | --- | --- |
| A | 5.0000 | 3.7251 | 0.8811 | 55.3191 | 89.0677 | 5.3952 |
| B | 8.0000 | 3.5046 | 0.6688 | 61.7021 | 91.8936 | 5.1198 |
| C | 10.0000 | 2.9467 | 0.6130 | 63.8298 | 93.1106 | 4.3021 |
| D | 8.0000 | 2.6898 | 0.5580 | 97.8723 | 94.9670 | 4.7297 |
| E | 5.0000 | 2.9733 | 0.6949 | 93.6170 | 94.6370 | 4.0943 |

来源：[TCN汇总CSV](TCN/out/results/TCN-comparison_all_metrics.csv)、[TCN+RLS汇总CSV](TCN+RLS/out/results/TCN_RLS_all_metrics.csv)。任务前区间中心为TCN任务总能耗；RLS累计WAPE将各窗口反馈前预测相加；反馈后在线覆盖率评价已观测消耗加剩余预测的最终总能耗区间。

D的TCN与RLS任务能耗WAPE均为五组最低。E的点预测误差高于D，反馈后区间评分为五组最低，不能据此判定一种激活在全部指标上更优。E任务前覆盖44/47=93.62%，反馈后在线覆盖94.64%；在25%、50%、75%进度的任务级覆盖分别为91.49%、93.62%、100.00%。

## 复现与报告更新

在仓库根目录运行。完整复现五组会重新搜索并训练TCN；仅刷新说明文档使用report命令。

```powershell
python TCN-comparison/TCN/main.py all --device cuda
python TCN-comparison/TCN+RLS/main.py all --device cuda
```

仅复现E时，两条命令均追加 `--group E`，先完成TCN再执行TCN+RLS。

```powershell
python TCN-comparison/TCN/main.py report
python TCN-comparison/TCN+RLS/main.py report
```

仅重绘图表使用各工程的plot模式。TCN的report只刷新README结果区与输出报告；TCN+RLS的report生成完整README与输出报告，持久说明同步维护在各工程report.py。

## 存储与日志

| 工程 | 数据集与加工数据 | 模型与配方 | 本地输出 |
| --- | --- | --- | --- |
| TCN（DJI历史实验） | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/TCN/processed` | `D:/Python-files/Energy-prediction/model/TCN` | `TCN/out` |
| TCN+RLS（DJI历史实验） | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/TCN+RLS/processed` | `D:/Python-files/Energy-prediction/model/TCN+RLS` | `TCN+RLS/out` |
| TCN-AMOVFLY | `D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/TCN-comparison` | `D:/Python-files/Energy-prediction/model/TCN-AMOVFLY` | `TCN-AMOVFLY/out` |
| TCN+RLS-AMOVFLY | `D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/TCN-comparison` | `D:/Python-files/Energy-prediction/model/TCN+RLS-AMOVFLY` | `TCN+RLS-AMOVFLY/out` |

TCN终端日志为 `TCN/out/logs/TCN-comparison_terminal.log`，E逐轮文件为 `TCN-comparison_E_tuning_epochs.csv`（160条）和 `TCN-comparison_E_training.csv`（80条）。RLS终端日志为 `TCN+RLS/out/logs/TCN_RLS_terminal.log`，逐候选表为 `TCN+RLS/out/results/TCN_RLS_E_rls_tuning.csv`（245条）。两工程out/logs均有 `E_validation_tests.log` 和 `E_artifact_audit.json`；E的环境快照分别为 `TCN-comparison_runtime_E.json` 与 `TCN_RLS_runtime_E.json`。

数据和模型留在D盘，out沿用Git忽略策略，仓库提交保留代码、参数和Markdown摘要。TCN与RLS冻结权重的SHA-256一致；逐采样CSV可能存在约0.000031W的float32写出格式差异。覆盖率复算应使用 `pd.read_csv(path, float_precision="round_trip")`。

## 结果解释

本实验只有种子42，TCN窗口和RLS参数均按组独立选择，尚无固定窗口或多种子结果。规划位置、风和部分状态来自实飞代理；校准flight曾参与TCN验证选模，同flight在线轨迹点相关。区间为未来观测值的经验预测区间，不能解释为模型参数的置信区间或理论95%覆盖保证。
