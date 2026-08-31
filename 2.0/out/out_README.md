# 四轴无人机飞行能耗预测实验 2.0：TCN + RLS 输出说明

本文件与 `2.0/out` 中当前生成的 CSV、JSON 和 PNG 一一对应，用于复核实验方案、读取结果并与 `1.0/out` 对比。2.0 保持 1.0 的数据源、特征工程、标签定义、flight 分组切分、随机种子、训练损失、优化器、评估指标和图表命名不变；仅将 1.0 的 MLP 候选替换为因果 TCN，并增加 RLS 在线校正。这样可以把模型结构变化与数据和评价口径变化区分开。

## 1. 实验方案

数据仍来自 DJI Matrice 100 四轴无人机公开数据集。程序读取 `flights.csv`，完成数值转换、缺失值清理、R/H 航线筛选、功率标签构造和 28 维特征工程。功率标签定义为：

$$
P_i=\max\left(U_i I_i,0\right)
$$

其中：
$P_i$：第 $i$ 个采样点的真实瞬时功率，单位 W。
$U_i$：第 $i$ 个采样点的电池电压，单位 V。
$I_i$：第 $i$ 个采样点的非负放电电流，单位 A。

处理后的 249210 条记录包含 198 次飞行，按 `flight` 整组随机划分，随机种子为 42：

| 集合 | 飞行数 | 记录数 |
| --- | ---: | ---: |
| train | 138 | 174653 |
| val | 30 | 34796 |
| test | 30 | 39761 |

测试集不参与网络参数更新。标准化统计量只由训练集计算，保存在 `model/scaler_2.0.json`。

## 2. 1.0 与 2.0 的对比边界

两版实验共同使用：

- 同一公开数据源和同一清洗规则；
- 同一 28 维输入特征、`power_w` 目标和能耗积分公式；
- 同一 flight 分组切分、随机种子 42 和 train/val/test 比例；
- 同一 Huber Loss、AdamW、学习率调度、早停和 CUDA 要求；
- 同一测试集指标：样本级功率 MAE、RMSE、R2、MAPE、WAPE，以及 flight 级能耗 MAE、RMSE、R2、MAPE、WAPE；
- 同一功率区间和 flight 汇总表结构，以及同名图表的轴含义。

两版的模型对比方法为：1.0 在验证集上比较 5 个 MLP 候选；2.0 在验证集上比较 5 个短时间窗 TCN 候选。2.0 每个窗口同时保存 TCN 原始结果和 RLS 校正结果，`tcn_*` 字段是校正前基线，不带前缀的字段是最终 RLS 输出。比较 1.0 与 2.0 时，应使用相同指标、相同测试集和相同单位，不应把标准化损失与 W 或 Wh 指标直接比较。

## 3. 2.0 模型流程

对每个 flight 按时间排序，在窗口左侧不足时使用该 flight 的首个样本填充。因果 TCN 只读取当前及历史窗口：

$$
\hat{P}^{\mathrm{TCN}}_t=f_{\theta}\left(X_{t-L+1:t}\right)
$$

其中：
$\hat{P}^{\mathrm{TCN}}_t$：时刻 $t$ 的 TCN 功率预测，单位 W。
$f_{\theta}$：因果 TCN 及其可学习参数。
$X_{t-L+1:t}$：当前时刻向前的输入窗口。
$L$：窗口采样步数。

RLS 在当前预测输出后更新，不使用未来观测：

$$
\tilde{P}_t=\hat{P}^{\mathrm{TCN}}_t+\boldsymbol{\phi}_t^{\mathsf{T}}\boldsymbol{\theta}_{t-1}
$$

$$
\boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}+\mathbf{K}_t\left(P_t-\hat{P}^{\mathrm{TCN}}_t-\boldsymbol{\phi}_t^{\mathsf{T}}\boldsymbol{\theta}_{t-1}\right)
$$

其中：
$\tilde{P}_t$：RLS 校正后的实时功率，单位 W。
$P_t$：当前采样点实测功率，单位 W。
$\boldsymbol{\phi}_t$：由偏置项和 TCN 功率组成的 RLS 回归向量。
$\boldsymbol{\theta}_t$：时刻 $t$ 的校正参数。
$\mathbf{K}_t$：RLS 增益向量。

输入含 `power_w` 时，程序执行“先输出、后更新”；部署输入没有 `power_w` 时，只执行 TCN 前向和已有 RLS 参数校正。

## 4. 时间窗口调参

当前输出的 `tuning_results_2.0.csv` 包含 5 个候选窗口：

| 候选窗口 | 采样步数 | 验证 TCN 功率 WAPE (%) | 验证 RLS 功率 WAPE (%) | 验证 TCN 能耗 WAPE (%) | 验证 RLS 能耗 WAPE (%) | 选择分数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 s | 5 | 44.7062 | 41.2298 | 11.9999 | 13.2518 | 21.4977 |
| 1.0 s | 8 | 48.6269 | 42.3288 | 11.8336 | 14.7531 | 23.2189 |
| 1.5 s | 12 | 41.1540 | 39.6627 | 10.7885 | 21.9997 | 29.9322 |
| 2.0 s | 17 | 35.8790 | 36.2785 | 11.1230 | 18.9697 | 26.2254 |
| 3.0 s | 25 | 35.5503 | 30.9180 | 9.1632 | 18.4697 | 24.6533 |

选择分数为：

$$
S=\mathrm{WAPE}_{\mathrm{flight}}+0.2\,\mathrm{WAPE}_{\mathrm{sample}}
$$

其中：
$S$：候选窗口选择分数，越小越优。
$\mathrm{WAPE}_{\mathrm{flight}}$：flight 级能耗 WAPE。
$\mathrm{WAPE}_{\mathrm{sample}}$：采样点功率 WAPE。

当前最优窗口为 0.6 s，即 5 个采样步。表中结果来自最小化验证训练，用于确认调参和输出链路；正式精度实验应使用默认 `epochs=55`、`tune_epochs=14` 重新运行。

## 5. 当前测试集结果

`model/evaluation_2.0.json` 和 `evaluation_2.0.csv` 的当前结果如下。TCN 基线与 RLS 校正使用同一测试集：

| 指标 | TCN 基线 | RLS 校正 | 单位 |
| --- | ---: | ---: | --- |
| 样本功率 MAE | 183.8165 | 156.6843 | W |
| 样本功率 RMSE | 217.1824 | 201.7209 | W |
| 样本功率 R2 | 0.1628 | 0.2777 | 无量纲 |
| 样本功率 WAPE | 46.6226 | 39.7409 | % |
| flight 能耗 MAE | 2.0139 | 2.8372 | Wh |
| flight 能耗 RMSE | 2.7398 | 3.5761 | Wh |
| flight 能耗 R2 | 0.7512 | 0.5762 | 无量纲 |
| flight 能耗 WAPE | 9.2484 | 13.0291 | % |

当前最小训练结果中，RLS 降低了逐点功率误差，但 flight 能耗指标高于 TCN 基线。该现象应作为本次输出的实际结果记录，不应通过删除 `tcn_*` 字段掩盖。完整训练轮数增加后需重新读取同一表格判断窗口和校正效果。

## 6. 文件与图表对应关系

- `data/processed_2.0/dataset_summary.json`：数据规模、flight 数量和切分结果。
- `model/scaler_2.0.json`：训练集标准化参数、典型采样间隔和 RLS 功率尺度。
- `model/tuning_results_2.0.csv`：五个时间窗候选及验证集比较结果。
- `model/training_log_2.0.csv`：最终窗口的训练损失、验证损失、学习率和窗口参数。
- `model/evaluation_2.0.json`、`evaluation_2.0.csv`：TCN 与 RLS 双指标结果。
- `model/flight_energy_summary_2.0.csv`：逐 flight 的真实能耗、TCN 能耗、RLS 能耗和误差。
- `model/power_bin_evaluation_2.0.csv`：按真实功率区间统计最终 `predicted_power_w` 的误差。
- `predictions/test_predictions_2.0.csv`：逐点保存 `tcn_predicted_power_w`、`rls_corrected_power_w`、`predicted_power_w`、校正量以及对应区间能耗。
- `figures/training/*`：损失、学习率、窗口排序和验证 WAPE 图。
- `figures/results/*`：总体指标、flight 能耗和功率区间图。
- `figures/prediction/*`：测试集散点、残差和 3 个典型 flight 时序图。
- `custom/*` 与 `figures/custom/*`：自定义工况输入、TCN/RLS 预测和累计能耗图。

图表文件名与 1.0 保持一致；2.0 的 `Predicted power` 表示 RLS 校正结果，原始 TCN 结果在 CSV 的 `tcn_predicted_power_w` 列中。

## 7. 推荐对比顺序

1. 对照两个版本的 `dataset_summary.json`，确认数据边界一致。
2. 对照 `tuning_results_1.0.csv` 与 `tuning_results_2.0.csv`，分别读取 MLP 候选和时间窗候选。
3. 在相同指标和单位下，对照 `evaluation_1.0.csv` 与 `evaluation_2.0.csv`。
4. 查看 2.0 的 `tcn_*` 和最终字段，单独判断 RLS 是否改善逐点功率和 flight 能耗。
5. 查看功率分段表、flight 汇总表和时序图，区分稳定巡航误差、低功率段误差和动态切换误差。
6. 自定义工况结果只表示模型推演，不替代真实飞行测试。
