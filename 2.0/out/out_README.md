# 四轴无人机飞行能耗预测实验 2.0：时间卷积网络（Temporal Convolutional Network, TCN）+ 递归最小二乘（Recursive Least Squares, RLS）输出说明

本文件与 `2.0/out` 当前生成的 CSV、JSON 和 PNG 一一对应，用于复核实验方案、读取结果，并和 `1.0/out` 做同口径对比。2.0 沿用 1.0 的数据源、特征工程、标签定义、flight 分组切分、随机种子、评价指标和输出命名；算法部分改为短时间窗因果 TCN 前向推理，再用 RLS 对功率输出进行实时在线校正。

## 1. 实验方案

数据来自 DJI Matrice 100 四轴无人机公开数据集。程序读取 `flights.csv`，完成数值转换、缺失值清理、R/H 航线筛选、功率标签构造和 28 维特征工程。功率标签定义为：

$$
P_i=\max\left(U_i I_i,0\right)
$$

其中：
$P_i$：第 $i$ 个采样点的真实瞬时功率，单位 W。
$U_i$：第 $i$ 个采样点的电池电压，单位 V。
$I_i$：第 $i$ 个采样点的非负放电电流，单位 A。

处理后共有 249210 条记录、198 次飞行。数据按 `flight` 整组随机划分，随机种子为 42：

| 集合 | 飞行数 | 记录数 |
| --- | ---: | ---: |
| train | 138 | 174653 |
| val | 30 | 34796 |
| test | 30 | 39761 |

测试集不参与网络参数更新；标准化统计量只由训练集计算，保存在 `out/model/scaler_2.0.json`。

## 2. 1.0 与 2.0 的对比边界

两版共同使用：

- 同一公开数据源、同一清洗规则和同一 28 维输入特征；
- 同一 `power_w` 目标、采样间隔积分公式、flight 分组切分和随机种子 42；
- 同一测试集指标：样本级功率平均绝对误差（Mean Absolute Error, MAE）、均方根误差（Root Mean Square Error, RMSE）、决定系数（Coefficient of Determination, R2）、平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）、加权绝对百分比误差（Weighted Absolute Percentage Error, WAPE），以及 flight 级能耗的 MAE、RMSE、R2、MAPE、WAPE；
- 同一功率区间和 flight 汇总表结构，同名图表的轴含义保持一致。

模型对比方法完全对应：1.0 在验证集比较 5 个多层感知机（Multilayer Perceptron, MLP）候选，2.0 在验证集比较 5 个短时间窗 TCN 候选。2.0 的每个候选同时计算 TCN 原始输出和 RLS 校正输出，`tcn_*` 字段表示校正前基线，不带 `tcn_` 前缀的字段表示最终结果。

## 3. 2.0 模型流程

每个 flight 按时间排序，窗口左侧不足时使用该 flight 的首个样本填充。因果 TCN 只读取当前及历史窗口：

$$
\hat{P}^{\mathrm{TCN}}_t=f_{\theta}\left(X_{t-L+1:t}\right)
$$

其中：
$\hat{P}^{\mathrm{TCN}}_t$：时刻 $t$ 的 TCN 功率预测，单位 W。
$f_{\theta}$：因果 TCN 及其可学习参数。
$X_{t-L+1:t}$：从当前时刻向前的输入窗口。
$L$：窗口采样步数。

TCN 由 4 个残差卷积块组成，通道为 `[64, 64, 64, 32]`，卷积核宽度为 3，膨胀率依次为 1、2、4、8，Dropout 为 0.08。卷积块使用 `GroupNorm`，模型头部增加当前时刻输入的 skip connection，使低功率和突变工况不只依赖卷积历史特征。训练损失为低功率和高功率加权的 Huber Loss，默认调参训练 20 轮、最终训练 80 轮。

RLS 将 TCN 输出映射到实测功率，使用归一化尺度 $s$ 构造回归向量：

$$
\boldsymbol{\phi}_t=\begin{bmatrix}1\\ \hat{P}^{\mathrm{TCN}}_t/s\end{bmatrix},\qquad z_t=P_t/s
$$

$$
\tilde{P}_t=s\boldsymbol{\phi}_t^{\mathsf T}\boldsymbol{\theta}_{t-1}
$$

$$
\mathbf{K}_t=\frac{\mathbf{C}_{t-1}\boldsymbol{\phi}_t}{\lambda+\boldsymbol{\phi}_t^{\mathsf T}\mathbf{C}_{t-1}\boldsymbol{\phi}_t},\qquad
\boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}+\mathbf{K}_t\left(z_t-\boldsymbol{\phi}_t^{\mathsf T}\boldsymbol{\theta}_{t-1}\right)
$$

$$
\mathbf{C}_t=\frac{\mathbf{C}_{t-1}-\mathbf{K}_t\boldsymbol{\phi}_t^{\mathsf T}\mathbf{C}_{t-1}}{\lambda}
$$

其中：
$\tilde{P}_t$：RLS 校正后的实时功率，单位 W。
$P_t$：当前已观测的实测功率，单位 W。
$s$：功率归一化尺度，保存在 `scaler_2.0.json`。
$\boldsymbol{\phi}_t$：由偏置项和归一化 TCN 功率组成的回归向量。
$z_t$：归一化实测功率。
$\boldsymbol{\theta}_t$：时刻 $t$ 的仿射校正参数，分别对应偏置和比例。
$\mathbf{K}_t$：RLS 增益向量。
$\mathbf{C}_t$：参数协方差矩阵。
$\lambda$：遗忘因子，本实验为 0.995。

程序按 flight 独立重置 RLS 状态，执行“先输出、后更新”：先用历史参数输出当前功率，再在当前实测功率到达后更新参数。部署输入没有 `power_w` 时，只执行 TCN 前向和已有 RLS 参数校正。

## 4. 时间窗口调参

`out/model/tuning_results_2.0.csv` 保存了 5 个候选时间窗。典型采样间隔约为 0.12 s，秒级窗口折算为下表采样步数。选择分数为：

$$
S=\mathrm{WAPE}_{\mathrm{flight}}+0.2\,\mathrm{WAPE}_{\mathrm{sample}}
$$

其中：
$S$：候选窗口选择分数，越小越优。
$\mathrm{WAPE}_{\mathrm{flight}}$：flight 级能耗 WAPE。
$\mathrm{WAPE}_{\mathrm{sample}}$：采样点功率 WAPE。

| 候选窗口 | 采样步数 | 验证 TCN 功率 WAPE (%) | 验证 RLS 功率 WAPE (%) | 验证 TCN 能耗 WAPE (%) | 验证 RLS 能耗 WAPE (%) | 选择分数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 s | 5 | 9.0892 | 8.7107 | 1.9527 | 1.2127 | 2.9549 |
| 1.0 s | 8 | 8.6550 | 8.1500 | 2.3278 | 0.7425 | 2.3725 |
| 1.5 s | 12 | 8.7691 | 8.2208 | 2.2372 | 0.7256 | 2.3698 |
| 2.0 s | 17 | 8.6453 | 8.2631 | 2.2357 | 0.7669 | 2.4195 |
| 3.0 s | 25 | 8.6406 | 8.1548 | 2.5450 | 0.8427 | 2.4737 |

综合分数最小的窗口为 **1.5 s、12 个采样步**。它与 1.0 s 的分数非常接近，但在 flight 能耗 WAPE 上略优，因此最终被选为 `best_candidate`。候选比较图如下。

![时间窗口候选比较](./figures/training/candidate_validation_wape.png)

![超参数排序](./figures/training/hyperparameter_ranking.png)

### 4.1 `best_*` 字段来源

`best_candidate`、`best_model_type`、`best_window_seconds`、`best_window_steps`、`best_channels`、`best_dropout`、`best_learning_rate` 和 `best_weight_decay` 都来自 `train.py` 中验证集最优候选 `best_params`，其值直接写入训练摘要和 `best_energy_tcn_rls_2.0.pt` 的 checkpoint。对应关系如下：

| 字段 | 具体来源 | 比较方式 |
| --- | --- | --- |
| `best_candidate` | `tuning_results_2.0.csv` 的 `candidate` 列和 `train.py` 返回的 `best_params["name"]` | 取 `selection_score` 最小的候选名 |
| `best_model_type` | `candidate_grid()` 中固定写死的 `tcn_rls` | 所有候选一致，不单独比较 |
| `best_window_seconds` | 最优候选行的 `window_seconds` | 在 0.6、1.0、1.5、2.0、3.0 s 中比较 |
| `best_window_steps` | `window_seconds_to_steps()` 由秒级窗口换算 | 由窗口秒数和 `sample_interval_seconds` 共同决定 |
| `best_channels` | `cfg.tcn_channels`，写入候选行后再随最优候选带回 | 所有候选一致，不单独比较 |
| `best_dropout` | `candidate_grid()` 中固定写死的 `0.08` | 所有候选一致，不单独比较 |
| `best_learning_rate` | `cfg.learning_rate` | 所有候选一致，不单独比较 |
| `best_weight_decay` | `cfg.weight_decay` | 所有候选一致，不单独比较 |

1. 先用 `candidate_grid()` 生成 5 个候选，只改变 `window_seconds`；
2. 每个候选都保持相同的 `model_type = tcn_rls`、`channels = [64, 64, 64, 32]`、`dropout = 0.08`、`learning_rate = 0.0003`、`weight_decay = 0.0001`、`kernel_size = 3` 和 `huber_delta = 0.65`；
3. 按候选窗口重建训练集和验证集序列，训练 TCN；
4. 在验证集上先算 TCN 基线，再用同一组初始 RLS 参数做校正；
5. 用 `selection_score = val_flight_energy_wape + 0.2 * val_sample_power_wape` 进行排序，分数最小者即为最优窗口。

`sample_interval_seconds` 则不是搜索出来的超参数，而是训练集估计得到的典型采样间隔，保存在 `scaler_2.0.json` 中；`window_steps` 由 `window_seconds / sample_interval_seconds` 四舍五入得到，所以 1.5 s 对应 12 步。

## 5. 当前测试集结果

以下数值直接读取 `out/model/evaluation_2.0.json`。RLS 校正和 TCN 基线使用同一测试集 39761 条记录、30 次飞行。

| 指标 | TCN 基线 | RLS 校正 | 单位 |
| --- | ---: | ---: | --- |
| 样本功率 MAE | 34.8041 | 33.1021 | W |
| 样本功率 RMSE | 63.9009 | 60.3974 | W |
| 样本功率 R2 | 0.9275 | 0.9353 | 无量纲 |
| 样本功率 WAPE | 8.8276 | 8.3959 | % |
| flight 能耗 MAE | 0.5548 | 0.3198 | Wh |
| flight 能耗 RMSE | 1.0484 | 0.9179 | Wh |
| flight 能耗 R2 | 0.9636 | 0.9721 | 无量纲 |
| flight 能耗 WAPE | 2.5478 | 1.4685 | % |

MAPE 分别为 550.8499%（RLS 样本功率）和 1.6960%（RLS flight 能耗）。功率接近 0 W 时分母很小，MAPE 会被少量低功率样本放大，因此功率主比较采用 MAE、R2 和 WAPE。

![总体评估指标](./figures/results/evaluation_metrics.png)

与 1.0 的最终输出对照如下：

| 版本 | 样本功率 MAE (W) | 样本功率 R2 | 样本功率 WAPE (%) | flight 能耗 MAE (Wh) | flight 能耗 R2 | flight 能耗 WAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 MLP | 36.4561 | 0.9196 | 9.2466 | 0.6298 | 0.9597 | 2.8922 |
| 2.0 TCN | 34.8041 | 0.9275 | 8.8276 | 0.5548 | 0.9636 | 2.5478 |
| 2.0 TCN + RLS | **33.1021** | **0.9353** | **8.3959** | **0.3198** | **0.9721** | **1.4685** |

相对 2.0 TCN 基线，RLS 将样本功率 MAE 降低 1.7020 W、WAPE 降低 0.4317 个百分点；flight 能耗 MAE 降低 0.2350 Wh、WAPE 降低 1.0792 个百分点。相对 1.0，2.0 的主要收益体现在样本功率和 flight 能耗的误差指标。

## 6. 训练过程

最终训练日志位于 `out/model/training_log_2.0.csv`，对应 1.5 s、12 个采样步的最优窗口。训练和验证损失均使用加权 Huber Loss，图中纵轴为标准化损失，不是 W 或 Wh。

![训练损失曲线](./figures/training/training_loss_history.png)

![学习率变化](./figures/training/learning_rate_schedule.png)

![最终损失曲线](./figures/training/loss_curve_2.0.png)

## 7. 功率分段与 flight 结果分析

`out/model/power_bin_evaluation_2.0.csv` 对最终 `predicted_power_w`（RLS 输出）按真实功率分段统计：

| 真实功率区间 | 样本数 | 真实均值 (W) | 预测均值 (W) | MAE (W) | WAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-50 W | 9549 | 1.2502 | 21.3224 | 20.8511 | 1667.7666 |
| 50-300 W | 704 | 162.6580 | 205.7373 | 98.1173 | 60.3213 |
| 300-450 W | 4471 | 409.9169 | 443.1479 | 46.5652 | 11.3597 |
| 450-600 W | 19985 | 520.0242 | 518.6631 | 29.4286 | 5.6591 |
| 600 W 以上 | 5052 | 658.0570 | 620.2663 | 49.8153 | 7.5701 |

450-600 W 是样本最多且最稳定的区间，预测均值与真实均值相差 1.3611 W。0-50 W 区间的真实均值接近 0，模型仍保留约 21 W 的输出，停机/着陆末段仍存在非零基线偏差；该区间不适合单独使用 MAPE 判断模型优劣。50-300 W 样本较少且动态状态比例高，误差仍然偏大，是后续改进的重点。

![功率区间 MAE](./figures/results/power_bin_mae.png)

![功率预测散点图](./figures/prediction/power_prediction_scatter.png)

![功率残差分布](./figures/prediction/power_residual_histogram.png)

散点图显示，改进后的 TCN 输出不再集中成旧版本的两条水平平台，预测点沿理想线形成连续分布；高功率段的离散程度较小，低功率段仍有明显的正偏差。残差直方图与分段表反映的是同一现象。

flight 能耗汇总保存在 `out/model/flight_energy_summary_2.0.csv`。大多数 R1 飞行的能耗误差低于 0.4 Wh，RLS 对不同飞行的偏置进行了逐段修正。`flight 224` 仍是最明显的异常案例：真实能耗约 17.657 Wh，RLS 预测约 22.461 Wh，误差约 4.804 Wh。该飞行属于 H 航线、编程速度为 0，末段出现近零功率，但输入特征中没有明确的着陆或停机状态标志，TCN 和 RLS 都会把部分末段状态延续为较高功率。这个案例说明，增加停机状态特征或单独建模着陆段比继续增大网络容量更有针对性。

![flight 能耗真实值与预测值](./figures/results/flight_energy_actual_vs_predicted.png)

![flight 能耗误差](./figures/results/flight_energy_error.png)

![flight 8 功率时序](./figures/prediction/flight_8_power_timeseries.png)

![flight 76 功率时序](./figures/prediction/flight_76_power_timeseries.png)

![flight 80 功率时序](./figures/prediction/flight_80_power_timeseries.png)

三张典型 flight 时序图分别展示无载荷、带载荷和不同速度/高度条件下的 TCN 基线与 RLS 输出。图中 `Predicted power` 为最终 RLS 结果，原始 TCN 结果可在预测 CSV 的 `tcn_predicted_power_w` 列中复核。

## 8. 输出文件与字段

- `data/processed_2.0/dataset_summary.json`：原始记录数、清洗后记录数、flight 数量和切分规模。
- `data/processed_2.0/uav_energy_features.csv`：清洗和特征工程后的建模数据。
- `model/scaler_2.0.json`：训练集标准化参数、特征列、典型采样间隔和功率尺度。
- `model/tuning_results_2.0.csv`：五个时间窗候选及验证集比较结果。
- `model/training_log_2.0.csv`：每轮训练损失、验证损失、学习率和窗口参数。
- `model/evaluation_2.0.json`、`evaluation_2.0.csv`：TCN 与 RLS 的双基线评估指标。
- `model/flight_energy_summary_2.0.csv`：逐 flight 的真实能耗、TCN 能耗、RLS 能耗和误差。
- `model/power_bin_evaluation_2.0.csv`：按真实功率区间统计最终预测误差。
- `model/uncertainty_calibration_2.0.npz`、`uncertainty_calibration_2.0.json`：验证集绝对残差校准数据和默认置信度摘要。
- `predictions/test_predictions_2.0.csv`：逐点保存输入、`tcn_predicted_power_w`、`rls_corrected_power_w`、`predicted_power_w`、校正量、能耗和 RLS 最终参数。
- `custom/custom_scenarios_2.0.csv`：自定义工况输入。
- `custom/custom_predictions_2.0.csv`：自定义工况的 TCN/RLS 预测序列。
- `custom/custom_prediction_summary_2.0.json`：自定义工况行数、平均/最大功率和累计能耗。

图表目录与文件的对应关系如下：

- `figures/training/`：训练损失、学习率、候选窗口验证 WAPE 和超参数排序。
- `figures/results/`：总体指标、flight 能耗对照、flight 能耗误差和功率分段 MAE。
- `figures/prediction/`：功率散点、残差直方图和 3 个典型 flight 功率时序。
- `figures/custom/`：自定义工况功率时序和累计能耗。

![自定义工况功率](./figures/custom/custom_power_timeseries.png)

![自定义工况累计能耗](./figures/custom/custom_cumulative_energy.png)

自定义工况结果只表示模型推演，不替代真实飞行测试。当前自定义输入共 16 个采样点，平均预测功率 372.0841 W，最大预测功率 492.2025 W，累计预测能耗 0.33074 Wh；95% 置信度下累计能耗区间为 0.23937~0.42211 Wh。

## 9. 终端输出字段说明

终端运行 `train`、`evaluate` 和 `visualize` 时，会分别以 `[train]`、`[evaluate]` 和 `[visualize]` 为标题打印结果字典。下面只说明各字段的含义，不固定记录具体数值；数值会随数据划分、训练过程和当前运行结果变化。

### 9.1 训练阶段 `[train]`

| 字段 | 含义 |
| --- | --- |
| `device` | 实际使用的训练设备，例如 CPU 或 CUDA GPU 编号。 |
| `cuda_device_name` | 实际参与训练的显卡型号。 |
| `best_candidate` | 验证集综合选择分数最低的候选模型名称。 |
| `best_model_type` | 最优模型类型，本实验为 `tcn_rls`，即 TCN 与 RLS 的组合。 |
| `best_window_seconds` | 最优输入时间窗口长度，单位为秒。 |
| `best_window_steps` | 最优时间窗口包含的离散采样步数。 |
| `sample_interval_seconds` | 相邻采样点的时间间隔，单位为秒。 |
| `best_channels` | 最优 TCN 各卷积层的通道数配置。 |
| `best_dropout` | TCN 的 Dropout 比例，用于抑制过拟合。 |
| `best_learning_rate` | 网络训练使用的学习率。 |
| `best_weight_decay` | 权重衰减系数，用于正则化网络参数。 |
| `rls_forgetting_factor` | RLS 遗忘因子，用于控制历史观测对当前参数更新的影响。 |
| `rls_initial_theta` | RLS 初始仿射校正参数向量，通常包含偏置项和比例项。 |
| `target_transform` | 目标功率值使用的变换方式；`none` 表示未做额外变换。 |
| `best_val_loss_standardized` | 最终训练过程中验证集上的最优标准化损失。 |
| `best_selection_score` | 时间窗口候选比较使用的综合选择分数，具体计算方式见第 4 节。 |
| `epochs_run` | 最终训练实际执行的轮数。 |
| `model_file` | 保存最优模型 checkpoint 的文件路径。 |
| `scaler_file` | 保存训练集标准化参数、特征信息和采样间隔的文件路径。 |

### 9.2 评估阶段 `[evaluate]`

评估结果分为两组：`sample_power_w` 表示采样点功率（单位 W）层面的指标，`flight_energy_wh` 表示单次飞行能耗（单位 Wh）层面的指标。字段名前缀为 `tcn_` 时，表示未经过 RLS 校正的 TCN 基线结果；不带该前缀时，表示 TCN 与 RLS 组合模型的最终结果。

字段末尾的指标缩写含义如下：`MAE` 为平均绝对误差，`RMSE` 为均方根误差，`R2` 为决定系数，`MAPE` 为平均绝对百分比误差，`WAPE` 为加权绝对百分比误差。一般情况下，MAE、RMSE、MAPE 和 WAPE 越小越好，R2 越接近 1 越好。

| 字段 | 含义 |
| --- | --- |
| `sample_power_w_mae`、`sample_power_w_rmse`、`sample_power_w_r2`、`sample_power_w_mape_percent`、`sample_power_w_wape_percent` | 最终 TCN + RLS 模型在采样点功率上的 MAE、RMSE、R2、MAPE 和 WAPE；百分比指标的单位为 `%`。 |
| `tcn_sample_power_w_mae`、`tcn_sample_power_w_rmse`、`tcn_sample_power_w_r2`、`tcn_sample_power_w_mape_percent`、`tcn_sample_power_w_wape_percent` | 纯 TCN 基线在采样点功率上的对应五项指标。 |
| `flight_energy_wh_mae`、`flight_energy_wh_rmse`、`flight_energy_wh_r2`、`flight_energy_wh_mape_percent`、`flight_energy_wh_wape_percent` | 最终 TCN + RLS 模型在 flight 级能耗上的对应五项指标；误差指标单位为 `Wh`，百分比指标单位为 `%`。 |
| `tcn_flight_energy_wh_mae`、`tcn_flight_energy_wh_rmse`、`tcn_flight_energy_wh_r2`、`tcn_flight_energy_wh_mape_percent`、`tcn_flight_energy_wh_wape_percent` | 纯 TCN 基线在 flight 级能耗上的对应五项指标。 |
| `test_rows` | 测试集中的采样点记录总数。 |
| `test_flights` | 测试集包含的独立 flight 数量。 |

功率接近 0 W 的样本会使 MAPE 的分母很小，导致该指标被放大，因此功率结果应结合 MAE、RMSE、R2 和 WAPE 一起判断。

### 9.3 评估文件字段

| 字段 | 含义 |
| --- | --- |
| `prediction_file` | 逐采样点测试预测结果文件路径。 |
| `flight_energy_summary_file` | 按 flight 汇总真实能耗、预测能耗和误差的文件路径。 |
| `power_bin_evaluation_file` | 按真实功率区间分箱统计评估结果的文件路径。 |

### 9.4 可视化阶段 `[visualize]`

| 字段 | 含义 |
| --- | --- |
| `figure_count` | 本次可视化流程实际生成的图像数量。 |
| `figures` | 本次生成的图像文件路径或文件名列表。 |

`figures` 中的文件按用途分为训练类、结果类和预测类。训练类包括训练损失、学习率、候选窗口验证 WAPE 和超参数排序图；结果类包括总体评价指标、flight 能耗对比、flight 能耗误差和功率分段 MAE 图；预测类包括功率预测散点图、功率残差直方图和典型 flight 功率时序图。

## 10. 复核和运行顺序

```powershell
python main.py train
python main.py evaluate
python main.py visualize
python main.py predict
```

复核时先对照两个版本的 `dataset_summary.json`，再对照 `tuning_results_1.0.csv` 与 `tuning_results_2.0.csv`，最后在相同指标和单位下比较 `evaluation_1.0.csv` 与 `evaluation_2.0.csv`。功率散点、分段表和 flight 汇总应结合阅读，以区分稳定巡航误差、低功率段误差和动态切换误差。

## 11. 预测置信度和上下阈值

预测文件同时给出点预测和置信区间。校准文件 `model/uncertainty_calibration_2.0.npz` 保存验证集上的绝对残差，`model/uncertainty_calibration_2.0.json` 保存样本数和默认半径。程序分别保存在线 RLS（输入含 `power_w`）和固定 RLS（部署或自定义工况无实测功率）的残差，避免把两种运行状态混用。

给定置信度 $c$，从对应校准残差计算有限样本保序分位数 $q_c$：

$$
q_c=\operatorname{sort}(e)_{\lceil(n+1)c\rceil},\qquad e_i=|P_i-\hat{P}_i|
$$

其中：
$c$：用户指定的置信度，例如 `0.90`、`0.95` 或 `0.99`。
$e_i$：验证集第 $i$ 个样本的功率绝对残差，单位 W。
$n$：校准样本数。
$q_c$：功率区间半径，单位 W。

逐采样点区间为：

$$
P_t^{\mathrm{low}}=\max(\hat{P}_t-q_c,0),\qquad P_t^{\mathrm{up}}=\hat{P}_t+q_c
$$

能耗和累计能耗上下限使用同一功率边界积分：

$$
E_t^{\mathrm{low/up}}=P_t^{\mathrm{low/up}}\frac{\Delta t_t}{3600},\qquad
C_t^{\mathrm{low/up}}=\sum_{j\le t}E_j^{\mathrm{low/up}}
$$

其中：
$\hat{P}_t$：TCN + RLS 点预测功率，单位 W。
$P_t^{\mathrm{low}}$、$P_t^{\mathrm{up}}$：置信度下的功率下限和上限，单位 W。
$E_t^{\mathrm{low/up}}$：单个采样间隔的能耗下限和上限，单位 Wh。
$C_t^{\mathrm{low/up}}$：截至时刻 $t$ 的累计能耗下限和上限，单位 Wh。
$\Delta t_t$：当前采样间隔，单位 s。

完整预测：

```powershell
python main.py predict --confidence 0.95
```

预测完成后即时改变置信度，无需重新加载模型或执行 TCN：

```powershell
python main.py interval --confidence 0.90
python main.py interval --confidence 0.99
```

`interval` 默认原地更新 `out/predictions/test_predictions_2.0.csv`。也可以通过 `--input-csv` 和 `--output-csv` 指定其他预测文件。输出字段包括 `predicted_power_w`、`predicted_power_lower_w`、`predicted_power_upper_w`、`cumulative_energy_wh`、`cumulative_energy_lower_wh` 和 `cumulative_energy_upper_wh`。评估 JSON 另外记录 `sample_power_interval_coverage_percent`、`flight_energy_interval_coverage_percent`、区间半径和平均区间宽度。这里的累计区间是逐点边界的累加结果，不等同于整条轨迹同时覆盖率保证。
