# 四轴无人机飞行能耗预测模型 3.0

## 1. 版本作用

3.0 保留 2.1 的 TCN、按 flight 切分、预测区间和自定义工况功能，并将 D 盘原始记录先按 flight 重采样到统一的 0.12 s 时间网格。模型输入在原 22 维工况特征上加入 `dt_seconds`，成为 23 维时间感知输入；TCN 仍输出逐采样点功率，但训练损失同时使用完整 1 秒窗口的能量差更新网络权重。RLS 根据 TCN 功率判断停机/飞行状态，在状态切换时恢复偏置 0、缩放 1，并在窗口结束后用真实能量更新下一窗口的校正参数。

## 2. 运行配置

在 `3.0` 目录执行：

```bash
python main.py prepare --force-prepare
python main.py tune-tcn
python main.py train-fixed
python main.py tune-rls
python main.py test
```

`tune-tcn` 只搜索 TCN；`train-fixed` 在终端打印当前参数并跳过搜索完成正式训练；`tune-rls` 只加载现有 TCN 搜索 RLS；`test` 只用固定模型与 RLS 参数测试和制图。本次使用 CUDA，最多训练 80 epoch，批大小 2048，学习率 `3e-4`，权重衰减 `1e-4`，TCN 通道 `[64,64,64,32]`，卷积核 3，Dropout 0.08。

## 3. 数据处理

固定采样原表位于 `D:/Python-files/Energy-prediction/data/dji_matrice_100`，旧不规则数据备份于同级 `dji_matrice_100_irregular_backup`。项目外一次性脚本已将 257896 条、209 个 flight 的记录重采样为 322904 条，典型间隔为 0.12 s；整理完成后脚本不保留在项目中。项目读取新原表后保留 R1 航线，得到 284758 条有效记录、182 个 flight。train/val/test 分别为 198469/41052/45237 条，对应 126/28/28 个 flight。

项目仍根据相邻时间戳显式计算 `dt_seconds`，而不假设数组下标等价于时间；除 flight 尾点外，其值基本为 0.12 s。功率由电压和非负放电电流计算，区间能量为 `power_w * dt_seconds / 3600`。每秒窗口使用 `floor(time)` 编号，并按真实间隔积分。

主要数据文件：

- `out/data/processed_3.0/uav_energy_features.csv`：逐采样点清洗和特征表。
- `out/data/processed_3.0/second_energy_3.0.csv`：每个 flight 的每个秒级窗口真实能量表。
- `out/data/processed_3.0/train.csv`、`val.csv`、`test.csv`：按完整 flight 切分的数据。
- `out/data/processed_3.0/feature_metadata.json`：输入特征和字段说明。
- `out/data/processed_3.0/dataset_summary.json`：数据规模和版本摘要。

## 4. TCN 训练和窗口选择

TCN 不把真实每秒能量作为前向输入。输入为 2.1 的 22 维工况特征加 `dt_seconds`，时间门控层根据每步真实间隔调节特征后再进入因果卷积。TCN 输出仍是每个采样点的 `power_w`。时间窗候选为 0.25、0.5、1、2、4、6、8、10、12、16 秒；本次沿用已选的 `2.0 s / 17 步`，未重新搜索十个候选。

训练批次按 `flight + floor(time)` 保持完整秒窗口，不能把同一窗口拆到两个批次。TCN 先输出各采样点功率，再按 `dt_seconds` 积分，联合损失为：

$$L=L_{power}+0.2L_{energy}$$

其中：

$L_{power}$：按功率区间加权的逐点 Huber 损失。

$L_{energy}$：完整秒窗口预测能量和真实能量之间的 Huber 损失；其梯度通过积分运算回传到该窗口内全部 TCN 功率输出。

窗口选择同时考虑逐点功率、秒级能量和 flight 总能量：

$$S_{TCN}=WAPE_{second}+0.2WAPE_{sample}+0.5WAPE_{flight}$$

其中：

$S_{TCN}$：TCN 窗口选择价值函数，越小越好。

$WAPE_{second}$：验证集秒级能量 WAPE。

$WAPE_{sample}$：验证集采样点功率 WAPE。

$WAPE_{flight}$：验证集按 flight 汇总能量 WAPE。

历史调参选定窗口为 `2.0 s / 17 步`，选择分数为 `7.38557800`。新数据与新损失下使用该窗口正式训练，63 轮早停，最佳权重出现在第 53 轮，验证联合损失为 `0.01641463`。完整重新运行 `tune-tcn` 后，十个候选会按新机制更新；当前结果不能解释为新机制下十个窗口的再次比较。

## 5. 秒级能量和 RLS 数据流

每个 flight 开始时 RLS 均从中性参数 `[0,1]` 和初始协方差开始。系统以当前秒窗口 TCN 平均功率是否达到 50 W 判断状态，0 表示停机，1 表示飞行；状态改变时立即恢复偏置 0、缩放 1。当前窗口内 RLS 使用窗口开始前的参数逐点校正，窗口内不更新。

窗口结束后，程序使用每个采样点的实际 `dt_seconds` 积分 TCN 功率、RLS 校正功率和真实功率，分别得到 `tcn_second_energy_wh`、`predicted_second_energy_wh` 和 `actual_second_energy_wh`。真实能量在窗口结束前不可用，因此不会参与当前窗口的功率校正。

RLS 将窗口预测能量和真实能量换算为窗口平均功率，再进行仿射回归：

$$\bar{P}_{f,k}^{true}\approx\theta_{0,f,k}+\theta_{1,f,k}\bar{P}_{f,k}^{pred}$$

其中：

$\bar{P}_{f,k}^{true}$：真实能量除以窗口实际时长得到的真实平均功率。

$\bar{P}_{f,k}^{pred}$：当前窗口预测能量除以窗口实际时长得到的预测平均功率。

$\theta_{0,f,k}$：RLS 的功率偏置参数。

$\theta_{1,f,k}$：RLS 的功率缩放参数。

$f$：flight 编号；每个 flight 独立重置 RLS。

$k$：当前秒级窗口编号。

更新完成后，`theta_{f,k+1}` 只用于下一个窗口。最后一个窗口更新的参数不会反向修改已经输出的结果。

## 6. RLS 超参数搜索

搜索固定最优 TCN 窗口的验证集预测，不重复训练 TCN。72 组候选均按以下顺序运行：按 flight 排序并重置 RLS；逐窗口预测；窗口结束后积分能量；使用当前窗口真实能量更新；将新参数传给下一个窗口。

每个 flight 先计算独立价值函数，再对 28 个验证 flight 等权平均，并增加 95 分位 flight 能量误差和 flight 间标准差惩罚：

$$S_{RLS,f}=WAPE_{second,f}+0.2WAPE_{sample,f}+0.5WAPE_{flight,f}$$

$$S_{RLS}=\frac{1}{N_{flight}}\sum_f S_{RLS,f}+0.25Q_{0.95}(WAPE_{flight,f})+0.1\sigma(S_{RLS,f})$$

其中：

$S_{RLS,f}$：单个 flight 的 RLS 价值函数。

$S_{RLS}$：所有验证 flight 的汇总价值函数。

$N_{flight}$：验证集 flight 数量，本次为 28。

$WAPE_{second,f}$：单个 flight 的秒级能量误差。

$WAPE_{sample,f}$：单个 flight 的逐点功率误差。

$WAPE_{flight,f}$：单个 flight 总能量误差。

本次网格为 `7 × 7 × 4 = 196` 组：遗忘因子 `0.86、0.88、0.90、0.91、0.92、0.94、0.96`，初始协方差 `0.01、0.025、0.05、0.1、0.25、0.5、1.0`，预热为 `0、1、2、3` 个完整窗口。最优为遗忘因子 `0.91`、初始协方差 `0.25`、预热 `0` 窗口，价值函数 `6.12551123`。遗忘因子和协方差均处于区间内部；预热 0 表示立即更新，是有实际意义的离散选择，不属于连续范围碰壁。

## 7. 最终训练、测试和结果

固定 `2.0 s / 17 步` 和上述 RLS 参数后，TCN 最多训练 80 epoch，并在第 63 轮早停，最佳权重来自第 53 轮。测试集只在参数全部固定后运行一次，共 45237 条记录、28 个 flight。

| 指标 | TCN | TCN + 秒级能量 RLS | 单位 |
|---|---:|---:|---|
| 样本功率 MAE | 31.3320 | 27.7797 | W |
| 样本功率 RMSE | 46.1691 | 43.0377 | W |
| 样本功率 R2 | 0.9574 | 0.9630 | 无量纲 |
| 样本功率 WAPE | 7.7114 | 6.8371 | % |
| flight 能耗 MAE | 0.6119 | 0.0674 | Wh |
| flight 能耗 RMSE | 0.8226 | 0.1135 | Wh |
| flight 能耗 R2 | 0.9693 | 0.9994 | 无量纲 |
| flight 能耗 WAPE | 2.7966 | 0.3079 | % |

RLS 后样本功率 WAPE 相对下降 `11.34%`，flight 能耗 WAPE 相对下降 `88.99%`。最差 flight 87 的 RLS 总能量绝对百分比误差为 `1.6675%`，未重现旧实验 flight 233 的明显过度校正。95% 功率区间实际覆盖率为 `93.6181%`，平均宽度 `145.2989 W`；flight 能耗区间覆盖率为 `100%`。

## 8. 结果图展示与分析

本节嵌入本次 3.0 实际运行生成的全部结果图。训练图用于检查候选窗口和最终模型的选择过程，评估图用于检查测试集整体误差，预测图用于观察逐点功率、每秒能量以及典型 flight 的时间变化。

### 8.1 TCN 窗口候选和训练过程

![TCN 候选验证集 WAPE](./out/figures/training/candidate_validation_wape.png)

图中比较了 10 个 TCN 时间窗候选的验证集误差。2.0 s 候选的综合选择分数为 `7.38557800`，低于其余候选，因此最终采用 2.0 s、17 步输入窗口。候选并非随时间窗变长而单调改善，说明扩大历史范围后，新增历史信息与模型训练误差之间存在权衡。

![TCN 和 RLS 超参数排序](./out/figures/training/hyperparameter_ranking.png)

该图保留历史 TCN 窗口排序。新 RLS 搜索选中遗忘因子 `0.91`、初始协方差 `0.25`、预热 0 个完整窗口，含尾部惩罚的汇总分数为 `6.12551123`。

![学习率变化](./out/figures/training/learning_rate_schedule.png)

学习率曲线反映训练过程中优化步长的变化。候选模型和最终模型使用同一套学习率调度，便于比较不同时间窗，而不会把学习率变化误认为时间窗带来的差异。

![3.0 损失曲线](./out/figures/training/loss_curve_3.0.png)

最终训练运行 63 个 epoch 后早停，最佳验证联合损失为 `0.01641463`，出现在第 53 个 epoch。

### 8.2 测试集总体结果

![总体评估指标](./out/figures/results/evaluation_metrics.png)

总体评估图同时比较纯 TCN 与 RLS 校正结果。RLS 后样本功率 WAPE 由 `7.7114%` 降至 `6.8371%`，flight 能耗 WAPE 由 `2.7966%` 降至 `0.3079%`，flight 能耗 R2 提升至 `0.9994`。

### 8.3 功率预测图

![功率分箱 MAE](./out/figures/results/power_bin_mae.png)

功率分箱图按真实功率区间统计 MAE，用来检查误差是否集中在某一功率范围。低功率段的相对误差容易被接近 0 的分母放大，因此应结合 MAE、WAPE 和散点图共同判断，不能仅凭低功率段的 MAPE 排名评价模型。

![功率预测散点图](./out/figures/prediction/power_prediction_scatter.png)

散点图以真实功率为横轴、RLS 功率为纵轴，测试集 R2 为 `0.9630`、RMSE 为 `43.0377 W`。

![功率残差直方图](./out/figures/prediction/power_residual_histogram.png)

残差主体集中在 0 附近，两侧仍有快速变化形成的尾部。RLS 后样本功率 MAE 为 `27.7797 W`、RMSE 为 `43.0377 W`。

### 8.4 flight 总能量结果

![flight 总能量真实值与预测值](./out/figures/results/flight_energy_actual_vs_predicted.png)

图中逐个比较测试 flight 的真实总能量、TCN 汇总能量和 RLS 校正能量。RLS 曲线更贴近真实值，且 flight 级 R2 达到 `0.9947`。这与算法的监督方式一致：RLS 在每个完整 1 秒窗口结束后利用真实能量更新参数，更新后的参数只传递给下一窗口，因此能逐步修正累计能量偏差。

![flight 总能量误差](./out/figures/results/flight_energy_error.png)

误差图展示每个 flight 的总能量误差及校正前后差异。不同 flight 的误差并不完全一致，这是飞行工况、持续时间和功率变化幅度差异共同造成的；评价时采用全部 28 个测试 flight 汇总的 MAE、RMSE、R2 和 WAPE，而不是只选取误差较小的 flight。

### 8.5 典型 flight 的逐点功率曲线

![flight 18 功率时序](./out/figures/prediction/flight_18_power_timeseries.png)

flight 18 的曲线用于观察一个测试 flight 内部的逐采样点功率跟踪情况。TCN 输出保留了原始采样时间轴，RLS 校正曲线在窗口边界后逐步调整，参数更新不会回写已经结束的窗口。

![flight 23 功率时序](./out/figures/prediction/flight_23_power_timeseries.png)

flight 23 展示另一种功率变化过程。曲线之间的局部差异说明 RLS 并非固定比例缩放，而是根据已经结束窗口的能量误差更新仿射参数，再作用于后续窗口。

![flight 83 功率时序](./out/figures/prediction/flight_83_power_timeseries.png)

flight 83 用于补充不同 flight 的工况对比。三个典型 flight 的曲线共同说明：TCN 负责逐点功率预测，RLS 负责利用秒级反馈修正后续窗口，二者输出应分别保留，便于区分基础预测误差和在线校正效果。

### 8.6 每秒能量图

![每秒能量预测散点图](./out/figures/prediction/second_energy_prediction_scatter.png)

该散点图以真实每秒能量为横轴、RLS 校正后的每秒预测能量为纵轴，并以理想线作为参照。点云越接近理想线，表示窗口级能量积分越准确。每秒能量使用不固定采样间隔 `dt_seconds` 加权计算，因而图中反映的是实际时间积分结果，而不是简单的样本数量求和。

![每秒能量残差直方图](./out/figures/prediction/second_energy_residual_histogram.png)

每秒能量残差主要围绕 0 集中。秒级积分平滑了部分采样噪声，但状态突变仍形成尾部；测试集 flight 能耗 WAPE 为 `0.3079%`。

![flight 18 每秒能量时序](./out/figures/prediction/flight_18_second_energy_timeseries.png)

flight 18 的每秒能量曲线把逐点功率压缩为 1 秒窗口积分结果。真实能量只能在窗口结束后形成，因此当前窗口的真实值不会参与当前窗口校正，RLS 更新从下一窗口开始体现。

![flight 23 每秒能量时序](./out/figures/prediction/flight_23_second_energy_timeseries.png)

flight 23 的窗口曲线用于观察不同能量水平下的跟踪效果。TCN 原始能量和 RLS 校正能量同时保留，能够区分 TCN 的积分误差与在线仿射修正带来的变化。

![flight 83 每秒能量时序](./out/figures/prediction/flight_83_second_energy_timeseries.png)

flight 83 的结果补充了典型 flight 对比。每秒能量图比逐点功率图更适合检查能量守恒和窗口级反馈效果，最终累计能量则由这些窗口能量逐段累加得到。

### 8.7 自定义展示航线

本次在 `out/custom` 中生成一条固定的 R1 展示航线，编号为 `999001`，总时长 180 s，输出间隔 1 s。航线剖面由三个阶段组成：0～约 27 s 起飞，约 27～153 s 巡航，约 153～180 s 降落。输入条件为风速 4 m/s、风向 0°、巡航速度 8 m/s、载荷 250 g、飞行高度 50 m。

![自定义展示航线功率](./out/figures/custom/custom_power_timeseries.png)

图中展示该航线的逐采样点功率预测和 95% 预测区间。蓝色曲线是经过 checkpoint 中 RLS 仿射参数校正后的预测功率，阴影是功率区间。起飞和降落阶段的垂直速度变化会改变实际速度、相对空速和热负载代理量，因此预测功率随航线阶段发生变化。

![自定义展示航线累计能耗](./out/figures/custom/custom_cumulative_energy.png)

图中展示逐点预测功率按 `dt_seconds` 积分后的累计能耗及 95% 区间。本次共 181 个采样点，平均预测功率为 `506.1544 W`，最大预测功率为 `577.6692 W`，累计预测能耗为 `25.4483 Wh`，95% 区间为 `20.8112～30.0854 Wh`。

自定义航线的处理顺序与真实测试一致：先根据工况生成 23 维输入特征；TCN 输出逐采样点功率；再使用固定的 RLS 参数进行功率校正；最后按 `dt_seconds` 计算能量并累加。该航线没有真实功率，所以不能计算误差指标，结果不替代真实飞行测试。

自定义产物如下：`custom_scenarios_3.0.csv` 保存航线输入，`custom_predictions_3.0.csv` 保存 TCN/RLS 预测和能量字段，`custom_prediction_summary_3.0.json` 保存摘要指标；对应图片位于 `out/figures/custom/`。

## 9. 输出文件

测试集 RLS 轨迹包含 5442 个秒窗口和 61 次状态切换。偏置归一化系数均值 `0.00697`，缩放系数均值 `0.97398`；少量窗口触及裁剪边界，说明后续仍可研究连续越界计数或更新平滑约束。

![flight 18 RLS 参数轨迹](./out/rls/flight_18_rls_parameter_trace.png)

红色竖线表示状态切换，切换后参数恢复为 `[0,1]`。

- `out/model/tuning_results_3.0.csv`：10 个 TCN 窗口的 80 轮训练结果和能量选择指标。
- `out/rls/rls_tuning_results_3.0.csv`：196 组 RLS 候选及尾部惩罚价值函数。
- `out/rls/rls_parameter_trace_3.0.csv`：逐秒窗口参数轨迹。
- `out/rls/rls_parameter_statistics_3.0.csv`：逐 flight 参数统计。
- `out/model/training_log_3.0.csv`：TCN 候选和最终训练日志。
- `out/model/evaluation_3.0.json/csv`：测试集总体指标。
- `out/model/flight_energy_summary_3.0.csv`：按 flight 汇总的真实、TCN 和 RLS 能量。
- `out/model/second_energy_evaluation_3.0.csv`：按 flight 和秒级窗口的能量对比及误差。
- `out/predictions/test_predictions_3.0.csv`：逐采样点功率、每秒能量、累计能量和预测区间。
- `out/figures/prediction/flight_*_power_timeseries.png`：逐点功率对比图。
- `out/figures/prediction/flight_*_second_energy_timeseries.png`：每秒能量真实值与预测值对比图。
- `out/figures/prediction/second_energy_prediction_scatter.png`：真实每秒能量与预测每秒能量的散点图及理想线。
- `out/figures/prediction/second_energy_residual_histogram.png`：每秒能量残差直方图。
- `out/figures/results/flight_energy_actual_vs_predicted.png`：flight 总能量对比图。
- `out/figures/results/flight_energy_error.png`：flight 总能量误差图。
- `out/custom/custom_scenarios_3.0.csv`：固定参数生成的 R1 自定义展示航线输入。
- `out/custom/custom_predictions_3.0.csv`：自定义航线的 TCN 功率、RLS 校正功率、预测能量和累计能量。
- `out/custom/custom_prediction_summary_3.0.json`：自定义航线预测摘要和置信区间。
- `out/figures/custom/custom_power_timeseries.png`：自定义航线功率及预测区间。
- `out/figures/custom/custom_cumulative_energy.png`：自定义航线累计能耗及预测区间。

## 10. 文件调用关系

`main.py` 调度全流程；`config.py` 统一管理路径、参数和版本化文件名；`data_utils.py` 下载、清理数据、计算不规则采样间隔并生成秒级能量表；`model.py` 定义因果 TCN 和窗口能量监督 RLS；`train.py` 执行 TCN 窗口搜索、RLS 搜索和最终训练；`predict.py` 生成逐点、秒级和累计能量字段；`evaluate.py` 生成测试指标及两个能量汇总文件；`visualize.py` 生成训练、功率、秒级能量和 flight 能量图表；`uncertainty.py` 保留 2.1 的预测区间校准功能。

## 11. 3.0 需求与数据流记录

3.0 的约束来自前期设计讨论并全部落实：真实每秒能量只能在窗口结束时得到；当前窗口不能使用当前窗口真实能量更新 RLS；更新后的参数只传给下一个窗口；TCN 输入不增加真实每秒能量，继续使用 2.1 的 22 维输入；每秒能量既参与 TCN/RLS 超参数选择，也作为在线流程的反馈监督，但 TCN 始终输出逐采样点功率；原始采样频率不固定，能量计算必须使用每个样本的 `dt_seconds`；每个 flight 独立重置 RLS；最终同时输出功率曲线、每秒能量曲线、累计能量曲线及真实值对比；3.0 与 2.0 并列保存，不能覆盖 2.0；运行时检查 `out/data`，缺失或不符合标准时从 D 盘原始数据重建；TCN 候选训练和最终训练均为 80 epoch。

数据传输顺序为：固定网格数据进入特征工程，显式计算 `dt_seconds` 和真实功率，再按 `flight + floor(time)` 汇总真实秒级能量。TCN 接收 23 维序列，经时间门控和因果卷积输出逐点功率。当前窗口用旧 RLS 参数校正；窗口结束后分别积分 TCN、RLS 和真实功率，RLS 再用 TCN 窗口能量与真实能量更新参数。新参数只用于下一窗口，最终输出逐点、秒级和 flight 级结果。

## 12. 变量解释

| 变量 | 含义 | 单位 |
|---|---|---|
| $P_{f,i}^{true}$ | flight $f$ 的第 $i$ 个样本真实功率 | W |
| $\Delta t_{f,i}$ | 该样本对应的不规则采样间隔 | s |
| $E_{f,k}^{true}$ | flight $f$ 第 $k$ 个完整 1 秒窗口真实能量 | Wh |
| $E_{f,k}^{TCN}$ | TCN 逐点功率积分得到的窗口能量 | Wh |
| $E_{f,k}^{RLS}$ | RLS 校正功率积分得到的窗口能量 | Wh |
| $\theta_{0,f,k}$ | 当前窗口使用的功率偏置参数 | W |
| $\theta_{1,f,k}$ | 当前窗口使用的功率缩放参数 | 无量纲 |
| $\lambda$ | RLS 遗忘因子，本次为 0.91 | 无量纲 |
| $\mathbf P_{f,k}$ | 当前 flight 的 RLS 协方差矩阵 | 参数协方差 |
| $T_w$ | TCN 输入时间窗，本次最优为 2 s | s |

变量使用 GitHub 可渲染的 $...$ 数学格式，独立公式使用 $$...$$，不再把数学表达式放在代码反引号中。
