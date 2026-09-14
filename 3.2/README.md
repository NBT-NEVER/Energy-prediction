# 四轴无人机飞行能耗预测模型 3.2

## 1. 版本作用

3.2 保留 2.1 的 TCN、按 flight 切分、预测区间和自定义工况功能，并将 D 盘原始记录先按 flight 重采样到统一的 0.12 s 时间网格。模型输入在原 22 维工况特征上加入 `dt_seconds`，成为 23 维时间感知输入；TCN 仍输出逐采样点功率，但训练损失同时使用完整 1 秒窗口的能量差更新网络权重。RLS 根据 TCN 功率判断停机/飞行状态，在状态切换时恢复偏置 0、缩放 1，并在窗口结束后用真实能量更新下一窗口的校正参数。

## 2. 运行配置

在 `3.2` 目录执行：

```bash
python main.py prepare --force-prepare
python main.py tune-tcn
python main.py train-fixed
python main.py tune-rls
python main.py test
```

3.2 的超参数选择和模型训练已经完成。复现本次补充结果时只需执行 `python main.py prepare`（若缺少切分文件）以及 `python main.py compare-rls-windows`、`python main.py route-visualize`；不要再次执行 `tune-tcn`、`train-fixed` 或 `tune-rls`，除非明确要重做实验。

`tune-tcn` 只搜索 TCN；`train-fixed` 在终端打印当前参数并跳过搜索完成正式训练；`tune-rls` 只加载现有 TCN 搜索 RLS；`test` 只用固定模型与 RLS 参数测试和制图。本次使用 CUDA，最多训练 80 epoch，批大小 2048，学习率 `3e-4`，权重衰减 `1e-4`，TCN 通道 `[64,64,64,32]`，卷积核 3，Dropout 0.08。

## 3. 数据处理

固定采样原表位于 `D:/Python-files/Energy-prediction/data/dji_matrice_100`，旧不规则数据备份于同级 `dji_matrice_100_irregular_backup`。项目外一次性脚本已将 257896 条、209 个 flight 的记录重采样为 322904 条，典型间隔为 0.12 s；整理完成后脚本不保留在项目中。项目读取新原表后保留 R1 航线，得到 284758 条有效记录、182 个 flight。train/val/test 分别为 198469/41052/45237 条，对应 126/28/28 个 flight。

项目仍根据相邻时间戳显式计算 `dt_seconds`，而不假设数组下标等价于时间；除 flight 尾点外，其值基本为 0.12 s。功率由电压和非负放电电流计算，区间能量为 `power_w * dt_seconds / 3600`。每秒窗口使用 `floor(time)` 编号，并按真实间隔积分。

主要数据文件：

- `out/data/processed_3.2/uav_energy_features.csv`：逐采样点清洗和特征表。
- `out/data/processed_3.2/second_energy_3.2.csv`：每个 flight 的每个秒级窗口真实能量表。
- `out/data/processed_3.2/train.csv`、`val.csv`、`test.csv`：按完整 flight 切分的数据。
- `out/data/processed_3.2/feature_metadata.json`：输入特征和字段说明。
- `out/data/processed_3.2/dataset_summary.json`：数据规模和版本摘要。

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

3.2 已完成的 RLS 参数来自仅含遗忘因子和初始协方差的 `7 × 7 = 49` 组搜索：遗忘因子候选为 `0.86、0.88、0.90、0.91、0.92、0.94、0.96`，初始协方差候选为 `0.01、0.025、0.05、0.1、0.25、0.5、1.0`，`warmup_windows` 固定为 `0`。当前权重实际保存的参数为遗忘因子 `0.86`、初始协方差 `1.0`、参考能量窗 `20 s`。RLS 能量窗不属于超参数搜索；不同能量窗的影响在后文单独进行变量分析。

## 7. 最终训练、测试和结果

本节复用已完成的 `12 s / 100 步` TCN 权重和上述已固定 RLS 参数，不重新训练模型。测试集共 45237 条记录、28 个 flight；参考能量窗为 `20 s`。

| 指标 | TCN | TCN + 秒级能量 RLS | 单位 |
|---|---:|---:|---|
| 样本功率 MAE | 31.3320 | 27.7797 | W |
| 样本功率 RMSE | 46.1691 | 43.2377 | W |
| 样本功率 R2 | 0.9574 | 0.9630 | 无量纲 |
| 样本功率 WAPE | 7.7114 | 6.8371 | % |
| flight 能耗 MAE | 0.6119 | 0.0674 | Wh |
| flight 能耗 RMSE | 0.8226 | 0.1135 | Wh |
| flight 能耗 R2 | 0.9693 | 0.9994 | 无量纲 |
| flight 能耗 WAPE | 2.7966 | 0.3079 | % |

RLS 后样本功率 WAPE 相对下降 `11.34%`，flight 能耗 WAPE 相对下降 `88.99%`。最差 flight 87 的 RLS 总能量绝对百分比误差为 `1.6675%`，未重现旧实验 flight 233 的明显过度校正。95% 功率区间实际覆盖率为 `93.6181%`，平均宽度 `145.2989 W`；flight 能耗区间覆盖率为 `100%`。

## 8. 结果图展示与分析

本节嵌入本次 3.2 实际运行生成的全部结果图。训练图用于检查候选窗口和最终模型的选择过程，评估图用于检查测试集整体误差，预测图用于观察逐点功率、每秒能量以及典型 flight 的时间变化。

### 8.1 TCN 窗口候选和训练过程

![TCN 候选验证集 WAPE](./out/figures/training/candidate_validation_wape.png)

图中比较了 10 个 TCN 时间窗候选的验证集误差。2.0 s 候选的综合选择分数为 `7.38557800`，低于其余候选，因此最终采用 2.0 s、17 步输入窗口。候选并非随时间窗变长而单调改善，说明扩大历史范围后，新增历史信息与模型训练误差之间存在权衡。

![TCN 和 RLS 超参数排序](./out/figures/training/hyperparameter_ranking.png)

该图保留历史 TCN 窗口排序。新 RLS 搜索选中遗忘因子 `0.91`、初始协方差 `0.25`、预热 0 个完整窗口，含尾部惩罚的汇总分数为 `6.12551123`。

![学习率变化](./out/figures/training/learning_rate_schedule.png)

学习率曲线反映训练过程中优化步长的变化。候选模型和最终模型使用同一套学习率调度，便于比较不同时间窗，而不会把学习率变化误认为时间窗带来的差异。

![3.2 损失曲线](./out/figures/training/loss_curve_3.2.png)

最终训练运行 63 个 epoch 后早停，最佳验证联合损失为 `0.01641463`，出现在第 53 个 epoch。

### 8.2 测试集总体结果

![总体评估指标](./out/figures/results/evaluation_metrics.png)

总体评估图同时比较纯 TCN 与 RLS 校正结果。RLS 后样本功率 WAPE 由 `7.7114%` 降至 `6.8371%`，flight 能耗 WAPE 由 `2.7966%` 降至 `0.3079%`，flight 能耗 R2 提升至 `0.9994`。

### 8.3 功率预测图

![功率分箱 MAE](./out/figures/results/power_bin_mae.png)

功率分箱图按真实功率区间统计 MAE，用来检查误差是否集中在某一功率范围。低功率段的相对误差容易被接近 0 的分母放大，因此应结合 MAE、WAPE 和散点图共同判断，不能仅凭低功率段的 MAPE 排名评价模型。

![功率预测散点图](./out/figures/prediction/power_prediction_scatter.png)

散点图以真实功率为横轴、RLS 功率为纵轴，测试集 R2 为 `0.9630`、RMSE 为 `43.2377 W`。

![功率残差直方图](./out/figures/prediction/power_residual_histogram.png)

残差主体集中在 0 附近，两侧仍有快速变化形成的尾部。RLS 后样本功率 MAE 为 `27.7797 W`、RMSE 为 `43.2377 W`。

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

自定义产物如下：`custom_scenarios_3.2.csv` 保存航线输入，`custom_predictions_3.2.csv` 保存 TCN/RLS 预测和能量字段，`custom_prediction_summary_3.2.json` 保存摘要指标；对应图片位于 `out/figures/custom/`。

## 9. 输出文件

测试集 RLS 轨迹包含 5442 个秒窗口和 61 次状态切换。偏置归一化系数均值 `0.00697`，缩放系数均值 `0.97398`；少量窗口触及裁剪边界，说明后续仍可研究连续越界计数或更新平滑约束。

![flight 18 RLS 参数轨迹](./out/rls/flight_18_rls_parameter_trace.png)

红色竖线表示状态切换，切换后参数恢复为 `[0,1]`。

- `out/model/tuning_results_3.2.csv`：10 个 TCN 窗口的 80 轮训练结果和能量选择指标。
- `out/rls/rls_tuning_results_3.2.csv`：49 组 RLS 遗忘因子和初始协方差候选及尾部惩罚价值函数；能量窗不在该搜索表内。
- `out/rls/rls_parameter_trace_3.2.csv`：逐秒窗口参数轨迹。
- `out/rls/rls_parameter_statistics_3.2.csv`：逐 flight 参数统计。
- `out/model/training_log_3.2.csv`：TCN 候选和最终训练日志。
- `out/model/evaluation_3.2.json/csv`：测试集总体指标。
- `out/model/flight_energy_summary_3.2.csv`：按 flight 汇总的真实、TCN 和 RLS 能量。
- `out/model/second_energy_evaluation_3.2.csv`：按 flight 和秒级窗口的能量对比及误差。
- `out/predictions/test_predictions_3.2.csv`：逐采样点功率、每秒能量、累计能量和预测区间。
- `out/figures/prediction/flight_*_power_timeseries.png`：逐点功率对比图。
- `out/figures/prediction/flight_*_second_energy_timeseries.png`：每秒能量真实值与预测值对比图。
- `out/figures/prediction/second_energy_prediction_scatter.png`：真实每秒能量与预测每秒能量的散点图及理想线。
- `out/figures/prediction/second_energy_residual_histogram.png`：每秒能量残差直方图。
- `out/figures/results/flight_energy_actual_vs_predicted.png`：flight 总能量对比图。
- `out/figures/results/flight_energy_error.png`：flight 总能量误差图。
- `out/custom/custom_scenarios_3.2.csv`：固定参数生成的 R1 自定义展示航线输入。
- `out/custom/custom_predictions_3.2.csv`：自定义航线的 TCN 功率、RLS 校正功率、预测能量和累计能量。
- `out/custom/custom_prediction_summary_3.2.json`：自定义航线预测摘要和置信区间。
- `out/figures/custom/custom_power_timeseries.png`：自定义航线功率及预测区间。
- `out/figures/custom/custom_cumulative_energy.png`：自定义航线累计能耗及预测区间。

## 10. 文件调用关系

`main.py` 调度全流程；`config.py` 统一管理路径、参数和版本化文件名；`data_utils.py` 下载、清理数据、计算不规则采样间隔并生成秒级能量表；`model.py` 定义因果 TCN 和窗口能量监督 RLS；`train.py` 执行 TCN 窗口搜索、RLS 搜索和最终训练；`predict.py` 生成逐点、秒级和累计能量字段；`evaluate.py` 生成测试指标及两个能量汇总文件；`visualize.py` 生成训练、功率、秒级能量和 flight 能量图表；`uncertainty.py` 保留 2.1 的预测区间校准功能。

## 11. 3.2 需求与数据流记录

3.2 的约束来自前期设计讨论并全部落实：真实每秒能量只能在窗口结束时得到；当前窗口不能使用当前窗口真实能量更新 RLS；更新后的参数只传给下一个窗口；TCN 输入不增加真实每秒能量，继续使用 2.1 的 22 维输入；每秒能量既参与 TCN/RLS 超参数选择，也作为在线流程的反馈监督，但 TCN 始终输出逐采样点功率；原始采样频率不固定，能量计算必须使用每个样本的 `dt_seconds`；每个 flight 独立重置 RLS；最终同时输出功率曲线、每秒能量曲线、累计能量曲线及真实值对比；3.2 与 2.0 并列保存，不能覆盖 2.0；运行时检查 `out/data`，缺失或不符合标准时从 D 盘原始数据重建；TCN 候选训练和最终训练均为 80 epoch。

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

---

## 13. 2026-09-07 全流程更新补充（最新结果）

本节为本次完整流程产生的新记录，追加在原 README 之后。前文中的参数、图表和指标属于历史运行记录，继续保留用于追溯；下面的“最新全流程结果”以当前 `out/model`、`out/rls`、`out/predictions` 和 `out/figures` 中的文件为准。

### 13.1 直接运行入口与阶段顺序

在 `3.2` 目录直接执行：

```bash
python main.py
```

默认模式为 `all`，按以下顺序执行，不会跳过 `tune-tcn`：

```text
prepare -> tune-tcn -> train-fixed -> tune-rls -> calibrate -> evaluate -> visualize
```

其中，`tune-tcn` 会训练并比较全部 TCN 时间窗候选，同时在终端显示候选编号、当前候选的逐 epoch 训练进度、验证集推理和每个候选的测试结果；`train-fixed` 读取已经选出的 TCN 窗口，只进行正式训练，不重复搜索 TCN；`tune-rls` 固定正式 TCN 的预测，只搜索 RLS 参数；后续阶段负责区间校准、测试评估和图表生成。也可以单独执行 `python main.py tune-tcn`、`python main.py train-fixed` 或 `python main.py tune-rls`。

终端显示阶段耗时、训练 epoch、当前损失、学习率、候选进度和 ETA。CSV/JSON 日志只保存 epoch 结果、候选汇总、最优参数和阶段摘要，不把动态进度条写入结果文件；完整终端记录另存为 `out/logs/terminal_3.2.log`。

### 13.2 最新超参数选择方式

数据按完整 flight 划分为训练集、验证集和测试集，测试集在全部参数固定后才使用。当前测试集有 45237 条采样记录、28 个 flight；TCN 和 RLS 的选择均只依据验证集。

TCN 阶段固定通道 `[64,64,64,32]`、卷积核 3、Dropout 0.08、学习率 `3e-4`、权重衰减 `1e-4` 和 Huber 参数 0.65，搜索 10 个时间窗：`0.25、0.5、1、2、4、6、8、10、12、16 s`。每组候选训练 80 轮，不使用调参早停，然后将验证集逐点功率、完整秒能量和 flight 总能量统一计算为：

$$
S_{TCN}=WAPE_{second}+0.2WAPE_{sample}+0.5WAPE_{flight}
$$

选择分数越小越好。最新最优候选是 `12 s / 100 步`，验证集采样点功率 WAPE 为 `6.4503%`，秒级能量 WAPE 为 `4.8466%`，flight 能量 WAPE 为 `1.7613%`，综合分数为 `7.017347`。候选完整记录见 `out/model/tuning_results_3.2.csv`。

RLS 阶段复用固定 TCN 在验证集上的逐点预测，不重新训练 TCN。当前只搜索遗忘因子 `0.86、0.88、0.90、0.91、0.92、0.94、0.96` 与初始协方差 `0.01、0.025、0.05、0.1、0.25、0.5、1.0`，共 `7 × 7 = 49` 组；`warmup_windows` 已从候选参数中删除，所有候选均固定为 `0`。已完成的 3.2 权重实际使用 `forgetting_factor=0.86`、`initial_covariance=1.0`、`warmup=0`、参考能量窗 `20 s`；能量窗不参与超参数搜索，单独变量分析见 13.7。完整记录见 `out/rls/rls_tuning_results_3.2.csv`。

### 13.3 数据流与算法边界

原始记录先按 flight 重采样到约 `0.12 s` 网格，计算相邻时间差 `dt_seconds`、电压电流得到的真实功率，以及每个完整 1 秒窗口的真实能量。TCN 输入是 22 维工况特征加 `dt_seconds` 的 23 维序列；真实秒级能量只作为训练监督和 RLS 反馈，不作为当前窗口的前向输入。

TCN 先输出每个采样点的功率 $\hat{P}_{f,i}^{TCN}$。对窗口 $k$ 积分得到能量：

$$
E_{f,k} = \sum_{i\in k} P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

窗口内，RLS 使用窗口开始前的参数对每个 TCN 功率进行仿射校正：

$$
\hat{P}_{f,i}^{RLS}=\theta_{0,f,k}+\theta_{1,f,k}\hat{P}_{f,i}^{TCN}
$$

窗口结束后才得到真实能量 $E_{f,k}^{true}$，程序据此更新 RLS；因此当前窗口不使用当前窗口标签，更新后的 $\theta_{f,k+1}$ 只传给下一个窗口。每个 flight 独立初始化并在状态切换时恢复中性校正，避免一个 flight 的参数泄漏到另一个 flight。

其中：

$f$：flight 编号；$k$：完整 1 秒窗口编号；$i$：窗口内采样点编号。

$P_{f,i}$：采样点功率，单位 W；$\hat{P}_{f,i}^{TCN}$ 和 $\hat{P}_{f,i}^{RLS}$：TCN 原始功率和 RLS 校正功率，单位 W。

$\Delta t_{f,i}$：采样间隔，单位 s；$E_{f,k}$：窗口能量，单位 Wh；$\theta_0$：偏置，单位 W；$\theta_1$：无量纲缩放系数。

### 13.4 最新正式训练与测试结果

本次不重复执行 `train-fixed`。直接复用已完成的 `12 s / 100 步` TCN 权重；测试阶段沿用权重中的 RLS 参数 `0.86 / 1.0 / warmup=0` 和参考能量窗 `20 s`。

| 指标 | 纯 TCN | TCN + RLS | 单位 |
|---|---:|---:|---|
| 样本功率 MAE | 31.7382 | 27.9911 | W |
| 样本功率 RMSE | 46.7863 | 43.4546 | W |
| 样本功率 R2 | 0.9563 | 0.9623 | 无量纲 |
| 样本功率 WAPE | 7.8114 | 6.8892 | % |
| flight 能耗 MAE | 0.5968 | 0.0640 | Wh |
| flight 能耗 RMSE | 0.8423 | 0.0890 | Wh |
| flight 能耗 R2 | 0.9679 | 0.9996 | 无量纲 |
| flight 能耗 WAPE | 2.7277 | 0.2923 | % |

RLS 使样本功率 WAPE 下降 `0.9222` 个百分点，相对下降 `11.81%`；flight 能耗 WAPE 下降 `2.4354` 个百分点，相对下降 `89.28%`。测试集真实总能量为 `612.6634 Wh`，纯 TCN 预测总能量为 `626.2884 Wh`，RLS 预测总能量为 `613.8128 Wh`。这说明秒级反馈主要消除了累计能量偏差，不能据此认为所有瞬态功率误差都已消失。

### 13.5 最新图表与数据分析

![最新 TCN 候选验证结果](./out/figures/training/candidate_validation_wape.png)

上图按验证集选择分数比较 10 个时间窗。12 s 候选最低；6 s 分数为 `7.132358`，16 s 为 `7.175536`。候选并不随时间窗单调改善，说明历史信息长度和训练误差之间存在折中。

![最新超参数排序](./out/figures/training/hyperparameter_ranking.png)

![修复后的学习率曲线](./out/figures/training/learning_rate_schedule.png)

学习率图现在只读取 `final_tcn` 的正式训练记录，不再把 10 个候选的 epoch 混在同一条曲线中。图像尺寸为 `1620×720`，正式训练从 `3e-4` 衰减到 `1.875e-5`；这保证横轴 epoch 与 12 s 正式模型一一对应。

![最新总体评估指标](./out/figures/results/evaluation_metrics.png)

![最新功率分箱误差](./out/figures/results/power_bin_mae.png)

测试功率主要集中在 `450--600 W`，共 22870 条，RLS MAE 为 `27.5046 W`、WAPE 为 `5.3152%`。`300--450 W` 区间 MAE 为 `39.4472 W`，`600 W` 以上为 `43.2195 W`；`50--300 W` 仅 859 条但 MAE 达 `96.9980 W`，通常对应起降或状态变化。`0--50 W` 的 MAE 只有 `5.6193 W`，但真实均值仅 `1.7965 W`，WAPE 被小分母放大到 `312.7965%`，该段应优先看绝对误差。

![最新功率预测散点](./out/figures/prediction/power_prediction_scatter.png)

![最新功率残差分布](./out/figures/prediction/power_residual_histogram.png)

散点主体集中在 `450--600 W`，全测试集 $R^2=0.9623$。残差主体靠近 0，但 RMSE 高于 MAE，说明少量快速变化和状态切换样本仍形成较大误差尾部；RLS 使用完整秒窗口反馈，不能提前修正当前窗口内部的瞬态。

![最新 flight 能耗对比](./out/figures/results/flight_energy_actual_vs_predicted.png)

![最新 flight 能耗误差](./out/figures/results/flight_energy_error.png)

28 个 flight 的能耗结果中，最大相对误差为 flight 87 的 `0.9005%`，其次为 flight 194 的 `0.8735%` 和 flight 135 的 `0.8616%`；最小为 flight 113 的 `0.0278%`。flight 级 $R^2=0.9996$，但区间仍应保留，因为不同 flight 的起降、负载和功率跃迁并不相同。

![最新每秒能量散点](./out/figures/prediction/second_energy_prediction_scatter.png)

![最新每秒能量残差](./out/figures/prediction/second_energy_residual_histogram.png)

测试集共有 5442 个完整秒窗口。秒级能量残差均值为 `0.000211 Wh`，平均绝对残差为 `0.005805 Wh`，第 5% 和第 95% 分位为 `-0.013975 Wh` 与 `0.013677 Wh`。残差尾部主要出现在起飞、降落和状态切换窗口，符合“窗口结束后才更新”的在线时序约束。

![最新 RLS 参数轨迹](./out/rls/flight_18_rls_parameter_trace.png)

全部测试 flight 的 RLS 轨迹共有 5442 个窗口、63 次状态切换。偏置均值/标准差为 `0.02107 / 0.20328`，缩放均值/标准差为 `0.95554 / 0.20592`；偏置限制在 `[-1,1]`，缩放限制在 `[0,2]`。参数触及边界的窗口提示强瞬态和低能量窗口仍是在线校正的主要风险点。

![最新自定义工况功率](./out/figures/custom/custom_power_timeseries.png)

![最新自定义工况累计能耗](./out/figures/custom/custom_cumulative_energy.png)

自定义 R1 工况为 180 s、1 s 输出间隔、风速 4 m/s、巡航速度 8 m/s、载荷 250 g、高度 50 m。181 个采样点的平均预测功率为 `506.1544 W`，最大功率为 `577.6692 W`，累计预测能耗为 `25.4483 Wh`，95% 区间为 `20.8112--30.0854 Wh`。该工况没有真实功率标签，因此只用于展示预测曲线和累计能耗，不能计算误差或替代测试集结论。

### 13.6 最新产物索引

- `out/model/tuning_results_3.2.csv`：10 个 TCN 候选的窗口、步数、验证指标和选择分数。
- `out/model/training_log_3.2.csv`：TCN 候选 epoch 结果及正式训练 epoch 结果；动态进度条不写入该文件。
- `out/rls/rls_tuning_results_3.2.csv`：49 个 RLS 候选，`warmup_windows` 全部为 0。
- `out/model/evaluation_3.2.json/csv`：最新测试指标、最优参数和候选数量。
- `out/model/flight_energy_summary_3.2.csv`、`power_bin_evaluation_3.2.csv`、`second_energy_evaluation_3.2.csv`：flight、功率分箱和秒级能量明细。
- `out/predictions/test_predictions_3.2.csv`：逐点功率、秒级能量、累计能量和预测区间。
- `out/rls/rls_parameter_trace_3.2.csv`、`rls_parameter_statistics_3.2.csv`、`rls/rls_parameter_summary_3.2.json`：RLS 在线参数轨迹和统计摘要。
- `out/figures/training/learning_rate_schedule.png`：只对应正式 TCN 训练的学习率图；其余图表按目录分别对应训练、预测、评估、RLS 和自定义工况。

### 13.7 3.2 的方法、结果和 RLS 能量窗变量分析

3.2 的最终网络和 RLS 参数已经完成，本次补充工作只做数据准备、固定模型推理、误差统计和图表生成，不重新选择 TCN 超参数、不重新训练模型，也不把 RLS 能量窗加入参数搜索。RLS 能量窗作为独立自变量，在固定 `forgetting_factor=0.86`、`initial_covariance=1.0`、`warmup_windows=0` 后，对 `1、2、5、10、20 s` 五种反馈长度分别运行验证集和测试集。

变量分析结果位于 `out/model/rls_energy_window_comparison_3.2.csv`，逐 flight 明细位于 `out/model/rls_energy_window_comparison_by_flight_3.2.csv`，摘要位于 `out/model/rls_energy_window_comparison_3.2.json`。测试集总体结果如下，窗口越长，窗口级 WAPE 下降，但逐点功率和 flight 级误差上升，说明长窗口能平滑能量反馈，却降低了参数对状态变化的响应速度。

| RLS 能量窗 | 功率 WAPE (%) | 窗口 MAE (Wh) | 窗口 RMSE (Wh) | 窗口 R² | 窗口 WAPE (%) | flight MAE (Wh) | flight RMSE (Wh) | flight R² | flight WAPE (%) | 完整窗口数 | 状态切换 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 s | 6.9564 | 0.005889 | 0.009279 | 0.9777 | 5.2311 | 0.070975 | 0.099100 | 0.9996 | 0.3244 | 5415/5442 | 63 |
| 2 s | 7.1229 | 0.010731 | 0.016609 | 0.9818 | 4.7784 | 0.085794 | 0.110130 | 0.9995 | 0.3921 | 2700/2728 | 58 |
| 5 s | 7.2606 | 0.022971 | 0.033544 | 0.9879 | 4.1242 | 0.124691 | 0.157910 | 0.9989 | 0.5699 | 1074/1100 | 55 |
| 10 s | 7.4266 | 0.041382 | 0.057393 | 0.9908 | 3.7622 | 0.145189 | 0.195245 | 0.9983 | 0.6635 | 530/557 | 51 |
| 20 s | 7.4950 | 0.072094 | 0.098180 | 0.9930 | 3.3655 | 0.182692 | 0.246447 | 0.9972 | 0.8349 | 259/286 | 29 |

因此，若目标是当前窗口的能量误差，较长反馈窗具有更低的 WAPE；若目标是兼顾逐点功率、flight 总能耗和飞行状态切换响应，`1--2 s` 更合适。这里的结论是变量敏感性分析，不应写成新的“最优超参数”，最终模型仍保留已完成权重中的 `20 s` 参考窗。

![RLS 能量窗误差对比](./out/figures/results/rls_energy_window_error_comparison.png)

![RLS 能量窗指标对比](./out/figures/results/rls_energy_window_energy_metrics.png)

![RLS 能量窗窗口数量对比](./out/figures/results/rls_energy_window_count_comparison.png)

### 13.8 18、23、83 航线轨迹产物

执行 `python main.py route-visualize` 可复用固定 3.2 权重生成航线产物。每条航线均使用原始 `flights.csv` 的 IMU 位置、速度、姿态和风场字段，并与测试预测按时间戳最近邻对齐；静态图和 GIF 的角标显示风速（m/s）与风向（°）。所有产物按航线隔离保存：

- `out/routes/flight_18/`：`flight_18_power_energy.png`、`flight_18_imu_trajectory.gif`、`flight_18_imu_aligned.csv`、`flight_18_summary.json`。
- `out/routes/flight_23/`：`flight_23_power_energy.png`、`flight_23_imu_trajectory.gif`、`flight_23_imu_aligned.csv`、`flight_23_summary.json`。
- `out/routes/flight_83/`：`flight_83_power_energy.png`、`flight_83_imu_trajectory.gif`、`flight_83_imu_aligned.csv`、`flight_83_summary.json`。

三条航线的总索引为 `out/routes/route_products_summary_3.2.json`，三航线统一对比图为 `out/routes/all_routes_trajectory_energy.png`。功率能量静态图上方展示实际功率、TCN 功率和 RLS 校正功率，下方展示实际、TCN 和 RLS 累计能量；3D GIF 使用 IMU 的局部米制 `position_x_m/position_y_m/position_z_m`，同步显示功率指示点、已飞轨迹、完整参考轨迹和风场角标。

![flight 18 IMU轨迹动图](./out/routes/flight_18/flight_18_imu_trajectory.gif)

![flight 23 IMU轨迹动图](./out/routes/flight_23/flight_23_imu_trajectory.gif)

![flight 83 IMU轨迹动图](./out/routes/flight_83/flight_83_imu_trajectory.gif)
