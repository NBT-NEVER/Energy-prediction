# 四轴无人机飞行能耗预测模型 3.0

## 1. 版本作用

3.0 保留 2.1 的 22 维 TCN 输入、TCN 结构、时间窗候选、RLS 候选参数、按 flight 切分数据、预测区间和自定义工况功能。新增内容是：按照不固定采样间隔计算每个 flight 的每秒真实能量；TCN 仍输出逐采样点功率；RLS 在完整 1 秒窗口结束后使用能量监督更新；最终输出逐点功率、每秒能量和累计能量。

## 2. 运行配置

在 `3.0` 目录执行：

```bash
python main.py all --device cuda
```

本次运行使用 CUDA，TCN 候选和最终训练均为 80 个 epoch。其他初始数据和 2.1 保持一致：R1 航线、随机种子 42、22 个输入特征、批大小 2048、学习率 `3e-4`、权重衰减 `1e-4`、TCN 通道 `[64,64,64,32]`、卷积核 3、Dropout 0.08、Huber 参数 0.65。

## 3. 数据处理

原始数据读取自 `D:/Python-files/Energy-prediction/data`。本次读取 257896 条原始记录，保留 R1 航线后得到 227551 条有效采样记录、182 个 flight。train/val/test 分别为 158625/32820/36106 条，对应 126/28/28 个 flight。

采样间隔由同一 flight 内相邻 `time` 差值计算，异常间隔使用训练数据有效间隔中位数补齐。功率由电压和非负放电电流计算，采样间隔能量为 `power_w * dt_seconds / 3600`。每秒窗口使用 `floor(time)` 编号，并按 `dt_seconds` 加权求和。

主要数据文件：

- `out/data/processed_3.0/uav_energy_features.csv`：逐采样点清洗和特征表。
- `out/data/processed_3.0/second_energy_3.0.csv`：每个 flight 的每个秒级窗口真实能量表。
- `out/data/processed_3.0/train.csv`、`val.csv`、`test.csv`：按完整 flight 切分的数据。
- `out/data/processed_3.0/feature_metadata.json`：输入特征和字段说明。
- `out/data/processed_3.0/dataset_summary.json`：数据规模和版本摘要。

## 4. TCN 训练和窗口选择

TCN 输入不直接使用真实每秒能量，输入仍为 2.1 的工况特征；TCN 输出每个采样点的 `power_w` 预测。每个候选时间窗仍转换为采样步数，候选为 0.4、0.6、0.8、1.0、1.2、1.5、2.0、2.5、3.0、4.0 秒，对应 3、5、7、8、10、12、17、21、25、33 步。

逐点功率训练损失仍使用 Huber 损失。每个候选训练 80 个 epoch，不使用调参阶段早停。窗口选择同时考虑逐点功率、秒级能量和 flight 总能量：

$$S_{TCN}=WAPE_{second}+0.2WAPE_{sample}+0.5WAPE_{flight}$$

其中：

`$S_{TCN}$`：TCN 窗口选择价值函数，越小越好。

`$WAPE_{second}$`：验证集秒级能量 WAPE。

`$WAPE_{sample}$`：验证集采样点功率 WAPE。

`$WAPE_{flight}$`：验证集按 flight 汇总能量 WAPE。

本次最优窗口为 `4.0 s / 33 步`，TCN 选择分数为 `7.15476262`，对应秒级能量 WAPE `4.96353661%`，采样点功率 WAPE `7.05047759%`，flight 能量 WAPE `1.56226098%`。

## 5. 秒级能量和 RLS 数据流

每个 flight 开始时初始化 RLS 参数和协方差矩阵。当前 1 秒窗口内，TCN 对全部采样点逐点预测功率，RLS 使用窗口开始前的参数校正这些功率。窗口内参数不更新。

窗口结束后，程序使用每个采样点的实际 `dt_seconds` 积分 TCN 功率、RLS 校正功率和真实功率，分别得到 `tcn_second_energy_wh`、`predicted_second_energy_wh` 和 `actual_second_energy_wh`。真实能量在窗口结束前不可用，因此不会参与当前窗口的功率校正。

RLS 将窗口预测能量和真实能量换算为窗口平均功率，再进行仿射回归：

$$\bar{P}_{f,k}^{true}\approx\theta_{0,f,k}+\theta_{1,f,k}\bar{P}_{f,k}^{pred}$$

其中：

`$\bar{P}_{f,k}^{true}$`：真实能量除以窗口实际时长得到的真实平均功率。

`$\bar{P}_{f,k}^{pred}$`：当前窗口预测能量除以窗口实际时长得到的预测平均功率。

`$\theta_{0,f,k}$`：RLS 的功率偏置参数。

`$\theta_{1,f,k}$`：RLS 的功率缩放参数。

`$f$`：flight 编号；每个 flight 独立重置 RLS。

`$k$`：当前秒级窗口编号。

更新完成后，`theta_{f,k+1}` 只用于下一个窗口。最后一个窗口更新的参数不会反向修改已经输出的结果。

## 6. RLS 超参数搜索

搜索固定最优 TCN 窗口的验证集预测，不重复训练 TCN。72 组候选均按以下顺序运行：按 flight 排序并重置 RLS；逐窗口预测；窗口结束后积分能量；使用当前窗口真实能量更新；将新参数传给下一个窗口。

每个 flight 先计算独立价值函数，再对 28 个验证 flight 等权平均：

$$S_{RLS,f}=WAPE_{second,f}+0.2WAPE_{sample,f}+0.5WAPE_{flight,f}$$

$$S_{RLS}=\frac{1}{N_{flight}}\sum_f S_{RLS,f}$$

其中：

`$S_{RLS,f}$`：单个 flight 的 RLS 价值函数。

`$S_{RLS}$`：所有验证 flight 的汇总价值函数。

`$N_{flight}$`：验证集 flight 数量，本次为 28。

`$WAPE_{second,f}$`：单个 flight 的秒级能量误差。

`$WAPE_{sample,f}$`：单个 flight 的逐点功率误差。

`$WAPE_{flight,f}$`：单个 flight 总能量误差。

本次 72 组搜索选择：遗忘因子 `0.95`、初始协方差 `100`、预热 `0.5 s`。汇总价值函数为 `7.03463516`，标准差为 `2.11632074`，包含 28 个验证 flight。参数记录位于 `out/model/rls_tuning_results_3.0.csv`。

## 7. 最终训练、测试和结果

固定 `4.0 s / 33 步` 和上述 RLS 参数后，最终 TCN 训练 80 个 epoch，最佳验证损失为 `0.01979637`，出现在第 43 个 epoch。测试集只在超参数全部固定后运行一次。

| 指标 | TCN | TCN + 秒级能量 RLS | 单位 |
|---|---:|---:|---|
| 样本功率 MAE | 33.2258 | 32.1707 | W |
| 样本功率 RMSE | 50.9658 | 51.7598 | W |
| 样本功率 R2 | 0.9485 | 0.9469 | 无量纲 |
| 样本功率 WAPE | 8.1785 | 7.9188 | % |
| flight 能耗 MAE | 0.5310 | 0.2453 | Wh |
| flight 能耗 RMSE | 0.7642 | 0.3605 | Wh |
| flight 能耗 R2 | 0.9735 | 0.9941 | 无量纲 |
| flight 能耗 WAPE | 2.4271 | 1.1213 | % |

RLS 后 flight 能耗 MAE 从 `0.5310 Wh` 降至 `0.2453 Wh`，WAPE 从 `2.4271%` 降至 `1.1213%`。样本功率 WAPE 从 `8.1785%` 降至 `7.9188%`，但 RMSE 和 R2 有小幅变化，说明窗口级能量校正主要改善累计能耗一致性，不保证每个采样点误差都同步下降。

## 8. 输出文件

- `out/model/tuning_results_3.0.csv`：10 个 TCN 窗口的 80 轮训练结果和能量选择指标。
- `out/model/rls_tuning_results_3.0.csv`：72 组 RLS 参数的验证价值函数。
- `out/model/training_log_3.0.csv`：TCN 候选和最终训练日志。
- `out/model/evaluation_3.0.json/csv`：测试集总体指标。
- `out/model/flight_energy_summary_3.0.csv`：按 flight 汇总的真实、TCN 和 RLS 能量。
- `out/model/second_energy_evaluation_3.0.csv`：按 flight 和秒级窗口的能量对比及误差。
- `out/predictions/test_predictions_3.0.csv`：逐采样点功率、每秒能量、累计能量和预测区间。
- `out/figures/prediction/flight_*_power_timeseries.png`：逐点功率对比图。
- `out/figures/prediction/flight_*_second_energy_timeseries.png`：每秒能量真实值与预测值对比图。
- `out/figures/results/flight_energy_actual_vs_predicted.png`：flight 总能量对比图。
- `out/figures/results/flight_energy_error.png`：flight 总能量误差图。

## 9. 文件调用关系

`main.py` 调度全流程；`config.py` 统一管理路径、参数和版本化文件名；`data_utils.py` 下载、清理数据、计算不规则采样间隔并生成秒级能量表；`model.py` 定义因果 TCN 和窗口能量监督 RLS；`train.py` 执行 TCN 窗口搜索、RLS 搜索和最终训练；`predict.py` 生成逐点、秒级和累计能量字段；`evaluate.py` 生成测试指标及两个能量汇总文件；`visualize.py` 生成训练、功率、秒级能量和 flight 能量图表；`uncertainty.py` 保留 2.1 的预测区间校准功能。
