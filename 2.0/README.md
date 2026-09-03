# 四轴无人机飞行能耗预测模型 2.0：TCN + RLS 实时矫正

本实验沿用 1.0 的 DJI Matrice 100 数据、特征工程、flight 分组切分、评估口径和输出目录结构，将预测模型替换为 4 个残差块的因果 TCN，并在前向推理后使用递推最小二乘（RLS）实时校正功率。2.0 仅保留 R1 航线进行训练、验证和测试；其他航线原始记录不会写回 D 盘，而是筛选后保存到项目内 `out/data/processed_2.0/excluded_routes_2.0.csv`。TCN 只读取当前样本及历史短窗；RLS 在当前预测输出后使用当前已观测功率更新，因此适用于在线矫正。

## 数据与路径

数据源仍为 `https://www.modelscope.cn/datasets/OmniData/Data_Collected_with_Package_etc.git`，原始数据默认位于 `D:/Python-files/Energy-prediction/data`。处理数据、日志、评估表、预测表和图表写入 `2.0/out`，模型权重写入 `D:/Python-files/Energy-prediction/model`。所有路径和文件名集中在 `config.py`。

本次实际数据摘要为：原始读取 257896 条，筛除非 R1 航线 30345 条，保留 R1 有效记录 227551 条、182 个 flight；train/val/test 分别为 158625/32820/36106 条，对应 126/28/28 个 flight。模型输入为 22 维，完整变量含义见 `out/data/processed_2.0/feature_metadata.json`。

## 目录结构

```text
2.0/
├── config.py          # 路径、训练参数、时间窗和RLS参数
├── data_utils.py      # 下载、解包、特征工程和flight切分
├── model.py           # 因果TCN和RLSCorrector
├── train.py           # 时间窗调参、TCN训练、RLS初始标定
├── predict.py         # TCN前向推理和RLS逐点在线校正
├── evaluate.py        # TCN与RLS两套结果的样本级、flight级评估
├── visualize.py       # 训练、评估、预测、自定义工况图表
├── main.py            # 统一命令行入口
├── progress.py        # 轻量级终端进度条
├── terminal_logger.py # 将终端输出和异常同步追加到UTF-8日志
├── uncertainty.py     # 保序残差校准和即时功率/能耗区间
├── device_utils.py    # CUDA设备检查
├── requirements.txt   # Python依赖
└── out                 # 2.0独立输出目录（含R1处理数据和排除航线记录）
```

## 运行方式

在 `2.0` 目录执行完整流程：

```bash
python main.py all
```

也可以分步执行：

```bash
python main.py prepare
python main.py train
python main.py evaluate
python main.py visualize
python main.py custom --wind-speed 5 --flight-speed 8 --payload-g 250 --altitude 50 --duration-s 180
python main.py interval --confidence 0.90
```

默认训练深度为 4 个 TCN 残差块，通道为 `64,64,64,32`，调参训练 20 轮、最终训练 80 轮。可通过 `--epochs`、`--tune-epochs` 和 `--tcn-channels` 覆盖；终端会显示总流程、时间窗、epoch、批次、耗时和 ETA。

每次通过 `main.py` 运行任意模式时，关键终端内容会同步追加到 `out/logs/terminal_2.0.log`。日志使用 UTF-8 编码，记录运行模式、开始与结束时间、阶段提示、结果摘要和异常信息；逐 flight、逐 batch 等动态进度条只在终端显示，不写入日志，历史运行记录不会被覆盖。

首次训练会使用验证集绝对残差生成在线 RLS 和固定 RLS 两套置信区间校准文件，默认置信度为 `0.95`。预测 CSV 会同时保存 `predicted_power_w`、`cumulative_energy_wh`，以及功率和累计能耗的上下限。预测完成后只需运行 `interval` 修改置信度，程序只读取已有 CSV 并即时重算，不会重新执行 TCN：

```bash
python main.py interval --confidence 0.90
python main.py interval --confidence 0.99 --input-csv out/predictions/test_predictions_2.0.csv
```

若需单独重建校准残差，可运行 `python main.py calibrate`。区间采用 split conformal 绝对残差分位数；累计能耗上下限由功率上下限按 `dt_seconds` 积分并逐 flight 累加得到。

时间窗候选以秒为单位，默认测试 `0.6, 1.0, 1.5, 2.0, 3.0` 秒；可覆盖候选，例如：

```bash
python main.py train --window-candidates 0.4,0.8,1.2,1.6
```

训练、验证和推理要求 CUDA。默认设备为 `cuda`，可使用 `--device cuda:0` 指定设备。

## 算法原理

对每个 flight 按时间排序，窗口左侧不足部分使用该 flight 的首个样本填充。TCN 输入形状为 `(batch, window_steps, feature_count)`，因果卷积的左填充保证时刻 $t$ 不读取 $t+1$ 之后的数据：

$$
\hat{P}^{\mathrm{TCN}}_t=f_{\theta}(X_{t-L+1:t})
$$

其中：
$\hat{P}^{\mathrm{TCN}}_t$：时刻 $t$ 的TCN功率预测，单位 W。
$f_{\theta}$：TCN网络及其参数。
$X_{t-L+1:t}$：从当前时刻向前的输入窗口。
$L$：窗口采样步数。

TCN 由 4 个残差卷积块组成，通道为 `[64,64,64,32]`，膨胀率为 1、2、4、8。RLS 对 TCN 预测和偏差项建立仿射校正器。实时输出先使用当前参数校正，再用当前实测功率更新参数：

当输入 CSV 包含 `power_w` 时，程序按“先输出、后更新”执行在线校正；没有 `power_w` 的部署输入只执行 TCN 前向和已有 RLS 参数校正，不会伪造观测值。

$$
\tilde{P}_t=s_P\boldsymbol{\phi}_t^{\mathsf{T}}\boldsymbol{\theta}_{t-1}
$$

$$
\boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}+\mathbf{K}_t\left(P_t/s_P-\boldsymbol{\phi}_t^{\mathsf{T}}\boldsymbol{\theta}_{t-1}\right)
$$

其中：
$\tilde{P}_t$：RLS校正后的实时功率，单位 W。
$P_t$：当前采样点实测功率，单位 W。
$\boldsymbol{\phi}_t=[1,\hat{P}^{\mathrm{TCN}}_t/s_P]^{\mathsf{T}}$：RLS回归向量。
$\boldsymbol{\theta}_t$：时刻 $t$ 的偏置和比例校正参数。
$\mathbf{K}_t$：RLS增益向量。
$s_P$：功率尺度，来自训练集非零功率中位数。

窗口选择分数与 1.0 的对比口径一致，使用验证集 flight 能耗 WAPE 和样本功率 WAPE：

$$
S=\mathrm{WAPE}_{\mathrm{flight}}+0.2\,\mathrm{WAPE}_{\mathrm{sample}}
$$

其中：
$S$：候选时间窗的选择分数，越小越优。
$\mathrm{WAPE}_{\mathrm{flight}}$：flight级能耗加权绝对百分比误差。
$\mathrm{WAPE}_{\mathrm{sample}}$：采样点功率加权绝对百分比误差。

## 输出与对比关系

- `out/data/processed_2.0/`：仅含 R1 的特征表、train/val/test 切分、逐变量元数据，以及 `excluded_routes_2.0.csv` 排除记录。
- `out/model/scaler_2.0.json`：TCN输入标准化、目标标准化、典型采样周期和RLS功率尺度。
- `out/model/tuning_results_2.0.csv`：每个时间窗的秒数、采样步数、TCN基线 WAPE、RLS校正 WAPE 和选择分数；这是窗口优劣的主对比表。
- `out/model/training_log_2.0.csv`：最终窗口每轮训练损失、验证损失、学习率和窗口信息。
- `out/model/evaluation_2.0.json`、`evaluation_2.0.csv`：同时包含 `tcn_*` 基线指标和不带前缀的 RLS 校正指标。
- `out/model/flight_energy_summary_2.0.csv`：同时保存 `tcn_predicted_energy_wh` 与 `predicted_energy_wh`，可逐 flight 对比校正前后能耗。
- `out/model/power_bin_evaluation_2.0.csv`：按真实功率区间统计 RLS 校正结果，字段与 1.0 对齐。
- `out/predictions/test_predictions_2.0.csv`：包含 `tcn_predicted_power_w`、`rls_corrected_power_w`、`predicted_power_w`、校正量和对应能耗字段。
- `out/model/uncertainty_calibration_2.0.npz/json`：在线 RLS、固定 RLS 的验证集绝对残差和校准摘要。
- 预测 CSV 还包含 `confidence_level`、`power_interval_radius_w`、功率上下限、单步能耗上下限和累计能耗上下限。
- `D:/Python-files/Energy-prediction/model/best_energy_tcn_rls_2.0.pt`、`final_energy_tcn_rls_2.0.pt`：TCN权重、最优窗口、采样周期和RLS初始参数。

## 图表输出

`visualize` 保留 1.0 的训练、评估、散点、残差、flight时序和自定义工况图文件名，数据内容改为 2.0 结果。训练候选图读取 `tuning_results_2.0.csv`；评估指标图同时显示 TCN 基线与 RLS 校正指标；测试 flight 时序图中的 `Predicted power` 为 RLS 校正结果。

## 文件调用关系

`main.py` 调用 `data_utils.prepare_dataset()`、`train.train_model()`、`evaluate.evaluate_model()` 和 `visualize.generate_all_visualizations()`。`train.py` 调用 `model.build_model()` 训练 TCN，调用 `apply_rls_correction()` 在验证集上比较窗口，并调用 `uncertainty.save_calibration()` 保存残差校准数据。`predict.py` 读取 checkpoint 的 `window_steps`，调用 `build_sequence_arrays()` 做因果短窗，调用 `predict_array()` 得到 TCN 前向结果，再按 flight 和时间顺序调用 RLS；`uncertainty.py` 为预测表添加功率和累计能耗区间。`interval` 模式只读取已有预测 CSV，因此修改置信度不会重复模型推理。`evaluate.py` 读取预测 CSV，同时计算 TCN 基线、RLS 校正误差和区间覆盖率，因此每一项结果都能与 1.0 的同名产物直接对照。
