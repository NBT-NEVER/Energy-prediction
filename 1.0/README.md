# 四轴无人机飞行能耗预测模型 1.0

本实验使用公开的 DJI Matrice 100 四轴无人机飞行数据，搭建普通 MLP 与残差 MLP 回归模型，用飞行期间的空气流速、飞行速度、载荷、高度、姿态变化、加速度，以及由运动状态推导出的热负荷、避障敏捷度、视觉算法耗能和通信耗能代理特征，预测瞬时电功率，并进一步积分得到单次飞行能耗。

## 数据来源

数据集：Data Collected with Package Delivery Quadcopter Drone

公开镜像：`https://www.modelscope.cn/datasets/OmniData/Data_Collected_with_Package_etc.git`

原始数据说明页：`https://theairlab.org/energy-dataset/`

原始公开数据仍存放在 `D:/Python-files/Energy-prediction/data`。根据当前要求，处理后数据、训练记录、评估结果、预测结果和可视化图表统一输出到 `1.0/out`；只有模型训练权重 `.pt` 文件保存到 `D:/Python-files/Energy-prediction/model`。

## 目录结构

```text
I:/STUDY/python/project/Energy-prediction/1.0
├── config.py          # 统一管理路径、训练参数和输出文件名
├── data_utils.py      # 下载、解包、特征工程和按flight切分数据
├── model.py           # 自建普通MLP和残差MLP回归模型
├── train.py           # GPU训练、调参、权重和日志保存
├── predict.py         # 加载模型并生成预测CSV
├── evaluate.py        # 样本级功率、flight级能耗和功率分段评估
├── visualize.py       # 训练、评估、预测和自定义工况可视化
├── main.py            # 统一入口
├── requirements.txt   # Python依赖
└── out                # 统一输出目录
```

输出目录：

```text
I:/STUDY/python/project/Energy-prediction/1.0/out
├── data/processed_1.0
├── model
├── predictions
├── custom
└── figures
    ├── training
    ├── results
    ├── prediction
    └── custom

D:/Python-files/Energy-prediction/model
├── best_energy_mlp_1.0.pt
└── final_energy_mlp_1.0.pt
```

## 运行方式

在 `1.0` 目录下运行完整流程：

```bash
python main.py all
```

常用分步命令：

```bash
python main.py download
python main.py prepare --force-prepare
python main.py train
python main.py evaluate
python main.py visualize
```

默认要求 CUDA GPU 训练。若只是调试流程且允许 CPU，可追加 `--allow-cpu`。

生成测试集预测：

```bash
python main.py predict
```

调取训练模型预测默认自定义工况并生成图表：

```bash
python main.py custom --wind-speed 5 --flight-speed 8 --payload-g 250 --altitude 50 --duration-s 180
```

读取自定义工况 CSV：

```bash
python main.py custom --custom-csv path/to/custom_conditions.csv
```

自定义 CSV 推荐字段包括 `time`、`dt_seconds`、`route`、`wind_speed`、`wind_sin`、`wind_cos`、`programmed_speed_mps`、`actual_speed_mps`、`horizontal_speed_mps`、`vertical_speed_mps`、`relative_air_speed_mps`、`payload_kg`、`altitude_m`、`obstacle_agility_index`、`thermal_load_proxy`、`vision_energy_proxy_w`、`communication_energy_proxy_w`。缺失字段会自动补 0；若不提供 CSV，程序会按命令行参数生成一段起飞、巡航、降落的简化工况剖面。

## 数据与模型输出

- `out/data/processed_1.0/uav_energy_features.csv`：完整特征表，包含真实功率、区间电能、实测飞行参数和代理工况特征。
- `out/data/processed_1.0/train.csv`、`val.csv`、`test.csv`：按 `flight` 分组切分的数据，避免同一次飞行同时进入训练集和测试集。
- `out/data/processed_1.0/feature_metadata.json`：特征列、目标列、代理特征和切分方式说明。
- `out/model/scaler_1.0.json`：训练集特征和目标的标准化参数，预测和自定义工况预测都会读取它。
- `out/model/training_log_1.0.csv`：最终模型每轮训练损失、验证损失、学习率和候选模型名称。
- `out/model/tuning_results_1.0.csv`：每组候选超参数的验证损失、样本功率 WAPE、flight 能耗 WAPE 和综合选择分数。
- `out/model/evaluation_1.0.json`、`evaluation_1.0.csv`：测试集样本级功率误差和 flight 级能耗误差指标。
- `out/model/flight_energy_summary_1.0.csv`：每次测试飞行的真实能耗、预测能耗、误差、飞行时长、载荷和高度摘要。
- `out/model/power_bin_evaluation_1.0.csv`：按真实功率区间统计 MAE、RMSE、R2、MAPE 和 WAPE。
- `out/predictions/test_predictions_1.0.csv`：测试集逐采样点预测功率、预测区间电能和真实区间电能。
- `D:/Python-files/Energy-prediction/model/best_energy_mlp_1.0.pt`、`final_energy_mlp_1.0.pt`：模型训练权重，外部模型目录只保留这类 `.pt` 权重文件。

## 可视化输出

`python main.py visualize` 会生成：

- `out/figures/training/training_loss_history.png`：训练集和验证集损失曲线。
- `out/figures/training/learning_rate_schedule.png`：学习率变化曲线。
- `out/figures/training/hyperparameter_ranking.png`：候选超参数排序图。
- `out/figures/training/candidate_validation_wape.png`：候选模型验证集 WAPE 对比图。
- `out/figures/results/evaluation_metrics.png`：核心评估指标柱状图。
- `out/figures/results/flight_energy_actual_vs_predicted.png`：flight 级真实能耗与预测能耗对比。
- `out/figures/results/flight_energy_error.png`：flight 级能耗误差图。
- `out/figures/results/power_bin_mae.png`：功率分段 MAE 图。
- `out/figures/prediction/power_prediction_scatter.png`：测试集真实功率与预测功率散点图。
- `out/figures/prediction/power_residual_histogram.png`：预测残差分布图。
- `out/figures/prediction/flight_*_power_timeseries.png`：典型 flight 功率时序对比图。

`python main.py custom` 会生成：

- `out/custom/custom_scenarios_1.0.csv`：默认自定义工况输入表。
- `out/custom/custom_predictions_1.0.csv`：自定义工况逐时刻预测结果。
- `out/custom/custom_prediction_summary_1.0.json`：自定义工况能耗摘要。
- `out/figures/custom/custom_power_timeseries.png`：自定义工况预测功率曲线。
- `out/figures/custom/custom_cumulative_energy.png`：自定义工况累计能耗曲线。

## 建模逻辑

目标变量为瞬时电功率：

$$
P = \max(U \times I, 0)
$$

其中，$P$：瞬时电功率，单位 W；$U$：电池电压，单位 V；$I$：放电电流，单位 A。模型预测的是 $P$，随后根据采样时间间隔积分得到区间电能：

$$
E_i = \frac{\hat{P_i}\Delta t_i}{3600}
$$

其中，$E_i$：第 $i$ 个采样间隔的预测电能，单位 Wh；$\hat{P_i}$：模型预测的瞬时功率，单位 W；$\Delta t_i$：相邻采样点时间间隔，单位 s。

模型结构包括普通 MLP 和残差 MLP 两类候选：

$$
\hat{P} = f_\theta(x)
$$

其中，$x$：标准化后的工况特征向量；$f_\theta$：由全连接层、归一化层、SiLU、Dropout 和可选残差连接组成的神经网络；$\hat{P}$：预测功率。训练时使用 Huber Loss 和 AdamW，并对普通 MLP 与残差 MLP 的隐藏层宽度、Dropout、学习率和权重衰减进行调参。候选选择使用验证集 flight 级能耗 WAPE 与样本级功率 WAPE 的综合分数，使调参目标更接近电能能耗预测。

## 文件调用关系

`main.py` 是统一入口。`all` 模式先调用 `data_utils.prepare_dataset()` 完成数据准备，再调用 `train.train_model()` 完成 GPU 调参训练，然后调用 `evaluate.evaluate_model()` 生成预测和评估，最后调用 `visualize.generate_all_visualizations()` 输出图表。`custom` 模式调用 `visualize.predict_custom_scenario()`，会加载 `D:/Python-files/Energy-prediction/model` 中的训练权重和 `out/model/scaler_1.0.json` 中的标准化参数进行自定义工况预测。
