# 四轴无人机飞行能耗预测模型 1.0

本实验使用公开的 DJI Matrice 100 四轴无人机飞行数据，搭建普通 MLP 与残差 MLP 回归模型，用飞行期间的空气流速、飞行速度、载荷、高度、姿态变化、加速度，以及由运动状态推导出的热负荷、避障敏捷度、视觉算法耗能和通信耗能代理特征，预测瞬时电功率，并进一步积分得到单次飞行能耗。

## 数据来源

数据集：Data Collected with Package Delivery Quadcopter Drone
公开镜像：`https://www.modelscope.cn/datasets/OmniData/Data_Collected_with_Package_etc.git`
原始数据说明页：`https://theairlab.org/energy-dataset/`

数据包含 209 次飞行实验，原始字段包括 `wind_speed`、`wind_angle`、`battery_voltage`、`battery_current`、速度、角速度、线加速度、载荷、巡航高度、巡航速度和航线编号。数据集中没有直接采集视觉算法耗能、通信耗能、热量和避障敏捷度，因此本项目将这些工况作为代理特征从运动状态、风速、高度和载荷推导，不把它们标记为实测传感器字段。

## 目录结构

```text
I:/STUDY/python/project/Energy-prediction/1.0
├── config.py          # 统一管理路径、训练参数和输出文件名
├── data_utils.py      # 下载、解包、特征工程和按flight切分数据
├── model.py           # 自建普通MLP和残差MLP回归模型
├── train.py           # GPU训练、调参、权重和日志保存
├── predict.py         # 加载模型并生成预测CSV
├── evaluate.py        # 样本级功率、flight级能耗和功率分段评估
├── main.py            # 统一入口
├── requirements.txt   # Python依赖
└── output             # 预测输出目录
```

数据和模型输出目录：

```text
D:/Python-files/Energy-prediction/data
├── Data_Collected_with_Package_etc
├── dji_matrice_100
└── processed_1.0

D:/Python-files/Energy-prediction/model
├── best_energy_mlp_1.0.pt
├── final_energy_mlp_1.0.pt
├── scaler_1.0.json
├── training_log_1.0.csv
├── tuning_results_1.0.csv
├── loss_curve_1.0.png
├── evaluation_1.0.json
├── evaluation_1.0.csv
├── flight_energy_summary_1.0.csv
└── power_bin_evaluation_1.0.csv
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
python main.py predict
python main.py evaluate
```

默认要求 CUDA GPU 训练。若只是调试代码且允许 CPU，可使用：

```bash
python main.py all --allow-cpu
```

如需尝试对功率目标做对数变换，可显式指定：

```bash
python main.py train --target-transform log1p
```

当前最终模型采用 `--target-transform none`。

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

主要输入特征包括：

- 实测或原始参数：空气流速、风向、巡航速度、三轴速度、三轴角速度、三轴线加速度、载荷、高度、航线。
- 推导特征：实际速度、相对空气速度、横风分量、飞行阶段进度、动态加速度模长、角速度模长。
- 代理工况：`obstacle_agility_index`、`thermal_load_proxy`、`vision_energy_proxy_w`、`communication_energy_proxy_w`。

模型结构包括普通 MLP 和残差 MLP 两类候选：

$$
\hat{P} = f_\theta(x)
$$

其中，$x$：标准化后的工况特征向量；$f_\theta$：由全连接层、归一化层、SiLU、Dropout 和可选残差连接组成的神经网络；$\hat{P}$：预测功率。训练时使用 Huber Loss 和 AdamW，并对普通 MLP 与残差 MLP 的隐藏层宽度、Dropout、学习率和权重衰减进行调参。候选选择不只看验证集 Huber Loss，还使用验证集 flight 级能耗 WAPE 与样本级功率 WAPE 的综合分数，使调参目标更接近电能能耗预测。

## 输出说明

- `processed_1.0/uav_energy_features.csv`：完整特征表。
- `processed_1.0/train.csv`、`val.csv`、`test.csv`：按 `flight` 分组切分的数据。
- `processed_1.0/feature_metadata.json`：特征列、代理特征和目标定义。
- `best_energy_mlp_1.0.pt`：当前调参标准下的最佳模型权重。
- `scaler_1.0.json`：训练集特征和目标标准化参数。
- `tuning_results_1.0.csv`：候选超参、验证损失、验证集样本 WAPE、验证集 flight 能耗 WAPE 和综合选择分数。
- `training_log_1.0.csv`：最终模型每轮训练和验证损失。
- `loss_curve_1.0.png`：训练曲线。
- `output/test_predictions_1.0.csv`：测试集逐采样点预测功率和预测电能。
- `evaluation_1.0.json`、`evaluation_1.0.csv`：样本级与 flight 级评估指标。
- `flight_energy_summary_1.0.csv`：每次测试飞行的真实能耗、预测能耗和误差。
- `power_bin_evaluation_1.0.csv`：按真实功率区间统计预测误差，用于观察低功率和高功率工况表现。

## 文件调用关系

`main.py` 是统一入口。`all` 模式先调用 `data_utils.prepare_dataset()` 完成数据准备，再调用 `train.train_model()` 完成 GPU 调参训练，最后调用 `evaluate.evaluate_model()`；评估函数内部会调用 `predict.predict_from_csv()` 生成测试集预测文件。所有路径和默认训练参数只从 `config.py` 读取。
