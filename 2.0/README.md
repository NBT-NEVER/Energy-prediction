# 四轴无人机飞行能耗预测模型 2.1：两阶段 TCN + RLS 选择

本版本将模型选择改为严格的两阶段流程：阶段一只根据 TCN 验证集原始预测选择时间窗；阶段二固定该时间窗和同一份 TCN 验证预测，逐点运行 72 组 RLS 参数；最终固定两者，在测试集上只做一次评估。

## 运行

在 `2.0` 目录执行：

```bash
python main.py all
```

也可以分步执行 `python main.py prepare`、`python main.py train`、`python main.py evaluate` 和 `python main.py visualize`。默认使用 CUDA，可通过 `--device cuda:0` 指定设备，可用 `--data-dir`、`--save-dir`、`--out-dir` 覆盖路径。

## 数据和代码

原始数据默认位于 `D:/Python-files/Energy-prediction/data`，模型权重默认位于 `D:/Python-files/Energy-prediction/model`，其余产物写入 `2.0/out`。本次运行仅使用 R1 航线：清洗后 227551 条记录、182 个 flight；train/val/test 为 158625/32820/36106 条，对应 126/28/28 个 flight；输入维度为 22。

```text
2.0/
├── config.py          # 路径、训练参数和两阶段搜索网格
├── data_utils.py      # 数据下载、清洗、特征工程和flight切分
├── model.py           # 因果TCN和RLSCorrector
├── train.py           # 两阶段选择、最终训练和区间校准
├── predict.py         # TCN推理和RLS逐点校正
├── evaluate.py        # 测试集样本级、flight级评估
├── visualize.py       # 训练、评估和预测图表
├── main.py            # 统一命令行入口
├── progress.py        # 终端进度显示
├── terminal_logger.py # UTF-8终端日志
├── uncertainty.py     # 预测区间校准
├── device_utils.py    # CUDA设备选择
├── requirements.txt   # Python依赖
└── out/               # 2.1独立输出目录
```

`main.py` 依次调用 `prepare_dataset()`、`train_model()`、`evaluate_model()` 和 `generate_all_visualizations()`。`train.py` 使用 `build_sequence_arrays()` 构造因果序列，使用 `run_training_loop()` 训练 TCN，使用 `apply_rls_correction()` 按 flight/time 顺序执行 RLS。

## 阶段一：TCN 时间窗

训练集先估计典型采样间隔。本次约为 `0.12 s`，每个候选换算为整数步数：

$$L=\max\left(2,\operatorname{round}\left(\frac{T}{\Delta t}\right)\right)$$

其中：`$T$` 为候选时间窗（s），`$\Delta t$` 为训练集典型采样间隔（s），`$L$` 为输入步数。

候选为 `0.4、0.6、0.8、1.0、1.2、1.5、2.0、2.5、3.0、4.0 s`，本次对应 `3、5、7、8、10、12、17、21、25、33` 步。所有候选使用同一 TCN 结构 `[64,64,64,32]`、卷积核 3、Dropout 0.08、学习率 `3e-4`、权重衰减 `1e-4`、Huber 参数 0.65 和 batch size 4096；每个候选单独初始化并固定完整训练 30 轮。每轮都计算验证损失，保存最低验证损失权重，调参阶段不启用早停。

窗口选择只使用 TCN 原始验证指标：

$$S_{TCN}=\mathrm{WAPE}_{flight}+0.2\,\mathrm{WAPE}_{sample}$$

其中：`$S_{TCN}$` 为窗口选择分数，`$\mathrm{WAPE}_{flight}$` 为 flight 能耗 WAPE，`$\mathrm{WAPE}_{sample}$` 为采样点功率 WAPE，均在验证集计算且分数越低越好。

本次选择 `0.8 s / 7 步`，TCN 验证选择分数为 `3.12761984`。10 个窗口的验证指标位于 `out/model/tuning_results_2.1.csv`，所有窗口的 30 轮日志位于 `out/model/training_log_2.1.csv` 的 `stage1_tcn_window` 行。

## 阶段二：RLS 超参数

阶段二不重新比较时间窗、不重新训练 TCN，72 组候选共用阶段一选中窗口的同一份 TCN 验证预测。网格如下：

| 参数 | 候选值 |
|---|---|
| 遗忘因子 | 0.95、0.97、0.98、0.99、0.995、0.999 |
| 初始协方差 | 100、500、1000、5000 |
| 预热长度 | 0.5 s、1.0 s、2.0 s |

每组都沿 flight 和 time 顺序逐点执行“先预测、后更新”。预热长度内只输出校正结果，达到预热时间后才用当前实测功率更新 RLS。最终选择 `遗忘因子=0.98、初始协方差=100、预热长度=0.5 s`，验证组合分数为 `1.62643982`。72 组结果位于 `out/model/rls_tuning_results_2.1.csv`。

## 最终训练和测试结果

固定 `0.8 s / 7 步` 和上述 RLS 参数后，TCN 最多训练 80 轮，仍使用 ReduceLROnPlateau、patience=10 早停以及验证损失最低权重回载。实际运行 61 轮，第 51 轮取得最佳验证损失 `0.02035396`，保存的不是最后一轮权重。

测试集只在所有选择完成后评估一次：

| 指标 | TCN | TCN + RLS | 单位 |
|---|---:|---:|---|
| 样本功率 MAE | 33.9359 | 30.8353 | W |
| 样本功率 R2 | 0.9471 | 0.9533 | 无量纲 |
| 样本功率 WAPE | 8.3533 | 7.5901 | % |
| flight能耗 MAE | 0.5064 | 0.0966 | Wh |
| flight能耗 R2 | 0.9792 | 0.9992 | 无量纲 |
| flight能耗 WAPE | 2.3146 | 0.4413 | % |

结果位于 `out/predictions/test_predictions_2.1.csv`、`out/model/evaluation_2.1.json`、`evaluation_2.1.csv`、`flight_energy_summary_2.1.csv` 和 `power_bin_evaluation_2.1.csv`。最终权重为 `D:/Python-files/Energy-prediction/model/best_energy_tcn_rls_2.1.pt` 和 `final_energy_tcn_rls_2.1.pt`。

## 输出说明

- `out/data/processed_2.1/`：R1 特征表、数据切分、字段元数据和排除航线记录。
- `out/model/scaler_2.1.json`：训练集标准化参数、典型采样间隔和功率尺度。
- `out/model/training_log_2.1.csv`：10 个窗口各 30 轮调参记录及最终训练记录。
- `out/model/tuning_results_2.1.csv`：10 个 TCN 窗口的采样步数和验证指标。
- `out/model/rls_tuning_results_2.1.csv`：72 组 RLS 参数及验证指标。
- `out/model/uncertainty_calibration_2.1.npz/json`：在线和固定 RLS 残差校准数据。
- `out/figures/`：损失、学习率、候选窗口、评估、功率分箱和 flight 时序图。

checkpoint 保存窗口步数和选中的 RLS 三项参数，`predict`、`evaluate`、`visualize`、`custom` 都读取 checkpoint，不会回退到默认 RLS 参数。测试集不参与窗口选择、RLS 选择、学习率调整、早停或 TCN 权重更新。
