# TCN-comparison：五组离线功率与能耗预测

## 目录

- [实验目标](#实验目标)
- [五组对照](#五组对照)
- [数据与输出位置](#数据与输出位置)
- [统一参数](#统一参数)
- [模型与数据处理](#模型与数据处理)
- [原始字段与派生字段](#原始字段与派生字段)
- [运行方法](#运行方法)
- [实验结果](#实验结果)
- [图表](#图表)
- [核验与文档更新](#核验与文档更新)
- [文件职责](#文件职责)
- [结果边界](#结果边界)

## 实验目标

本目录只评价离线 TCN 的逐时刻功率和累计能耗预测，不包含 RLS、置信区间、在线更新和电池决策。五组实验使用相同的训练/验证/测试 flight 划分、损失函数、优化器、通道数、TCN 时间窗候选和训练轮数。

## 五组对照

| 组别 | 输入 | 隐藏激活 | 目的 |
| --- | --- | --- | --- |
| A | 原始 10 维 | 当前 SiLU/ReLU 混合 | 4.1 实现的重新训练基线 |
| B | 原始 10 维 | 统一 SiLU | 检查残差出口 ReLU 替换为 SiLU 的影响 |
| C | 原始 10 维 + 5 维 SMA，共 15 维 | 当前 SiLU/ReLU 混合 | 检查 10 步 SMA 特征的影响 |
| D | 原始 10 维 + 5 维 SMA，共 15 维 | 统一 SiLU | 检查组合效果及交互影响 |
| E | 原始 10 维 + 5 维 SMA，共 15 维 | 统一 LeakyReLU(0.05) | 检查统一 LeakyReLU 激活的附加效果 |

这里的“15 维 SMA”表示在原始 10 维基础上增加 5 个 SMA 派生通道：三维位置、风速和风向。SMA 不替换原始输入，按 flight 分组、只使用当前和历史值计算，窗口固定为 10 个重采样点，即 2.0 s 名义窗口。

## 数据与输出位置

输入数据来自 DJI planned 加工表：

```text
D:/Python-files/Energy-prediction/data/dji_matrice_100_data/TCN-planned/
```

包含 124 个训练 flight、29 个验证 flight 和 47 个测试 flight。模型统一保存到：

```text
D:/Python-files/Energy-prediction/model/TCN-planned/
```

日志、CSV 指标和 PNG 图表保存在当前工程的 `out/` 下。原始数据、4.1 加工表和 A-D 旧模型不会被覆盖；E 使用独立的模型、scaler、日志和结果文件名。

## 统一参数

| 参数 | 设置 |
| --- | --- |
| 重采样间隔 | 0.2 s |
| SMA 窗口 | 10 步，因果 rolling mean |
| TCN 窗口搜索 | 1、2、5、10 s，共 4 个候选 |
| 搜索训练 | 每个候选 8 轮 |
| 正式训练 | 80 轮 |
| 提前停止 | 不启用，候选完整8轮、正式完整80轮 |
| 通道 | `(32, 32)` |
| 卷积核 | 3 |
| 膨胀率 | 1、2 |
| Dropout | 0.05 |
| 损失 | SmoothL1Loss |
| 优化器 | AdamW |
| 学习率 | `3e-4` |
| 权重衰减 | `1e-4` |
| 批大小 | 2048 |
| 梯度裁剪 | 2.0 |
| 随机种子 | 42 |

每组会独立搜索最佳 TCN 窗口。窗口选择分数为：1 s 能耗 WAPE + 0.2 × 功率 WAPE + 0.5 × flight 总能耗 WAPE。正式模型使用该组验证集分数最低的窗口。每个候选及正式训练开始前重置种子42。搜索和正式训练都完整执行指定轮数，正式权重保留验证SmoothL1损失最低的轮次，不使用测试指标选择权重。

每个窗口候选开始前重新设置随机种子，使候选不受搜索顺序影响；结果只代表当前数据划分和单次随机种子。

## 模型与数据处理

TCN 采用因果卷积，序列最后一个时间位置输出当前功率。输入标准化器只用训练集拟合，测试集只使用冻结参数。功率预测反标准化后裁剪为非负值，再按 0.2 s 积分得到区间能耗；按 flight 汇总得到任务总能耗。

10 步 SMA 对每个选定连续字段 $x$ 计算：

$$
a_t=\frac{1}{n_t}\sum_{q=0}^{n_t-1}x_{t-q},\qquad n_t=\min(10,t+1)
$$

其中：

$a_t$：当前时刻的 SMA 派生特征。

$x_{t-q}$：当前时刻及历史时刻的原始连续特征。

$n_t$：序列起点处实际可用的历史点数。

$t$：单个flight内从0开始的采样索引。

$q$：向历史回溯的采样点数。

统一 SiLU 组只替换残差相加后的 ReLU，卷积块内部和预测头中间层保持 SiLU，预测头最后一层保持线性。E组将卷积块、残差出口和预测头中间层的隐藏激活统一为 LeakyReLU(negative_slope=0.05)，最后一层保持线性。

完整结构为两个残差块，每块含两层因果Conv1d，每层后接GroupNorm(1,32)、隐藏激活和Dropout(0.05)；A/C使用SiLU卷积激活与ReLU残差出口，B/D使用SiLU，E的卷积激活、残差出口和预测头中间层全部使用LeakyReLU(0.05)。输入维数不等于32时跳连用1×1卷积。最后取窗口末端编码，接Linear(32,32)、组别对应的隐藏激活、Linear(32,1)。A/B有12001个参数，C/D/E有12641个参数。

GroupNorm使用单个输入窗口内的时间与通道统计量。本工程按目标时刻分别构造历史窗口；不能将整段轨迹一次前向或单纯卷积缓存视为与当前逐窗口推理等价。SMA在完整flight上先计算，再构造TCN窗口，因此可以利用窗口左侧仍然有效的任务历史。

## 原始字段与派生字段

| 顺序 | 原始字段 | 单位 | 含义 | C/D/E是否增加SMA |
| --- | --- | --- | --- | --- |
| 1 | `time_s` | s | 任务相对时间 | 否 |
| 2 | `task_duration_s` | s | 任务总时长 | 否 |
| 3 | `position_x_m` | m | 归一化位置X | 是 |
| 4 | `position_y_m` | m | 归一化位置Y | 是 |
| 5 | `position_z_m` | m | 归一化位置Z | 是 |
| 6 | `wind_speed` | m/s | 风速 | 是 |
| 7 | `wind_angle` | rad | 风向角 | 是 |
| 8 | `payload_g` | g | 有效载荷 | 否 |
| 9 | `uav_type` | 1/2/3/4 | 机型编号 | 否 |
| 10 | `planned` | 0/1/2 | 停机/原地启动/飞行状态 | 否 |

新增通道按位置X、位置Y、位置Z、风速、风向的顺序追加，名称统一为对应原字段加 `_sma10`，单位不变。每列独立用训练集均值和标准差缩放。真实功率、区间能耗、flight编号和route标签均不作为神经网络输入。

## 运行方法

在项目根目录执行：

```powershell
python TCN-comparison/TCN/main.py all --device cuda
```

分阶段运行：

```powershell
python TCN-comparison/TCN/main.py prepare --force-prepare
python TCN-comparison/TCN/main.py tune --device cuda
python TCN-comparison/TCN/main.py train --device cuda
python TCN-comparison/TCN/main.py evaluate --device cuda
python TCN-comparison/TCN/main.py plot --device cuda
python TCN-comparison/TCN/main.py report
python -m unittest discover -s TCN-comparison/TCN/tests -v
```

只运行某组时添加 `--group A`、`--group B`、`--group C`、`--group D` 或 `--group E`。支持 `--data-dir`、`--save-dir`、`--out-dir`、`--seed`、`--epochs`、`--tune-epochs` 和 `--batch-size` 覆盖。若改变训练超参数，应使用独立模型与输出目录并重新搜索、训练；不能将旧窗口搜索表当作新配置的结果。如果机器没有 CUDA，可显式使用 `--device cpu`，但完整五组训练会明显变慢。

五组均按当前配置完成训练；每组窗口候选逐轮记录位于 `out/logs/`，正式训练记录为80轮，终端日志为 `TCN-comparison_terminal.log`。

## 实验结果

运行完成后，五组汇总结果位于 [`out/results/TCN-comparison_all_metrics.csv`](./out/results/TCN-comparison_all_metrics.csv)，每组 JSON 指标、测试预测和窗口搜索表也在同一目录。下表由保存的实际测试指标自动更新。

<!-- RESULTS_START -->

| 组别 | 最佳窗口(s) | 输入维度 | 功率WAPE(%) | 1s能耗WAPE(%) | flight能耗WAPE(%) | 功率RMSE(W) | flight能耗RMSE(Wh) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.0000 | 10 | 10.9428 | 9.6081 | 2.7155 | 68.3027 | 0.8370 |
| B | 5.0000 | 10 | 10.2656 | 8.8946 | 3.7001 | 63.7145 | 1.3742 |
| C | 2.0000 | 15 | 10.3059 | 8.9447 | 3.1647 | 64.9338 | 1.0559 |
| D | 5.0000 | 15 | 10.3832 | 9.0215 | 3.6226 | 64.7863 | 1.4560 |
| E | 5.0000 | 15 | 10.4266 | 9.0953 | 3.4659 | 64.4370 | 1.2261 |

| 组别 | 激活 | TCN步数 | 候选轮数 | 正式轮数 | 最佳轮次 | 参数量 | 正式训练秒数 | 种子 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | mixed | 5 | 8 | 80 | 60 | 12001 | 135.1876 | 42 |
| B | silu | 25 | 8 | 80 | 78 | 12001 | 140.3945 | 42 |
| C | mixed | 10 | 8 | 80 | 63 | 12641 | 141.9115 | 42 |
| D | silu | 25 | 8 | 80 | 69 | 12641 | 142.8682 | 42 |
| E | leaky_relu | 25 | 8 | 80 | 74 | 12641 | 142.3446 | 42 |

当前时间步功率WAPE最低的是 B 组，为 10.2656%。

当前1 s窗口能耗WAPE最低的是 B 组，为 8.8946%。

当前flight总能耗WAPE最低的是 A 组，为 2.7155%。

B 对 A（统一SiLU）的总能耗WAPE上升 0.9846 个百分点，相对上升 36.26%。

C 对 A（增加SMA）的总能耗WAPE上升 0.4492 个百分点，相对上升 16.54%。

D 对 B（SiLU条件下增加SMA）的总能耗WAPE下降 0.0774 个百分点，相对下降 2.09%。

D 对 C（SMA条件下统一SiLU）的总能耗WAPE上升 0.4580 个百分点，相对上升 14.47%。

E 对 D（同15维输入，统一SiLU改为统一LeakyReLU(0.05)）的总能耗WAPE下降 0.1567 个百分点，相对下降 4.33%。

E 对 C（同15维输入，混合激活改为统一LeakyReLU(0.05)）的总能耗WAPE上升 0.3013 个百分点，相对上升 9.52%。

以上是固定种子42的单次完整实验；各组独立搜索窗口，差异同时包含所选上下文变化。未进行多种子重复或新的独立留出验证，不据此宣称统计显著性或跨数据集优势。

数据：训练 113766 行/124 个 flight；验证 26300 行/29 个 flight；测试 48724 行/47 个 flight。完整推导、逐任务误差和逐图说明见 [当前输出报告](./out/outreadme.md)。

<!-- RESULTS_END -->

## 图表

`main.py all` 或 `main.py plot` 会生成以下图形的PNG与SVG版本：

- `out/figures/TCN-comparison_power_wape.png`：功率 WAPE 对比。
- `out/figures/TCN-comparison_window_energy_wape.png`：1 s 窗口能耗 WAPE 对比。
- `out/figures/TCN-comparison_flight_energy_wape.png`：flight 总能耗 WAPE 对比。
- `out/figures/TCN-comparison_representative_power.png`：代表 flight 的真实功率和五组预测曲线。
- `out/figures/TCN-comparison_flight_energy_scatter.png`：逐 flight 实际与预测总能耗散点图。
- `out/figures/TCN-comparison_representative_energy.png`：同一代表flight的累计能耗曲线。
- `out/figures/TCN-comparison_flight_error_distribution.png`：逐flight绝对误差分布。
- `out/figures/TCN-comparison_window_search.png`：全部窗口验证选择分数。
- `out/figures/TCN-comparison_training_loss.png`：五组80轮正式训练/验证损失。

代表任务按测试集时长的中位位置选择，不挑选误差最小的flight。图中没有置信带或在线反馈。所有图的源CSV、轴单位和结论边界见 [输出报告](./out/outreadme.md)。

![五组任务能耗误差](./out/figures/TCN-comparison_flight_energy_wape.png)

![五组窗口搜索](./out/figures/TCN-comparison_window_search.png)

## 核验与文档更新

E的模型文件为 `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_E_best.pt`，标准化器为同目录 `TCN-comparison_E_scaler.json`；输入15维，激活元数据为leaky_relu，负半轴斜率由模型实现固定为0.05。保存轮次以checkpoint元数据为准，训练日志保留完整80轮。

[测试日志](out/logs/E_validation_tests.log)记录6项行为测试通过，[产物核验](out/logs/E_artifact_audit.json)记录模型指纹、历史结构前向兼容与数据划分核验。E环境快照见 [TCN-comparison_runtime_E.json](out/TCN-comparison_runtime_E.json)。

仅更新报告使用 `python TCN-comparison/TCN/main.py report`，仅重绘图表使用plot模式；均不训练。README结果区和out报告由report.py更新，稳定说明在本README维护，工程历史在improve.md按日期追加。全部五组参数、窗口表与结果源文件见 [当前输出报告](out/outreadme.md)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 集中管理 D 盘数据、D 盘模型、输出路径和统一超参数 |
| `data_utils.py` | 复制加工数据、计算 10 步 SMA、检查输入字段 |
| `model.py` | 因果TCN、混合激活、统一SiLU和统一LeakyReLU(0.05)结构 |
| `train.py` | 数据标准化、序列构造、20 轮窗口搜索和 80 轮正式训练 |
| `evaluate.py` | 生成离线功率、1 s 能耗和 flight 总能耗指标 |
| `visualize.py` | 生成五组对比图表 |
| `main.py` | 统一命令行入口 |
| `report.py` | 根据当前CSV/JSON同步README结果区和out/outreadme.md |
| `tests/test_pipeline.py` | 检查SMA分组、序列边界、标签隔离、能量单位、模型路径及LeakyReLU负半轴行为 |

## 结果边界

本实验的测试指标只反映任务前规划特征到功率/能耗标签的离线映射。没有 RLS 或真实窗口反馈，因此不能把结果解释为在线校正性能；没有置信区间，因此不支持覆盖率或安全裕度结论。正式报告应同时给出五组的实际运行参数、最佳窗口、均值指标、逐 flight 误差和图表路径。

五组各自选择窗口，因此比较的是相同搜索预算下的方案表现；固定结构下的激活或SMA纯效应需要另做固定窗口对照。本次只使用种子42，未完成多种子稳定性试验。历史数据中的规划位置与风来自实飞测量代理，历史电机状态也使用实际功率阈值标注，不能将该离线结果直接当作独立计划/预报来源下的部署精度。

本README保存工程入口、详细统一参数和本次实测结果摘要；[improve.md](./improve.md)追加实施与验证记录；[out/outreadme.md](./out/outreadme.md)随运行同步，记录完整窗口表、最差flight、算法口径和逐图说明。D盘数据和模型不纳入Git，`out/`也沿用仓库默认忽略策略，源代码和结果摘要可独立查阅。
