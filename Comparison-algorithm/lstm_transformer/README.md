# LSTM-Transformer

**算法来源：** Feng 等，2024，Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework。正文引用 <sup>[A5]</sup>。

## 1. 方法概述

Feng 等的方法先由 LSTM 提取短期变化，再由 Transformer 建模较长依赖，原始对象是电动汽车。当前 UAV 适配版在 $D=32/64$、1/2 个 Encoder 等候选中选中 $D=64$、1 层 LSTM、2 层 Transformer Encoder、4 个注意力头、128 维前馈层和 Dropout 0.10，共 105,089 个可训练参数。

1. **输入与填充层**：同一 flight 的当前点和之前最多 16 点组成 $B\times17\times23$ 张量；起点不足 17 步时复制最早记录到左侧，23 个特征按训练集统计量标准化。
2. **线性投影层**：`Linear(23,64)` 独立作用于每个时间步，把不同量纲的 23 维输入映射到统一的 64 维模型空间，输出 $B\times17\times64$。
3. **可学习位置编码层**：加入形状 $1\times17\times64$ 的参数矩阵，使注意力能够区分窗口首端、内部和当前时刻；位置参数与投影结果逐元素相加。
4. **单层 LSTM**：64 个记忆单元依次处理 17 个位置，输出仍为 $B\times17\times64$。该层先汇总相邻点的短期顺序信息，再交给自注意力。
5. **Transformer Encoder 1**：4 个注意力头并行工作，每个头的键/查询维度为 $64/4=16$；多头输出经残差连接和 LayerNorm，再通过 `Linear(64,128) → GELU → Dropout → Linear(128,64)` 前馈网络及第二组残差归一化。
6. **Transformer Encoder 2**：重复同样的 4 头注意力和 128 维前馈结构，在第一层关系表示上再次组合窗口内位置。实现未使用因果 mask，但整个 17 步输入窗只包含当前及过去记录，因此不会读取预测时刻之后的数据。
7. **归一化与回归头**：取第二个 Encoder 最后时间位置的 64 维向量，经 `LayerNorm(64) → Linear(64,32) → GELU → Linear(32,1)` 输出标准化功率，再还原为 W 并截断为非负值。
8. **训练选择层**：使用 AdamW、SmoothL1、梯度裁剪和验证早停。本次选中候选的最佳保存轮次为第 21 轮。

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V$$
$$head_r=\operatorname{Attention}(XW_r^Q,XW_r^K,XW_r^V),\qquad \operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_h)W^O$$
其中：

- $X$：LSTM 输出的窗口序列矩阵，每条样本为 $17\times64$。
- $Q$、$K$、$V$：查询、键和值矩阵，用于计算位置间权重并加权汇总信息。
- $d_k$：单个注意力头的键维度，本次为 16；$\sqrt{d_k}$：抑制点积随维度增大的缩放项。
- $W_r^Q,W_r^K,W_r^V$：第 $r$ 个头的查询、键、值投影矩阵。
- $head_r$：第 $r$ 个注意力头输出；$r$：注意力头索引。
- $h$：注意力头总数，本次为 4；$\operatorname{Concat}$：沿特征维拼接各头输出。
- $W^O$：把多头拼接结果映射回 64 维的输出投影矩阵。
- $\operatorname{softmax}$：在键位置维度上把缩放点积转换为和为 1 的注意力权重。

## 2. 统一实验条件

本算法与其他方法使用同一 R1 源数据和同一完整 flight 划分：训练 198,469 条、126 个 flight；验证 41,052 条、28 个 flight；测试 45,237 条、28 个 flight。实际模型输入为：23 个公共特征，17 步（约 2.0 s）历史至当前窗口。测试功率目标统一为 `power_w`，能耗统一按每条记录的 `dt_seconds` 积分；测试集只用于最终报告，不参与候选选择。

本次最终测试结果：功率 MAE=29.630 W，RMSE=45.541 W，R²=0.9586，WAPE=7.293%；flight 能耗 MAE=0.556 Wh，RMSE=0.766 Wh，R²=0.9734，WAPE=2.542%。

## 3. 验证集调参记录

| 候选 | 验证最优轮次 | 验证功率WAPE(%) | 验证能耗WAPE(%) | 选择分数 | 最终选择 | model_dim | lstm_layers | transformer_layers | heads | feedforward | dropout | learning_rate | weight_decay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 6.2906 | 1.5571 | 7.0691 | 否 | 32 | 1 | 1 | 4 | 64 | 0.0800 | 0.0020 | 0.0001 |
| 2 | 21 | 6.2213 | 1.6296 | 7.0361 | 是 | 64 | 1 | 2 | 4 | 128 | 0.1000 | 0.0010 | 0.0001 |

最终选择候选 `2`：`model_dim=64.0000`、`lstm_layers=1.0000`、`transformer_layers=2.0000`、`heads=4.0000`、`feedforward=128.0000`、`dropout=0.1000`、`learning_rate=0.0010`、`weight_decay=0.0001`。验证集选择分数为 `7.0361`；测试集没有参与候选筛选。

## 4. 评价指标和变量

### 评价指标与能耗积分

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
$$
$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\qquad
\mathrm{MAPE}=\frac{100\%}{n_0}\sum_{|y_i|>\epsilon}\left|\frac{y_i-\hat y_i}{y_i}\right|
$$
$$
\mathrm{WAPE}=\frac{\sum_i|y_i-\hat y_i|}{\sum_i|y_i|}\times100\%,\qquad
E_f=\sum_{i\in f}P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

其中：

- $y_i$：第 $i$ 个真实值；在样本级为功率 W，在 flight 级为能耗 Wh。
- $\hat y_i$：与 $y_i$ 对应的预测值；$y_i-\hat y_i$：评价指标采用的误差差值。
- $n$：参与 MAE、RMSE、$R^2$ 或 WAPE 的样本数；$i$：样本索引。
- $n_0$：满足 $|y_i|>\epsilon$ 的 MAPE 有效样本数；$\epsilon=10^{-6}$ 是近零筛选阈值。
- $\bar y$：全部真实值的算术平均数。
- $P_{f,i}$：flight $f$ 第 $i$ 个采样点的功率，单位 W。
- $\Delta t_{f,i}$：对应记录的真实采样间隔，单位 s。
- $E_f$：flight $f$ 的总能耗，单位 Wh；$f$：flight 编号。
- $3600$：从 W·s 换算到 Wh 的秒数因子。

MAE、RMSE、MAPE、WAPE 越小越好，$R^2$ 越接近 1 越好。能耗先按真实 `dt_seconds` 积分，再在 28 个 flight 上计算指标，避免把不等间隔采样误当成等权样本。

本实验的选择分数为：

$$S_{val}=\mathrm{WAPE}_{power,val}+0.5\,\mathrm{WAPE}_{energy,val}$$

其中：

- $S_{val}$：验证集候选选择分数，越小越好。
- $\mathrm{WAPE}_{power,val}$：验证集逐点功率 WAPE，单位 %。
- $\mathrm{WAPE}_{energy,val}$：验证集按 flight 汇总后的能耗 WAPE，单位 %。
- $0.5$：能耗 WAPE 在选择分数中的固定权重。

该分数只在 28 个 validation flight 上计算，用于比较候选结构、正则强度和相位阈值；测试集 28 个 flight 在参数固定后才运行。

### 变量、坐标轴和字段词典

| 字段或指标 | 中文全称 | English full name | 单位/范围 | 计算方式和含义 |
|---|---|---|---|---|
| `flight` | 飞行任务编号 | Flight identifier | 无量纲 | 同一编号的连续记录属于同一任务，能耗按该字段分组。 |
| `route` | 航线编号 | Route identifier | — | 本实验仅保留 R1。 |
| `time` | 飞行内相对时间 | Flight-relative time | s | 每个 flight 从 0 s 开始的时间坐标，所有时序图横轴使用它。 |
| `dt_seconds`（$\Delta t_i$） | 采样时间间隔 | Sampling time interval | s | 相邻时间戳之差；能耗积分使用真实值，不用固定步长替代。 |
| `power_w`（$y_i$） | 实测瞬时功率 | Measured instantaneous power | W | `max(battery_voltage × battery_current, 0)`，作为监督目标和散点图横坐标。 |
| `predicted_power_w`（$\hat y_i$） | 预测瞬时功率 | Predicted instantaneous power | W | 算法输出的非负功率，作为时序图预测线和散点图纵坐标。 |
| `residual_w`（$e_i$） | 功率残差 | Power residual | W | $e_i=\hat y_i-y_i$；正值表示高估，负值表示低估。 |
| `absolute_error_w` | 功率绝对误差 | Absolute power error | W | $|e_i|$，用于箱线图、CDF、分箱误差和小提琴图。 |
| `actual_energy_wh` | 实测采样能量 | Measured interval energy | Wh | $P_i\Delta t_i/3600$。 |
| `predicted_energy_wh` | 预测采样能量 | Predicted interval energy | Wh | $\hat P_i\Delta t_i/3600$。 |
| `actual_energy_wh`（flight汇总） | 实测 flight 总能耗 | Measured flight energy | Wh | 同一 flight 的 `actual_energy_wh` 求和。 |
| `predicted_energy_wh`（flight汇总） | 预测 flight 总能耗 | Predicted flight energy | Wh | 同一 flight 的 `predicted_energy_wh` 求和。 |
| `energy_error_wh` | flight 能耗误差 | Flight-energy error | Wh | 预测总能耗减实测总能耗；正值为任务级高估。 |
| `phase_name` | 飞行阶段 | Flight phase | idle/ascent/descent/cruise | 由 $v_z$ 和水平速度阈值推导，不是人工标注。 |
| MAE | 平均绝对误差 | Mean absolute error | W 或 Wh | 所有误差绝对值的算术平均，代表典型偏差。 |
| RMSE | 均方根误差 | Root mean square error | W 或 Wh | 误差平方平均后开方，对尖峰误差更敏感。 |
| $R^2$ | 决定系数 | Coefficient of determination | 无量纲 | 相对于真实均值基线的解释度。 |
| MAPE | 平均绝对百分比误差 | Mean absolute percentage error | % | 逐点相对误差平均；接近 0 的真实功率被排除。 |
| WAPE | 加权绝对百分比误差 | Weighted absolute percentage error | % | 总绝对误差除以真实值绝对和，适合跨功率段总体比较。 |
| CDF | 经验累积分布函数 | Empirical cumulative distribution function | 0–1 | $F(a)=\#\{|e_i|\le a\}/n$，读取某误差阈值以内的样本比例。 |
| P50/P90/P95 | 误差分位数 | Error percentiles | W | 绝对误差排序后第 50%、90%、95% 位置，反映典型和尾部风险。 |


下表列出公共数据中的 23 个候选字段。当前算法是否直接使用某字段，以第 2 节的实际输入和第 1 节逐层结构为准。

| 变量 | 中文全称 | English full name | 单位/范围 | 在模型中的作用 |
|---|---|---|---|---|
| `time` | 飞行内相对时间 | Flight-relative time | s | 飞行内相对时间戳，单位为秒。 |
| `dt_seconds` | 采样时间间隔 | Sampling time interval | s | 相邻采样点时间间隔，单位为秒；用于把功率积分为能耗。 |
| `flight_progress` | 飞行进度 | Flight progress | 0–1 | 当前时间除以本次飞行最大时间，范围 0 到 1，表示飞行进度。 |
| `wind_speed` | 风速 | Wind speed | m/s | 风速，单位为米每秒。 |
| `wind_sin` | 风向正弦分量 | Sine of wind direction | — | 风向角的正弦分量，用于连续表达风向。 |
| `wind_cos` | 风向余弦分量 | Cosine of wind direction | — | 风向角的余弦分量，用于连续表达风向。 |
| `programmed_speed_mps` | 规划速度 | Programmed speed | m/s | 任务规划速度，单位为米每秒。 |
| `actual_speed_mps` | 实际合速度 | Actual resultant speed | m/s | 三轴速度合成后的实际速度，单位为米每秒。 |
| `horizontal_speed_mps` | 水平速度 | Horizontal speed | m/s | 水平面速度合成值，单位为米每秒。 |
| `vertical_speed_mps` | 垂直速度 | Vertical speed | m/s | 垂直速度分量，单位为米每秒；正负表示升降方向。 |
| `vertical_speed_abs_mps` | 垂直速度绝对值 | Absolute vertical speed | m/s | 垂直速度绝对值，单位为米每秒，表示垂直机动强度。 |
| `relative_air_speed_mps` | 相对空速 | Relative air speed | m/s | 相对空气速度，结合机体速度和风速计算，单位为米每秒。 |
| `wind_alignment` | 风速方向对齐度 | Wind alignment cosine | −1–1 | 水平速度与风速方向的余弦相似度，范围 -1 到 1。 |
| `wind_cross_component_mps` | 横向风速分量 | Crosswind component | m/s | 风速在水平速度横向上的分量，单位为米每秒。 |
| `payload_kg` | 载荷质量 | Payload mass | kg | 载荷质量，单位为千克。 |
| `altitude_m` | 飞行高度 | Flight altitude | m | 飞行高度，单位为米。 |
| `dynamic_accel_norm` | 动态加速度模 | Dynamic acceleration norm | m/s² | 去除重力影响后的三轴线性加速度合成强度，单位为米每二次方秒。 |
| `angular_rate_norm` | 角速度模 | Angular-rate norm | rad/s | 三轴角速度合成强度，单位为弧度每秒。 |
| `obstacle_agility_index` | 障碍机动代理指标 | Obstacle-agility proxy index | — | 由加速度、角速度和垂直速度构成的机动性代理指标，无量纲。 |
| `thermal_load_proxy` | 热负荷代理量 | Thermal-load proxy | — | 由相对气流、载荷和爬升状态估计的热负荷代理指标，用于表征附加能耗。 |
| `vision_energy_proxy_w` | 视觉计算附加功率代理量 | Vision-energy power proxy | W | 由速度、机动性和高度估计的视觉计算附加功率代理值，单位为瓦特。 |
| `communication_energy_proxy_w` | 通信附加功率代理量 | Communication-energy power proxy | W | 由高度、风速和速度估计的通信附加功率代理值，单位为瓦特。 |
| `route_R1` | R1航线独热变量 | R1 route one-hot variable | 0/1 | R1 航线独热编码；R1 样本取 1。 |

### 按飞行阶段的误差

| 阶段 | 中文名称 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |
|---|---|---:|---:|---:|---:|
| idle | 停机/低速 | 10,616 | 11.83 | 36.41 | 20.88 |
| ascent | 上升 | 19,819 | 35.61 | 47.72 | 6.63 |
| descent | 下降 | 12,284 | 35.09 | 49.56 | 7.34 |
| cruise | 巡航 | 2,518 | 31.00 | 41.99 | 6.22 |

统一诊断阶段由速度阈值推导：$v_z>0.15$ m/s 为 ascent，$v_z<-0.15$ m/s 为 descent，$|v_z|<0.15$ m/s 且 $v_{xy}\le1.0$ m/s 为 idle，$|v_z|<0.15$ m/s 且 $v_{xy}>1.0$ m/s 为 cruise。边界值也保留为 cruise。该规则只用于让七种算法按同一口径生成阶段图；RF-TLATT 轻量基线预测时使用验证集另选的 0.25/1.5 m/s 阈值。idle 的真实功率中位数接近 0，百分比误差分母较小，阅读时应结合 MAE（W）和样本数；ascent/descent 的功率变化更大，RMSE 对少量尖峰更敏感。

### 按实测功率区间的误差

| 实测功率区间 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |
|---|---:|---:|---:|---:|
| 0–50 W | 9,300 | 4.76 | 19.50 | 264.91 |
| 50–300 W | 859 | 84.79 | 108.51 | 51.82 |
| 300–450 W | 6,595 | 50.48 | 65.34 | 12.27 |
| 450–600 W | 22,870 | 28.07 | 38.21 | 5.42 |
| >600 W | 5,613 | 44.26 | 57.60 | 6.76 |

测试集中样本最多的是 `450–600 W`（22,870 条），它对总体 MAE 的贡献最大；当前算法的绝对误差最高区间是 `50–300 W`（MAE=84.79 W），通常对应起飞、降落或功率过渡段。

## 5. 图表与数据分析

该算法在测试集上的功率 MAE 为 **29.630 W**、RMSE 为 **45.541 W**、$R^2=0.9586$、WAPE 为 **7.293%**；flight 能耗 MAE 为 **0.556 Wh**、RMSE 为 **0.766 Wh**、WAPE 为 **2.542%**。残差均值为 7.126 W，样本标准差为 44.980 W，绝对误差 P50/P90/P95 为 19.47/71.49/94.78 W。下面逐图说明横纵轴、每条线或颜色、计算方式以及当前图中的数值。

### `power_scatter.png`

![power_scatter.png](./out/figures/power_scatter.png)

横轴是实测瞬时功率（Measured instantaneous power，W），纵轴是预测瞬时功率（Predicted instantaneous power，W）。绘图从 45,237 条测试记录中每隔 20 条取 1 条，共显示 2,262 个点；黑色虚线 $y=x$ 是理想线，线上方代表高估，点到理想线的垂直距离是该点绝对误差。抽样只用于减轻点云遮挡，指标仍由全部记录计算。
图中显示等间隔抽样点，$R^2=0.9586$ 与 WAPE=7.293% 则由全部 45,237 条记录计算。WAPE 表示全部绝对误差之和占实测功率绝对值之和的比例。

### `flight_power_timeseries.png`

![flight_power_timeseries.png](./out/figures/flight_power_timeseries.png)

横轴是飞行时间（Flight time，s），纵轴是功率（Power，W）。同一 flight 的实线是实测功率，虚线是预测功率；蓝、橙、绿分别对应图例中的前三个测试 flight。曲线峰值错位表示响应滞后，预测线持续高于实线表示该时段高估。
图中前三个测试 flight 为 18、23、83；flight 18 的区间 MAE=28.40 W，flight 23 的区间 MAE=32.08 W，flight 83 的区间 MAE=43.14 W。同色实线/虚线分别是实测/预测，峰值错位会增加尾部误差。

### `residual_histogram.png`

![residual_histogram.png](./out/figures/residual_histogram.png)

横轴是功率残差 $e=\hat P-P$（Residual，W），纵轴是记录数（Count）。蓝色柱表示残差频数，红色竖虚线是零误差。中心偏右/偏左分别表示总体高估/低估，分布宽度和长尾对应 RMSE 与 P95。
残差均值 7.126 W 表明整体存在轻微高估倾向；标准差 44.98 W 与 RMSE 的差异共同反映了尖峰样本。

### `residual_vs_actual.png`

![residual_vs_actual.png](./out/figures/residual_vs_actual.png)

横轴是实测功率（Measured power，W），纵轴是残差（Residual，W）；颜色分别表示 idle、ascent、descent、cruise。红色水平虚线为零残差。该图每隔 4 条记录取 1 条，显示 11,310 个点；抽样只服务于显示。某个功率区间持续偏离零线说明该区间存在系统偏差，扇形扩散说明误差随功率增大而增大。
该图每隔 4 条记录绘制 1 点，指标不抽样。若点云在高功率端展开更宽，说明残差离散度随功率改变；颜色只表示统一诊断阶段，不是模型一定使用的阶段输入。

### `power_bin_error.png`

![power_bin_error.png](./out/figures/power_bin_error.png)

横轴是实测功率区间；左纵轴是区间 MAE（W），橙色柱表示误差；右纵轴是样本数（Samples），蓝色折线和圆点表示样本量。两条纵轴不能混读。柱高用于比较区间难度，折线用于判断该区间对总体指标的权重。
橙柱最高的区间是 50–300 W（MAE=84.79 W）；蓝线显示该区间只有 859 条样本，不能仅凭柱高判断其对总体指标的贡献。

### `flight_energy_scatter.png`

![flight_energy_scatter.png](./out/figures/flight_energy_scatter.png)

横轴是实测单 flight 总能耗（Measured flight energy，Wh），纵轴是预测总能耗（Predicted flight energy，Wh）。蓝点代表一个 flight，黑色虚线是 $y=x$；线上方为任务级高估，点距直线越远表示累计能量偏差越大。
28 个蓝点对应 28 个 flight；任务级 WAPE=2.542%，点到 $y=x$ 的垂直距离就是 `energy_error_wh`。

### `flight_energy_error_ranking.png`

![flight_energy_error_ranking.png](./out/figures/flight_energy_error_ranking.png)

横轴是 `predicted_energy_wh−actual_energy_wh`（Flight-energy error，Wh），纵轴是 flight 编号。红柱表示高估，蓝柱表示低估，黑色竖线为零误差。柱长直接给出每个任务的累计偏差。
任务误差范围为 -0.494～2.438 Wh；红柱是高估，蓝柱是低估。

### `cumulative_energy_timeseries.png`

![cumulative_energy_timeseries.png](./out/figures/cumulative_energy_timeseries.png)

横轴是飞行时间（s），纵轴是累计能耗（Cumulative energy，Wh）。同色实线是实测能量累加，虚线是预测能量累加；末端垂直距离是该 flight 总能耗误差，曲线斜率对应当前功率水平。
每条实线/虚线对使用同一 `dt_seconds` 积分；曲线末端差异与 flight 能耗误差表一致，斜率变化对应当前功率变化。

### `phase_error_metrics.png`

![phase_error_metrics.png](./out/figures/phase_error_metrics.png)

横轴依次为 idle（停机/低速）、ascent（上升）、descent（下降）、cruise（巡航）；三个子图纵轴分别是 MAE（W）、RMSE（W）和 WAPE（%）。每根柱表示一个阶段的聚合误差，RMSE 明显高于 MAE 时说明阶段内存在尖峰。
按 WAPE 看，当前算法误差最低的阶段是 cruise（巡航，6.22%）；若某阶段 RMSE 明显高于 MAE，说明少量突变点主导了平方误差。

### `absolute_error_cdf.png`

![absolute_error_cdf.png](./out/figures/absolute_error_cdf.png)

横轴是绝对功率误差（Absolute power error，W），纵轴是经验 CDF（0–1）。蓝线在阈值 $a$ 处的高度表示误差不超过 $a$ W 的比例；红色水平线是 90% 参考线，曲线越靠左上越好。
CDF 在 19.47 W、71.49 W 和 94.78 W 附近分别达到 0.50、0.90 和 0.95，表示对应比例的测试点误差不超过这些阈值。

### `prediction_zoom.png`

![prediction_zoom.png](./out/figures/prediction_zoom.png)

上子图横轴为局部飞行时间（s），纵轴为功率（W）；深蓝实线是实测功率，橙色虚线是预测功率。下子图横轴相同，纵轴为残差（W），紫线为 $\hat P-P$，黑色虚线为零误差。窗口以该算法最大绝对误差为中心，专门观察突变响应。
最大绝对误差位于 flight 135、时间 15.00 s；实测/预测功率为 582.88/21.84 W，残差为 -561.04 W。上图显示该点前后最多各 100 条记录，下图给出同一窗口的残差符号和持续时间。

### `flight_energy_error_histogram.png`

![flight_energy_error_histogram.png](./out/figures/flight_energy_error_histogram.png)

横轴是 flight 能耗误差（Wh），纵轴是 flight 数量。蓝色柱表示误差分布，黑色虚线是零误差，红线是平均误差；红线偏离零说明存在任务级系统偏差。
红色均值线对应平均任务误差 0.384 Wh；若它偏离黑色零线，说明误差不是单纯随机波动。

### `phase_error_distribution.png`

![phase_error_distribution.png](./out/figures/phase_error_distribution.png)

横轴是四个飞行阶段，纵轴是绝对功率误差（W）。箱体为第 25–75 百分位，红线为中位数，须线表示非离群范围；箱体高说明该阶段误差离散程度大。
四阶段绝对误差中位数为 idle 0.93 W、ascent 27.64 W、descent 24.95 W、cruise 23.06 W。箱体高表示中间 50% 样本跨度大；图中隐藏离群点，完整尾部应结合 CDF 和 P95。


## 6. 输出文件

- `out/predictions.csv`：每条测试记录的 flight、时间、真实功率、预测功率和采样能量。
- `out/flight_energy_summary.csv`：每个 flight 的实测能耗、预测能耗、记录数和 `energy_error_wh`。
- `out/metrics.csv`、`out/metrics.json`：功率级和 flight 能耗级 MAE、RMSE、R²、MAPE、WAPE。
- `out/model/tuning_results.csv`：验证集候选参数、验证指标、选择分数和 `selected` 标记（固定 checkpoint 方法没有该文件）。
- `out/model/training_log.csv`：深度模型各候选每轮训练损失、验证损失和设备信息。
- `out/figures/`：本 README 中嵌入的 13 张诊断图。

## 7. 应用场景与限制

原文面向电动汽车能耗，本实验将其短期 LSTM 与窗口内自注意力结构迁移到 UAV。它适合 17 步窗口中既有局部变化又有跨位置关联的工况，但这里的窗口只有约 2.0 s，结果只能说明短窗依赖建模效果，不能外推为整段航程的长期预测能力。 <sup>[A5]</sup>

统一诊断阶段来自速度阈值而非人工标签；所有指标只对应当前 R1 固定划分和一个随机种子。不同论文的原始对象、输入字段、采样条件和预测目标并不相同，其原文指标不能与本目录数值直接横向相减。本实验未做重复训练、置信区间或显著性检验。

## 8. 参考文献

- **[A5]** Feng, Z., Zhang, J., Jiang, H., Yao, X., Qian, Y., Zhang, H. *Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework*. Energy, 2024, 302: 131780. DOI: 10.1016/j.energy.2024.131780.

背景和应用文献见上级 [README](../README.md) 的 B 类列表。PDF 原件目录：`C:\Users\18030\Desktop\GPT-files\Energy-prediction-Comparison-algorithm\pdf`。
