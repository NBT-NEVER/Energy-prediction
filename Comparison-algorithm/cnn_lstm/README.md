# CNN-LSTM（Convolutional Neural Network–Long Short-Term Memory，卷积神经网络-长短期记忆网络）

**算法来源：** Luo 等，2025，RF-TLATT 论文表 8 列出的 CNN-LSTM 对比名称（二手来源）。正文引用 <sup>[A1]</sup>。

## 1. 方法概述

Luo 等仅在表 8 给出 CNN-LSTM 的对比结果，正文将其归于“Xuebing et al. (2024)”，但所给论文的参考文献表缺少相应完整条目，因而无法核实原模型层数与参数。当前目录实现的是可复现的 CNN-LSTM 工程基线，不冒充该未知原模型的直接复现。验证集在 48/64 通道候选中选中卷积通道 $C=64$、LSTM隐层 $H=64$、Dropout 0.15，共 29,377 个可训练参数。

1. **输入与填充层**：当前点与同一 flight 的前19点组成 $B\times20\times10$ 标准化序列；flight 开头不足部分复制最早记录。转置后得到 Conv1d 所需的 $B\times10\times20$。
2. **卷积层 1**：`Conv1d(10,64,k=3,padding=1)` 同时汇总 10 个输入通道和相邻 3 个时间位置，输出 $B\times64\times20$；随后执行 `BatchNorm1d(64) → LeakyReLU(0.05) → Dropout(0.15)`。
3. **卷积层 2**：`Conv1d(64,64,k=3,padding=1) → BatchNorm1d(64) → LeakyReLU(0.05)`，继续组合第一层局部模式，输出形状仍为 $B\times64\times20$。第二层后没有额外 Dropout。
4. **单层 LSTM**：转回 $B\times20\times64$，由 64 个记忆单元顺序聚合卷积特征；输出整个序列并取最后时间位置的 $B\times64$ 向量。
5. **回归头**：对最后时刻向量执行 `Dropout(0.15) → Linear(64,1)`，得到标准化功率，再还原为 W 并截断为非负值。
6. **时间信息边界**：两层卷积使用对称 `padding=1`，中间位置会组合其左右相邻位置；但送入模型的 20 步缓冲区以当前预测时刻结束，所以不会访问该时刻之后的数据。训练使用 AdamW、SmoothL1 和梯度裁剪，本次最佳状态保存于第 20 轮。

$$z_{c,t}=\\operatorname{LeakyReLU}\left(\\operatorname{BN}\left(\sum_{r=1}^{C_{in}}\sum_{\tau=-1}^{1}W_{c,r,\tau}x_{r,t+\tau}+b_c\right)\right)$$
其中：

- $x_{r,t+\tau}$：输入通道 $r$ 在窗口位置 $t+\tau$ 的值；边界外位置由零填充。
- $C_{in}$：该卷积层的输入通道数，第一层为 10、第二层为 64。
- $W_{c,r,\tau}$：从输入通道 $r$、相对位置 $\tau$ 到输出通道 $c$ 的卷积权重。
- $b_c$：输出通道 $c$ 的偏置；$z_{c,t}$：卷积、批归一化和激活后的局部特征。
- $c$：输出通道索引；$r$：输入通道索引；$t$：窗口内位置；$\tau\in\{-1,0,1\}$：三点卷积核的相对位置。
- $\\operatorname{BN}$：批归一化；$\\operatorname{LeakyReLU}$：负半轴斜率为 0.05 的激活函数。

## 2. 统一实验条件

本算法与其他方法使用同一 4.1 十维任务前数据和同一完整 flight 划分：训练 113,766 条、124 个 flight；验证 26,300 条、29 个 flight；测试 48,724 条、47 个 flight。实际模型输入为：10 个4.1任务前特征，20 步（4 s）历史至当前窗口。测试功率目标统一为 `power_w`，能耗统一按每条记录的 `dt_seconds` 积分；测试集只用于最终报告，不参与候选选择。

本次最终测试结果：功率 MAE=41.137 W，RMSE=64.871 W，R²=0.9249，WAPE=10.525%；flight 能耗 MAE=0.798 Wh，RMSE=1.234 Wh，R²=0.9463，WAPE=3.544%。

## 3. 验证集调参记录

| 候选 | 验证最优轮次 | 验证功率WAPE(%) | 验证能耗WAPE(%) | 选择分数 | 最终选择 | cnn_channels | lstm_hidden | dropout | learning_rate | weight_decay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 9.4838 | 1.9940 | 10.4808 | 否 | 48 | 48 | 0.1000 | 0.0020 | 0.0001 |
| 2 | 24 | 9.3068 | 1.8115 | 10.2125 | 是 | 64 | 64 | 0.1500 | 0.0010 | 0.0001 |

最终选择候选 `2`：`cnn_channels=64.0000`、`lstm_hidden=64.0000`、`dropout=0.1500`、`learning_rate=0.0010`、`weight_decay=0.0001`。验证集选择分数为 `10.2125`；测试集没有参与候选筛选。

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

MAE、RMSE、MAPE、WAPE 越小越好，$R^2$ 越接近 1 越好。能耗先按真实 `dt_seconds` 积分，再在 47 个 flight 上计算指标，避免把不等间隔采样误当成等权样本。

本实验的选择分数为：

$$S_{val}=\mathrm{WAPE}_{power,val}+0.5\,\mathrm{WAPE}_{energy,val}$$

其中：

- $S_{val}$：验证集候选选择分数，越小越好。
- $\mathrm{WAPE}_{power,val}$：验证集逐点功率 WAPE，单位 %。
- $\mathrm{WAPE}_{energy,val}$：验证集按 flight 汇总后的能耗 WAPE，单位 %。
- $0.5$：能耗 WAPE 在选择分数中的固定权重。

该分数只在 29 个 validation flight 上计算，用于比较候选结构、正则强度和相位阈值；测试集 47 个 flight 在参数固定后才运行。

### 变量、坐标轴和字段词典

| 字段或指标 | 中文全称 | English full name | 单位/范围 | 计算方式和含义 |
|---|---|---|---|---|
| `flight` | 飞行任务编号 | Flight identifier | 无量纲 | 同一编号的连续记录属于同一任务，能耗按该字段分组。 |
| `route` | 航线编号 | Route identifier | — | route 仅作追踪与分层统计，不作为模型输入；4.1数据包含 H、R1–R7（排除 A1–A3）。 |
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


下表列出公共数据中的10个4.1任务前字段。当前算法是否直接使用某字段，以第 2 节的实际输入和第 1 节逐层结构为准。

| 变量 | 中文全称 | English full name | 单位/范围 | 在模型中的作用 |
|---|---|---|---|---|
| `time_s` | 相对任务时间 | Relative task time | s | 任务开始后的相对时间，来自每个flight首时刻归零的规划时间轴。 |
| `task_duration_s` | 任务计划时长 | Planned task duration | s | 完整任务计划时长，用于表达当前任务所处的时间尺度。 |
| `planned_position_east_m` | 规划东向位置 | Planned east position | m | ENU东向规划位置，反映水平航迹位移。 |
| `planned_position_north_m` | 规划北向位置 | Planned north position | m | ENU北向规划位置，反映水平航迹位移。 |
| `planned_position_up_m` | 规划上向位置 | Planned up position | m | ENU上向规划位置，反映爬升和下降状态。 |
| `wind_east_mps` | 东向风速分量 | East wind component | m/s | 风矢量东向分量，参与风致功率变化建模。 |
| `wind_north_mps` | 北向风速分量 | North wind component | m/s | 风矢量北向分量，参与风致功率变化建模。 |
| `payload_g` | 有效载荷质量 | Payload mass | g | 有效载荷质量，影响悬停和机动所需功率。 |
| `planned_motor_on` | 计划电机运行状态 | Planned motor-on state | 0/1 | 计划电机状态，区分停机和动力运行阶段。 |
| `planned_airborne` | 计划离地状态 | Planned airborne state | 0/1 | 计划离地状态，辅助区分地面、起降和空中航段。 |

### 按飞行阶段的误差

| 阶段 | 中文名称 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |
|---|---|---:|---:|---:|---:|
| idle | 停机/低速 | 14,086 | 17.58 | 48.13 | 25.08 |
| ascent | 上升 | 10,185 | 59.51 | 81.12 | 10.54 |
| descent | 下降 | 13,988 | 49.76 | 71.36 | 9.95 |
| cruise | 巡航 | 10,465 | 43.44 | 57.13 | 8.57 |

统一诊断阶段由速度阈值推导：$v_z>0.15$ m/s 为 ascent，$v_z<-0.15$ m/s 为 descent，$|v_z|<0.15$ m/s 且 $v_{xy}\le1.0$ m/s 为 idle，$|v_z|<0.15$ m/s 且 $v_{xy}>1.0$ m/s 为 cruise。边界值也保留为 cruise。该规则只用于让七种算法按同一口径生成阶段图；RF-TLATT 轻量基线预测时使用验证集另选的 0.25/1.5 m/s 阈值。idle 的真实功率中位数接近 0，百分比误差分母较小，阅读时应结合 MAE（W）和样本数；ascent/descent 的功率变化更大，RMSE 对少量尖峰更敏感。

### 按实测功率区间的误差

| 实测功率区间 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |
|---|---:|---:|---:|---:|
| 0–50 W | 11,783 | 6.20 | 18.35 | 401.71 |
| 50–300 W | 856 | 156.56 | 199.58 | 93.80 |
| 300–450 W | 6,779 | 64.04 | 81.14 | 15.53 |
| 450–600 W | 23,081 | 37.63 | 50.47 | 7.25 |
| >600 W | 6,225 | 79.46 | 101.04 | 12.05 |

测试集中样本最多的是 `450–600 W`（23,081 条），它对总体 MAE 的贡献最大；当前算法的绝对误差最高区间是 `50–300 W`（MAE=156.56 W），通常对应起飞、降落或功率过渡段。

## 5. 图表与数据分析

该算法在测试集上的功率 MAE 为 **41.137 W**、RMSE 为 **64.871 W**、$R^2=0.9249$、WAPE 为 **10.525%**；flight 能耗 MAE 为 **0.798 Wh**、RMSE 为 **1.234 Wh**、WAPE 为 **3.544%**。残差均值为 5.921 W，样本标准差为 64.601 W，绝对误差 P50/P90/P95 为 24.80/102.05/135.74 W。下面逐图说明横纵轴、每条线或颜色、计算方式以及当前图中的数值。

### `power_scatter.png`

![power_scatter.png](./out/figures/power_scatter.png)

横轴是实测瞬时功率（Measured instantaneous power，W），纵轴是预测瞬时功率（Predicted instantaneous power，W）。绘图从 48,724 条测试记录中每隔 20 条取 1 条，共显示 2,262 个点；黑色虚线 $y=x$ 是理想线，线上方代表高估，点到理想线的垂直距离是该点绝对误差。抽样只用于减轻点云遮挡，指标仍由全部记录计算。
图中显示等间隔抽样点，$R^2=0.9249$ 与 WAPE=10.525% 则由全部 48,724 条记录计算。WAPE 表示全部绝对误差之和占实测功率绝对值之和的比例。

### `flight_power_timeseries.png`

![flight_power_timeseries.png](./out/figures/flight_power_timeseries.png)

横轴是飞行时间（Flight time，s），纵轴是功率（Power，W）。同一 flight 的实线是实测功率，虚线是预测功率；蓝、橙、绿分别对应图例中的前三个测试 flight。曲线峰值错位表示响应滞后，预测线持续高于实线表示该时段高估。
图中前三个测试 flight 为 1、5、6；flight 1 的区间 MAE=32.45 W，flight 5 的区间 MAE=37.56 W，flight 6 的区间 MAE=32.88 W。同色实线/虚线分别是实测/预测，峰值错位会增加尾部误差。

### `residual_histogram.png`

![residual_histogram.png](./out/figures/residual_histogram.png)

横轴是功率残差 $e=\hat P-P$（Residual，W），纵轴是记录数（Count）。蓝色柱表示残差频数，红色竖虚线是零误差。中心偏右/偏左分别表示总体高估/低估，分布宽度和长尾对应 RMSE 与 P95。
残差均值 5.921 W 表明整体存在轻微高估倾向；标准差 64.60 W 与 RMSE 的差异共同反映了尖峰样本。

### `residual_vs_actual.png`

![residual_vs_actual.png](./out/figures/residual_vs_actual.png)

横轴是实测功率（Measured power，W），纵轴是残差（Residual，W）；颜色分别表示 idle、ascent、descent、cruise。红色水平虚线为零残差。该图每隔 4 条记录取 1 条，显示 11,310 个点；抽样只服务于显示。某个功率区间持续偏离零线说明该区间存在系统偏差，扇形扩散说明误差随功率增大而增大。
该图每隔 4 条记录绘制 1 点，指标不抽样。若点云在高功率端展开更宽，说明残差离散度随功率改变；颜色只表示统一诊断阶段，不是模型一定使用的阶段输入。

### `power_bin_error.png`

![power_bin_error.png](./out/figures/power_bin_error.png)

横轴是实测功率区间；左纵轴是区间 MAE（W），橙色柱表示误差；右纵轴是样本数（Samples），蓝色折线和圆点表示样本量。两条纵轴不能混读。柱高用于比较区间难度，折线用于判断该区间对总体指标的权重。
橙柱最高的区间是 50–300 W（MAE=156.56 W）；蓝线显示该区间只有 856 条样本，不能仅凭柱高判断其对总体指标的贡献。

### `flight_energy_scatter.png`

![flight_energy_scatter.png](./out/figures/flight_energy_scatter.png)

横轴是实测单 flight 总能耗（Measured flight energy，Wh），纵轴是预测总能耗（Predicted flight energy，Wh）。蓝点代表一个 flight，黑色虚线是 $y=x$；线上方为任务级高估，点距直线越远表示累计能量偏差越大。
47 个蓝点对应 47 个 flight；任务级 WAPE=3.544%，点到 $y=x$ 的垂直距离就是 `energy_error_wh`。

### `flight_energy_error_ranking.png`

![flight_energy_error_ranking.png](./out/figures/flight_energy_error_ranking.png)

横轴是 `predicted_energy_wh−actual_energy_wh`（Flight-energy error，Wh），纵轴是 flight 编号。红柱表示高估，蓝柱表示低估，黑色竖线为零误差。柱长直接给出每个任务的累计偏差。
任务误差范围为 -2.457～5.622 Wh；红柱是高估，蓝柱是低估。

### `cumulative_energy_timeseries.png`

![cumulative_energy_timeseries.png](./out/figures/cumulative_energy_timeseries.png)

横轴是飞行时间（s），纵轴是累计能耗（Cumulative energy，Wh）。同色实线是实测能量累加，虚线是预测能量累加；末端垂直距离是该 flight 总能耗误差，曲线斜率对应当前功率水平。
每条实线/虚线对使用同一 `dt_seconds` 积分；曲线末端差异与 flight 能耗误差表一致，斜率变化对应当前功率变化。

### `phase_error_metrics.png`

![phase_error_metrics.png](./out/figures/phase_error_metrics.png)

横轴依次为 idle（停机/低速）、ascent（上升）、descent（下降）、cruise（巡航）；三个子图纵轴分别是 MAE（W）、RMSE（W）和 WAPE（%）。每根柱表示一个阶段的聚合误差，RMSE 明显高于 MAE 时说明阶段内存在尖峰。
按 WAPE 看，当前算法误差最低的阶段是 cruise（巡航，8.57%）；若某阶段 RMSE 明显高于 MAE，说明少量突变点主导了平方误差。

### `absolute_error_cdf.png`

![absolute_error_cdf.png](./out/figures/absolute_error_cdf.png)

横轴是绝对功率误差（Absolute power error，W），纵轴是经验 CDF（0–1）。蓝线在阈值 $a$ 处的高度表示误差不超过 $a$ W 的比例；红色水平线是 90% 参考线，曲线越靠左上越好。
CDF 在 24.80 W、102.05 W 和 135.74 W 附近分别达到 0.50、0.90 和 0.95，表示对应比例的测试点误差不超过这些阈值。

### `prediction_zoom.png`

![prediction_zoom.png](./out/figures/prediction_zoom.png)

上子图横轴为局部飞行时间（s），纵轴为功率（W）；深蓝实线是实测功率，橙色虚线是预测功率。下子图横轴相同，纵轴为残差（W），紫线为 $\hat P-P$，黑色虚线为零误差。窗口以该算法最大绝对误差为中心，专门观察突变响应。
最大绝对误差位于 flight 84、时间 15.40 s；实测/预测功率为 674.55/49.74 W，残差为 -624.81 W。上图显示该点前后最多各 100 条记录，下图给出同一窗口的残差符号和持续时间。

### `flight_energy_error_histogram.png`

![flight_energy_error_histogram.png](./out/figures/flight_energy_error_histogram.png)

横轴是 flight 能耗误差（Wh），纵轴是 flight 数量。蓝色柱表示误差分布，黑色虚线是零误差，红线是平均误差；红线偏离零说明存在任务级系统偏差。
红色均值线对应平均任务误差 0.341 Wh；若它偏离黑色零线，说明误差不是单纯随机波动。

### `phase_error_distribution.png`

![phase_error_distribution.png](./out/figures/phase_error_distribution.png)

横轴是四个飞行阶段，纵轴是绝对功率误差（W）。箱体为第 25–75 百分位，红线为中位数，须线表示非离群范围；箱体高说明该阶段误差离散程度大。
四阶段绝对误差中位数为 idle 4.21 W、ascent 44.17 W、descent 34.99 W、cruise 34.32 W。箱体高表示中间 50% 样本跨度大；图中隐藏离群点，完整尾部应结合 CDF 和 P95。


## 6. 输出文件

- `out/predictions.csv`：每条测试记录的 flight、时间、真实功率、预测功率和采样能量。
- `out/flight_energy_summary.csv`：每个 flight 的实测能耗、预测能耗、记录数和 `energy_error_wh`。
- `out/metrics.csv`、`out/metrics.json`：功率级和 flight 能耗级 MAE、RMSE、R²、MAPE、WAPE。
- `out/model/tuning_results.csv`：验证集候选参数、验证指标、选择分数和 `selected` 标记（固定 checkpoint 方法没有该文件）。
- `out/model/training_log.csv`：深度模型各候选每轮训练损失、验证损失和设备信息。
- `out/figures/`：本 README 中嵌入的 13 张诊断图。

## 7. 应用场景与限制

适合先从相邻采样点提取局部波形，再由 LSTM 汇总短时演化的功率估计。当前来源仅为 Luo 等论文表 8 的二手算法名称，所给材料无法核实原始 CNN-LSTM 论文和完整层配置，因此本实现是明确标注的工程适配基线。 <sup>[A1]</sup>

统一诊断阶段来自速度阈值而非人工标签；所有指标只对应当前4.1固定划分和一个随机种子。不同论文的原始对象、输入字段、采样条件和预测目标并不相同，其原文指标不能与本目录数值直接横向相减。本实验未做重复训练、置信区间或显著性检验。

## 8. 参考文献

- **[A1]** Luo, W., Li, N., Xiong, Z., Chen, W., Li, Y., Tang, C., Li, Y., Dong, C. *Phase-based power prediction for quadrotor UAVs with RF-TLATT*. Energy, 2025, 335: 138208. DOI: 10.1016/j.energy.2025.138208.

> 来源层级说明：Luo 等只在表 8 报告 CNN-LSTM，并在正文写作未编号的 “Xuebing et al. (2024)”；其参考文献表缺少相应题录。所给材料无法核实该模型的一手作者、题名和层配置，因此 [A1] 在本目录属于二手来源。

背景和应用文献见上级 [README](../README.md) 的 B 类列表。PDF 原件目录：`C:\Users\18030\Desktop\GPT-files\Energy-prediction-Comparison-algorithm\pdf`。
