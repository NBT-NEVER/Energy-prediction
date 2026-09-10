# Comparison-algorithm：四轴无人机功率与能耗对比实验

## 摘要

针对 R1 航线上的四轴无人机瞬时功率与单次任务能耗预测，本文在固定 train/validation/test flight 划分上比较 3.0 TCN+RLS 主方法和六种论文来源基线。两项原候选因功率 WAPE 达到 26.516% 和 18.088% 而被替换，新加入 LSTM-Transformer 与 CNN-LSTM。最终主方法在 45,237 条测试记录上取得 27.991 W 的功率 MAE 和 6.889% 的功率 WAPE，在 28 个测试 flight 上取得 0.292% 的能耗 WAPE；最强无测试期目标反馈对照 LSTM-Transformer 的功率 WAPE 为 7.293%。由于主方法在测试期使用已结束秒窗的真实能量更新下一窗 RLS 参数，任务级能耗差异同时包含在线反馈收益。所有结论限于当前固定划分和单次随机种子。

**关键词：** 四轴无人机；功率预测；能耗预测；时间卷积网络；递推最小二乘；对比实验

## 1. 研究背景与实验目的

无人机逐点功率可由飞行内电压、电流和工况记录建立监督数据<sup>[B1]</sup>；预测结果可为返航阈值、剩余航时和任务调度提供输入。任务总能耗还能进入配送路径成本<sup>[B2]</sup>、编队任务代价<sup>[B3]</sup>和多旋翼任务可行性判断<sup>[B4]</sup>。本实验的目的，是在相同 flight 划分、相同功率目标、相同真实时间积分和相同评价指标下，比较带秒级能量反馈的 TCN+RLS 与六种时序、相位和物理基线。各方法使用同一份源数据，但依据论文结构选择不同输入表示，因此“统一条件”不等于所有模型都读取相同特征子集或相同时间窗。

## 2. 数据、场景与实验边界

数据来自 3.0 的 R1 处理集：训练集 198,469 条、126 个 flight；验证集 41,052 条、28 个 flight；测试集 45,237 条、28 个 flight。典型采样间隔约为 0.12 s，功率目标为 `power_w`（W），能耗按每条记录的 `dt_seconds` 积分为 Wh。数据按完整 flight 划分，避免同一飞行同时出现在训练与测试中。

当前场景是单架四轴无人机在 R1 航线上的逐点功率与任务能耗估计。Luo 等把相位识别与专用时序模型用于四旋翼功率预测<sup>[A1]</sup>；Muli 等比较无人机数据驱动能耗模型<sup>[A3]</sup>；Dudukcu 等研究带 SMA 的 LR-TCN 瞬时功率模型<sup>[A4]</sup>；Jastrzębska 等从物理运动量建立 UAV 能耗模型<sup>[A2]</sup>；Feng 等在电动汽车上验证 LSTM-Transformer 的长短期组合<sup>[A5]</sup>；Cabuk 等把小型无人机能耗用于编队连接恢复、拓扑维护和队形变换的任务代价<sup>[B3]</sup>。本实验只迁移这些建模思想，论文原数据集上的数值不作为本项目结果。

主方法的部署边界需要单列说明：3.0 在每个完整约 1 s 窗口结束后读取该窗口真实能量，RLS 更新后的偏置和缩放只用于下一窗口；六种对比方法在测试时没有同样的目标反馈。因此主方法的 flight 能耗优势包含在线反馈带来的收益，不能把全部差异归因于 TCN 的前馈特征提取。

## 3. 方法与文献来源


| 算法目录                                           | 方法                     | 代码中实际结构                                        | 算法来源                                      |
| -------------------------------------------------- | ------------------------ | ----------------------------------------------------- | --------------------------------------------- |
| [`proposed_tcn_rls`](./proposed_tcn_rls/README.md) | 3.0 TCN + 秒级能量 RLS   | TCN 四残差块 + 时间门控 + 秒级能量反馈 RLS            | <sup>[A0]</sup><sup>[A6]</sup><sup>[A7]</sup> |
| [`rf_tlatt_lite`](./rf_tlatt_lite/README.md)       | RF-TLATT 思路相位岭回归  | 速度相位分类 + 四个相位专用岭回归头                   | <sup>[A1]</sup>                               |
| [`physical_mlr`](./physical_mlr/README.md)         | 物理特征 MLR（全局适配） | 物理量展开 + 标准化多元线性/岭回归                    | <sup>[A2]</sup>                               |
| [`lstm`](./lstm/README.md)                         | LSTM 时序功率预测        | 两层双向 LSTM + Dropout + Tanh 回归头                 | <sup>[A3]</sup>                               |
| [`lr_tcn_sma`](./lr_tcn_sma/README.md)             | LR-TCN-SMA               | 因果 LeakyReLU TCN 残差块 + 历史 SMA                  | <sup>[A4]</sup>                               |
| [`lstm_transformer`](./lstm_transformer/README.md) | LSTM-Transformer         | 线性投影 + 位置编码 + LSTM + 多头 Transformer Encoder | <sup>[A5]</sup>                               |
| [`cnn_lstm`](./cnn_lstm/README.md)                 | CNN-LSTM                 | 两层 Conv1d/BatchNorm + LSTM + 线性头                 | <sup>[A1]</sup>                               |

算法来源文献只在本节及各算法 README 的‘算法来源’中列出；研究背景和应用场景文献放在文末的 B 类列表，避免把方法来源与背景引用混在一起。正文中的 `<sup>[A*]</sup>` 和 `<sup>[B*]</sup>` 是规范上标引用。

## 4. 统一运行方式

```powershell
cd I:\STUDY\python\project\Energy-prediction\Comparison-algorithm
python main.py
python main.py --algorithm lstm_transformer cnn_lstm
python generate_documentation.py
```

`main.py` 从 `config.py` 读取路径和算法注册表；`comparison_engine.py` 读取 train/validation/test，先在 validation 上选择候选，再生成测试预测、指标、诊断图和总体图；`generate_documentation.py` 只读取最终 CSV/JSON 并重建本文档。每个算法目录均包含 `algorithm.py`、独立 `README.md`、`out/predictions.csv`、`out/flight_energy_summary.csv`、`out/metrics.csv/json`、`out/model/tuning_results.csv`（适用时）和 `out/figures/`。

## 5. 统一输入变量

公共处理表提供 23 个候选字段：`time`, `dt_seconds`, `flight_progress`, `wind_speed`, `wind_sin`, `wind_cos`, `programmed_speed_mps`, `actual_speed_mps`, `horizontal_speed_mps`, `vertical_speed_mps`, `vertical_speed_abs_mps`, `relative_air_speed_mps`, `wind_alignment`, `wind_cross_component_mps`, `payload_kg`, `altitude_m`, `dynamic_accel_norm`, `angular_rate_norm`, `obstacle_agility_index`, `thermal_load_proxy`, `vision_energy_proxy_w`, `communication_energy_proxy_w`, `route_R1`。TCN+RLS 和四个深度基线读取全部 23 个字段；RF-TLATT 轻量基线和 Physical MLR 从这些字段构造较小的当前时刻特征集。各模型实际输入如下。


| 算法                     | 实际输入与时间范围                                                                            |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| 3.0 TCN + 秒级能量 RLS   | 23 个公共特征，100 步（12 s）历史至当前窗口；末端 TCN 输出再由上一完整秒窗更新的 RLS 参数校正 |
| RF-TLATT 思路相位岭回归  | 从公共字段构造 16 个当前时刻物理与交互量；无序列窗口，按预测专用速度阈值选择相位回归头        |
| 物理特征 MLR（全局适配） | 从公共字段构造 7 个当前时刻物理量；无序列窗口，使用单一全局回归及 moving 指示变量             |
| LSTM 时序功率预测        | 23 个公共特征，17 步（约 2.0 s）历史至当前窗口                                                |
| LR-TCN-SMA               | 23 个公共特征先按 flight 做 10 点历史 SMA，再组成 17 步（约 2.0 s）窗口                       |
| LSTM-Transformer         | 23 个公共特征，17 步（约 2.0 s）历史至当前窗口                                                |
| CNN-LSTM                 | 23 个公共特征，17 步（约 2.0 s）历史至当前窗口                                                |

下表说明公共字段的中英文全称、单位、构造方式和物理含义。字段出现在公共表中不等于每个算法都直接使用它。


| 变量                           | 中文全称               | English full name                | 单位/范围 | 在模型中的作用                                                     |
| ------------------------------ | ---------------------- | -------------------------------- | --------- | ------------------------------------------------------------------ |
| `time`                         | 飞行内相对时间         | Flight-relative time             | s         | 飞行内相对时间戳，单位为秒。                                       |
| `dt_seconds`                   | 采样时间间隔           | Sampling time interval           | s         | 相邻采样点时间间隔，单位为秒；用于把功率积分为能耗。               |
| `flight_progress`              | 飞行进度               | Flight progress                  | 0–1      | 当前时间除以本次飞行最大时间，范围 0 到 1，表示飞行进度。          |
| `wind_speed`                   | 风速                   | Wind speed                       | m/s       | 风速，单位为米每秒。                                               |
| `wind_sin`                     | 风向正弦分量           | Sine of wind direction           | —        | 风向角的正弦分量，用于连续表达风向。                               |
| `wind_cos`                     | 风向余弦分量           | Cosine of wind direction         | —        | 风向角的余弦分量，用于连续表达风向。                               |
| `programmed_speed_mps`         | 规划速度               | Programmed speed                 | m/s       | 任务规划速度，单位为米每秒。                                       |
| `actual_speed_mps`             | 实际合速度             | Actual resultant speed           | m/s       | 三轴速度合成后的实际速度，单位为米每秒。                           |
| `horizontal_speed_mps`         | 水平速度               | Horizontal speed                 | m/s       | 水平面速度合成值，单位为米每秒。                                   |
| `vertical_speed_mps`           | 垂直速度               | Vertical speed                   | m/s       | 垂直速度分量，单位为米每秒；正负表示升降方向。                     |
| `vertical_speed_abs_mps`       | 垂直速度绝对值         | Absolute vertical speed          | m/s       | 垂直速度绝对值，单位为米每秒，表示垂直机动强度。                   |
| `relative_air_speed_mps`       | 相对空速               | Relative air speed               | m/s       | 相对空气速度，结合机体速度和风速计算，单位为米每秒。               |
| `wind_alignment`               | 风速方向对齐度         | Wind alignment cosine            | −1–1    | 水平速度与风速方向的余弦相似度，范围 -1 到 1。                     |
| `wind_cross_component_mps`     | 横向风速分量           | Crosswind component              | m/s       | 风速在水平速度横向上的分量，单位为米每秒。                         |
| `payload_kg`                   | 载荷质量               | Payload mass                     | kg        | 载荷质量，单位为千克。                                             |
| `altitude_m`                   | 飞行高度               | Flight altitude                  | m         | 飞行高度，单位为米。                                               |
| `dynamic_accel_norm`           | 动态加速度模           | Dynamic acceleration norm        | m/s²     | 去除重力影响后的三轴线性加速度合成强度，单位为米每二次方秒。       |
| `angular_rate_norm`            | 角速度模               | Angular-rate norm                | rad/s     | 三轴角速度合成强度，单位为弧度每秒。                               |
| `obstacle_agility_index`       | 障碍机动代理指标       | Obstacle-agility proxy index     | —        | 由加速度、角速度和垂直速度构成的机动性代理指标，无量纲。           |
| `thermal_load_proxy`           | 热负荷代理量           | Thermal-load proxy               | —        | 由相对气流、载荷和爬升状态估计的热负荷代理指标，用于表征附加能耗。 |
| `vision_energy_proxy_w`        | 视觉计算附加功率代理量 | Vision-energy power proxy        | W         | 由速度、机动性和高度估计的视觉计算附加功率代理值，单位为瓦特。     |
| `communication_energy_proxy_w` | 通信附加功率代理量     | Communication-energy power proxy | W         | 由高度、风速和速度估计的通信附加功率代理值，单位为瓦特。           |
| `route_R1`                     | R1航线独热变量         | R1 route one-hot variable        | 0/1       | R1 航线独热编码；R1 样本取 1。                                     |

## 6. 算法原理与逐层结构

### 3.0 TCN + 秒级能量 RLS（主方法） <sup>[A0]</sup><sup>[A6]</sup><sup>[A7]</sup>

**实现来源：** 本项目 3.0 实现；TCN 与 RLS 理论分别参考 Bai 等和 Sayed，实验图表组织参考 Luo 等人的 RF-TLATT 研究。

输入张量为 $B\times100\times23$：$B$ 是批量大小，100 是从历史到当前的时间步数，23 是每步特征数。该 checkpoint 的窗口覆盖 12 s，测试集典型采样间隔约 0.12 s。网络共有 81,049 个可训练参数。

1. **时间门控层**：取 23 维输入中已标准化的 `dt_seconds`，经 `Linear(1,8) → SiLU → Linear(8,1)` 得到形状 $B\times100\times1$ 的门值 $g_t=1+0.2\tanh(\cdot)$，再广播乘到同一时刻的 23 个特征。门值范围为 0.8–1.2，使不规则采样间隔参与输入缩放。
2. **TCN 残差块 1**：两层 $k=3$、膨胀率 $d=1$ 的左填充因果卷积，通道依次为 $23\to64\to64$。每层依次执行 GroupNorm（1 组）、SiLU 和 Dropout(0.08)；旁路用 $1\times1$ 卷积把 23 维映射到 64 维，主路与旁路相加后执行 ReLU。
3. **TCN 残差块 2**：通道为 $64\to64\to64$，两层卷积的膨胀率均为 2。输入输出通道相同，残差旁路为恒等映射。
4. **TCN 残差块 3**：通道保持 64，两层卷积的膨胀率均为 4，用更稀疏的历史位置补充中尺度变化。
5. **TCN 残差块 4**：通道为 $64\to32\to32$，膨胀率为 8，并用 $1\times1$ 旁路降到 32 维。四个残差块共有 8 层时间卷积，理论感受野为 $1+2(k-1)(1+2+4+8)=61$ 步，且每层都只在左侧补零，不读取预测时刻之后的数据。
6. **双支路回归头**：主路取 TCN 最后时刻的 32 维向量，执行 `Linear(32,16) → SiLU → Dropout(0.08) → Linear(16,1)`；旁路只取当前时刻原始 23 维输入，执行 `LayerNorm(23) → Linear(23,16) → SiLU → Linear(16,1)`。两路标量相加后反标准化为 TCN 功率。
7. **在线 RLS 校正层**：每个 flight 从 $[\theta_0,\theta_1]=[0,1]$ 和 $0.25I$ 协方差开始。程序先用旧参数预测当前完整秒窗内的所有点；当该窗累计时长不小于 0.95 s 时，才用其真实能量与 TCN 能量更新参数，更新结果从下一秒窗生效。遗忘因子为 0.90，偏置与缩放分别限制在 $[-1,1]$ 和 $[0,2]$。当 TCN 窗口均值跨过 50 W 的飞行状态阈值时恢复中性参数，避免停机与飞行状态共用一组校正量。

$$
\hat P_{f,i}^{\mathrm{RLS}}=\max\left(0,\;s\theta_{0,f,k}+\theta_{1,f,k}\hat P_{f,i}^{\mathrm{TCN}}\right)
$$

$$
K_k=\frac{P_k\phi_k}{\lambda+\phi_k^\mathsf{T}P_k\phi_k},\quad \theta_{k+1}=\theta_k+K_k(y_k-\phi_k^\mathsf{T}\theta_k),\quad P_{k+1}=\frac{P_k-K_k\phi_k^\mathsf{T}P_k}{\lambda}
$$

其中：

- $\hat P_{f,i}^{\mathrm{RLS}}$：flight $f$ 中第 $i$ 个采样点经 RLS 校正的功率，单位 W。
- $\hat P_{f,i}^{\mathrm{TCN}}$：同一采样点的 TCN 原始功率，单位 W。
- $k$：当前已经结束的完整秒窗编号；$i$：秒窗内采样点编号；$f$：flight 编号。
- $s$：训练集功率标准差形成的功率尺度，单位 W。
- $\theta_k=[\theta_{0,k},\theta_{1,k}]^\mathsf{T}$：第 $k$ 个秒窗预测时使用的偏置与缩放参数。
- $\phi_k=[1,\bar P_k^{\mathrm{TCN}}/s]^\mathsf{T}$：由第 $k$ 个秒窗 TCN 平均功率构成的回归向量。
- $y_k=\bar P_k^{\mathrm{true}}/s$：第 $k$ 个秒窗真实平均功率的尺度化目标。
- $P_k$：RLS 参数协方差矩阵；初始值为 $0.25I$。
- $\lambda$：遗忘因子，本次取 0.90；$K_k$：第 $k$ 次更新的 RLS 增益。
- $I$：二阶单位矩阵；$\max(0,\cdot)$：把物理上无意义的负功率截断为 0。

### RF-TLATT 思路的相位感知岭回归（轻量基线） <sup>[A1]</sup>

**实现来源：** Luo 等，2025，RF-TLATT 论文表 8 与相位分类流程。

Luo 等的完整 RF-TLATT 先用随机森林识别飞行阶段，再为每个阶段训练包含 LSTM、Transformer 和注意力的专用预测器。本目录只保留“先分相位、再使用相位专用模型”的控制变量思想：相位由速度阈值直接判定，预测器改为岭回归。因此本结果不能写成 RF-TLATT 的复现结果。

1. **预测相位判别层**：验证集比较垂直/水平速度阈值 $(0.10,0.5)$、$(0.15,1.0)$、$(0.25,1.5)$ m/s，最终选中 $(0.25,1.5)$ m/s。其规则为：$v_z>0.25$ m/s 判为 ascent，$v_z<-0.25$ m/s 判为 descent，$|v_z|<0.25$ m/s 且 $v_{xy}\le1.5$ m/s 判为 idle，$|v_z|<0.25$ m/s 且 $v_{xy}>1.5$ m/s 判为 cruise。README 的统一诊断图仍使用固定的 $(0.15,1.0)$ m/s 阈值划分阶段，两者用途不同。
2. **16 维特征层**：每条记录依次使用实际合速度、水平速度、垂直速度、垂直速度绝对值、载荷、高度、风速、动态加速度模、角速度模、相对空速、风向对齐度、横向风速分量、合速度平方、垂直速度平方、合速度×载荷、合速度×风速。该方法只读当前记录，不构造 17 步序列。
3. **标准化与截距层**：每个回归头都用训练子集的均值和标准差标准化 16 个输入，再在首列加入常数 1。每个相位头有 16 个特征系数和 1 个截距；岭惩罚不作用于截距。
4. **四个相位回归头**：idle、ascent、descent、cruise 分别求闭式岭回归。若某相位训练样本少于 $2\times16=32$ 条，则回退到用全部训练记录拟合的全局头；输出经 $\max(0,\cdot)$ 截断后得到瞬时功率。
5. **验证选择层**：每组阈值与 $\alpha\in\{0.01,0.1,1,10\}$ 组合均只在验证集打分。最终选中 $\alpha=10$、垂直阈值 0.25 m/s、水平阈值 1.5 m/s，选择分数为功率 WAPE 加 0.5 倍 flight 能耗 WAPE。

$$
\hat P_i=\max\left(0,\;[1,\mathbf z_i^\mathsf{T}]\hat\beta_{c_i}\right),\qquad \hat\beta_c=(X_c^\mathsf{T}X_c+\alpha I)^{-1}X_c^\mathsf{T}\mathbf y_c
$$

其中：

- $\hat P_i$：第 $i$ 条测试记录的非负预测功率，单位 W。
- $c_i$：第 $i$ 条记录由预测阈值判定的相位类别。
- $\mathbf z_i$：第 $i$ 条记录经训练统计量标准化后的 16 维特征向量。
- $[1,\mathbf z_i^\mathsf{T}]$：含截距常数 1 的行向量。
- $\hat\beta_c$：相位 $c$ 的 17 维回归系数估计。
- $X_c$：训练集中相位 $c$ 的含截距设计矩阵；$\mathbf y_c$：对应真实功率列向量。
- $\alpha$：岭正则强度，本次验证集选中 10；$I$：截距位置为 0、其余对角元素为 1 的惩罚矩阵。
- $i$：测试记录索引；$c$：相位类别索引。

### 物理特征 MLR（单一全局回归适配版） <sup>[A2]</sup>

**实现来源：** Jastrzębska 等，2026，UAV energy consumption prediction。

Jastrzębska 等的原方法分别建立静止与飞行子模型，并使用物理运动量解释能耗。本工程为适配现有 R1 字段，只建立一个全局回归，在输入中加入 `moving` 指示量；同时由于数据没有论文所需的独立垂直加速度字段，代码用带符号垂直速度项代替对应垂直运动项。下面描述的是当前可运行实现，而不是论文两套子模型的逐项复现。

1. **总质量代理层**：按 $m=1.0+payload_{kg}$ 构造总质量，其中 1.0 kg 是项目采用的机体基准质量，`payload_kg` 为载荷质量。
2. **运动状态指示层**：实际合速度大于 0.1 m/s 时令 $I_{move}=1$，否则为 0。该变量只是单一全局模型中的一列，不会切换到另一套回归系数。
3. **7 维物理项展开层**：依次形成 $I_{move}$、$am$、$v_zm$、$v_{xy}^2m^{2/3}$、$v_z^2m^{2/3}$、$m$ 和风速 $w$。其中 $a$ 使用三轴动态加速度模；代码中的 $\operatorname{sign}(v_z)|v_z|m$ 数值上等于 $v_zm$。每项只使用当前记录，不包含历史序列。
4. **标准化层**：用训练集均值和标准差分别标准化 7 个输入量，零标准差以 1 替代；再加入常数截距列，因此模型共有 8 个回归系数。
5. **线性/岭闭式解层**：验证集比较 $\alpha\in\{0,0.01,0.1,1,10,100\}$，最终选择 $\alpha=0$，即普通最小二乘。预测值经 $\max(0,\cdot)$ 截断。该模型无记忆单元，不能显式利用阶段切换前后的历史。

$$
\mathbf u_i=\left[I_{move},\;am,\;v_zm,\;v_{xy}^2m^{2/3},\;v_z^2m^{2/3},\;m,\;w\right]_i^\mathsf{T},\qquad z_{i,j}=\frac{u_{i,j}-\mu_j}{\sigma_j}
$$

$$
\hat P_i=\max\left(0,\;\beta_0+\sum_{j=1}^{7}\beta_j z_{i,j}\right)
$$

其中：

- $\mathbf u_i$：第 $i$ 条记录的 7 维物理构造量向量；$u_{i,j}$：该向量的第 $j$ 个分量。
- $I_{move}$：运动状态指示量，实际合速度大于 0.1 m/s 时为 1，否则为 0。
- $a$：三轴动态加速度模，单位 $\mathrm{m/s^2}$；$m$：机体基准质量与载荷质量之和，单位 kg。
- $v_z$：带符号垂直速度，单位 m/s；$v_{xy}$：水平速度，单位 m/s。
- $w$：风速，单位 m/s。
- $\mu_j$、$\sigma_j$：第 $j$ 个构造量在训练集上的均值和标准差；$z_{i,j}$：对应标准化值。
- $\hat P_i$：第 $i$ 条记录截断为非负值后的预测功率，单位 W。
- $\beta_0$：截距；$\beta_j$：第 $j$ 个标准化物理项的回归系数。
- $i$：记录索引；$j$：物理构造量索引；$\max(0,\cdot)$：把负预测截断为 0。

### 两层双向 LSTM <sup>[A3]</sup>

**实现来源：** Muli 等，2022，A Comparative Study on Energy Consumption Models for Drones。

Muli 等使用两层堆叠双向 LSTM，每层每方向 128 个 hidden cells，并在后端使用 Dropout 与 Dense(tanh)。本工程保留两层双向结构，在每方向 64 和 128 单元之间用验证集选择；最终选中每方向 64 单元、Dropout 0.30，共 149,057 个可训练参数。

1. **输入与填充层**：每条样本由同一 flight 中当前点及之前最多 16 个点组成，得到 $B\times17\times23$ 张量。flight 开头不足 17 步时复制最早记录到左侧；23 个通道均使用训练集统计量标准化。
2. **第 1 个双向 LSTM 层**：正向和反向各有 64 个记忆单元。两方向在 17 步缓存窗口内分别从首端和末端扫描，逐步更新输入门、遗忘门、候选记忆和输出门；沿特征维拼接后，每个时间位置输出 128 维。
3. **第 2 个双向 LSTM 层**：输入和输出均为 $B\times17\times128$。PyTorch LSTM 在第 1 层到第 2 层之间使用 0.30 Dropout；第二层正向与反向最终 hidden state 各为 $B\times64$。
4. **状态拼接层**：将第二层正向与反向最终 hidden state 拼成 $B\times128$。反向分支只反向读取已经缓存的“历史至当前”窗口，不访问预测时刻之后的记录。
5. **回归头**：执行 `Dropout(0.30) → Linear(128,32) → Tanh → Linear(32,1)`，输出一个标准化功率，再按训练目标均值与标准差还原为 W 并截断为非负值。
6. **训练选择层**：使用 AdamW、SmoothL1 损失和梯度范数 5.0 裁剪；每个候选按验证损失保存最佳 epoch，再用验证功率 WAPE 与 flight 能耗 WAPE 的组合分数比较候选。本次 64 单元候选在第 24 轮取得最终保存状态。

$$
i_t=\sigma(W_i x_t+U_i h_{t-1}+b_i),\quad f_t=\sigma(W_f x_t+U_f h_{t-1}+b_f)
$$

$$
\tilde c_t=\tanh(W_cx_t+U_ch_{t-1}+b_c),\quad c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$$

$$
o_t=\sigma(W_o x_t+U_o h_{t-1}+b_o),\quad h_t=o_t\odot\tanh(c_t)
$$

其中：

- $x_t$：时间步 $t$ 的 23 维标准化输入；$h_{t-1}$：前一时间步隐状态。
- $i_t$、$f_t$、$o_t$：输入门、遗忘门和输出门向量。
- $\tilde c_t$：候选记忆；$c_t$：当前记忆状态；$c_{t-1}$：前一时间步记忆状态。
- $h_t$：当前隐状态；双向层分别计算正向与反向 $h_t$ 后再拼接。
- $W_i,W_f,W_c,W_o$：输入到各门或候选记忆的权重矩阵。
- $U_i,U_f,U_c,U_o$：上一隐状态到各门或候选记忆的循环权重矩阵。
- $b_i,b_f,b_c,b_o$：相应偏置向量。
- $\sigma$：Sigmoid 激活函数；$\tanh$：双曲正切；$\odot$：逐元素乘法；$t$：窗口内时间步索引。

### LR-TCN-SMA <sup>[A4]</sup>

**实现来源：** Dudukcu 等，2024，UAV instantaneous power prediction using LR-TCN with simple moving average。

Dudukcu 等提出以 LeakyReLU 替代常规 TCN 激活并引入简单移动平均（SMA）特征。本工程把 23 个字段全部先做历史 SMA，再送入统一 TCN，没有复现原文原始特征与 SMA 特征的并行接口。验证集在 48/64 通道和 5/10 点 SMA 两组候选中选中 64 通道、10 点 SMA、Dropout 0.12，共 67,841 个可训练参数。

1. **SMA 输入层**：对每个 flight、每个特征在当前点及之前最多 9 个点上求均值；flight 开头使用实际可用点数。平滑后再用训练集统计量标准化，并构造 $B\times17\times23$ 的历史至当前窗口。
2. **残差块 1**：第一层为左因果 `Conv1d(23,64,k=3,d=1)`，第二层为 `Conv1d(64,64,k=3,d=1)`；每层后依次执行 LeakyReLU(0.05) 和 Dropout(0.12)。旁路用 $1\times1$ 卷积把 23 通道投影到 64 通道，主路与旁路相加。
3. **残差块 2**：两层 $64\to64$ 左因果卷积，膨胀率均为 2；旁路为恒等映射。该块扩大当前输出能够访问的历史间隔。
4. **残差块 3**：两层 $64\to64$ 左因果卷积，膨胀率均为 4。三块理论感受野为 $1+2(k-1)(1+2+4)=29$ 步，但实际输入只有 17 步，因此真实数据上下文最多为当前点及前 16 点，其余位置来自左填充。
5. **当前时刻读出层**：取第三个残差块最后时间位置的 64 维向量，经 `Linear(64,1)` 输出标准化功率，再还原为 W 并截断为非负值。TCN 卷积不读取预测时刻之后的记录，`dt_seconds` 与其他特征一起参与 SMA 和卷积，原始真实时间间隔仍用于最终能耗积分。
6. **训练选择层**：使用 AdamW、SmoothL1 和梯度裁剪；候选内部按验证损失保留最佳 epoch，再按统一选择分数确定结构。本次最终候选保存于第 24 轮。

$$
q_t=\min(q,t+1),\qquad x_t^{SMA}=\frac{1}{q_t}\sum_{j=0}^{q_t-1}x_{t-j},\qquad h_t=\operatorname{LeakyReLU}(W*x_{\le t}+b)
$$

其中：

- $x_t$：时间步 $t$ 的 23 维原始输入向量。
- $x_t^{SMA}$：时间步 $t$ 的历史移动平均输入；实际 flight 起点按可用点数调整分母。
- $q$：设定的 SMA 最大窗口长度，本次为 10；$q_t$：时间步 $t$ 实际可用的平均点数；$j$：回看步索引。
- $x_{t-j}$：当前点之前第 $j$ 步的输入；$x_{\le t}$：不晚于当前点的输入序列。
- $W$：因果卷积核权重；$b$：卷积偏置；$*$：仅左侧填充的膨胀卷积运算。
- $h_t$：当前时间步的卷积特征；$t$：窗口内时间步索引。
- $\operatorname{LeakyReLU}(u)=\max(u,0)+0.05\min(u,0)$：本实现使用的带负半轴斜率激活函数。

### LSTM-Transformer <sup>[A5]</sup>

**实现来源：** Feng 等，2024，Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework。

Feng 等的方法先由 LSTM 提取短期变化，再由 Transformer 建模较长依赖，原始对象是电动汽车。当前 UAV 适配版在 $D=32/64$、1/2 个 Encoder 等候选中选中 $D=64$、1 层 LSTM、2 层 Transformer Encoder、4 个注意力头、128 维前馈层和 Dropout 0.10，共 105,089 个可训练参数。

1. **输入与填充层**：同一 flight 的当前点和之前最多 16 点组成 $B\times17\times23$ 张量；起点不足 17 步时复制最早记录到左侧，23 个特征按训练集统计量标准化。
2. **线性投影层**：`Linear(23,64)` 独立作用于每个时间步，把不同量纲的 23 维输入映射到统一的 64 维模型空间，输出 $B\times17\times64$。
3. **可学习位置编码层**：加入形状 $1\times17\times64$ 的参数矩阵，使注意力能够区分窗口首端、内部和当前时刻；位置参数与投影结果逐元素相加。
4. **单层 LSTM**：64 个记忆单元依次处理 17 个位置，输出仍为 $B\times17\times64$。该层先汇总相邻点的短期顺序信息，再交给自注意力。
5. **Transformer Encoder 1**：4 个注意力头并行工作，每个头的键/查询维度为 $64/4=16$；多头输出经残差连接和 LayerNorm，再通过 `Linear(64,128) → GELU → Dropout → Linear(128,64)` 前馈网络及第二组残差归一化。
6. **Transformer Encoder 2**：重复同样的 4 头注意力和 128 维前馈结构，在第一层关系表示上再次组合窗口内位置。实现未使用因果 mask，但整个 17 步输入窗只包含当前及过去记录，因此不会读取预测时刻之后的数据。
7. **归一化与回归头**：取第二个 Encoder 最后时间位置的 64 维向量，经 `LayerNorm(64) → Linear(64,32) → GELU → Linear(32,1)` 输出标准化功率，再还原为 W 并截断为非负值。
8. **训练选择层**：使用 AdamW、SmoothL1、梯度裁剪和验证早停。本次选中候选的最佳保存轮次为第 21 轮。

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

$$
head_r=\operatorname{Attention}(XW_r^Q,XW_r^K,XW_r^V),\qquad \operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_h)W^O
$$

其中：

- $X$：LSTM 输出的窗口序列矩阵，每条样本为 $17\times64$。
- $Q$、$K$、$V$：查询、键和值矩阵，用于计算位置间权重并加权汇总信息。
- $d_k$：单个注意力头的键维度，本次为 16；$\sqrt{d_k}$：抑制点积随维度增大的缩放项。
- $W_r^Q,W_r^K,W_r^V$：第 $r$ 个头的查询、键、值投影矩阵。
- $head_r$：第 $r$ 个注意力头输出；$r$：注意力头索引。
- $h$：注意力头总数，本次为 4；$\operatorname{Concat}$：沿特征维拼接各头输出。
- $W^O$：把多头拼接结果映射回 64 维的输出投影矩阵。
- $\operatorname{softmax}$：在键位置维度上把缩放点积转换为和为 1 的注意力权重。

### CNN-LSTM <sup>[A1]</sup>

**实现来源：** Luo 等，2025，RF-TLATT 论文表 8 列出的 CNN-LSTM 对比名称（二手来源）。

Luo 等仅在表 8 给出 CNN-LSTM 的对比结果，正文将其归于“Xuebing et al. (2024)”，但所给论文的参考文献表缺少相应完整条目，因而无法核实原模型层数与参数。当前目录实现的是可复现的 CNN-LSTM 工程基线，不冒充该未知原模型的直接复现。验证集在 48/64 通道候选中选中卷积通道 $C=48$、LSTM 隐层 $H=48$、Dropout 0.10，共 29,377 个可训练参数。

1. **输入与填充层**：当前点与同一 flight 的前 16 点组成 $B\times17\times23$ 标准化序列；flight 开头不足部分复制最早记录。转置后得到 Conv1d 所需的 $B\times23\times17$。
2. **卷积层 1**：`Conv1d(23,48,k=3,padding=1)` 同时汇总 23 个输入通道和相邻 3 个时间位置，输出 $B\times48\times17$；随后执行 `BatchNorm1d(48) → LeakyReLU(0.05) → Dropout(0.10)`。
3. **卷积层 2**：`Conv1d(48,48,k=3,padding=1) → BatchNorm1d(48) → LeakyReLU(0.05)`，继续组合第一层局部模式，输出形状仍为 $B\times48\times17$。第二层后没有额外 Dropout。
4. **单层 LSTM**：转回 $B\times17\times48$，由 48 个记忆单元顺序聚合卷积特征；输出整个序列并取最后时间位置的 $B\times48$ 向量。
5. **回归头**：对最后时刻向量执行 `Dropout(0.10) → Linear(48,1)`，得到标准化功率，再还原为 W 并截断为非负值。
6. **时间信息边界**：两层卷积使用对称 `padding=1`，中间位置会组合其左右相邻位置；但送入模型的 17 步缓冲区以当前预测时刻结束，所以不会访问该时刻之后的数据。训练使用 AdamW、SmoothL1 和梯度裁剪，本次最佳状态保存于第 20 轮。

$$
z_{c,t}=\operatorname{LeakyReLU}\left(\operatorname{BN}\left(\sum_{r=1}^{C_{in}}\sum_{\tau=-1}^{1}W_{c,r,\tau}x_{r,t+\tau}+b_c\right)\right)
$$

其中：

- $x_{r,t+\tau}$：输入通道 $r$ 在窗口位置 $t+\tau$ 的值；边界外位置由零填充。
- $C_{in}$：该卷积层的输入通道数，第一层为 23、第二层为 48。
- $W_{c,r,\tau}$：从输入通道 $r$、相对位置 $\tau$ 到输出通道 $c$ 的卷积权重。
- $b_c$：输出通道 $c$ 的偏置；$z_{c,t}$：卷积、批归一化和激活后的局部特征。
- $c$：输出通道索引；$r$：输入通道索引；$t$：窗口内位置；$\tau\in\{-1,0,1\}$：三点卷积核的相对位置。
- $\operatorname{BN}$：批归一化；$\operatorname{LeakyReLU}$：负半轴斜率为 0.05 的激活函数。

## 7. 调参与公平性设置

所有需要学习的候选均只使用训练 flight 拟合，在独立 validation flight 上选择，test flight 只在结构与参数固定后用于最终评价。每个深度候选以 SmoothL1 损失最多训练 24 轮，验证损失连续 4 轮不改善则早停，并保存该候选验证损失最低的 epoch；不同候选再按功率 WAPE + 0.5×flight 能耗 WAPE 选择。线性模型只在验证集选择正则项和 RF 轻量基线的相位阈值。候选明细位于各目录 `out/model/tuning_results.csv`。

主方法 3.0 的固定 TCN checkpoint 使用 12 s/100 步输入窗和 `[64,64,64,32]` 通道；RLS 使用遗忘因子 0.90、初始协方差 0.25、无预热窗口，这些参数来自 3.0 验证集搜索。LSTM、LR-TCN-SMA、LSTM-Transformer、CNN-LSTM 使用统一 17 步窗口并在本项目验证集重新选参；RF 相位基线和 Physical MLR 是当前记录回归，不使用序列窗口。由于主方法另有在线真实能量反馈，本实验在数据划分与指标上保持一致，但并非严格同信息量的消融比较。

### 7.1 低效候选删除与替换

先前两项候选在相同测试集上的误差显著高于时序模型，因此不再进入有效算法清单。替换算法均来自所给论文体系：LSTM-Transformer 是 Feng 等提出的混合框架<sup>[A5]</sup>，CNN-LSTM 是 Luo 等表 8 报告的对照名称<sup>[A1]</sup>。替换前后的固定结果如下。


| 被淘汰候选               | 来源            | 原功率 MAE(W) | 原功率 WAPE(%) | 原能耗 WAPE(%) | 替换算法         | 新功率 WAPE(%) | 新能耗 WAPE(%) |
| ------------------------ | --------------- | ------------: | -------------: | -------------: | ---------------- | -------------: | -------------: |
| 任务级多项式 Elastic Net | <sup>[B4]</sup> |       107.736 |         26.516 |          4.982 | LSTM-Transformer |          7.293 |          2.542 |
| 经验多相位能耗剖面       | <sup>[B5]</sup> |        73.492 |         18.088 |          6.779 | CNN-LSTM         |          7.770 |          2.997 |

LSTM-Transformer 相对任务级多项式 Elastic Net 的功率 WAPE 下降 72.50%，能耗 WAPE 下降 48.98%；CNN-LSTM 相对经验多相位能耗剖面的功率 WAPE 下降 57.04%，能耗 WAPE 下降 55.79%。两项旧候选已从有效算法清单和项目目录中删除，表中只保留替换依据。

## 8. 评价指标与结果

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

$$
S_{val}=\mathrm{WAPE}_{power,val}+0.5\,\mathrm{WAPE}_{energy,val}
$$

其中：

- $S_{val}$：验证集候选选择分数，越小越好。
- $\mathrm{WAPE}_{power,val}$：验证集逐点功率 WAPE，单位 %。
- $\mathrm{WAPE}_{energy,val}$：验证集按 flight 汇总后的能耗 WAPE，单位 %。
- $0.5$：能耗 WAPE 在选择分数中的固定权重。

该分数只在 28 个 validation flight 上计算，用于比较候选结构、正则强度和相位阈值；测试集 28 个 flight 在参数固定后才运行。

### 8.1 测试集总体指标


| 方法                     | 功率 MAE(W) | 功率 RMSE(W) | 功率 R² | 功率 WAPE(%) | flight能耗 MAE(Wh) | flight能耗 RMSE(Wh) | flight能耗 WAPE(%) |
| ------------------------ | ----------: | -----------: | -------: | -----------: | -----------------: | ------------------: | -----------------: |
| 3.0 TCN + 秒级能量 RLS   |      27.991 |       43.455 |   0.9623 |        6.889 |              0.064 |               0.089 |              0.292 |
| RF-TLATT 思路相位岭回归  |      45.667 |       73.844 |   0.8911 |       11.240 |              0.569 |               0.722 |              2.600 |
| 物理特征 MLR（全局适配） |      50.533 |       89.757 |   0.8391 |       12.437 |              0.693 |               0.873 |              3.169 |
| LSTM 时序功率预测        |      31.026 |       48.138 |   0.9537 |        7.636 |              0.504 |               0.681 |              2.303 |
| LR-TCN-SMA               |      31.304 |       47.853 |   0.9543 |        7.705 |              0.532 |               0.725 |              2.430 |
| LSTM-Transformer         |      29.630 |       45.541 |   0.9586 |        7.293 |              0.556 |               0.766 |              2.542 |
| CNN-LSTM                 |      31.569 |       47.149 |   0.9556 |        7.770 |              0.656 |               0.886 |              2.997 |

### 8.2 结果与讨论

测试集共有 **45,237** 条采样记录、**28** 个 flight。实测瞬时功率均值为 406.31 W、中位数为 484.54 W、范围为 0.00–919.29 W；按真实 `dt_seconds` 积分后，28 个 flight 的实测总能耗为 612.663 Wh。阶段样本数为 idle（停机/低速）10,616、ascent（上升）19,819、descent（下降）12,284、cruise（巡航）2,518。功率区间样本数为 0–50 W 9,300、50–300 W 859、300–450 W 6,595、450–600 W 22,870、>600 W 5,613。

按样本功率 WAPE 排名，3.0 TCN + 秒级能量 RLS 最低（6.889%），其次为 LSTM-Transformer（7.293%）；按 flight 能耗 WAPE 排名，3.0 TCN + 秒级能量 RLS 最低（0.292%）。主方法的绝对误差 P50/P90/P95 为 17.79/65.84/88.75 W，说明大多数采样点误差集中在较小范围，但 P95 仍保留少量起降和状态切换尾部。

主方法与最强无测试期目标反馈对照 LSTM-Transformer 的功率 MAE 分别为 27.991 W 和 29.630 W，差值为 1.639 W；功率 WAPE 相对下降 5.53%。flight 能耗 WAPE 分别为 0.292% 和 2.542%。

同一 3.0 checkpoint 的原始 TCN 功率 WAPE 为 7.811%，加入在线 RLS 后为 6.889%，相对下降 11.81%；原始 TCN 的 flight 能耗 WAPE 为 2.728%，RLS 后为 0.292%，相对下降 89.28%。这组同模型前后值说明，主方法的任务级优势主要受在线校正影响。RLS 在每个完整秒窗结束后读取该窗真实能量，并把更新参数用于下一窗；其余对比算法没有同样的测试期目标反馈，因此不能把能耗差异全部归因于 TCN 表征能力。

阶段结果也呈现出不同的误差来源。idle 的功率分母较小，WAPE 往往高于 MAE 所反映的绝对误差；ascent 和 descent 的垂直速度变化使功率尖峰更密集，RMSE 对这些点更敏感；cruise 的输入变化相对连续，通常更适合时序模型。功率分箱中 450–600 W 是样本主体，因而该区间的 MAE 对总体结果贡献最大；50–300 W 样本较少但多出现在过渡段，适合用于检查模型响应延迟。

模型之间的差异与结构特点一致：LSTM-Transformer 的 idle MAE 为 11.83 W，是七种方法中的最低值；LR-TCN-SMA 的 idle RMSE 为 34.58 W，是该阶段最低值。主方法在 ascent、descent 和 cruise 的 MAE 分别为 32.33、34.09 和 27.71 W，均低于六个对照。Physical MLR 在 descent 的 MAE 为 62.67 W，相位岭回归在 idle 的 MAE 为 46.05 W，反映无历史线性模型对低功率混合分布和状态变化的刻画较弱。以上只描述当前固定划分和单次训练结果；未做重复试验、置信区间或消融，不能据此断言结构差异具有统计显著性。

### 8.3 总体图谱与逐图分析

13 张总体图均基于同一批 7 种算法、45,237 条测试记录和 28 个 flight；散点密集的图按图注规则等间隔抽样，但所有表格指标仍用完整测试集计算。图例中的每条线或颜色都对应一个算法目录。当前功率 WAPE 最低的是 3.0 TCN + 秒级能量 RLS（6.889%），第二名为 LSTM-Transformer（7.293%）。

#### `comparison_metrics.png`

![comparison_metrics.png](./out/comparison_metrics.png)

左右两个横向柱图分别以算法为纵轴、样本功率 WAPE（Power WAPE，%）和 flight 能耗 WAPE（Flight-energy WAPE，%）为横轴。蓝柱对应逐点功率，橙柱对应任务累计能耗，柱越短越好。
蓝色功率柱从主方法的 6.889% 到 物理特征 MLR（全局适配） 的 12.437%；橙色能耗柱从主方法的 0.292% 到 物理特征 MLR（全局适配） 的 3.169%。两组柱使用不同评价对象，不能把蓝柱与橙柱直接相减。

#### `comparison_error_boxplot.png`

![comparison_error_boxplot.png](./out/comparison_error_boxplot.png)

横轴为算法，纵轴为绝对功率误差（W）。箱体从 Q1 到 Q3，红线为 P50，图中隐藏极端离群点以便比较主体分布；中位数低且箱体短表示典型误差小、稳定性高。
3.0 TCN + 秒级能量 RLS 的绝对误差中位数为 17.79 W，P95 为 88.75 W。
RF-TLATT 思路相位岭回归 的绝对误差中位数为 28.67 W，P95 为 148.37 W。
物理特征 MLR（全局适配） 的绝对误差中位数为 29.54 W，P95 为 159.32 W。
LSTM 时序功率预测 的绝对误差中位数为 20.28 W，P95 为 100.83 W。
LR-TCN-SMA 的绝对误差中位数为 20.62 W，P95 为 98.97 W。
LSTM-Transformer 的绝对误差中位数为 19.47 W，P95 为 94.78 W。
CNN-LSTM 的绝对误差中位数为 20.85 W，P95 为 96.75 W。

#### `comparison_phase_wape.png`

![comparison_phase_wape.png](./out/comparison_phase_wape.png)

横轴为 idle、ascent、descent、cruise，纵轴为算法；颜色和格内数字都是阶段功率 WAPE（%）。颜色越深表示该算法在该阶段误差越大，适合定位阶段性退化。
每个格子由该阶段全部记录先汇总分子、分母后计算 WAPE。idle 最低为 LSTM-Transformer 20.88%；ascent 最低为 3.0 TCN + 秒级能量 RLS 6.02%；descent 最低为 3.0 TCN + 秒级能量 RLS 7.13%；cruise 最低为 3.0 TCN + 秒级能量 RLS 5.56%。

#### `comparison_flight_energy_scatter.png`

![comparison_flight_energy_scatter.png](./out/comparison_flight_energy_scatter.png)

每个小图对应一种算法，横轴为实测 flight 能耗（Wh），纵轴为预测 flight 能耗（Wh）；蓝点为同一测试 flight，黑色虚线为理想线。小图之间可以直接比较点云离线程度。
每个小图有 28 个点，线上方为能耗高估、下方为低估。3.0 TCN + 秒级能量 RLS 平均偏差 +0.041 Wh、最大绝对偏差 0.261 Wh；RF-TLATT 思路相位岭回归 平均偏差 +0.111 Wh、最大绝对偏差 2.230 Wh；物理特征 MLR（全局适配） 平均偏差 +0.173 Wh、最大绝对偏差 1.990 Wh；LSTM 时序功率预测 平均偏差 +0.326 Wh、最大绝对偏差 1.938 Wh；LR-TCN-SMA 平均偏差 +0.299 Wh、最大绝对偏差 2.313 Wh；LSTM-Transformer 平均偏差 +0.384 Wh、最大绝对偏差 2.438 Wh；CNN-LSTM 平均偏差 +0.584 Wh、最大绝对偏差 2.564 Wh。

#### `comparison_metric_heatmap.png`

![comparison_metric_heatmap.png](./out/comparison_metric_heatmap.png)

横轴为 Power MAE、Power RMSE、Power WAPE、Energy MAE、Energy WAPE，纵轴为算法。底色为按每列最小值和最大值归一化后的误差，绿色较优、红色较差；格内数字保留原始单位值。
主方法在五列中均为最小值，因此五格都位于绿色端；最强无测试期目标反馈方法 LSTM-Transformer 的功率 MAE/RMSE/WAPE 为 29.63 W、45.54 W、7.29%。底色按每一列单独做最小-最大归一化，格内数字仍保留 W、Wh 或 % 的原单位。

#### `comparison_error_cdf.png`

![comparison_error_cdf.png](./out/comparison_error_cdf.png)

横轴为绝对功率误差（W），纵轴为经验 CDF（经验累积分布函数）；每条线由图例标识一种算法。在固定阈值处，线越高表示更多样本达到该误差要求，曲线越靠左上越好。
在纵轴 $F(a)=0.50/0.90/0.95$ 处向下读取横轴，分别得到 P50/P90/P95；当前数据依次为：3.0 TCN + 秒级能量 RLS 17.79/65.84/88.75 W；RF-TLATT 思路相位岭回归 28.67/104.40/148.37 W；物理特征 MLR（全局适配） 29.54/107.81/159.32 W；LSTM 时序功率预测 20.28/75.36/100.83 W；LR-TCN-SMA 20.62/74.82/98.97 W；LSTM-Transformer 19.47/71.49/94.78 W；CNN-LSTM 20.85/74.12/96.75 W。

#### `comparison_overall_rank.png`

![comparison_overall_rank.png](./out/comparison_overall_rank.png)

横轴为五项误差指标的平均名次，纵轴为算法。每项指标先按从小到大排名，再求平均；柱越短表示功率与能耗折中越好。该图是描述性排序，不是统计显著性检验。
五列误差分别从小到大排名后求平均，结果为：3.0 TCN + 秒级能量 RLS 1.0、LSTM-Transformer 2.8、LSTM 时序功率预测 3.0、LR-TCN-SMA 3.6、CNN-LSTM 5.0、RF-TLATT 思路相位岭回归 5.6、物理特征 MLR（全局适配） 7.0。该值只作描述性折中，不是显著性检验。

#### `comparison_power_timeseries.png`

![comparison_power_timeseries.png](./out/comparison_power_timeseries.png)

横轴为测试集中功率标准差最大的同一 flight 的飞行时间（s），纵轴为功率（W）。黑色粗线是实测功率，其他彩色线分别对应图例算法；若该 flight 超过 700 条记录，图中只显示前 700 条。所有线共享同一时间轴，可比较峰值位置、过冲和稳态偏差。
程序选择功率标准差最大的 flight 113，显示其前 700 条记录；黑线实测功率范围为 0.00–781.10 W。该显示区间内，3.0 TCN + 秒级能量 RLS MAE 26.90 W、RF-TLATT 思路相位岭回归 MAE 44.47 W、物理特征 MLR（全局适配） MAE 41.97 W、LSTM 时序功率预测 MAE 35.72 W、LR-TCN-SMA MAE 33.41 W、LSTM-Transformer MAE 35.22 W、CNN-LSTM MAE 39.35 W。

#### `comparison_error_violin.png`

![comparison_error_violin.png](./out/comparison_error_violin.png)

横轴为算法，纵轴为绝对功率误差（W）。每个算法每隔 5 条测试记录取 1 条，共 9,048 个误差值；小提琴宽度表示该误差附近的核密度，内部两组水平标记分别表示均值和中位数。抽样不参与指标计算。
图中密度和内部标记按每 5 条取 1 条的显示样本计算：3.0 TCN + 秒级能量 RLS 抽样均值/中位数 27.70/17.35 W；RF-TLATT 思路相位岭回归 抽样均值/中位数 45.42/28.29 W；物理特征 MLR（全局适配） 抽样均值/中位数 50.52/29.21 W；LSTM 时序功率预测 抽样均值/中位数 31.01/20.43 W；LR-TCN-SMA 抽样均值/中位数 31.18/20.83 W；LSTM-Transformer 抽样均值/中位数 29.55/19.59 W；CNN-LSTM 抽样均值/中位数 31.46/20.96 W。

#### `comparison_bland_altman.png`

![comparison_bland_altman.png](./out/comparison_bland_altman.png)

每个子图是一种算法，横轴为实测与预测功率的均值（W），纵轴为残差（W）。45,237 条记录按步长 7 抽样，共显示 6,463 个点；红线为该抽样点集的平均偏差，橙色虚线为平均偏差 ±1.96 个样本标准差。若残差随横轴明显倾斜，说明存在功率相关偏差。
以下为红色平均残差及两条橙色 95% 一致性界限：3.0 TCN + 秒级能量 RLS +0.39 W（-86.64, 87.41 W）；RF-TLATT 思路相位岭回归 +1.89 W（-144.33, 148.11 W）；物理特征 MLR（全局适配） +3.08 W（-173.19, 179.34 W）；LSTM 时序功率预测 +5.83 W（-89.93, 101.58 W）；LR-TCN-SMA +5.29 W（-89.98, 100.57 W）；LSTM-Transformer +6.94 W（-83.35, 97.23 W）；CNN-LSTM +10.59 W（-82.20, 103.37 W）。界限是描述性样本界限，不是预测区间。

#### `comparison_energy_error_distribution.png`

![comparison_energy_error_distribution.png](./out/comparison_energy_error_distribution.png)

横轴为算法，纵轴为 flight 能耗误差（Wh）；箱体表示任务级误差的主体范围，红线为中位数，黑色水平虚线为零误差。它与功率箱线图不同，关注的是整段任务的累计偏差。
任务级误差按预测能耗减实测能耗计算：3.0 TCN + 秒级能量 RLS 中位数 +0.020 Wh、范围 -0.118 至 +0.261 Wh；RF-TLATT 思路相位岭回归 中位数 +0.128 Wh、范围 -1.183 至 +2.230 Wh；物理特征 MLR（全局适配） 中位数 +0.073 Wh、范围 -1.354 至 +1.990 Wh；LSTM 时序功率预测 中位数 +0.293 Wh、范围 -0.574 至 +1.938 Wh；LR-TCN-SMA 中位数 +0.304 Wh、范围 -0.688 至 +2.313 Wh；LSTM-Transformer 中位数 +0.299 Wh、范围 -0.494 至 +2.438 Wh；CNN-LSTM 中位数 +0.471 Wh、范围 -0.451 至 +2.564 Wh。正负逐点误差可能在积分中抵消，因此应与功率箱线图联合阅读。

#### `comparison_phase_metrics.png`

![comparison_phase_metrics.png](./out/comparison_phase_metrics.png)

三个并列热图分别给出阶段 MAE（W）、RMSE（W）和 WAPE（%），横轴为四阶段，纵轴为算法，格内数字为原始指标。并列观察可以区分低功率分母放大的百分比误差与真实瓦特误差。
三张热图分别使用 W、W 和 %。idle 的最低 MAE 为 LSTM-Transformer 11.83 W，最低 RMSE 为 LR-TCN-SMA 34.58 W；ascent 的最低 MAE 为 3.0 TCN + 秒级能量 RLS 32.33 W，最低 RMSE 为 3.0 TCN + 秒级能量 RLS 43.78 W；descent 的最低 MAE 为 3.0 TCN + 秒级能量 RLS 34.09 W，最低 RMSE 为 3.0 TCN + 秒级能量 RLS 49.54 W；cruise 的最低 MAE 为 3.0 TCN + 秒级能量 RLS 27.71 W，最低 RMSE 为 3.0 TCN + 秒级能量 RLS 38.22 W。idle 的 WAPE 还会被接近 0 W 的真实值放大。

#### `comparison_metric_radar.png`

![comparison_metric_radar.png](./out/comparison_metric_radar.png)

极坐标轴分别为 Power MAE、Power RMSE、Power WAPE、Energy MAE、Energy WAPE；每项先转换为 $1-(x-x_{min})/(x_{max}-x_{min})$ 的相对得分，越靠外表示该列相对误差越低。它只反映当前七种方法之间的相对位置。
主方法在五项误差中均为当前最小值，因而五个顶点都位于外圈；六种无测试期目标反馈对照中，LSTM-Transformer 的三项功率误差最低，LSTM 的两项能耗误差最低。雷达值是七种方法内的最小-最大相对得分，不代表绝对精度，也不包含 R² 或计算成本。

### 变量、坐标轴和字段词典


| 字段或指标                          | 中文全称           | English full name                          | 单位/范围                  | 计算方式和含义                                                             |
| ----------------------------------- | ------------------ | ------------------------------------------ | -------------------------- | -------------------------------------------------------------------------- |
| `flight`                            | 飞行任务编号       | Flight identifier                          | 无量纲                     | 同一编号的连续记录属于同一任务，能耗按该字段分组。                         |
| `route`                             | 航线编号           | Route identifier                           | —                         | 本实验仅保留 R1。                                                          |
| `time`                              | 飞行内相对时间     | Flight-relative time                       | s                          | 每个 flight 从 0 s 开始的时间坐标，所有时序图横轴使用它。                  |
| `dt_seconds`（$\Delta t_i$）        | 采样时间间隔       | Sampling time interval                     | s                          | 相邻时间戳之差；能耗积分使用真实值，不用固定步长替代。                     |
| `power_w`（$y_i$）                  | 实测瞬时功率       | Measured instantaneous power               | W                          | `max(battery_voltage × battery_current, 0)`，作为监督目标和散点图横坐标。 |
| `predicted_power_w`（$\hat y_i$）   | 预测瞬时功率       | Predicted instantaneous power              | W                          | 算法输出的非负功率，作为时序图预测线和散点图纵坐标。                       |
| `residual_w`（$e_i$）               | 功率残差           | Power residual                             | W                          | $e_i=\hat y_i-y_i$；正值表示高估，负值表示低估。                           |
| `absolute_error_w`                  | 功率绝对误差       | Absolute power error                       | W                          | $                                                                          |
| `actual_energy_wh`                  | 实测采样能量       | Measured interval energy                   | Wh                         | $P_i\Delta t_i/3600$。                                                     |
| `predicted_energy_wh`               | 预测采样能量       | Predicted interval energy                  | Wh                         | $\hat P_i\Delta t_i/3600$。                                                |
| `actual_energy_wh`（flight汇总）    | 实测 flight 总能耗 | Measured flight energy                     | Wh                         | 同一 flight 的`actual_energy_wh` 求和。                                    |
| `predicted_energy_wh`（flight汇总） | 预测 flight 总能耗 | Predicted flight energy                    | Wh                         | 同一 flight 的`predicted_energy_wh` 求和。                                 |
| `energy_error_wh`                   | flight 能耗误差    | Flight-energy error                        | Wh                         | 预测总能耗减实测总能耗；正值为任务级高估。                                 |
| `phase_name`                        | 飞行阶段           | Flight phase                               | idle/ascent/descent/cruise | 由$v_z$ 和水平速度阈值推导，不是人工标注。                                 |
| MAE                                 | 平均绝对误差       | Mean absolute error                        | W 或 Wh                    | 所有误差绝对值的算术平均，代表典型偏差。                                   |
| RMSE                                | 均方根误差         | Root mean square error                     | W 或 Wh                    | 误差平方平均后开方，对尖峰误差更敏感。                                     |
| $R^2$                               | 决定系数           | Coefficient of determination               | 无量纲                     | 相对于真实均值基线的解释度。                                               |
| MAPE                                | 平均绝对百分比误差 | Mean absolute percentage error             | %                          | 逐点相对误差平均；接近 0 的真实功率被排除。                                |
| WAPE                                | 加权绝对百分比误差 | Weighted absolute percentage error         | %                          | 总绝对误差除以真实值绝对和，适合跨功率段总体比较。                         |
| CDF                                 | 经验累积分布函数   | Empirical cumulative distribution function | 0–1                       | $F(a)=\#\{                                                                 |
| P50/P90/P95                         | 误差分位数         | Error percentiles                          | W                          | 绝对误差排序后第 50%、90%、95% 位置，反映典型和尾部风险。                  |

## 9. 输出文件与目录

```text
Comparison-algorithm/
├─ config.py                         # 路径、23维特征、候选参数和图表名称
├─ main.py                           # 全部算法统一入口
├─ comparison_engine.py              # 算法、训练、评估和绘图
├─ generate_documentation.py         # 从最终CSV生成README
├─ proposed_tcn_rls/out/             # 主方法逐点/flight结果和13张图
├─ rf_tlatt_lite/out/                # 相位轻量基线结果和13张图
├─ physical_mlr/out/                 # 物理MLR结果和13张图
├─ lstm/out/                         # 双向LSTM结果和13张图
├─ lr_tcn_sma/out/                   # LR-TCN-SMA结果和13张图
├─ lstm_transformer/out/             # LSTM-Transformer结果和13张图
├─ cnn_lstm/out/                     # CNN-LSTM结果和13张图
└─ out/                              # 跨算法CSV、JSON和13张总体图
```

每个算法 `out/figures/` 的 13 张图依次覆盖散点、典型时序、残差直方图、残差-功率、功率分箱、flight 能耗散点、flight 误差排序、累计能耗、阶段柱图、CDF、最大误差局部放大、flight 能耗误差直方图和阶段误差箱线图；根目录 `out/` 另有 13 张跨算法图，包括柱图、箱线图、阶段热图、散点矩阵、归一化热图、CDF、综合排名、同 flight 曲线、小提琴图、Bland–Altman 图、能耗误差分布、阶段三指标热图和雷达图。

## 10. 参考文献

### 10.1 对比算法与方法来源文献

- **[A0]** Energy-prediction 3.0 本项目 TCN+RLS 方法及固定模型输出；TCN 与 RLS 的理论依据分别见 [A6]、[A7]，实验结果组织方式参照 [A1]。
- **[A1]** Luo, W., Li, N., Xiong, Z., Chen, W., Li, Y., Tang, C., Li, Y., Dong, C. *Phase-based power prediction for quadrotor UAVs with RF-TLATT*. Energy, 2025, 335: 138208. DOI: 10.1016/j.energy.2025.138208.
- **[A2]** Jastrzębska, A., Lerke, M., Kwiatkowski, W. *Prediction of energy consumption in unmanned aerial vehicles*. Electric Power Systems Research, 2026, 257: 113008. DOI: 10.1016/j.epsr.2026.113008.
- **[A3]** Muli, C., Park, S., Liu, M. *A Comparative Study on Energy Consumption Models for Drones*. In: Internet of Things, GIoTS 2022, Lecture Notes in Computer Science, vol. 13533, Springer, 2022, pp. 199–210. DOI: 10.1007/978-3-031-20936-9_16.
- **[A4]** Dudukcu, H. V., Taskiran, M., Kahraman, N. *UAV instantaneous power consumption prediction using LR-TCN with simple moving average*. Concurrency and Computation: Practice and Experience, 2024, 36(3): e7913. DOI: 10.1002/cpe.7913.
- **[A5]** Feng, Z., Zhang, J., Jiang, H., Yao, X., Qian, Y., Zhang, H. *Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework*. Energy, 2024, 302: 131780. DOI: 10.1016/j.energy.2024.131780.
- **[A6]** Bai, S., Kolter, J. Z., Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271, 2018. DOI: 10.48550/arXiv.1803.01271.
- **[A7]** Sayed, A. H. *Adaptive Filters*. Wiley, 2008. DOI: 10.1002/9780470374122.

### 10.2 背景与应用参考文献

- **[B1]** Rodrigues, T. A., Patrikar, J., Choudhry, A., et al. *In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery*. Scientific Data, 2021, 8(1): 155. DOI: 10.1038/s41597-021-00930-x.
- **[B2]** Dorling, K., Heinrichs, J., Messier, G. G., Magierowski, S. *Vehicle Routing Problems for Drone Delivery*. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2017, 47(1): 70–85. DOI: 10.1109/TSMC.2016.2582745.
- **[B3]** Cabuk, U. C., Tosun, M., Dagdeviren, O., Ozturk, Y. *Modeling Energy Consumption of Small Drones for Swarm Missions*. IEEE Transactions on Intelligent Transportation Systems, 2024, 25(8): 10176–10189. DOI: 10.1109/TITS.2024.3350042.
- **[B4]** Prasetia, A. S., Wai, R.-J., Wen, Y.-L., Wang, Y.-K. *Mission-Based Energy Consumption Prediction of Multirotor UAV*. IEEE Access, 2019, 7: 33055–33063. DOI: 10.1109/ACCESS.2019.2903644.
- **[B5]** Dietrich, T., Krug, S., Zimmermann, A. *An Empirical Study on Generic Multicopter Energy Consumption Profiles*. 2017 Annual IEEE International Systems Conference (SysCon), 2017. DOI: 10.1109/SYSCON.2017.7934762.

PDF 原件目录：`C:\Users\18030\Desktop\GPT-files\Energy-prediction-Comparison-algorithm\pdf`。所收录论文分别采用瞬时功率、任务能耗和固定时长归一化能耗等口径，研究对象也涵盖四旋翼、多旋翼和电动汽车；其原文指标不能与本实验按真实 `dt_seconds` 积分得到的 R1 指标直接换算或相减，相关论文只用于说明算法机制、应用场景和图表组织依据。
