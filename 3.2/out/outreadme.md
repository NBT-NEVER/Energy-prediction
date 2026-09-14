# 实验 3.2 输出、算法与结果报告

## 目录

- [A. 方法与数学建模](#a-方法与数学建模)
  - [A.1 问题背景与预测对象](#a1-问题背景与预测对象)
  - [A.2 数据来源与无泄漏切分](#a2-数据来源与无泄漏切分)
  - [A.3 特征与时间积分](#a3-特征与时间积分)
  - [A.4 时间感知 TCN](#a4-时间感知-tcn)
  - [A.5 联合功率和能量损失](#a5-联合功率和能量损失)
  - [A.6 优化器、梯度处理与停止条件](#a6-优化器梯度处理与停止条件)
  - [A.7 TCN 超参数搜索](#a7-tcn-超参数搜索)
  - [A.8 RLS 在线能量反馈校正](#a8-rls-在线能量反馈校正)
  - [A.9 RLS 超参数搜索](#a9-rls-超参数搜索)
  - [A.10 RLS 能量窗变量分析](#a10-rls-能量窗变量分析)
  - [A.11 预测区间](#a11-预测区间)
  - [A.12 评价指标](#a12-评价指标)
  - [A.13 数据流和文件流](#a13-数据流和文件流)
- [B. 本次完整运行摘要](#b-本次完整运行摘要)
- [C. 当前可视化结果与逐图分析](#c-当前可视化结果与逐图分析)
- [D. 产物完整性与复现边界](#d-产物完整性与复现边界)

## A. 方法与数学建模

### A.1 问题背景与预测对象

实验以四轴无人机飞行遥测和工况参数为输入，预测每个采样点的电功率，并按真实采样间隔积分得到能量。对 flight $f$ 的第 $i$ 个采样点，监督标签为：

$$
P_{f,i}^{true}=\max(V_{f,i}I_{f,i}^{discharge},0)
$$

其中：

- $P_{f,i}^{true}$：真实放电功率，单位 W。
- $V_{f,i}$：电池电压，单位 V。
- $I_{f,i}^{discharge}$：非负放电电流，单位 A。
- $f$：完整飞行编号；$i$：flight 内按时间排序的采样点编号。

模型需要同时控制三个尺度的误差：逐采样点功率、反馈窗口能量和整次 flight 总能量。测试集只用于最终评估，TCN 与 RLS 的参数选择均只使用训练集和验证集。

### A.2 数据来源与无泄漏切分

程序读取 `D:/Python-files/Energy-prediction/data/dji_matrice_100/flights.csv` 和 `parameters.csv`，处理结果写入 [`data/processed_3.2/`](./data/processed_3.2/)。特征工程覆盖原始数据中的全部 11 种 route，不从项目内旧 `data/` 目录读取副本。

切分以完整 flight 为最小单位，并按 route 分层。对 flight 数不少于 3 的 route，验证集与测试集数量分别按比例向上取整，同时至少保留 1 个训练 flight；不足 3 个 flight 的 route 全部进入训练集，避免模型从未见过该 route。随机数种子固定为 42，同一 flight 不会跨集合，从而避免相邻采样点泄漏。

### A.3 特征与时间积分

特征包含电池状态、速度、加速度、角速度、风速风向、载荷、高度、航线独热编码，以及相对空速、横风分量、动态加速度、机动强度、热负载、视觉和通信能耗代理量。`dt_seconds` 由同一 flight 内相邻时间戳计算，即使原始表已接近 0.12 s 固定网格，也不把数组下标当作真实时间。

采样区间能量为：

$$
E_{f,i}=P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

窗口能量为：

$$
E_{f,k}=\sum_{i\in\mathcal I_{f,k}}P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

其中：

- $E_{f,i}$：单个采样区间能量，单位 Wh。
- $E_{f,k}$：flight $f$ 的第 $k$ 个时间窗口能量，单位 Wh。
- $\Delta t_{f,i}$：采样点覆盖的真实时间长度，单位 s。
- $\mathcal I_{f,k}$：属于窗口 $k$ 的采样点集合。
- 3600：把 W·s 换算为 Wh 的单位因子。

### A.4 时间感知 TCN

对当前采样点构造长度为 $T$ 的历史序列 $\mathbf X_{f,i-T+1:i}$。序列开头不足 $T$ 步时用该 flight 的首个样本左侧填充，不使用未来数据。`dt_seconds` 先通过两层门控网络：

$$
g_{f,j}=1+0.2\tanh\left(\mathbf W_2\operatorname{SiLU}(\mathbf W_1\Delta t_{f,j}+\mathbf b_1)+b_2\right)
$$

$$
\widetilde{\mathbf x}_{f,j}=g_{f,j}\mathbf x_{f,j}
$$

其中：

- $g_{f,j}$：时间间隔对应的正值门控系数，范围约为 $[0.8,1.2]$。
- $\mathbf x_{f,j}$、$\widetilde{\mathbf x}_{f,j}$：门控前后的标准化特征向量。
- $\mathbf W_1$、$\mathbf W_2$、$\mathbf b_1$、$b_2$：门控网络可学习参数。

门控序列进入 4 个 TCN 残差块，通道数为 `[64,64,64,32]`。每个块含两层核宽 3 的左侧填充因果卷积、GroupNorm、SiLU 和 Dropout，膨胀率依次为 1、2、4、8；输入输出通道不一致时用 $1\times1$ 卷积形成残差捷径。网络只取最后时刻的卷积表示，并与最后时刻输入经过 LayerNorm 和线性层得到的捷径表示相加，输出标准化功率。

按当前 4 块、每块 2 层卷积计算，卷积分支理论感受野为 61 个采样步。超过该感受野的窗口候选不会继续增加卷积分支可见的历史长度，长窗候选之间的差异还会受随机初始化影响；因此长时间窗排序只用于当前固定结构，不能外推为任意深度 TCN 的结论。

### A.5 联合功率和能量损失

逐点误差在标准化功率域使用 Huber 损失，阈值 $\delta=0.65$：

$$
h_\delta(e)=
\begin{cases}
\frac{1}{2}e^2,& |e|\le\delta\\
\delta(|e|-\frac{1}{2}\delta),& |e|>\delta
\end{cases}
$$

低于 100 W 的样本权重为 1.8，高于 650 W 的样本权重为 1.5，其余为 1.0。逐点损失为：

$$
L_{power}=\frac{1}{N}\sum_{n=1}^{N}w_nh_\delta(\hat y_n-y_n)
$$

完整秒窗口在原始功率域按 `dt_seconds` 积分。预测能量与真实能量之差除以 $P_{scale}/3600$ 后再计算 Smooth L1：

$$
L_{energy}=\operatorname{SmoothL1}_\delta\left(\frac{\hat E_{f,k}-E_{f,k}^{true}}{P_{scale}/3600},0\right)
$$

$$
L=L_{power}+0.2L_{energy}
$$

其中：

- $e$：标准化预测与标签之差。
- $w_n$：功率区间平衡权重。
- $N$：当前批次采样点数。
- $\hat E_{f,k}$、$E_{f,k}^{true}$：预测与真实秒级能量，单位 Wh。
- $P_{scale}$：训练集功率尺度，单位 W。
- $L$：用于反向传播的联合损失。

批采样器保证同一个完整秒窗口不被拆到两个批次。能量由预测功率张量通过 `scatter_add` 聚合，因而 $L_{energy}$ 的梯度会沿积分关系回传到窗口内每个功率输出；不完整末窗口不参与能量损失。

### A.6 优化器、梯度处理与停止条件

TCN 使用 AdamW，基础学习率 `3e-4`，权重衰减 `1e-4`，批量上限 2048。CUDA 环境启用 FP16 自动混合精度和 `GradScaler`；反向传播后先反缩放，再将全模型梯度范数裁剪到 5.0，随后更新参数。

`ReduceLROnPlateau` 监控验证联合损失，连续 3 轮未改善时把学习率乘 0.5。TCN 候选比较关闭早停，每个候选固定跑满 80 轮，以相同训练预算比较；正式训练最多 80 轮，若验证损失连续 10 轮未刷新最佳值则提前停止，并恢复验证损失最低轮次的权重。

### A.7 TCN 超参数搜索

本次搜索窗口为 0.25、0.5、1、2、4、6、8、10、12、16 s。秒数按训练集典型采样周期换算为步数，每组保持相同网络通道、核宽、Dropout、学习率、权重衰减和 Huber 阈值。候选选择分数为：

$$
S_{TCN}=WAPE_{second}+0.2WAPE_{sample}+0.5WAPE_{flight}
$$

其中：

- $S_{TCN}$：TCN 候选选择分数，越小越好。
- $WAPE_{second}$：验证集秒级能量 WAPE，单位 %。
- $WAPE_{sample}$：验证集逐采样点功率 WAPE，单位 %。
- $WAPE_{flight}$：验证集整 flight 能量 WAPE，单位 %。

选择过程不读取测试集。最优候选保存为外部阶段权重，随后 `train-fixed` 用同一窗口从头执行正式训练。

### A.8 RLS 在线能量反馈校正

RLS 使用 TCN 功率到真实功率的归一化仿射映射。对反馈窗口 $k$：

$$
\boldsymbol\phi_k=
\begin{bmatrix}
1\\
\bar P_k^{TCN}/P_{scale}
\end{bmatrix},\qquad
\hat P_{k}^{RLS}=P_{scale}\boldsymbol\phi_k^T\boldsymbol\theta_k
$$

窗口结束后才计算真实平均功率并更新：

$$
\mathbf K_k=\frac{\mathbf C_k\boldsymbol\phi_k}{\lambda+\boldsymbol\phi_k^T\mathbf C_k\boldsymbol\phi_k}
$$

$$
\boldsymbol\theta_{k+1}=\boldsymbol\theta_k+\mathbf K_k\left(\frac{\bar P_k^{true}}{P_{scale}}-\boldsymbol\phi_k^T\boldsymbol\theta_k\right)
$$

$$
\mathbf C_{k+1}=\frac{\mathbf C_k-\mathbf K_k\boldsymbol\phi_k^T\mathbf C_k}{\lambda}
$$

其中：

- $\bar P_k^{TCN}$、$\bar P_k^{true}$：TCN 和真实窗口能量除以窗口时长得到的平均功率，单位 W。
- $\boldsymbol\theta_k=[\theta_{0,k},\theta_{1,k}]^T$：窗口开始时的归一化偏置和缩放参数。
- $\mathbf C_k$：RLS 协方差矩阵。
- $\mathbf K_k$：RLS 增益。
- $\lambda$：遗忘因子。

每个 flight 开始时使用中性参数 $[0,1]$ 和初始协方差。程序按窗口内 TCN 平均功率是否达到 50 W 区分停机与飞行，状态切换时再次恢复中性参数。偏置裁剪到 $[-1,1]$，缩放裁剪到 $[0,2]$，校正功率下限为 0 W。

在线顺序是“使用旧参数预测当前窗口 -> 窗口结束后积分 -> 读取真实能量 -> 更新下一窗口参数”。只有持续时间达到目标窗长 95% 的窗口才更新；末尾不完整窗口仍输出预测，但不更新参数。该顺序保证当前窗口标签不会提前进入当前窗口预测。

### A.9 RLS 超参数搜索

RLS 固定正式 TCN 的验证集输出，只搜索 7 个遗忘因子 `[0.86,0.88,0.90,0.91,0.92,0.94,0.96]` 与 7 个初始协方差 `[0.01,0.025,0.05,0.1,0.25,0.5,1.0]` 的 49 组笛卡尔积。`warmup_windows` 固定为 0；参考能量窗固定为 1 s，不参与搜索。

每个 flight 的分数为：

$$
S_f=WAPE_{window,f}+0.2WAPE_{power,f}+0.5APE_{flight,f}
$$

总分加入尾部和跨 flight 波动惩罚：

$$
S_{RLS}=\frac{1}{F}\sum_{f=1}^{F}S_f+0.25Q_{0.95}(APE_{flight})+0.1\sigma(S_f)
$$

其中：

- $S_f$：单个 flight 的窗口能量、逐点功率和总能量综合分数。
- $F$：验证集 flight 数量。
- $Q_{0.95}(APE_{flight})$：flight 总能量绝对百分比误差的 95% 分位。
- $\sigma(S_f)$：不同 flight 综合分数的总体标准差。

该目标不会只优化总样本加权均值，还会惩罚少数 flight 的大能量误差。测试集不参与最优 RLS 参数选择。

### A.10 RLS 能量窗变量分析

RLS 能量窗不是超参数。程序先固定验证集选出的遗忘因子和初始协方差，再分别以 1、2、5、10、20 s 运行验证集和测试集。每个窗长独立计算逐点功率、反馈窗口能量和 flight 总能量的 MAE、RMSE、MAPE、$R^2$、WAPE，并统计完整窗口数、状态切换数、参数均值/标准差和裁剪边界触达次数。

这种设计把“RLS 记忆速度”和“真实能量反馈频率”分开：前者属于模型参数选择，后者用于分析反馈延迟与能量平滑之间的折中。测试集窗口结果只用于报告，不反向选择模型或参考窗。

### A.11 预测区间

验证集在线 RLS 与固定 RLS 的绝对功率残差分别保存。给定置信度 $c$ 和 $n$ 个校准残差，采用有限样本保序分位数：

$$
r_c=e_{(\min(\lceil(n+1)c\rceil,n))}
$$

$$
[\hat P_i^{lower},\hat P_i^{upper}]=[\max(\hat P_i-r_c,0),\hat P_i+r_c]
$$

其中：

- $e_{(j)}$：按升序排列后的第 $j$ 个绝对残差，单位 W。
- $r_c$：置信度 $c$ 对应的功率区间半径，单位 W。
- $\hat P_i^{lower}$、$\hat P_i^{upper}$：功率区间上下界，单位 W。

能量区间由功率上下界逐点积分并累加。这种累加是假定同一方向边界持续成立的保守构造，不等同于对 flight 总能量单独完成保序校准。

### A.12 评价指标

对真实值 $y_i$、预测值 $\hat y_i$ 和样本数 $N$：

$$
MAE=\frac{1}{N}\sum_i|\hat y_i-y_i|,\qquad
RMSE=\sqrt{\frac{1}{N}\sum_i(\hat y_i-y_i)^2}
$$

$$
MAPE=\frac{100}{N}\sum_i\frac{|\hat y_i-y_i|}{\max(|y_i|,\epsilon)},\qquad
WAPE=100\frac{\sum_i|\hat y_i-y_i|}{\max(\sum_i|y_i|,\epsilon)}
$$

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
$$

其中：

- MAE、RMSE：功率任务单位为 W，能量任务单位为 Wh。
- MAPE、WAPE：百分比指标，单位 %。
- $R^2$：决定系数，无量纲。
- $\epsilon=10^{-6}$：避免真实值为零时除零。

低功率或近零能量窗口的 MAPE 会被小分母放大，必须与 MAE、样本数和真实量级一起解释，不能只凭百分比判断模型失效。

### A.13 数据流和文件流

```text
原始 flights.csv + parameters.csv
-> data_utils.py 清洗、特征工程、dt_seconds、flight切分
-> out/data/processed_3.2/{train,val,test}.csv
-> train.py 搜索10个TCN窗口
-> 外部 best_energy_tcn_rls_3.2.pt
-> train.py 正式TCN训练（最多80轮）
-> 外部 final_energy_tcn_rls_3.2.pt
-> 固定TCN的49组RLS搜索
-> 固定RLS参数的5种能量窗变量分析
-> uncertainty.py 验证集残差校准
-> evaluate.py 测试指标与分层明细
-> predict.py 测试集/全航线预测
-> visualize.py 常规SVG
-> route_visualization.py 各route的SVG/HTML/GIF
-> out/outreadme.md 当前输出解释
```

## B. 本次完整运行摘要

本次运行时间为 2026-09-14 23:48:14 至 2026-09-15 05:48:48（总耗时 21633.22 s）。流程包含数据准备、10 个 TCN 候选、固定窗口正式训练、49 组 RLS 参数搜索、5 个能量窗变量对比、校准、评估、全航线预测和航线可视化。日志文件为 [`logs/terminal_3.2.log`](./logs/terminal_3.2.log)，本节数值均从最终 CSV/JSON 读取。

### B.1 数据和训练状态

- 清洗后保留 322904 条记录、209 个 flight、33 个输入特征；按完整 flight 切分为训练 221220 条/141 flight、验证 48841 条/34 flight、测试 52843 条/34 flight。
- TCN 搜索窗口为 0.25、0.5、1、2、4、6、8、10、12、16 s，对应 2、4、8、17、33、50、67、83、100、133 步；每个候选均执行 80 轮，搜索分数最低的是 2 s（7.708195）。
- 正式 2 s 模型在第 31 轮按验证损失连续 10 轮未改善触发早停，最佳轮为第 21 轮，最佳验证损失 0.0226905；这不改变候选阶段固定 80 轮的比较预算。
- RLS 搜索固定参考窗 1 s，49 组笛卡尔积全部完成；最优为 `rls_ff0.86_cov1`，选择分数 9604.822160。RLS 能量窗没有进入该搜索。
- 95% 功率区间半径为 91.91 W；测试采样覆盖率 94.83%，flight 能量区间覆盖率 97.06%。

### B.2 测试集主要指标

| 任务 | 方法 | MAE | RMSE | MAPE (%) | WAPE (%) | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 采样功率 | TCN | 29.468 W | 45.891 W | 247.721 | 7.752 | 0.9630 |
| 采样功率 | TCN+RLS | 28.260 W | 50.304 W | 71.712 | 7.434 | 0.9555 |
| flight 总能量 | TCN | 0.449 Wh | 0.591 Wh | 2.675 | 2.281 | 0.9938 |
| flight 总能量 | TCN+RLS | 0.126 Wh | 0.336 Wh | 4.970 | 0.639 | 0.9980 |

RLS 将采样功率 WAPE 降低 0.318 个百分点，将 flight 能量 WAPE 降低 1.642 个百分点；功率 RMSE 和 R² 略有下降，说明在线校正主要改善系统性偏差和累计能量，而不是保证每个瞬时尖峰都更平滑。MAPE 在接近零功率/能量样本上受小分母放大，不能脱离 MAE、WAPE 和真实量级单独解释。

### B.3 RLS 能量窗变量分析

下表来自 [`rls_energy_window_comparison_3.2.csv`](./model/rls_energy_window_comparison_3.2.csv)。遗忘因子 0.86、初始协方差 1.0 在所有行保持不变，能量窗仅作为独立变量；窗口越长，反馈次数减少，但每次反馈包含的能量误差绝对值增大。

| 集合 | 窗长 (s) | 完整窗数 | 功率 WAPE (%) | 窗口能量 MAE (Wh) | 窗口能量 WAPE (%) | flight 能量 MAE (Wh) | flight 能量 WAPE (%) | flight R² | 状态切换 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 验证 | 1 | 5844 | 7.251 | 0.00604 | 5.509 | 0.1481 | 0.782 | 0.9967 | 68 |
| 验证 | 2 | 2917 | 7.071 | 0.01029 | 4.711 | 0.1171 | 0.618 | 0.9991 | 68 |
| 验证 | 5 | 1157 | 7.100 | 0.02174 | 4.012 | 0.1415 | 0.747 | 0.9991 | 64 |
| 验证 | 10 | 571 | 7.231 | 0.03895 | 3.640 | 0.1685 | 0.889 | 0.9988 | 58 |
| 验证 | 20 | 275 | 7.281 | 0.06752 | 3.218 | 0.1840 | 0.971 | 0.9988 | 35 |
| 测试 | 1 | 6324 | 7.434 | 0.00595 | 5.652 | 0.1258 | 0.639 | 0.9980 | 60 |
| 测试 | 2 | 3156 | 7.260 | 0.00992 | 4.723 | 0.0793 | 0.403 | 0.9998 | 60 |
| 测试 | 5 | 1252 | 7.351 | 0.02051 | 3.936 | 0.0922 | 0.9997 | 0.9997 | 60 |
| 测试 | 10 | 618 | 7.458 | 0.03622 | 3.527 | 0.1094 | 0.9997 | 0.9997 | 58 |
| 测试 | 20 | 301 | 7.555 | 0.06434 | 3.219 | 0.1258 | 0.6387 | 0.9995 | 35 |

验证集的 flight 能量 WAPE 在 2 s 达到 0.618%，测试集在 2 s 达到 0.403%，因此 2 s 是本次数据上累计能量精度与反馈频率的折中点。5--20 s 的窗口能量相对 WAPE 继续下降，但单窗 MAE 和反馈延迟增大，且测试 flight WAPE 回升；该现象说明“窗口能量更平滑”不等于“单次飞行能量更准确”。

## C. 当前可视化结果与逐图分析

所有静态图均为 SVG，交互轨迹为 Plotly HTML，动态轨迹为 2240×980 像素 GIF（8 fps、最多 120 帧）。静态图的数据来自 `out/model/*.csv`、`out/predictions/test_predictions_3.2.csv`、`out/rls/*.csv`；航线图来自 `out/routes/route_*/*_aligned.csv`。坐标图均使用三条航线代表 flight 起点的共同参考点：X 为东向米、Y 为北向米、Z 为相对高度米。

### C.1 训练和搜索图

| 文件 | 标题/坐标 | 图例与数据含义 | 分析 |
| --- | --- | --- | --- |
| [`figures/training/loss_curve_3.2.svg`](./figures/training/loss_curve_3.2.svg) | UAV Energy TCN Training Curve；横轴 Epoch，纵轴 Huber Loss（无量纲） | train_loss 与 val_loss，来自 `training_log_3.2.csv` | 训练损失下降后趋稳；验证损失在第 21 轮最佳，第 31 轮早停，显示继续训练收益有限。 |
| [`figures/training/learning_rate_schedule.svg`](./figures/training/learning_rate_schedule.svg) | Learning Rate Schedule；Epoch 与 learning rate | 最终训练学习率，单位无量纲 | `ReduceLROnPlateau` 多次减半，后期学习率很低，与早停判断一致。 |
| [`figures/training/hyperparameter_ranking.svg`](./figures/training/hyperparameter_ranking.svg) | Hyperparameter Candidate Ranking；候选与选择分数 | 10 个 TCN 窗口，分数越低越好 | 2 s 分数最低；0.25 s 和长于 6 s 的候选明显较差，支持选择中等历史长度。 |
| [`figures/training/candidate_validation_wape.svg`](./figures/training/candidate_validation_wape.svg) | Validation WAPE by Candidate；WAPE (%) | 逐点功率和 flight 能量 WAPE | 2 s 在功率和能量两个尺度上同时保持较低误差，避免只优化单一指标。 |

### C.2 预测和评估图

| 文件 | 标题/坐标 | 图例与数据含义 | 分析 |
| --- | --- | --- | --- |
| [`figures/results/evaluation_metrics.svg`](./figures/results/evaluation_metrics.svg) | Evaluation Metrics；指标值 | TCN 与 TCN+RLS 的功率/能量 MAE、WAPE | 直观看出 RLS 对累计能量误差的改善，同时功率 RMSE 未必下降。 |
| [`figures/results/flight_energy_actual_vs_predicted.svg`](./figures/results/flight_energy_actual_vs_predicted.svg) | Flight-Level Energy: Actual vs Predicted；flight 与能量 (Wh) | Actual、Predicted | 大多数 flight 的 RLS 柱线贴近实际值；A1 等近零能量 flight 的相对误差需谨慎解释。 |
| [`figures/results/flight_energy_error.svg`](./figures/results/flight_energy_error.svg) | Flight-Level Energy Prediction Error；flight 与误差 (Wh) | 每个 flight 的 `energy_error_wh` | 误差围绕零分布，少数长 flight 贡献较大绝对误差。 |
| [`figures/results/power_bin_mae.svg`](./figures/results/power_bin_mae.svg) | Prediction MAE by Power Bin；功率区间与 MAE (W) | 按真实功率分箱 | 高功率段误差受尖峰和样本量影响，需结合 WAPE 及分箱计数读取。 |
| [`figures/prediction/power_prediction_scatter.svg`](./figures/prediction/power_prediction_scatter.svg) | 功率预测散点；真实功率与预测功率 (W) | TCN、RLS 与理想对角线 | 点云靠近对角线说明整体拟合有效，低功率区域的离散会抬高 MAPE。 |
| [`figures/prediction/power_residual_histogram.svg`](./figures/prediction/power_residual_histogram.svg) | 功率残差分布；残差 (W) 与频数 | TCN/RLS 残差 | 残差中心偏移表示系统性偏差，RLS 的作用是在线压低该偏移。 |
| [`figures/prediction/second_energy_prediction_scatter.svg`](./figures/prediction/second_energy_prediction_scatter.svg) | 秒级能量预测散点；能量 (Wh) | TCN、RLS 与理想线 | 秒级积分误差比瞬时功率更平滑，但近零窗口会产生相对百分比放大。 |
| [`figures/prediction/second_energy_residual_histogram.svg`](./figures/prediction/second_energy_residual_histogram.svg) | 秒级能量残差分布；残差 (Wh) | TCN/RLS | 反映积分窗口内误差抵消和剩余偏差。 |
| [`figures/prediction/rls_energy_window_prediction_scatter.svg`](./figures/prediction/rls_energy_window_prediction_scatter.svg) | RLS 能量窗散点；窗口能量 (Wh) | 五种窗长的预测与真实值 | 结合 B.3 表格判断反馈频率与能量误差的折中，不能只看散点密度。 |
| [`figures/prediction/rls_energy_window_residual_histogram.svg`](./figures/prediction/rls_energy_window_residual_histogram.svg) | RLS 能量窗残差；残差 (Wh) | 五种窗长 | 窗长增大后残差尺度变大，但相对 WAPE 可能下降，体现平滑效应。 |
| [`figures/results/rls_energy_window_error_comparison.svg`](./figures/results/rls_energy_window_error_comparison.svg) | RLS Energy Window Error Comparison；窗长与误差 | MAE、RMSE 等误差曲线 | 测试集 2 s 的 flight 能量 MAE 最低，是本次运行的折中点。 |
| [`figures/results/rls_energy_window_energy_metrics.svg`](./figures/results/rls_energy_window_energy_metrics.svg) | RLS Energy Window Energy Metrics；窗长与 WAPE/R² | 窗口和 flight 能量指标 | WAPE 与 R² 同时展示，避免用单一指标误判长窗优势。 |
| [`figures/results/rls_energy_window_count_comparison.svg`](./figures/results/rls_energy_window_count_comparison.svg) | RLS Energy Window Count Comparison；窗长与窗口数量 | 总窗口、完整窗口 | 1 s 反馈最频繁，20 s 反馈次数约降至其二十分之一，解释了响应延迟差异。 |

另有 [`figures/custom/custom_power_timeseries.svg`](./figures/custom/custom_power_timeseries.svg) 和 [`figures/custom/custom_cumulative_energy.svg`](./figures/custom/custom_cumulative_energy.svg)，分别展示自定义工况的时间-功率（W）和时间-累计能量（Wh）；数据由 `predict_custom_scenario` 生成，用于检查模型在非测试 flight 工况下的接口和趋势，不作为测试指标。

### C.3 代表 flight 功率、能量和轨迹图

flight 1、18、23 分别对应 R5、R1、R1 航线，保留用于与既有报告中的三条代表性航线对照；其功率/能量 SVG 位于 `figures/prediction/flight_1_*`、`flight_18_*`、`flight_23_*`。每张图横轴为时间 (s)，功率纵轴为 W，秒级能量图纵轴为 Wh；实际值、TCN 原始预测和 RLS 校正预测分别用图例标出。曲线的积分关系可用对应 `second_energy_evaluation_3.2.csv` 复核，不能把功率曲线的纵轴数值当作能量。

### C.4 RLS 参数轨迹

[`rls/flight_1_rls_parameter_trace.svg`](./rls/flight_1_rls_parameter_trace.svg)、[`rls/flight_2_rls_parameter_trace.svg`](./rls/flight_2_rls_parameter_trace.svg)、[`rls/flight_3_rls_parameter_trace.svg`](./rls/flight_3_rls_parameter_trace.svg) 的横轴为窗口序号，纵轴分别为无量纲偏置和缩放参数。每条线表示一个 flight 的在线参数，边界为偏置 [-1,1]、缩放 [0,2]；参数在停机/飞行状态切换时重置。它们用于检查 RLS 是否频繁触及裁剪边界，不是功率误差曲线。

### C.5 11 条航线的静态、交互和动图产物

总览图 [`routes/all_routes_trajectory_energy.svg`](./routes/all_routes_trajectory_energy.svg) 左侧为 11 条代表 flight 的统一局部三维轨迹（X/Y/Z 均为 m），右侧为实际累计能量与 RLS 累计能量（Wh）；颜色按航线区分，实线/虚线区分实际/RLS。每条航线目录均包含同一代表 flight 的功率能量 SVG、交互式 3D HTML 和高清 3D GIF，代表 flight 如下：

| 航线 | flight | 目录 | 风速均值 (m/s) | 风向均值 (°) | 实际能量 (Wh) | RLS 能量 (Wh) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A1 | 212 | [`routes/route_A1/flight_212/`](./routes/route_A1/flight_212/) | 0.139 | 163.8 | 0.000006 | 0.007370 |
| A2 | 216 | [`routes/route_A2/flight_216/`](./routes/route_A2/flight_216/) | 0.087 | 158.0 | 0.141569 | 0.127667 |
| A3 | 217 | [`routes/route_A3/flight_217/`](./routes/route_A3/flight_217/) | 0.303 | 118.8 | 1.261676 | 3.758692 |
| H | 222 | [`routes/route_H/flight_222/`](./routes/route_H/flight_222/) | 2.374 | 275.2 | 18.694820 | 18.888881 |
| R1 | 86 | [`routes/route_R1/flight_86/`](./routes/route_R1/flight_86/) | 5.697 | 177.3 | 19.904715 | 19.911337 |
| R2 | 5 | [`routes/route_R2/flight_5/`](./routes/route_R2/flight_5/) | 3.330 | 172.5 | 19.052936 | 19.594042 |
| R3 | 6 | [`routes/route_R3/flight_6/`](./routes/route_R3/flight_6/) | 4.027 | 193.0 | 20.116680 | 19.525708 |
| R4 | 7 | [`routes/route_R4/flight_7/`](./routes/route_R4/flight_7/) | 4.885 | 213.3 | 21.525185 | 21.534446 |
| R5 | 1 | [`routes/route_R5/flight_1/`](./routes/route_R5/flight_1/) | 3.897 | 133.7 | 21.762434 | 21.736414 |
| R6 | 270 | [`routes/route_R6/flight_270/`](./routes/route_R6/flight_270/) | 5.644 | 140.8 | 24.067533 | 24.004800 |
| R7 | 278 | [`routes/route_R7/flight_278/`](./routes/route_R7/flight_278/) | 4.688 | 191.8 | 19.021313 | 18.931740 |

每个 `*_power_energy.svg` 的上面板横轴为时间 (s)、纵轴为功率 (W)，图例为实际/TCN/RLS；下面板纵轴为累计能量 (Wh)。每个 `*_imu_trajectory.html` 可用鼠标旋转、缩放和悬停，轨迹颜色表示 IMU 路径，角标显示时间、无人机速度 (m/s)、风速 (m/s) 和风向 (°)。对应 GIF 左侧为功率时间曲线，右侧为 3D 轨迹；两侧均有图例，动画点表示当前时刻，灰线表示完整参考轨迹，蓝线表示已飞轨迹。A3 的 RLS 能量明显高于实际值，提示低能量、短航线对在线反馈较敏感；R1、R4、R5、R6 的累计能量较贴近实际，说明在持续飞行且功率较高的航线中校正更稳定。

## D. 产物完整性与复现边界

最终 `out/` 清单为 40 个 SVG、11 个交互 HTML、11 个 GIF、0 个 PNG；所有图片类产物均已改为 SVG，未发现旧 PNG 被 README 引用。每条航线目录同时包含功率能量 SVG、交互 HTML、3D GIF、对齐 CSV 和摘要 JSON；总览 SVG 位于 `routes/all_routes_trajectory_energy.svg`。模型权重为 `D:/Python-files/Energy-prediction/model/best_energy_tcn_rls_3.2.pt` 与 `final_energy_tcn_rls_3.2.pt`，项目内结构化结果和图表位于本目录相对路径。

README 中的相对链接已按最终文件名检查；运行摘要中的绝对路径仅来自程序写出的元数据，不作为跨机器复现路径。原始数据和处理后 CSV、模型权重、预测 CSV 等体积较大或包含实验数据，按项目 `.gitignore` 规则不纳入本次代码提交；克隆项目后需先配置原始数据路径，再运行 `python main.py prepare`。复现实验时，`tune-tcn`/`all` 会清空并重建终端日志，其他模式只追加日志；RLS 能量窗仍必须作为独立变量分析，不得改回 TCN/RLS 超参搜索。
