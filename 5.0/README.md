# 无人车真实道路能耗预测实验 5.0

## 目录

- [项目定位](#项目定位)
- [数据来源与内容](#数据来源与内容)
- [数据处理](#数据处理)
- [双层预测方法](#双层预测方法)
- [目录与结果文件](#目录与结果文件)
- [运行方法](#运行方法)
- [文档分工](#文档分工)

## 项目定位

5.0 使用 MOTIVES-LAB GECR 公开的纯电动汽车（BEV）真实道路行程，验证“TCN 一步功率预测 + RLS 历史能耗反馈校正”的双层框架。当前时刻使用车速、环境量、档位和当前已测得的电池电功率预测下一秒功率。RLS 在完整观测窗口结束后读取该窗口实测净能耗，校正后续窗口；因此 RLS 是在线反馈估计，不是出发前预测，也不是无反馈整段预测。

GECR 面向 BEV 能耗，不是自动驾驶专用数据，也不是无人机数据。它可以验证车辆域的时序能耗方法，不能直接支持自动驾驶路线规划或无人机任务能耗结论。

## 数据来源与内容

- 作者仓库：[MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev](https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev)，固定提交 `557a4de432c057b4d7d97a2800e4a48d414e132c`。
- 原始目录：`D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/Source/`。含 106 条真实道路行程 Excel（约 262.8 MB）、46 个实验室循环 Excel（约 60.4 MB），以及作者 README、LICENSE 和 MATLAB 参考脚本。本轮只用 106 条真实道路行程训练和测试，实验室循环尚未纳入模型。
- 清洗目录：`D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/5.0/processed/`。
- 数据筛选优劣、字段来源与限制、作者论文和许可说明见[GECR 数据适配说明](./GECR_DATA_GUIDE.md)，来源副本元数据见 `Source/DATA_SOURCE.md`。
- 主要数据论文：Yuan et al., “Data-driven evaluation of electric vehicle energy consumption for generalizing standard testing to real-world driving,” *Patterns* 5(4), 100950 (2024), [DOI: 10.1016/j.patter.2024.100950](https://doi.org/10.1016/j.patter.2024.100950)。作者仓库和论文均确认该研究使用 106 条道路行程。论文勘误为 [DOI: 10.1016/j.patter.2025.101173](https://doi.org/10.1016/j.patter.2025.101173)。

`Continuous20Hz` 工作表提供 `PhaseTime`、实测车速、REESS 电流/电压、环境温度、海拔、档位和累计放电/回充计数器。作者 MATLAB 代码注明速度单位为 km/h，并以 `-sum(voltage .* current, 2)` 计算电池功率。当前道路数据第二组电压/电流通道恒为 0。累计计数器的 Excel 表头没有单位，代码不把它直接当逐秒监督标签。

## 数据处理

`vehicle_pipeline.py` 对每条 Excel 排序去重、剔除关键值无效采样，以 `PhaseTime` 聚合为 1 秒记录。功率监督标签按 $P_t=-V_tI_t$ 重建，放电为正、再生回馈为负；单步能量为 $E_t=P_t\Delta t/3600$ Wh，$Δt=1$ s。缺失速度表头 `DAScheduleSpeed` 在道路集恒为 0，故不作为模型输入。

输入包含车速、速度变化、近 5 秒速度均值/标准差、温度、海拔变化率、档位、当前行程相对时间、电池电压/电流、当前已观测电功率及近 5 秒功率均值/标准差，共 14 维。模型输入截至时刻 $t$，标签为 $t+1$ 秒功率；不使用整段行程终点派生的未来进度。输入与目标标准化参数只由训练行程拟合。

以完整 trip 隔离训练、验证、测试，随机种子固定为 42；69/16/21 条行程分别属于训练/验证/测试，不允许同一行程跨集合。加工表保留来源行数、行程编号、切分、时间、速度、电量和派生特征，以便审计。

## 双层预测方法

**第一层：TCN。** 两层一维卷积、ReLU、dropout 与线性输出头，卷积仅使用当前窗口内历史观测。用标准化功率上的 SmoothL1 损失训练，AdamW 更新，梯度范数裁剪为 5；根据验证损失保存最优 epoch。搜索先比较 3 个窗口 × 4 组网络/优化参数、每组 2 个随机种子，再对前三配置以 3 个种子复核；按验证功率、5 秒窗口净能量和行程净能量 WAPE 的加权分数选择。

$$S=0.2W_{P}+W_{5s}+0.5W_{trip}$$

其中：

$S$：TCN 验证配置的综合选择分数，越低越好。

$W_P$：逐秒电池功率加权绝对百分比误差（WAPE）。

$W_{5s}$：5 秒窗口净能量 WAPE。

$W_{trip}$：完整测试行程净能量 WAPE。

**第二层：RLS。** 将冻结 TCN 功率积分成窗口能量，用偏置和比例系数校正窗口预测。每个行程初始化参数为 $\theta_0=0,\theta_1=1$、协方差矩阵为 $P_0=\alpha I$。当前窗口先输出预测，待窗口实测能量可用后再递推更新：

$$\hat E_k=\phi_k^T\theta_{k-1},\quad K_k=\frac{P_{k-1}\phi_k}{\lambda+\phi_k^TP_{k-1}\phi_k},\quad \theta_k=\theta_{k-1}+K_k(E_k-\hat E_k),\quad P_k=\frac{P_{k-1}-K_k\phi_k^TP_{k-1}}{\lambda}$$

其中：

$E_k$：第 $k$ 个完整观测窗口的实测净能量（Wh）。

$\hat E_k$：在获得窗口 $k$ 的实测能量之前输出的校正预测（Wh）。

$\phi_k=[1,E_{\mathrm{TCN},k}]^T$：偏置及 TCN 窗口能量输入。

$K_k$：RLS 增益向量。

$P_k$：参数估计协方差矩阵。

$\lambda$：遗忘因子，控制旧窗口信息衰减。

$\alpha$：初始协方差倍率。

RLS 的窗口长度、遗忘因子和初始协方差在验证集 18 组配置中，以整程能量 WAPE 选择。测试逐行程重置 RLS，不把一个 trip 的已观测能量传给另一 trip。

## 目录与结果文件

| 路径 | 内容 |
| --- | --- |
| [`config.py`](./config.py) | 原始/加工数据、模型、输出路径及搜索参数 |
| [`main.py`](./main.py) | `prepare`、`search`、`train`、`evaluate`、`all` 入口 |
| [`vehicle_pipeline.py`](./vehicle_pipeline.py) | 清洗、窗口构造、训练、搜索、RLS、评估和图表 |
| `data/MOTIVES-LAB-GECR/5.0/processed/` | 全量清洗表、train/val/test、数据摘要 |
| `model/vehicle_tcn_5.0.pt` | 三种子模型权重和模型元数据 |
| [`out/outreadme.md`](./out/outreadme.md) | 本次算法、实际结果、逐图解读和限制 |
| `out/predictions/` | 测试集逐秒预测及 RLS 窗口预测 |
| `out/figures/` | 搜索、收敛、误差、行程能耗图 |
| `out/` 中 CSV/JSON | 搜索试验、训练日志、指标和逐行程统计 |

## 运行方法

从仓库根目录执行，Python 环境需具备 `requirements.txt` 中依赖：

```powershell
python 5.0/main.py prepare --force-prepare
python 5.0/main.py search
python 5.0/main.py train
python 5.0/main.py predict
python 5.0/main.py evaluate
python 5.0/main.py all --force-prepare
python -m unittest discover -s 5.0/tests -v
```

`search` 只使用训练和验证数据，并更新最优 TCN/RLS 配置；`train` 按最优配置以三个固定种子训练并集成；`evaluate` 读取冻结的测试预测并生成指标和图表。默认使用 CUDA，未检测到 CUDA 时回退 CPU。源数据和加工数据路径集中在 `config.py`。

## 文档分工

- `README.md` 记录稳定的数据定义、模型接口、目录和运行方式。
- `improve.md` 追加工程变化、纠错和验证结果。
- `out/outreadme.md` 记录当前模型、搜索、测试结果及 `out/` 下全部产物；重跑、改图或改变结果文件后同步更新。
