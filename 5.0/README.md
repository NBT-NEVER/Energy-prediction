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

5.0 使用 MOTIVES-LAB GECR 公开的纯电动汽车（BEV）真实道路行程，评估“TCN 逐时刻功率曲线 + RLS 历史能耗反馈校正”的双层框架。调度器 API 接收任务开始前可用的逐秒计划轨迹，返回逐秒功率、区间能量和累计能量。RLS 在完整观测窗口结束后读取该窗口实测净能耗，校正后续窗口；因此 RLS 是在线反馈估计，不是出发前预测，也不是无反馈整段预测。

GECR 面向 BEV 能耗，不是自动驾驶专用数据，也不是无人机数据。它可以验证车辆域的时序能耗方法，不能直接支持自动驾驶路线规划或无人机任务能耗结论。

## 数据来源与内容

- 作者仓库：[MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev](https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev)，固定提交 `557a4de432c057b4d7d97a2800e4a48d414e132c`。
- 原始目录：`D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/Source/`。含 106 条真实道路行程 Excel（约 262.8 MB）、46 个实验室循环 Excel（约 60.4 MB），以及作者 README、LICENSE 和 MATLAB 参考脚本。本轮只用 106 条真实道路行程训练和测试，实验室循环尚未纳入模型。
- 清洗目录：`D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/5.0/processed/`。
- 数据筛选优劣、字段来源与限制、作者论文和许可说明见[GECR 数据适配说明](./GECR_DATA_GUIDE.md)，来源副本元数据见 `Source/DATA_SOURCE.md`。
- 主要数据论文：Yuan et al., “Data-driven evaluation of electric vehicle energy consumption for generalizing standard testing to real-world driving,” *Patterns* 5(4), 100950 (2024), [DOI: 10.1016/j.patter.2024.100950](https://doi.org/10.1016/j.patter.2024.100950)。作者仓库和论文均确认该研究使用 106 条道路行程。论文勘误为 [DOI: 10.1016/j.patter.2025.101173](https://doi.org/10.1016/j.patter.2025.101173)。

`Continuous20Hz` 工作表提供 `PhaseTime`、实测车速、REESS 电流/电压、环境温度、海拔、GPS、档位和累计放电/回充计数器。作者 MATLAB 代码注明速度单位为 km/h，并以 `-sum(voltage .* current, 2)` 计算电池功率。当前道路数据第二组电压/电流通道恒为 0。累计计数器的 Excel 表头没有单位，代码不把它直接当逐秒监督标签。GECR 不提供真实调度器计划轨迹或载荷质量；本轮 `planned_*` 是从已执行行程派生的计划代理，测试属于按完整行程隔离的离线回放，不代表出发前泛化或载荷敏感性能。

## 数据处理

`vehicle_pipeline.py` 对每条 Excel 排序去重、剔除关键值无效采样，以 `PhaseTime` 聚合为 1 秒记录。功率监督标签按 $P_t=-V_tI_t$ 重建，放电为正、再生回馈为负；单步能量为 $E_t=P_t\Delta t/3600$ Wh，$Δt=1$ s。缺失速度表头 `DAScheduleSpeed` 在道路集恒为 0，故不作为模型输入。

调度器契约遵循 4.1 的任务前信号筛选原则，不把十维字段误当成模型必须维数。完整兼容格式由 `task_id`、`payload_kg`（仅审计，不用于模型）及十个逐秒计划特征组成；当前模型实际消费三维 `planned_speed_kmh`、`planned_acceleration_kmhps`、`planned_grade_percent`。最小输入可省略未被 checkpoint 使用的路线、环境和停车字段，接口会将缺失加速度按速度差分、缺失坡度按海拔/距离或零值补齐。监督目标为逐秒 `power_w`，区间能量由功率积分得到。训练时将 GECR 已执行行程的速度、距离、海拔、GPS 航向和环境温度构造为计划轨迹代理；输入不包含电压、电流、档位或监督功率等运行期能耗状态。因为计划代理取自已执行行程，本轮只能说明离线轨迹回放能力，不是独立真实调度器计划的出发前泛化。

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `time_s` | s | 任务相对时间，固定 1 s 网格 |
| `task_duration_s` | s | 完整任务计划时长 |
| `planned_speed_kmh` | km/h | 计划速度 |
| `planned_acceleration_kmhps` | (km/h)/s | 计划速度差分 |
| `planned_distance_m` | m | 计划累计距离 |
| `planned_elevation_m` | m | 相对起点的计划海拔 |
| `planned_grade_percent` | % | 由计划海拔和距离变化估计的坡度 |
| `planned_heading_deg` | degree | 计划航向角 |
| `ambient_temperature_c` | 摄氏度 | 任务环境温度 |
| `planned_stop_flag` | 0/1 | 计划停车标志 |

以完整 trip 隔离训练、验证、校准、测试，随机种子固定为 42；当前 106 条行程按约 50%/10%/20%/20% 分配，实际行程数见 `out/model/split_statistics_5.0.csv`。校准行程不参与权重训练或超参选择；测试行程只用于最终一次性评估。训练和验证 DataLoader 均使用 2048 批次，训练窗口不抽稀。加工表保留来源行数、行程编号、切分、时间、速度、电量和派生特征，以便审计。

## 双层预测方法

**第一层：TCN。** 三个双卷积因果残差块，膨胀系数为 1、2、4，采用只在左侧补零的卷积；卷积核长为 3 时理论感受野为 29 个 1 Hz 点，核长为 5 时为 57 点。用标准化功率上的 SmoothL1 损失训练，AdamW 更新，梯度范数裁剪为 5；根据验证损失保存最优 epoch。部署接口要求完整、严格递增、每秒采样的任务轨迹。GECR 无载荷训练字段，`payload_kg` 不影响模型预测。

$$S=0.2W_{P}+W_{5s}+0.5W_{trip}$$

其中：

$S$：TCN 验证配置的综合选择分数，越低越好。

$W_P$：逐秒电池功率加权绝对百分比误差（WAPE）。

$W_{5s}$：5 秒窗口净能量 WAPE。

$W_{trip}$：完整测试行程净能量 WAPE。

模型权重由逐秒功率上的 SmoothL1 损失更新。上式只在训练完成后比较验证集候选配置，能耗误差影响超参数选择，不会直接产生能耗梯度；若要直接以能耗调整权重，需将窗口能量误差纳入训练损失并重新训练。

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

RLS 每完成一个 5 秒窗口，先接收该窗口实测净能量并更新参数，再重算整段最终能耗：

$$\hat E_{\mathrm{final},k}=E_{\mathrm{observed},1:k}+\sum_{j=k+1}^{K}\hat E_{\mathrm{RLS},j\mid k}$$

四类预测区间分别校准和评估：TCN逐秒能耗区间由按预测功率幅值分档的90%逐秒功率残差换算；TCN整段能耗区间直接校准每条行程的最终总能耗误差；RLS逐秒能耗区间由在线回放的逐秒等效残差校准；RLS整段区间覆盖“已观测真实能耗 + 尚未完成窗口预测”，半径结合剩余窗口预测误差方差和RLS参数协方差，并用独立校准行程在线回放的95%标准化终值误差校准。最后一个窗口反馈后整段能耗已知，区间归零；在线覆盖率排除该确定性终点。区间是预测区间，不是种子差异或均值参数置信区间；有限校准行程下，经验覆盖率可能偏离名义覆盖率，预测带不会被强制单调收窄。

## 目录与结果文件

| 路径 | 内容 |
| --- | --- |
| [`config.py`](./config.py) | 原始/加工数据、模型、输出路径及搜索参数 |
| [`main.py`](./main.py) | `prepare`、`search`、`train`、`predict`、`predict-task`、`evaluate`、`all` 入口 |
| [`task_api.py`](./task_api.py) | 校验调度器输入并生成任务逐秒功率和能耗曲线 |
| [`terminal_logger.py`](./terminal_logger.py) | 将终端输出、阶段摘要和异常同步记录到 UTF-8 文本日志 |
| [`vehicle_pipeline.py`](./vehicle_pipeline.py) | 清洗、窗口构造、训练、搜索、RLS、评估和图表 |
| `data/MOTIVES-LAB-GECR/5.0/processed/` | 全量清洗表、train/val/test、数据摘要 |
| `model/vehicle_tcn_5.0.pt` | 三种子模型权重和模型元数据 |
| [`out/outreadme.md`](./out/outreadme.md) | 本次算法、实际结果、逐图解读和限制 |
| `out/predictions/` | 测试/校准集逐秒预测及调度器预测输出 |
| `out/model/` | 搜索、消融、训练、数据划分统计、评估指标及校准参数 |
| `out/figures/{training,prediction,results}/` | 训练搜索图、预测结果图和数据分析图 |
| `out/rls/` | RLS 搜索、逐窗口轨迹、在线区间图和 GIF |
| `out/routes/` | 多路线展示及测试路线统计 |
| `out/tasks/` | 调度器任务预测结果 |
| `out/rls/rls_final_energy_updates_5.0.gif` | RLS 每次窗口反馈后的整段最终能耗估计及校准区间 |
| `out/rls/rls_interval_width_updates_5.0.gif` | 同一在线回放中的区间全宽变化 |
| `out/rls/test_rls_trajectory_5.0.csv` | 测试行程逐窗口观测能耗、剩余预测、终值估计和区间轨迹 |
| `out/model/power_interval_calibration_5.0.json` | 独立行程校准的功率区间分位数，供任务接口复用 |
| `out/logs/terminal_5.0.log` | 完整终端阶段记录；`all/search` 清空后重记，单项模式追加 |

## 运行方法

从仓库根目录执行，Python 环境需具备 `requirements.txt` 中依赖：

```powershell
python 5.0/main.py prepare --force-prepare
python 5.0/main.py search
python 5.0/main.py train
python 5.0/main.py predict
python 5.0/main.py predict-task --task-input 5.0/examples/scheduled_task_template.csv --task-output 5.0/out/tasks/scheduled_task_predictions_5.0.csv
python 5.0/main.py evaluate
python 5.0/main.py all --force-prepare
Set-Location 5.0; python -m unittest discover -s tests -v
```

`search` 只使用训练和验证数据，并更新最优 TCN/RLS 配置；`train` 按最优配置以三个固定种子训练并集成；`evaluate` 读取冻结的测试预测并生成指标和图表。默认使用 CUDA，未检测到 CUDA 时回退 CPU。源数据和加工数据路径集中在 `config.py`。

`predict-task` 的 CSV 至少要求 `task_id`、`payload_kg`、`time_s`、`task_duration_s` 和 `planned_speed_kmh`，也接受上表十个计划字段的完整格式；不得包含实测功率、电压、电流、实际速度等结果字段，一次只预测一个任务。输出包括 `predicted_power_w`、`predicted_energy_wh`、`cumulative_energy_wh`、90%逐秒边际功率区间，以及按校准行程进度匹配的90%累计能耗区间和运行平均能耗率区间，摘要还提供 `predicted_total_energy_wh`。累计能耗区间根据独立校准行程的累计误差生成，不把逐秒功率上下界当作持续同向误差累加；其适用性仍受计划代理数据域限制。

## 文档分工

- `README.md` 记录稳定的数据定义、模型接口、目录和运行方式。
- `improve.md` 追加工程变化、纠错和验证结果。
- `out/outreadme.md` 记录当前模型、搜索、测试结果及 `out/` 下全部产物；重跑、改图或改变结果文件后同步更新。
