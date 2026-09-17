# 无人机任务规划能耗预测 4.0

## 目录

- [1. 版本定位与调度背景](#1-版本定位与调度背景)
- [2. 项目结构](#2-项目结构)
- [3. 环境与依赖](#3-环境与依赖)
- [4. 数据、模型与产物位置](#4-数据模型与产物位置)
- [5. 数据策略与离散化](#5-数据策略与离散化)
- [6. 任务前输入契约](#6-任务前输入契约)
- [7. 23 维 TCN 输入](#7-23-维-tcn-输入)
- [8. 坐标、风向与历史数据映射](#8-坐标风向与历史数据映射)
- [9. 任务前与飞行中输出契约](#9-任务前与飞行中输出契约)
- [10. 运行方式](#10-运行方式)
- [11. 命令行模式](#11-命令行模式)
- [12. 算法概览](#12-算法概览)
- [13. 训练与超参数约束](#13-训练与超参数约束)
- [14. 产物管理](#14-产物管理)
- [15. 文件职责与调用关系](#15-文件职责与调用关系)
- [16. 日志规则](#16-日志规则)
- [17. 文档职责](#17-文档职责)
- [18. 当前边界与复现约束](#18-当前边界与复现约束)

## 1. 版本定位与调度背景

4.0 面向无人机任务分配、路线比较和充电决策，不再只对飞行结束后的功率记录做离线拟合。完整业务链分为三层：上层调度器确定无人机、起终点、任务载荷、时间要求和候选路径；中层轨迹规划器把路径离散为每个时刻的位置、三轴速度、三轴加速度和高度；4.0 接收整条规划轨迹与风场、载荷信息，在起飞前估计能耗，并在飞行中用已经结束窗口的实际能耗修正剩余路线。

系统支持两个阶段：

- **任务执行前**：TCN 只读取规划时已经确定或可计算的数据，输出每个 0.2 s 时间步的基础动力功率、附加设备功率、总功率、分步能耗、累计能耗、整条任务能耗区间、安全储备和电池可行性。
- **任务执行中**：保留任务前生成的整条路线基线。每个 1 s RLS 窗口结束后，外部电池计量模块才提供该窗口的实际能耗；程序用同一窗口的“任务前预测能耗—实际能耗”更新 RLS，并重新计算其后的全部剩余时间步和任务级区间。已经执行的部分累计为实际能耗，不被后续预测覆盖。

4.0 与 3.2 相互独立。它不读取 3.2 的权重、scaler、特征元数据、预测 CSV、置信区间或日志，也不修改 3.2 的历史产物。外部原始公开数据可以作为共同只读数据源，但 4.0 的处理后数据、模型和输出都使用自己的版本化路径。

本文件是 4.0 的稳定项目入口。当前运行的指标、完整算法说明和逐产物分析见 [`out/outreadme.md`](./out/outreadme.md)，工程演进与历史纠正见 [`improve.md`](./improve.md)。

## 2. 项目结构

```text
4.0/
├── config.py            # 路径、0.2 s步长、训练参数和版本化文件名
├── data_utils.py        # 原始数据检查、重采样、23维特征、flight切分和元数据
├── model.py             # 因果TCN与窗口能耗DynamicEnergyRLS
├── train.py             # scaler、8窗TCN搜索、正式训练、区间模型和49组RLS搜索
├── evaluate.py          # 样本/窗口/flight评估、在线模拟和RLS修正轨迹
├── task_api.py          # 任务前批量接口与飞行中OnlineTaskSession
├── visualize.py         # 训练、评估、预测、RLS和分路线产物
├── terminal_logger.py   # UTF-8终端日志的清空、追加和异常记录
├── main.py              # 统一命令行入口与完整流程调度
├── requirements.txt     # Python依赖
├── README.md            # 当前稳定接口和复现入口
├── improve.md           # 追加式工程改进日志
├── model/               # 4.0独立权重与scaler
└── out/                 # 当前正式运行的全部结构化产物和图表
```

4.0 不使用 `out/runs/`。每次正式运行直接更新 `out/data`、`out/model`、`out/predictions`、`out/tasks`、`out/rls`、`out/figures`、`out/routes` 和 `out/logs` 中的固定版本化文件，避免同一结果同时存在多套嵌套路径。

## 3. 环境与依赖

安装依赖：

```bash
python -m pip install -r 4.0/requirements.txt
```

当前依赖包括 `numpy`、`pandas`、`matplotlib`、`torch`、`pillow` 和 `plotly`。训练和模型推理默认并强制使用 CUDA；`train.py` 会验证 `torch.cuda.is_available()` 和指定设备，CUDA 不可用或传入 `--device cpu` 时明确报错，不静默降级到 CPU。数据准备 `prepare` 本身不执行模型推理，可单独完成。

推荐从项目根目录使用 `--device cuda` 或 `--device cuda:0`。显卡型号、CUDA 版本、实际耗时和本轮训练状态属于运行结果，应从终端日志和 [`out/outreadme.md`](./out/outreadme.md) 读取，不在稳定 README 中固定填写。

中文图表使用程序注册的中文字体，标题、坐标轴、图例和注释均保留中文；静态图以 SVG 保存，动态过程以 GIF 保存，交互轨迹以 HTML 保存。图表生成后仍应检查字体是否实际可用，不能只依赖终端无报错判断中文显示正常。

## 4. 数据、模型与产物位置

所有默认路径集中在 `config.py`，可通过命令行覆盖原始数据根目录、模型目录和输出目录。

| 内容 | 默认位置 | 用途 |
|---|---|---|
| 原始数据根目录 | `D:/Python-files/Energy-prediction/data` | 项目外长期保存的只读数据源 |
| 原始飞行记录 | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.0/raw/flights.csv` | 历史轨迹、环境记录和离线监督标签来源 |
| 处理后数据 | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.0/processed/` | 0.2 s 特征表、train/val/test、元数据和重采样映射 |
| 最优阶段权重 | `4.0/model/best_planning_tcn_4.0.pt` | 当前最优窗口的阶段 checkpoint |
| 最终权重 | `4.0/model/final_planning_tcn_4.0.pt` | 任务前预测和离线评估使用的正式权重 |
| 标准化参数 | `4.0/model/planning_scaler_4.0.json` | 只用训练集估计的 23 维输入和基础功率缩放参数 |
| 当前运行产物 | `4.0/out/` | 数据、评估、预测、接口示例、RLS、图表、路线产物和日志 |

执行 `prepare --force-prepare` 会按当前原始数据重新生成 4.0 的处理后数据。具体行数、flight 数、路线集合和切分数量以 `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.0/processed/dataset_summary_4.0.json` 为准；逐维来源、单位、可获得时机和坐标约定以同目录的 `feature_metadata_4.0.json` 为准。

`--data-dir` 应指向当前版本的数据目录，其下必须包含 `raw/flights.csv`，处理后数据固定写入同目录的 `processed/`；`--save-dir` 只改变 4.0 权重与 scaler 位置；`--out-dir` 只重新派生日志、评估、预测、图表和任务产物目录。不得把这些参数指向 3.2 的版本目录。

## 5. 数据策略与离散化

### 5.1 训练与测试边界

- A1、A2、A3 是地面辅助测试，在读取原始数据时排除，不进入训练集、验证集、正常飞行指标或 TCN/RLS 选择。
- H 悬停数据保留，用于统一模型学习低水平位移和悬停状态；起飞、爬升、巡航、转弯和降落不拆成多个模型。
- 数据按完整 `flight` 划分，同一 flight 不跨训练、验证和测试集合。
- R2、R3、R4、R7 的正常 flight 当前整体留作未见路线测试；其他路线在 flight 级划分后用于训练、验证和测试。
- 验证集用于 TCN 时间窗、RLS 参数和残差区间模型拟合；测试集只用于最终评估。
- `route` 保留在处理表、预测结果和统计中，用于来源追踪、分路线查询和异常定位，但不进入 `FEATURE_COLUMNS`，也不生成 route 独热编码。

### 5.2 0.2 s 时间网格

原始记录不是固定 0.2 s 采样。程序按每个 flight 的真实 `time` 排序、去重，再插值到固定 0.2 s 网格；不能把数组下标或“5 帧”直接当作 0.2 s。当前原始采样周期统计及其范围写入 `feature_metadata_4.0.json`，每个新时间步关联的原始左右记录索引和时间范围写入 `resample_map_4.0.csv`。

每个离散步的能量按固定时间积分：

$$
E_t=P_t\frac{\Delta t}{3600},\qquad \Delta t=0.2\ \mathrm{s}
$$

其中：

- $E_t$：第 $t$ 个离散步的能量，单位 Wh。
- $P_t$：第 $t$ 个离散步的功率，单位 W。
- $\Delta t$：统一重采样步长，固定为 0.2 s。
- $t$：任务离散时间步索引。

电压和电流仅在离线数据准备时生成 `power_w` 标签；在线阶段的实际窗口能耗应由外部计量模块在窗口结束后给出，RLS 不读取电压、电流原始序列。

## 6. 任务前输入契约

`task_api.predict_task_before` 一次预测一个完整 `task_id`；`task_api.predict_candidate_tasks` 接收多个 `task_id`，供调度器批量比较候选任务。任务表必须按 0.2 s 提供完整轨迹，字段如下。

### 6.1 必填字段

| 字段 | 单位 | 说明 |
|---|---:|---|
| `task_id` | 字符串 | 上层调度器分配的任务标识；不进入 TCN |
| `time_s` | s | 任务时间轴；每个任务内必须严格递增，接口会转为相对时间 |
| `dt_seconds` | s | 每行固定为 0.2 |
| `planned_position_x_m` | m | 任务起点局部 ENU 坐标系中的东向位置 |
| `planned_position_y_m` | m | 任务起点局部 ENU 坐标系中的北向位置 |
| `planned_altitude_m` | m | 规划高度轨迹 |
| `planned_vx_mps`、`planned_vy_mps`、`planned_vz_mps` | m/s | 中层规划器给出的机体系 X/Y/Z 三轴速度；实验中直接采用原始飞行数据的对应三轴轨迹 |
| `planned_ax_mps2`、`planned_ay_mps2`、`planned_az_mps2` | m/s² | 中层规划器给出的机体系 X/Y/Z 线加速度；Z 轴值不含重力项，实验中由原始 Z 加速度加 $g$ 得到 |
| `planned_heading_deg` | ° | 机体系 X 轴相对规划坐标的航向，只用于把全局风向转换到机体系，不进入 TCN |
| `wind_speed_mps` | m/s | 对应任务时空位置的风速预报，必须非负 |
| `wind_direction_deg` | ° | 全局规划坐标中的风矢量指向角；0° 向北，90° 向东，气象学来向由接口转换 |
| `payload_kg` | kg | 任务有效载荷质量，必须非负 |
| `wind_forecast_source` | 字符串 | 风预报来源标识 |
| `wind_forecast_timestamp` | ISO-8601 | 风预报发布时间或有效时刻 |
| `wind_speed_error_mps`、`wind_direction_error_deg` | m/s、° | 风速和风向预报误差范围 |
| `wind_coordinate_system`、`wind_direction_convention` | 枚举 | 风向坐标基准和角度定义（`VECTOR_TO` 或 `METEOROLOGICAL_FROM`） |

### 6.2 可选字段

| 字段 | 单位 | 说明 |
|---|---:|---|
| `route` | 字符串 | 路线来源和结果查询标签；缺省时使用 `PLANNER_ROUTE`，不进入 TCN |
| `auxiliary_power_w` | W | 已知相机、通信、计算或投递设备额定功率之和；默认 0，不进入 TCN，在输出端确定性相加 |

公开原始数据没有可确认的相机、通信和计算设备额定功率字段，因此历史训练记录中的 `auxiliary_power_w` 固定为 0。部署时只能填写设备规格书或标定试验给出的确定功率，不能把旧版 `thermal_load_proxy`、`vision_energy_proxy_w`、`communication_energy_proxy_w` 当作瓦特值代入。

### 6.3 禁止字段

任务前接口拒绝电池电压、电池电流、真实功率、真实能耗、实飞速度、实飞加速度、实飞角速度和三个旧代理字段。任务结束后才能得到的最终能耗或实飞最终进度也不得进入任务前表。`power_w`、`base_power_w`、`energy_interval_wh` 等标签字段只存在于离线数据和评估结果中。

## 7. 23 维 TCN 输入

任务接口先校验必填字段，再派生完整的 23 维 `FEATURE_COLUMNS`。速度模长、水平速度、加速度模长和固定步长下的单步距离都可以从三轴分量确定，因此不作为独立输入重复训练；`planned_heading_deg` 和风预报元数据只参与风场坐标统一；`segment_distance_m` 保留在路线输出和特征分析表中，但不进入 TCN。

| 序号 | TCN 字段 | 单位 | 具体含义与部署来源 |
|---:|---|---:|---|
| 1 | `time_s` | s | 任务开始后的相对时间；由规划时间轴减去首时刻得到 |
| 2 | `dt_seconds` | s | 当前离散步时长，固定为 0.2 |
| 3 | `task_progress` | 1 | 当前相对时间占完整任务计划时长的比例 |
| 4 | `task_duration_s` | s | 完整候选路线的计划总时长 |
| 5 | `planned_position_x_m` | m | 以任务起点为原点的 ENU 东向局部位置 |
| 6 | `planned_position_y_m` | m | 以任务起点为原点的 ENU 北向局部位置 |
| 7 | `planned_altitude_m` | m | 当前规划高度 |
| 8 | `planned_vx_mps` | m/s | 机体系 X 轴（机头方向）规划速度分量；实验中直接使用原始 `velocity_x` |
| 9 | `planned_vy_mps` | m/s | 机体系 Y 轴规划速度分量；实验中直接使用原始 `velocity_y` |
| 10 | `planned_vz_mps` | m/s | 机体系 Z 轴规划速度分量；实验中直接使用原始 `velocity_z` |
| 11 | `planned_ax_mps2` | m/s² | 机体系 X 轴规划线加速度；实验中直接使用原始 `linear_acceleration_x` |
| 12 | `planned_ay_mps2` | m/s² | 机体系 Y 轴规划线加速度；实验中直接使用原始 `linear_acceleration_y` |
| 13 | `planned_az_mps2` | m/s² | 机体系 Z 轴去重力规划线加速度；实验中使用原始 Z 加速度加 $9.80665$ |
| 14 | `cumulative_distance_m` | m | 从任务开始累计的计划飞行距离 |
| 15 | `cumulative_climb_m` | m | 规划高度正向差分的累计值 |
| 16 | `cumulative_descent_m` | m | 规划高度负向差分绝对值的累计值 |
| 17 | `wind_speed_mps` | m/s | 任务时空位置对应的风速预报 |
| 18 | `wind_sin` | 1 | 全局风向相对机体航向的正弦，即机体系横向风单位分量 |
| 19 | `wind_cos` | 1 | 全局风向相对机体航向的余弦，即机体系纵向风单位分量 |
| 20 | `relative_air_speed_mps` | m/s | 机体系三轴规划速度与机体系水平风矢量之差的模长 |
| 21 | `headwind_mps` | m/s | 机体系沿水平规划速度反方向的风分量；正值表示逆风 |
| 22 | `crosswind_mps` | m/s | 机体系垂直于水平规划速度的风分量绝对值 |
| 23 | `payload_kg` | kg | 任务携带的有效载荷质量 |

本轮特征修剪删除了以下输入：

- 原始 `speed`：数据说明表示巡航设定水平速度，不是逐时刻真实三轴速度，不适合直接替代规划轨迹。
- `planned_speed_mps`、`planned_horizontal_speed_mps`、`planned_acceleration_mps2`：均可由三轴分量计算，继续同时输入会重复表达相同信息。
- 四个设备启用状态和旧 `payload_rated_power_w`：历史数据中没有有效变化或可靠物理来源。
- `thermal_load_proxy`、`vision_energy_proxy_w`、`communication_energy_proxy_w`：属于经验构造代理，量纲或标定依据不足，不参与 TCN 精度评价。
- `obstacle_agility_index` 和 route 独热：前者重复组合运动量，后者会把模型限制在已见路线编号。

特征的非空数、唯一值数量、零值比例、均值、标准差和保留理由写入 `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.0/processed/feature_analysis_4.0.csv`，可用于检查零列、常量列和重复输入是否重新出现。

## 8. 坐标、风向与历史数据映射

4.0 对坐标采用“位置 ENU、运动机体系”的明确约定。位置以任务起点为原点，局部 $X$ 轴向东、$Y$ 轴向北、$Z$ 轴向上；`planned_vx/vy/vz` 和 `planned_ax/ay/az` 保持规划器输出的机体系三轴分量。这里不再把历史速度、加速度旋转成 ENU：任务规划器生成的轨迹被视为无人机必须跟踪的参考轨迹，实验原始飞行数据的轨迹、三轴速度和三轴加速度直接作为该规划输入的历史样本。路线 GIF 中的“完整规划轨迹”和“已执行轨迹”因此使用同一组位置坐标，当前位置标记只表示任务推进到的时间点。

`planned_heading_deg` 只用于把全局风预报转换到机体系。全局风向角按从北向顺时针量取，接口统一使用“风矢量吹向”约定；若预报源使用气象学“风从何处吹来”，接入层先加 180°。令 $\psi_h$ 为规划航向，$\psi_w$ 为全局风矢量方向，二者均以同一个 `wind_coordinate_system` 表示，则相对风角为：

$$
\psi_{\mathrm{rel}}=\psi_h-\psi_w
$$

$$
w_x=v_w\cos\psi_{\mathrm{rel}},\qquad w_y=v_w\sin\psi_{\mathrm{rel}}
$$

其中：

- $\psi_{\mathrm{rel}}$：相对机体航向的风角，单位 ° 或弧度（计算时使用弧度）。
- $\psi_h$：机体系 X 轴相对全局规划坐标的航向角，单位 °。
- $\psi_w$：全局风矢量吹向角，单位 °。
- $v_w$：风速预报值，单位 m/s。
- $w_x$、$w_y$：机体系 X/Y 方向的风速分量，单位 m/s。

由 $w_x$、$w_y$ 与机体系 `planned_vx_mps`、`planned_vy_mps`、`planned_vz_mps` 计算相对空速、逆风和侧风。位置和速度不属于同一个坐标系并不矛盾：位置用于描述路线几何，机体系运动量用于描述飞行器受力和动力功率。`wind_forecast_source`、`wind_forecast_timestamp`、`wind_speed_error_mps`、`wind_direction_error_deg`、`wind_coordinate_system` 和 `wind_direction_convention` 记录风预报的可追溯性与误差范围。

原始历史数据用于训练时采用以下映射：

- 原始 `position_x` 为经度，按起点纬度缩放为东向 `planned_position_x_m`；原始 `position_y` 为纬度，转换为北向 `planned_position_y_m`。
- 原始 `position_z` 映射为 `planned_altitude_m`。
- 原始 `velocity_x/y/z` 直接映射为机体系 `planned_vx/vy/vz`；不使用巡航设定列 `speed`。
- 原始 `linear_acceleration_x/y/z` 直接映射为机体系加速度，Z 轴逐点加 9.80665 m/s² 去除静态重力，得到不含重力项的 `planned_az_mps2`。
- 原始姿态四元数只用于提取 `planned_heading_deg`，再与风向求相对风角。
- 原始 `wind_speed`、`wind_angle` 和 `payload` 分别映射为风场与载荷输入，其中原始载荷由 g 换算为 kg。

这套映射把“规划器要求的轨迹状态”与历史飞行记录对齐：离线训练使用原始飞行已经执行的轨迹作为当时的参考规划轨迹，部署时由调度器和中层规划器提供同字段，任务前阶段不读取实时 IMU。历史风角没有明确区分真北和磁北，因此实际接入风预报时必须先把角度转换到 `wind_coordinate_system`，不能把地理角、磁北角、机体系角和规划器角直接混合。

## 9. 任务前与飞行中输出契约

### 9.1 任务前接口

`predict_task_before` 返回逐时间步 DataFrame 和任务摘要字典。逐步表至少包含：

- `predicted_base_power_w`：TCN 预测的基础飞行动力功率；
- `auxiliary_power_w`：规划器提供的已知设备额定功率；
- `predicted_power_w`：两者相加后的总功率；
- `predicted_base_energy_wh`、`auxiliary_energy_wh`、`predicted_energy_wh`；
- 每步能耗上下界、累计能耗及累计上下界。

任务摘要提供基础动力总能耗、附加设备总能耗、任务总能耗、任务总能耗上下界、置信度、可用能量、返航能耗上界、安全储备、所需总能量、电池约束是否满足和 `continue` / `charge_or_replace` 决策。批量接口返回各 `task_id` 的合并逐步表和摘要列表，调度器不需要解析图像。

命令行演示把结构化结果写入：

- `out/tasks/task_before_prediction_4.0.csv`
- `out/tasks/task_before_summary_4.0.json`

### 9.2 飞行中接口

`OnlineTaskSession` 在创建时固定保存任务前基线。每次 `update` 只接受连续的完整窗口编号、该窗口结束后的实际能耗，以及可选的任务前窗口预测能耗校验值、电池可用能量、返航能耗上界和安全储备。

RLS 默认窗口为 1 s，即 5 个 0.2 s 时间步。第一个完整窗口结束前使用中性状态 $(b,s)=(0,1)$，只输出 TCN 基线。窗口结束后按同窗数据更新参数，新参数从下一行开始作用于所有剩余窗口。输出包括：

- 未执行时间步的新功率、能耗和上下界；
- 已执行实际能耗、剩余任务能耗和剩余区间；
- 预测最终总能耗和最终区间；
- 偏置、缩放、协方差、最近残差、更新次数和本次观测对预测/区间宽度的影响；
- `continue`、`replan_required`、`degraded_execution_recommended`、`pause_required` 和 `return_or_charge` 在线决策状态。相对窗口偏差达到 15%、30% 和 50% 时分别触发重新规划、降级执行和暂停建议；电池可用能量不足时优先输出返航或充电。

命令行演示把结构化结果写入：

- `out/tasks/online_remaining_prediction_4.0.csv`
- `out/tasks/online_decision_4.0.json`

## 10. 运行方式

从项目根目录运行完整流程：

```bash
python 4.0/main.py all --device cuda --force-prepare
```

也可以进入 `4.0` 后运行：

```bash
python main.py all --device cuda --force-prepare
```

`all` 的执行顺序为：

```text
prepare
-> tune-tcn
-> train
-> tune-rls
-> evaluate
-> task-demo
-> online-demo
-> visualize
```

常用分阶段命令：

```bash
python 4.0/main.py prepare --force-prepare
python 4.0/main.py tune-tcn --device cuda --tune-epochs 20
python 4.0/main.py train --device cuda --epochs 80
python 4.0/main.py tune-rls --device cuda
python 4.0/main.py evaluate --device cuda
python 4.0/main.py task-demo --device cuda
python 4.0/main.py online-demo --device cuda
python 4.0/main.py visualize --device cuda
```

路径和批量大小覆盖示例：

```bash
python 4.0/main.py all --data-dir D:/data --save-dir D:/model-4.0 --out-dir D:/energy-out-4.0 --device cuda:0 --batch-size 2048 --force-prepare
```

完整参数说明：

```bash
python 4.0/main.py --help
```

## 11. 命令行模式

| 模式 | 功能 | 是否训练 TCN | 日志处理 |
|---|---|---:|---|
| `prepare` | 排除地面测试、按真实时间重采样、生成 23 维特征与 flight 切分 | 否 | 追加 |
| `tune-tcn` | 对 8 个 TCN 时间窗分别执行 20 轮搜索 | 是 | 清空后记录 |
| `train` | 读取当前最优时间窗并正式训练 80 轮 | 是 | 追加 |
| `tune-rls` | 固定 TCN，在验证集搜索 7×7 组 RLS 参数 | 否训练 TCN；需要 GPU 推理 | 追加 |
| `evaluate` | 生成测试/全路线预测、分层指标和在线修正轨迹 | 否训练 TCN；需要 GPU 推理 | 追加 |
| `task-demo` | 生成一条新路线并调用任务前接口 | 否训练 TCN；需要 GPU 推理 | 追加 |
| `online-demo` | 输入一个完整窗口实际能耗并演示剩余路线更新 | 否训练 TCN；需要 GPU 推理 | 追加 |
| `visualize` | 根据当前结构化结果重建 SVG、GIF、HTML 和路线产物 | 否训练 TCN；当前入口仍检查 GPU | 追加 |
| `all` | 执行上述完整八阶段流程 | 是 | 清空后记录 |

单独执行后置模式前，应确认依赖产物存在。例如 `train` 需要处理后 train/val 表；`tune-rls` 需要最终 TCN 权重和 scaler；`evaluate` 还需要 RLS 搜索表；`online-demo` 需要任务前演示输入或由程序重新构造演示任务。

## 12. 算法概览

### 12.1 基础动力功率与附加功率

TCN 使用长度为 $L$ 的当前及历史规划序列预测基础动力功率：

$$
\widehat{P}^{\mathrm{base}}_t=f_{\Theta}\!\left(\mathbf{X}_{t-L+1:t}\right),\qquad
\widehat{P}^{\mathrm{total}}_t=\widehat{P}^{\mathrm{base}}_t+P^{\mathrm{aux}}_t
$$

其中：

- $\widehat{P}^{\mathrm{base}}_t$：第 $t$ 步 TCN 预测的基础飞行动力功率，单位 W。
- $f_{\Theta}$：参数为 $\Theta$ 的因果 TCN。
- $\mathbf{X}_{t-L+1:t}$：截至第 $t$ 步、长度为 $L$ 的 23 维规划特征序列。
- $L$：TCN 时间窗对应的离散步数。
- $\widehat{P}^{\mathrm{total}}_t$：基础动力与已知设备相加后的总预测功率，单位 W。
- $P^{\mathrm{aux}}_t$：规划时已知的设备额定功率总和，单位 W，不参与 TCN 拟合。
- $t$：任务离散时间步索引。

网络由两层残差 TCN block 组成，通道数为 32、32，膨胀率为 1、2，卷积核大小为 3，dropout 为 0.05。每个 block 使用因果卷积、GroupNorm、SiLU 和残差连接，回归头读取最后时刻特征。训练目标是标准化基础动力功率的 Smooth L1 损失；电压、电流和总功率标签不进入输入张量。

### 12.2 任务前不确定性

验证集基础功率残差用于拟合条件标准差。条件变量包括风速、风向正弦/余弦、载荷、任务时长、任务进度和由三轴速度计算的速度模长。任务级标准差按各步能量方差的平方和合成：

$$
\sigma_{E,\mathrm{task}}=\sqrt{\sum_t\left(\sigma_{P,t}\frac{\Delta t}{3600}\right)^2}
$$

其中：

- $\sigma_{E,\mathrm{task}}$：整条任务预测能量的标准差，单位 Wh。
- $\sigma_{P,t}$：根据第 $t$ 步工况估计的功率残差标准差，单位 W。
- $\Delta t$：固定离散步长 0.2 s。
- $t$：任务离散时间步索引。

这种计算不会把一个固定功率半径按时间步线性累加。它仍是基于历史残差的统计近似，真实覆盖率和分工况表现必须以本轮评估产物为准。

### 12.3 RLS 剩余路线校正

第 $k$ 个完整窗口结束前，RLS 使用旧参数预测该窗口；实际能耗到达后才更新：

$$
\widehat{E}^{\mathrm{corr}}_k=b_k+s_k\widehat{E}^{\mathrm{base}}_k
$$

更新后的状态对剩余 $n_{\mathrm{rem}}$ 个窗口整体生效：

$$
\widehat{E}_{\mathrm{rem}}=n_{\mathrm{rem}}b_{k+1}+s_{k+1}\widehat{E}^{\mathrm{base}}_{\mathrm{rem}}+E^{\mathrm{aux}}_{\mathrm{rem}}
$$

其中：

- $\widehat{E}^{\mathrm{corr}}_k$：更新前对第 $k$ 个窗口的校正能耗预测，单位 Wh。
- $\widehat{E}^{\mathrm{base}}_k$：任务前 TCN 对同一窗口的基础动力能耗预测，单位 Wh。
- $b_k$：第 $k$ 个窗口开始时的每窗口偏置参数，单位 Wh/窗口。
- $s_k$：第 $k$ 个窗口开始时的无量纲缩放参数。
- $\widehat{E}_{\mathrm{rem}}$：第 $k$ 个窗口结束后重新计算的剩余任务总能耗，单位 Wh。
- $n_{\mathrm{rem}}$：剩余 RLS 窗口数。
- $b_{k+1}$、$s_{k+1}$：用第 $k$ 个窗口实际能耗更新后的参数。
- $\widehat{E}^{\mathrm{base}}_{\mathrm{rem}}$：所有剩余时间步的 TCN 基础能耗总和，单位 Wh。
- $E^{\mathrm{aux}}_{\mathrm{rem}}$：剩余时间步中确定性附加设备能耗总和，单位 Wh。
- $k$：已结束的 RLS 窗口索引。

RLS 初始参数为偏置 0、缩放 1。区间同时使用参数协方差和近期窗口残差方差；稳定观测通常使参数不确定性减小，新残差或工况变化也允许区间重新扩大。剩余窗口为 0 时，剩余能耗和剩余区间均归零，最终任务能耗等于累计实际观测能耗。

完整递推、区间公式、评估定义和当前参数结果见 [`out/outreadme.md`](./out/outreadme.md)。

## 13. 训练与超参数约束

- TCN 时间窗候选固定为 0.6、1、2、3、4、5、8、10 s，对应 3、5、10、15、20、25、40、50 个 0.2 s 步。
- 每个候选训练 20 轮，使用验证集基础功率 WAPE 排序。搜索阶段共执行 8×20 个候选训练 epoch。
- 最优时间窗进入正式训练，正式训练固定 80 轮；当前 `patience=80`，不会因少量验证波动在预算中途提前结束。
- RLS 窗口固定为 1 s。遗忘因子候选为 0.85、0.88、0.90、0.93、0.96、0.98、0.99；初始协方差候选为 0.05、0.10、0.25、0.50、1.00、2.00、4.00，共 7×7=49 组。
- TCN 和 RLS 参数均只用训练集/验证集选择，测试集不参与模型选择。
- 本版本不执行输入组合消融实验，也不把 RLS 窗口长度作为额外搜索维度。
- 默认随机种子为 42，batch size 为 2048，学习率为 $3\times10^{-4}$，权重衰减为 $10^{-4}$。

命令行允许覆盖 `--tune-epochs` 和 `--epochs` 以便诊断，但正式复现实验应使用 20 轮搜索和 80 轮正式训练。任何覆盖都必须保留在日志和结构化训练表中，不能与正式结果混写而不说明。

## 14. 产物管理

`out/` 是 4.0 唯一的正式运行产物根目录，不建立 `runs` 归档层。

```text
out/
├── outreadme.md                 # 当前产物对应的算法、结果和逐图分析
├── D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.0/processed/  # 特征、切分、逐维元数据、关系分析和重采样映射
├── model/                       # 搜索日志、评估指标、flight/window/route汇总和区间模型
├── predictions/                 # 测试集与全部正常路线逐步预测
├── tasks/                       # 调度器可读取的任务前/在线CSV与JSON
├── rls/                         # 参数搜索、参数变化和未来路线修正轨迹/GIF
├── figures/
│   ├── training/                # 窗口排名、损失和训练过程
│   ├── results/                 # 总体、flight、分箱和RLS比较
│   ├── prediction/              # 功率/窗口能耗时序、散点和残差
│   └── custom/                  # 任务前与在线接口示例
├── routes/                      # 每条route的代表flight独立产物
└── logs/terminal_4.0.log        # UTF-8完整终端记录
```

`out/rls/` 只保留与参数变化和未来路线修正直接相关的文件：49 组参数搜索表、参数轨迹/统计/摘要、代表 flight 参数图、未来修正轨迹 CSV/摘要 JSON 和收敛 GIF。未来修正轨迹应展示实际窗口逐步进入后，预测最终能耗、剩余能耗和置信区间如何变化；区间不要求人为单调缩小，中途残差增大时应如实扩大。

每个 `out/routes/route_<route>/flight_<id>/` 与 3.2 保持相同用途，保存：

```text
flight_<id>_imu_aligned.csv
flight_<id>_imu_trajectory.gif
flight_<id>_imu_trajectory.html
flight_<id>_power_energy.svg
flight_<id>_summary.json
```

路线根目录另保存全部路线汇总 SVG 和 `route_products_summary_4.0.json`。SVG/GIF/HTML 用于人工审查，CSV/JSON 才是调度器和后处理程序的读取接口。所有图表必须具有中文表头、标题、坐标、单位和图例；`out/outreadme.md` 直接嵌入图表，并在每张图下说明来源、读图方法、数据现象、结论和局限。

## 15. 文件职责与调用关系

`main.py` 通过 `config.py` 构建统一配置，再按模式调用下游模块：

```text
main.py
├── data_utils.py
│   └── 原始记录 -> 0.2 s重采样 -> 23维规划特征 -> flight级切分
├── train.py
│   ├── model.py -> PlanningTCN
│   ├── 8窗TCN搜索与80轮正式训练
│   ├── 验证集条件残差模型
│   └── evaluate.py -> 49组RLS参数评价
├── evaluate.py
│   └── model.py -> DynamicEnergyRLS无泄漏在线模拟与剩余路线重算
├── task_api.py
│   ├── predict_task_before / predict_candidate_tasks
│   └── OnlineTaskSession
├── visualize.py -> SVG、GIF、HTML和分路线产物
└── terminal_logger.py -> 日志生命周期和异常回溯
```

各模块通过 `ExperimentConfig` 读取路径和参数，不应自行创建另一套输出根目录。`data_utils.py` 是特征定义的唯一代码来源，`feature_metadata_4.0.json` 是运行后对应的结构化说明；`task_api.py` 必须与两者保持字段和单位一致。

## 16. 日志规则

`terminal_logger.py` 把标准输出、标准错误、阶段摘要、耗时和异常回溯写入 `out/logs/terminal_4.0.log`，文件编码为 UTF-8。

只有包含完整 TCN 搜索的 `tune-tcn` 和 `all` 在开始时清空旧日志。`prepare`、`train`、`tune-rls`、`evaluate`、`task-demo`、`online-demo` 和 `visualize` 均追加，不覆盖此前阶段记录。运行头会标明“清空后记录”或“追加记录”。

结构化训练曲线写入 `out/model/training_log_4.0.csv`，并用 `phase`、`candidate`、`window_seconds` 区分 8 个搜索候选和正式训练。终端日志用于追踪流程、GPU、耗时和异常，不替代 CSV/JSON 指标。

## 17. 文档职责

### 17.1 `README.md`

稳定项目入口。记录当前 4.0 的业务定位、接口、字段、坐标、目录、命令、训练预算、日志和长期复现约束。例行重训不在这里追加本轮指标，也不逐张解释图表。

### 17.2 `improve.md`

追加式工程改进日志。历史记录不得删除；旧配置或旧结果被新实现替代时，新增一条纠正记录，明确被纠正的条目、实现范围、验证状态和输出影响。尚无最终证据时写“未验证”或“以本轮结构化产物为准”，不能预填指标。

### 17.3 `out/outreadme.md`

与当前 `out/` 一一对应的运行报告。它负责完整算法与公式、数据规模、GPU 运行状态、最优超参数、最终指标、警告和局限，并枚举和直接展示每个 CSV、JSON、SVG、GIF、HTML 产物。`out/` 下任何文件重建、移动或删除后，都必须同步检查该报告的清单、相对链接和数值。

三个文档均使用 UTF-8 无 BOM，链接使用仓库相对路径。README 与 outreadme 的公式采用 GitHub 可渲染的 `$...$` 和 `$$...$$`，公式后逐项解释关键变量。

## 18. 当前边界与复现约束

- 任务前模型依赖的是中层规划器能给出的完整轨迹。若部署规划值和历史实飞替代值存在明显分布偏移，离线指标不能直接代表线上精度，需要用真实“规划—执行”成对数据重新标定。
- 当前输入没有独立的电机启停、飞行阶段或相对地面高度字段。低速度、低加速度状态既可能对应地面等待，也可能对应空中悬停；两类状态的动力功率差异可超过 300 W，单靠现有 23 维连续特征不能稳定区分。测试 Flight 278 中，实际功率连续超过 100 W 的时刻为 42.0 s，TCN 在 19.8 s 已提前抬升，起飞前实际/TCN 平均功率为 3.16/233.61 W。部署规划轨迹应增加 `planned_motor_on`、`planned_flight_phase` 或可信的离地高度，且累计爬升/下降应设置高度噪声死区。
- 当前 RLS 使用一组缩放和偏置修正整段剩余路线。长时间地面等待会把缩放参数压低，阶段切换到真实起飞后可能暂时低估动力功率；因此 flight 总能耗误差可能由“等待阶段高估”和“起飞阶段低估”抵消，不能替代启动过渡、时间窗和剩余任务误差检查。完整统计见 [`out/outreadme.md` 的 B.9 节](./out/outreadme.md#b9-启动阶段偏差专项诊断)。
- 公开数据没有设备额定功率，历史 `auxiliary_power_w` 全为 0；当前实验只能验证基础动力功率学习，不能据此评价设备附加功率估计误差。
- 公开数据没有明确说明 `wind_angle` 以真北还是磁北为基准。部署接入必须先把风预报转换到任务规划器声明的坐标系，并在每条任务中记录来源、时间戳、风速误差范围、风向误差范围和 `VECTOR_TO`/`METEOROLOGICAL_FROM` 约定；这些字段已经由任务前接口校验并写入任务摘要。
- 当前未见路线测试由 R2、R3、R4、R7 构成。未见风速范围、未见风向范围、未见载荷组合和不同任务长度仍需结合最终结构化评估表确认，不能仅用样本量较大的 R1 代表整体性能。
- 当前任务前决策为 `continue` / `charge_or_replace`；在线阶段除 `continue` 和 `return_or_charge` 外，还会按窗口相对偏差输出 `replan`、`degrade` 和 `pause` 信号。阈值由 `config.py` 中的 15%、30% 和 50% 配置，实际动作仍由上层调度器执行。
- 安全判断使用“剩余能耗上界 + 返航能耗上界 + 安全储备”。返航能耗上界和可用电量由外部规划器/电池管理模块提供，4.0 不从电压、电流原始序列自行估算。
- 4.0 不做输入组合消融实验，不依赖固定路线编号，不允许真实窗口标签提前进入当前窗口预测。
- 本目录的版本化文件名统一为 `4.0`；不得读取或覆盖 3.2 的模型和产物，也不得恢复 `out/runs/` 嵌套结构。
