# 无人机任务前功率与能耗预测实验 4.1

## 目录

- [项目定位](#项目定位)
- [数据来源与输出](#数据来源与输出)
- [十维模型输入](#十维模型输入)
- [数据处理](#数据处理)
- [任务前接口](#任务前接口)
- [模型与在线修正](#模型与在线修正)
- [运行方法](#运行方法)
- [目录与文件职责](#目录与文件职责)
- [当前结果边界](#当前结果边界)
- [文档分工](#文档分工)

## 项目定位

实验 4.1 用任务执行前可获得的规划轨迹、任务时长、风场、载荷和计划状态预测无人机逐时刻功率与任务能耗。TCN 固定接收 10 维输入，`route` 只参与数据追踪、切分和分路线统计。真实功率、电流、电压、实飞运动量和姿态只能用于离线标签或原始数据分析，任务前接口会拒绝这些字段。

模型给出任务前功率基线，在线阶段每收到一个完整 1 s 窗口的实际能耗后，用二维 RLS 修正后续未执行路线。测试集只在 TCN、RLS 参数和区间倍率冻结后使用。

## 数据来源与输出

默认数据根目录为：

```text
D:/Python-files/Energy-prediction/data/dji_matrice_100_data
```

原始数据由 [`prepare_data.py`](./prepare_data.py) 直接读取：

```text
Source/Data_Collected_with_Package_etc/raw/12683453.zip::flights.csv
```

加工结果写入 `4.1/processed/`：

| 文件 | 用途 |
| --- | --- |
| `planning_energy_features_4.1.csv` | 全部 200 个 flight 的 0.2 s 加工表 |
| `train_4.1.csv` | 124 个完整 flight 的训练集 |
| `val_4.1.csv` | 29 个完整 flight 的验证集 |
| `test_4.1.csv` | 47 个完整 flight 的测试集 |
| `feature_metadata_4.1.json` | 10 维顺序、单位、来源、目标和接口禁用项 |
| `feature_analysis_4.1.csv` | 输入与监督字段的数值统计和选用理由 |
| `dataset_summary_4.1.json` | 数据规模、状态组合和一致性检查 |
| `resample_map_4.1.csv` | 新时间点与原始时间范围的审计映射 |

每行数据分为三类：

- 模型输入：下表 10 维，训练、验证、测试和任务接口顺序完全相同。
- 监督目标：`power_w`、`energy_interval_wh`，不进入 TCN 输入。
- 追踪审计：`flight`、`route`、`split`、`split_role`、`time`、`dt_seconds`、状态标注方法、质量标记和能耗守恒字段。

## 十维模型输入

| 维度 | 字段 | 单位 | 含义 | 任务前来源 |
| ---: | --- | --- | --- | --- |
| 1 | `time_s` | s | 当前 flight 开始后的相对时间 | 规划时间轴 |
| 2 | `task_duration_s` | s | 完整任务计划时长 | 规划轨迹末时刻 |
| 3 | `planned_position_east_m` | m | 相对任务起点的 ENU 东向位置 | 轨迹规划器 |
| 4 | `planned_position_north_m` | m | 相对任务起点的 ENU 北向位置 | 轨迹规划器 |
| 5 | `planned_position_up_m` | m | 相对任务起点的 ENU 上向位置 | 规划高度减起点高度 |
| 6 | `wind_east_mps` | m/s | 风矢量的 ENU 东向分量 | 风场预报转换 |
| 7 | `wind_north_mps` | m/s | 风矢量的 ENU 北向分量 | 风场预报转换 |
| 8 | `payload_g` | g | 任务有效载荷 | 调度器任务信息 |
| 9 | `planned_motor_on` | 0/1 | 计划电机运行状态 | 飞控任务状态或起降计划 |
| 10 | `planned_airborne` | 0/1 | 计划离地状态 | 起飞、降落和航段状态 |

`planned_motor_on` 和 `planned_airborne` 必须严格为 0 或 1，并满足：

$$
\mathrm{planned\_airborne}_t \le \mathrm{planned\_motor\_on}_t
$$

其中：

$t$：任务离散时刻。

$\mathrm{planned\_airborne}_t$：时刻 $t$ 的计划离地状态。

$\mathrm{planned\_motor\_on}_t$：时刻 $t$ 的计划电机状态。

## 数据处理

[`data_utils.py`](./data_utils.py) 和独立入口 [`prepare_data.py`](./prepare_data.py) 执行以下处理：

1. 从 Source 压缩包读取指定字段，排除 A1、A2、A3 地面辅助路线。
2. 按 `flight` 和真实 `time` 稳定排序，对相同 flight 内的重复时刻保留最后一条。
3. 每个 flight 独立换算任务起点局部 ENU 位置和 ENU 风分量，禁止跨 flight 差分。
4. 将原始记录重采样到固定 0.2 s 网格。
5. 原始功率达到 100 W 的首末证据间标记为电机运行阶段；相对起点水平位移达到 1 m 或垂直位移绝对值达到 0.5 m 的首末证据间标记为离地阶段。
6. 对状态执行二值检查和逻辑约束，再按 flight 校准重采样功率，使区间能耗总和与原始真实时间梯形积分一致。
7. 按完整 flight 划分数据。R2、R3、R4、R7，以及高风速、大载荷或超长任务条件进入强制留出测试集；其余 flight 在路线内随机划分。

单步监督能耗为：

$$
E_i = \frac{P_i\Delta t}{3600}
$$

其中：

$E_i$：第 $i$ 个 0.2 s 区间的能耗，单位 Wh。

$P_i$：第 $i$ 个区间的监督功率，单位 W。

$\Delta t$：固定采样周期，取 0.2 s。

当前加工数据共 188,790 行、200 个 flight，无 NaN、Inf 和重复 flight/time。逐 flight 重采样能耗最大相对误差为 $3.76\times10^{-14}\%$，低于 0.1% 容差。

## 任务前接口

[`task_api.py`](./task_api.py) 的外部表只允许 `task_id` 和固定 10 维字段。`dt_seconds` 在内部按 0.2 s 生成，不由调用方提供。接口检查：

- `time_s` 从 0 开始并严格按 0.2 s 递增；
- 同一任务的 `task_duration_s` 保持一致并覆盖完整时间轴；
- 载荷非负，所有数值有限；
- 两个状态字段为二值且满足离地状态不大于电机状态；
- 任何额外字段都会被拒绝，防止监督标签或实飞观测混入部署输入。

demo 输入由 [`main.py`](./main.py) 直接构造位置轨迹、风场和计划状态，不生成实飞运动量或姿态字段。正式接口预测示例：

```python
from config import build_config
from task_api import predict_task_before

cfg = build_config()
prediction, summary = predict_task_before(task_dataframe, cfg)
```

## 模型与在线修正

TCN 使用因果卷积，仅访问当前及历史任务前输入。8 个候选窗口在验证集比较时间步功率、1 s 窗口能耗和 flight 总能耗后，当前选择 4 s（20 步）窗口。网络通道为 `(32, 32)`，卷积核为 3，dropout 为 0.05；AdamW 学习率为 $3\times10^{-4}$，正式训练 80 轮。

RLS 以完整 1 s 窗口为更新单位，估计偏置与缩放系数：

$$
\hat{E}_k = \theta_{0,k} + \theta_{1,k}E_{\mathrm{TCN},k}
$$

其中：

$\hat{E}_k$：第 $k$ 个窗口经 RLS 修正后的能耗。

$\theta_{0,k}$：第 $k$ 次更新后的窗口偏置。

$\theta_{1,k}$：第 $k$ 次更新后的 TCN 能耗缩放系数。

$E_{\mathrm{TCN},k}$：第 $k$ 个窗口的 TCN 基线能耗。

验证集搜索 49 组参数后，当前遗忘因子为 0.93，初始协方差为 1.0。截断尾窗计入任务能耗但不更新 RLS。

## 运行方法

环境依赖见 [`requirements.txt`](./requirements.txt)。已有依赖和 CUDA 环境下可直接运行：

```powershell
# 仅从Source重建4.1数据
python 4.1/prepare_data.py --force

# 依次执行完整流程
python 4.1/main.py prepare --force-prepare
python 4.1/main.py tune-tcn --device cuda
python 4.1/main.py train --device cuda
python 4.1/main.py tune-rls --device cuda
python 4.1/main.py evaluate --device cuda
python 4.1/main.py task-demo --device cuda
python 4.1/main.py online-demo --device cuda
python 4.1/main.py visualize --device cuda

# 一条命令清理并重建所有运行产物
python 4.1/main.py all --device cuda --force-prepare

# 单元测试
python -m unittest discover -s 4.1/tests -v
```

## 目录与文件职责

| 文件 | 职责 |
| --- | --- |
| [`config.py`](./config.py) | 集中管理 Source、4.1 数据、模型和输出路径及超参数 |
| [`prepare_data.py`](./prepare_data.py) | 可独立运行的数据处理入口 |
| [`data_utils.py`](./data_utils.py) | 排序去重、重采样、状态标注、切分和一致性检查 |
| [`model.py`](./model.py) | 因果 TCN 与二维动态 RLS |
| [`train.py`](./train.py) | scaler、时间窗搜索、正式训练、不确定性和 RLS 调参 |
| [`evaluate.py`](./evaluate.py) | 测试集评价、分层统计、区间与在线轨迹审计 |
| [`task_api.py`](./task_api.py) | 严格任务前接口和飞行中剩余路线修正 |
| [`visualize.py`](./visualize.py) | 只读取 4.1 加工数据、预测和统计生成图形 |
| [`report.py`](./report.py) | 根据当前结构化结果重建 `out/outreadme.md` |
| [`main.py`](./main.py) | 统一命令行调度 |
| [`tests/test_4_1_contract.py`](./tests/test_4_1_contract.py) | 10 维、状态、数据和模型产物契约测试 |

模型文件保存在 `4.1/model/`，包括最终 checkpoint、最佳 checkpoint 和 scaler。运行结果保存在 `4.1/out/`。外部加工数据只保存在数据根目录的 `4.1/processed/`，绘图不回读 Source 或其他版本目录。

## 当前结果边界

正式测试集含 47 个 flight。TCN 时间步功率 WAPE 为 10.454%，TCN flight 总能耗 WAPE 为 3.774%；1 s 在线 RLS 后，时间步功率 WAPE 为 9.498%，flight 总能耗 WAPE 为 0.956%。

点预测优于任务级区间校准。当前名义 95% 的任务级覆盖率为 76.596%，超长任务组只有 33.333%，单个未见大载荷 flight 的覆盖率为 0%。这些分组样本量较小，但结果说明任务前区间不能当作已充分校准的安全保证。完整指标、逐图数据源和局限见 [`out/outreadme.md`](./out/outreadme.md)。

## 文档分工

- 本文件保存稳定接口、数据规则、目录和运行方法。
- [`improve.md`](./improve.md) 记录 4.1 的工程变更和验证证据。
- [`out/outreadme.md`](./out/outreadme.md) 与当前 `out/` 同步，保存算法细节、正式指标和每张 SVG/GIF 的图解。
