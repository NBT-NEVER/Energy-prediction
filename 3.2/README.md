# 四轴无人机飞行能耗预测 3.2

## 目录

- [1. 版本定位](#1-版本定位)
- [2. 项目结构](#2-项目结构)
- [3. 环境与依赖](#3-环境与依赖)
- [4. 数据与模型存放位置](#4-数据与模型存放位置)
- [5. 运行方式](#5-运行方式)
- [6. 命令行模式](#6-命令行模式)
- [7. 算法概览](#7-算法概览)
- [8. 产物管理](#8-产物管理)
- [9. 文件职责与调用关系](#9-文件职责与调用关系)
- [10. 日志规则](#10-日志规则)
- [11. 文档职责](#11-文档职责)
- [12. 版本约束](#12-版本约束)

## 1. 版本定位

3.2 面向四轴无人机飞行过程的逐采样点功率与累计能耗预测。该版本读取全部原始航线，按完整 `flight` 划分训练集、验证集和测试集，使用时间感知 TCN 生成基础功率预测，再用真实能量窗口结束后得到的反馈执行 RLS 在线仿射校正。

本版本保留批量预测、全航线预测、预测区间、自定义工况和航线可视化功能。RLS 能量窗不是超参数搜索项：TCN 时间窗、RLS 遗忘因子和初始协方差由验证集选择，RLS 能量窗在最优 RLS 参数固定后作为独立变量比较。

本文件是 3.2 的稳定项目入口，只在版本、长期接口、目录结构或持久化规则发生变化时更新。本次运行的指标、完整算法推导和逐图分析见 [`out/outreadme.md`](./out/outreadme.md)，历次工程改进见 [`improve.md`](./improve.md)。

## 2. 项目结构

```text
3.2/
├── config.py                    # 路径、训练参数和版本化文件名
├── data_utils.py                # 原始数据检查、清洗、特征工程和flight切分
├── device_utils.py              # CUDA/CPU设备选择
├── evaluate.py                  # 测试指标和分层评估
├── main.py                      # 命令行入口与完整流程调度
├── model.py                     # 时间感知TCN和RLS实现
├── plot_rls_window_analysis.py  # RLS能量窗独立变量分析图
├── predict.py                   # 逐点、窗口、flight和区间预测
├── progress.py                  # 终端进度与预计剩余时间
├── route_visualization.py       # 分航线静态/交互/三维动图产物
├── terminal_logger.py           # 终端输出捕获与日志清空/追加控制
├── train.py                     # TCN搜索、正式训练、RLS搜索和窗口比较
├── uncertainty.py               # 预测区间校准与重算
├── visualize.py                 # 训练、评估、预测和RLS结果制图
├── requirements.txt             # Python依赖
├── README.md                    # 稳定项目说明
├── improve.md                   # 追加式改进日志
└── out/
    ├── outreadme.md             # 当前输出对应的详细算法与结果报告
    ├── D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/processed/  # 处理后数据、切分和元数据
    ├── model/                   # 调参、训练、评估和能量窗结果
    ├── rls/                     # RLS搜索、参数轨迹和区间校准
    ├── predictions/             # 测试集与全航线预测
    ├── figures/                 # SVG静态图
    ├── routes/                  # 各航线独立产物目录
    ├── custom/                  # 自定义工况输入与预测
    └── logs/                    # 完整终端日志
```

每个 `route` 在 `out/routes/route_<航线>/flight_<编号>/` 中保存一个代表 `flight` 的功率图、能量图、交互式 3D 轨迹 HTML 和高清 3D 轨迹 GIF，不与其他航线共用产物目录。

## 3. 环境与依赖

建议使用支持 CUDA 的 Python 环境。安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

依赖包括 `numpy`、`pandas`、`matplotlib`、`torch`、`pillow` 和 `plotly`。训练设备默认是 `cuda`；没有可用 CUDA 时可通过 `--device cpu` 明确切换，但完整 TCN 搜索耗时会明显增加。

## 4. 数据与模型存放位置

默认路径集中在 `config.py`，命令行可以覆盖数据、权重和输出根目录。

| 内容 | 默认位置 | 说明 |
|---|---|---|
| 原始数据根目录 | `D:/Python-files/Energy-prediction/data` | 项目外长期保存，不复制进版本目录 |
| 原始飞行记录 | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/raw/flights.csv` | 全部原始航线的逐采样记录 |
| 原始工况参数 | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/raw/parameters.csv` | flight级工况参数 |
| 处理后数据 | `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/processed/` | 特征表、秒级能量表和重新划分的数据集 |
| 最优阶段权重 | `D:/Python-files/Energy-prediction/model/best_energy_tcn_rls_3.2.pt` | TCN搜索阶段的最优 checkpoint |
| 最终权重 | `D:/Python-files/Energy-prediction/model/final_energy_tcn_rls_3.2.pt` | 正式训练及选定RLS参数 |
| 运行产物 | `3.2/out/` | 评估、预测、日志和全部可视化 |

处理后数据按完整 `flight` 划分，单个 flight 不跨训练、验证和测试集合。执行 `prepare --force-prepare` 或完整 `all --force-prepare` 时会根据当前原始数据重新生成划分；具体记录数、flight 数、航线分布和切分清单以 `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/processed/dataset_summary.json` 为准。

仓库不把原始数据、逐采样大 CSV 和外部模型权重当作源码管理。复现实验时应保持上述默认目录，或同时通过 `--data-dir`、`--save-dir` 和 `--out-dir` 指向新的明确位置。

## 5. 运行方式

在 `3.2` 目录执行完整流程：

```bash
python main.py all --device cuda --force-prepare
```

默认不写 `mode` 时也执行 `all`。完整流程顺序为：

```text
prepare
-> tune-tcn
-> train-fixed
-> tune-rls
-> compare-rls-windows
-> calibrate
-> evaluate
-> predict-all
-> visualize
-> route-visualize
```

3.2 默认对 10 个 TCN 时间窗候选分别训练 80 轮，最优窗口再正式训练 80 轮；RLS 搜索 7 个遗忘因子与 7 个初始协方差构成的 49 组组合。完整搜索规模较大，应优先使用 CUDA。

常用覆盖参数示例：

```bash
python main.py all --data-dir D:/data --save-dir D:/model --out-dir ./out --device cuda
python main.py tune-tcn --tune-epochs 80 --window-candidates 0.25,0.5,1,2,4,6,8,10,12,16
python main.py train-fixed --epochs 80
python main.py interval --confidence 0.90
```

## 6. 命令行模式

| 模式 | 功能 | 是否训练TCN | 日志处理 |
|---|---|---:|---|
| `download` | 检查并获取公开源数据 | 否 | 追加 |
| `prepare` | 清洗、构造特征并按flight划分数据 | 否 | 追加 |
| `tune-tcn` | 搜索TCN时间窗候选 | 是 | 清空后记录 |
| `train` | 执行TCN搜索、正式训练和RLS选择 | 是 | 清空后记录 |
| `train-fixed` | 使用已选TCN窗口正式训练 | 是，不搜索 | 追加 |
| `tune-rls` | 固定TCN，仅搜索RLS参数 | 否 | 追加 |
| `compare-rls-windows` | 固定RLS参数，比较能量窗 | 否 | 追加 |
| `calibrate` | 使用验证集建立预测区间 | 否 | 追加 |
| `evaluate` | 生成测试预测和评估指标 | 否 | 追加 |
| `test` | 评估并生成常规结果图 | 否 | 追加 |
| `predict` | 对指定CSV执行批量预测 | 否 | 追加 |
| `predict-all` | 对全部处理后原始航线预测 | 否 | 追加 |
| `interval` | 不重跑模型，仅重算置信区间 | 否 | 追加 |
| `visualize` | 根据当前结果重新生成图表 | 否 | 追加 |
| `route-visualize` | 为每种航线生成代表flight产物 | 否 | 追加 |
| `custom` | 构造或读取自定义工况并预测 | 否 | 追加 |
| `all` | 执行十阶段完整流程 | 是 | 清空后记录 |

完整参数说明可执行：

```bash
python main.py --help
```

## 7. 算法概览

原始记录先按 flight 清洗和重采样，程序仍根据相邻时间戳计算每个样本的 `dt_seconds`。真实功率由电压与非负放电电流计算，能量按真实时间间隔积分：

$$
E_{f,k}=\sum_{i\in k}P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

其中：

- $E_{f,k}$：flight $f$ 的第 $k$ 个能量窗口能量，单位 Wh。
- $P_{f,i}$：窗口内第 $i$ 个采样点功率，单位 W。
- $\Delta t_{f,i}$：该采样点对应的时间间隔，单位 s。

TCN 接收工况特征和 `dt_seconds` 组成的时间序列，经过时间门控与因果膨胀卷积后输出逐点功率。训练损失联合约束逐点功率和完整秒窗口能量，窗口选择同时考察采样点功率、秒级能量和 flight 总能量。

RLS 在每个 flight 开始和飞行状态切换时恢复中性参数。当前窗口只使用窗口开始前的参数校正 TCN 功率；真实能量在窗口结束后才参与更新，新参数仅作用于下一窗口，避免标签提前泄漏。RLS 超参数搜索不包含能量窗，1、2、5、10、20 s 窗口只在固定最优 RLS 参数后做变量影响分析。

完整数学建模、模型结构、联合损失、梯度回传、RLS 递推公式、停止条件和预测区间方法见 [`out/outreadme.md`](./out/outreadme.md)。

## 8. 产物管理

`out/` 是 3.2 唯一的运行产物根目录。各类产物不得散落到代码目录：

- `D:/Python-files/Energy-prediction/data/dji_matrice_100_data/3.2/processed/`：处理后特征、切分 CSV、数据摘要和特征元数据。
- `out/model/`：TCN 调参表、训练日志、评估指标、分箱统计和能量窗比较结果。
- `out/rls/`：RLS 调参表、逐窗口参数轨迹、参数统计和区间校准文件。
- `out/predictions/`：测试集预测与全部原始航线预测。
- `out/figures/`：训练、预测、评估和自定义工况 SVG；不再生成 PNG。
- `out/routes/`：每条航线自己的代表 flight 目录及其 SVG、HTML、GIF 和摘要 JSON。
- `out/custom/`：自定义工况输入、预测和摘要。
- `out/logs/terminal_3.2.log`：完整终端记录。
- `out/outreadme.md`：与当前 `out/` 内容同步的详细报告。

静态图统一使用 SVG。IMU 轨迹静态产物使用内嵌 Plotly 的 HTML，可用鼠标旋转、缩放和悬停查看；轨迹 GIF 使用 3D 视角并同时展示时间、无人机速度、风速和风向信息。全部轨迹采用三条基准航线共享参考点附近的局部米制坐标，坐标值以便于观察的共同平移原点表示，轴标题使用 `X (m)`、`Y (m)`、`Z (m)`，方向由箭头标识。

每次 `out/` 中任何文件发生变化，都必须同步更新 `out/outreadme.md` 并检查相对链接。逐采样大数据和模型权重若因体积不纳入 Git，报告仍需写清预期位置、生成命令和排除原因。

## 9. 文件职责与调用关系

`main.py` 构建 `config.py` 中的配置对象，并按模式调用下游模块。主要调用链如下：

```text
main.py
├── data_utils.py -> 原始数据、特征、flight切分
├── train.py
│   ├── model.py -> 时间感知TCN与RLS
│   └── device_utils.py / progress.py
├── uncertainty.py -> 验证集残差校准
├── predict.py -> 逐点与累计预测
├── evaluate.py -> 指标和分层明细
├── visualize.py -> 常规SVG图表
├── route_visualization.py -> 分航线SVG/HTML/GIF
└── terminal_logger.py -> 终端日志生命周期
```

`plot_rls_window_analysis.py` 读取 `out/model/rls_energy_window_comparison_3.2.csv`，独立生成能量窗比较 SVG。各模块通过 `ExperimentConfig` 获取路径和参数，不应自行创建另一套硬编码输出位置。

## 10. 日志规则

`terminal_logger.py` 将标准输出、标准错误、阶段摘要和异常回溯写入 `out/logs/terminal_3.2.log`，动态 `\r` 进度刷新不重复写入日志。

只有确实重新执行 TCN 超参数搜索的 `tune-tcn`、`train` 和 `all` 模式会在开始时清空旧日志。`train-fixed`、`tune-rls`、评估、预测、制图和其他模式均追加到现有日志，不得覆盖此前运行记录。终端运行头会明确显示“清空后记录”或“追加记录”。

结构化训练与评估结果仍分别写入 CSV/JSON；终端日志用于追踪阶段状态、耗时、异常和关键摘要，不替代结构化结果文件。

## 11. 文档职责

### 11.1 `README.md`

稳定项目入口，说明版本定位、结构、环境、运行方法、数据/权重/产物地址、文件职责、算法概览和长期约束。仅在版本或长期接口变化时修改，不追加每次训练指标和逐图分析。

### 11.2 `improve.md`

追加式工程改进日志。历史记录不得删除；每个新的功能优化节点记录时间、对应实现提交哈希、修改范围、原问题、处理方式、验证结果和输出影响。已经完成的实现提交由下一次文档提交引用，避免提交哈希自引用。

### 11.3 `out/outreadme.md`

当前运行的详细算法与结果报告。A 部分记录问题建模、公式、数据流程、停止条件、模型结构和梯度校正；B 部分从当前日志和结构化结果摘取运行状态与指标；C 部分用相对路径展示全部可视化并逐图说明数据来源、标题、坐标、单位、图例、数值现象、图外数据、结论和局限。`out/` 发生任何变化都要同步更新该文件。

三个文档都必须有目录、逻辑层级和有效相对链接。冲突内容按当前代码和最新结构化结果逐点修正，不能通过整段删除仍然有效的功能说明或历史记录解决。

## 12. 版本约束

- 本目录和内部版本化文件名统一使用 `3.2`，不得回写为 `3.0`。
- 3.2 使用全部原始航线重新构造数据并按完整 flight 切分，不读取项目内旧 `data/` 副本。
- 默认 TCN 搜索和正式训练均为 80 epoch；命令行覆盖参数必须在日志和结果元数据中体现。
- 当前 TCN 候选为 10 组时间窗，RLS 候选为 49 组参数组合。
- RLS 能量窗只作独立变量分析，不得加入 RLS 参数搜索或据此选择模型权重。
- 测试集只用于最终评估，不参与 TCN 或 RLS 参数选择。
- 静态图统一为 SVG，交互轨迹为 HTML，轨迹动图为高清 GIF；各航线产物相互隔离。
