# 实验 4.0：面向任务规划的无人机能耗预测

## 1. 版本定位

4.0 是一个可供任务分配和充电决策直接调用的能耗预测组件。它把任务规划器给出的完整候选轨迹转换为 0.2 s 时间步，先生成整条路线的基线能耗；飞行过程中，每个完整 RLS 窗口结束后再接收外部计量模块提供的实际窗口能耗，更新校正状态，并重新计算当前窗口之后的全部剩余路线。

4.0 是 3.2 的独立副本。3.2 的代码、权重、日志和历史 `out/` 不由本目录读取，也不被本目录覆盖。

## 2. 输入契约

任务前接口 `task_api.predict_task_before` 接收完整候选路线，至少包括：

- `task_id`、`time_s`、`dt_seconds`（必须为 0.2 s）；
- `planned_vx_mps`、`planned_vy_mps`、`planned_vz_mps`、`planned_speed_mps`；
- `planned_ax_mps2`、`planned_ay_mps2`、`planned_az_mps2`、`planned_altitude_m`；
- `wind_speed_mps`、`wind_direction_deg`；
- `payload_kg`，以及可选的设备启用状态和额定功率。

风向在 4.0 中按规划器坐标系解释，并同时转换为正弦、余弦、顺风/逆风和侧风分量。`route` 可以保留用于查询和统计，但不进入 `feature_columns`，因此可以处理未见路线编号。

任务前输入禁止出现电压、电流、瞬时功率、真实能耗、实飞速度、实飞加速度和实飞角速度。电压、电流只在离线数据处理中生成 `power_w` 监督标签，或由外部计量模块在窗口结束时提供实际能耗；它们不会进入 TCN 或在线 RLS 输入。

## 3. 数据与离散化

原始数据位置由 `config.py` 指定。4.0 删除 A1、A2、A3 地面辅助测试，不将其用于正常飞行训练、验证或测试；H 悬停和 R1-R7 正常飞行统一作为连续轨迹状态处理。数据按完整 `flight` 划分，当前生成 138/31/31 个 train/val/test flight。

原始记录的中位采样周期约为 0.12 s，实际范围约为 0.04–0.12 s。程序按真实 `time` 重采样到固定 0.2 s 网格，并在 `out/data/processed_4.0/resample_map_4.0.csv` 记录每个新时间步对应的原始记录范围。每个时间步能耗为 `power_w × 0.2 / 3600` Wh，不能把数组下标当作时间。

## 4. 特征与模型

4.0 使用 29 个规划可得或由规划轨迹派生的低维字段，删除 route 独热、`obstacle_agility_index`、虚拟热负荷/视觉/通信功率代理以及重复的实测运动变量。完整字段、单位、来源和可获得时机见 [`feature_metadata_4.0.json`](./out/data/processed_4.0/feature_metadata_4.0.json)。

TCN 时间窗候选扩展为 0.6、1、2、3、4、5、8、10 s，对应 3、5、10、15、20、25、40、50 个 0.2 s 步；每组搜索和正式训练均严格完成 80 轮，不因短期验证波动提前终止。RLS 使用窗口能耗而非原始电压/电流，遗忘因子和初始协方差各 7 个候选，共 49 组。当前窗口的真实能耗只能在窗口结束后更新，更新后的参数只作用于后续时间步。

任务级区间由基线能耗、RLS 参数协方差和近期残差共同计算，不把固定单步半径机械累加到整条路线；新工况或残差增大时区间允许扩大。

## 5. 运行

在项目根目录执行：

```bash
python 4.0/main.py prepare --device cuda --force-prepare
python 4.0/main.py all --device cuda
```

也可以分阶段运行：

```bash
python 4.0/main.py tune-tcn --device cuda
python 4.0/main.py train --device cuda
python 4.0/main.py tune-rls --device cuda
python 4.0/main.py evaluate --device cuda
python 4.0/main.py task-demo --device cuda
python 4.0/main.py online-demo --device cuda
```

## 6. 目录与接口产物

```text
4.0/
├── config.py                 # 独立路径、采样周期和候选参数
├── data_utils.py             # 排除地面测试、重采样、特征和flight切分
├── model.py                  # 规划TCN和动态能耗RLS
├── train.py                  # TCN 8窗搜索、80轮正式训练、RLS 49组搜索
├── evaluate.py               # 测试集、窗口级和flight级指标
├── task_api.py               # 任务前和飞行中接口
├── main.py                   # 命令行入口
├── model/                    # 4.0独立权重和scaler
└── out/
    ├── data/processed_4.0/   # 特征、切分、元数据和重采样映射
    ├── model/                 # TCN、测试评估和flight/窗口汇总
    ├── rls/                   # 49组RLS搜索结果和参数轨迹
    ├── predictions/           # 测试集结构化预测CSV
    ├── tasks/                 # 任务前和在线接口CSV/JSON
    ├── figures/               # 分训练/结果/预测目录的SVG分析图
    ├── routes/                # 各route的SVG、GIF、CSV和JSON追踪产物
    └── logs/                  # terminal_4.0.log
```

任务调度器应读取 `task_before_summary_4.0.json`、`task_before_prediction_4.0.csv`、`online_decision_4.0.json` 和 `online_remaining_prediction_4.0.csv`；图表只承担人工分析和审查作用。

## 7. 文档分工

- `README.md`：长期稳定的接口、目录和复现入口。
- `improve.md`：追加式工程变更记录。
- `out/outreadme.md`：与本次 `out/` 同步的运行结果、图表和逐图分析。

## 8. 项目背景与后续调度用途

无人机任务调度不能只在飞行结束后统计功率。任务分配器在派遣前需要比较多条候选航线、载荷和风场组合，判断哪架无人机有足够可用能量；飞行中还需要根据电池计量模块已经结束的时间窗修正剩余任务，及时决定继续执行、降级、返航、暂停或充电。因此 4.0 的预测对象从“已完成飞行的功率拟合”扩展为“任务规划阶段的整条路线能耗预算 + 飞行中的剩余能耗更新”。

任务前阶段的输入来自规划器：每个离散时间步的速度、加速度、高度、路线距离、风预报和载荷状态。模型输出逐步功率、逐步能耗、累计能耗、总能耗区间和安全储备，结果可以直接被任务分配器读取。任务执行阶段不把未来真实标签带入当前预测，而是在窗口结束后将“同一窗口的基线预测能耗—外部计量实际能耗”作为 RLS 配对样本，重新计算剩余路线。

## 9. 4.0 文件流与追踪产物

原始 `flights.csv` → `data_utils.py` 真实时间戳重采样 → `processed_4.0` 特征和切分 → `train.py` TCN/RLS 搜索 → `evaluate.py` 测试表、分箱表、路线表 → `visualize.py` 训练/结果/预测/RLS/路线图表。每条 route 还会生成代表 flight 的规划能量 CSV、功率能量 SVG、轨迹 GIF 和摘要 JSON。SVG/GIF 用于审查，CSV/JSON 用于调度器和后处理程序。

## 10. 图表文字与格式约束

图表统一使用 `DejaVu Sans` 和 ASCII 英文标题、坐标、图例，避免 Windows 环境缺少中文字体造成方框或乱码。静态图只输出 SVG；动态轨迹只输出 GIF，不生成大量 PNG 中间文件。图表目录按 `training`、`results`、`prediction`、`custom` 分类，路线追踪单独位于 `out/routes/`。
