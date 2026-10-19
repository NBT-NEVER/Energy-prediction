# 4.1 外部真实道路数据评估：GECR

## 数据入口

- 数据与作者代码仓库：[MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev](https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev)
- 106次真实道路行程目录：[real_world_data](https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev/tree/main/data/real_world_data)
- 单条行程直链：[001.xlsx](https://raw.githubusercontent.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev/main/data/real_world_data/001.xlsx)
- 数据集论文：[Data-driven evaluation of electric vehicle energy consumption for generalizing standard testing to real-world driving](https://doi.org/10.1016/j.patter.2024.100950)，Patterns 5(4), 100950, 2024。开放全文：[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11026974/)
- 论文勘误：[Erratum](https://doi.org/10.1016/j.patter.2025.101173)。勘误说明作者仓库中速度阈值按 15 m/s 使用（相当于 54 km/h），正文误写为 15 km/h；复用其算法结果时应注意。

仓库公开 106 个 `xlsx` 文件，总计约 263 MB。作者仓库 README 标注 MIT，但这不能自动证明数据表本身采用 MIT；原文全文标注 CC BY-NC-ND。研究复用可以先按论文引用，重新分发数据或用于商业项目之前应核对数据文件的明确许可。

## 已核验字段与适配

我检查了 `001.xlsx` 的 `Continuous20Hz` 工作表：30,384 行，`PhaseTime` 中位采样间隔 0.05 s；字段包括实际/计划车速、REESS 电流电压、温度、海拔、经纬度、档位和累计放电/回充量。该文件的 `tot_disc`、`tot_char` 均单调不减；样例中放电电流为负，放电累计量增加。

| 4.1字段 | GECR来源或处理 | 限制 |
| --- | --- | --- |
| `time_s` / `task_duration_s` | `PhaseTime`及行程时间范围 | 行程起止受采集记录定义影响 |
| `planned_position_east_m` / `north_m` | 经纬度换算为起点局部平面坐标 | 这是实测GPS轨迹回放，不是行程开始前拿到的路线规划 |
| `planned_position_up_m` | 海拔减起点海拔 | GPS/车辆海拔噪声需审查 |
| `power_w` | `max(-REESSVoltage * REESSCurrent, 0)` | 极性从字段与样例趋势推断，必须在正式实验前与论文单位/累计量核对；该标签不是直接给出的功率列 |
| `energy_interval_wh` | 上述功率乘真实采样间隔积分 | 电池回馈能量被截断为零，不能评估含再生制动的净能耗 |
| `wind_east_mps` / `wind_north_mps` | 置零并明确视为缺失特征基线 | 数据没有风场 |
| `payload_g` | 置零并明确视为缺失特征基线 | 没有车辆载荷质量 |
| `planned_motor_on` | 实测放电功率大于等于 100 W | 由标签反推，只能作格式/回放诊断 |
| `planned_airborne` | 实测车速大于 1 的车辆移动状态代理，并受 `motor_on` 约束 | 车辆没有“离地”概念；不可按该字段解释 |

适配脚本刻意把输出放进 `replay_proxy/`，且每行标注代理语义。它提供 4.1 十维字段的结构化输入和实测功率监督数据，适合检查数据读取、能耗积分、时间窗预测与在线RLS流程；不能直接并入现有无人机训练集，也不能把所得精度写成无人机任务前预测精度。数据中存在 `DAScheduleSpeed`，但道路数据的采集调度速度并不等于规划器完整轨迹，也没有可用的计划海拔、风、载荷信息。

## 下载与转换

使用仓库已有 Python 环境中的 pandas、numpy 和 openpyxl（已加入 `4.1/requirements.txt`）：

```powershell
# 先取一条行程确认本机下载与Excel读取链路
python 4.1/gecr_adapter.py download --trip 1
python 4.1/gecr_adapter.py convert --trip 1

# 下载/转换全部106条，约263 MB原始文件
python 4.1/gecr_adapter.py download
python 4.1/gecr_adapter.py convert
```

默认数据位置为仓库根目录下的 `data/gecr/`；可用 `--data-dir` 改到外部数据盘。原始文件保存于 `raw/`，适配输出保存于 `replay_proxy/`。不建议直接运行现有 `4.1/main.py all` 来处理此数据，因为该入口的正式切分策略、`flight` 含义、无人机状态和特征物理意义均针对原始无人机数据。

## 论文使用情况

截至本次检索，能由数据仓库作者 README 和论文正文直接确认使用这 106 条真实道路行程的论文为：

1. Yuan, X. et al. “Data-driven evaluation of electric vehicle energy consumption for generalizing standard testing to real-world driving.” *Patterns*, 5(4), 100950 (2024). DOI: [10.1016/j.patter.2024.100950](https://doi.org/10.1016/j.patter.2024.100950)。论文明确说明以 106 次真实道路行程验证其 GECR 方法，摘要报告这些行程的能耗估计误差为 7.12%。其作者代码仓库README直接把该论文列为引用要求。

2025 年发表的勘误是同一论文的更正记录，不是另一个独立的数据使用研究。没有把“引用该论文”的后续论文自动计作使用这份原始数据；要证明后续论文实际使用数据，还需在全文中确认数据来源或仓库文件。

## 适用性结论

| 评估目标 | 适配度 | 判断 |
| --- | --- | --- |
| 检查双层功率/能量模型的程序管线 | 中 | 高频电功率、轨迹和累计能耗可构造真实标签；电动车与无人机的动力学分布不同 |
| 检查时间窗能耗预测与在线RLS校正 | 中 | 行程可按时间切分并顺序回放，但RLS反馈窗口和标签极性需单独验证 |
| 验证无人机跨任务泛化 | 低 | 106个行程可按行程留出，但设备/驾驶区域/路线多样性不能替代无人机飞行工况 |
| 验证真正任务前输入性能 | 低 | `planned_*` 轨迹由事后实测位置回放；风、载荷和飞行状态缺失 |
| 验证无人车/BEV道路能耗方法 | 高 | 原始数据和论文均直接面向BEV真实行程能耗分析 |

因此，它是当前能找到的有真实能源标签、规模明确、公开可取的道路电动车候选中最适合作为外部管线验证集的一个；如果研究结论必须是“无人机任务前能耗预测在真实无人车数据上有效”，该结论不成立。更稳妥的实验名称是“跨域真实道路电动车回放验证”，单独报告，并将无人机测试集结果作为主结果。
