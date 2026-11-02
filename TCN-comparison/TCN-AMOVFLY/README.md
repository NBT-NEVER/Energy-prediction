# TCN-AMOVFLY

## 目录

- [实验与输入](#实验与输入)
- [数据与运行](#数据与运行)
- [模块与复现](#模块与复现)
- [结果与边界](#结果与边界)

## 实验与输入

本项目是原始10维/扩展15维不同TCN结构对比实验，使用AMOVFLY ready data的裁剪飞行状态预测功率并积分能耗。重新训练TCN，比较输入SMA和隐藏激活。

| 组别 | 输入维数 | 激活 | SMA |
| --- | --- | --- | --- |
| A | 10 | mixed | 无 |
| B | 10 | silu | 无 |
| C | 15 | mixed | 位置3维+风速和风向2维 |
| D | 15 | silu | 位置3维+风速和风向2维 |
| E | 15 | leaky_relu | 位置3维+风速和风向2维 |

| 字段 | 维数 | 单位 | 构造 |
| --- | --- | --- | --- |
| position_x_m, position_y_m, position_z_m | 3 | m | 裁剪航段首点归一化gps_x/y/z |
| wind_speed | 1 | m/s | 由正交风分量还原 |
| wind_angle | 1 | rad | 由正交风分量还原，范围[-π, π] |
| time_s, task_duration_s | 2 | s | 航段已过时间和完整航段时长 |
| payload_g | 1 | g | 文件名P参数 |
| uav_R, uav_G | 2 | 0/1 | 机型指示 |

功率、电池、电流、电压、四元数、加速度、气压和经纬度目标列只作标签或审计，不进入模型。C/D/E的5个SMA通道按flight只用当前及历史10点计算，窗口起点按已有点数求均值。

## 数据与运行

- 原始目录：`D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\sourse`。
- 加工目录：`D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\TCN-comparison`，train/val/test共用一套CSV和SHA-256。
- 模型目录：`D:\Python-files\Energy-prediction\model\TCN-AMOVFLY`；输出目录：`I:\STUDY\python\project\Energy-prediction\TCN-comparison\TCN-AMOVFLY\out`。

接受256个飞行，拒绝21个；训练/验证/测试为183/36/37个flight、505176/99980/105370行。完整flight及Multi-UAV关联组按种子42划分70/15/15；验证再拆成18个选模和18个校准flight。清洗去非有限和重复时间，保留gps_z>2m首末航段，航段内断点>2s整文件拒收；0.2s线性重采样，功率按源航段梯形能量守恒校准，最后时间箱保留真实时长。

```powershell
python main.py all --device cuda
python main.py evaluate --device cuda
python main.py plot
python main.py report
python -m unittest discover -s tests -v
```

TCN窗口1/2/5/10s各训练8轮，最佳窗口重新初始化训练至多40轮，验证损失耐心值8；batch=8192，AdamW学习率0.0003，权衰减0.001，Dropout=0.15，种子42。

## 模块与复现

`config.py`集中路径、字段、组别和参数；`main.py`调度；`data_utils.py`加载数据并按flight生成SMA；`model.py`定义TCN；`train.py`只用训练集拟合scaler，搜索窗口并正式训练；evaluate.py负责功率、1秒能量、完整航段及场景/运动状态指标，visualize.py绘图。报告由 [report.py](report.py) 调用 [共享生成器](../amovfly_reporting.py) 生成。

首次all或prepare调用 ../prepare_amovfly.py；prepare --force-prepare重建原始加工数据，之后需重新执行全部训练。
模式支持 prepare/tune/train/evaluate/plot/report/all；`--group A`指定组，`--device cpu`切换CPU；`--data-dir`、`--save-dir`、`--out-dir`覆盖路径。--tune-epochs、--epochs、--batch-size、--seed可覆盖预算与种子。 修改参数后使用独立模型/输出目录全量重跑，避免不同配置混合。

当前数据审计：[加工脚本](../prepare_amovfly.py)；当前算法和结果：[输出报告](out/outreadme.md)；工程历史：[improve.md](improve.md)。数据、权重和out因仓库既有存储策略保留本机，`all`可重建；未执行Git提交。

## 结果与边界

| group | feature_count | window_seconds | power_wape_percent | window_energy_wape_percent | flight_energy_wape_percent |
| --- | --- | --- | --- | --- | --- |
| A | 10 | 1.0000 | 10.1318 | 9.9306 | 5.3715 |
| B | 10 | 1.0000 | 10.1517 | 9.9478 | 5.0734 |
| C | 15 | 10.0000 | 8.7603 | 8.5677 | 5.7913 |
| D | 15 | 10.0000 | 9.0605 | 8.8751 | 6.0683 |
| E | 15 | 10.0000 | 8.9641 | 8.7729 | 5.5684 |

WAPE按样本/1秒窗口/完整flight分别统计；flight能量误差可能因正负误差抵消而低于瞬时功率误差。

功率WAPE最低为C组（8.7603%）。

任务能耗WAPE最低为B组（5.0734%）。

单种子和有限训练预算下，结果只代表当前划分。RLS反馈后的覆盖率低于名义95%时，区间不能解释为理论覆盖保证。

位置、速度和相对风来自实测航段，不等于起飞前已知计划；裁剪外起降待机不计入能耗。RLS任务前与剩余任务区间使用完整实测轨迹的TCN回归基线，是已知轨迹条件下的回放，不能解释为真实起飞前预测。线性重采样使用相邻右端点，SMA/TCN只在加工网格上使用历史，不保证原始传感器逐点严格实时。风分量使用相对风角正余弦，不是地理ENU风；route列为场景名而非实际航线。关联组隔离不等于跨日期留出，同日其他航班仍可能共享环境；仅当前随机划分和种子42，无多种子显著性结论。审计CSV记录接受文件的源哈希、重复时间、缺失行和裁剪量；JSON另列全部拒收原因。

README维护稳定接口；improve.md追加工程变化；out/outreadme.md记录当前算法、结果、图表和结构化产物。数据、模型和out按仓库规则保留在本机。