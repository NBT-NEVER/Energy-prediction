# TCN+RLS-AMOVFLY

## 目录

- [实验与输入](#实验与输入)
- [数据与运行](#数据与运行)
- [模块与复现](#模块与复现)
- [结果与边界](#结果与边界)

## 实验与输入

使用AMOVFLY ready data的裁剪飞行状态预测功率并积分能耗。冻结TCN，只搜索RLS遗忘因子、初始协方差、反馈窗并校准区间。

| 组别 | 输入维数 | 激活 | SMA |
| --- | --- | --- | --- |
| A | 10 | mixed | 无 |
| B | 10 | silu | 无 |
| C | 15 | mixed | 位置3维+风2维 |
| D | 15 | silu | 位置3维+风2维 |
| E | 15 | leaky_relu | 位置3维+风2维 |

| 字段 | 维数 | 单位 | 构造 |
| --- | --- | --- | --- |
| position_x_m, position_y_m, position_z_m | 3 | m | 裁剪航段首点归一化gps_x/y/z |
| wind_sensor_sin_mps, wind_sensor_cos_mps | 2 | m/s | wind_speed乘wind_angle的正弦/余弦 |
| payload_g | 1 | g | 文件名P参数 |
| speed_mps | 1 | m/s | v_x/y/z三维模长 |
| vertical_speed_mps | 1 | m/s | 源v_z |
| uav_R, uav_G | 2 | 0/1 | 机型指示 |

功率、电池、电流、电压、四元数、加速度、气压和经纬度目标列只作标签或审计，不进入模型。C/D/E的5个SMA通道按flight只用当前及历史10点计算，窗口起点按已有点数求均值。

## 数据与运行

- 原始目录：`D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\sourse`。
- 加工目录：`D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\TCN-comparison`，train/val/test共用一套CSV和SHA-256。
- 模型目录：`D:\Python-files\Energy-prediction\model\TCN+RLS-AMOVFLY`；输出目录：`I:\STUDY\python\project\Energy-prediction\TCN-comparison\TCN+RLS-AMOVFLY\out`。

接受256个飞行，拒绝21个；训练/验证/测试为183/36/37个flight、505176/99980/105370行。完整flight及Multi-UAV关联组按种子42划分70/15/15；验证再拆成18个选模和18个校准flight。清洗去非有限和重复时间，保留gps_z>2m首末航段，航段内断点>2s整文件拒收；0.2s线性重采样，功率按源航段梯形能量守恒校准，最后时间箱保留真实时长。

```powershell
python main.py all --device cuda
python main.py evaluate --device cuda
python main.py plot
python main.py report
python -m unittest discover -s tests -v
```

RLS网格为7×7×5=245候选，验证集18个flight搜索、18个独立flight校准；反馈窗结束后才更新，短尾窗不更新。

## 模块与复现

`config.py`集中路径、字段、组别和参数；`main.py`调度；`data_utils.py`加载数据并按flight生成SMA；`model.py`定义TCN；`train.py`加载冻结模型推理；rls.py负责反馈和区间，rls_evaluate.py负责搜索/校准/测试，visualize_rls.py绘图。报告由 [report.py](report.py) 调用 [共享生成器](../amovfly_reporting.py) 生成。

先在 ../TCN-AMOVFLY 执行all，生成AMOVFLY新权重；RLS没有重训接口。
模式支持 prepare/tune/calibrate/evaluate/plot/report/all；`--group A`指定组，`--device cpu`切换CPU；`--data-dir`、`--save-dir`、`--out-dir`覆盖路径。RLS固定使用config.py网格，TCN源权重路径见tcn_source_model_dir。 修改参数后使用独立模型/输出目录全量重跑，避免不同配置混合。

当前数据审计：[加工脚本](../prepare_amovfly.py)；当前算法和结果：[输出报告](out/outreadme.md)；工程历史：[improve.md](improve.md)。数据、权重和out因仓库既有存储策略保留本机，`all`可重建；未执行Git提交。

## 结果与边界

| group | tcn_flight_energy_wape_percent | flight_energy_wape_percent | task_start_interval_coverage_percent | post_feedback_interval_coverage_percent | post_feedback_interval_mean_width_wh | post_feedback_interval_score_wh |
| --- | --- | --- | --- | --- | --- | --- |
| A | 4.3628 | 0.8377 | 48.6486 | 89.6981 | 2.2600 | 19.6268 |
| B | 4.6295 | 0.7825 | 56.7568 | 88.7877 | 2.0840 | 24.5309 |
| C | 4.5733 | 0.8929 | 48.6486 | 89.5544 | 2.2113 | 21.4714 |
| D | 5.0193 | 1.0315 | 40.5405 | 89.1711 | 2.1197 | 24.7137 |
| E | 4.9995 | 0.8598 | 64.8649 | 90.1294 | 2.2774 | 24.3220 |

RLS区间为验证残差比率95%经验分位数校准；反馈后区间评分同时惩罚宽度和未覆盖。

功率WAPE最低为C组（4.5408%）。

任务能耗WAPE最低为B组（0.7825%）。

A组TCN任务能耗WAPE为4.3628%，RLS后为0.8377%；任务前覆盖48.65%，反馈后覆盖89.70%。

B组TCN任务能耗WAPE为4.6295%，RLS后为0.7825%；任务前覆盖56.76%，反馈后覆盖88.79%。

C组TCN任务能耗WAPE为4.5733%，RLS后为0.8929%；任务前覆盖48.65%，反馈后覆盖89.55%。

D组TCN任务能耗WAPE为5.0193%，RLS后为1.0315%；任务前覆盖40.54%，反馈后覆盖89.17%。

E组TCN任务能耗WAPE为4.9995%，RLS后为0.8598%；任务前覆盖64.86%，反馈后覆盖90.13%。

单种子和有限训练预算下，结果只代表当前划分。RLS反馈后的覆盖率低于名义95%时，区间不能解释为理论覆盖保证。

位置、速度和相对风来自实测航段，不等于起飞前已知计划；裁剪外起降待机不计入能耗。RLS任务前与剩余任务区间使用完整实测轨迹的TCN回归基线，是已知轨迹条件下的回放，不能解释为真实起飞前预测。线性重采样使用相邻右端点，SMA/TCN只在加工网格上使用历史，不保证原始传感器逐点严格实时。风分量使用相对风角正余弦，不是地理ENU风；route列为场景名而非实际航线。关联组隔离不等于跨日期留出，同日其他航班仍可能共享环境；仅当前随机划分和种子42，无多种子显著性结论。审计CSV记录接受文件的源哈希、重复时间、缺失行和裁剪量；JSON另列全部拒收原因。

README维护稳定接口；improve.md追加工程变化；out/outreadme.md记录当前算法、结果、图表和结构化产物。数据、模型和out按仓库规则保留在本机。