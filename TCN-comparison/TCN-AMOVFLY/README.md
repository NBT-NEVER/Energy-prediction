# TCN-AMOVFLY：planned状态五组实验

## 目录

- [输入和实验](#输入和实验)
- [数据与状态](#数据与状态)
- [训练与运行](#训练与运行)
- [当前结果](#当前结果)
- [文件与存储](#文件与存储)

## 输入和实验

| 字段 | 维数 | 单位 | 构造 |
| --- | --- | --- | --- |
| position_x_m, position_y_m, position_z_m | 3 | m | 完整记录相对首个有效点的三维实测位置 |
| wind_speed | 1 | m/s | 相对风正交分量模长 |
| wind_angle | 1 | rad | atan2重建，[-π,π] |
| time_s, task_duration_s | 2 | s | 已过时间和完整记录时长 |
| payload_g | 1 | g | 文件名P参数 |
| uav_type | 1 | 类别编码 | Y=1，R=2，G=3，按普通数值列标准化 |
| planned | 1 | 状态编码 | 0停机，1原地启动，2飞行；DJI同阈值历史标注 |

| 组别 | 输入维数 | 激活 | SMA |
| --- | --- | --- | --- |
| A | 10 | mixed | 无 |
| B | 10 | silu | 无 |
| C | 15 | mixed | 三维位置、风速、风向共5维 |
| D | 15 | silu | 三维位置、风速、风向共5维 |
| E | 15 | leaky_relu | 三维位置、风速、风向共5维 |

## 数据与状态

从 `D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\sourse` 的FAFS/FAVS/VAFS/VAVS/Random同名ready子目录读取，不重复读取raw_data。本次277个CSV接受252个，拒收25个。必要列去非有限行，时间稳定排序去重，至少20个有效点；完整记录内任意采样缺口>2秒则整文件拒收。源哈希、拒收原因、清洗量见加工目录amovfly_flight_audit.csv和摘要JSON。

保留完整有效时间范围，不再按gps_z>2m裁掉两端；位置减首个有效点。原始非负battery_voltage乘battery_current形成源功率，用真实时间梯形积分取得能量。风角先转换正余弦分量再插值，连续量重采样至0.2秒网格，末箱使用实际剩余时长；监督功率整体缩放到与源能量守恒，逐箱能量为功率乘实际秒数再除3600。全零记录保留零值并避免除零。

状态与DJI 4.1/data_utils.py一致：功率≥100W的首末证据之间连续标注motor_on；相对起点水平位移≥1m或垂直位移绝对值≥0.5m的首末证据之间连续标注airborne，再令airborne乘motor_on。两列分别线性插值后np.rint取整，限制airborne≤motor_on，planned为两列之和。因此0=(0,0)、1=(1,0)、2=(1,1)；无证据时相应列全零。二进制列只作审计，不额外进入TCN，不直接插值三级标签。

planned含功率阈值信息，首末证据填充使用完整记录，是指定的历史代理状态；不能解释为无需真实功率的起飞前精度或严格在线状态估计。部署时需由飞控或计划提供状态。GPS漂移可能提前触发位移阈值，本实验不另调阈值。uav_type以Y/R/G=1/2/3作为单一数值输入，不使用one-hot或embedding。

按完整flight和Multi-UAV关联组隔离，种子42将关联组随机分为70/15/15；验证关联组交替分角色，19个flight选窗/选最佳轮次，19个校准flight不参与本目录选模。完整记录和拒收范围改变，数据划分及任务能量定义随之变化，不与旧裁剪实验直接计算改善率。

| 划分 | rows | flights | 停机行 | 原地启动行 | 飞行行 |
| --- | --- | --- | --- | --- | --- |
| train | 519698 | 174 | 35441 | 984 | 483273 |
| val | 114661 | 38 | 11810 | 215 | 102636 |
| test | 120930 | 40 | 12743 | 211 | 107976 |

加工目录：`D:\Python-files\Energy-prediction\data\AMOVFLY-Dataset\TCN-AMOVFLY-planned`。契约：`AMOVFLY-TCN-planned-1.0`。

## 训练与运行

通道(32, 32)，Dropout=0.05，权重衰减=0.0001，batch=2048；窗口(1.0, 2.0, 5.0, 10.0)秒各完整8轮，正式完整80轮，无早停，种子42，学习率0.0003。

在本目录运行：

```powershell
python main.py all --device cuda
python main.py prepare --force-prepare
python main.py tune --device cuda
python main.py train --device cuda
python main.py evaluate --device cuda
python main.py plot
python main.py report
python -X utf8 -m unittest discover -s tests -v
python audit.py
```

all自动准备数据；force-prepare后重新选窗和训练。`--group A`至E指定组；`--data-dir`、`--save-dir`、`--out-dir`覆盖根路径；`--epochs`、`--tune-epochs`、`--batch-size`、`--seed`覆盖预算。配置改变时使用独立模型/输出目录，防止新旧结果混用。

## 当前结果

| group | feature_count | window_seconds | power_wape_percent | window_energy_wape_percent | flight_energy_wape_percent |
| --- | --- | --- | --- | --- | --- |
| A | 10 | 5.0000 | 6.3004 | 6.0990 | 2.8410 |
| B | 10 | 5.0000 | 6.3385 | 6.1336 | 2.8836 |
| C | 15 | 5.0000 | 5.8947 | 5.6914 | 2.8640 |
| D | 15 | 2.0000 | 5.8607 | 5.6624 | 3.0212 |
| E | 15 | 10.0000 | 6.1887 | 5.9677 | 2.6308 |

本次功率WAPE最低为D组（5.8607%）；任务能耗WAPE最低为E组（2.6308%）。各组独立选窗，仅代表当前种子与划分。

## 文件与存储

`config.py`集中管理路径、状态阈值和超参数；`prepare_data.py`独立读取sourse生成完整记录、机型和状态；`data_utils.py`核验加工表和计算SMA；`model.py`定义TCN；`train.py`标准化、选窗和正式训练；`evaluate.py`输出逐点/窗口/任务/分层指标；`visualize.py`生成9套PNG/SVG；`report.py`根据实际产物生成报告；`audit.py`核验源数据、划分、权重和指标。

权重/scaler：`D:\Python-files\Energy-prediction\model\TCN-AMOVFLY-planned`。输出：`I:\STUDY\python\project\Energy-prediction\TCN-comparison\TCN-AMOVFLY\out`；完整算法与结果见 [out/outreadme.md](out/outreadme.md)。旧输出归档于 `D:/Python-files/Energy-prediction/model/TCN-AMOVFLY-history/20261102-before-planned`，旧权重仍在原TCN-AMOVFLY目录。D盘数据、模型和out按仓库规则不纳入Git，可通过all重建。

README维护当前入口及结果摘要；[improve.md](improve.md)追加历史；out/outreadme.md随输出同步。
