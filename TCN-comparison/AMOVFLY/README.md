# AMOVFLY数据统计与可视化

## 目录

- [项目范围](#项目范围)
- [运行方式](#运行方式)
- [统计口径](#统计口径)
- [目录结构](#目录结构)

## 项目范围

独立分析AMOVFLY数据集，不训练或调用TCN/RLS模型。默认读取 `D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/sourse` 下FAFS、FAVS、VAFS、VAVS和Random场景的ready CSV。原始采集子目录raw_data、航线素材及独立风观测不混入ready逐点统计。

项目统计数据维度、类型、单位、缺失率、分位数与极值；按场景分析样本量和时长；检查重复时间、长采样间隔、明显物理边界、功率与电压电流一致性、四元数模长，并用IQR筛查统计极端候选。异常阈值用于人工复核，探索阶段不会自动删除原始样本。

代表性航迹按每个场景行数最多的单航段选择，输出实际经纬度投影轨迹、有效参考航点，以及gps_z/aim_z高度曲线。

当前统计：277个ready文件、829,021行记录、27个源字段，新增3个派生统计字段；全体ready数据覆盖约47.25小时。主要缺失为wind_speed与wind_angle，各31,275行；时间重复集中在1个航段，共2,032条重复时间记录；超过2秒的采样间隔分布于20个航段、共85处。其它异常和口径见输出报告。

## 运行方式

在本目录运行：

```powershell
python main.py
```

覆盖输入和输出位置：

```powershell
python main.py --data-dir "D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/sourse" --out-dir "./out"
python -m unittest discover -s tests -v
```

`main.py`调用 `analysis.py`生成统计CSV/JSON，`visualize.py`生成图表，`report.py`生成 `out/outreadme.md`。`config.py`集中管理数据路径和异常阈值。可将 `--max-gap`改为其他秒数，默认2秒；该值只影响异常事件标记，不裁剪数据。

依赖见 `requirements.txt`。当前Python环境已有NumPy、Pandas和Matplotlib，无需另行安装。

## 统计口径

字段缺失率按全体ready数据行计算。`Unnamed: 0`是CSV行序号，不作为传感器维度；另新增 `payload_g`（文件名载荷）、`speed_mps`（三轴速度模长）及 `power_from_vi_w`（电压乘电流）用于描述和核查。IQR边界是全数据字段的Tukey范围，极端候选不等同坏数据。

每个文件独立统计时间顺序、重复时间和最大采样间隔；异常表保留文件名、场景、事件代码、数量和阈值含义。电量百分比按[0,1]比例检查；功率一致性容差为max(2W,5%)；四元数模长偏离1超过0.05计为复核候选。此类标记不替代传感器校准或原始采集日志。

详细统计与图表口径见 `out/outreadme.md`。源ready表只读，分析结果保存在 `out/results` 和 `out/figures`。

## 目录结构

```text
AMOVFLY/
├── config.py
├── main.py
├── analysis.py
├── visualize.py
├── report.py
├── tests/
│   └── test_analysis.py
├── requirements.txt
├── README.md
├── improve.md
└── out/
    ├── outreadme.md
    ├── figures/
    └── results/
```

`README.md`记录稳定入口和统计定义；`improve.md`追加实现变化；`out/outreadme.md`对应当前生成数据、发现和图表。重新运行 `python main.py` 会刷新当前统计输出。
