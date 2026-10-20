# MOTIVES-LAB GECR 数据适配说明

## 数据来源

- 代码与数据仓库：[MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev](https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev)
- 固定版本：`557a4de432c057b4d7d97a2800e4a48d414e132c`
- 当前本地副本：`D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/Source/`
- 原始内容：106 个真实道路 BEV 行程 Excel（约 262.8 MB）、46 个实验室循环 Excel（约 60.4 MB）、作者 README、MIT LICENSE 和 MATLAB 参考脚本。
- 本次模型实验只用 106 条真实道路行程；实验室循环暂时保留作后续参考，未进入训练/验证/测试。
- 数据论文：Yuan, X. et al. “Data-driven evaluation of electric vehicle energy consumption for generalizing standard testing to real-world driving.” *Patterns* 5(4), 100950 (2024). [DOI](https://doi.org/10.1016/j.patter.2024.100950)；[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11026974/)。作者仓库说明 106 条道路行程用于论文中的 GECR 真实道路验证。
- 勘误：2025 年 [DOI: 10.1016/j.patter.2025.101173](https://doi.org/10.1016/j.patter.2025.101173)，说明代码阈值 15 m/s 与论文文字的 15 km/h 不一致；引用原论文实验结论时应同时说明这一点。

## 对双层框架的适用性

| 维度 | 优势 | 局限与风险 |
| --- | --- | --- |
| 能耗标签 | 20 Hz 电池电压、电流和累计放电/回充字段；可构造带再生制动的电池侧净功率 | 功率由 $-VI$ 重建，非独立功率测量；累计量单位未标注，不能直接当 Wh |
| 行驶状态 | 含实测/计划车速、温度、海拔、GPS、档位，足以检验车辆功率时序预测 | 不含加速度踏板、制动压力、HVAC、载荷质量、道路坡度真值等完整能耗因素 |
| 数据规模 | 106 条真实道路行程、实验室标准循环，20 Hz 原始采样 | 行程数量远少于采样点数量；同一车辆/区域数据不等于跨车型泛化 |
| 双层预测 | 可留出完整行程，测试 TCN 基线及使用历史能耗观测的 RLS 在线校正 | RLS 结果是获得真实历史能耗反馈后的在线修正，不能作为无反馈一次性预测结果 |
| 自动驾驶关联 | 有 GPS 轨迹和道路速度过程，可用于车辆行驶能耗预测对照 | 数据不是自动驾驶专用数据；轨迹为实测事后轨迹，不能验证任务前路线规划或自动驾驶策略 |
| 合规与复现 | 仓库公开源码和 Excel；固定提交、保留原作者说明及许可证 | 仓库 README 的 MIT 标签未必自动覆盖所有数据文件；商业再分发前须核实数据许可及论文 CC BY-NC-ND 条款 |

## 标签和清洗

真实道路 Excel 使用 `Continuous20Hz` 工作表。实验 5.0 将 `PhaseTime` 排序并去重，关键输入缺失或非有限样本剔除，再聚合为 1 Hz。放电电流在样例中为负，因此定义 $P_t=-V_tI_t$，放电为正、再生回馈为负；区间能量 $E_t=P_t\Delta t/3600$，其中 $\Delta t=1$ s。累计计数器不作为标签。功率标签受采样对齐、符号定义和缺少独立功率测量影响，应作为数据重建值理解。

加工表和训练、验证、测试拆分写入 `D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/5.0/processed/`。完整行程切分，避免同一行程相邻采样跨集合。数据清洗结果、切分编号、无效文件和特征定义由 `processed/data_summary_5.0.json` 追踪。

## 已确认使用该数据的论文

目前根据作者仓库 README 与论文内容能直接确认的独立研究为上述 Yuan et al. (2024) *Patterns* 论文，明确使用 106 条真实道路行程验证 GECR。2025 年勘误是同一研究的更正记录，不是另一项数据使用研究。没有全文证据时，不把引用该论文的后续文章算作使用该原始数据的论文。

## 运行

```powershell
python 5.0/main.py prepare --force-prepare
python 5.0/main.py train
python 5.0/main.py evaluate
python 5.0/main.py all --force-prepare
```

本地已下载原始数据，因此不需要再次运行下载命令。新环境可通过仓库下载或 Git 克隆上述固定版本；只需把 `Source` 恢复到 `config.py` 设定的数据路径。
