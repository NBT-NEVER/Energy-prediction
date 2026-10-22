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
| 行驶状态 | 含实测车速、温度、海拔和 GPS，可构造计划路线代理并进行按行程留出回放 | 不含真实调度器计划轨迹、载荷质量、踏板、制动压力、HVAC 或坡度真值 |
| 数据规模 | 106 条真实道路行程、实验室标准循环，20 Hz 原始采样 | 行程数量远少于采样点数量；同一车辆/区域数据不等于跨车型泛化 |
| 双层预测 | 可留出完整行程，测试 TCN 基线及使用历史能耗观测的 RLS 在线校正 | RLS 结果是获得真实历史能耗反馈后的在线修正，不能作为无反馈一次性预测结果 |
| 调度器输入 | 计划速度、路线、坡度、时长和环境适合作为任务前输入，可与逐步功率/能量曲线接口对齐 | 数据中没有这些计划信号；实测轨迹代理不能验证真实任务开始前的预测泛化 |
| 载荷 | API 可接收并回显 `payload_kg` 任务元信息 | GECR 无载荷字段，模型不使用载荷，也不能预测载荷变化的能耗效应 |
| 合规与复现 | 仓库公开源码和 Excel；固定提交、保留原作者说明及许可证 | 仓库 README 的 MIT 标签未必自动覆盖所有数据文件；商业再分发前须核实数据许可及论文 CC BY-NC-ND 条款 |

## 标签和清洗

真实道路 Excel 使用 `Continuous20Hz` 工作表。实验 5.0 将 `PhaseTime` 排序并去重，关键输入缺失或非有限样本剔除，再聚合为 1 Hz。放电电流在样例中为负，因此定义 $P_t=-V_tI_t$，放电为正、再生回馈为负；区间能量 $E_t=P_t\Delta t/3600$，其中 $\Delta t=1$ s。累计计数器不作为标签。功率标签受采样对齐、符号定义和缺少独立功率测量影响，应作为数据重建值理解。

加工表和训练、验证、测试拆分写入 `D:/Python-files/Energy-prediction/data/MOTIVES-LAB-GECR/5.0/processed/`。完整行程切分，避免同一行程相邻采样跨集合。数据清洗结果、切分编号、无效文件和特征定义由 `processed/data_summary_5.0.json` 追踪。

### 计划输入与监督目标边界

参考 4.1 的输入筛选原则，字段依据任务开始前可用性、输入/监督目标/审计字段分离，以及训练、推理、标准化器和 checkpoint 顺序一致性确定，不机械复制十维。接口计划字段包括任务时长和时间、速度与加速度、累计距离、海拔与坡度、航向、环境温度及停车状态。`payload_kg` 只作为审计字段；GECR 没有载荷监督数据，因此当前模型不支持载荷敏感预测。

GECR 只有已执行行程。训练时将实测速度、GPS/海拔路线和环境温度映射为计划轨迹代理；标签单独由电压和电流构造，模型输入排除了功率、电压、电流、档位等运行期能耗结果。这满足部署接口的信息隔离，但代理计划来自与标签同一条已执行行程，并非调度器下发的独立计划。因此本轮结果是完整 trip 留出下的离线轨迹回放表现，不是实证出发前泛化。获得真实计划数据后需要重新训练，并按任务/路线分组评估。

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
