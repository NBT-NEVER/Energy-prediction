# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: amovfly_reporting.py
# 开发时间: 2026-11-01
# 文件名: amovfly_reporting.py
# 功能说明: 生成AMOVFLY TCN对比实验的结果说明
# 版本号：1.0
from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

FEATURES = [
    ("position_x_m, position_y_m, position_z_m", "3", "m", "裁剪航段首点归一化gps_x/y/z"),
    ("wind_sensor_sin_mps, wind_sensor_cos_mps", "2", "m/s", "wind_speed乘wind_angle的正弦/余弦"),
    ("payload_g", "1", "g", "文件名P参数"), ("speed_mps", "1", "m/s", "v_x/y/z三维模长"),
    ("vertical_speed_mps", "1", "m/s", "源v_z"), ("uav_R, uav_G", "2", "0/1", "机型指示")]

FIGURE_NOTES = {
    "power_wape": "来源：汇总metrics.csv；横轴A-E，纵轴功率WAPE(%)。",
    "window_energy_wape": "来源：汇总metrics.csv；横轴A-E，纵轴1秒能耗WAPE(%)。",
    "flight_energy_wape": "来源：汇总metrics.csv和逐flight表；纵轴完整航段能耗WAPE(%)。",
    "representative_power": "来源：test_predictions.csv；横轴时间(s)，纵轴功率(W)，黑线为实际。",
    "representative_energy": "来源：test_predictions.csv；横轴时间(s)，纵轴累计能耗(Wh)。",
    "flight_energy_scatter": "来源：TCN-comparison_*_flight_energy.csv；实际/预测航段能耗单位Wh，虚线为y=x。",
    "flight_error_distribution": "来源：TCN-comparison_*_flight_energy.csv；纵轴航段绝对误差Wh，箱体为分布。",
    "window_search": "来源：window_tuning.csv；横轴TCN窗口秒数，纵轴验证综合选择分数。",
    "training_loss": "来源：training.csv；横轴epoch，纵轴标准化SmoothL1。",
    "interval_coverage": "来源：TCN_RLS_all_metrics.csv；纵轴不同粒度区间覆盖率(%)，虚线为名义95%。",
    "interval_coverage_width": "来源：TCN_RLS_all_metrics.csv；横轴任务前区间宽度Wh，纵轴任务前覆盖率%。",
    "rls_window_search": "来源：rls_tuning.csv；横轴反馈窗秒数，纵轴验证航段能耗WAPE%。",
    "remaining_energy_trace": "来源：test_trace.csv；横轴反馈窗数，纵轴最终能耗预测Wh，阴影为区间。",
    "stage_interval_coverage": "来源：stage_intervals.csv；横轴观察进度，纵轴任务区间覆盖率%。",
}


def algorithm_text(online):
    """功能: 返回报告中的核心算法与评价口径。
    参数: online为RLS标记。
    返回: Markdown文本。
    调用位置: write_reports。
    """
    text = r"""读取FAFS/FAVS/VAFS/VAVS/Random中与父目录同名的ready文件夹，共277个CSV；不重复导入raw_data。Flight_info列出270个文件，本地额外7个也按同一规则清洗。必要数值字段去非有限行，时间稳定排序后去重，gps_z>2m首末证据之间保留连续航段，航段内部可以存在短暂低空段。航段内有效采样间隔>2s时整文件拒收。清洗后256个航段、21个拒收；详情在D盘数据目录的审计CSV和摘要JSON。

源位置先减航段首点，风角先变成正余弦分量再插值，避免角度跨360度的插值错误。0.2s线性重采样后按源航段梯形能量整体缩放监督功率，最后时间箱用实际剩余秒数，非完整0.2s不补长。标签因此是能量守恒加工功率，不能声称与原始每点传感器读数完全相同。

$$E_f=\frac{1}{3600}\int_{t_0}^{t_1}P_f(t)\,dt,\quad c_f=\frac{E_f}{\sum_i\widetilde P_{f,i}\Delta t_{f,i}/3600},\quad P^*_{f,i}=c_f\widetilde P_{f,i}$$

其中：

- $f$：裁剪航段索引。
- $t_0,t_1$：源航段首末时刻，s。
- $P_f(t)$：非负源功率，W；积分采用梯形法。
- $E_f$：源航段能量，Wh。
- $i$：加工网格时间箱索引。
- $\widetilde P_{f,i}$：重采样功率，W。
- $\Delta t_{f,i}$：时间箱实际长度，s。
- $c_f$：航段能量守恒缩放系数。
- $P^*_{f,i}$：最终监督功率，W。

按完整flight和Multi-UAV关联组划分，种子42随机排列分组，70%/15%/15%组数经四舍五入形成183/36/37个flight。验证关联组排序后交替指定选模与校准角色，各18个flight；校准flight不参与TCN选窗、最优轮次或RLS搜索。训练使用全部183个flight、505176行；测试37个flight、105370行，未抽取训练子集。

标准化仅用训练集均值和总体标准差，常数列的标准差置1。C/D/E在10维后拼接位置3维、相对风2维的10点SMA，原始列和未经SMA的功率标签保留，SMA不跨flight。

$$s_{f,i,j}=\frac{1}{n_i}\sum_{q=0}^{n_i-1}x_{f,i-q,j},\quad n_i=\min(10,i+1)$$

其中：

- $s_{f,i,j}$：新增SMA通道。
- $x_{f,i-q,j}$：当前或历史原始输入；$j$为位置或风的特征索引。
- $n_i$：当前实际参与均值的样本数。
- $q$：历史回溯步数。

TCN窗口只包含截至目标时刻的加工特征；不足时重复本flight首行。两个残差块通道均32，膨胀率1/2，每块含两次核3的左补齐卷积、GroupNorm(1,32)、隐藏激活、Dropout(0.05)，跳连必要时1×1卷积。A/C卷积后SiLU、残差出口ReLU；B/D卷积后及出口SiLU；E四次卷积后、两次残差出口和预测头隐藏层均用LeakyReLU(0.05)。预测头Linear(32,32)→SiLU（E为LeakyReLU）→Linear(32,1)，输出标准化功率，反标准化后截到非负。参数量A/B为12001，C/D/E为12641。GroupNorm在整个已截取的历史窗统计，不能把整段一次前向与逐窗推理等同。

候选1/2/5/10s各训练5轮，候选内保留验证SmoothL1最小权重，以“1s能量WAPE+0.2×功率WAPE+0.5×航段能量WAPE”选窗。正式训练重新按种子42初始化，完整20轮后保留最小验证SmoothL1权重；无早停。AdamW学习率0.0003、权衰减0.0001，batch8192，梯度范数裁剪2。损失是标准化功率SmoothL1，未额外加入能量损失或RLS梯度。训练预算与原DJI工程不同，因此仅解释AMOVFLY内部比较。

$$\widehat E_f=\sum_i\widehat P_{f,i}\Delta t_{f,i}/3600,\quad \mathrm{WAPE}=100\frac{\sum_u|\widehat y_u-y_u|}{\sum_u|y_u|}$$

其中：

- $\widehat P_{f,i}$：非负功率预测，W。
- $\widehat E_f$：预测航段能量，Wh。
- $u$：当前评价粒度的索引（功率点、1秒箱或完整航段）。
- $y_u,\widehat y_u$：对应实际值和预测值。
- $\mathrm{WAPE}$：百分数误差，总绝对误差除总实际绝对值；不能与flight等权MAPE混用。

每flight连续5行构成名义1s箱，尾箱可不足1s，仍按实际dt积分。功率指标按采样点计权；航段WAPE按实际能量大小计权。运动状态分层用垂直速度绝对值≥0.5m/s、其余速度<0.5m/s、其余点依次标为垂直运动、悬停、水平运动。
"""
    if online:
        text += r"""

### RLS更新和预测区间

每组245候选由遗忘因子0.85/0.88/0.90/0.93/0.96/0.98/0.99、初始协方差0.05/0.1/0.25/0.5/1/2/4、反馈窗1/2/5/10/20s组合。18个选模flight最小化逐窗累计航段WAPE；18个校准flight固定区间倍率；测试参数冻结。

$$\boldsymbol\phi_k=[1,E_k^{\mathrm{TCN}}]^\top,\quad r_k=E_k^{\mathrm{true}}-\boldsymbol\phi_k^\top\boldsymbol\theta_{k-1}$$

$$\mathbf g_k=\frac{\mathbf C_{k-1}\boldsymbol\phi_k}{\lambda+\boldsymbol\phi_k^\top\mathbf C_{k-1}\boldsymbol\phi_k},\quad\boldsymbol\theta_k=\boldsymbol\theta_{k-1}+\mathbf g_kr_k,\quad\mathbf C_k=(\mathbf C_{k-1}-\mathbf g_k\boldsymbol\phi_k^\top\mathbf C_{k-1})/\lambda$$

其中：

- $k$：完整反馈窗序号。
- $E_k^{\mathrm{TCN}},E_k^{\mathrm{true}}$：基线和实际窗口能量，Wh。
- $\boldsymbol\phi_k$：窗口回归向量。
- $\boldsymbol\theta_k$：偏置（Wh/整窗）与比例参数。
- $r_k$：更新前未截断线性预测残差，Wh。
- $\lambda$：遗忘因子。
- $\mathbf C_k,\mathbf g_k$：参数矩阵和递推增益。

每flight初始化参数[0,1]、矩阵diag(初始协方差×0.01²,初始协方差)。当前窗先用旧参数预测，结束后取得实际能量再更新；不回改当前窗。偏置截到±max(实际窗能量,0.001)，比例截到[0.25,1.75]，矩阵对称化。短尾窗的偏置按真实时长比例缩放，只预测不反馈。剩余窗数同样按实际剩余秒数除名义窗长。

校正后的窗口能量按该窗TCN逐点积分占比分配回功率，基线全零时等分。这一步需要完整窗TCN基线，逐点结果是离线窗口分配，并非无等待的实时功率输出。剩余任务使用未来实测特征对应的TCN基线，属于条件回放。

残差均值按0.85旧值+0.15残差更新，方差按0.85旧方差+0.15(残差−旧均值)²更新，初始均值0、标准差0.01Wh。创新方差取残差方差、(0.25×残差均值绝对值)²、10⁻⁸的最大值。以RLS参数矩阵和剩余等效窗数构造正态近似原始半径，95%分位常数1.96。

$$\mathbf z=[h,E_{\mathrm{rem}}^{\mathrm{TCN}}]^\top,\quad a=1.96\sqrt{w\{\max(0,\mathbf z^\top\mathbf C_k\mathbf z)+\max(h,1)\}},\quad I=[\max(0,\mu-ca),\max(0,\mu+ca)]$$

其中：

- $h$：剩余真实秒数除名义反馈窗长。
- $E_{\mathrm{rem}}^{\mathrm{TCN}}$：剩余基线能量，Wh。
- $\mathbf z$：剩余任务回归向量。
- $w$：创新方差。
- $a$：原始区间半径，Wh。
- $\mu=\max(0,\mathbf z^\top\boldsymbol\theta_k)$：剩余能量预测，Wh。
- $c$：校准倍率，校准集绝对残差/原始半径的95%经验分位数，至少1。
- $I$：剩余能量预测区间；加已经消耗的真实能量得到最终任务区间。

任务倍率混合初始和反馈后的任务预测点，窗口倍率用完整窗口。二者分别校准，每组独立倍率。功率区间从窗口边界按TCN权重分配，未独立按逐点残差校准。时间相关性、有限校准航段和分布偏移不支持理论覆盖率保证。

$$\mathrm{IS}_\alpha=(U-L)+\frac{2}{\alpha}(L-y)\mathbf1(y<L)+\frac{2}{\alpha}(y-U)\mathbf1(y>U)$$

其中：

- $\alpha=0.05$：名义未覆盖概率。
- $L,U$：区间下上界，Wh。
- $y$：实际能量，Wh。
- $\mathbf1$：条件成立为1，否则0。
- $\mathrm{IS}_\alpha$：区间评分，越小越好。

任务前覆盖每flight一个点，在线汇总只取反馈后且任务未结束点，长任务贡献更多时间点；0/25/50/75%阶段各flight等权。RLS逐窗累计能量使用每窗预测和，与“实际已耗能+剩余预测”的任务轨迹不是同一个对象。
"""
    return text


def table(frame):
    """功能: 生成Markdown表格。
    参数: frame为数据表。
    返回: Markdown文本。
    调用位置: write_reports。
    """
    def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v).replace("|", "\\|")
    return "| " + " | ".join(map(str, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def write_reports(cfg, online=False):
    """功能: 从当前CSV生成稳定README和详细输出报告。
    参数: cfg为配置，online为RLS标记。
    返回: 文档路径。
    调用位置: main.py。
    """
    from config import GROUPS, group_paths
    metrics_path = cfg.summary_csv
    if not metrics_path.exists(): return {"status": "尚无汇总结果"}
    metrics = pd.read_csv(metrics_path).sort_values("group")
    summary = json.loads((cfg.data_dir / "dataset_summary_TCN-comparison.json").read_text(encoding="utf-8"))
    name = "TCN+RLS-AMOVFLY" if online else "TCN-AMOVFLY"
    feature_table = table(pd.DataFrame(FEATURES, columns=["字段", "维数", "单位", "构造"]))
    group_table = table(pd.DataFrame([{"组别": g, "输入维数": 15 if s["use_sma"] else 10, "激活": s["activation"], "SMA": "位置3维+风2维" if s["use_sma"] else "无"} for g, s in GROUPS.items()]))
    data_path = summary["output_directory"]
    source = summary["source_directory"]
    extra = ("冻结TCN，只搜索RLS遗忘因子、初始协方差、反馈窗并校准区间。" if online else "重新训练TCN，比较输入SMA和隐藏激活。")
    param = ("RLS网格为7×7×5=245候选，验证集18个flight搜索、18个独立flight校准；反馈窗结束后才更新，短尾窗不更新。" if online else "TCN窗口1/2/5/10s各训练5轮，最佳窗口重新初始化训练20轮；batch=8192，AdamW学习率0.0003，权衰减0.0001，种子42。")
    result_cols = (["group", "tcn_flight_energy_wape_percent", "flight_energy_wape_percent", "task_start_interval_coverage_percent", "post_feedback_interval_coverage_percent", "post_feedback_interval_mean_width_wh", "post_feedback_interval_score_wh"] if online else ["group", "feature_count", "window_seconds", "power_wape_percent", "window_energy_wape_percent", "flight_energy_wape_percent"])
    result = table(metrics[result_cols])
    analysis = []
    for col, label in (("power_wape_percent", "功率WAPE"), ("flight_energy_wape_percent", "任务能耗WAPE")):
        row = metrics.loc[metrics[col].idxmin()]; analysis.append(f"{label}最低为{row.group}组（{row[col]:.4f}%）。")
    for g in "ABCDE":
        if online:
            row = metrics[metrics.group == g].iloc[0]
            analysis.append(f"{g}组TCN任务能耗WAPE为{row.tcn_flight_energy_wape_percent:.4f}%，RLS后为{row.flight_energy_wape_percent:.4f}%；任务前覆盖{row.task_start_interval_coverage_percent:.2f}%，反馈后覆盖{row.post_feedback_interval_coverage_percent:.2f}%。")
    analysis.append("单种子和有限训练预算下，结果只代表当前划分。RLS反馈后的覆盖率低于名义95%时，区间不能解释为理论覆盖保证。")
    boundary = ("位置、速度和相对风来自实测航段，不等于起飞前已知计划；裁剪外起降待机不计入能耗。"
                "RLS任务前与剩余任务区间使用完整实测轨迹的TCN回归基线，是已知轨迹条件下的回放，不能解释为真实起飞前预测。"
                "线性重采样使用相邻右端点，SMA/TCN只在加工网格上使用历史，不保证原始传感器逐点严格实时。"
                "风分量使用相对风角正余弦，不是地理ENU风；route列为场景名而非实际航线。"
                "关联组隔离不等于跨日期留出，同日其他航班仍可能共享环境；仅当前随机划分和种子42，无多种子显著性结论。"
                "审计CSV记录接受文件的源哈希、重复时间、缺失行和裁剪量；JSON另列全部拒收原因。")
    readme = f"""# {name}

## 目录

- [实验与输入](#实验与输入)
- [数据与运行](#数据与运行)
- [模块与复现](#模块与复现)
- [结果与边界](#结果与边界)

## 实验与输入

使用AMOVFLY ready data的裁剪飞行状态预测功率并积分能耗。{extra}

{group_table}

{feature_table}

功率、电池、电流、电压、四元数、加速度、气压和经纬度目标列只作标签或审计，不进入模型。C/D/E的5个SMA通道按flight只用当前及历史10点计算，窗口起点按已有点数求均值。

## 数据与运行

- 原始目录：`{source}`。
- 加工目录：`{data_path}`，train/val/test共用一套CSV和SHA-256。
- 模型目录：`{cfg.model_dir}`；输出目录：`{cfg.out_dir}`。

接受{summary['accepted_flights']}个飞行，拒绝{len(summary['rejected_flights'])}个；训练/验证/测试为{summary['train_flights']}/{summary['val_flights']}/{summary['test_flights']}个flight、{summary['train_rows']}/{summary['val_rows']}/{summary['test_rows']}行。完整flight及Multi-UAV关联组按种子42划分70/15/15；验证再拆成18个选模和18个校准flight。清洗去非有限和重复时间，保留gps_z>2m首末航段，航段内断点>2s整文件拒收；0.2s线性重采样，功率按源航段梯形能量守恒校准，最后时间箱保留真实时长。

```powershell
python main.py all --device cuda
python main.py evaluate --device cuda
python main.py plot
python main.py report
python -m unittest discover -s tests -v
```

{param}

## 模块与复现

`config.py`集中路径、字段、组别和参数；`main.py`调度；`data_utils.py`加载数据并按flight生成SMA；`model.py`定义TCN；`train.py`{'加载冻结模型推理' if online else '只用训练集拟合scaler，搜索窗口并正式训练'}；{'rls.py负责反馈和区间，rls_evaluate.py负责搜索/校准/测试，visualize_rls.py绘图' if online else 'evaluate.py负责功率、1秒能量、完整航段及场景/运动状态指标，visualize.py绘图'}。报告由 [report.py](report.py) 调用 [共享生成器](../amovfly_reporting.py) 生成。

{'先在 ../TCN-AMOVFLY 执行all，生成AMOVFLY新权重；RLS没有重训接口。' if online else '首次all或prepare调用 ../prepare_amovfly.py；prepare --force-prepare重建原始加工数据，之后需重新执行全部训练。'}
模式支持 {'prepare/tune/calibrate/evaluate/plot/report/all' if online else 'prepare/tune/train/evaluate/plot/report/all'}；`--group A`指定组，`--device cpu`切换CPU；`--data-dir`、`--save-dir`、`--out-dir`覆盖路径。{'RLS固定使用config.py网格，TCN源权重路径见tcn_source_model_dir。' if online else '--tune-epochs、--epochs、--batch-size、--seed可覆盖预算与种子。'} 修改参数后使用独立模型/输出目录全量重跑，避免不同配置混合。

当前数据审计：[加工脚本](../prepare_amovfly.py)；当前算法和结果：[输出报告](out/outreadme.md)；工程历史：[improve.md](improve.md)。数据、权重和out因仓库既有存储策略保留本机，`all`可重建；未执行Git提交。

## 结果与边界

{result}

{'RLS区间为验证残差比率95%经验分位数校准；反馈后区间评分同时惩罚宽度和未覆盖。' if online else 'WAPE按样本/1秒窗口/完整flight分别统计；flight能量误差可能因正负误差抵消而低于瞬时功率误差。'}

{'\n\n'.join(analysis)}

{boundary}

README维护稳定接口；improve.md追加工程变化；out/outreadme.md记录当前算法、结果、图表和结构化产物。数据、模型和out按仓库规则保留在本机。"""
    cfg.project_root.joinpath("README.md").write_text(readme, encoding="utf-8")
    lines = [f"# {name} 当前输出报告", "", "## 目录", "",
             "- [A 数据与算法](#a-数据与算法)", "- [B 当前结果](#b-当前结果)",
             "- [C 图表与产物](#c-图表与产物)", "", "## A 数据与算法", "", feature_table, "", group_table, "", boundary, "",
             algorithm_text(online), "", "## B 当前结果", "", result, "", "\n\n".join(analysis), ""]
    if not online:
        lookup = metrics.set_index("group")
        comparisons = []
        for left, right, question in (("B", "A", "统一SiLU"), ("C", "A", "混合激活下增加SMA"),
                                      ("D", "B", "SiLU下增加SMA"), ("E", "D", "统一LeakyReLU")):
            delta = lookup.loc[left, "flight_energy_wape_percent"] - lookup.loc[right, "flight_energy_wape_percent"]
            comparisons.append(f"{left}相对{right}（{question}）任务WAPE变化{delta:+.4f}个百分点。")
        lines += ["\n\n".join(comparisons), "",
                  "A/C选择相同10s窗口、B/D相同2s窗口，可作对应SMA比较。两对结果均未支持SMA改善；E/D窗口不同，差异同时包含窗口选择。20轮预算不保证收敛，不能扩大为激活函数的普遍排序。", ""]
    else:
        lines += ["任务前覆盖只有40.54%—64.86%，反馈后覆盖88.79%—90.13%，均未达到95%。任务倍率使用初始点和大量反馈后点混合校准，初始点占比低；验证/测试残差分布可能不同，不能用总体倍率保证任务前覆盖。A反馈后区间评分最低，B任务点误差最低，二者不是同一评价目标。", ""]
    for group in metrics.group:
        paths = group_paths(cfg, group)
        tuning = pd.read_csv(paths["rls_tuning" if online else "tuning"])
        lines += [f"### {group}组验证搜索", "", f"完整候选数量：{len(tuning)}。测试标签未参与本表排序。", "",
                  table(tuning.sort_values("flight_energy_wape_percent" if online else "selection_score").head(5)), ""]
        flights = pd.read_csv(paths["flight_metrics"])
        flights["absolute_error_wh"] = (flights.predicted_energy_wh - flights.actual_energy_wh).abs()
        lines += ["测试航段绝对能量误差最大的5项：", "", table(flights.sort_values("absolute_error_wh", ascending=False).head(5)), ""]
        if online:
            lines += ["进度阶段覆盖与宽度（每flight每阶段一个点）：", "", table(pd.read_csv(paths["stage_intervals"])), ""]
        else:
            lines += ["场景分层（route是场景名）：", "", table(pd.read_csv(paths["route_metrics"])), "",
                      "运动状态：垂直速度绝对值≥0.5m/s优先为vertical_motion，其余速度<0.5m/s为hover，其余为horizontal_motion。", "",
                      table(pd.read_csv(paths["stage_metrics"])), ""]
    if cfg.runtime_json.exists():
        runtime = json.loads(cfg.runtime_json.read_text(encoding="utf-8"))
        lines += ["### 运行环境", "", f"开始时间：{runtime['started_at']}；Python {runtime['python'].split()[0]}；PyTorch {runtime['torch']}；CUDA {runtime['cuda']}；{runtime['gpu']}。"
                  "正式训练CSV每组20轮，调参逐轮CSV每组4×5=20轮。追加终端日志包含早期被中断联调，不以历史日志的试跑预算作为当前结果。", ""]
    lines += ["## C 图表与产物", ""]
    for figure in sorted(cfg.figure_dir.glob("*.svg")):
        rel = os.path.relpath(figure, cfg.out_dir).replace("\\", "/")
        key = figure.stem.replace("TCN-comparison_", "").replace("TCN_RLS_", "")
        description = FIGURE_NOTES.get(key, "来源：当前结果表；坐标和单位见图。")
        lines += [f"### {figure.stem}", "", description + " 各组精确数值见B部分及源表，仅解释当前划分。", "",
                  f"![{figure.stem}]({rel})", "", f"[PNG]({rel[:-4]}.png) / [SVG]({rel})", ""]
    lines += ["## 结构化产物", "", "下列文件与当前结果配套，逐航段、逐窗口、逐点CSV和日志均以最终运行生成文件为准："]
    artifacts = []
    for path in sorted(cfg.out_dir.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".log"}:
            artifacts.append({"文件": f"[{path.name}]({os.path.relpath(path, cfg.out_dir).replace(chr(92), '/')})", "字节": path.stat().st_size})
    lines += ["", table(pd.DataFrame(artifacts)), ""]
    cfg.report_md.write_text("\n".join(lines), encoding="utf-8")
    return {"readme": str(cfg.project_root / "README.md"), "report": str(cfg.report_md)}
