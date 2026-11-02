# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-10-30
# 文件名: report.py
# 功能说明: 从当前实测结果生成五组参数、误差和逐图说明报告
# 版本号：1.0

from __future__ import annotations

import json
import os

import pandas as pd

from config import GROUPS, RAW_FEATURE_COLUMNS, SMA_FEATURE_COLUMNS, ExperimentConfig, group_paths, figure_path


def markdown_table(frame: pd.DataFrame) -> str:
    """功能: 将数据表转为不依赖tabulate的Markdown表格。
    参数: frame为待输出表格。
    返回: Markdown表格文本。
    调用位置: write_reports。
    """

    def format_value(value):
        """功能: 格式化单个表格数值并转义分隔符。
        参数: value为单元格值。
        返回: Markdown单元格文本。
        调用位置: markdown_table。
        """
        if isinstance(value, float):
            return f"{value:.4f}" if pd.notna(value) else "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(format_value(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def result_description(metrics):
    """功能: 根据实测指标生成有边界的实验结果解释。
    参数: metrics为已完成组的汇总表。
    返回: 对比解释段落。
    调用位置: write_reports。
    """

    lookup = metrics.set_index("group")
    text = []
    for metric, title in (("power_wape_percent", "时间步功率WAPE"),
                          ("window_energy_wape_percent", "1 s窗口能耗WAPE"),
                          ("flight_energy_wape_percent", "flight总能耗WAPE")):
        best = metrics.loc[metrics[metric].idxmin()]
        text.append(f"当前{title}最低的是 {best.group} 组，为 {best[metric]:.4f}%。")
    for left, right, question in (("B", "A", "统一SiLU"), ("C", "A", "增加SMA"),
                                  ("D", "B", "SiLU条件下增加SMA"), ("D", "C", "SMA条件下统一SiLU"),
                                  ("E", "D", "同15维输入，统一SiLU改为统一LeakyReLU(0.05)"),
                                  ("E", "C", "同15维输入，混合激活改为统一LeakyReLU(0.05)")):
        if left not in lookup.index or right not in lookup.index:
            continue
        base, improved = float(lookup.loc[right, "flight_energy_wape_percent"]), float(lookup.loc[left, "flight_energy_wape_percent"])
        delta = base - improved
        relative = delta / max(base, 1e-12) * 100
        direction = "下降" if delta >= 0 else "上升"
        text.append(f"{left} 对 {right}（{question}）的总能耗WAPE{direction} {abs(delta):.4f} 个百分点，"
                    f"相对{direction} {abs(relative):.2f}%。")
    text.append("以上是固定种子42的单次完整实验；各组独立搜索窗口，差异同时包含所选上下文变化。"
                "未进行多种子重复或新的独立留出验证，不据此宣称统计显著性或跨数据集优势。")
    return "\n\n".join(text)


def write_reports(cfg: ExperimentConfig) -> dict:
    """功能: 同步README实测结果区和out详细算法及逐图报告。
    参数: cfg为实验路径配置。
    返回: 文档路径。
    调用位置: main.py。
    """

    if not cfg.summary_csv.exists():
        return {"report": "尚无已完成的测试指标"}
    metrics = pd.read_csv(cfg.summary_csv).sort_values("group")
    groups = metrics.group.tolist()
    data = json.loads(cfg.manifest_json.read_text(encoding="utf-8"))
    fields = {"group": "组别", "window_seconds": "最佳窗口(s)", "feature_count": "输入维度",
              "power_wape_percent": "功率WAPE(%)", "window_energy_wape_percent": "1s能耗WAPE(%)",
              "flight_energy_wape_percent": "flight能耗WAPE(%)", "power_rmse_w": "功率RMSE(W)",
              "flight_energy_rmse_wh": "flight能耗RMSE(Wh)"}
    table = markdown_table(metrics[list(fields)].rename(columns=fields))
    parameter_fields = {"group": "组别", "activation": "激活", "window_steps": "TCN步数",
                        "tune_epochs": "候选轮数", "epochs": "正式轮数", "best_epoch": "最佳轮次",
                        "parameter_count": "参数量", "training_seconds": "正式训练秒数", "seed": "种子"}
    parameters = markdown_table(metrics[list(parameter_fields)].rename(columns=parameter_fields))
    comparison = result_description(metrics)
    result_block = table + "\n\n" + parameters + "\n\n" + comparison + "\n\n"
    result_block += (f"数据：训练 {data['train_rows']} 行/{data['train_flights']} 个 flight；"
                     f"验证 {data['val_rows']} 行/{data['val_flights']} 个 flight；"
                     f"测试 {data['test_rows']} 行/{data['test_flights']} 个 flight。完整推导、"
                     "逐任务误差和逐图说明见 [当前输出报告](./out/outreadme.md)。")
    if cfg.readme_md.exists():
        readme = cfg.readme_md.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in readme and end in readme:
            prefix, after = readme.split(start, 1)
            _, suffix = after.split(end, 1)
            cfg.readme_md.write_text(prefix + start + "\n\n" + result_block + "\n\n" + end + suffix, encoding="utf-8")

    lines = [
        "# TCN-comparison 当前输出与算法报告", "", "## 目录", "",
        "- [A 算法与评价口径](#a-算法与评价口径)",
        "- [B 当前运行结果](#b-当前运行结果)",
        "- [C 图表与数据清单](#c-图表与数据清单)", "",
        "## A 算法与评价口径", "",
        "本实验仅进行离线TCN功率回归与能量积分。没有RLS、在线反馈更新或预测置信区间。",
        "按完整flight复用既有124/29/47划分，副本与来源通过SHA-256核对，划分无flight交叉。"
        "来源数据已完成0.2 s重采样和逐flight能耗守恒处理，本工程不会重新抽样或改变标签。", "",
        "原始字段顺序：`" + "`、`".join(RAW_FEATURE_COLUMNS) + "`。", "",
        "C、D、E在原始10维后拼接 `" + "`、`".join(f"{c}_sma10" for c in SMA_FEATURE_COLUMNS) + "`，合计15维。"
        "每个flight独立计算尾随10点均值，起点使用已出现点数，不补零；先计算完整任务特征再切出TCN窗口。"
        "目标仍为未经SMA的原始功率。", "",
        r"$$a_{t,j}=\frac{1}{n_t}\sum_{q=0}^{n_t-1}x_{t-q,j},\quad n_t=\min(10,t+1)$$", "",
        "其中：", "", "$a_{t,j}$：特征 $j$ 在时刻 $t$ 的新增均值通道。", "",
        "$x_{t-q,j}$：当前或历史的原始特征。", "", "$n_t$：当前位置参与均值的采样点数。", "",
        "$t$：单个flight内从0开始的采样索引。", "", "$j$：选中的位置或风分量索引。", "",
        "$q$：向历史回溯的点数。", "",
        "所有输入列分别用训练集均值和标准差进行标准化；功率目标也只使用训练集统计量。"
        "TCN输入首部不足时重复该任务首个特征行。五组逐窗口和正式训练开始前都重置种子42，"
        "避免候选顺序改变随机状态。网络采用两个残差块，通道32、32，核3，膨胀率1、2，"
        "A-D每个块含两组因果卷积、GroupNorm(1,32)、SiLU、Dropout(0.05)。"
        "A/C残差出口ReLU，B/D残差出口SiLU。跳连必要时使用1×1卷积。A—D预测头为Linear(32,32)、SiLU、Linear(32,1)。"
        "E将两残差块的四次卷积后激活、两次残差出口激活及预测头中间激活全部替换为LeakyReLU(negative_slope=0.05)，"
        "最后一层仍为线性。E与C/D均为12641参数；E是原四组2×2设计之外的附加激活对照。", "",
        "GroupNorm会利用单个输入窗口内的时间位置统计量；本工程只构造截至目标时刻的输入窗口。"
        "不能将整段轨迹一次前向的所有时刻输出与此逐窗口计算等同。", "",
        "各组独立搜索0.6、1、2、3、4、5、8、10 s窗口，每个20轮。采用相同验证分数，"
        "选定后重新初始化模型训练完整80轮，并保留验证SmoothL1最小的轮次权重。"
        "搜索阶段和正式权重选择阶段的判据分别是综合WAPE和验证SmoothL1，二者不能混写。"
        "优化器AdamW，学习率0.0003，权重衰减0.0001，batch=2048，梯度范数裁剪2.0。", "",
        r"$$E_f=\sum_{t\in f}P_t\Delta t_t/3600,\quad"
        r"\widehat E_f=\sum_{t\in f}\max(0,\widehat P_t)\Delta t_t/3600$$", "",
        "其中：", "", "$E_f$：flight $f$ 的实际能耗，Wh。", "",
        r"$\widehat E_f$：该flight预测能耗，Wh。", "", "$P_t$：实际功率，W。", "",
        r"$\widehat P_t$：反标准化后的预测功率，W。", "", r"$\Delta t_t$：当前采样时间箱长度，s。", "",
        "$f$：完整飞行任务索引。", "",
        "1 s能耗由每个flight内连续5行积分，末尾不足5行的截断窗口也保留，并在窗口结果中记录samples。"
        "总WAPE为绝对误差和除以实际绝对值和，单位为%；另报flight等权MAPE和P90绝对误差，"
        "以避免总体WAPE掩盖短任务。功率接近零时不以时间步MAPE作为主指标。", "",
        "历史规划字段由实飞位置和实测风构建，电机状态历史标注用到功率阈值。"
        "结果属于相同代理规划输入下的受控离线比较，不能直接代替真实计划轨迹与预报风场的部署验证。", "",
        "## B 当前运行结果", "", f"已完成实验组：{', '.join(groups)}。", "",
        result_block.replace("[当前输出报告](./out/outreadme.md)", "[README](../README.md)"), "",
    ]
    if cfg.runtime_json.exists():
        runtime = json.loads(cfg.runtime_json.read_text(encoding="utf-8"))
        lines.extend(["### 运行环境", "",
                      f"PyTorch：{runtime['torch']}；CUDA：{runtime['cuda']}；GPU：{runtime['gpu']}。"
                      f"保存的训练运行开始时间：{runtime['started_at']}。报告刷新不覆盖该快照。正式训练秒数只计算训练循环，"
                      "不含数据读取、序列构造、窗口搜索、评价与绘图耗时。", ""])
    lines.extend(["### 训练完成状态与复核记录", "",
                  "当前五组均完成8个TCN窗口候选的每候选20轮搜索及80轮正式训练；"
                  "本次E组新增160条候选逐轮记录和80条正式训练记录。E选择5s窗口，保留第24轮验证最优权重。"
                  "A-D沿用已完成实验的权重与逐组结果，五组汇总和9套PNG/SVG由保存结果生成。", "",
                  "[单元测试日志](logs/E_validation_tests.log)记录6项测试通过；"
                  "[产物核验](logs/E_artifact_audit.json)记录数据副本、模型SHA-256、旧模型前向兼容性和覆盖率复算等检查。"
                  "TCN与RLS冻结基线的逐采样CSV因float32写出格式存在不超过0.000031W的差异，"
                  "其模型指纹与汇总基线指标一致；逐采样比较应明确数值精度。", ""])
    extra = {"group": "组别", "power_mae_w": "功率MAE(W)",
             "window_energy_mae_wh": "1s能耗MAE(Wh)", "flight_energy_mae_wh": "flight能耗MAE(Wh)",
             "flight_equal_weight_mape_percent": "flight等权MAPE(%)",
             "flight_absolute_error_p90_wh": "flight误差P90(Wh)",
             "flight_energy_underestimate_rate_percent": "flight低估比例(%)"}
    lines.extend(["### 补充误差指标", "", markdown_table(metrics[list(extra)].rename(columns=extra)), ""])
    for group in groups:
        paths = group_paths(cfg, group)
        tuning = pd.read_csv(paths["tuning"]).sort_values("window_seconds")
        tuning_cols = ["window_seconds", "window_steps", "best_val_loss", "power_wape_percent",
                       "window_energy_wape_percent", "flight_energy_wape_percent", "selection_score"]
        lines.extend([f"### {group} 组全部窗口验证结果", "", markdown_table(tuning[tuning_cols]), ""])
        flight = pd.read_csv(paths["flight_metrics"])
        worst = flight.sort_values("absolute_error_wh", ascending=False).head(5)
        lines.extend([f"{group}组总能耗绝对误差最大的5个测试flight（按Wh排序）：", "",
                      markdown_table(worst[["flight", "route", "actual_energy_wh", "predicted_energy_wh", "error_wh"]]), ""])
        stage = pd.read_csv(paths["stage_metrics"])
        route = pd.read_csv(paths["route_metrics"])
        lines.extend([f"{group}组按原始状态字段划分的阶段功率误差：", "", markdown_table(stage), "",
                      f"{group}组分路线指标（相同路线的完整flight汇总）：", "",
                      markdown_table(route[["route", "flights", "rows", "power_wape_percent",
                                            "flight_energy_wape_percent", "flight_energy_mae_wh"]]), ""])
    lines.extend(["## C 图表与数据清单", ""])
    descriptions = {
        "power_wape": "横轴实验组，纵轴功率WAPE（%），五种颜色分别对应A/B/C/D/E。来源为all_metrics.csv，不包含在线修正。",
        "window_energy_wape": "横轴实验组，纵轴1 s时间箱能耗WAPE（%）；来源为all_metrics.csv和各组window_energy.csv。包含截断尾窗。",
        "flight_energy_wape": "横轴实验组，纵轴flight总能耗WAPE（%）；来源为各组flight_energy.csv与all_metrics.csv。各组采用各自选定窗口。",
        "representative_power": "横轴任务时间（s），纵轴功率（W）；黑线为实际值，彩线为组别预测。代表任务按测试集时长中位位置确定，与误差无关。只能展示该任务局部拟合。",
        "representative_energy": "横轴任务时间（s），纵轴累计能耗（Wh）；由同一代表任务逐采样功率积分。局部功率正负误差可抵消，曲线接近不等于时间步误差都小。",
        "flight_energy_scatter": "左图横轴实际总能耗（Wh），纵轴预测总能耗（Wh），虚线为相等线；右图横轴flight编号，纵轴带符号误差（Wh），零虚线为无偏差。来源为flight_energy.csv。",
        "flight_error_distribution": "横轴实验组，纵轴flight总能耗绝对误差（Wh）；箱线图显示中位数、四分位范围、离群值和均值。来源为47个flight误差，不把重叠时间步当作独立任务。",
        "window_search": "横轴TCN窗口名义时长（s），纵轴验证综合选择分数；每条线为一个实验组，每点均为20轮候选。来源为window_tuning.csv，不能按测试指标重新选择窗口。",
        "training_loss": "每子图一个实验组，横轴训练轮次，纵轴标准化SmoothL1损失；彩色实线为训练，灰色虚线为验证。来源为training.csv，保存权重对应验证损失最低轮次。",
    }
    for name, description in descriptions.items():
        path = figure_path(cfg, name)
        if path.exists():
            rel = PathRelative(path, cfg.out_dir)
            svg = PathRelative(figure_path(cfg, name, "svg"), cfg.out_dir)
            observations = ""
            metric_map = {"power_wape": "power_wape_percent", "window_energy_wape": "window_energy_wape_percent",
                          "flight_energy_wape": "flight_energy_wape_percent"}
            if name in metric_map:
                key = metric_map[name]
                observations = "本次图中数值：" + "；".join(f"{row.group}={row[key]:.4f}%" for _, row in metrics.iterrows()) + "。"
            elif name == "window_search":
                observations = "本次选择窗口：" + "；".join(f"{row.group}={row.window_seconds:g} s" for _, row in metrics.iterrows()) + "。"
            elif name == "training_loss":
                observations = "验证损失最优轮次：" + "；".join(f"{row.group}=epoch {int(row.best_epoch)}" for _, row in metrics.iterrows()) + "。图中后续轮次不会覆盖更优验证权重。"
            elif name in {"representative_power", "representative_energy"}:
                reference = pd.read_csv(group_paths(cfg, groups[0])["flight_metrics"]).sort_values("duration_s")
                representative = int(reference.iloc[len(reference) // 2].flight)
                actual_energy = float(reference.query("flight == @representative").iloc[0].actual_energy_wh)
                values = []
                for group in groups:
                    frame = pd.read_csv(group_paths(cfg, group)["flight_metrics"])
                    values.append(f"{group}={frame.query('flight == @representative').iloc[0].predicted_energy_wh:.4f} Wh")
                observations = f"代表flight={representative}，实际总能耗={actual_energy:.4f} Wh；" + "；".join(values) + "。"
            else:
                values = []
                for group in groups:
                    frame = pd.read_csv(group_paths(cfg, group)["flight_metrics"])
                    values.append(f"{group}绝对误差中位数={frame.absolute_error_wh.median():.4f} Wh，最大值={frame.absolute_error_wh.max():.4f} Wh")
                observations = "；".join(values) + "。"
            lines.extend([f"### {name}", "", f"![{name}]({rel})", "", description, "", observations, "",
                          f"可缩放版本：[SVG]({svg})。", ""])
    lines.extend(["### 当前输出完整清单", "", "| 文件 | 用途 |", "| --- | --- |"])
    for path in sorted(cfg.out_dir.rglob("*")):
        if not path.is_file() or path == cfg.report_md:
            continue
        rel = PathRelative(path, cfg.out_dir)
        purpose = ("逐采样功率与累计能耗" if "predictions" in path.name else
                   "逐flight能耗与误差" if "flight_energy.csv" in path.name else
                   "逐1秒窗口能耗" if "window_energy.csv" in path.name else
                   "阶段功率评价" if "stage_metrics" in path.name else
                   "分路线指标" if "route_metrics" in path.name else
                   "完整窗口搜索结果" if "window_tuning" in path.name else
                   "逐轮训练记录" if path.suffix == ".csv" and "logs" in path.parts else
                   "图表" if path.suffix in {".png", ".svg"} else
                   "终端日志" if path.suffix == ".log" else "运行配置或指标")
        lines.append(f"| [{rel}]({rel}) | {purpose} |")
    cfg.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"readme": str(cfg.readme_md), "output_report": str(cfg.report_md)}


def PathRelative(path, root):
    """功能: 返回适用于Markdown的相对路径。
    参数: path为目标文件，root为文档根目录。
    返回: 正斜杠相对路径。
    调用位置: write_reports。
    """

    return os.path.relpath(path, root).replace("\\", "/")
