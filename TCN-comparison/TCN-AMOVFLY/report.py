# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-11-01
# 文件名: report.py
# 功能说明: 从当前实测结果生成五组参数、误差和逐图说明报告
# 版本号：1.0

from __future__ import annotations

import json
import os
import pandas as pd

from config import GROUPS, RAW_FEATURE_COLUMNS, SCHEMA_VERSION, group_paths


def table(frame):
    """功能: 将结果表转换为Markdown表格。
    参数: frame为结构化数据。
    返回: Markdown表格文本。
    调用位置: write_reports。
    """
    def format_value(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    return "| " + " | ".join(map(str, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(format_value(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))


def write_reports(cfg):
    """功能: 根据当前加工摘要和五组CSV生成一致的实验说明。
    参数: cfg为集中路径和超参数。
    返回: 入口说明与输出报告路径。
    调用位置: main.py。
    """
    if not cfg.summary_csv.exists():
        return {"status": "尚无汇总结果"}
    metrics = pd.read_csv(cfg.summary_csv).sort_values("group")
    if not metrics.schema_version.eq(SCHEMA_VERSION).all():
        raise ValueError("汇总表混入旧特征契约")
    summary = json.loads(cfg.manifest_json.read_text(encoding="utf-8"))
    features = table(pd.DataFrame([
        ("position_x_m, position_y_m, position_z_m", 3, "m", "完整记录相对首个有效点的三维实测位置"),
        ("wind_speed", 1, "m/s", "相对风正交分量模长"), ("wind_angle", 1, "rad", "atan2重建，[-π,π]"),
        ("time_s, task_duration_s", 2, "s", "已过时间和完整记录时长"), ("payload_g", 1, "g", "文件名P参数"),
        ("uav_type", 1, "类别编码", "Y=1，R=2，G=3，按普通数值列标准化"),
        ("planned", 1, "状态编码", "0停机，1原地启动，2飞行；DJI同阈值历史标注"),
    ], columns=["字段", "维数", "单位", "构造"]))
    groups = table(pd.DataFrame([
        {"组别": group, "输入维数": 15 if spec["use_sma"] else 10, "激活": spec["activation"],
         "SMA": "三维位置、风速、风向共5维" if spec["use_sma"] else "无"} for group, spec in GROUPS.items()]))
    counts = table(pd.DataFrame([
        {"划分": split, **values, "停机行": summary["state_counts"][split]["0"],
         "原地启动行": summary["state_counts"][split]["1"], "飞行行": summary["state_counts"][split]["2"]}
        for split, values in summary["counts"].items()]))
    roles = summary["validation_roles"]
    preprocessing = f"""从 `{cfg.source_processed_dir}` 的FAFS/FAVS/VAFS/VAVS/Random同名ready子目录读取，不重复读取raw_data。本次{summary['source_files']}个CSV接受{summary['accepted_flights']}个，拒收{len(summary['rejected_flights'])}个。必要列去非有限行，时间稳定排序去重，至少20个有效点；完整记录内任意采样缺口>2秒则整文件拒收。源哈希、拒收原因、清洗量见加工目录amovfly_flight_audit.csv和摘要JSON。

保留完整有效时间范围，不再按gps_z>2m裁掉两端；位置减首个有效点。原始非负battery_voltage乘battery_current形成源功率，用真实时间梯形积分取得能量。风角先转换正余弦分量再插值，连续量重采样至{cfg.resample_seconds}秒网格，末箱使用实际剩余时长；监督功率整体缩放到与源能量守恒，逐箱能量为功率乘实际秒数再除3600。全零记录保留零值并避免除零。

状态与DJI 4.1/data_utils.py一致：功率≥100W的首末证据之间连续标注motor_on；相对起点水平位移≥1m或垂直位移绝对值≥0.5m的首末证据之间连续标注airborne，再令airborne乘motor_on。两列分别线性插值后np.rint取整，限制airborne≤motor_on，planned为两列之和。因此0=(0,0)、1=(1,0)、2=(1,1)；无证据时相应列全零。二进制列只作审计，不额外进入TCN，不直接插值三级标签。

planned含功率阈值信息，首末证据填充使用完整记录，是指定的历史代理状态；不能解释为无需真实功率的起飞前精度或严格在线状态估计。部署时需由飞控或计划提供状态。GPS漂移可能提前触发位移阈值，本实验不另调阈值。uav_type以Y/R/G=1/2/3作为单一数值输入，不使用one-hot或embedding。

按完整flight和Multi-UAV关联组隔离，种子{cfg.random_seed}将关联组随机分为70/15/15；验证关联组交替分角色，{len(roles['val'])}个flight选窗/选最佳轮次，{len(roles['calibration'])}个校准flight不参与本目录选模。完整记录和拒收范围改变，数据划分及任务能量定义随之变化，不与旧裁剪实验直接计算改善率。

{counts}"""
    algorithm = f"""输入顺序：`{', '.join(RAW_FEATURE_COLUMNS)}`。A/B使用10维；C/D/E在末尾追加三维位置、风速、风向的10点因果SMA，合计15维；不跨flight，起点不足10点按已有点数平均。风向SMA平均正弦/余弦再atan2重建，仍保留[-π,π]切口。机型和planned不计算SMA。

标准化仅用全部训练行的均值、总体标准差，常数列标准差置1，目标功率单独缩放。历史窗口不足时重复本flight首行。预处理线性插值使用右端点，不表示原始传感器严格实时。

两个TCN残差块通道{cfg.tcn_channels}，核{cfg.tcn_kernel_size}，膨胀率1/2；每块两次左补齐因果Conv1d，各接GroupNorm(1,32)、激活和Dropout({cfg.tcn_dropout})，必要时1×1跳连。A/C卷积后SiLU、残差出口ReLU；B/D均SiLU；E全部隐藏激活LeakyReLU(0.05)。取末端编码，接Linear(32,32)、隐藏激活、Linear(32,1)，反标准化后裁到非负。A/B参数12001，C/D/E参数12641。GroupNorm统计整个历史窗口的通道和时间，整段一次推理不等于逐窗口推理。

窗口{cfg.tcn_window_seconds}秒各完整训练{cfg.tune_epochs}轮，每候选重置种子并保留最低验证SmoothL1权重，以“1秒能量WAPE+0.2×功率WAPE+0.5×任务能量WAPE”选窗。重新初始化后正式完整训练{cfg.epochs}轮，无早停，保存最低验证SmoothL1权重。AdamW学习率{cfg.learning_rate}、权衰减{cfg.weight_decay}、batch={cfg.batch_size}，梯度范数裁剪2；无额外能量损失或RLS梯度。

评价冻结权重/scaler，核对列顺序、组别、SMA、激活、schema与manifest SHA-256。每flight连续5行合为名义1秒箱，短尾按实际dt积分；全flight积分为任务能量。WAPE为绝对误差之和除真实绝对值之和乘100%，整任务按实际能量大小加权。功率误差符号可以在积分中抵消，局部与全程排序可能不同。另存flight等权MAPE、绝对误差90%分位及逐场景/状态指标。

运动分层：垂直速度绝对值≥0.5m/s优先vertical_motion，其余速度<0.5m/s为hover，其余horizontal_motion；另按planned给出motor_off/ground_motor_on/airborne。两套分层会重叠，不可相加。

本实验仅TCN，实测位置和相对风不等于独立计划；route为场景名；单种子42，无多种子或跨日期显著性结论。原TCN+RLS-AMOVFLY使用不同数据和特征契约，不能直接复用本次结果。"""
    result = table(metrics[["group", "feature_count", "window_seconds", "power_wape_percent", "window_energy_wape_percent", "flight_energy_wape_percent"]])
    power_best = metrics.loc[metrics.power_wape_percent.idxmin()]
    flight_best = metrics.loc[metrics.flight_energy_wape_percent.idxmin()]
    conclusion = f"本次功率WAPE最低为{power_best.group}组（{power_best.power_wape_percent:.4f}%）；任务能耗WAPE最低为{flight_best.group}组（{flight_best.flight_energy_wape_percent:.4f}%）。各组独立选窗，仅代表当前种子与划分。"
    readme = f"""# TCN-AMOVFLY：planned状态五组实验

## 目录

- [输入和实验](#输入和实验)
- [数据与状态](#数据与状态)
- [训练与运行](#训练与运行)
- [当前结果](#当前结果)
- [文件与存储](#文件与存储)

## 输入和实验

{features}

{groups}

## 数据与状态

{preprocessing}

加工目录：`{cfg.data_dir}`。契约：`{SCHEMA_VERSION}`。

## 训练与运行

通道{cfg.tcn_channels}，Dropout={cfg.tcn_dropout}，权重衰减={cfg.weight_decay}，batch={cfg.batch_size}；窗口{cfg.tcn_window_seconds}秒各完整{cfg.tune_epochs}轮，正式完整{cfg.epochs}轮，无早停，种子{cfg.random_seed}，学习率{cfg.learning_rate}。

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

{result}

{conclusion}

## 文件与存储

`config.py`集中管理路径、状态阈值和超参数；`prepare_data.py`独立读取sourse生成完整记录、机型和状态；`data_utils.py`核验加工表和计算SMA；`model.py`定义TCN；`train.py`标准化、选窗和正式训练；`evaluate.py`输出逐点/窗口/任务/分层指标；`visualize.py`生成9套PNG/SVG；`report.py`根据实际产物生成报告；`audit.py`核验源数据、划分、权重和指标。

权重/scaler：`{cfg.model_dir}`。输出：`{cfg.out_dir}`；完整算法与结果见 [out/outreadme.md](out/outreadme.md)。旧输出归档于 `D:/Python-files/Energy-prediction/model/TCN-AMOVFLY-history/20261102-before-planned`，旧权重仍在原TCN-AMOVFLY目录。D盘数据、模型和out按仓库规则不纳入Git，可通过all重建。

README维护当前入口及结果摘要；[improve.md](improve.md)追加历史；out/outreadme.md随输出同步。
"""
    cfg.readme_md.write_text(readme, encoding="utf-8")
    lines = ["# TCN-AMOVFLY planned当前输出报告", "", "## 目录", "", "- [A 数据与算法](#a-数据与算法)",
             "- [B 当前结果](#b-当前结果)", "- [C 图表与产物](#c-图表与产物)", "", "## A 数据与算法", "", preprocessing,
             "", features, "", groups, "", algorithm, "", "## B 当前结果", "", result, "", conclusion, "",
             table(metrics[["group", "epochs", "best_epoch", "tune_epochs", "parameter_count", "seed", "training_seconds"]]), ""]
    for group in metrics.group:
        paths = group_paths(cfg, group)
        lines += [f"### {group}组", "", "全部验证窗口：", "", table(pd.read_csv(paths["tuning"])), "",
                  "任务绝对误差最大5项：", "", table(pd.read_csv(paths["flight_metrics"]).sort_values("absolute_error_wh", ascending=False).head(5)), "",
                  "场景分层：", "", table(pd.read_csv(paths["route_metrics"])), "", "运动/电机状态分层：", "", table(pd.read_csv(paths["stage_metrics"])), ""]
    if cfg.runtime_json.exists():
        runtime = json.loads(cfg.runtime_json.read_text(encoding="utf-8"))
        lines += ["### 运行环境", "", f"开始{runtime['started_at']}；Python {runtime['python'].split()[0]}；PyTorch {runtime['torch']}；CUDA {runtime['cuda']}；{runtime['gpu']}。", ""]
    notes = {
        "power_wape": ("汇总CSV", "横轴A-E，纵轴功率WAPE(%)", "power_wape_percent"),
        "window_energy_wape": ("汇总CSV与window_energy.csv", "横轴A-E，纵轴1秒箱能耗WAPE(%)", "window_energy_wape_percent"),
        "flight_energy_wape": ("汇总CSV与flight_energy.csv", "横轴A-E，纵轴任务能耗WAPE(%)", "flight_energy_wape_percent"),
        "representative_power": ("test_predictions.csv", "横轴时间(s)，纵轴功率(W)，黑线真实，彩线A-E", None),
        "representative_energy": ("test_predictions.csv", "横轴时间(s)，纵轴累计能量(Wh)，黑线真实，彩线A-E", None),
        "flight_energy_scatter": ("flight_energy.csv", "左图实际/预测能量(Wh)及等值虚线；右图flight/有符号误差(Wh)", None),
        "flight_error_distribution": ("flight_energy.csv", "横轴A-E，纵轴任务绝对误差(Wh)，箱体四分位，标记均值/离群点", None),
        "window_search": ("window_tuning.csv", "横轴窗口秒数，纵轴验证选择分数，彩线A-E", None),
        "training_loss": ("training.csv", "横轴epoch，纵轴标准化SmoothL1，实线训练，虚线验证", None),
    }
    lines += ["## C 图表与产物", ""]
    for figure in sorted(cfg.figure_dir.glob("*.svg")):
        key = figure.stem.removeprefix("TCN-comparison_")
        source, axes, column = notes[key]
        rel = figure.relative_to(cfg.out_dir).as_posix()
        values = "；".join(f"{row.group}={getattr(row, column):.4f}%" for row in metrics.itertuples()) if column else "精确指标与选窗见B部分；代表flight按测试时长中位选择，不按误差挑选"
        lines += [f"### {key}", "", f"来源：{source}；{axes}。{values}。仅解释当前划分。", "",
                  f"![{key}]({rel})", "", f"[PNG]({rel[:-4]}.png) / [SVG]({rel})", ""]
    artifacts = [{"文件": f"[{path.name}]({os.path.relpath(path, cfg.out_dir).replace(chr(92), '/')})", "字节": path.stat().st_size}
                 for path in sorted(cfg.out_dir.rglob("*")) if path.is_file() and path.suffix in {".csv", ".json", ".log"}]
    lines += ["### 结构化产物", "", table(pd.DataFrame(artifacts)), ""]
    cfg.report_md.write_text("\n".join(lines), encoding="utf-8")
    return {"readme": str(cfg.readme_md), "report": str(cfg.report_md)}
