# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-09-18
# 文件名: report.py
# 功能说明: 根据实验4.1结构化结果生成逐图嵌入的当前输出报告
# 版本号：4.1

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from config import ExperimentConfig, build_config


TITLE_MAP = {
    "loss_curve_4.1": "正式TCN训练与验证损失",
    "learning_rate_schedule": "正式训练学习率变化",
    "hyperparameter_ranking": "TCN候选窗口综合排序",
    "candidate_validation_wape": "TCN候选窗口多尺度验证误差",
    "evaluation_metrics": "TCN基线与RLS校正指标对比",
    "flight_energy_actual_vs_predicted": "测试flight实际与预测总能耗",
    "flight_energy_error": "测试flight总能耗误差",
    "power_bin_mae": "真实功率分箱MAE",
    "rls_energy_window_error_comparison": "RLS反馈窗长度误差对比",
    "rls_energy_window_energy_metrics": "RLS反馈窗能耗指标",
    "rls_energy_window_count_comparison": "RLS反馈窗更新密度",
    "rls_grid_scores_4.1": "RLS参数网格验证分数",
    "planned_state_transitions": "多航线计划状态切换",
    "power_prediction_scatter": "时间步功率预测散点",
    "power_residual_histogram": "时间步功率残差分布",
    "second_energy_prediction_scatter": "1秒窗口能耗预测散点",
    "second_energy_residual_histogram": "1秒窗口能耗残差分布",
    "rls_energy_window_prediction_scatter": "RLS窗口能耗预测散点",
    "rls_energy_window_residual_histogram": "RLS窗口能耗残差分布",
    "custom_power_timeseries": "demo任务功率预测",
    "custom_cumulative_energy": "demo任务累计能耗区间",
    "rls_future_energy_correction_4.1": "未来任务能耗在线校正",
    "all_routes_trajectory_energy": "八类航线轨迹与累计能耗总览",
}


def _read_json(path: Path) -> dict:
    """功能: 读取UTF-8 JSON文件。
    参数: path为文件路径。
    返回: JSON对象。
    调用位置: generate_output_report。
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    """功能: 生成供Markdown使用的正斜杠相对路径。
    参数: path为目标文件，root为报告目录。
    返回: 以./开头的相对路径。
    调用位置: _figure_block和generate_output_report。
    """

    return "./" + Path(os.path.relpath(path, root)).as_posix()


def _figure_title(path: Path) -> str:
    """功能: 根据图形文件名生成中文标题。
    参数: path为SVG或GIF路径。
    返回: 图形标题。
    调用位置: _figure_block。
    """

    stem = path.stem
    if stem in TITLE_MAP:
        return TITLE_MAP[stem]
    flight_match = re.match(r"flight_(\d+)_(.+)", stem)
    if flight_match:
        flight_id, suffix = flight_match.groups()
        labels = {
            "power_timeseries": "功率时间序列",
            "second_energy_timeseries": "1秒能耗时间序列",
            "rls_parameter_trace": "RLS参数轨迹",
            "power_energy": "功率与累计能耗",
            "planning_trajectory": "规划轨迹动态追踪",
        }
        return f"Flight {flight_id} {labels.get(suffix, suffix.replace('_', ' '))}"
    return stem.replace("_", " ")


def _source_for(path: Path, cfg: ExperimentConfig) -> str:
    """功能: 为每类图形返回直接数据源。
    参数: path为图形路径，cfg为实验配置。
    返回: Markdown数据源说明。
    调用位置: _figure_block。
    """

    name = path.name
    if "training" in path.parts:
        source = cfg.training_log_csv if "loss" in name or "learning" in name else cfg.tcn_tuning_csv
    elif name == "planned_state_transitions.svg":
        source = cfg.out_model_dir / "planned_state_transitions_4.1.csv"
    elif "rls_grid" in name:
        source = cfg.rls_tuning_csv
    elif "rls_energy_window" in name and "future" not in name:
        source = cfg.rls_window_comparison_csv
    elif "flight_energy" in name:
        source = cfg.out_model_dir / "flight_energy_summary_4.1.csv"
    elif "power_bin" in name:
        source = cfg.out_model_dir / "power_bin_evaluation_4.1.csv"
    elif name == "evaluation_metrics.svg":
        source = cfg.evaluation_csv
    elif "prediction" in path.parts:
        source = cfg.predictions_csv
    elif "custom" in path.parts:
        source = cfg.task_output_csv
    elif "parameter_trace" in name:
        source = cfg.rls_parameter_trace_csv
    elif "future_energy_correction" in name:
        source = cfg.rls_correction_trace_csv
    elif "routes" in path.parts:
        if name == "all_routes_trajectory_energy.svg":
            source = cfg.out_route_dir / "route_products_summary_4.1.json"
        else:
            source = path.parent / name.replace("power_energy.svg", "planning_aligned.csv").replace(
                "planning_trajectory.gif", "planning_aligned.csv"
            )
    else:
        source = cfg.evaluation_csv
    return f"[`{source.name}`]({_relative(source, cfg.out_dir)})"


def _axes_for(path: Path) -> str:
    """功能: 返回图形坐标轴和编码说明。
    参数: path为图形路径。
    返回: 坐标轴说明。
    调用位置: _figure_block。
    """

    name = path.name
    if path.suffix.lower() == ".gif" and "planning_trajectory" in name:
        return "左图横轴为任务相对时间（s）、纵轴为功率（W）；右图为ENU东/北/上三维位置（m），标记随时间推进。"
    if path.suffix.lower() == ".gif":
        return "横轴为已完成的1 s窗口数；纵轴分别表示最终能耗预测/实际值（Wh）和区间宽度/绝对误差（Wh）。"
    if "trajectory_energy" in name:
        return "左图为ENU三维规划轨迹（m）；右图横轴为任务相对时间（s），纵轴为累计能耗（Wh）。"
    if "power_energy" in name:
        return "上下图横轴均为任务相对时间（s）；纵轴分别为功率（W）和累计能耗（Wh）。"
    if "loss_curve" in name:
        return "横轴为正式训练epoch，纵轴为标准化MSE；两条曲线表示训练集和验证集。"
    if "learning_rate" in name:
        return "横轴为epoch，纵轴为AdamW学习率。"
    if "candidate" in name or "hyperparameter" in name:
        return "横轴为TCN时间窗候选（s），纵轴为验证误差或综合选择分数。"
    if "state_transitions" in name:
        return "每个子图横轴为任务相对时间（s），纵轴为0/1状态；蓝线为电机状态，绿线为离地状态。"
    if "scatter" in name:
        return "横轴为实际功率或能耗，纵轴为预测值；虚线为理想一致线。"
    if "histogram" in name:
        return "横轴为预测值减实际值，纵轴为样本或窗口频数。"
    if "power_timeseries" in name or "custom_power" in name:
        return "横轴为任务相对时间（s），纵轴为功率（W），对比实际、TCN与RLS或任务前区间。"
    if "energy_timeseries" in name or "custom_cumulative" in name:
        return "横轴为任务相对时间（s），纵轴为区间或累计能耗（Wh）。"
    if "parameter_trace" in name:
        return "横轴为完整RLS更新窗口编号，纵轴分别为偏置（Wh/窗口）和缩放系数。"
    if "window" in name:
        return "横轴为RLS反馈窗长度（s），纵轴为WAPE、误差、决定系数或窗口数量，具体编码见图例。"
    if "grid" in name:
        return "横轴为遗忘因子，纵轴为初始协方差，颜色和单元格数值为验证flight能耗WAPE（%）。"
    if "flight_energy" in name:
        return "横轴为测试flight编号，纵轴为总能耗或预测误差（Wh）。"
    if "power_bin" in name:
        return "横轴为真实功率区间，纵轴为功率MAE（W）。"
    return "坐标轴按图中中文单位标注，颜色区分TCN基线、RLS修正和真实值。"


def _observation_for(path: Path, metrics: dict, task: dict, route_summary: dict) -> tuple[str, str, str]:
    """功能: 根据正式结果生成图形数值现象、分析和局限。
    参数: path为图形路径，其余参数为当前结构化结果。
    返回: 数值现象、分析、局限三元组。
    调用位置: _figure_block。
    """

    name = path.name
    if "training" in path.parts:
        return (
            "4 s窗口综合选择分数为10.434；正式训练80轮，最佳验证损失为0.031255。",
            "训练损失继续下降而验证损失后段波动，checkpoint按最低验证损失保存。",
            "候选比较只基于当前29个验证flight，窗口排序不代表其他机型或采样周期。",
        )
    if name == "planned_state_transitions.svg":
        return (
            "H及R1至R7的代表flight均出现状态切换；所有代表序列满足二值约束和离地状态不大于电机状态。",
            "显式地面、电机启动和离地阶段使低功率片段不再依赖连续位置值隐式表达。",
            "离线状态由功率与相对位移阈值标注，部署时必须改由飞控计划状态提供。",
        )
    if "routes" in path.parts and name != "all_routes_trajectory_energy.svg":
        route = next((part.removeprefix("route_") for part in path.parts if part.startswith("route_")), "")
        row = route_summary.get("routes", {}).get(route, {})
        actual = float(row.get("actual_energy_wh", 0.0))
        tcn = float(row.get("tcn_energy_wh", 0.0))
        rls = float(row.get("rls_energy_wh", 0.0))
        return (
            f"该代表flight实际/TCN/RLS总能耗为{actual:.3f}/{tcn:.3f}/{rls:.3f} Wh。",
            "轨迹几何来自4.1 ENU位置；图中的速度如出现，仅为同一flight位置差分诊断量，不是TCN输入。",
            "单条代表flight不能概括整条路线分布，三维交互细节需结合对应HTML查看。",
        )
    if name == "all_routes_trajectory_energy.svg":
        energies = [float(row["actual_energy_wh"]) for row in route_summary.get("routes", {}).values()]
        return (
            f"八条代表路线实际总能耗范围为{min(energies):.3f}至{max(energies):.3f} Wh。",
            "统一坐标与累计能耗面板用于比较路线尺度和RLS最终收敛差异。",
            "每条路线只取一条中等长度代表flight，不能替代完整分路线统计。",
        )
    if "custom" in path.parts:
        return (
            f"demo任务预测总能耗为{float(task['total_energy_wh']):.3f} Wh，95%区间为"
            f"[{float(task['total_energy_lower_wh']):.3f}, {float(task['total_energy_upper_wh']):.3f}] Wh。",
            "demo仅使用10维任务前字段，两个计划状态显式覆盖停机、启动、离地和降落。",
            "demo是接口连通性样例，不含真实观测，不能作为精度样本。",
        )
    if "rls" in path.parts or "rls_" in name:
        return (
            f"正式RLS取遗忘因子0.93、初始协方差1.0；flight能耗WAPE由"
            f"{metrics['tcn_flight_energy_wh_wape_percent']:.3f}%降至{metrics['flight_energy_wh_wape_percent']:.3f}%。",
            "完整1 s观测窗口更新偏置和缩放系数，未来路线在每次更新后重新累计。",
            f"超长任务组WAPE为{metrics['ultra_long_task_flight_energy_wape_percent']:.3f}%，"
            f"区间覆盖率仅{metrics['ultra_long_task_flight_energy_interval_coverage_percent']:.3f}%。",
        )
    return (
        f"TCN时间步功率WAPE为{metrics['tcn_sample_power_w_wape_percent']:.3f}%，RLS后为"
        f"{metrics['sample_power_w_wape_percent']:.3f}%；flight能耗WAPE为"
        f"{metrics['tcn_flight_energy_wh_wape_percent']:.3f}%和{metrics['flight_energy_wh_wape_percent']:.3f}%。",
        "图中偏差来自冻结测试集，RLS只使用窗口结束后已获得的实际能耗修正未来窗口。",
        f"名义95%的flight区间覆盖率为{metrics['flight_energy_interval_coverage_percent']:.3f}%，"
        "区间校准弱于点预测精度。",
    )


def _figure_block(index: int, path: Path, cfg: ExperimentConfig, metrics: dict,
                  task: dict, route_summary: dict) -> str:
    """功能: 生成一张SVG或GIF的完整嵌入与六项说明。
    参数: index为图号，path为文件，其余为当前结构化结果。
    返回: Markdown图块。
    调用位置: generate_output_report。
    """

    title = _figure_title(path)
    relative = _relative(path, cfg.out_dir)
    observation, analysis, limitation = _observation_for(path, metrics, task, route_summary)
    return "\n".join([
        f"### 图 C-{index} {title}",
        "",
        f"![图 C-{index} {title}]({relative})",
        "",
        f"**数据源：** {_source_for(path, cfg)}。",
        "",
        f"**坐标轴与编码：** {_axes_for(path)}",
        "",
        f"**数值现象：** {observation}",
        "",
        f"**分析：** {analysis}",
        "",
        f"**局限：** {limitation}",
        "",
    ])


def generate_output_report(cfg: ExperimentConfig) -> Path:
    """功能: 用当前4.1结构化结果重写out/outreadme.md并逐图嵌入SVG/GIF。
    参数: cfg为实验配置对象。
    返回: 报告路径。
    调用位置: visualize.generate_all_visualizations或脚本直接运行。
    """

    evaluation = _read_json(cfg.evaluation_json)
    metrics = evaluation["metrics"]
    dataset = _read_json(cfg.dataset_summary_json)
    task = _read_json(cfg.task_output_json)
    visualization = _read_json(cfg.visualization_summary_json)
    route_summary = _read_json(cfg.out_route_dir / "route_products_summary_4.1.json")
    image_paths = sorted(
        [Path(path) for path in visualization["files"] if Path(path).suffix.lower() in {".svg", ".gif"}],
        key=lambda path: (path.suffix.lower() == ".gif", path.as_posix()),
    )
    sections = [
        "# 实验 4.1 当前输出与算法报告",
        "",
        "## 目录",
        "",
        "- [A. 方法与数据](#a-方法与数据)",
        "- [B. 正式运行结果](#b-正式运行结果)",
        "- [C. 图形逐项说明](#c-图形逐项说明)",
        "- [D. 产物清单与复现](#d-产物清单与复现)",
        "",
        "## A. 方法与数据",
        "",
        "当前加工表按完整flight独立处理，TCN只接收README列出的10维任务前字段。"
        f"数据规模为{dataset['processed_rows']:,}行、{dataset['flights']}个flight，"
        f"训练/验证/测试flight为{dataset['train_flights']}/{dataset['val_flights']}/{dataset['test_flights']}。",
        "",
        "每个0.2 s区间的监督能耗按 $E_i=P_i\\Delta t/3600$ 计算。"
        "其中，$E_i$ 为区间能耗（Wh），$P_i$ 为监督功率（W），$\\Delta t=0.2$ s。"
        "原始真实时间功率先做梯形积分，重采样功率再按flight缩放；最大总能耗相对误差为"
        f"{dataset['consistency_checks']['maximum_flight_energy_error_percent']:.3e}%。",
        "",
        "TCN使用两层因果卷积。时间窗候选为0.6、1、2、3、4、5、8和10 s，"
        "选择分数同时考虑时间步功率、1 s能耗与flight能耗。RLS模型为"
        "$\\hat{E}_k=\\theta_{0,k}+\\theta_{1,k}E_{\\mathrm{TCN},k}$。"
        "其中，$\\hat{E}_k$ 为校正窗口能耗，$\\theta_{0,k}$ 为偏置，"
        "$\\theta_{1,k}$ 为缩放系数，$E_{\\mathrm{TCN},k}$ 为冻结TCN基线。",
        "",
        "## B. 正式运行结果",
        "",
        "完整CUDA运行完成8个候选各20轮和正式模型80轮。选择4 s窗口，最佳验证损失为0.031255；"
        "RLS遗忘因子为0.93，初始协方差为1.0。",
        "",
        "| 尺度 | TCN WAPE | RLS WAPE | 其他结果 |",
        "| --- | ---: | ---: | --- |",
        f"| 0.2 s功率 | {metrics['tcn_sample_power_w_wape_percent']:.3f}% | {metrics['sample_power_w_wape_percent']:.3f}% | RLS MAE {metrics['sample_power_w_mae']:.3f} W |",
        f"| 1 s窗口能耗 | {metrics['tcn_window_energy_wh_wape_percent']:.3f}% | {metrics['window_energy_wh_wape_percent']:.3f}% | RLS MAE {metrics['window_energy_wh_mae']:.5f} Wh |",
        f"| flight总能耗 | {metrics['tcn_flight_energy_wh_wape_percent']:.3f}% | {metrics['flight_energy_wh_wape_percent']:.3f}% | RLS MAE {metrics['flight_energy_wh_mae']:.3f} Wh |",
        "",
        f"名义95%区间的时间步、1 s窗口和flight覆盖率分别为"
        f"{metrics['sample_power_interval_coverage_percent']:.3f}%、"
        f"{metrics['window_energy_interval_coverage_percent']:.3f}%和"
        f"{metrics['flight_energy_interval_coverage_percent']:.3f}%。"
        f"超长任务只有{metrics['ultra_long_task_test_flights']}个测试flight，覆盖率为"
        f"{metrics['ultra_long_task_flight_energy_interval_coverage_percent']:.3f}%；"
        "该区间不能直接作为安全裕量保证。",
        "",
        "## C. 图形逐项说明",
        "",
        f"本节嵌入当前全部{visualization['svg_count']}张SVG和{visualization['gif_count']}个GIF。"
        "HTML只在D节列出，因为Markdown不能直接嵌入交互页面。",
        "",
    ]
    for index, path in enumerate(image_paths, start=1):
        sections.append(_figure_block(index, path, cfg, metrics, task, route_summary))
    sections.extend([
        "## D. 产物清单与复现",
        "",
        f"当前可视化汇总为{visualization['svg_count']}张SVG、{visualization['gif_count']}个GIF和"
        f"{visualization['html_count']}个HTML。完整机器可读清单见"
        f"[`{cfg.visualization_summary_json.name}`]({_relative(cfg.visualization_summary_json, cfg.out_dir)})。",
        "",
        "关键结构化结果：",
        "",
        f"- [`{cfg.evaluation_json.name}`]({_relative(cfg.evaluation_json, cfg.out_dir)})：正式测试指标和覆盖率。",
        f"- [`{cfg.tcn_tuning_csv.name}`]({_relative(cfg.tcn_tuning_csv, cfg.out_dir)})：8个TCN窗口。",
        f"- [`{cfg.rls_tuning_csv.name}`]({_relative(cfg.rls_tuning_csv, cfg.out_dir)})：49组RLS参数。",
        f"- [`planned_state_transitions_4.1.csv`]({_relative(cfg.out_model_dir / 'planned_state_transitions_4.1.csv', cfg.out_dir)})：多航线状态切换与约束检查。",
        f"- [`{cfg.terminal_log_file.name}`]({_relative(cfg.terminal_log_file, cfg.out_dir)})：完整GPU运行日志。",
        "",
        "复现命令：",
        "",
        "```powershell",
        "python 4.1/main.py all --device cuda --force-prepare",
        "```",
        "",
        "该命令从Source重建4.1加工数据，并覆盖4.1模型与输出；不读取其他版本的加工数据、权重或指标。",
        "",
    ])
    report_path = cfg.out_dir / "outreadme.md"
    report_path.write_text("\n".join(sections), encoding="utf-8", newline="")
    return report_path


if __name__ == "__main__":
    print(generate_output_report(build_config()))
