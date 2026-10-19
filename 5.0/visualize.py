# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: visualize.py
# 开发时间: 2026-09-16
# 功能说明: 生成实验4.1训练、评估、预测、在线RLS修正和路线追踪可视化产物
# 版本号：4.1

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go

from config import ExperimentConfig, ensure_directories


MICROSOFT_YAHEI_PATH = Path("C:/Windows/Fonts/msyh.ttc")
COLORS = {
    "actual": "#2f2f2f",
    "tcn": "#4c78a8",
    "rls": "#e45756",
    "interval": "#72b7b2",
    "accent": "#f28e2b",
    "green": "#59a14f",
    "purple": "#b279a2",
    "grid": "#d9d9d9",
}


def _style() -> str:
    """注册微软雅黑并设置适用于SVG、GIF的统一中文绘图样式。"""

    if MICROSOFT_YAHEI_PATH.exists():
        font_manager.fontManager.addfont(str(MICROSOFT_YAHEI_PATH))
        font_name = font_manager.FontProperties(fname=str(MICROSOFT_YAHEI_PATH)).get_name()
    else:
        font_name = "Microsoft YaHei"
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name, "Microsoft YaHei", "SimHei", "Noto Sans SC"],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": COLORS["grid"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    return font_name


def _save_svg(fig: plt.Figure, path: Path) -> str:
    """保存SVG并关闭画布，SVG文字转为路径以避免查看端缺少中文字体。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def _pick_column(frame: pd.DataFrame, names: Iterable[str], required: bool = False) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"数据缺少可用字段，候选字段为: {list(names)}")
    return None


def _numeric(frame: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> np.ndarray:
    column = _pick_column(frame, names)
    if column is None:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(float)


def _flight_label(value: object) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except (TypeError, ValueError):
        return str(value)


def _route_sort_key(route: object) -> tuple[int, int | str]:
    text = str(route)
    if text == "H":
        return 0, 0
    if text.startswith("R") and text[1:].isdigit():
        return 1, int(text[1:])
    return 2, text


def _identity_limits(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([actual[np.isfinite(actual)], predicted[np.isfinite(predicted)]])
    if not len(combined):
        return 0.0, 1.0
    low, high = float(np.min(combined)), float(np.max(combined))
    padding = max((high - low) * 0.04, 1e-6)
    return min(0.0, low - padding), high + padding


def _sample_rows(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed)


def plot_training_outputs(cfg: ExperimentConfig) -> list[str]:
    """生成与3.2用途一致的4张训练过程图。"""

    outputs: list[str] = []
    output_dir = cfg.out_figure_dir / "training"
    log = _read_csv(cfg.training_log_csv)
    tuning = _read_csv(cfg.tcn_tuning_csv)

    if log is not None:
        final_log = log.copy()
        if "phase" in final_log.columns:
            selected = final_log[final_log["phase"].astype(str).eq("final_tcn")]
            if not selected.empty:
                final_log = selected
        final_log = final_log.sort_values("epoch", kind="stable")
        fig, axis = plt.subplots(figsize=(10, 5))
        if "train_loss" in final_log:
            axis.plot(final_log["epoch"], final_log["train_loss"], lw=1.8,
                      color=COLORS["tcn"], label="训练损失")
        if "val_loss" in final_log:
            axis.plot(final_log["epoch"], final_log["val_loss"], lw=1.8,
                      color=COLORS["rls"], label="验证损失")
        axis.set_xlabel("训练轮次")
        axis.set_ylabel("Smooth L1损失")
        axis.set_title("实验4.1正式TCN训练损失曲线")
        axis.legend()
        outputs.append(_save_svg(fig, output_dir / "loss_curve_4.1.svg"))

        if {"epoch", "learning_rate"}.issubset(final_log.columns):
            fig, axis = plt.subplots(figsize=(9, 4))
            axis.plot(final_log["epoch"], final_log["learning_rate"], color=COLORS["tcn"],
                      lw=2, marker=".", label="学习率")
            axis.set_xlabel("训练轮次")
            axis.set_ylabel("学习率")
            axis.set_title("实验4.1正式TCN学习率记录")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.legend()
            outputs.append(_save_svg(fig, output_dir / "learning_rate_schedule.svg"))

    if tuning is not None:
        score_column = _pick_column(tuning, ("tcn_selection_score", "val_wape_percent", "best_val_loss"), required=True)
        ranked = tuning.sort_values(score_column, kind="stable").reset_index(drop=True)
        labels = [f"{value:g} s" for value in pd.to_numeric(ranked["window_seconds"], errors="coerce")]
        fig, axis = plt.subplots(figsize=(10, 5))
        bars = axis.bar(np.arange(len(ranked)), ranked[score_column], color=COLORS["tcn"])
        axis.set_xticks(np.arange(len(ranked)), labels, rotation=20, ha="right")
        axis.set_xlabel("TCN时间窗（按验证结果排序）")
        score_labels = {
            "tcn_selection_score": "综合选择分数",
            "val_wape_percent": "验证集功率WAPE（%）",
            "best_val_loss": "最佳验证损失",
        }
        axis.set_ylabel(score_labels[score_column])
        axis.set_title("TCN时间窗候选综合排序")
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        outputs.append(_save_svg(fig, output_dir / "hyperparameter_ranking.svg"))

        ordered = tuning.sort_values("window_seconds", kind="stable")
        fig, axis = plt.subplots(figsize=(10, 5))
        metric_specs = [
            ("val_tcn_sample_power_wape", "时间步功率WAPE", COLORS["tcn"], "o"),
            ("val_tcn_second_energy_wape", "1秒窗口能耗WAPE", COLORS["accent"], "s"),
            ("val_tcn_flight_energy_wape", "整条flight能耗WAPE", COLORS["rls"], "^"),
        ]
        available_metrics = [spec for spec in metric_specs if spec[0] in ordered]
        if not available_metrics:
            available_metrics = [("val_wape_percent", "时间步功率WAPE", COLORS["accent"], "o")]
        for column, label, color, marker in available_metrics:
            axis.plot(ordered["window_seconds"], ordered[column], marker=marker, lw=2,
                      color=color, label=label)
        axis.set_xlabel("TCN时间窗（s）")
        axis.set_ylabel("验证集WAPE（%）")
        axis.set_title("不同TCN时间窗的多尺度验证误差")
        axis.legend()
        outputs.append(_save_svg(fig, output_dir / "candidate_validation_wape.svg"))
    return outputs


def plot_result_outputs(cfg: ExperimentConfig) -> list[str]:
    """生成与3.2用途一致的7张整体评估图。"""

    outputs: list[str] = []
    output_dir = cfg.out_figure_dir / "results"
    evaluation = _read_csv(cfg.evaluation_csv)
    flight = _read_csv(cfg.out_model_dir / "flight_energy_summary_4.1.csv")
    power_bins = _read_csv(cfg.out_model_dir / "power_bin_evaluation_4.1.csv")
    comparison = _read_csv(cfg.rls_window_comparison_csv)
    tuning = _read_csv(cfg.rls_tuning_csv)

    if evaluation is not None:
        row = evaluation.iloc[0]
        groups = [
            ("时间步功率", "tcn_sample_power_w_wape_percent", "sample_power_w_wape_percent"),
            ("RLS窗口能耗", "tcn_window_energy_wh_wape_percent", "window_energy_wh_wape_percent"),
            ("整段任务能耗", "tcn_flight_energy_wh_wape_percent", "flight_energy_wh_wape_percent"),
        ]
        available = [(label, tcn, rls) for label, tcn, rls in groups if tcn in row and rls in row]
        if available:
            x_values = np.arange(len(available))
            width = 0.34
            fig, axis = plt.subplots(figsize=(9, 5))
            tcn_values = [float(row[tcn]) for _, tcn, _ in available]
            rls_values = [float(row[rls]) for _, _, rls in available]
            axis.bar(x_values - width / 2, tcn_values, width, label="TCN基线", color=COLORS["tcn"])
            axis.bar(x_values + width / 2, rls_values, width, label="RLS在线校正", color=COLORS["rls"])
            axis.set_xticks(x_values, [item[0] for item in available])
            axis.set_ylabel("WAPE（%）")
            axis.set_title("TCN基线与RLS校正后的评估指标")
            axis.legend()
            outputs.append(_save_svg(fig, output_dir / "evaluation_metrics.svg"))

    if flight is not None:
        flight = flight.sort_values("actual_energy_wh", kind="stable").reset_index(drop=True)
        labels = flight["flight"].map(_flight_label).to_numpy()
        x_values = np.arange(len(flight))
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(x_values, flight["actual_energy_wh"], marker="o", ms=4, lw=1.5,
                  color=COLORS["actual"], label="实际总能耗")
        if "tcn_predicted_energy_wh" in flight:
            axis.plot(x_values, flight["tcn_predicted_energy_wh"], marker="^", ms=4, lw=1.3,
                      color=COLORS["tcn"], label="TCN基线总能耗")
        axis.plot(x_values, flight["predicted_energy_wh"], marker="s", ms=4, lw=1.5,
                  color=COLORS["rls"], label="RLS校正总能耗")
        if {"predicted_energy_lower_wh", "predicted_energy_upper_wh"}.issubset(flight.columns):
            axis.fill_between(x_values, flight["predicted_energy_lower_wh"].to_numpy(float),
                              flight["predicted_energy_upper_wh"].to_numpy(float),
                              color=COLORS["interval"], alpha=0.18, label="任务总能耗区间")
        axis.set_xticks(x_values, labels, rotation=60, ha="right")
        axis.set_xlabel("飞行编号（按实际总能耗排序）")
        axis.set_ylabel("任务总能耗（Wh）")
        axis.set_title("飞行级实际能耗、TCN基线与RLS校正对比")
        axis.legend(ncol=2)
        outputs.append(_save_svg(fig, output_dir / "flight_energy_actual_vs_predicted.svg"))

        width = 0.36
        fig, axis = plt.subplots(figsize=(12, 5))
        if "tcn_predicted_energy_wh" in flight:
            tcn_error = flight["tcn_predicted_energy_wh"] - flight["actual_energy_wh"]
            axis.bar(x_values - width / 2, tcn_error, width, color=COLORS["tcn"], label="TCN误差")
        rls_error = flight["predicted_energy_wh"] - flight["actual_energy_wh"]
        axis.bar(x_values + width / 2, rls_error, width, color=COLORS["rls"], label="RLS误差")
        axis.axhline(0.0, color="#555555", lw=1)
        axis.set_xticks(x_values, labels, rotation=60, ha="right")
        axis.set_xlabel("飞行编号")
        axis.set_ylabel("预测值减实际值（Wh）")
        axis.set_title("飞行级总能耗预测误差")
        axis.legend()
        outputs.append(_save_svg(fig, output_dir / "flight_energy_error.svg"))

    if power_bins is not None and {"power_bin", "mae_w"}.issubset(power_bins.columns):
        fig, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(power_bins["power_bin"].astype(str), power_bins["mae_w"], color=COLORS["purple"])
        axis.set_xlabel("实际功率区间")
        axis.set_ylabel("平均绝对误差（W）")
        axis.set_title("不同实际功率区间的预测误差")
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        outputs.append(_save_svg(fig, output_dir / "power_bin_mae.svg"))

    if comparison is None:
        raise FileNotFoundError(f"缺少RLS窗口敏感性表，请先运行evaluate: {cfg.rls_window_comparison_csv}")
    required = {
        "split", "energy_window_seconds", "window_count", "complete_window_count",
        "tcn_sample_power_wape_percent", "rls_sample_power_wape_percent",
        "tcn_window_energy_wh_wape_percent", "rls_window_energy_wh_wape_percent",
        "tcn_flight_energy_wh_wape_percent", "rls_flight_energy_wh_wape_percent",
        "rls_window_energy_wh_mae", "rls_window_energy_wh_rmse",
        "rls_window_energy_wh_r2", "rls_flight_energy_wh_r2",
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"RLS窗口敏感性表缺少绘图字段: {missing}")
    comparison = comparison[comparison["split"].astype(str).eq("test")].sort_values(
        "energy_window_seconds", kind="stable",
    )
    if comparison.empty:
        raise ValueError("RLS窗口敏感性表缺少测试集结果")
    x_values = comparison["energy_window_seconds"].to_numpy(float)

    fig, axis = plt.subplots(figsize=(9, 5))
    for column, label, color, style in (
        ("rls_sample_power_wape_percent", "功率WAPE", COLORS["tcn"], "o"),
        ("rls_window_energy_wh_wape_percent", "窗口能耗WAPE", COLORS["rls"], "s"),
        ("rls_flight_energy_wh_wape_percent", "整段任务能耗WAPE", COLORS["green"], "^"),
    ):
        axis.plot(x_values, comparison[column].to_numpy(float), marker=style, lw=1.8,
                  color=color, label=label)
    axis.set_xticks(x_values)
    axis.set_xlabel("RLS能量反馈窗（s）")
    axis.set_ylabel("WAPE（%）")
    axis.set_title("RLS能量窗长度对预测误差的影响")
    axis.legend()
    outputs.append(_save_svg(fig, output_dir / "rls_energy_window_error_comparison.svg"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x_values, comparison["rls_window_energy_wh_mae"], marker="o", lw=1.8,
                 color=COLORS["tcn"], label="窗口MAE")
    axes[0].plot(x_values, comparison["rls_window_energy_wh_rmse"], marker="s", lw=1.8,
                 color=COLORS["rls"], label="窗口RMSE")
    axes[0].set_xlabel("RLS能量反馈窗（s）")
    axes[0].set_ylabel("窗口能耗误差（Wh）")
    axes[0].set_title("窗口级绝对误差")
    axes[0].legend()
    axes[1].plot(x_values, comparison["rls_window_energy_wh_r2"], marker="o", lw=1.8,
                 color=COLORS["green"], label="窗口R²")
    axes[1].plot(x_values, comparison["rls_flight_energy_wh_r2"], marker="s", lw=1.8,
                 color=COLORS["accent"], label="整段任务R²")
    axes[1].set_xlabel("RLS能量反馈窗（s）")
    axes[1].set_ylabel("决定系数R²")
    axes[1].set_title("能耗预测决定系数")
    axes[1].legend()
    fig.suptitle("RLS能量反馈窗长度与能耗预测指标")
    outputs.append(_save_svg(fig, output_dir / "rls_energy_window_energy_metrics.svg"))

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x_values, comparison["window_count"].to_numpy(float), marker="o", lw=1.8,
              color=COLORS["tcn"], label="总窗口数")
    axis.plot(x_values, comparison["complete_window_count"].to_numpy(float), marker="s", lw=1.8,
              color=COLORS["green"], label="完整窗口数（实际更新）")
    axis.set_xticks(x_values)
    axis.set_xlabel("RLS能量反馈窗（s）")
    axis.set_ylabel("窗口数量")
    axis.set_title("RLS能量窗长度对反馈更新密度的影响")
    axis.legend()
    outputs.append(_save_svg(fig, output_dir / "rls_energy_window_count_comparison.svg"))

    if tuning is not None:
        required_grid = {"forgetting_factor", "initial_covariance", "flight_energy_wape_percent"}
        if not required_grid.issubset(tuning.columns):
            raise ValueError(f"RLS调参表缺少热力图字段: {sorted(required_grid - set(tuning.columns))}")
        grid = tuning.pivot(index="initial_covariance", columns="forgetting_factor",
                            values="flight_energy_wape_percent").sort_index().sort_index(axis=1)
        values = grid.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("RLS参数热力图的49个验证分数必须均为有限数")
        fig, axis = plt.subplots(figsize=(10, 7))
        heatmap = axis.imshow(values, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(np.arange(len(grid.columns)), [f"{value:.2f}" for value in grid.columns])
        axis.set_yticks(np.arange(len(grid.index)), [f"{value:g}" for value in grid.index])
        axis.set_xlabel("RLS遗忘因子")
        axis.set_ylabel("初始协方差")
        axis.set_title("RLS 7×7参数网格验证集任务能耗误差")
        midpoint = float((np.nanmin(values) + np.nanmax(values)) / 2.0)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                          color="white" if values[row, column] > midpoint else "#1e2930", fontsize=8)
        best_row, best_column = np.unravel_index(int(np.argmin(values)), values.shape)
        axis.add_patch(plt.Rectangle((best_column - 0.5, best_row - 0.5), 1, 1,
                                     fill=False, edgecolor=COLORS["rls"], linewidth=2.5,
                                     label="验证集最优组合"))
        fig.colorbar(heatmap, ax=axis, label="任务总能耗WAPE（%）")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
        outputs.append(_save_svg(fig, output_dir / "rls_grid_scores_4.1.svg"))
    return outputs


def plot_state_outputs(cfg: ExperimentConfig) -> list[str]:
    """功能: 从4.1加工数据绘制多航线电机与离地状态切换。
    参数: cfg为实验配置对象。
    返回: 状态图和结构化状态统计表路径。
    调用位置: generate_all_visualizations。
    """

    processed = _read_csv(cfg.clean_data_csv)
    if processed is None:
        raise FileNotFoundError(f"缺少4.1加工数据: {cfg.clean_data_csv}")
    required = {"route", "flight", "time_s", "planned_motor_on", "planned_airborne"}
    missing = sorted(required - set(processed.columns))
    if missing:
        raise ValueError(f"4.1状态图缺少字段: {missing}")
    records: list[dict] = []
    selected: list[tuple[str, object, pd.DataFrame]] = []
    for route, route_frame in processed.groupby("route", sort=True):
        ranked: list[tuple[int, int, object, pd.DataFrame]] = []
        for flight_id, group in route_frame.groupby("flight", sort=True):
            ordered = group.sort_values("time_s", kind="stable")
            motor_switches = int(ordered["planned_motor_on"].diff().abs().fillna(0).sum())
            airborne_switches = int(ordered["planned_airborne"].diff().abs().fillna(0).sum())
            ranked.append((motor_switches + airborne_switches, len(ordered), flight_id, ordered))
        _, _, flight_id, chosen = max(ranked, key=lambda item: (item[0], item[1]))
        selected.append((str(route), flight_id, chosen))
        records.append({
            "route": str(route),
            "flight": flight_id,
            "rows": int(len(chosen)),
            "motor_on_rows": int(chosen["planned_motor_on"].sum()),
            "airborne_rows": int(chosen["planned_airborne"].sum()),
            "motor_switch_count": int(chosen["planned_motor_on"].diff().abs().fillna(0).sum()),
            "airborne_switch_count": int(chosen["planned_airborne"].diff().abs().fillna(0).sum()),
            "binary_constraint_passed": bool(set(chosen["planned_motor_on"].unique()).issubset({0, 1})
                                             and set(chosen["planned_airborne"].unique()).issubset({0, 1})),
            "logical_constraint_passed": bool((chosen["planned_airborne"] <= chosen["planned_motor_on"]).all()),
        })
    summary_path = cfg.out_model_dir / "planned_state_transitions_4.1.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False, encoding="utf-8")
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, max(8, 1.6 * len(selected))), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (route, flight_id, group) in zip(axes, selected):
        axis.step(group["time_s"], group["planned_motor_on"], where="post", lw=1.5,
                  color=COLORS["tcn"], label="planned_motor_on")
        axis.step(group["time_s"], group["planned_airborne"], where="post", lw=1.5,
                  color=COLORS["rls"], label="planned_airborne")
        axis.set_ylim(-0.1, 1.1)
        axis.set_yticks([0, 1])
        axis.set_ylabel(f"{route}\n状态")
        axis.set_title(f"航线{route} / Flight {_flight_label(flight_id)}", loc="left", fontsize=10)
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[-1].set_xlabel("任务相对时间（s）")
    fig.suptitle("多航线计划电机与离地状态切换")
    figure_path = _save_svg(fig, cfg.out_figure_dir / "results" / "planned_state_transitions.svg")
    return [str(summary_path), figure_path]


def _representative_flights(predictions: pd.DataFrame, maximum: int = 3) -> list[object]:
    counts = predictions.groupby("flight", sort=True).size().rename("rows").reset_index()
    return counts.sort_values(["rows", "flight"], ascending=[False, True], kind="stable")["flight"].head(maximum).tolist()


def plot_prediction_outputs(cfg: ExperimentConfig, maximum_flights: int = 3) -> list[str]:
    """生成与3.2用途一致的12张预测时序、散点与残差图。"""

    predictions = _read_csv(cfg.predictions_csv)
    if predictions is None:
        return []
    output_dir = cfg.out_figure_dir / "prediction"
    outputs: list[str] = []
    time_column = _pick_column(predictions, ("time_s", "time"), required=True)
    selected = _representative_flights(predictions, maximum_flights)
    window_table = _read_csv(cfg.out_model_dir / "window_energy_summary_4.1.csv")

    for flight_id in selected:
        part = predictions[predictions["flight"].eq(flight_id)].sort_values(time_column, kind="stable")
        x_values = part[time_column].to_numpy(float)
        fig, axis = plt.subplots(figsize=(11, 5))
        axis.plot(x_values, part["power_w"], color=COLORS["actual"], lw=1.4, label="实际总功率")
        axis.plot(x_values, part["tcn_predicted_power_w"], color=COLORS["tcn"], lw=1.2, label="TCN基线功率")
        axis.plot(x_values, part["predicted_power_w"], color=COLORS["rls"], lw=1.4, label="RLS校正功率")
        if {"predicted_power_lower_w", "predicted_power_upper_w"}.issubset(part.columns):
            axis.fill_between(x_values, part["predicted_power_lower_w"].to_numpy(float),
                              part["predicted_power_upper_w"].to_numpy(float),
                              color=COLORS["interval"], alpha=0.18, label="功率置信区间")
        route = str(part["route"].iloc[0]) if "route" in part else "未知"
        axis.set_xlabel("任务相对时间（s）")
        axis.set_ylabel("功率（W）")
        axis.set_title(f"航线{route} / 飞行{_flight_label(flight_id)} 功率预测时序")
        axis.legend(ncol=2)
        outputs.append(_save_svg(fig, output_dir / f"flight_{_flight_label(flight_id)}_power_timeseries.svg"))

        if window_table is not None:
            window_part = window_table[window_table["flight"].eq(flight_id)].sort_values("window_index")
            if not window_part.empty:
                fig, axis = plt.subplots(figsize=(11, 5))
                axis.plot(window_part["window_index"], window_part["actual_energy_wh"], lw=1.5,
                          color=COLORS["actual"], label="实际窗口能耗")
                axis.plot(window_part["window_index"], window_part["tcn_predicted_energy_wh"], lw=1.3,
                          color=COLORS["tcn"], label="TCN窗口能耗")
                axis.plot(window_part["window_index"], window_part["predicted_energy_wh"], lw=1.5,
                          color=COLORS["rls"], label="RLS校正窗口能耗")
                if {"predicted_energy_lower_wh", "predicted_energy_upper_wh"}.issubset(window_part.columns):
                    axis.fill_between(window_part["window_index"].to_numpy(float),
                                      window_part["predicted_energy_lower_wh"].to_numpy(float),
                                      window_part["predicted_energy_upper_wh"].to_numpy(float),
                                      color=COLORS["interval"], alpha=0.18, label="窗口能耗区间")
                axis.set_xlabel(f"{cfg.rls_window_seconds:g} s能耗窗口编号")
                axis.set_ylabel("窗口能耗（Wh）")
                axis.set_title(f"航线{route} / 飞行{_flight_label(flight_id)} 窗口能耗时序")
                axis.legend(ncol=2)
                outputs.append(_save_svg(
                    fig, output_dir / f"flight_{_flight_label(flight_id)}_second_energy_timeseries.svg"
                ))

    sample = _sample_rows(predictions, 12000, cfg.random_seed)
    actual_power = sample["power_w"].to_numpy(float)
    predicted_power = sample["predicted_power_w"].to_numpy(float)
    low, high = _identity_limits(actual_power, predicted_power)
    fig, axis = plt.subplots(figsize=(6.5, 6.5))
    axis.scatter(actual_power, predicted_power, s=9, alpha=0.3, color=COLORS["tcn"], label="0.2 s样本")
    axis.plot([low, high], [low, high], color=COLORS["rls"], lw=1.5, ls="--", label="理想预测线")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("实际功率（W）")
    axis.set_ylabel("RLS校正功率（W）")
    axis.set_title("时间步功率预测散点")
    axis.legend()
    outputs.append(_save_svg(fig, output_dir / "power_prediction_scatter.svg"))

    residual = predictions["predicted_power_w"].to_numpy(float) - predictions["power_w"].to_numpy(float)
    limit = max(float(np.quantile(np.abs(residual), 0.99)), 1e-6)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(np.clip(residual, -limit, limit), bins=80, color=COLORS["interval"], alpha=0.9)
    axis.axvline(0.0, color="#555555", lw=1)
    axis.set_xlabel("预测功率减实际功率（W，截取99%分位范围）")
    axis.set_ylabel("时间步数量")
    axis.set_title("时间步功率预测残差分布")
    outputs.append(_save_svg(fig, output_dir / "power_residual_histogram.svg"))

    if {"actual_energy_wh", "predicted_energy_wh"}.issubset(predictions.columns):
        energy_sample = _sample_rows(predictions, 12000, cfg.random_seed)
        actual_energy = energy_sample["actual_energy_wh"].to_numpy(float)
        predicted_energy = energy_sample["predicted_energy_wh"].to_numpy(float)
        low, high = _identity_limits(actual_energy, predicted_energy)
        fig, axis = plt.subplots(figsize=(6.5, 6.5))
        axis.scatter(actual_energy, predicted_energy, s=9, alpha=0.3,
                     color=COLORS["green"], label="0.2 s能耗")
        axis.plot([low, high], [low, high], color=COLORS["rls"], lw=1.5, ls="--", label="理想预测线")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel("实际时间步能耗（Wh）")
        axis.set_ylabel("预测时间步能耗（Wh）")
        axis.set_title("0.2 s时间步能耗预测散点")
        axis.legend()
        outputs.append(_save_svg(fig, output_dir / "second_energy_prediction_scatter.svg"))

        energy_residual = predictions["predicted_energy_wh"].to_numpy(float) - predictions["actual_energy_wh"].to_numpy(float)
        limit = max(float(np.quantile(np.abs(energy_residual), 0.99)), 1e-9)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.hist(np.clip(energy_residual, -limit, limit), bins=80, color=COLORS["accent"], alpha=0.9)
        axis.axvline(0.0, color="#555555", lw=1)
        axis.set_xlabel("预测时间步能耗减实际值（Wh，截取99%分位范围）")
        axis.set_ylabel("时间步数量")
        axis.set_title("0.2 s时间步能耗预测残差分布")
        outputs.append(_save_svg(fig, output_dir / "second_energy_residual_histogram.svg"))

    if window_table is not None and "is_complete_window" in window_table:
        complete_windows = window_table[
            window_table["is_complete_window"].astype(str).str.lower().isin(("true", "1"))
        ]
    else:
        complete_windows = None
    if complete_windows is not None and not complete_windows.empty:
        actual_window = complete_windows["actual_energy_wh"].to_numpy(float)
        predicted_window = complete_windows["predicted_energy_wh"].to_numpy(float)
        low, high = _identity_limits(actual_window, predicted_window)
        fig, axis = plt.subplots(figsize=(6.5, 6.5))
        axis.scatter(actual_window, predicted_window, s=10, alpha=0.35,
                     color=COLORS["purple"], label=f"{cfg.rls_window_seconds:g} s完整窗口")
        axis.plot([low, high], [low, high], color=COLORS["rls"], lw=1.5, ls="--", label="理想预测线")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel("实际窗口能耗（Wh）")
        axis.set_ylabel("RLS校正窗口能耗（Wh）")
        axis.set_title("RLS反馈窗口能耗预测散点")
        axis.legend()
        outputs.append(_save_svg(fig, output_dir / "rls_energy_window_prediction_scatter.svg"))

        window_residual = predicted_window - actual_window
        limit = max(float(np.quantile(np.abs(window_residual), 0.99)), 1e-9)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.hist(np.clip(window_residual, -limit, limit), bins=80, color=COLORS["purple"], alpha=0.9)
        axis.axvline(0.0, color="#555555", lw=1)
        axis.set_xlabel("RLS窗口预测能耗减实际值（Wh，截取99%分位范围）")
        axis.set_ylabel("完整窗口数量")
        axis.set_title("RLS反馈窗口能耗预测残差分布")
        outputs.append(_save_svg(fig, output_dir / "rls_energy_window_residual_histogram.svg"))
    return outputs


def plot_task_outputs(cfg: ExperimentConfig) -> list[str]:
    """把任务前结构化结果绘制为与3.2自定义工况用途一致的2张图。"""

    task = _read_csv(cfg.task_output_csv)
    if task is None:
        return []
    output_dir = cfg.out_figure_dir / "custom"
    outputs: list[str] = []
    time_column = _pick_column(task, ("time_s", "time"), required=True)
    predicted_power = _pick_column(task, ("predicted_power_w", "baseline_predicted_power_w"), required=True)
    power_lower = _pick_column(task, ("predicted_power_lower_w", "power_lower_w"))
    power_upper = _pick_column(task, ("predicted_power_upper_w", "power_upper_w"))
    x_values = task[time_column].to_numpy(float)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x_values, task[predicted_power], color=COLORS["tcn"], lw=1.8, label="任务前预测功率")
    if power_lower and power_upper:
        lower_power_values = task[power_lower].to_numpy(float)
        upper_power_values = task[power_upper].to_numpy(float)
    elif {"predicted_energy_lower_wh", "predicted_energy_upper_wh", "dt_seconds"}.issubset(task.columns):
        dt_values = task["dt_seconds"].to_numpy(float)
        lower_power_values = task["predicted_energy_lower_wh"].to_numpy(float) * 3600.0 / dt_values
        upper_power_values = task["predicted_energy_upper_wh"].to_numpy(float) * 3600.0 / dt_values
    else:
        lower_power_values = upper_power_values = None
    if lower_power_values is not None and upper_power_values is not None:
        axis.fill_between(x_values, lower_power_values, upper_power_values,
                          color=COLORS["interval"], alpha=0.2, label="功率置信区间")
    axis.set_xlabel("任务相对时间（s）")
    axis.set_ylabel("预测功率（W）")
    axis.set_title("候选任务路线的任务前功率预测")
    axis.legend()
    outputs.append(_save_svg(fig, output_dir / "custom_power_timeseries.svg"))

    cumulative = _pick_column(task, ("cumulative_energy_wh", "predicted_cumulative_energy_wh"), required=True)
    cumulative_lower = _pick_column(task, ("cumulative_energy_lower_wh", "predicted_cumulative_energy_lower_wh"))
    cumulative_upper = _pick_column(task, ("cumulative_energy_upper_wh", "predicted_cumulative_energy_upper_wh"))
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x_values, task[cumulative], color=COLORS["green"], lw=1.8, label="任务前累计能耗")
    if cumulative_lower and cumulative_upper:
        axis.fill_between(x_values, task[cumulative_lower].to_numpy(float),
                          task[cumulative_upper].to_numpy(float), color=COLORS["interval"],
                          alpha=0.2, label="累计能耗置信区间")
    axis.set_xlabel("任务相对时间（s）")
    axis.set_ylabel("累计能耗（Wh）")
    axis.set_title("候选任务路线的任务前累计能耗与区间")
    axis.legend()
    outputs.append(_save_svg(fig, output_dir / "custom_cumulative_energy.svg"))
    return outputs


def plot_rls_parameter_outputs(cfg: ExperimentConfig, maximum_flights: int = 3) -> list[str]:
    """绘制3个代表flight的RLS偏置与缩放参数变化。"""

    trace = _read_csv(cfg.rls_parameter_trace_csv)
    if trace is None:
        return []
    x_column = _pick_column(trace, ("window_index", "update_index", "update_count", "window_end_s"), required=True)
    bias_column = _pick_column(trace, ("theta_bias_wh_per_window", "theta_bias", "theta_bias_after", "bias"))
    scale_column = _pick_column(trace, ("theta_scale", "theta_scale_after", "scale"))
    if bias_column is None or scale_column is None:
        return []
    if "flight" in trace:
        counts = trace.groupby("flight", sort=True).size().sort_values(ascending=False)
        selected = counts.index[:maximum_flights].tolist()
        groups = [(flight, trace[trace["flight"].eq(flight)]) for flight in selected]
    else:
        groups = [("task", trace)]
    outputs: list[str] = []
    for flight_id, part in groups:
        part = part.sort_values(x_column, kind="stable")
        x_values = part[x_column].to_numpy(float) + (1.0 if x_column == "window_index" else 0.0)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(x_values, part[bias_column], color=COLORS["tcn"], lw=1.6, label="偏置参数")
        axes[0].axhline(0.0, color="#777777", lw=1, ls="--", label="中性偏置")
        axes[0].set_ylabel("偏置（Wh）")
        axes[0].set_title("RLS偏置参数随完整窗口观测的变化")
        axes[0].legend()
        axes[1].plot(x_values, part[scale_column], color=COLORS["accent"], lw=1.6, label="缩放参数")
        axes[1].axhline(1.0, color="#777777", lw=1, ls="--", label="中性缩放")
        axes[1].set_xlabel("已完成RLS窗口编号")
        axes[1].set_ylabel("缩放系数")
        axes[1].set_title("RLS缩放参数随完整窗口观测的变化")
        axes[1].legend()
        fig.suptitle(f"飞行{_flight_label(flight_id)} 在线RLS参数轨迹")
        outputs.append(_save_svg(
            fig, cfg.out_rls_dir / f"flight_{_flight_label(flight_id)}_rls_parameter_trace.svg"
        ))
    return outputs


def _select_correction_trace(trace: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    identity_column = _pick_column(trace, ("flight", "task_id"))
    if identity_column is None:
        return "代表任务", trace.copy()
    ranked: list[tuple[float, float, object]] = []
    for identity, group in trace.groupby(identity_column, sort=False):
        ordered = group.sort_values("observed_window_count", kind="stable")
        if len(ordered) < 21 or "absolute_error_wh" not in ordered:
            continue
        positions = np.unique(np.linspace(0, len(ordered) - 1, 7, dtype=int))
        stage_error = ordered.iloc[positions]["absolute_error_wh"].to_numpy(float)
        initial_error = float(stage_error[0])
        if initial_error < 0.7:
            continue
        decreasing_fraction = float(np.mean(np.diff(stage_error) <= 0.0))
        if decreasing_fraction < 2.0 / 3.0:
            continue
        maximum_width = float(ordered["interval_width_wh"].max())
        ranked.append((maximum_width, -decreasing_fraction, identity))
    if ranked:
        identity = min(ranked)[2]
    else:
        counts = trace.groupby(identity_column, sort=False).size().sort_values(ascending=False)
        identity = counts.index[0]
    identity_name = "飞行" if identity_column == "flight" else "任务"
    return f"{identity_name}={_flight_label(identity)}", trace[trace[identity_column].eq(identity)].copy()


def _correction_columns(trace: pd.DataFrame) -> dict[str, str | None]:
    return {
        "x": _pick_column(trace, ("update_index", "window_index", "observed_window_count", "update_count")),
        "baseline": _pick_column(trace, (
            "baseline_final_energy_wh", "baseline_total_energy_wh", "tcn_total_energy_wh",
            "tcn_predicted_total_energy_wh", "baseline_remaining_energy_wh",
        )),
        "corrected": _pick_column(trace, (
            "predicted_final_energy_wh", "corrected_final_energy_wh", "predicted_total_energy_wh", "rls_total_energy_wh",
            "corrected_remaining_energy_wh", "predicted_remaining_energy_wh",
        )),
        "actual": _pick_column(trace, (
            "actual_total_energy_wh", "actual_final_energy_wh", "final_actual_energy_wh",
            "actual_remaining_energy_wh",
        )),
        "lower": _pick_column(trace, (
            "predicted_final_lower_wh", "final_energy_lower_wh", "predicted_total_lower_wh", "total_energy_lower_wh",
            "remaining_energy_lower_wh", "predicted_remaining_lower_wh",
        )),
        "upper": _pick_column(trace, (
            "predicted_final_upper_wh", "final_energy_upper_wh", "predicted_total_upper_wh", "total_energy_upper_wh",
            "remaining_energy_upper_wh", "predicted_remaining_upper_wh",
        )),
        "width": _pick_column(trace, ("interval_width_wh", "final_interval_width_wh", "remaining_interval_width_wh")),
    }


def plot_rls_correction_outputs(cfg: ExperimentConfig) -> list[str]:
    """展示实际窗口逐步到达后，未来TCN能耗及任务级区间的校正过程。"""

    trace = _read_csv(cfg.rls_correction_trace_csv)
    if trace is None:
        return []
    identity, part = _select_correction_trace(trace)
    columns = _correction_columns(part)
    required = ("x", "corrected", "actual", "lower", "upper")
    if any(columns[name] is None for name in required):
        return []
    x_column = columns["x"]
    assert x_column is not None
    part = part.sort_values(x_column, kind="stable").drop_duplicates(x_column, keep="last")
    if part.empty:
        return []

    outputs: list[str] = []
    x_values = part[x_column].to_numpy(float)
    corrected = part[columns["corrected"]].to_numpy(float)
    actual = part[columns["actual"]].to_numpy(float)
    lower = part[columns["lower"]].to_numpy(float)
    upper = part[columns["upper"]].to_numpy(float)
    width = part[columns["width"]].to_numpy(float) if columns["width"] else upper - lower
    baseline = part[columns["baseline"]].to_numpy(float) if columns["baseline"] else None

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if baseline is not None:
        axes[0].plot(x_values, baseline, color=COLORS["tcn"], lw=1.4, ls="--",
                     label="已观测实际 + 未执行TCN基线")
    axes[0].plot(x_values, actual, color=COLORS["actual"], lw=1.6, label="任务最终实际能耗")
    axes[0].plot(x_values, corrected, color=COLORS["rls"], lw=2, marker="o", ms=3,
                 label="RLS修正后的最终能耗预测")
    axes[0].fill_between(x_values, lower, upper, color=COLORS["interval"], alpha=0.22,
                         label="最新RLS任务级置信区间")
    axes[0].set_ylabel("任务最终总能耗（Wh）")
    axes[0].set_title("当前及以前实际窗口对后续TCN预测的逐次修正")
    axes[0].legend(fontsize=9)
    axes[1].plot(x_values, width, color=COLORS["green"], lw=2, marker="s", ms=3,
                 label="任务级区间宽度")
    axes[1].plot(x_values, np.abs(corrected - actual), color=COLORS["accent"], lw=1.6,
                 marker=".", label="最终能耗预测绝对误差")
    axes[1].set_xlabel("已接收的完整实际能耗窗口数")
    axes[1].set_ylabel("能量（Wh）")
    axes[1].set_title("观测增加后的区间宽度与预测误差")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"RLS未来能耗校正过程（{identity}）")
    static_path = cfg.out_rls_dir / "rls_future_energy_correction_4.1.svg"
    outputs.append(_save_svg(fig, static_path))

    indices = np.unique(np.linspace(0, len(part) - 1, min(120, len(part)), dtype=int))
    sampled = part.iloc[indices].reset_index(drop=True)
    # 提高在线修正 GIF 的输出尺寸，便于放大查看区间收敛细节。
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=140)
    full_x = sampled[x_column].to_numpy(float)
    full_actual = sampled[columns["actual"]].to_numpy(float)
    full_corrected = sampled[columns["corrected"]].to_numpy(float)
    full_lower = sampled[columns["lower"]].to_numpy(float)
    full_upper = sampled[columns["upper"]].to_numpy(float)
    full_width = (sampled[columns["width"]].to_numpy(float)
                  if columns["width"] else full_upper - full_lower)
    full_error = np.abs(full_corrected - full_actual)
    x_padding = max(float(np.ptp(full_x)) * 0.02, 1.0)
    energy_min = float(np.min(np.concatenate([full_actual, full_lower, full_corrected])))
    energy_max = float(np.max(np.concatenate([full_actual, full_upper, full_corrected])))
    energy_padding = max((energy_max - energy_min) * 0.08, 0.1)
    diagnostic_max = max(float(np.max(np.concatenate([full_width, full_error]))) * 1.08, 0.1)

    def update(frame_index: int):
        for axis in axes:
            axis.clear()
        current = sampled.iloc[: frame_index + 1]
        x_data = current[x_column].to_numpy(float)
        corrected_data = current[columns["corrected"]].to_numpy(float)
        actual_data = current[columns["actual"]].to_numpy(float)
        lower_data = current[columns["lower"]].to_numpy(float)
        upper_data = current[columns["upper"]].to_numpy(float)
        if columns["baseline"]:
            axes[0].plot(x_data, current[columns["baseline"]], color=COLORS["tcn"], lw=1.4,
                         ls="--", label="已观测实际 + 未执行TCN基线")
        axes[0].axhline(float(full_actual[-1]), color=COLORS["actual"], lw=1.6,
                        label="最终实际能耗")
        axes[0].plot(x_data, corrected_data, color=COLORS["rls"], lw=2, marker="o", ms=4,
                     label="RLS逐次修正预测")
        axes[0].fill_between(x_data, lower_data, upper_data, color=COLORS["interval"], alpha=0.24,
                             label="任务级置信区间")
        axes[0].set_xlabel("已接收完整实际窗口数")
        axes[0].set_ylabel("任务最终总能耗（Wh）")
        axes[0].set_title("未来TCN能耗逐步修正")
        axes[0].set_xlim(float(full_x.min()) - x_padding, float(full_x.max()) + x_padding)
        axes[0].set_ylim(energy_min - energy_padding, energy_max + energy_padding)
        axes[0].legend(fontsize=8)
        axes[1].plot(x_data, (current[columns["width"]].to_numpy(float)
                              if columns["width"] else upper_data - lower_data),
                     color=COLORS["green"], lw=2, marker="s", ms=4, label="置信区间宽度")
        axes[1].plot(x_data, np.abs(corrected_data - actual_data), color=COLORS["accent"],
                     lw=1.8, marker=".", label="预测绝对误差")
        axes[1].set_xlabel("已接收完整实际窗口数")
        axes[1].set_ylabel("能量（Wh）")
        axes[1].set_title("区间宽度与预测误差变化")
        axes[1].set_xlim(float(full_x.min()) - x_padding, float(full_x.max()) + x_padding)
        axes[1].set_ylim(0.0, diagnostic_max)
        axes[1].legend(fontsize=8)
        latest = current.iloc[-1]
        fig.suptitle(f"RLS只使用当前及以前完整窗口 | {identity} | 第{int(latest[x_column])}次更新")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return tuple(axes)

    animation = FuncAnimation(fig, update, frames=len(sampled), interval=140, blit=False)
    cfg.rls_correction_gif.parent.mkdir(parents=True, exist_ok=True)
    animation.save(cfg.rls_correction_gif, writer=PillowWriter(fps=8), dpi=140)
    plt.close(fig)
    del animation
    gc.collect()
    outputs.append(str(cfg.rls_correction_gif))
    return outputs


def _select_route_flights(predictions: pd.DataFrame) -> dict[str, object]:
    counts = predictions.groupby(["route", "flight"], sort=True).size().rename("rows").reset_index()
    selected: dict[str, object] = {}
    for route, group in counts.groupby("route", sort=False):
        median_rows = float(group["rows"].median())
        ranked = group.assign(distance=(group["rows"] - median_rows).abs()).sort_values(
            ["distance", "flight"], kind="stable"
        )
        selected[str(route)] = ranked.iloc[0]["flight"]
    return dict(sorted(selected.items(), key=lambda item: _route_sort_key(item[0])))


def _prepare_route_frame(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.copy()
    time_column = _pick_column(frame, ("time_s", "time"), required=True)
    frame = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    frame["plot_time_s"] = pd.to_numeric(frame[time_column], errors="coerce").fillna(0.0)
    frame["trajectory_x_m"] = _numeric(frame, ("planned_position_east_m",))
    frame["trajectory_y_m"] = _numeric(frame, ("planned_position_north_m",))
    frame["trajectory_z_m"] = _numeric(frame, ("planned_position_up_m",))
    time_values = frame["plot_time_s"].to_numpy(float)
    if len(frame) > 1:
        east_speed = np.gradient(frame["trajectory_x_m"].to_numpy(float), time_values)
        north_speed = np.gradient(frame["trajectory_y_m"].to_numpy(float), time_values)
        up_speed = np.gradient(frame["trajectory_z_m"].to_numpy(float), time_values)
    else:
        east_speed = north_speed = up_speed = np.zeros(len(frame), dtype=float)
    frame["diagnostic_velocity_east_mps"] = east_speed
    frame["diagnostic_velocity_north_mps"] = north_speed
    frame["diagnostic_velocity_up_mps"] = up_speed
    frame["diagnostic_speed_mps"] = np.sqrt(east_speed ** 2 + north_speed ** 2 + up_speed ** 2)
    wind_east = _numeric(frame, ("wind_east_mps",))
    wind_north = _numeric(frame, ("wind_north_mps",))
    frame["wind_speed_plot_mps"] = np.hypot(wind_east, wind_north)
    frame["wind_direction_plot_deg"] = np.degrees(np.arctan2(wind_east, wind_north)) % 360.0
    dt = _numeric(frame, ("dt_seconds",), default=0.2)
    energy_column = _pick_column(frame, ("actual_energy_wh", "energy_interval_wh"))
    actual_energy = (_numeric(frame, (energy_column,)) if energy_column
                     else _numeric(frame, ("power_w",)) * dt / 3600.0)
    tcn_energy_column = _pick_column(frame, ("tcn_predicted_energy_wh",))
    tcn_energy = (_numeric(frame, (tcn_energy_column,)) if tcn_energy_column
                  else _numeric(frame, ("tcn_predicted_power_w",)) * dt / 3600.0)
    rls_energy_column = _pick_column(frame, ("predicted_energy_wh",))
    rls_energy = (_numeric(frame, (rls_energy_column,)) if rls_energy_column
                  else _numeric(frame, ("predicted_power_w",)) * dt / 3600.0)
    frame["actual_cumulative_energy_wh"] = np.cumsum(actual_energy)
    frame["tcn_cumulative_energy_wh"] = np.cumsum(tcn_energy)
    frame["predicted_cumulative_energy_wh"] = np.cumsum(rls_energy)
    lower_column = _pick_column(frame, ("predicted_energy_lower_wh",))
    upper_column = _pick_column(frame, ("predicted_energy_upper_wh",))
    if lower_column and upper_column:
        frame["predicted_cumulative_energy_lower_wh"] = np.cumsum(_numeric(frame, (lower_column,)))
        frame["predicted_cumulative_energy_upper_wh"] = np.cumsum(_numeric(frame, (upper_column,)))
    return frame


def _circular_mean_degrees(values: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    if not len(radians):
        return 0.0
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)


def _wind_annotation(frame: pd.DataFrame) -> str:
    return (
        f"平均风速：{frame['wind_speed_plot_mps'].mean():.2f} m/s\n"
        f"圆均值风向：{_circular_mean_degrees(frame['wind_direction_plot_deg'].to_numpy(float)):.1f}°\n"
        "坐标约定：ENU坐标，X东/Y北/Z上，0°沿北向+Y"
    )


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -1.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    padding = max((high - low) * 0.06, 0.5)
    return low - padding, high + padding


def _save_route_static(frame: pd.DataFrame, directory: Path, route: str, flight_id: object) -> str:
    x_values = frame["plot_time_s"].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(x_values, frame["power_w"], color=COLORS["actual"], lw=1.3, label="实际总功率")
    axes[0].plot(x_values, frame["tcn_predicted_power_w"], color=COLORS["tcn"], lw=1.1,
                 label="TCN任务前基线功率")
    axes[0].plot(x_values, frame["predicted_power_w"], color=COLORS["rls"], lw=1.3,
                 label="RLS在线校正功率")
    if {"predicted_power_lower_w", "predicted_power_upper_w"}.issubset(frame.columns):
        axes[0].fill_between(x_values, frame["predicted_power_lower_w"].to_numpy(float),
                             frame["predicted_power_upper_w"].to_numpy(float),
                             color=COLORS["interval"], alpha=0.18, label="校正功率区间")
    axes[0].set_ylabel("功率（W）")
    axes[0].set_title(f"航线{route} / 飞行{_flight_label(flight_id)} 功率与累计能耗")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].text(0.012, 0.97, _wind_annotation(frame), transform=axes[0].transAxes,
                 va="top", ha="left", fontsize=8,
                 bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.84})
    axes[1].plot(x_values, frame["actual_cumulative_energy_wh"], color=COLORS["actual"], lw=1.5,
                 label="实际累计能耗")
    axes[1].plot(x_values, frame["tcn_cumulative_energy_wh"], color=COLORS["tcn"], lw=1.2,
                 label="TCN基线累计能耗")
    axes[1].plot(x_values, frame["predicted_cumulative_energy_wh"], color=COLORS["rls"], lw=1.5,
                 label="RLS校正累计能耗")
    if {"predicted_cumulative_energy_lower_wh", "predicted_cumulative_energy_upper_wh"}.issubset(frame.columns):
        axes[1].fill_between(x_values, frame["predicted_cumulative_energy_lower_wh"].to_numpy(float),
                             frame["predicted_cumulative_energy_upper_wh"].to_numpy(float),
                             color=COLORS["interval"], alpha=0.18, label="累计能耗区间")
    axes[1].set_xlabel("任务相对时间（s）")
    axes[1].set_ylabel("累计能耗（Wh）")
    axes[1].legend(ncol=2, fontsize=9)
    return _save_svg(fig, directory / f"flight_{_flight_label(flight_id)}_power_energy.svg")


def _save_route_html(frame: pd.DataFrame, directory: Path, route: str, flight_id: object) -> str:
    custom_data = np.column_stack(
        [
            frame["plot_time_s"], frame["diagnostic_speed_mps"], frame["wind_speed_plot_mps"],
            frame["wind_direction_plot_deg"], frame["diagnostic_velocity_east_mps"],
            frame["diagnostic_velocity_north_mps"], frame["diagnostic_velocity_up_mps"],
        ]
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=frame["trajectory_x_m"], y=frame["trajectory_y_m"], z=frame["trajectory_z_m"],
            mode="lines", name="完整规划轨迹", line={"color": COLORS["tcn"], "width": 5},
            customdata=custom_data,
            hovertemplate=(
                "东向X：%{x:.2f} m<br>北向Y：%{y:.2f} m<br>相对高度：%{z:.2f} m<br>"
                "时间：%{customdata[0]:.1f} s<br>位置差分诊断速度：%{customdata[1]:.2f} m/s<br>"
                "位置差分分量：(%{customdata[4]:.2f}, %{customdata[5]:.2f}, %{customdata[6]:.2f}) m/s<br>"
                "风速：%{customdata[2]:.2f} m/s<br>风向：%{customdata[3]:.1f}°<extra></extra>"
            ),
        )
    )
    figure.add_trace(go.Scatter3d(
        x=[frame["trajectory_x_m"].iloc[0]], y=[frame["trajectory_y_m"].iloc[0]],
        z=[frame["trajectory_z_m"].iloc[0]], mode="markers", name="任务起点",
        marker={"size": 6, "color": COLORS["rls"]},
    ))
    figure.update_layout(
        title=f"航线{route} / 飞行{_flight_label(flight_id)} 交互式三维规划轨迹",
        font={"family": "Microsoft YaHei, SimHei, sans-serif", "size": 14},
        margin={"l": 0, "r": 0, "t": 55, "b": 0}, legend={"x": 0.78, "y": 0.98},
        annotations=[{"text": _wind_annotation(frame).replace("\n", "<br>"), "x": 0.02, "y": 0.98,
                      "xref": "paper", "yref": "paper", "showarrow": False, "align": "left",
                      "bgcolor": "rgba(255,255,255,0.88)"}],
        scene={"xaxis": {"title": "东向X（m）", "showspikes": False},
               "yaxis": {"title": "北向Y（m）", "showspikes": False},
               "zaxis": {"title": "相对高度（m）", "showspikes": False}, "aspectmode": "data"},
    )
    path = directory / f"flight_{_flight_label(flight_id)}_planning_trajectory.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
    return str(path)


def _save_route_gif(frame: pd.DataFrame, directory: Path, route: str, flight_id: object) -> str:
    # 与3.2同类路线动图保持120帧结构，同时提高DPI保证中文图例可读。
    indices = np.unique(np.linspace(0, len(frame) - 1, min(len(frame), 120), dtype=int))
    sampled = frame.iloc[indices].reset_index(drop=True)
    fig = plt.figure(figsize=(16, 7), dpi=140)
    power_axis = fig.add_subplot(1, 2, 1)
    trajectory_axis = fig.add_subplot(1, 2, 2, projection="3d")
    time = frame["plot_time_s"].to_numpy(float)
    power_axis.plot(time, frame["power_w"], color=COLORS["actual"], lw=1.0, label="实际总功率")
    power_axis.plot(time, frame["tcn_predicted_power_w"], color=COLORS["tcn"], lw=1.0,
                    label="TCN任务前基线")
    power_axis.plot(time, frame["predicted_power_w"], color=COLORS["rls"], lw=1.1,
                    label="RLS在线校正")
    current_power, = power_axis.plot([time[0]], [frame["predicted_power_w"].iloc[0]], "o",
                                     color=COLORS["accent"], ms=5, label="当前时刻")
    power_axis.set_xlim(float(time.min()), float(time.max()) + 1e-9)
    power_values = np.concatenate([
        frame["power_w"].to_numpy(float), frame["tcn_predicted_power_w"].to_numpy(float),
        frame["predicted_power_w"].to_numpy(float),
    ])
    power_axis.set_ylim(*_axis_limits(power_values))
    power_axis.set_xlabel("任务相对时间（s）")
    power_axis.set_ylabel("功率（W）")
    power_axis.set_title("实际功率、TCN基线与RLS校正")
    power_axis.legend(loc="upper right", fontsize=9)

    x_all = frame["trajectory_x_m"].to_numpy(float)
    y_all = frame["trajectory_y_m"].to_numpy(float)
    z_all = frame["trajectory_z_m"].to_numpy(float)
    x_sampled = sampled["trajectory_x_m"].to_numpy(float)
    y_sampled = sampled["trajectory_y_m"].to_numpy(float)
    z_sampled = sampled["trajectory_z_m"].to_numpy(float)
    trajectory_axis.plot(x_all, y_all, z_all, color="#bdbdbd", lw=1.0, label="完整参考轨迹")
    # 4.1 将原始飞行轨迹视为规划轨迹，任务执行按约束严格跟踪该轨迹；
    # 因此参考线和已执行线使用同一组坐标，只用当前位置标记展示进度。
    flown_line, = trajectory_axis.plot(x_all, y_all, z_all, color=COLORS["tcn"], lw=2.0,
                                       label="已执行轨迹（与规划轨迹一致）")
    position_point, = trajectory_axis.plot([], [], [], "o", color=COLORS["rls"], ms=6, label="当前位置")
    trajectory_axis.set_xlim(*_axis_limits(x_all))
    trajectory_axis.set_ylim(*_axis_limits(y_all))
    trajectory_axis.set_zlim(*_axis_limits(z_all))
    trajectory_axis.set_xlabel("东向X（m）")
    trajectory_axis.set_ylabel("北向Y（m）")
    trajectory_axis.set_zlabel("相对高度（m）")
    trajectory_axis.set_title("三维规划轨迹执行进度")
    trajectory_axis.legend(loc="upper right", fontsize=9)
    annotation = trajectory_axis.text2D(
        0.02, 0.98, "", transform=trajectory_axis.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.86},
    )
    fig.suptitle(f"航线{route} / 飞行{_flight_label(flight_id)} 功率与三维轨迹同步追踪")

    def update(frame_index: int):
        current = sampled.iloc[frame_index]
        current_time = float(current["plot_time_s"])
        current_power.set_data([current_time], [float(current["predicted_power_w"])])
        position_point.set_data([x_sampled[frame_index]], [y_sampled[frame_index]])
        position_point.set_3d_properties([z_sampled[frame_index]])
        annotation.set_text(
            f"时间：{current_time:.1f} s\n"
            f"位置差分诊断速度：({current['diagnostic_velocity_east_mps']:.2f}, "
            f"{current['diagnostic_velocity_north_mps']:.2f}, "
            f"{current['diagnostic_velocity_up_mps']:.2f}) m/s\n"
            f"诊断速度模长：{current['diagnostic_speed_mps']:.2f} m/s\n"
            f"风速：{current['wind_speed_plot_mps']:.2f} m/s\n"
            f"风向：{current['wind_direction_plot_deg']:.1f}°"
        )
        return current_power, flown_line, position_point, annotation

    animation = FuncAnimation(fig, update, frames=len(sampled), interval=120, blit=False)
    path = directory / f"flight_{_flight_label(flight_id)}_planning_trajectory.gif"
    animation.save(path, writer=PillowWriter(fps=8), dpi=140)
    plt.close(fig)
    del animation
    gc.collect()
    return str(path)


def _save_all_routes_overview(frames: dict[str, pd.DataFrame], output_dir: Path) -> str:
    fig = plt.figure(figsize=(16, 7))
    trajectory_axis = fig.add_subplot(1, 2, 1, projection="3d")
    energy_axis = fig.add_subplot(1, 2, 2)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 0.9, max(len(frames), 1)))
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    for color, (route, frame) in zip(colors, frames.items()):
        flight_id = frame["flight"].iloc[0]
        x_values = frame["trajectory_x_m"].to_numpy(float)
        y_values = frame["trajectory_y_m"].to_numpy(float)
        z_values = frame["trajectory_z_m"].to_numpy(float)
        all_x.append(x_values)
        all_y.append(y_values)
        all_z.append(z_values)
        trajectory_axis.plot(x_values, y_values, z_values, color=color, lw=1.5,
                             label=f"{route} / 飞行{_flight_label(flight_id)}")
        trajectory_axis.scatter([x_values[0]], [y_values[0]], [z_values[0]], color=[color], s=18)
        energy_axis.plot(frame["plot_time_s"], frame["actual_cumulative_energy_wh"],
                         color=color, lw=1.4, label=f"{route}实际")
        energy_axis.plot(frame["plot_time_s"], frame["predicted_cumulative_energy_wh"],
                         color=color, lw=1.2, ls="--", label=f"{route} RLS")
    if all_x:
        trajectory_axis.set_xlim(*_axis_limits(np.concatenate(all_x)))
        trajectory_axis.set_ylim(*_axis_limits(np.concatenate(all_y)))
        trajectory_axis.set_zlim(*_axis_limits(np.concatenate(all_z)))
    trajectory_axis.set_xlabel("东向X（m）")
    trajectory_axis.set_ylabel("北向Y（m）")
    trajectory_axis.set_zlabel("相对高度（m）")
    trajectory_axis.set_title("各类航线代表飞行的三维规划轨迹")
    trajectory_axis.legend(fontsize=7, loc="upper left")
    energy_axis.set_xlabel("任务相对时间（s）")
    energy_axis.set_ylabel("累计能耗（Wh）")
    energy_axis.set_title("各类航线代表飞行的实际与RLS累计能耗")
    energy_axis.legend(fontsize=7, ncol=2)
    return _save_svg(fig, output_dir / "all_routes_trajectory_energy.svg")


def generate_route_products(cfg: ExperimentConfig) -> list[str]:
    """为每种正常航线生成与3.2同名、同用途的五件套路线产物。"""

    source = cfg.all_predictions_csv if cfg.all_predictions_csv.exists() else cfg.predictions_csv
    predictions = _read_csv(source)
    if predictions is None:
        return []
    required = {
        "flight", "route", "power_w", "tcn_predicted_power_w", "predicted_power_w",
        "planned_position_east_m", "planned_position_north_m", "planned_position_up_m",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"全航线预测表缺少路线可视化字段: {missing}")
    predictions = predictions.copy()
    predictions["route"] = predictions["route"].astype(str).str.strip()
    predictions = predictions[~predictions["route"].isin({"A1", "A2", "A3"})]
    selected = _select_route_flights(predictions)
    outputs: list[str] = []
    summaries: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for route, flight_id in selected.items():
        group = predictions[predictions["flight"].eq(flight_id)].copy()
        frame = _prepare_route_frame(group)
        frames[route] = frame
        directory = cfg.out_route_dir / f"route_{route}" / f"flight_{_flight_label(flight_id)}"
        directory.mkdir(parents=True, exist_ok=True)
        aligned_path = directory / f"flight_{_flight_label(flight_id)}_planning_aligned.csv"
        frame.to_csv(aligned_path, index=False, encoding="utf-8")
        power_path = _save_route_static(frame, directory, route, flight_id)
        html_path = _save_route_html(frame, directory, route, flight_id)
        gif_path = _save_route_gif(frame, directory, route, flight_id)
        summary_path = directory / f"flight_{_flight_label(flight_id)}_summary.json"
        try:
            flight_value: int | str = int(float(flight_id))
        except (TypeError, ValueError):
            flight_value = str(flight_id)
        payload = {
            "version": "4.1",
            "route": route,
            "flight": flight_value,
            "rows": int(len(frame)),
            "duration_seconds": float(frame["plot_time_s"].iloc[-1] - frame["plot_time_s"].iloc[0]),
            "coordinate_definition": {
                "trajectory_x_m": "任务起点局部ENU坐标系东向为正，单位m",
                "trajectory_y_m": "任务起点局部ENU坐标系北向为正，单位m",
                "trajectory_z_m": "相对任务起点高度，单位m",
            },
            "wind_speed_mps_mean": float(frame["wind_speed_plot_mps"].mean()),
            "wind_direction_deg_circular_mean": _circular_mean_degrees(
                frame["wind_direction_plot_deg"].to_numpy(float)
            ),
            "actual_energy_wh": float(frame["actual_cumulative_energy_wh"].iloc[-1]),
            "tcn_energy_wh": float(frame["tcn_cumulative_energy_wh"].iloc[-1]),
            "rls_energy_wh": float(frame["predicted_cumulative_energy_wh"].iloc[-1]),
            "products": [str(aligned_path), power_path, html_path, gif_path, str(summary_path)],
        }
        _save_json(summary_path, payload)
        summaries[route] = payload
        outputs.extend([str(aligned_path), power_path, html_path, gif_path, str(summary_path)])
    overview = _save_all_routes_overview(frames, cfg.out_route_dir)
    outputs.append(overview)
    summary_path = cfg.out_route_dir / "route_products_summary_4.1.json"
    _save_json(
        summary_path,
        {
            "version": "4.1",
            "prediction_source": str(source),
            "excluded_ground_routes": ["A1", "A2", "A3"],
            "representative_flights": {route: _flight_label(value) for route, value in selected.items()},
            "comparison_figure": overview,
            "routes": summaries,
        },
    )
    outputs.append(str(summary_path))
    return outputs


def generate_all_visualizations(cfg: ExperimentConfig) -> dict:
    """统一生成4.1的中文SVG、GIF、HTML、CSV与JSON追踪产物。"""

    ensure_directories(cfg)
    font_name = _style()
    for name in ("training", "results", "prediction", "custom"):
        (cfg.out_figure_dir / name).mkdir(parents=True, exist_ok=True)
    categories = {
        "training": plot_training_outputs(cfg),
        "results": [*plot_result_outputs(cfg), *plot_state_outputs(cfg)],
        "prediction": plot_prediction_outputs(cfg),
        "tasks": plot_task_outputs(cfg),
        "rls_parameters": plot_rls_parameter_outputs(cfg),
        "rls_correction": plot_rls_correction_outputs(cfg),
        "routes": generate_route_products(cfg),
    }
    files = [path for paths in categories.values() for path in paths]
    summary = {
        "version": "4.1",
        "chart_font": font_name,
        "svg_text_mode": "path",
        "all_chart_labels": "中文",
        "no_png_outputs": True,
        "svg_count": int(sum(path.lower().endswith(".svg") for path in files)),
        "gif_count": int(sum(path.lower().endswith(".gif") for path in files)),
        "html_count": int(sum(path.lower().endswith(".html") for path in files)),
        "categories": categories,
        "files": files,
    }
    _save_json(cfg.visualization_summary_json, summary)
    from report import generate_output_report
    summary["output_report"] = str(generate_output_report(cfg))
    _save_json(cfg.visualization_summary_json, summary)
    return summary
