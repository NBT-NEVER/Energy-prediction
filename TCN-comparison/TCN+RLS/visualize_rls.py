# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: visualize_rls.py
# 开发时间: 2026-10-30
# 文件名: visualize_rls.py
# 功能说明: 绘制四组TCN加RLS误差、区间覆盖率和在线轨迹图表
# 版本号：1.0

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from config import GROUPS, ExperimentConfig, figure_path, group_paths


def _save(fig, cfg: ExperimentConfig, name: str) -> list[str]:
    """功能: 保存PNG和SVG图表。
    参数: fig为matplotlib图形，cfg为配置，name为图表名。
    返回: 两个图表路径字符串。
    调用位置: plot_all。
    """
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    png, svg = figure_path(cfg, name), figure_path(cfg, name, "svg")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(svg)]


def plot_all(cfg: ExperimentConfig) -> dict:
    """功能: 从当前CSV结果生成全部RLS对照图。
    参数: cfg为实验配置。
    返回: 图表名称到文件路径的字典。
    调用位置: main.py。
    """
    metrics = pd.read_csv(cfg.summary_csv).sort_values("group")
    paths: dict[str, list[str]] = {}
    groups = metrics.group.tolist()
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.36
    ax.bar(x - width / 2, metrics.tcn_flight_energy_wape_percent, width, label="TCN基线")
    ax.bar(x + width / 2, metrics.flight_energy_wape_percent, width, label="TCN+RLS")
    ax.set_xticks(x, groups); ax.set_ylabel("Flight能耗 WAPE (%)")
    ax.set_title("四组TCN基线与RLS校正能耗误差"); ax.legend(); ax.grid(axis="y", alpha=.25)
    for col, offset in [("tcn_flight_energy_wape_percent", -width / 2), ("flight_energy_wape_percent", width / 2)]:
        for i, value in enumerate(metrics[col]): ax.text(i + offset, value + .05, f"{value:.2f}", ha="center", fontsize=8)
    paths["flight_wape"] = _save(fig, cfg, "flight_energy_wape")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    coverage_cols = ["sample_power_interval_coverage_percent", "window_interval_coverage_percent", "flight_interval_coverage_percent", "remaining_trace_interval_coverage_percent"]
    labels = ["功率采样", "RLS窗口", "任务前总能耗", "在线轨迹(含初始)"]
    width = .18
    for i, (col, label) in enumerate(zip(coverage_cols, labels)):
        ax.bar(x + (i - 1.5) * width, metrics[col], width, label=label)
    ax.axhline(metrics.confidence_level.iloc[0] * 100, color="black", ls="--", lw=1.2, label="名义95%")
    ax.set_xticks(x, groups); ax.set_ylabel("覆盖率 (%)"); ax.set_ylim(0, 105)
    ax.set_title("四组置信区间覆盖率"); ax.legend(ncol=3, fontsize=8); ax.grid(axis="y", alpha=.25)
    paths["coverage"] = _save(fig, cfg, "interval_coverage")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(metrics.flight_interval_mean_width_wh, metrics.flight_interval_coverage_percent, s=70)
    for _, row in metrics.iterrows(): ax.annotate(row.group, (row.flight_interval_mean_width_wh, row.flight_interval_coverage_percent), xytext=(5, 4), textcoords="offset points")
    ax.axhline(metrics.confidence_level.iloc[0] * 100, color="black", ls="--", lw=1)
    ax.set_xlabel("任务前区间平均宽度 (Wh)"); ax.set_ylabel("任务前覆盖率 (%)")
    ax.set_title("置信区间覆盖率与平均宽度"); ax.grid(alpha=.25)
    paths["coverage_width"] = _save(fig, cfg, "interval_coverage_width")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for group in groups:
        tuning = pd.read_csv(group_paths(cfg, group)["rls_tuning"]).groupby("window_seconds", as_index=False).flight_energy_wape_percent.min()
        ax.plot(tuning.window_seconds, tuning.flight_energy_wape_percent, marker="o", label=group)
    ax.set_xlabel("RLS反馈窗口 (s)"); ax.set_ylabel("验证Flight能耗 WAPE (%)")
    ax.set_title("RLS时间窗搜索结果（每个时间窗取最优参数）"); ax.legend(); ax.grid(alpha=.25)
    paths["window_search"] = _save(fig, cfg, "rls_window_search")

    reference_group = groups[0]
    reference = pd.read_csv(group_paths(cfg, reference_group)["test_predictions"])
    representative = int(reference.groupby("flight").size().sort_values().index[len(reference.flight.unique()) // 2])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    part = reference[reference.flight == representative]
    ax.plot(part.time_s, part.power_w, color="black", lw=1.2, label="实际功率")
    for group in groups:
        data = pd.read_csv(group_paths(cfg, group)["test_predictions"])
        data = data[data.flight == representative]
        ax.plot(data.time_s, data.predicted_power_w, lw=1.0, label=f"{group} RLS")
    ax.set_xlabel("任务时间 (s)"); ax.set_ylabel("功率 (W)")
    ax.set_title(f"代表测试flight {representative} 的TCN+RLS功率预测"); ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.25)
    paths["representative_power"] = _save(fig, cfg, "representative_power")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, group in zip(axes.flat, groups):
        trace = pd.read_csv(group_paths(cfg, group)["test_trace"])
        trace = trace[(trace.flight == representative) & (trace.remaining_window_count > 0)]
        ax.plot(trace.observed_window_count, trace.predicted_final_energy_wh, lw=1.2, label=f"{group}预测")
        ax.fill_between(trace.observed_window_count, trace.predicted_final_lower_wh,
                        trace.predicted_final_upper_wh, alpha=.2, label="95%名义预测区间")
        ax.axhline(trace.actual_final_energy_wh.iloc[0], color="black", ls="--", label="实际总能耗")
        ax.set_xlabel("已完成反馈窗口数"); ax.set_ylabel("最终总能耗预测 (Wh)")
        ax.set_title(f"{group}组 / flight {representative}"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle("相同代表任务的RLS在线预测与区间（完成反馈后更新）")
    fig.tight_layout()
    paths["remaining_trace"] = _save(fig, cfg, "remaining_energy_trace")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for group in groups:
        stages = pd.read_csv(group_paths(cfg, group)["stage_intervals"])
        ax.plot(stages.observed_fraction * 100, stages.coverage_percent, marker="o", label=group)
    ax.axhline(metrics.confidence_level.iloc[0] * 100, color="black", ls="--", label="名义95%")
    ax.set_xlabel("已观察任务进度阈值 (%)"); ax.set_ylabel("任务总能耗区间覆盖率 (%)")
    ax.set_xticks([0, 25, 50, 75]); ax.set_ylim(0, 105)
    ax.set_title("任务不同阶段的区间覆盖率（每个flight每阶段一个点）")
    ax.legend(); ax.grid(alpha=.25)
    paths["stage_coverage"] = _save(fig, cfg, "stage_interval_coverage")
    return paths
