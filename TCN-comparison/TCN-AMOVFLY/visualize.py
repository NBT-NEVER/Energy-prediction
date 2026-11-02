# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: visualize.py
# 开发时间: 2026-11-01
# 文件名: visualize.py
# 功能说明: 从CSV绘制五组TCN功率、能耗、调参与训练曲线
# 版本号：1.0

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from config import ExperimentConfig, group_paths, figure_path


COLORS = {"A": "#2878B5", "B": "#E98B2A", "C": "#27976E", "D": "#C8444B", "E": "#7B61A8"}


def setup_fonts():
    """功能: 选择已安装的中文字体并设置图形风格。
    参数: 无。
    返回: None。
    调用位置: plot_results。
    """

    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
        if name in names:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 12, "figure.dpi": 160})


def save_figure(fig, cfg, name):
    """功能: 将同一图形保存为PNG和可缩放SVG。
    参数: fig为图形，cfg为配置，name为图形名称。
    返回: 输出路径列表。
    调用位置: plot_results。
    """

    fig.tight_layout()
    paths = []
    for suffix in ("png", "svg"):
        path = figure_path(cfg, name, suffix)
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_results(cfg: ExperimentConfig) -> list[str]:
    """功能: 绘制已完成实验组的误差、曲线及训练调参图。
    参数: cfg为实验配置。
    返回: 本次生成的图表路径。
    调用位置: main.py。
    """

    setup_fonts()
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(cfg.summary_csv).sort_values("group")
    groups = metrics.group.tolist()
    paths = []
    predictions = {g: pd.read_csv(group_paths(cfg, g)["predictions"]) for g in groups}
    flights = {g: pd.read_csv(group_paths(cfg, g)["flight_metrics"]) for g in groups}
    for key, ylabel, name in (
        ("power_wape_percent", "功率 WAPE (%)", "power_wape"),
        ("flight_energy_wape_percent", "Flight 总能耗 WAPE (%)", "flight_energy_wape"),
        ("window_energy_wape_percent", "1 s 窗口能耗 WAPE (%)", "window_energy_wape"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        bars = ax.bar(groups, metrics[key], color=[COLORS[g] for g in groups], width=.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"离线TCN测试集：{ylabel}")
        ax.set_ylim(0, metrics[key].max() * 1.18)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, metrics[key]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + metrics[key].max() * .015,
                    f"{value:.3f}", ha="center", va="bottom")
        paths.extend(save_figure(fig, cfg, name))

    reference = flights[groups[0]].sort_values("duration_s")
    representative = int(reference.iloc[len(reference) // 2].flight)
    for cumulative, name in ((False, "representative_power"), (True, "representative_energy")):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        actual = predictions[groups[0]].query("flight == @representative")
        ax.plot(actual.time_s, actual.actual_cumulative_energy_wh if cumulative else actual.power_w,
                label="实际值", color="#202020", linewidth=1.5, alpha=.8)
        for group in groups:
            data = predictions[group].query("flight == @representative")
            ax.plot(data.time_s, data.predicted_cumulative_energy_wh if cumulative else data.predicted_power_w,
                    label=f"{group} 预测", color=COLORS[group], linewidth=1.15, alpha=.9)
        ax.set_xlabel("任务时间 (s)")
        ax.set_ylabel("累计能耗 (Wh)" if cumulative else "功率 (W)")
        ax.set_title(f"中位时长测试任务 flight {representative}：{'累计能耗' if cumulative else '功率'}")
        ax.legend(ncol=5)
        ax.grid(alpha=.22)
        paths.extend(save_figure(fig, cfg, name))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    max_energy = max(data.actual_energy_wh.max() for data in flights.values()) * 1.05
    for group in groups:
        data = flights[group]
        axes[0].scatter(data.actual_energy_wh, data.predicted_energy_wh, s=22, alpha=.65,
                        label=group, color=COLORS[group])
        axes[1].scatter(data.flight, data.error_wh, s=22, alpha=.7, label=group, color=COLORS[group])
    axes[0].plot([0, max_energy], [0, max_energy], "k--", linewidth=1)
    axes[0].set(xlabel="实际 flight 能耗 (Wh)", ylabel="预测 flight 能耗 (Wh)", title="逐任务能耗一致性")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set(xlabel="Flight 编号", ylabel="预测 − 实际能耗 (Wh)", title="逐任务带符号误差")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend()
    paths.extend(save_figure(fig, cfg, "flight_energy_scatter"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    boxes = ax.boxplot([flights[g].absolute_error_wh for g in groups], tick_labels=groups,
                       patch_artist=True, showmeans=True)
    for box, group in zip(boxes["boxes"], groups):
        box.set_facecolor(COLORS[group])
        box.set_alpha(.65)
    ax.set(ylabel="Flight 总能耗绝对误差 (Wh)", title="测试任务能耗误差分布")
    ax.grid(axis="y", alpha=.22)
    paths.extend(save_figure(fig, cfg, "flight_error_distribution"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for group in groups:
        data = pd.read_csv(group_paths(cfg, group)["tuning"]).sort_values("window_seconds")
        ax.plot(data.window_seconds, data.selection_score, "o-", label=group, color=COLORS[group])
    ax.set(xlabel="TCN 输入窗口名义时长 (s)", ylabel="验证集综合选择分数", title=f"每候选{cfg.tune_epochs}轮TCN窗口搜索")
    ax.legend()
    ax.grid(alpha=.22)
    paths.extend(save_figure(fig, cfg, "window_search"))

    ncols = 3 if len(groups) > 4 else 2
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, group in zip(axes, groups):
        data = pd.read_csv(group_paths(cfg, group)["training_log"])
        ax.plot(data.epoch, data.train_loss, label="训练", color=COLORS[group])
        ax.plot(data.epoch, data.val_loss, label="验证", color="#404040", linestyle="--")
        ax.set(xlabel="Epoch", ylabel="标准化SmoothL1损失", title=f"{group} 组正式训练")
        ax.grid(alpha=.22)
        ax.legend()
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    paths.extend(save_figure(fig, cfg, "training_loss"))
    return paths
