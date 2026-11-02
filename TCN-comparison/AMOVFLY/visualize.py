# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: visualize.py
# 开发时间: 2026-11-02
# 文件名: visualize.py
# 功能说明: 绘制AMOVFLY场景、数据分布、异常和典型航线统计图
# 版本号：1.0

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from analysis import NUMERIC_METADATA
from config import AnalysisConfig


COLORS = {"FAFS": "#2878B5", "FAVS": "#E98B2A", "VAFS": "#27976E", "VAVS": "#C8444B", "Random": "#7B61A8"}


def setup_style() -> None:
    """功能: 设置中文字体、坐标轴和输出图形风格。
    参数: 无。
    返回: None。
    调用位置: create_figures。
    """
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
        if name in names:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.dpi": 180})


def save_figure(fig, cfg: AnalysisConfig, stem: str) -> list[str]:
    """功能: 将图形保存为PNG和SVG。
    参数: fig为Matplotlib图形，cfg为输出配置，stem为图名。
    返回: 保存文件路径列表。
    调用位置: create_figures。
    """
    fig.tight_layout()
    paths = []
    for suffix in ("png", "svg"):
        path = cfg.figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def create_figures(cfg: AnalysisConfig, result: dict) -> list[str]:
    """功能: 使用统计表和原始ready记录绘制全套分析图。
    参数: cfg为配置，result为analyze_dataset的返回值。
    返回: 本次图形文件路径列表。
    调用位置: main.py。
    """
    setup_style()
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    flights = result["frames"]["flights"]
    scene = result["frames"]["scene"].set_index("scene")
    stats = result["frames"]["stats"].set_index("字段")
    paths = []

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    counts = flights.scene.value_counts().reindex(cfg.scenes, fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[COLORS[s] for s in counts.index])
    ax.set(title="AMOVFLY各场景ready飞行文件数", ylabel="文件数")
    ax.grid(axis="y", alpha=.22)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, value, str(value), ha="center", va="bottom")
    paths += save_figure(fig, cfg, "01_scene_flight_counts")

    fig, ax = plt.subplots(figsize=(9, 5))
    values = [flights.loc[flights.scene == s, "duration_s"].dropna().to_numpy()/60 for s in cfg.scenes]
    box = ax.boxplot(values, tick_labels=cfg.scenes, patch_artist=True, showfliers=False)
    for patch, scene_name in zip(box["boxes"], cfg.scenes):
        patch.set_facecolor(COLORS[scene_name]); patch.set_alpha(.72)
    ax.set(title="单次飞行时长分布", ylabel="时长 (min)")
    ax.grid(axis="y", alpha=.22)
    paths += save_figure(fig, cfg, "02_duration_by_scene")

    missing = result["frames"]["dimensions"].query("缺失数 > 0").sort_values("缺失率_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.2, .3*len(missing))))
    if len(missing):
        ax.barh(missing["字段"], missing["缺失率_percent"], color="#C8444B")
        ax.set_xlabel("缺失比例 (%)")
    else:
        ax.text(.5, .5, "未发现字段缺失", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("字段缺失比例（全体原始记录）")
    ax.grid(axis="x", alpha=.22)
    paths += save_figure(fig, cfg, "03_missingness")

    distribution_fields = ["wind_speed", "gps_z", "v_x", "v_y", "v_z", "power",
                           "battery_voltage", "battery_current", "battery_remain"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, column in zip(axes.flat, distribution_fields):
        meta = NUMERIC_METADATA[column]
        q = stats.loc[column]
        lower, upper = max(q["P01"], q["IQR下界"]), min(q["P99"], q["IQR上界"])
        for scene_name in cfg.scenes:
            files = flights.loc[flights.scene == scene_name, "source_file"]
            samples = []
            for file in files:
                data = pd.read_csv(file, usecols=[column], encoding="utf-8-sig")
                values = pd.to_numeric(data[column], errors="coerce").dropna().to_numpy()
                samples.extend(values[::max(1, len(values)//4000)][:4000])
            samples = np.asarray(samples)
            samples = samples[(samples >= lower) & (samples <= upper)]
            if len(samples):
                ax.hist(samples, bins=28, density=True, histtype="step", linewidth=1.3,
                        color=COLORS[scene_name], label=scene_name)
        ax.set_title(f"{column} ({meta[0]})")
        ax.set_ylabel("概率密度")
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("关键飞行状态与能耗变量分布（按场景）", y=1.01)
    paths += save_figure(fig, cfg, "04_feature_distributions")

    columns = ["wind_speed", "gps_z", "v_x", "v_y", "v_z", "battery_voltage",
               "battery_current", "battery_remain", "power"]
    corr_parts = []
    for file in flights.source_file:
        sample = pd.read_csv(file, usecols=columns, encoding="utf-8-sig").iloc[::10].copy()
        sample["speed_mps"] = np.sqrt(sample[["v_x", "v_y", "v_z"]].pow(2).sum(axis=1))
        corr_parts.append(sample)
    corr = pd.concat(corr_parts, ignore_index=True).corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    fig.colorbar(image, ax=ax, label="Pearson相關係數")
    ax.set_title("飛行状态与能耗变量相关矩阵（每10行抽样）")
    paths += save_figure(fig, cfg, "05_correlation_heatmap")

    representatives = result["representatives"]
    fig, axes = plt.subplots(len(representatives), 2, figsize=(12, 3.4*len(representatives)))
    if len(representatives) == 1:
        axes = np.asarray([axes])
    for row, representative in enumerate(representatives):
        frame = pd.read_csv(representative["source_file"], encoding="utf-8-sig")
        ax = axes[row, 0]
        valid_gps = frame.real_lat.between(-90, 90) & frame.real_long.between(-180, 180)
        latitude0 = float(frame.loc[valid_gps, "real_lat"].iloc[0])
        longitude0 = float(frame.loc[valid_gps, "real_long"].iloc[0])
        gps_north = (frame.real_lat - latitude0) * 111320.
        gps_east = (frame.real_long - longitude0) * 111320. * np.cos(np.deg2rad(latitude0))
        points = ax.scatter(gps_east[valid_gps], gps_north[valid_gps], c=frame.loc[valid_gps, "time"], s=5, cmap="viridis")
        ax.plot(gps_east[valid_gps].iloc[0], gps_north[valid_gps].iloc[0], marker="o", color="#202020", label="起点")
        valid = (frame.aim_lat.abs() > 1) & (frame.aim_long.abs() > 1)
        if valid.any():
            north = (frame.loc[valid, "aim_lat"] - latitude0) * 111320.
            east = (frame.loc[valid, "aim_long"] - longitude0) * 111320. * np.cos(np.deg2rad(latitude0))
            ax.plot(east, north, "k--", linewidth=1, alpha=.65, label="参考航点连线（经纬度局部投影）")
            ax.scatter(east, north, s=10, marker="x", color="#C8444B", label="参考航点")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="局部东向位置 (m)", ylabel="局部北向位置 (m)",
               title=f"{representative['scene']} / {representative['flight_name']} 实际航迹")
        ax.legend(fontsize=7, loc="best")
        fig.colorbar(points, ax=ax, label="任务时间 (s)", fraction=.046)
        ax = axes[row, 1]
        ax.plot(frame.time, frame.gps_z, label="实际高度 gps_z", color=COLORS[representative["scene"]])
        if "aim_z" in frame:
            ax.step(frame.time, frame.aim_z, label="参考高度 aim_z", color="#444444", alpha=.7, where="post")
        ax.set(xlabel="任务时间 (s)", ylabel="高度 (m)", title="实际与参考高度")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    paths += save_figure(fig, cfg, "06_representative_trajectories")

    fields = ["power", "battery_voltage", "battery_current", "wind_speed", "gps_z", "speed_mps"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.flat, fields):
        candidates = stats.loc[column, "IQR极端候选数"]
        ax.bar(["Tukey范围内", "极端值候选"],
               [stats.loc[column, "有效数"]-candidates, candidates], color=["#2878B5", "#E98B2A"])
        ax.set_title(f"{column}：{int(candidates):,}个候选")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("关键变量的IQR统计极端值候选（并非自动判错）", y=1.02)
    paths += save_figure(fig, cfg, "07_iqr_extreme_candidates")

    intervals = []
    for file in flights.source_file:
        time_values = pd.read_csv(file, usecols=["time"], encoding="utf-8-sig").time.to_numpy(float)
        intervals.extend(np.diff(time_values).tolist())
    intervals = np.asarray(intervals)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0) & (intervals <= 2.)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist(intervals, bins=60, color="#27976E", alpha=.86)
    ax.axvline(cfg.expected_sample_seconds, color="#C8444B", linestyle="--", label="名义采样周期0.2s")
    ax.set(title="相邻采样间隔分布（排除非正和大于2秒间隔）", xlabel="采样间隔 (s)", ylabel="间隔数量")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    paths += save_figure(fig, cfg, "08_sampling_interval_distribution")
    return paths
