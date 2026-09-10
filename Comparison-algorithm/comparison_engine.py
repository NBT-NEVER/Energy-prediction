# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: comparison_engine.py
# 开发时间: 2026-09-09
# 文件名: comparison_engine.py
# 功能说明: 在3.0固定数据划分上训练和评估多种对比算法并生成多维图表
# 版本号：3.0

from __future__ import annotations

import json
import gc
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import (ALGORITHMS, BATCH_SIZE, DEEP_EPOCHS, DEEP_PATIENCE, FEATURE_COLUMNS,
                    FIGURE_NAMES, RANDOM_SEED,
                    MODEL_CANDIDATES, SEQUENCE_WINDOW, TEST_CSV, TRAIN_CSV, VAL_CSV, ExperimentConfig,
                    ensure_directories)

PHASE_NAMES = ["idle", "ascent", "descent", "cruise"]
POWER_BIN_EDGES = [-np.inf, 50.0, 300.0, 450.0, 600.0, np.inf]
POWER_BIN_NAMES = ["0-50 W", "50-300 W", "300-450 W", "450-600 W", ">600 W"]


def figure_file(directory: Path, key: str) -> Path:
    """功能: 根据统一配置返回图表输出路径。
    参数: directory为目标图表目录，key为图表配置键。
    返回: 规范化的PNG路径。
    调用位置: 所有绘图函数。
    """
    return directory / FIGURE_NAMES[key]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """功能: 计算功率回归指标。
    参数: y为真实功率，pred为预测功率。
    返回: MAE、RMSE、R2、MAPE和WAPE。
    调用位置: run_algorithm。
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    e = pred - y
    den = np.sum((y - y.mean()) ** 2)
    valid = np.abs(y) > 1e-6
    return {
        "sample_power_mae_w": float(np.mean(np.abs(e))),
        "sample_power_rmse_w": float(np.sqrt(np.mean(e ** 2))),
        "sample_power_r2": float(1 - np.sum(e ** 2) / den) if den > 1e-12 else 0.0,
        "sample_power_mape_percent": float(np.mean(np.abs(e[valid] / y[valid])) * 100) if valid.any() else 0.0,
        "sample_power_wape_percent": float(np.sum(np.abs(e)) / max(np.sum(np.abs(y)), 1e-6) * 100),
    }


def flight_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """功能: 计算flight级能耗指标。
    参数: frame为测试数据，pred为逐点功率预测。
    返回: flight级MAE、RMSE、R2、MAPE和WAPE。
    调用位置: run_algorithm。
    """
    temp = frame[["flight", "dt_seconds", "power_w"]].copy()
    temp["pred"] = pred
    temp["actual_wh"] = temp.power_w * temp.dt_seconds / 3600
    temp["pred_wh"] = temp.pred * temp.dt_seconds / 3600
    g = temp.groupby("flight", sort=True)[["actual_wh", "pred_wh"]].sum()
    base = metrics(g.actual_wh.to_numpy(), g.pred_wh.to_numpy())
    return {
        "flight_energy_mae_wh": base["sample_power_mae_w"],
        "flight_energy_rmse_wh": base["sample_power_rmse_w"],
        "flight_energy_r2": base["sample_power_r2"],
        "flight_energy_mape_percent": base["sample_power_mape_percent"],
        "flight_energy_wape_percent": base["sample_power_wape_percent"],
    }


def build_diagnostic_frame(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """功能: 构造图表诊断所需的逐样本误差表。
    参数: frame为测试数据，pred为预测功率数组。
    返回: 含残差、能耗、功率分箱和飞行阶段字段的数据表。
    调用位置: plot_algorithm_diagnostics和plot_comparison_diagnostics。
    """
    out = frame[["flight", "dt_seconds", "power_w", "time", "actual_speed_mps", "horizontal_speed_mps", "vertical_speed_mps", "payload_kg", "wind_speed"]].copy()
    out["predicted_power_w"] = np.asarray(pred, dtype=float)
    out["residual_w"] = out["predicted_power_w"] - out["power_w"]
    out["absolute_error_w"] = out["residual_w"].abs()
    out["actual_energy_wh"] = out["power_w"] * out["dt_seconds"] / 3600.0
    out["predicted_energy_wh"] = out["predicted_power_w"] * out["dt_seconds"] / 3600.0
    out["phase"] = np.asarray(phase_label(frame), dtype=int)
    out["phase_name"] = out["phase"].map(dict(enumerate(PHASE_NAMES)))
    out["power_bin"] = pd.cut(out["power_w"], POWER_BIN_EDGES, labels=POWER_BIN_NAMES)
    return out


def save_plot(fig: plt.Figure, path: Path) -> None:
    """功能: 统一保存高分辨率图表并关闭画布。
    参数: fig为Matplotlib画布，path为输出文件路径。
    返回: 无。
    调用位置: 各图表诊断函数。
    """
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_algorithm_diagnostics(name: str, frame: pd.DataFrame, pred: np.ndarray, cfg: ExperimentConfig) -> None:
    """功能: 为单个算法生成多维预测效果图。
    参数: name为算法标识，frame为测试数据，pred为预测功率，cfg为输出配置。
    返回: 无，图表写入算法out/figures目录。
    调用位置: run_algorithm。
    """
    diag = build_diagnostic_frame(frame, pred)
    figure_dir = cfg.figure_dir
    # 1. 残差分布：检查偏差方向、离群点和误差尾部。
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diag["residual_w"], bins=60, color="#4472C4", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#C00000", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (predicted - measured, W)"); ax.set_ylabel("Count"); ax.set_title(f"{name}: residual distribution")
    save_plot(fig, figure_file(figure_dir, "residual_histogram"))
    # 2. 残差与真实功率关系：检查异方差和低/高功率段系统性误差。
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = diag.iloc[::max(1, len(diag) // 10000)]
    scatter = ax.scatter(sample["power_w"], sample["residual_w"], s=5, alpha=0.25, c=sample["phase"], cmap="viridis")
    # 阶段颜色图例说明，避免只看颜色而无法判断阶段。
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=plt.get_cmap("viridis")(k / 3), markersize=5, label=PHASE_NAMES[k]) for k in range(4)]
    ax.legend(handles=handles, title="Phase", fontsize=7, title_fontsize=7, ncol=4, loc="upper right")
    ax.axhline(0, color="#C00000", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Measured power (W)"); ax.set_ylabel("Residual (W)"); ax.set_title(f"{name}: residual vs measured power")
    save_plot(fig, figure_file(figure_dir, "residual_vs_actual"))
    # 3. 功率区间误差：展示不同功率水平下的MAE和样本量。
    grouped = diag.groupby("power_bin", observed=False).agg(mae=("absolute_error_w", "mean"), rows=("power_w", "size")).reindex(POWER_BIN_NAMES)
    fig, ax1 = plt.subplots(figsize=(7, 4)); ax2 = ax1.twinx()
    x = np.arange(len(POWER_BIN_NAMES)); ax1.bar(x, grouped["mae"], color="#ED7D31", alpha=0.85, label="MAE")
    ax2.plot(x, grouped["rows"], color="#4472C4", marker="o", linewidth=1.8, label="Samples")
    ax1.set_xticks(x, POWER_BIN_NAMES); ax1.set_ylabel("MAE (W)"); ax2.set_ylabel("Samples"); ax1.set_title(f"{name}: error by measured-power bin")
    save_plot(fig, figure_file(figure_dir, "power_bin_error"))
    # 4. flight级真实与预测能耗：检查累计误差和相对偏差。
    flights = diag.groupby("flight", sort=True).agg(actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"))
    fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(flights["actual_energy_wh"], flights["predicted_energy_wh"], s=28, alpha=.8, color="#4472C4")
    lim = max(flights.to_numpy().max(), 1.0); ax.plot([0, lim], [0, lim], "k--", linewidth=1.0); ax.set_xlabel("Measured flight energy (Wh)"); ax.set_ylabel("Predicted flight energy (Wh)"); ax.set_title(f"{name}: flight-energy agreement")
    save_plot(fig, figure_file(figure_dir, "flight_energy_scatter"))
    # 5. flight级误差排序：突出最差和最好任务，避免只看总体均值。
    flights["error_wh"] = flights["predicted_energy_wh"] - flights["actual_energy_wh"]
    ordered = flights.sort_values("error_wh")
    fig, ax = plt.subplots(figsize=(8, 5)); colors = ["#C00000" if v > 0 else "#4472C4" for v in ordered["error_wh"]]
    ax.barh(np.arange(len(ordered)), ordered["error_wh"], color=colors); ax.axvline(0, color="black", linewidth=.8); ax.set_yticks(np.arange(len(ordered)), ordered.index.astype(str), fontsize=6); ax.set_xlabel("Energy error (Wh)"); ax.set_ylabel("Flight"); ax.set_title(f"{name}: flight-energy error ranking")
    save_plot(fig, figure_file(figure_dir, "flight_energy_error_ranking"))
    # 6. 累计能耗曲线：观察误差是否随任务进程累积。
    fig, ax = plt.subplots(figsize=(8, 4)); ids = diag.flight.drop_duplicates().to_numpy()[:3]
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for fid in ids:
        part = diag[diag.flight == fid].sort_values("time").copy(); actual = part.actual_energy_wh.cumsum(); predicted = part.predicted_energy_wh.cumsum(); color = colors[len(ax.lines) // 2]
        ax.plot(part.time, actual, color=color, linewidth=1.5, label=f"flight {fid} measured"); ax.plot(part.time, predicted, "--", color=color, linewidth=1.2, label=f"flight {fid} predicted")
    ax.set_xlabel("Flight time (s)"); ax.set_ylabel("Cumulative energy (Wh)"); ax.set_title(f"{name}: cumulative energy")
    ax.legend(ncol=2, fontsize=7); save_plot(fig, figure_file(figure_dir, "cumulative_energy_timeseries"))
    # 7. 阶段误差：比较不同运动阶段的MAE、RMSE和WAPE。
    phase_rows = []
    for phase, part in diag.groupby("phase_name", sort=False):
        mm = metrics(part.power_w.to_numpy(), part.predicted_power_w.to_numpy()); phase_rows.append({"phase": phase, "MAE": mm["sample_power_mae_w"], "RMSE": mm["sample_power_rmse_w"], "WAPE": mm["sample_power_wape_percent"]})
    phase_df = pd.DataFrame(phase_rows).set_index("phase").reindex(PHASE_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4)); phase_df["MAE"].plot.bar(ax=axes[0], color="#4472C4", title="MAE (W)"); phase_df["RMSE"].plot.bar(ax=axes[1], color="#ED7D31", title="RMSE (W)"); phase_df["WAPE"].plot.bar(ax=axes[2], color="#70AD47", title="WAPE (%)")
    for ax in axes: ax.set_xlabel("Phase"); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"{name}: phase-wise error", y=1.03); save_plot(fig, figure_file(figure_dir, "phase_error_metrics"))
    # 8. 绝对误差累积分布：比较大误差样本比例。
    values = np.sort(diag.absolute_error_w.to_numpy()); cdf = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(values, cdf, color="#4472C4"); ax.axhline(.9, color="#C00000", linestyle="--", linewidth=.9); ax.set_xlabel("Absolute error (W)"); ax.set_ylabel("Empirical CDF"); ax.set_title(f"{name}: absolute-error CDF"); ax.set_ylim(0, 1.02)
    save_plot(fig, figure_file(figure_dir, "absolute_error_cdf"))
    # 9. 局部放大图：以最大绝对误差为中心展示真实线、预测线和残差线。
    center = int(np.argmax(diag.absolute_error_w.to_numpy()))
    flight_id = diag.iloc[center].flight
    flight_part = diag[diag.flight == flight_id].reset_index(drop=True)
    local_center = int(np.argmax(flight_part.absolute_error_w.to_numpy()))
    local_start = max(0, local_center - 100)
    local_end = min(len(flight_part), local_center + 101)
    zoom = flight_part.iloc[local_start:local_end]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(zoom.time, zoom.power_w, color="#1F4E79", linewidth=1.5, label="Measured power")
    axes[0].plot(zoom.time, zoom.predicted_power_w, color="#C55A11", linewidth=1.3, linestyle="--", label="Predicted power")
    axes[0].set_ylabel("Power (W)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].plot(zoom.time, zoom.residual_w, color="#7030A0", linewidth=1.1, label="Residual")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[1].set_xlabel("Flight time (s)"); axes[1].set_ylabel("Residual (W)"); axes[1].grid(alpha=.25)
    fig.suptitle(f"{name}: local response around the largest error (flight {flight_id})")
    save_plot(fig, figure_file(figure_dir, "prediction_zoom"))
    # 10. flight能耗误差分布：同时标出零误差和均值，检查系统性高估或低估。
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flights.error_wh, bins=min(12, max(5, len(flights) // 2)), color="#5B9BD5", alpha=.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label="Zero error")
    ax.axvline(flights.error_wh.mean(), color="#C00000", linewidth=1.2, label="Mean error")
    ax.set_xlabel("Flight-energy error (Wh)"); ax.set_ylabel("Flights"); ax.set_title(f"{name}: flight-energy error distribution"); ax.legend(fontsize=8)
    save_plot(fig, figure_file(figure_dir, "flight_energy_error_histogram"))
    # 11. 分阶段绝对误差箱线图：补充阶段均值柱图无法表达的离散程度。
    phase_data = [diag.loc[diag.phase_name == phase, "absolute_error_w"].to_numpy() for phase in PHASE_NAMES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(phase_data, labels=PHASE_NAMES, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#E2F0D9"}, medianprops={"color": "#C00000"})
    ax.set_xlabel("Flight phase"); ax.set_ylabel("Absolute power error (W)"); ax.set_title(f"{name}: phase error distribution"); ax.grid(axis="y", alpha=.25)
    save_plot(fig, figure_file(figure_dir, "phase_error_distribution"))


def phase_label(frame: pd.DataFrame, vertical_threshold: float = 0.15,
                horizontal_threshold: float = 1.0) -> np.ndarray:
    """功能: 按运动状态生成经验剖面阶段标签。
    参数: frame为含速度特征的数据表。
    返回: idle、ascent、descent、cruise四类整数标签。
    调用位置: rf_tlatt_lite和诊断绘图。
    """
    vz = frame.vertical_speed_mps.to_numpy()
    hs = frame.horizontal_speed_mps.to_numpy()
    labels = np.full(len(frame), 3, dtype=int)
    labels[np.abs(vz) < vertical_threshold] = 0
    labels[vz > vertical_threshold] = 1
    labels[vz < -vertical_threshold] = 2
    labels[(np.abs(vz) < vertical_threshold) & (hs > horizontal_threshold)] = 3
    return labels


def _ridge_solution(x_train: np.ndarray, y_train: np.ndarray, alpha: float):
    """对非截距列标准化后求岭回归闭式解。"""
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-9, 1.0, feature_std)
    normalized = (x_train - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = (
        np.linalg.lstsq(design, y_train, rcond=None)[0]
        if alpha == 0.0
        else np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    )

    def predict(x: np.ndarray) -> np.ndarray:
        values = (x - feature_mean) / feature_std
        return np.column_stack([np.ones(len(values)), values]) @ coefficients

    return predict


def _validation_score(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    """用验证集功率 WAPE 与 flight 能耗 WAPE 的加权和选择参数。"""
    return (
        metrics(frame.power_w.to_numpy(float), prediction)["sample_power_wape_percent"]
        + 0.5 * flight_metrics(frame, prediction)["flight_energy_wape_percent"]
    )


def physical_mlr(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 cfg: ExperimentConfig) -> np.ndarray:
    def design(f: pd.DataFrame) -> np.ndarray:
        mass = 1.0 + f.payload_kg.to_numpy(float)
        ax = f.dynamic_accel_norm.to_numpy(float)
        az = f.vertical_speed_abs_mps.to_numpy(float)
        vxy = f.horizontal_speed_mps.to_numpy(float)
        vz = f.vertical_speed_mps.to_numpy(float)
        moving = (f.actual_speed_mps.to_numpy(float) > 0.1).astype(float)
        return np.column_stack([
            moving,
            ax * mass,
            np.sign(vz) * az * mass,
            vxy ** 2 * mass ** (2 / 3),
            vz ** 2 * mass ** (2 / 3),
            mass,
            f.wind_speed.to_numpy(float),
        ])

    rows = []
    best = None
    for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
        predictor = _ridge_solution(design(train), train.power_w.to_numpy(float), alpha)
        validation_prediction = np.maximum(0.0, predictor(design(validation)))
        score = _validation_score(validation, validation_prediction)
        row = {
            "alpha": alpha,
            "validation_power_wape_percent": metrics(
                validation.power_w.to_numpy(float), validation_prediction
            )["sample_power_wape_percent"],
            "validation_flight_energy_wape_percent": flight_metrics(
                validation, validation_prediction
            )["flight_energy_wape_percent"],
            "selection_score": score,
        }
        rows.append(row)
        if best is None or score < best[0]:
            best = (score, alpha, predictor)
    tuning = pd.DataFrame(rows)
    tuning["selected"] = tuning["alpha"] == best[1]
    tuning.to_csv(cfg.tuning_csv, index=False, encoding="utf-8")
    return np.maximum(0.0, best[2](design(test)))


def rf_tlatt_lite(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                  cfg: ExperimentConfig) -> np.ndarray:
    """功能: 实现RF-TLATT的可复现轻量相位专用基线。
    参数: train和test为公共数据划分。
    返回: 按相位分别拟合的功率预测。
    调用位置: run_algorithm。
    """
    def design(f: pd.DataFrame) -> np.ndarray:
        speed = f.actual_speed_mps.to_numpy(float)
        vertical = f.vertical_speed_mps.to_numpy(float)
        payload = f.payload_kg.to_numpy(float)
        wind = f.wind_speed.to_numpy(float)
        return np.column_stack([
            speed,
            f.horizontal_speed_mps,
            vertical,
            f.vertical_speed_abs_mps,
            payload,
            f.altitude_m,
            wind,
            f.dynamic_accel_norm,
            f.angular_rate_norm,
            f.relative_air_speed_mps,
            f.wind_alignment,
            f.wind_cross_component_mps,
            speed ** 2,
            vertical ** 2,
            speed * payload,
            speed * wind,
        ])

    def fit_predict(source: pd.DataFrame, target: pd.DataFrame, vertical_threshold: float,
                    horizontal_threshold: float, alpha: float) -> np.ndarray:
        source_labels = phase_label(source, vertical_threshold, horizontal_threshold)
        target_labels = phase_label(target, vertical_threshold, horizontal_threshold)
        source_design = design(source)
        target_design = design(target)
        source_power = source.power_w.to_numpy(float)
        output = np.zeros(len(target), dtype=float)
        global_predictor = _ridge_solution(source_design, source_power, alpha)
        for phase_index in range(len(PHASE_NAMES)):
            source_mask = source_labels == phase_index
            target_mask = target_labels == phase_index
            predictor = (
                _ridge_solution(source_design[source_mask], source_power[source_mask], alpha)
                if source_mask.sum() >= source_design.shape[1] * 2
                else global_predictor
            )
            output[target_mask] = predictor(target_design[target_mask])
        return np.maximum(0.0, output)

    rows = []
    best = None
    for vertical_threshold, horizontal_threshold in ((0.10, 0.5), (0.15, 1.0), (0.25, 1.5)):
        for alpha in (0.01, 0.1, 1.0, 10.0):
            validation_prediction = fit_predict(
                train, validation, vertical_threshold, horizontal_threshold, alpha
            )
            score = _validation_score(validation, validation_prediction)
            row = {
                "vertical_threshold_mps": vertical_threshold,
                "horizontal_threshold_mps": horizontal_threshold,
                "alpha": alpha,
                "validation_power_wape_percent": metrics(
                    validation.power_w.to_numpy(float), validation_prediction
                )["sample_power_wape_percent"],
                "validation_flight_energy_wape_percent": flight_metrics(
                    validation, validation_prediction
                )["flight_energy_wape_percent"],
                "selection_score": score,
            }
            rows.append(row)
            if best is None or score < best[0]:
                best = (score, vertical_threshold, horizontal_threshold, alpha)
    tuning = pd.DataFrame(rows)
    tuning["selected"] = (
        (tuning["vertical_threshold_mps"] == best[1])
        & (tuning["horizontal_threshold_mps"] == best[2])
        & (tuning["alpha"] == best[3])
    )
    tuning.to_csv(cfg.tuning_csv, index=False, encoding="utf-8")
    return fit_predict(train, test, best[1], best[2], best[3])


class CausalConv1d(nn.Conv1d):
    """只在时间轴左侧补零的一维卷积，保证当前预测不读取未来样本。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(torch.nn.functional.pad(x, (left, 0)))


class LRTemporalBlock(nn.Module):
    """LR-TCN-SMA 的双卷积残差块。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, 3, dilation=dilation),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, 3, dilation=dilation),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + self.shortcut(x)


class SequenceRegressor(nn.Module):
    """根据候选参数构建 LSTM、LR-TCN-SMA、LSTM-Transformer 或 CNN-LSTM。"""

    def __init__(self, kind: str, features: int, params: dict[str, float | int | bool]):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            hidden = int(params["hidden_size"])
            layers = int(params["layers"])
            bidirectional = bool(params["bidirectional"])
            self.lstm_bidirectional = bidirectional
            self.body = nn.LSTM(
                features,
                hidden,
                num_layers=layers,
                dropout=float(params["dropout"]) if layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            output_dim = hidden * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(float(params["dropout"]))
            self.head = nn.Sequential(nn.Linear(output_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        elif kind == "lstm_transformer":
            model_dim = int(params["model_dim"])
            self.input_projection = nn.Linear(features, model_dim)
            self.position = nn.Parameter(torch.zeros(1, SEQUENCE_WINDOW, model_dim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.lstm = nn.LSTM(
                model_dim,
                model_dim,
                num_layers=int(params["lstm_layers"]),
                batch_first=True,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=int(params["heads"]),
                dim_feedforward=int(params["feedforward"]),
                dropout=float(params["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(params["transformer_layers"]),
            )
            self.norm = nn.LayerNorm(model_dim)
            self.head = nn.Sequential(nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1))
        elif kind == "cnn_lstm":
            channels = int(params["cnn_channels"])
            hidden = int(params["lstm_hidden"])
            dropout = float(params["dropout"])
            self.cnn = nn.Sequential(
                nn.Conv1d(features, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.05),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.05),
            )
            self.lstm = nn.LSTM(channels, hidden, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)
        elif kind == "lr_tcn_sma":
            channels = int(params["channels"])
            blocks = int(params["blocks"])
            layers = []
            in_channels = features
            for block_index in range(blocks):
                layers.append(LRTemporalBlock(in_channels, channels, 2 ** block_index, float(params["dropout"])))
                in_channels = channels
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(channels, 1)
        else:
            raise ValueError(f"Unsupported sequence model: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lstm":
            _, (hidden, _) = self.body(x)
            encoded = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.lstm_bidirectional else hidden[-1]
            return self.head(self.dropout(encoded))[:, 0]
        if self.kind == "lstm_transformer":
            z = self.input_projection(x) + self.position[:, :x.shape[1]]
            z, _ = self.lstm(z)
            z = self.transformer(z)
            return self.head(self.norm(z[:, -1]))[:, 0]
        if self.kind == "cnn_lstm":
            z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            z, _ = self.lstm(z)
            return self.head(self.dropout(z[:, -1]))[:, 0]
        z = self.body(x.transpose(1, 2))
        return self.head(z[:, :, -1])[:, 0]


def make_sequences(frame: pd.DataFrame, feature_mean, feature_std):
    arr = ((frame[FEATURE_COLUMNS].to_numpy(float) - feature_mean) / feature_std).astype(np.float32)
    y = frame.power_w.to_numpy(np.float32)
    xs, ys, indices = [], [], []
    for _, g in frame.groupby("flight", sort=False):
        ids = g.index.to_numpy()
        pos = np.arange(len(ids))
        for p in pos:
            start = max(0, p - SEQUENCE_WINDOW + 1)
            seq = arr[ids[start:p + 1]]
            if len(seq) < SEQUENCE_WINDOW:
                seq = np.pad(seq, ((SEQUENCE_WINDOW - len(seq), 0), (0, 0)), mode="edge")
            xs.append(seq); ys.append(y[ids[p]]); indices.append(ids[p])
    return np.asarray(xs), np.asarray(ys), np.asarray(indices)


def _batched_predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """分批执行模型推理，避免一次性把全部序列放入显存。"""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.from_numpy(features[start:start + BATCH_SIZE]).to(device)
            values.append(model(batch).cpu().numpy())
    return np.concatenate(values)


def _smooth_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """按 flight 对输入做只读取当前和过去样本的简单移动平均。"""
    result = frame.reset_index(drop=True).copy()
    result[FEATURE_COLUMNS] = result.groupby("flight", sort=False)[FEATURE_COLUMNS].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    return result


def deep_predict(train: pd.DataFrame, test: pd.DataFrame, kind: str, cfg: ExperimentConfig,
                 validation: pd.DataFrame | None = None) -> np.ndarray:
    """在独立验证集上搜索有限候选，固定最优配置后仅对测试集推理。"""
    if validation is None or validation.empty:
        raise ValueError("Deep-model tuning requires the independent validation split.")

    train = train.reset_index(drop=True).copy()
    validation = validation.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    target_mean = float(train.power_w.mean())
    target_std = float(train.power_w.std()) or 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.SmoothL1Loss()
    candidates = MODEL_CANDIDATES[kind]
    tuning_rows: list[dict[str, float | int | str | bool]] = []
    log_rows: list[dict[str, float | int | str | bool]] = []
    selected: dict[str, object] | None = None

    for candidate_index, candidate in enumerate(candidates, start=1):
        torch.manual_seed(RANDOM_SEED + candidate_index)
        np.random.seed(RANDOM_SEED + candidate_index)
        candidate_train = train
        candidate_validation = validation
        if kind == "lr_tcn_sma":
            sma_window = int(candidate["sma_window"])
            candidate_train = _smooth_features(train, sma_window)
            candidate_validation = _smooth_features(validation, sma_window)

        feature_mean = candidate_train[FEATURE_COLUMNS].to_numpy(float).mean(axis=0)
        feature_std = candidate_train[FEATURE_COLUMNS].to_numpy(float).std(axis=0)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        x_train, y_train, _ = make_sequences(candidate_train, feature_mean, feature_std)
        x_validation, y_validation, _ = make_sequences(candidate_validation, feature_mean, feature_std)
        y_train = (y_train - target_mean) / target_std
        y_validation_scaled = (y_validation - target_mean) / target_std

        model = SequenceRegressor(kind, len(FEATURE_COLUMNS), candidate).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        order = np.arange(len(x_train))
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        stale = 0
        for epoch in range(1, DEEP_EPOCHS + 1):
            np.random.shuffle(order)
            model.train()
            train_total = 0.0
            for start in range(0, len(order), BATCH_SIZE):
                ids = order[start:start + BATCH_SIZE]
                x_batch = torch.from_numpy(x_train[ids]).to(device)
                y_batch = torch.from_numpy(y_train[ids]).to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_total += loss.detach().item() * len(ids)

            model.eval()
            validation_total = 0.0
            with torch.no_grad():
                for start in range(0, len(x_validation), BATCH_SIZE):
                    x_batch = torch.from_numpy(x_validation[start:start + BATCH_SIZE]).to(device)
                    y_batch = torch.from_numpy(y_validation_scaled[start:start + BATCH_SIZE]).to(device)
                    validation_total += loss_fn(model(x_batch), y_batch).detach().item() * len(x_batch)
            train_loss = train_total / len(x_train)
            validation_loss = validation_total / len(x_validation)
            improved = validation_loss < best_loss - 1e-7
            log_rows.append({
                "candidate": candidate_index,
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "best_at_epoch": improved,
                "device": str(device),
            })
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= DEEP_PATIENCE:
                break

        if best_state is None:
            raise RuntimeError(f"No valid checkpoint was produced for {kind} candidate {candidate_index}.")
        model.load_state_dict(best_state)
        validation_prediction = np.maximum(
            0.0,
            _batched_predict(model, x_validation, device) * target_std + target_mean,
        )
        validation_metrics = metrics(y_validation, validation_prediction)
        validation_energy = flight_metrics(candidate_validation, validation_prediction)
        selection_score = (
            validation_metrics["sample_power_wape_percent"]
            + 0.5 * validation_energy["flight_energy_wape_percent"]
        )
        tuning_row = {
            "candidate": candidate_index,
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "validation_power_mae_w": validation_metrics["sample_power_mae_w"],
            "validation_power_wape_percent": validation_metrics["sample_power_wape_percent"],
            "validation_flight_energy_wape_percent": validation_energy["flight_energy_wape_percent"],
            "selection_score": selection_score,
            **candidate,
        }
        tuning_rows.append(tuning_row)
        if selected is None or selection_score < float(selected["selection_score"]):
            selected = {
                "candidate": candidate_index,
                "params": dict(candidate),
                "selection_score": selection_score,
                "state": best_state,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "best_epoch": best_epoch,
            }

        del model, x_train, y_train, x_validation, y_validation_scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if selected is None:
        raise RuntimeError(f"No hyperparameter candidate was selected for {kind}.")
    pd.DataFrame(log_rows).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    tuning_frame = pd.DataFrame(tuning_rows)
    tuning_frame["selected"] = tuning_frame["candidate"] == int(selected["candidate"])
    tuning_frame.to_csv(cfg.tuning_csv, index=False, encoding="utf-8")

    selected_params = selected["params"]
    candidate_test = test
    if kind == "lr_tcn_sma":
        candidate_test = _smooth_features(test, int(selected_params["sma_window"]))
    x_test, _, _ = make_sequences(
        candidate_test,
        np.asarray(selected["feature_mean"]),
        np.asarray(selected["feature_std"]),
    )
    model = SequenceRegressor(kind, len(FEATURE_COLUMNS), selected_params).to(device)
    model.load_state_dict(selected["state"])
    prediction = np.maximum(0.0, _batched_predict(model, x_test, device) * target_std + target_mean)
    torch.save(
        {
            "model_state": selected["state"],
            "algorithm": kind,
            "hyperparameters": selected_params,
            "feature_mean": np.asarray(selected["feature_mean"]),
            "feature_std": np.asarray(selected["feature_std"]),
            "target_mean": target_mean,
            "target_std": target_std,
            "sequence_window": SEQUENCE_WINDOW,
            "validation_selection_score": selected["selection_score"],
        },
        cfg.model_dir / "best_model.pt",
    )
    return prediction


def run_algorithm(name: str, train: pd.DataFrame, test: pd.DataFrame,
                  validation: pd.DataFrame | None = None) -> dict:
    cfg = ExperimentConfig(name); ensure_directories(cfg)
    kind = ALGORITHMS[name]["kind"]
    if kind == "baseline":
        src = Path(__file__).resolve().parent.parent / "3.0" / "out" / "predictions" / "test_predictions_3.0.csv"
        baseline = pd.read_csv(src)
        if len(baseline) != len(test):
            raise ValueError("3.0 baseline prediction count does not match the common test split.")
        for column in ("flight", "time"):
            if column in baseline and not np.allclose(
                baseline[column].to_numpy(float), test[column].to_numpy(float), rtol=0.0, atol=1e-7
            ):
                raise ValueError(f"3.0 baseline {column} is not aligned with the common test split.")
        pred = baseline.predicted_power_w.to_numpy(float)
    elif kind == "physical_mlr":
        pred = physical_mlr(train, validation, test, cfg)
    elif kind == "rf_tlatt_lite":
        pred = rf_tlatt_lite(train, validation, test, cfg)
    else:
        pred = deep_predict(train, test, kind, cfg, validation=validation)
    m = {**metrics(test.power_w.to_numpy(), pred), **flight_metrics(test, pred), "algorithm": name, "algorithm_name": ALGORITHMS[name]["name"], "reference": ALGORITHMS[name]["source"]}
    out = test[["flight", "route", "dt_seconds", "power_w", "time"]].copy(); out["predicted_power_w"] = pred; out["actual_energy_wh"] = out.power_w * out.dt_seconds / 3600; out["predicted_energy_wh"] = out.predicted_power_w * out.dt_seconds / 3600; out.to_csv(cfg.prediction_csv, index=False, encoding="utf-8")
    summary = out.groupby("flight", sort=True).agg(actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"), rows=("flight", "size")).reset_index(); summary["energy_error_wh"] = summary.predicted_energy_wh - summary.actual_energy_wh; summary.to_csv(cfg.flight_summary_csv, index=False, encoding="utf-8")
    cfg.metrics_json.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame([m]).to_csv(cfg.metrics_csv, index=False, encoding="utf-8")
    plt.figure(figsize=(7, 4)); plt.scatter(test.power_w.to_numpy()[::20], pred[::20], s=3, alpha=.25); lim=max(test.power_w.max(), pred.max()); plt.plot([0, lim], [0, lim], "k--"); plt.xlabel("Measured power (W)"); plt.ylabel("Predicted power (W)"); plt.title(name); plt.tight_layout(); plt.savefig(figure_file(cfg.figure_dir, "power_scatter"), dpi=180); plt.close()
    plt.figure(figsize=(8, 4)); ids = test.flight.unique()[:3]; colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for j, fid in enumerate(ids):
        mask = test.flight.to_numpy() == fid; plt.plot(test.time.to_numpy()[mask], test.power_w.to_numpy()[mask], color=colors[j], alpha=.7, label=f"flight {fid} measured"); plt.plot(test.time.to_numpy()[mask], pred[mask], "--", color=colors[j], alpha=.7, label=f"flight {fid} predicted")
    plt.xlabel("Flight time (s)"); plt.ylabel("Power (W)"); plt.legend(ncol=2, fontsize=7); plt.tight_layout(); plt.savefig(figure_file(cfg.figure_dir, "flight_power_timeseries"), dpi=180); plt.close()
    plot_algorithm_diagnostics(name, test, pred, cfg)
    return m


def plot_comparison_diagnostics(result_frame: pd.DataFrame, test: pd.DataFrame, root: Path) -> None:
    """功能: 生成跨算法的误差分布、阶段误差和能耗对照图。
    参数: result_frame为总体指标表，test为公共测试集，root为总体输出目录。
    返回: 无，图表写入总体out目录。
    调用位置: main。
    """
    names = result_frame.algorithm.tolist()
    diagnostic_frames = {}
    for name in names:
        prediction = pd.read_csv(ExperimentConfig(name).prediction_csv)
        diagnostic_frames[name] = build_diagnostic_frame(test, prediction.predicted_power_w.to_numpy())
    # 1. 算法绝对误差箱线图：展示中位数、四分位距和离群点。
    fig, ax = plt.subplots(figsize=(10, 5)); data = [diagnostic_frames[n].absolute_error_w.to_numpy() for n in names]
    ax.boxplot(data, labels=names, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#D9E2F3"), medianprops=dict(color="#C00000")); ax.set_ylabel("Absolute power error (W)"); ax.set_title("Cross-algorithm absolute-error distribution"); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    save_plot(fig, figure_file(root, "comparison_error_boxplot"))
    # 2. 各阶段WAPE热图：检查算法在不同飞行状态下的稳定性。
    phase_values = []
    for name in names:
        row = []
        diag = diagnostic_frames[name]
        for phase in PHASE_NAMES:
            part = diag[diag.phase_name == phase]; row.append(metrics(part.power_w.to_numpy(), part.predicted_power_w.to_numpy())["sample_power_wape_percent"] if len(part) else np.nan)
        phase_values.append(row)
    phase_arr = np.asarray(phase_values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5)); im = ax.imshow(phase_arr, aspect="auto", cmap="YlOrRd"); ax.set_xticks(range(len(PHASE_NAMES)), PHASE_NAMES); ax.set_yticks(range(len(names)), names); ax.set_title("Power WAPE by flight phase (%)")
    for i in range(len(names)):
        for j in range(len(PHASE_NAMES)): ax.text(j, i, f"{phase_arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="WAPE (%)"); save_plot(fig, figure_file(root, "comparison_phase_wape"))
    # 3. flight能耗真实值-预测值小 multiples：检查各算法在同一flight上的偏差。
    cols = 3; rows = int(np.ceil(len(names) / cols)); fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False); axes = axes.ravel()
    for idx, name in enumerate(names):
        summary = pd.read_csv(ExperimentConfig(name).flight_summary_csv); ax = axes[idx]; ax.scatter(summary.actual_energy_wh, summary.predicted_energy_wh, s=20, alpha=.8, color="#4472C4"); lim = max(summary.actual_energy_wh.max(), summary.predicted_energy_wh.max()); ax.plot([0, lim], [0, lim], "k--", linewidth=.8); ax.set_title(name); ax.set_xlabel("Measured Wh"); ax.set_ylabel("Predicted Wh")
    for ax in axes[len(names):]: ax.axis("off")
    fig.suptitle("Flight-energy agreement across algorithms", y=1.01); save_plot(fig, figure_file(root, "comparison_flight_energy_scatter"))
    # 4. 指标归一化热图：同时比较尺度不同的功率和能耗指标，数值越小越好。
    metric_cols = ["sample_power_mae_w", "sample_power_rmse_w", "sample_power_wape_percent", "flight_energy_mae_wh", "flight_energy_wape_percent"]
    metric_names = ["Power MAE", "Power RMSE", "Power WAPE", "Energy MAE", "Energy WAPE"]
    raw = result_frame[metric_cols].to_numpy(float); normalized = (raw - raw.min(axis=0)) / np.maximum(raw.max(axis=0) - raw.min(axis=0), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(normalized, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1); ax.set_xticks(range(len(metric_names)), metric_names, rotation=30, ha="right"); ax.set_yticks(range(len(names)), names); ax.set_title("Normalized error metrics (lower is better)")
    for i in range(len(names)):
        for j in range(len(metric_names)): ax.text(j, i, f"{raw[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Min-max normalized error"); save_plot(fig, figure_file(root, "comparison_metric_heatmap"))
    # 5. 误差CDF：直观看到达到给定误差阈值的样本比例。
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        values = np.sort(diagnostic_frames[name].absolute_error_w.to_numpy()); ax.plot(values, np.arange(1, len(values) + 1) / len(values), linewidth=1.5, label=name)
    ax.set_xlabel("Absolute power error (W)"); ax.set_ylabel("Empirical CDF"); ax.set_ylim(0, 1.02); ax.set_title("Absolute-error CDF comparison"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    save_plot(fig, figure_file(root, "comparison_error_cdf"))
    # 6. 综合排名：按五个误差指标的平均名次汇总。
    ranks = result_frame[metric_cols].rank(method="min", ascending=True); rank_score = ranks.mean(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(np.arange(len(rank_score)), rank_score.to_numpy(), color="#70AD47"); ax.set_yticks(np.arange(len(rank_score)), result_frame.loc[rank_score.index, "algorithm"]); ax.invert_yaxis(); ax.set_xlabel("Mean rank across five error metrics"); ax.set_title("Overall algorithm ranking"); ax.grid(axis="x", alpha=.25)
    save_plot(fig, figure_file(root, "comparison_overall_rank"))
    # 7. 同一flight功率曲线：所有算法与实测线共享同一时间轴。
    flight_variability = test.groupby("flight").power_w.std().sort_values(ascending=False)
    flight_id = flight_variability.index[0]
    flight_mask = test.flight.to_numpy() == flight_id
    flight_rows = np.flatnonzero(flight_mask)
    if len(flight_rows) > 700:
        flight_rows = flight_rows[:700]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.time.to_numpy()[flight_rows], test.power_w.to_numpy()[flight_rows], color="black", linewidth=2.0, label="Measured")
    for name in names:
        values = diagnostic_frames[name].predicted_power_w.to_numpy()[flight_rows]
        ax.plot(test.time.to_numpy()[flight_rows], values, linewidth=1.0, alpha=.85, label=name)
    ax.set_xlabel("Flight time (s)"); ax.set_ylabel("Power (W)"); ax.set_title(f"Prediction traces on flight {flight_id}")
    ax.legend(ncol=2, fontsize=7); ax.grid(alpha=.2)
    save_plot(fig, figure_file(root, "comparison_power_timeseries"))
    # 8. 绝对误差小提琴图：显示主体密度、偏斜和长尾结构。
    sampled_errors = [diagnostic_frames[name].absolute_error_w.to_numpy()[::5] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    violin = ax.violinplot(sampled_errors, showmeans=True, showmedians=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#5B9BD5"); body.set_alpha(.55)
    ax.set_xticks(np.arange(1, len(names) + 1), names, rotation=35, ha="right")
    ax.set_ylabel("Absolute power error (W)"); ax.set_title("Absolute-error density comparison"); ax.grid(axis="y", alpha=.25)
    save_plot(fig, figure_file(root, "comparison_error_violin"))
    # 9. Bland-Altman图：检查误差均值以及误差随功率水平变化的趋势。
    columns = 3; rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    for index, name in enumerate(names):
        ax = axes.ravel()[index]
        diag = diagnostic_frames[name].iloc[::max(1, len(test) // 6000)]
        mean_power = (diag.power_w + diag.predicted_power_w) / 2.0
        residual = diag.residual_w
        residual_mean = residual.mean(); residual_std = residual.std()
        ax.scatter(mean_power, residual, s=4, alpha=.18, color="#4472C4")
        ax.axhline(residual_mean, color="#C00000", linewidth=1.0)
        ax.axhline(residual_mean + 1.96 * residual_std, color="#ED7D31", linestyle="--", linewidth=.9)
        ax.axhline(residual_mean - 1.96 * residual_std, color="#ED7D31", linestyle="--", linewidth=.9)
        ax.set_title(name); ax.set_xlabel("Mean of measured and predicted power (W)"); ax.set_ylabel("Residual (W)")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Bland-Altman power agreement", y=1.01)
    save_plot(fig, figure_file(root, "comparison_bland_altman"))
    # 10. flight能耗误差箱线图：比较任务级累计误差的中心和离散范围。
    energy_errors = []
    for name in names:
        summary = pd.read_csv(ExperimentConfig(name).flight_summary_csv)
        energy_errors.append(summary.energy_error_wh.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(energy_errors, labels=names, showfliers=True, patch_artist=True,
               boxprops={"facecolor": "#FCE4D6"}, medianprops={"color": "#C00000"})
    ax.axhline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_ylabel("Flight-energy error (Wh)"); ax.set_title("Flight-energy error distribution"); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    save_plot(fig, figure_file(root, "comparison_energy_error_distribution"))
    # 11. 三指标阶段热图：补齐仅用WAPE时对低功率阶段可能产生的误判。
    phase_metric_arrays = {"MAE (W)": [], "RMSE (W)": [], "WAPE (%)": []}
    for name in names:
        diag = diagnostic_frames[name]
        metric_rows = []
        for phase in PHASE_NAMES:
            part = diag[diag.phase_name == phase]
            metric_rows.append(metrics(part.power_w.to_numpy(), part.predicted_power_w.to_numpy()))
        phase_metric_arrays["MAE (W)"].append([value["sample_power_mae_w"] for value in metric_rows])
        phase_metric_arrays["RMSE (W)"].append([value["sample_power_rmse_w"] for value in metric_rows])
        phase_metric_arrays["WAPE (%)"].append([value["sample_power_wape_percent"] for value in metric_rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric_name, values) in zip(axes, phase_metric_arrays.items()):
        array = np.asarray(values)
        image = ax.imshow(array, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(PHASE_NAMES)), PHASE_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [])
        ax.set_title(metric_name)
        for row_index in range(len(names)):
            for column_index in range(len(PHASE_NAMES)):
                ax.text(column_index, row_index, f"{array[row_index, column_index]:.1f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Phase-wise power error metrics")
    save_plot(fig, figure_file(root, "comparison_phase_metrics"))
    # 12. 雷达图：把五项误差转换为0到1的相对得分，越靠外表示相对误差越低。
    score = 1.0 - normalized
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"polar": True})
    for row_index, name in enumerate(names):
        closed_score = np.concatenate([score[row_index], score[row_index, :1]])
        ax.plot(closed_angles, closed_score, linewidth=1.2, label=name)
    ax.set_xticks(angles, metric_names); ax.set_ylim(0, 1); ax.set_yticks([.25, .5, .75, 1.0])
    ax.set_title("Relative error score (outer is better)"); ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=7)
    save_plot(fig, figure_file(root, "comparison_metric_radar"))


def main(names=None):
    np.random.seed(RANDOM_SEED); torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train = pd.read_csv(TRAIN_CSV); validation = pd.read_csv(VAL_CSV); test = pd.read_csv(TEST_CSV)
    names = names or list(ALGORITHMS)
    results = [run_algorithm(n, train, test, validation=validation) for n in names]
    root = Path(__file__).resolve().parent / "out"
    root.mkdir(parents=True, exist_ok=True)
    # 单算法重跑后仍汇集全部现有完整结果，避免把根目录总体图覆盖成单算法图。
    available_results = []
    for algorithm, metadata in ALGORITHMS.items():
        cfg = ExperimentConfig(algorithm)
        required_outputs = (cfg.metrics_csv, cfg.prediction_csv, cfg.flight_summary_csv)
        if not all(path.is_file() for path in required_outputs):
            continue
        frame = pd.read_csv(cfg.metrics_csv, encoding="utf-8")
        if len(frame) != 1:
            raise ValueError(f"算法 {algorithm} 的指标文件应仅包含一行: {cfg.metrics_csv}")
        record = frame.iloc[0].to_dict()
        record["algorithm"] = algorithm
        record["algorithm_name"] = metadata["name"]
        record["reference"] = metadata["source"]
        available_results.append(record)
    result_frame = pd.DataFrame(available_results)
    result_frame.to_csv(root / "comparison_metrics.csv", index=False, encoding="utf-8")
    with open(root / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result_frame.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    # 总览图沿用论文的并列柱状图方式，同时展示功率和flight能耗WAPE。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = result_frame.algorithm.tolist()
    y = np.arange(len(labels))
    axes[0].barh(y, result_frame.sample_power_wape_percent, color="#4472C4")
    axes[1].barh(y, result_frame.flight_energy_wape_percent, color="#ED7D31")
    for ax, title, xlabel in ((axes[0], "Sample power WAPE", "WAPE (%)"), (axes[1], "Flight energy WAPE", "WAPE (%)")):
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.25)
    save_plot(fig, figure_file(root, "comparison_metrics"))
    plot_comparison_diagnostics(result_frame, test, root)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--algorithm" in args:
        args = args[args.index("--algorithm") + 1:]
    main(args or None)
