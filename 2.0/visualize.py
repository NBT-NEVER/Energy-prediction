# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: visualize.py
# 开发时间: 2026-07-08
# 文件名: visualize.py
# 功能说明: 生成实验2.0的TCN、RLS评估和自定义工况可视化图表
# 版本号：2.0

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories
from data_utils import load_json
from device_utils import describe_cuda_device, select_cuda_device
from predict import load_trained_model, predict_array
from train import apply_rls_correction, build_sequence_arrays
from progress import TerminalProgress
from uncertainty import add_prediction_intervals, load_calibration_scores

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """功能: 以UTF-8保存JSON摘要。
    参数: path为保存路径，payload为待保存内容。
    返回: None。
    调用位置: generate_all_visualizations、predict_custom_scenario。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def set_plot_style() -> None:
    """功能: 设置图表默认样式。
    参数: 无。
    返回: None。
    调用位置: 各绘图函数。
    """

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(path: Path) -> Path:
    """功能: 保存当前Matplotlib图表。
    参数: path为图片路径。
    返回: 图片路径。
    调用位置: 各绘图函数。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_training_history(cfg: ExperimentConfig) -> list[Path]:
    """功能: 绘制训练损失、学习率和调参候选表现。
    参数: cfg为实验配置对象。
    返回: 已生成图片路径列表。
    调用位置: generate_all_visualizations。
    """

    set_plot_style()
    outputs: list[Path] = []
    if cfg.training_log_csv.exists():
        log_frame = pd.read_csv(cfg.training_log_csv)
        plt.figure(figsize=(9, 5))
        plt.plot(log_frame["epoch"], log_frame["train_loss"], label="Train loss", linewidth=2)
        plt.plot(log_frame["epoch"], log_frame["val_loss"], label="Validation loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Huber loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        outputs.append(save_figure(cfg.training_vis_dir / "training_loss_history.png"))

        plt.figure(figsize=(9, 4))
        plt.plot(log_frame["epoch"], log_frame["learning_rate"], color="#2f6fbb", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning Rate Schedule")
        outputs.append(save_figure(cfg.training_vis_dir / "learning_rate_schedule.png"))

    if cfg.tuning_results_csv.exists():
        tuning = pd.read_csv(cfg.tuning_results_csv)
        score_column = "selection_score" if "selection_score" in tuning.columns else "best_val_loss"
        tuning = tuning.sort_values(score_column)
        plt.figure(figsize=(10, 5))
        plt.bar(tuning["candidate"], tuning[score_column], color="#4c78a8")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(score_column)
        plt.title("Hyperparameter Candidate Ranking")
        outputs.append(save_figure(cfg.training_vis_dir / "hyperparameter_ranking.png"))

        available = [col for col in ["val_sample_power_wape", "val_flight_energy_wape"] if col in tuning.columns]
        if available:
            x = np.arange(len(tuning))
            width = 0.35
            plt.figure(figsize=(10, 5))
            for index, column in enumerate(available):
                offset = (index - (len(available) - 1) / 2) * width
                plt.bar(x + offset, tuning[column], width=width, label=column)
            plt.xticks(x, tuning["candidate"], rotation=25, ha="right")
            plt.ylabel("WAPE (%)")
            plt.title("Validation WAPE by Candidate")
            plt.legend()
            outputs.append(save_figure(cfg.training_vis_dir / "candidate_validation_wape.png"))
    return outputs


def plot_result_summary(cfg: ExperimentConfig) -> list[Path]:
    """功能: 绘制模型评估结果和flight级能耗对比图。
    参数: cfg为实验配置对象。
    返回: 已生成图片路径列表。
    调用位置: generate_all_visualizations。
    """

    set_plot_style()
    outputs: list[Path] = []
    if cfg.evaluation_json.exists():
        metrics = load_json(cfg.evaluation_json)
        metric_names = [
            "tcn_sample_power_w_mae",
            "sample_power_w_mae",
            "tcn_sample_power_w_wape_percent",
            "sample_power_w_wape_percent",
            "tcn_flight_energy_wh_mae",
            "flight_energy_wh_mae",
        ]
        values = [float(metrics[name]) for name in metric_names if name in metrics]
        names = [name.replace("_", "\n") for name in metric_names if name in metrics]
        plt.figure(figsize=(9, 5))
        plt.bar(names, values, color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#e45756", "#54a24b"])
        plt.ylabel("Metric value")
        plt.title("Evaluation Metrics")
        outputs.append(save_figure(cfg.result_vis_dir / "evaluation_metrics.png"))

    if cfg.flight_energy_summary_csv.exists():
        flight = pd.read_csv(cfg.flight_energy_summary_csv).sort_values("actual_energy_wh")
        plt.figure(figsize=(10, 5))
        plt.plot(flight["flight"].astype(str), flight["actual_energy_wh"], marker="o", label="Actual")
        plt.plot(flight["flight"].astype(str), flight["predicted_energy_wh"], marker="s", label="Predicted")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Flight")
        plt.ylabel("Energy (Wh)")
        plt.title("Flight-Level Energy: Actual vs Predicted")
        plt.legend()
        outputs.append(save_figure(cfg.result_vis_dir / "flight_energy_actual_vs_predicted.png"))

        plt.figure(figsize=(9, 5))
        plt.bar(flight["flight"].astype(str), flight["energy_error_wh"], color="#e45756")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Flight")
        plt.ylabel("Prediction error (Wh)")
        plt.title("Flight-Level Energy Prediction Error")
        outputs.append(save_figure(cfg.result_vis_dir / "flight_energy_error.png"))

    if cfg.power_bin_evaluation_csv.exists():
        bins = pd.read_csv(cfg.power_bin_evaluation_csv)
        bin_labels = bins["power_bin"].astype(str).str.replace("以上", "+", regex=False)
        plt.figure(figsize=(9, 5))
        plt.bar(bin_labels, bins["power_w_mae"], color="#b279a2")
        plt.xlabel("Power bin")
        plt.ylabel("MAE (W)")
        plt.title("Prediction MAE by Power Bin")
        outputs.append(save_figure(cfg.result_vis_dir / "power_bin_mae.png"))
    return outputs


def plot_prediction_outputs(cfg: ExperimentConfig, max_flights: int = 3) -> list[Path]:
    """功能: 绘制测试集预测散点、残差分布和典型flight时序。
    参数: cfg为实验配置对象，max_flights为时序图最多展示的flight数量。
    返回: 已生成图片路径列表。
    调用位置: generate_all_visualizations。
    """

    set_plot_style()
    outputs: list[Path] = []
    if not cfg.prediction_csv.exists():
        return outputs

    predictions = pd.read_csv(cfg.prediction_csv)
    if "power_w" in predictions.columns:
        sample = predictions.sample(min(len(predictions), 12000), random_state=cfg.random_seed)
        plt.figure(figsize=(6, 6))
        plt.scatter(sample["power_w"], sample["predicted_power_w"], s=8, alpha=0.35, color="#4c78a8")
        max_value = float(max(sample["power_w"].max(), sample["predicted_power_w"].max()))
        plt.plot([0, max_value], [0, max_value], color="#e45756", linewidth=2, label="Ideal")
        plt.xlabel("Actual power (W)")
        plt.ylabel("Predicted power (W)")
        plt.title("Power Prediction Scatter")
        plt.legend()
        outputs.append(save_figure(cfg.prediction_vis_dir / "power_prediction_scatter.png"))

        residual = predictions["predicted_power_w"] - predictions["power_w"]
        plt.figure(figsize=(9, 5))
        plt.hist(residual.clip(-300, 300), bins=80, color="#72b7b2", alpha=0.85)
        plt.xlabel("Prediction residual (W)")
        plt.ylabel("Count")
        plt.title("Power Prediction Residual Distribution")
        outputs.append(save_figure(cfg.prediction_vis_dir / "power_residual_histogram.png"))

    if {"flight", "time", "predicted_power_w"}.issubset(predictions.columns):
        flight_ids = predictions["flight"].drop_duplicates().head(max_flights).tolist()
        for flight_id in flight_ids:
            flight_frame = predictions[predictions["flight"] == flight_id].sort_values("time")
            plt.figure(figsize=(10, 5))
            if "power_w" in flight_frame.columns:
                plt.plot(flight_frame["time"], flight_frame["power_w"], label="Actual power", linewidth=1.8)
            plt.plot(flight_frame["time"], flight_frame["predicted_power_w"], label="Predicted power", linewidth=1.8)
            if {"predicted_power_lower_w", "predicted_power_upper_w"}.issubset(flight_frame.columns):
                plt.fill_between(
                    flight_frame["time"].to_numpy(),
                    flight_frame["predicted_power_lower_w"].to_numpy(),
                    flight_frame["predicted_power_upper_w"].to_numpy(),
                    color="#4c78a8",
                    alpha=0.16,
                    label=f"{float(flight_frame['confidence_level'].iloc[0]):.0%} interval",
                )
            plt.xlabel("Time (s)")
            plt.ylabel("Power (W)")
            plt.title(f"Flight {flight_id} Power Prediction")
            plt.legend()
            outputs.append(save_figure(cfg.prediction_vis_dir / f"flight_{flight_id}_power_timeseries.png"))
    return outputs


def build_default_custom_conditions(
    cfg: ExperimentConfig,
    duration_s: float,
    sample_dt: float,
    wind_speed: float,
    wind_angle: float,
    flight_speed: float,
    payload_g: float,
    altitude_m: float,
    route: str,
) -> pd.DataFrame:
    """功能: 根据命令行自定义工况生成一段简化飞行剖面。
    参数: cfg为实验配置对象，其余参数为自定义工况。
    返回: 自定义特征DataFrame。
    调用位置: predict_custom_scenario。
    """

    meta = load_json(cfg.feature_meta_json)
    feature_columns = meta["feature_columns"]
    times = np.arange(0.0, duration_s + sample_dt, sample_dt)
    progress = np.clip(times / max(duration_s, sample_dt), 0.0, 1.0)
    vertical_speed = np.where(progress < 0.15, 1.8, np.where(progress > 0.85, -1.5, 0.0))
    horizontal_speed = np.where((progress >= 0.15) & (progress <= 0.85), flight_speed, flight_speed * 0.35)
    actual_speed = np.sqrt(horizontal_speed**2 + vertical_speed**2)

    wind_rad = math.radians(wind_angle)
    wind_x = wind_speed * math.cos(wind_rad)
    wind_y = wind_speed * math.sin(wind_rad)
    velocity_x = horizontal_speed
    velocity_y = np.zeros_like(velocity_x)
    relative_air_speed = np.sqrt((velocity_x - wind_x) ** 2 + wind_y**2 + vertical_speed**2)
    wind_alignment = velocity_x * wind_x / np.maximum(horizontal_speed * wind_speed, 1e-6)
    wind_cross = np.abs(velocity_x * wind_y) / np.maximum(horizontal_speed, 1e-6)
    payload_kg = payload_g / 1000.0
    dynamic_accel = 0.35 + np.abs(np.gradient(actual_speed, sample_dt))
    angular_rate = 0.03 + 0.04 * np.sin(progress * np.pi) ** 2
    obstacle_agility = dynamic_accel * (1.0 + angular_rate) + np.abs(vertical_speed) * 0.2
    thermal_load = (
        0.5 * np.maximum(relative_air_speed, 0) ** 3
        + payload_kg * dynamic_accel * np.maximum(actual_speed, 0)
        + np.maximum(vertical_speed, 0) * (3.6 + payload_kg) * 9.80665
    )

    frame = pd.DataFrame(
        {
            "flight": 999001,
            "route": route,
            "time": times,
            "dt_seconds": sample_dt,
            "flight_progress": progress,
            "wind_speed": wind_speed,
            "wind_sin": math.sin(wind_rad),
            "wind_cos": math.cos(wind_rad),
            "programmed_speed_mps": flight_speed,
            "actual_speed_mps": actual_speed,
            "horizontal_speed_mps": horizontal_speed,
            "vertical_speed_mps": vertical_speed,
            "vertical_speed_abs_mps": np.abs(vertical_speed),
            "relative_air_speed_mps": relative_air_speed,
            "wind_alignment": np.clip(wind_alignment, -1.0, 1.0),
            "wind_cross_component_mps": wind_cross,
            "payload_kg": payload_kg,
            "altitude_m": altitude_m,
            "dynamic_accel_norm": dynamic_accel,
            "angular_rate_norm": angular_rate,
            "obstacle_agility_index": obstacle_agility,
            "thermal_load_proxy": thermal_load,
            "vision_energy_proxy_w": 2.5 + 0.15 * actual_speed + 0.30 * obstacle_agility + 0.002 * altitude_m,
            "communication_energy_proxy_w": 0.8 + 0.004 * altitude_m + 0.04 * wind_speed + 0.01 * actual_speed,
        }
    )
    for column in feature_columns:
        if column.startswith("route_"):
            frame[column] = 1.0 if column == f"route_{route}" else 0.0
        elif column not in frame.columns:
            frame[column] = 0.0
    return frame


def normalize_custom_input(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 将用户自定义CSV补齐为模型特征表。
    参数: frame为用户CSV内容，cfg为实验配置对象。
    返回: 补齐后的特征DataFrame。
    调用位置: predict_custom_scenario。
    """

    meta = load_json(cfg.feature_meta_json)
    feature_columns = meta["feature_columns"]
    if "dt_seconds" not in frame.columns:
        frame["dt_seconds"] = frame["time"].diff().fillna(0.2) if "time" in frame.columns else 0.2
    if "time" not in frame.columns:
        frame["time"] = np.arange(len(frame)) * frame["dt_seconds"].astype(float)
    if "flight" not in frame.columns:
        frame["flight"] = 999001
    if "route" not in frame.columns:
        frame["route"] = cfg.training_route
    routes = frame["route"].astype(str).str.strip()
    invalid_routes = sorted(set(routes) - {cfg.training_route})
    if invalid_routes:
        raise ValueError(f"实验2.0自定义工况仅支持{cfg.training_route}航线，输入包含: {invalid_routes}")
    frame["route"] = cfg.training_route
    for column in feature_columns:
        if column.startswith("route_"):
            route_name = column.replace("route_", "", 1)
            frame[column] = (frame["route"].astype(str) == route_name).astype(float)
        elif column not in frame.columns:
            frame[column] = 0.0
    return frame


def predict_custom_scenario(
    cfg: ExperimentConfig,
    custom_csv: Path | None,
    duration_s: float,
    sample_dt: float,
    wind_speed: float,
    wind_angle: float,
    flight_speed: float,
    payload_g: float,
    altitude_m: float,
    route: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """功能: 调取训练模型预测自定义工况能耗并生成图表。
    参数: cfg为实验配置对象，custom_csv为可选自定义输入CSV，其余为默认工况参数，confidence为置信度。
    返回: 自定义预测摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    if custom_csv:
        custom_frame = normalize_custom_input(pd.read_csv(custom_csv), cfg)
    else:
        custom_frame = build_default_custom_conditions(
            cfg, duration_s, sample_dt, wind_speed, wind_angle, flight_speed, payload_g, altitude_m, route
        )
        custom_frame.to_csv(cfg.custom_scenario_csv, index=False, encoding="utf-8")

    progress = TerminalProgress("自定义工况总流程", 6)
    progress.update(1, f"已准备 {len(custom_frame)} 条自定义工况记录")
    device = select_cuda_device(cfg.device)
    print(f"自定义工况推理设备: {describe_cuda_device(device)}")
    model, checkpoint, scaler = load_trained_model(cfg, device)
    progress.update(2, f"已加载 {len(checkpoint['channels'])}块TCN，窗口 {checkpoint['window_seconds']:g}s")
    feature_columns = scaler["feature_columns"]
    for column in feature_columns:
        if column not in custom_frame.columns:
            custom_frame[column] = 0.0
    custom_frame = custom_frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    sequences, _ = build_sequence_arrays(custom_frame, scaler, int(checkpoint["window_steps"]), "自定义序列构造")
    progress.update(3, f"已构造 {len(sequences)} 条输入序列")
    tcn_power = predict_array(model, sequences, scaler, device, cfg.batch_size)
    progress.update(4, "TCN前向推理完成")
    corrected_power, _ = apply_rls_correction(tcn_power, custom_frame, scaler, cfg, checkpoint.get("rls_theta"), update=False, progress_label="自定义RLS校正")
    custom_frame["tcn_predicted_power_w"] = tcn_power
    custom_frame["rls_corrected_power_w"] = corrected_power
    custom_frame["predicted_power_w"] = corrected_power
    custom_frame["rls_update_enabled"] = False
    confidence = cfg.default_confidence if confidence is None else float(confidence)
    scores = load_calibration_scores(cfg.uncertainty_calibration_npz, online_update=False)
    custom_frame, radius = add_prediction_intervals(custom_frame, scores, confidence)
    custom_frame.to_csv(cfg.custom_prediction_csv, index=False, encoding="utf-8")
    progress.update(5, f"RLS校正和预测文件已保存：{cfg.custom_prediction_csv.name}")

    set_plot_style()
    plt.figure(figsize=(10, 5))
    plt.plot(custom_frame["time"], custom_frame["predicted_power_w"], color="#4c78a8", linewidth=2, label="Predicted power")
    plt.fill_between(custom_frame["time"], custom_frame["predicted_power_lower_w"], custom_frame["predicted_power_upper_w"], color="#4c78a8", alpha=0.16, label=f"{confidence:.0%} interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Predicted power (W)")
    plt.title("Custom Scenario Predicted Power")
    plt.legend()
    power_chart = save_figure(cfg.custom_vis_dir / "custom_power_timeseries.png")

    plt.figure(figsize=(10, 5))
    plt.plot(custom_frame["time"], custom_frame["cumulative_energy_wh"], color="#54a24b", linewidth=2, label="Predicted cumulative energy")
    plt.fill_between(custom_frame["time"], custom_frame["cumulative_energy_lower_wh"], custom_frame["cumulative_energy_upper_wh"], color="#54a24b", alpha=0.16, label=f"{confidence:.0%} interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative energy (Wh)")
    plt.title("Custom Scenario Cumulative Energy")
    plt.legend()
    energy_chart = save_figure(cfg.custom_vis_dir / "custom_cumulative_energy.png")

    summary = {
        "custom_prediction_csv": str(cfg.custom_prediction_csv),
        "custom_power_chart": str(power_chart),
        "custom_energy_chart": str(energy_chart),
        "rows": int(len(custom_frame)),
        "mean_predicted_power_w": float(custom_frame["predicted_power_w"].mean()),
        "max_predicted_power_w": float(custom_frame["predicted_power_w"].max()),
        "total_predicted_energy_wh": float(custom_frame["predicted_energy_wh"].sum()),
        "confidence_level": confidence,
        "power_interval_radius_w": radius,
        "total_predicted_energy_lower_wh": float(custom_frame["predicted_energy_lower_wh"].sum()),
        "total_predicted_energy_upper_wh": float(custom_frame["predicted_energy_upper_wh"].sum()),
    }
    save_json(cfg.custom_prediction_summary_json, summary)
    progress.finish(f"图表和摘要已保存，累计能耗={summary['total_predicted_energy_wh']:.5f} Wh")
    return summary


def generate_all_visualizations(cfg: ExperimentConfig) -> dict[str, Any]:
    """功能: 生成训练过程、训练结果和测试预测结果全部可视化图表。
    参数: cfg为实验配置对象。
    返回: 可视化输出摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    outputs = []
    progress = TerminalProgress("生成图表总流程", 3)
    outputs.extend(plot_training_history(cfg))
    progress.update(1, f"训练过程图表已生成（累计 {len(outputs)} 张）")
    outputs.extend(plot_result_summary(cfg))
    progress.update(2, f"评估结果图表已生成（累计 {len(outputs)} 张）")
    outputs.extend(plot_prediction_outputs(cfg))
    progress.finish(f"预测结果图表已生成，共 {len(outputs)} 张")
    summary = {
        "figure_count": len(outputs),
        "figures": [str(path) for path in outputs],
        "out_dir": str(cfg.out_dir),
    }
    save_json(cfg.visualization_summary_json, summary)
    return summary
