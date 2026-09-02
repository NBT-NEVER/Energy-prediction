# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: evaluate.py
# 开发时间: 2026-07-07
# 文件名: evaluate.py
# 功能说明: 评估实验2.0的TCN基线和RLS校正样本级功率、飞行级能量误差
# 版本号：2.0

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories
from data_utils import save_json
from predict import predict_from_csv
from progress import TerminalProgress


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """功能: 计算回归任务的MAE、RMSE、R2和MAPE。
    参数: y_true为真实值，y_pred为预测值，prefix为指标名前缀。
    返回: 指标字典。
    调用位置: evaluate_model。
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom > 1e-12 else 0.0
    valid = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(error[valid] / y_true[valid])) * 100.0) if valid.any() else 0.0
    wape = float(np.sum(np.abs(error)) / np.maximum(np.sum(np.abs(y_true)), 1e-6) * 100.0)
    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_mape_percent": mape,
        f"{prefix}_wape_percent": wape,
    }


def build_flight_energy_summary(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """功能: 按flight汇总真实和预测能耗。
    参数: prediction_frame为包含预测结果的DataFrame。
    返回: flight级能耗汇总表。
    调用位置: evaluate_model。
    """

    required = {"flight", "actual_energy_wh", "predicted_energy_wh", "tcn_predicted_energy_wh", "power_w", "predicted_power_w", "tcn_predicted_power_w"}
    missing = sorted(required - set(prediction_frame.columns))
    if missing:
        raise ValueError(f"预测结果缺少评估字段: {missing}")
    grouped = prediction_frame.groupby("flight", sort=True)
    summary = grouped.agg(
        route=("route", "first"),
        rows=("flight", "size"),
        duration_s=("dt_seconds", "sum"),
        actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"),
        tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        actual_mean_power_w=("power_w", "mean"),
        predicted_mean_power_w=("predicted_power_w", "mean"),
        tcn_predicted_mean_power_w=("tcn_predicted_power_w", "mean"),
        programmed_speed_mps=("programmed_speed_mps", "median"),
        payload_kg=("payload_kg", "median"),
        altitude_m=("altitude_m", "median"),
    ).reset_index()
    if {"predicted_energy_lower_wh", "predicted_energy_upper_wh"}.issubset(prediction_frame.columns):
        interval = grouped.agg(
            predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"),
            predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
        ).reset_index()
        summary = summary.merge(interval, on="flight", how="left")
    summary["energy_error_wh"] = summary["predicted_energy_wh"] - summary["actual_energy_wh"]
    summary["energy_abs_error_wh"] = summary["energy_error_wh"].abs()
    summary["energy_abs_percent_error"] = (
        summary["energy_abs_error_wh"] / summary["actual_energy_wh"].clip(lower=1e-6) * 100.0
    )
    return summary


def build_power_bin_summary(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """功能: 按真实功率区间汇总预测误差。
    参数: prediction_frame为包含预测结果的DataFrame。
    返回: 功率分段评估表。
    调用位置: evaluate_model。
    """

    bins = [-np.inf, 50.0, 300.0, 450.0, 600.0, np.inf]
    labels = ["0-50W", "50-300W", "300-450W", "450-600W", "600W以上"]
    frame = prediction_frame.copy()
    frame["power_bin"] = pd.cut(frame["power_w"], bins=bins, labels=labels)
    rows = []
    for label, group in frame.groupby("power_bin", observed=True):
        metrics = regression_metrics(group["power_w"].to_numpy(), group["predicted_power_w"].to_numpy(), "power_w")
        rows.append(
            {
                "power_bin": str(label),
                "rows": int(len(group)),
                "actual_mean_power_w": float(group["power_w"].mean()),
                "predicted_mean_power_w": float(group["predicted_power_w"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(cfg: ExperimentConfig) -> dict:
    """功能: 生成预测文件并保存模型评估结果。
    参数: cfg为实验配置对象。
    返回: 评估指标字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    progress = TerminalProgress("模型评估总流程", 5)
    prediction_path = predict_from_csv(cfg, cfg.test_csv, cfg.prediction_csv)
    progress.update(1, "测试集预测文件已生成")
    predictions = pd.read_csv(prediction_path)
    sample_metrics = regression_metrics(
        predictions["power_w"].to_numpy(),
        predictions["predicted_power_w"].to_numpy(),
        "sample_power_w",
    )
    tcn_sample_metrics = regression_metrics(
        predictions["power_w"].to_numpy(),
        predictions["tcn_predicted_power_w"].to_numpy(),
        "tcn_sample_power_w",
    )
    progress.update(2, f"样本级功率误差已计算，RLS WAPE={sample_metrics['sample_power_w_wape_percent']:.4f}%")

    flight_summary = build_flight_energy_summary(predictions)
    power_bin_summary = build_power_bin_summary(predictions)
    progress.update(3, f"已汇总 {len(flight_summary)} 个flight、{len(power_bin_summary)} 个功率区间")
    flight_metrics = regression_metrics(
        flight_summary["actual_energy_wh"].to_numpy(),
        flight_summary["predicted_energy_wh"].to_numpy(),
        "flight_energy_wh",
    )
    tcn_flight_metrics = regression_metrics(
        flight_summary["actual_energy_wh"].to_numpy(),
        flight_summary["tcn_predicted_energy_wh"].to_numpy(),
        "tcn_flight_energy_wh",
    )
    interval_metrics = {}
    if {"predicted_power_lower_w", "predicted_power_upper_w"}.issubset(predictions.columns):
        actual_power = predictions["power_w"].to_numpy(dtype=float)
        lower_power = predictions["predicted_power_lower_w"].to_numpy(dtype=float)
        upper_power = predictions["predicted_power_upper_w"].to_numpy(dtype=float)
        interval_metrics = {
            "confidence_level": float(predictions["confidence_level"].iloc[0]),
            "power_interval_radius_w": float(predictions["power_interval_radius_w"].iloc[0]),
            "sample_power_interval_coverage_percent": float(np.mean((actual_power >= lower_power) & (actual_power <= upper_power)) * 100.0),
            "sample_power_interval_mean_width_w": float(np.mean(upper_power - lower_power)),
        }
        if {"predicted_energy_lower_wh", "predicted_energy_upper_wh"}.issubset(flight_summary.columns):
            actual_energy = flight_summary["actual_energy_wh"].to_numpy(dtype=float)
            lower_energy = flight_summary["predicted_energy_lower_wh"].to_numpy(dtype=float)
            upper_energy = flight_summary["predicted_energy_upper_wh"].to_numpy(dtype=float)
            interval_metrics.update(
                {
                    "flight_energy_interval_coverage_percent": float(np.mean((actual_energy >= lower_energy) & (actual_energy <= upper_energy)) * 100.0),
                    "flight_energy_interval_mean_width_wh": float(np.mean(upper_energy - lower_energy)),
                }
            )
    metrics = {
        **sample_metrics,
        **tcn_sample_metrics,
        **flight_metrics,
        **tcn_flight_metrics,
        **interval_metrics,
        "test_rows": int(len(predictions)),
        "test_flights": int(flight_summary["flight"].nunique()),
        "prediction_file": str(prediction_path),
        "flight_energy_summary_file": str(cfg.flight_energy_summary_csv),
        "power_bin_evaluation_file": str(cfg.power_bin_evaluation_csv),
    }
    progress.update(4, f"flight能耗 WAPE={flight_metrics['flight_energy_wh_wape_percent']:.4f}%")

    pd.DataFrame([metrics]).to_csv(cfg.evaluation_csv, index=False, encoding="utf-8")
    flight_summary.to_csv(cfg.flight_energy_summary_csv, index=False, encoding="utf-8")
    power_bin_summary.to_csv(cfg.power_bin_evaluation_csv, index=False, encoding="utf-8")
    save_json(cfg.evaluation_json, metrics)
    progress.finish(f"评估结果已保存：{cfg.evaluation_json.name}")
    return metrics
