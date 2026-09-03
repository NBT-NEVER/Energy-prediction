# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: uncertainty.py
# 开发时间: 2026-09-02
# 文件名: uncertainty.py
# 功能说明: 基于验证集保序残差计算实验2.1的功率和累计能耗置信区间
# 版本号：2.1

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_confidence(confidence: float) -> float:
    """功能: 校验并规范置信度参数。
    参数: confidence为0到1之间的目标覆盖率。
    返回: 可用于保序分位数计算的浮点置信度。
    调用位置: calibration_quantile、add_prediction_intervals。
    """

    value = float(confidence)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence必须在0和1之间，例如95%应填写0.95。")
    return value


def calibration_quantile(scores: np.ndarray, confidence: float) -> float:
    """功能: 计算有限样本保序校准的绝对残差半径。
    参数: scores为验证集绝对残差，confidence为目标覆盖率。
    返回: 功率区间半径，单位W。
    调用位置: add_prediction_intervals、save_calibration。
    """

    values = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("置信区间校准残差为空，无法计算区间。")
    rank = min(max(int(math.ceil((len(values) + 1) * validate_confidence(confidence))), 1), len(values))
    return float(values[rank - 1])


def save_calibration(
    npz_path: Path,
    json_path: Path,
    online_scores: np.ndarray,
    static_scores: np.ndarray,
    default_confidence: float,
) -> dict[str, Any]:
    """功能: 保存在线RLS和固定RLS两种预测状态的校准残差。
    参数: npz_path和json_path为校准文件路径，online_scores/static_scores为绝对残差，default_confidence为默认置信度。
    返回: 校准摘要字典。
    调用位置: train_model、calibrate_uncertainty。
    """

    confidence = validate_confidence(default_confidence)
    online = np.asarray(online_scores, dtype=np.float32).reshape(-1)
    static = np.asarray(static_scores, dtype=np.float32).reshape(-1)
    if not len(online) or not len(static):
        raise ValueError("在线RLS和固定RLS都至少需要一个校准残差。")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, online_abs_residual_w=online, static_abs_residual_w=static)
    summary = {
        "method": "split_conformal_absolute_residual",
        "default_confidence": confidence,
        "online_samples": int(len(online)),
        "static_samples": int(len(static)),
        "online_default_radius_w": calibration_quantile(online, confidence),
        "static_default_radius_w": calibration_quantile(static, confidence),
        "npz_file": str(npz_path),
    }
    with open(json_path, "w", encoding="utf-8", newline="") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def load_calibration_scores(npz_path: Path, online_update: bool) -> np.ndarray:
    """功能: 读取对应RLS运行状态的校准绝对残差。
    参数: npz_path为校准文件路径，online_update表示是否有实测值在线更新。
    返回: 校准绝对残差数组，单位W。
    调用位置: add_prediction_intervals。
    """

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到置信区间校准文件: {npz_path}，请先运行 train 或 calibrate。")
    with np.load(npz_path) as payload:
        key = "online_abs_residual_w" if online_update else "static_abs_residual_w"
        if key not in payload:
            raise ValueError(f"校准文件缺少字段: {key}")
        return np.asarray(payload[key], dtype=float)


def add_prediction_intervals(
    frame: pd.DataFrame,
    calibration_scores: np.ndarray,
    confidence: float,
) -> tuple[pd.DataFrame, float]:
    """功能: 将置信度对应的功率、单步能耗和累计能耗区间写入预测表。
    参数: frame为含预测功率和时间间隔的DataFrame，calibration_scores为绝对残差，confidence为目标覆盖率。
    返回: 增加区间字段的DataFrame和功率区间半径。
    调用位置: predict_from_csv、recalculate_prediction_intervals、predict_custom_scenario。
    """

    confidence = validate_confidence(confidence)
    required = {"predicted_power_w", "dt_seconds"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"预测结果缺少置信区间字段: {missing}")
    output = frame.copy()
    radius = calibration_quantile(calibration_scores, confidence)
    prediction = output["predicted_power_w"].to_numpy(dtype=float)
    output["confidence_level"] = confidence
    output["power_interval_radius_w"] = radius
    output["predicted_power_lower_w"] = np.maximum(prediction - radius, 0.0)
    output["predicted_power_upper_w"] = prediction + radius
    dt = output["dt_seconds"].to_numpy(dtype=float)
    output["predicted_energy_wh"] = prediction * dt / 3600.0
    output["predicted_energy_lower_wh"] = output["predicted_power_lower_w"] * dt / 3600.0
    output["predicted_energy_upper_wh"] = output["predicted_power_upper_w"] * dt / 3600.0
    if "flight" in output.columns:
        output["cumulative_energy_wh"] = output.groupby("flight", sort=False)["predicted_energy_wh"].cumsum()
        output["cumulative_energy_lower_wh"] = output.groupby("flight", sort=False)["predicted_energy_lower_wh"].cumsum()
        output["cumulative_energy_upper_wh"] = output.groupby("flight", sort=False)["predicted_energy_upper_wh"].cumsum()
    else:
        output["cumulative_energy_wh"] = output["predicted_energy_wh"].cumsum()
        output["cumulative_energy_lower_wh"] = output["predicted_energy_lower_wh"].cumsum()
        output["cumulative_energy_upper_wh"] = output["predicted_energy_upper_wh"].cumsum()
    return output, radius


def interval_summary(frame: pd.DataFrame, calibration_mode: str, radius: float) -> dict[str, Any]:
    """功能: 汇总当前置信度下的区间结果。
    参数: frame为已增加区间字段的预测表，calibration_mode为校准状态，radius为功率区间半径。
    返回: 置信区间摘要字典。
    调用位置: recalculate_prediction_intervals、predict_custom_scenario。
    """

    confidence = float(frame["confidence_level"].iloc[0]) if len(frame) else 0.0
    return {
        "confidence_level": confidence,
        "calibration_mode": calibration_mode,
        "calibration_power_radius_w": float(radius),
        "rows": int(len(frame)),
        "mean_predicted_power_w": float(frame["predicted_power_w"].mean()) if len(frame) else 0.0,
        "total_predicted_energy_wh": float(frame["predicted_energy_wh"].sum()) if len(frame) else 0.0,
        "total_predicted_energy_lower_wh": float(frame["predicted_energy_lower_wh"].sum()) if len(frame) else 0.0,
        "total_predicted_energy_upper_wh": float(frame["predicted_energy_upper_wh"].sum()) if len(frame) else 0.0,
    }
