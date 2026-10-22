# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: task_api.py
# 开发时间: 2026-10-20
# 文件名: task_api.py
# 功能说明: 校验车辆调度器计划输入并输出逐秒功率与能耗曲线
# 版本号：5.0

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import config
from vehicle_pipeline import ARCHITECTURE_VERSION, FEATURES, TCN, _device


SCHEDULER_COLUMNS = ["task_id", "payload_kg", *FEATURES]
FORBIDDEN_COLUMNS = {
    "power_w", "energy_wh", "actual_power_w", "actual_energy_wh", "voltage_v", "current_a",
    "REESSVoltage", "REESSCurrent", "speed_kmh", "temperature_c", "elevation_m", "gear",
}


def validate_scheduled_task(task: pd.DataFrame) -> None:
    """功能: 校验任务调度器输入字段、数值、时间网格和任务属性。
    参数: task为一个完整车辆任务的逐秒计划表。
    返回: None；输入不符合调度器契约时抛出ValueError。
    调用位置: predict_scheduled_task。
    """
    if task.empty:
        raise ValueError("任务计划不能为空")
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(task.columns))
    if forbidden:
        raise ValueError(f"任务前输入禁止包含实测状态或监督标签: {forbidden}")
    missing = sorted(set(SCHEDULER_COLUMNS) - set(task.columns))
    extra = sorted(set(task.columns) - set(SCHEDULER_COLUMNS))
    if missing or extra:
        raise ValueError(f"任务输入字段不符合契约；缺少={missing}，多余={extra}")
    if task["task_id"].nunique(dropna=False) != 1 or task["task_id"].isna().any():
        raise ValueError("一次只能预测一个非空task_id")
    numeric = task[SCHEDULER_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("任务计划字段必须是有限数值")
    task = task.copy()
    task[numeric.columns] = numeric
    task = task.sort_values("time_s", kind="stable").reset_index(drop=True)
    if (task.payload_kg < 0).any():
        raise ValueError("payload_kg不能为负")
    if task.payload_kg.nunique() != 1:
        raise ValueError("同一任务的payload_kg必须保持一致")
    times = task.time_s.to_numpy(float)
    if not np.isclose(times[0], 0.0, atol=1e-6) or np.any(np.diff(times) <= 0):
        raise ValueError("time_s必须从0开始并严格递增")
    if len(times) > 1 and not np.allclose(np.diff(times), config.SAMPLE_SECONDS, atol=1e-6):
        raise ValueError(f"任务轨迹必须按{config.SAMPLE_SECONDS:g}秒间隔采样")
    duration = task.task_duration_s.to_numpy(float)
    expected_duration = times[-1] + config.SAMPLE_SECONDS
    if not np.allclose(duration, duration[0], atol=1e-6) or not np.isclose(duration[0], expected_duration, atol=1e-6):
        raise ValueError("task_duration_s必须与完整计划轨迹的采样点数一致")
    if ((task.planned_stop_flag != 0) & (task.planned_stop_flag != 1)).any():
        raise ValueError("planned_stop_flag只能为0或1")


def predict_scheduled_task(task: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """功能: 依据调度器计划轨迹输出任务逐秒功率、区间能量及累计能量。
    参数: task为满足固定字段契约的单任务计划表。
    返回: 逐秒预测表和调度器可读任务摘要。
    调用位置: main.py的predict-task模式。
    """
    validate_scheduled_task(task)
    task = task.sort_values("time_s", kind="stable").reset_index(drop=True)
    checkpoint = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("当前模型权重架构版本不匹配，请先搜索并训练")
    scaler = json.loads(config.SCALER_FILE.read_text(encoding="utf-8"))
    features = checkpoint["features"]
    if not features or not set(features).issubset(FEATURES):
        raise ValueError("模型特征不属于调度器计划输入契约")
    if scaler.get("features") != features:
        raise ValueError("scaler与checkpoint特征顺序不一致")
    for key in ("mean", "std"):
        if not np.allclose(scaler.get(key, []), checkpoint.get(key, []), rtol=0, atol=1e-10):
            raise ValueError(f"scaler与checkpoint的{key}不一致")
    for key in ("target_mean", "target_std"):
        if not np.isclose(scaler.get(key, np.nan), checkpoint.get(key, np.nan), rtol=0, atol=1e-10):
            raise ValueError(f"scaler与checkpoint的{key}不一致")
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if mean.shape != (len(features),) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("模型标准化参数维数或取值无效")
    values = ((task[features].to_numpy(np.float32) - mean) / std).astype(np.float32)
    steps = int(checkpoint["window_steps"])
    padded = np.pad(values, ((steps - 1, 0), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, steps, axis=0).transpose(0, 2, 1).copy()
    params = checkpoint["params"]
    predictions = []
    for state in checkpoint["state_dicts"]:
        model = TCN(len(features), int(params["channels"]), int(params["kernel_size"]), float(params["dropout"]))
        model.load_state_dict(state)
        model.to(_device()).eval()
        with torch.no_grad():
            normalized = model(torch.from_numpy(windows).to(_device())).cpu().numpy()
        predictions.append(normalized * checkpoint["target_std"] + checkpoint["target_mean"])
    output = task.sort_values("time_s", kind="stable").reset_index(drop=True).copy()
    seed_predictions = np.stack(predictions, axis=0)
    output["predicted_power_w"] = np.mean(seed_predictions, axis=0)
    seed_std = np.std(seed_predictions, axis=0, ddof=1) if len(predictions) > 1 else np.zeros(len(output), dtype=float)
    calibration = json.loads(config.POWER_INTERVAL_CALIBRATION_JSON.read_text(encoding="utf-8"))
    power_quantile = float(calibration["quantile_w"])
    ci_half = np.full(len(output), power_quantile, dtype=float)
    output["power_seed_std_w"] = seed_std
    output["power_interval_half_width_w"] = ci_half
    output["power_prediction_interval_low_w"] = output.predicted_power_w - ci_half
    output["power_prediction_interval_high_w"] = output.predicted_power_w + ci_half
    output["power_ci95_low_w"] = output.power_prediction_interval_low_w
    output["power_ci95_high_w"] = output.power_prediction_interval_high_w
    output["predicted_energy_wh"] = output.predicted_power_w * config.SAMPLE_SECONDS / 3600.0
    output["energy_ci95_low_wh"] = output.power_ci95_low_w * config.SAMPLE_SECONDS / 3600.0
    output["energy_ci95_high_wh"] = output.power_ci95_high_w * config.SAMPLE_SECONDS / 3600.0
    output["cumulative_energy_wh"] = output.predicted_energy_wh.cumsum()
    output["cumulative_energy_ci95_low_wh"] = output.energy_ci95_low_wh.cumsum()
    output["cumulative_energy_ci95_high_wh"] = output.energy_ci95_high_wh.cumsum()
    elapsed_s = (np.arange(len(output)) + 1) * config.SAMPLE_SECONDS
    output["mean_energy_rate_wh_per_s"] = output.cumulative_energy_wh / elapsed_s
    output["mean_energy_rate_ci95_low_wh_per_s"] = output.cumulative_energy_ci95_low_wh / elapsed_s
    output["mean_energy_rate_ci95_high_wh_per_s"] = output.cumulative_energy_ci95_high_wh / elapsed_s
    summary = {
        "task_id": str(output.task_id.iloc[0]),
        "payload_kg": float(output.payload_kg.iloc[0]),
        "payload_used_by_model": False,
        "payload_limitation": "GECR训练数据没有载荷字段，当前预测不含载荷敏感效应。",
        "step_count": len(output),
        "task_duration_s": float(output.task_duration_s.iloc[0]),
        "predicted_total_energy_wh": float(output.predicted_energy_wh.sum()),
        "model_input_columns": features,
        "prediction_mode": "task_before_planned_trajectory",
        "confidence_interval": "trip-calibrated pointwise power interval; not a whole-trip simultaneous band",
        "power_interval_nominal_coverage": calibration["target_coverage"],
        "power_interval_calibration_trips": calibration["calibration_trip_count"],
        "power_interval_half_width_w": power_quantile,
    }
    return output, summary
