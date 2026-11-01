# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: evaluate.py
# 开发时间: 2026-10-30
# 文件名: evaluate.py
# 功能说明: 评价四组TCN在测试集上的离线功率和能耗预测
# 版本号：1.0

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import GROUPS, ExperimentConfig, group_paths
from data_utils import load_tables, prepare_features, feature_columns, validate_features
from train import _metrics, _model_from_checkpoint, build_sequences, predict_array


def evaluate_group(group: str, cfg: ExperimentConfig) -> dict:
    """功能: 读取指定组模型并生成测试集逐时刻预测、功率和能耗指标。
    参数: group为A、B、C或D，cfg为配置。
    返回: 测试指标摘要。
    调用位置: main.py。
    """

    paths = group_paths(cfg, group)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    scaler = json.loads(paths["scaler"].read_text(encoding="utf-8"))
    expected_columns = feature_columns(GROUPS[group]["use_sma"])
    for data in (checkpoint, scaler):
        if data["group"] != group or data["feature_columns"] != expected_columns:
            raise ValueError("模型、标准化器与实验组的特征契约不一致")
        if data["sma_window_steps"] != cfg.sma_window_steps:
            raise ValueError("模型与当前SMA配置不一致")
    if checkpoint["activation"] != GROUPS[group]["activation"]:
        raise ValueError("模型激活模式与实验组不一致")
    _, _, test = load_tables(cfg)
    test = prepare_features(test, GROUPS[group]["use_sma"], cfg.sma_window_steps)
    columns = list(checkpoint["feature_columns"])
    validate_features(test, columns)
    sequence, _ = build_sequences(test, scaler, int(checkpoint["window_steps"]), require_target=False)
    model = _model_from_checkpoint(checkpoint, len(columns))
    predicted = predict_array(model, sequence, scaler, cfg)
    output = test[["flight", "route", "time", "time_s", "dt_seconds", "power_w", "energy_interval_wh"]].copy()
    output["predicted_power_w"] = predicted
    output["predicted_energy_wh"] = predicted * output["dt_seconds"].to_numpy(float) / 3600.0
    output["power_error_w"] = output["predicted_power_w"] - output["power_w"]
    output["energy_error_wh"] = output["predicted_energy_wh"] - output["energy_interval_wh"]
    output["actual_cumulative_energy_wh"] = output.groupby("flight")["energy_interval_wh"].cumsum()
    output["predicted_cumulative_energy_wh"] = output.groupby("flight")["predicted_energy_wh"].cumsum()
    output.to_csv(paths["predictions"], index=False, encoding="utf-8")
    flights = output.groupby(["flight", "route"], sort=False).agg(
        actual_energy_wh=("energy_interval_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"),
        duration_s=("dt_seconds", "sum"), samples=("time", "size")).reset_index()
    flights["error_wh"] = flights.predicted_energy_wh - flights.actual_energy_wh
    flights["absolute_error_wh"] = flights.error_wh.abs()
    flights["absolute_error_percent"] = 100 * flights.absolute_error_wh / flights.actual_energy_wh.clip(lower=1e-9)
    flights.to_csv(paths["flight_metrics"], index=False, encoding="utf-8")
    windows = output.assign(window=output.groupby("flight").cumcount() // 5).groupby(
        ["flight", "window"], sort=False).agg(actual_energy_wh=("energy_interval_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"), samples=("time", "size")).reset_index()
    windows.to_csv(paths["window_metrics"], index=False, encoding="utf-8")
    stages = np.select([test.planned_motor_on.eq(0), test.planned_airborne.eq(0)],
                       ["motor_off", "ground_motor_on"], default="airborne")
    rows = []
    for stage in ("motor_off", "ground_motor_on", "airborne"):
        mask = stages == stage
        if mask.any():
            stage_metrics = _metrics(test.loc[mask].reset_index(drop=True), predicted[mask], cfg)
            rows.append({"stage": stage, **{key: value for key, value in stage_metrics.items()
                                            if key.startswith("power_") or key in {"rows", "flights"}}})
    pd.DataFrame(rows).to_csv(paths["stage_metrics"], index=False, encoding="utf-8")
    route_rows = []
    for route, subset in test.groupby("route"):
        route_rows.append({"route": route, **_metrics(subset.reset_index(drop=True), predicted[subset.index], cfg)})
    pd.DataFrame(route_rows).to_csv(paths["route_metrics"], index=False, encoding="utf-8")
    metrics = _metrics(test, predicted, cfg)
    metrics.update({"group": group, "name": GROUPS[group]["name"], "activation": GROUPS[group]["activation"],
                    "use_sma": GROUPS[group]["use_sma"], "sma_window_steps": cfg.sma_window_steps,
                    "feature_count": len(columns), "window_steps": int(checkpoint["window_steps"]),
                    "window_seconds": float(checkpoint["window_seconds"]), "epochs": checkpoint["epochs_run"],
                    "tune_epochs": checkpoint["tune_epochs"], "best_epoch": checkpoint["best_epoch"],
                    "seed": checkpoint["seed"], "parameter_count": checkpoint["parameter_count"],
                    "training_seconds": checkpoint["training_seconds"],
                    "flight_equal_weight_mape_percent": float(flights.absolute_error_percent.mean()),
                    "flight_absolute_error_p90_wh": float(flights.absolute_error_wh.quantile(.9)),
                    "model_file": str(paths["checkpoint"]),
                    "prediction_file": str(paths["predictions"])})
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_all(cfg: ExperimentConfig, groups=None) -> pd.DataFrame:
    """功能: 评价A、B、C、D四组并汇总结果。
    参数: cfg为配置。
    返回: 四组测试指标表。
    调用位置: main.py。
    """

    rows = [evaluate_group(group, cfg) for group in (groups or GROUPS)]
    if cfg.summary_csv.exists():
        previous = pd.read_csv(cfg.summary_csv)
        selected = [row["group"] for row in rows]
        rows.extend(previous.loc[~previous.group.isin(selected)].to_dict("records"))
    result = pd.DataFrame(rows)
    result = result.sort_values("group").reset_index(drop=True)
    result.to_csv(cfg.summary_csv, index=False, encoding="utf-8")
    return result
