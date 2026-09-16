# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: evaluate.py
# 开发时间: 2026-09-16
# 功能说明: 评估实验4.0规划TCN、窗口级动态RLS和任务能耗区间
# 版本号：4.0

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories
from model import DynamicEnergyRLS
from train import predict_frame


def _metrics(actual: np.ndarray, predicted: np.ndarray, prefix: str) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    denominator = max(float(np.sum(np.abs(actual))), 1e-9)
    return {
        f"{prefix}_mae": float(np.mean(np.abs(error))) if len(error) else 0.0,
        f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else 0.0,
        f"{prefix}_wape_percent": float(np.sum(np.abs(error)) / denominator * 100.0),
        f"{prefix}_underestimate_rate_percent": float(np.mean(predicted < actual) * 100.0) if len(error) else 0.0,
    }


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("flight", "time", "route") if c in frame.columns]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True).copy()


def _simulate(frame: pd.DataFrame, baseline_power: np.ndarray, factor: float, covariance: float,
              cfg: ExperimentConfig, keep_rows: bool = False) -> tuple[dict, pd.DataFrame | None]:
    ordered = _ordered(frame)
    baseline_power = np.asarray(baseline_power, dtype=float)
    if len(ordered) != len(baseline_power):
        raise ValueError("baseline预测长度与评估表不一致")
    steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
    row_predictions = np.zeros(len(ordered), dtype=float)
    row_lower = np.zeros(len(ordered), dtype=float)
    row_upper = np.zeros(len(ordered), dtype=float)
    flight_rows = []
    for flight, group in ordered.groupby("flight", sort=False):
        indices = group.index.to_numpy()
        rls = DynamicEnergyRLS(factor, covariance)
        actual_total = 0.0
        predicted_total = 0.0
        for start in range(0, len(indices), steps):
            current = indices[start:start + steps]
            baseline_energy = float(np.sum(baseline_power[current]) * cfg.resample_seconds / 3600.0)
            actual_energy = float(np.sum(ordered.loc[current, "power_w"].to_numpy(float)) * cfg.resample_seconds / 3600.0)
            corrected_energy = rls.predict_energy(baseline_energy)
            weights = np.maximum(baseline_power[current], 1e-6)
            row_predictions[current] = corrected_energy * weights / float(weights.sum()) * 3600.0 / cfg.resample_seconds
            lower, upper = rls.interval(baseline_energy, cfg.default_confidence, 1)
            row_lower[current] = lower * weights / float(weights.sum()) * 3600.0 / cfg.resample_seconds
            row_upper[current] = upper * weights / float(weights.sum()) * 3600.0 / cfg.resample_seconds
            predicted_total += corrected_energy
            actual_total += actual_energy
            rls.update_window(baseline_energy, actual_energy)
        flight_rows.append({"flight": flight, "route": str(group["route"].iloc[0]),
                            "actual_energy_wh": actual_total, "predicted_energy_wh": predicted_total,
                            "rls_updates": rls.update_count})
    result = pd.DataFrame(flight_rows)
    rows = None
    if keep_rows:
        rows = ordered.copy()
        rows["tcn_predicted_power_w"] = baseline_power
        rows["predicted_power_w"] = row_predictions
        rows["predicted_power_lower_w"] = row_lower
        rows["predicted_power_upper_w"] = row_upper
        rows["actual_energy_wh"] = rows["power_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_wh"] = rows["predicted_power_w"] * rows["dt_seconds"] / 3600.0
        rows["tcn_predicted_energy_wh"] = rows["tcn_predicted_power_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_lower_wh"] = rows["predicted_power_lower_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_upper_wh"] = rows["predicted_power_upper_w"] * rows["dt_seconds"] / 3600.0
    return {"flight_energy_wape_percent": _metrics(result.actual_energy_wh, result.predicted_energy_wh, "flight_energy")["flight_energy_wape_percent"]}, rows


def evaluate_rls_parameters(frame: pd.DataFrame, baseline: np.ndarray, factor: float,
                            covariance: float, cfg: ExperimentConfig) -> dict:
    """按完整flight模拟窗口结束更新的RLS，返回飞行级验证误差。"""
    metrics, _ = _simulate(frame, baseline, factor, covariance, cfg, keep_rows=False)
    return metrics


def evaluate_model(cfg: ExperimentConfig) -> dict:
    """生成测试集结构化预测、分层指标和动态区间摘要。"""
    ensure_directories(cfg)
    test = _ordered(pd.read_csv(cfg.test_csv))
    baseline = predict_frame(test, cfg)
    if cfg.rls_tuning_csv.exists() and len(pd.read_csv(cfg.rls_tuning_csv)):
        tuning = pd.read_csv(cfg.rls_tuning_csv).sort_values("flight_energy_wape_percent")
        best = tuning.iloc[0]
        factor, covariance = float(best.forgetting_factor), float(best.initial_covariance)
    else:
        factor, covariance = 0.96, 0.5
    _, predictions = _simulate(test, baseline, factor, covariance, cfg, keep_rows=True)
    assert predictions is not None
    predictions.to_csv(cfg.predictions_csv, index=False, encoding="utf-8")
    flight = predictions.groupby("flight", sort=True).agg(
        route=("route", "first"), actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"), tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"), predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    window_steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
    window = predictions.assign(window_index=(predictions.groupby("flight").cumcount() // window_steps))
    window = window.groupby(["flight", "window_index"], sort=True).agg(
        route=("route", "first"), actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"), tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"), predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    metrics = {
        **_metrics(predictions.power_w.to_numpy(float), predictions.predicted_power_w.to_numpy(float), "sample_power_w"),
        **_metrics(predictions.power_w.to_numpy(float), predictions.tcn_predicted_power_w.to_numpy(float), "tcn_sample_power_w"),
        **_metrics(flight.actual_energy_wh.to_numpy(float), flight.predicted_energy_wh.to_numpy(float), "flight_energy_wh"),
        **_metrics(flight.actual_energy_wh.to_numpy(float), flight.tcn_predicted_energy_wh.to_numpy(float), "tcn_flight_energy_wh"),
        **_metrics(window.actual_energy_wh.to_numpy(float), window.predicted_energy_wh.to_numpy(float), "window_energy_wh"),
        **_metrics(window.actual_energy_wh.to_numpy(float), window.tcn_predicted_energy_wh.to_numpy(float), "tcn_window_energy_wh"),
        "test_rows": int(len(predictions)), "test_flights": int(flight.flight.nunique()),
        "window_count": int(len(window)), "rls_window_seconds": cfg.rls_window_seconds,
        "rls_forgetting_factor": factor, "rls_initial_covariance": covariance,
        "confidence_level": cfg.default_confidence,
        "sample_power_interval_coverage_percent": float(np.mean((predictions.power_w >= predictions.predicted_power_lower_w) & (predictions.power_w <= predictions.predicted_power_upper_w)) * 100.0),
        "flight_energy_interval_coverage_percent": float(np.mean((flight.actual_energy_wh >= flight.predicted_energy_lower_wh) & (flight.actual_energy_wh <= flight.predicted_energy_upper_wh)) * 100.0),
        "flight_energy_interval_mean_width_wh": float(np.mean(flight.predicted_energy_upper_wh - flight.predicted_energy_lower_wh)),
        "prediction_file": str(cfg.predictions_csv),
    }
    cfg.out_model_dir.mkdir(parents=True, exist_ok=True)
    flight.to_csv(cfg.out_model_dir / "flight_energy_summary_4.0.csv", index=False, encoding="utf-8")
    window.to_csv(cfg.out_model_dir / "window_energy_summary_4.0.csv", index=False, encoding="utf-8")
    power_bins = [-np.inf, 50.0, 150.0, 300.0, 450.0, 600.0, np.inf]
    power_labels = ["0-50W", "50-150W", "150-300W", "300-450W", "450-600W", "600W+"]
    power_table = predictions.copy()
    power_table["power_bin"] = pd.cut(power_table["power_w"], bins=power_bins, labels=power_labels)
    power_rows = []
    for label, group in power_table.groupby("power_bin", observed=True):
        error = group["predicted_power_w"] - group["power_w"]
        power_rows.append({"power_bin": str(label), "rows": int(len(group)),
                           "actual_mean_power_w": float(group.power_w.mean()),
                           "predicted_mean_power_w": float(group.predicted_power_w.mean()),
                           "mae_w": float(np.mean(np.abs(error))),
                           "rmse_w": float(np.sqrt(np.mean(error ** 2))),
                           "wape_percent": float(np.sum(np.abs(error)) / max(np.sum(np.abs(group.power_w)), 1e-9) * 100.0)})
    pd.DataFrame(power_rows).to_csv(cfg.out_model_dir / "power_bin_evaluation_4.0.csv", index=False, encoding="utf-8")
    route = predictions.groupby("route", sort=True).agg(
        flights=("flight", "nunique"), rows=("flight", "size"),
        actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"),
        tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"),
        predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    route["energy_error_wh"] = route.predicted_energy_wh - route.actual_energy_wh
    route["energy_abs_error_wh"] = route.energy_error_wh.abs()
    route["energy_wape_percent"] = route.energy_abs_error_wh / route.actual_energy_wh.abs().clip(lower=1e-9) * 100.0
    route.to_csv(cfg.out_model_dir / "route_energy_summary_4.0.csv", index=False, encoding="utf-8")
    pd.DataFrame([{
        "metric": "power_wape_percent", "tcn_value": metrics["tcn_sample_power_w_wape_percent"], "rls_value": metrics["sample_power_w_wape_percent"]
    }, {
        "metric": "flight_energy_wape_percent", "tcn_value": metrics["tcn_flight_energy_wh_wape_percent"], "rls_value": metrics["flight_energy_wh_wape_percent"]
    }, {
        "metric": "window_energy_wape_percent", "tcn_value": metrics["tcn_window_energy_wh_wape_percent"], "rls_value": metrics["window_energy_wh_wape_percent"]
    }]).to_csv(cfg.out_model_dir / "tcn_vs_rls_summary_4.0.csv", index=False, encoding="utf-8")
    cfg.evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(cfg.evaluation_csv, index=False, encoding="utf-8")
    cfg.evaluation_json.write_text(json.dumps({"version": "4.0", "metrics": metrics,
                                                "input_policy": "power_w仅作为离线标签，RLS只使用窗口预测能耗与实际能耗"},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
