# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: rls.py
# 开发时间: 2026-10-30
# 文件名: rls.py
# 功能说明: 实现因果RLS窗口更新、区间计算和覆盖率所需的模拟
# 版本号：1.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def normal_quantile(confidence: float) -> float:
    """功能: 返回常用正态近似分位点。
    参数: confidence为置信水平。
    返回: 双侧区间使用的正态分位点。
    调用位置: DynamicEnergyRLS.interval。
    """
    if confidence >= 0.99:
        return 2.576
    if confidence >= 0.95:
        return 1.960
    if confidence >= 0.90:
        return 1.645
    return 1.282


class DynamicEnergyRLS:
    """功能: 用完整反馈窗口更新能耗偏置和比例参数。
    参数: forgetting_factor、initial_covariance、interval_scale为RLS参数。
    返回: 提供窗口预测、在线更新和预测区间方法的状态对象。
    调用位置: simulate_rls。
    """

    def __init__(self, forgetting_factor: float, initial_covariance: float,
                 interval_scale: float = 1.0, window_interval_scale: float = 1.0,
                 initial_residual_std_wh: float = 0.0) -> None:
        self.forgetting_factor = float(forgetting_factor)
        self.initial_covariance = float(initial_covariance)
        self.interval_scale = max(float(interval_scale), 1.0)
        self.window_interval_scale = max(float(window_interval_scale), 1.0)
        self.theta = np.array([0.0, 1.0], dtype=float)
        self.covariance = np.diag([
            self.initial_covariance * 0.01 ** 2,
            self.initial_covariance,
        ]).astype(float)
        self.residual_ema = 0.0
        self.residual_var = max(float(initial_residual_std_wh), 1e-6) ** 2
        self.last_residual_wh = 0.0
        self.update_count = 0

    def predict_energy(self, baseline_energy_wh: float, fraction: float = 1.0) -> float:
        """功能: 预测当前完整或截断窗口的校正能耗。
        参数: baseline_energy_wh为TCN窗口能耗，fraction为窗口长度比例。
        返回: 非负RLS窗口能耗预测。
        调用位置: simulate_rls。
        """
        phi = np.array([max(float(fraction), 0.0), float(baseline_energy_wh)])
        return max(0.0, float(phi @ self.theta))

    def predict_total(self, baseline_total_wh: float, horizon_windows: float) -> float:
        """功能: 预测剩余等效窗口的校正总能耗。
        参数: baseline_total_wh为剩余TCN能耗，horizon_windows为等效窗口数。
        返回: 非负剩余能耗预测。
        调用位置: simulate_rls。
        """
        if horizon_windows <= 0:
            return 0.0
        phi = np.array([float(horizon_windows), float(baseline_total_wh)])
        return max(0.0, float(phi @ self.theta))

    def update(self, baseline_energy_wh: float, actual_energy_wh: float) -> dict:
        """功能: 在窗口实际能耗到达后执行一次RLS更新。
        参数: baseline_energy_wh为更新前窗口预测，actual_energy_wh为窗口实测值。
        返回: 更新后的RLS状态摘要。
        调用位置: simulate_rls。
        """
        phi = np.array([1.0, float(baseline_energy_wh)])
        denominator = self.forgetting_factor + phi @ self.covariance @ phi
        gain = self.covariance @ phi / max(float(denominator), 1e-12)
        residual = float(actual_energy_wh) - float(phi @ self.theta)
        self.theta = self.theta + gain * residual
        self.theta[0] = np.clip(self.theta[0], -max(abs(actual_energy_wh), 1e-3), max(abs(actual_energy_wh), 1e-3))
        self.theta[1] = np.clip(self.theta[1], 0.25, 1.75)
        self.covariance = (self.covariance - np.outer(gain, phi) @ self.covariance) / self.forgetting_factor
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        centered = residual - self.residual_ema
        self.residual_var = 0.85 * self.residual_var + 0.15 * centered ** 2
        self.residual_ema = 0.85 * self.residual_ema + 0.15 * residual
        self.last_residual_wh = residual
        self.update_count += 1
        return self.state()

    def interval(self, baseline_total_wh: float, confidence: float, horizon_windows: float,
                 scale: float | None = None) -> tuple[float, float, float]:
        """功能: 根据RLS参数协方差和残差方差计算预测区间。
        参数: baseline_total_wh为基线总能耗，confidence为置信水平，horizon_windows为剩余窗口数，scale为校准倍率。
        返回: 下界、上界和未校准半径。
        调用位置: simulate_rls。
        """
        if horizon_windows <= 0:
            return 0.0, 0.0, 0.0
        phi = np.array([float(horizon_windows), float(baseline_total_wh)])
        innovation_var = max(self.residual_var, (0.25 * abs(self.residual_ema)) ** 2, 1e-8)
        parameter_var = innovation_var * max(0.0, float(phi @ self.covariance @ phi))
        residual_var = innovation_var * max(float(horizon_windows), 1.0)
        raw_radius = normal_quantile(confidence) * math.sqrt(parameter_var + residual_var)
        applied = self.interval_scale if scale is None else max(float(scale), 1.0)
        mean = self.predict_total(baseline_total_wh, horizon_windows)
        radius = applied * raw_radius
        return max(0.0, mean - radius), max(0.0, mean + radius), raw_radius

    def state(self) -> dict:
        """功能: 导出可写入日志的RLS状态。
        参数: 无。
        返回: 参数、协方差和残差状态字典。
        调用位置: simulate_rls。
        """
        return {
            "theta_bias_wh_per_window": float(self.theta[0]),
            "theta_scale": float(self.theta[1]),
            "covariance_trace": float(np.trace(self.covariance)),
            "residual_std_wh": float(math.sqrt(max(self.residual_var, 0.0))),
            "last_residual_wh": float(self.last_residual_wh),
            "update_count": int(self.update_count),
        }


@dataclass
class SimulationOutput:
    """功能: 保存RLS模拟产生的逐样本、逐窗口和逐flight结果。
    参数: predictions、windows、flights、trace为四类结果表。
    返回: 可供调参、评价和绘图使用的结果对象。
    调用位置: rls_evaluate.py。
    """

    predictions: pd.DataFrame
    windows: pd.DataFrame
    flights: pd.DataFrame
    trace: pd.DataFrame


def simulate_rls(frame: pd.DataFrame, baseline_power: np.ndarray, forgetting_factor: float,
                 initial_covariance: float, window_seconds: float, resample_seconds: float,
                 confidence: float = 0.95, interval_scale: float = 1.0,
                 window_interval_scale: float = 1.0,
                 initial_residual_std_wh: float = 0.01) -> SimulationOutput:
    """功能: 按完整flight执行无未来信息泄漏的RLS和区间模拟。
    参数: frame为按flight/time排序的数据，baseline_power为TCN功率，后续参数为RLS与采样配置。
    返回: 逐样本、逐窗口、逐flight及状态轨迹结果。
    调用位置: tune_rls、calibrate_intervals、evaluate_group。
    """
    ordered = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True).copy()
    baseline_power = np.asarray(baseline_power, dtype=float)
    if len(ordered) != len(baseline_power):
        raise ValueError("TCN基线长度与RLS评估表不一致")
    steps = max(1, int(round(float(window_seconds) / resample_seconds)))
    row_power = np.zeros(len(ordered), dtype=float)
    row_lower = np.zeros(len(ordered), dtype=float)
    row_upper = np.zeros(len(ordered), dtype=float)
    window_rows: list[dict] = []
    flight_rows: list[dict] = []
    trace_rows: list[dict] = []
    for flight_id, group in ordered.groupby("flight", sort=False):
        indices = group.index.to_numpy()
        base_power = baseline_power[indices]
        dt = group["dt_seconds"].to_numpy(float)
        actual_power = group["power_w"].to_numpy(float)
        base_energy = base_power * dt / 3600.0
        actual_energy = actual_power * dt / 3600.0
        rls = DynamicEnergyRLS(forgetting_factor, initial_covariance, interval_scale,
                               window_interval_scale, initial_residual_std_wh)
        horizon = len(indices) / steps
        initial_base = float(base_energy.sum())
        initial_lower, initial_upper, initial_raw_radius = rls.interval(
            initial_base, confidence, horizon, scale=interval_scale,
        )
        actual_total = float(actual_energy.sum())
        trace_rows.append({
            "flight": flight_id, "route": str(group["route"].iloc[0]), "stage": "initial",
            "observed_window_count": 0, "remaining_window_count": horizon,
            "actual_final_energy_wh": actual_total, "predicted_final_energy_wh": initial_base,
            "predicted_final_lower_wh": initial_lower, "predicted_final_upper_wh": initial_upper,
            "raw_radius_wh": initial_raw_radius, "interval_width_wh": initial_upper - initial_lower,
            "rls_update_applied": False, **rls.state(),
        })
        windows = [np.arange(start, min(start + steps, len(indices))) for start in range(0, len(indices), steps)]
        predicted_total = 0.0
        for window_index, positions in enumerate(windows):
            n = len(positions)
            fraction = n / steps
            base_window = float(base_energy[positions].sum())
            actual_window = float(actual_energy[positions].sum())
            lower_window, upper_window, raw_window_radius = rls.interval(
                base_window, confidence, fraction, scale=window_interval_scale,
            )
            corrected_window = rls.predict_energy(base_window, fraction)
            weights = base_energy[positions].copy()
            if float(weights.sum()) <= 1e-12:
                weights[:] = 1.0 / n
            else:
                weights /= float(weights.sum())
            row_power[indices[positions]] = corrected_window * weights * 3600.0 / dt[positions]
            row_lower[indices[positions]] = lower_window * weights * 3600.0 / dt[positions]
            row_upper[indices[positions]] = upper_window * weights * 3600.0 / dt[positions]
            predicted_total += corrected_window
            complete = n == steps
            window_rows.append({
                "flight": flight_id, "route": str(group["route"].iloc[0]), "window": window_index,
                "samples": n, "is_complete_window": complete,
                "actual_energy_wh": actual_window, "tcn_energy_wh": base_window,
                "predicted_energy_wh": corrected_window, "predicted_lower_wh": lower_window,
                "predicted_upper_wh": upper_window, "raw_radius_wh": raw_window_radius,
                "rls_update_applied": complete,
            })
            remaining_start = int(positions[-1]) + 1
            remaining_base = float(base_energy[remaining_start:].sum())
            remaining_windows = max(len(indices) - remaining_start, 0) / steps
            before_remaining = rls.predict_total(remaining_base, remaining_windows)
            if complete:
                state = rls.update(base_window, actual_window)
            else:
                state = rls.state()
            after_remaining = rls.predict_total(remaining_base, remaining_windows)
            lower_remaining, upper_remaining, raw_remaining_radius = rls.interval(
                remaining_base, confidence, remaining_windows, scale=interval_scale,
            )
            consumed = float(actual_energy[:remaining_start].sum())
            trace_rows.append({
                "flight": flight_id, "route": str(group["route"].iloc[0]), "stage": "post_update",
                "observed_window_count": int(rls.update_count), "remaining_window_count": remaining_windows,
                "actual_final_energy_wh": actual_total,
                "predicted_final_energy_wh": consumed + after_remaining,
                "predicted_final_lower_wh": consumed + lower_remaining,
                "predicted_final_upper_wh": consumed + upper_remaining,
                "raw_radius_wh": raw_remaining_radius,
                "interval_width_wh": upper_remaining - lower_remaining,
                "remaining_prediction_before_update_wh": consumed + before_remaining,
                "window_actual_energy_wh": actual_window,
                "window_predicted_energy_wh": corrected_window,
                "window_lower_wh": lower_window, "window_upper_wh": upper_window,
                "window_raw_radius_wh": raw_window_radius,
                "rls_update_applied": complete, **state,
            })
        flight_rows.append({
            "flight": flight_id, "route": str(group["route"].iloc[0]),
            "actual_energy_wh": actual_total, "tcn_energy_wh": initial_base,
            "predicted_energy_wh": predicted_total, "predicted_lower_wh": initial_lower,
            "predicted_upper_wh": initial_upper, "rls_updates": int(rls.update_count),
        })
    predictions = ordered[["flight", "route", "time", "time_s", "dt_seconds", "power_w"]].copy()
    predictions["tcn_predicted_power_w"] = baseline_power
    predictions["predicted_power_w"] = row_power
    predictions["predicted_power_lower_w"] = row_lower
    predictions["predicted_power_upper_w"] = row_upper
    predictions["actual_energy_wh"] = predictions.power_w * predictions.dt_seconds / 3600.0
    predictions["predicted_energy_wh"] = predictions.predicted_power_w * predictions.dt_seconds / 3600.0
    predictions["predicted_energy_lower_wh"] = predictions.predicted_power_lower_w * predictions.dt_seconds / 3600.0
    predictions["predicted_energy_upper_wh"] = predictions.predicted_power_upper_w * predictions.dt_seconds / 3600.0
    predictions["power_error_w"] = predictions.predicted_power_w - predictions.power_w
    predictions["tcn_power_error_w"] = predictions.tcn_predicted_power_w - predictions.power_w
    return SimulationOutput(predictions, pd.DataFrame(window_rows), pd.DataFrame(flight_rows), pd.DataFrame(trace_rows))


def interval_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   confidence: float) -> float:
    """功能: 计算双侧预测区间的平均区间评分。
    参数: actual、lower、upper为实际值和区间边界，confidence为置信水平。
    返回: 平均区间评分。
    调用位置: summarise_output。
    """
    alpha = max(1.0 - float(confidence), 1e-6)
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    penalty = np.where(actual < lower, 2.0 / alpha * (lower - actual),
                       np.where(actual > upper, 2.0 / alpha * (actual - upper), 0.0))
    return float(np.mean(upper - lower + penalty))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """功能: 计算实际值落入预测区间的百分比。
    参数: actual、lower、upper为实际值和区间边界。
    返回: 覆盖率百分比。
    调用位置: summarise_output。
    """
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    return float(np.mean((actual >= lower) & (actual <= upper)) * 100.0)


def calibrate_scales(output: SimulationOutput, confidence: float) -> dict:
    """功能: 用验证残差将原始RLS区间半径校准到目标置信水平。
    参数: output为验证集RLS模拟结果，confidence为置信水平。
    返回: 任务级和窗口级区间倍率及校准样本数。
    调用位置: rls_evaluate.py。
    """
    trace = output.trace.copy()
    active = trace[trace.remaining_window_count > 0].copy()
    task_raw = np.maximum(active.raw_radius_wh.to_numpy(float), 1e-8)
    task_ratio = np.abs(active.actual_final_energy_wh.to_numpy(float) - active.predicted_final_energy_wh.to_numpy(float)) / task_raw
    windows = output.windows[output.windows.is_complete_window].copy()
    window_raw = np.maximum(windows.raw_radius_wh.to_numpy(float), 1e-8)
    window_ratio = np.abs(windows.actual_energy_wh.to_numpy(float) - windows.predicted_energy_wh.to_numpy(float)) / window_raw
    quantile = confidence
    task_scale = max(1.0, float(np.quantile(task_ratio[np.isfinite(task_ratio)], quantile))) if len(task_ratio) else 1.0
    window_scale = max(1.0, float(np.quantile(window_ratio[np.isfinite(window_ratio)], quantile))) if len(window_ratio) else 1.0
    return {
        "confidence_level": float(confidence), "method": "validation_residual_ratio_quantile",
        "task_interval_scale": task_scale, "window_interval_scale": window_scale,
        "task_calibration_points": int(len(task_ratio)), "window_calibration_points": int(len(window_ratio)),
        "task_ratio_quantile": float(quantile), "window_ratio_quantile": float(quantile),
    }
