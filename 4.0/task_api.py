# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: task_api.py
# 开发时间: 2026-09-16
# 文件名: task_api.py
# 功能说明: 提供实验4.0任务前候选路线预测和飞行中剩余路线更新接口
# 版本号：4.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data_utils import FEATURE_COLUMNS
from model import DynamicEnergyRLS
from train import predict_frame


REQUIRED_PLANNING_COLUMNS = [
    "task_id", "time_s", "dt_seconds", "planned_vx_mps", "planned_vy_mps", "planned_vz_mps",
    "planned_speed_mps", "planned_ax_mps2", "planned_ay_mps2", "planned_az_mps2",
    "planned_altitude_m", "wind_speed_mps", "wind_direction_deg", "payload_kg",
]


def validate_task_input(task: pd.DataFrame, cfg: ExperimentConfig) -> None:
    """功能: 检查任务前规划器输入是否满足4.0数据契约。
    参数: task为候选路线表，cfg为实验配置对象。
    返回: None；不符合契约时抛出ValueError。
    调用位置: planning_to_model_frame、predict_task_before。
    """

    missing = sorted(set(REQUIRED_PLANNING_COLUMNS) - set(task.columns))
    if missing:
        raise ValueError(f"任务前输入缺少字段: {missing}")
    dt = pd.to_numeric(task["dt_seconds"], errors="coerce")
    if dt.isna().any() or not np.allclose(dt.to_numpy(), cfg.resample_seconds, atol=1e-6):
        raise ValueError(f"任务前输入必须统一使用{cfg.resample_seconds:g}s离散步长")
    forbidden = {"battery_voltage", "battery_current", "power_w", "energy_interval_wh", "actual_speed_mps", "actual_acceleration_mps2"}
    present = sorted(forbidden.intersection(task.columns))
    if present:
        raise ValueError(f"任务前输入禁止包含实测或标签字段: {present}")


def planning_to_model_frame(task: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 将规划器字段转换为TCN需要的低维规划特征。
    参数: task为完整候选路线，cfg为配置对象。
    返回: 不含真实电池测量字段的模型输入表。
    调用位置: predict_task_before、OnlineTaskSession。
    """

    validate_task_input(task, cfg)
    result = task.copy().sort_values("time_s").reset_index(drop=True)
    result["flight"] = result["task_id"].astype(str)
    result["route"] = result.get("route", "PLANNER_ROUTE").astype(str) if "route" in result else "PLANNER_ROUTE"
    result["time"] = result["time_s"].astype(float)
    result["task_progress"] = result["time_s"] / max(float(result["time_s"].iloc[-1]), cfg.resample_seconds)
    result["task_duration_s"] = max(float(result["time_s"].iloc[-1]), cfg.resample_seconds)
    result["planned_horizontal_speed_mps"] = np.sqrt(result.planned_vx_mps**2 + result.planned_vy_mps**2)
    result["wind_sin"] = np.sin(np.deg2rad(result.wind_direction_deg))
    result["wind_cos"] = np.cos(np.deg2rad(result.wind_direction_deg))
    wind_x = result.wind_speed_mps * result.wind_cos
    wind_y = result.wind_speed_mps * result.wind_sin
    horizontal = result.planned_horizontal_speed_mps.clip(lower=1e-6)
    result["relative_air_speed_mps"] = np.sqrt((result.planned_vx_mps - wind_x) ** 2 + (result.planned_vy_mps - wind_y) ** 2 + result.planned_vz_mps**2)
    result["headwind_mps"] = -(result.planned_vx_mps * wind_x + result.planned_vy_mps * wind_y) / horizontal
    result["crosswind_mps"] = np.abs(result.planned_vx_mps * wind_y - result.planned_vy_mps * wind_x) / horizontal
    result["planned_acceleration_mps2"] = np.sqrt(result.planned_ax_mps2**2 + result.planned_ay_mps2**2 + result.planned_az_mps2**2)
    result["segment_distance_m"] = result.planned_speed_mps * result.dt_seconds
    result["cumulative_distance_m"] = result.segment_distance_m.cumsum()
    result["cumulative_climb_m"] = (result.planned_vz_mps.clip(lower=0) * result.dt_seconds).cumsum()
    for column, default in (("payload_camera_enabled", 0.0), ("payload_communication_enabled", 0.0),
                            ("payload_compute_enabled", 0.0), ("payload_delivery_enabled", 0.0),
                            ("payload_rated_power_w", 0.0)):
        result[column] = pd.to_numeric(result.get(column, default), errors="coerce").fillna(default)
    result["power_w"] = 0.0
    result["energy_interval_wh"] = 0.0
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result[["flight", "route", "time", "power_w", "energy_interval_wh", *FEATURE_COLUMNS]].copy()


def _normal_z(confidence: float) -> float:
    return 1.96 if confidence >= 0.95 else 1.645


def predict_task_before(task: pd.DataFrame, cfg: ExperimentConfig, confidence: float | None = None,
                        available_energy_wh: float | None = None, return_energy_wh: float = 0.0,
                        safety_reserve_wh: float | None = None) -> tuple[pd.DataFrame, dict]:
    """功能: 生成整条候选路线的任务前分段、累计、区间和决策输出。
    参数: task为规划器路线，cfg为配置对象，其他参数用于电池安全判断。
    返回: 每步预测表和可被调度器读取的任务摘要。
    调用位置: main.py、OnlineTaskSession。
    """

    confidence = confidence or cfg.default_confidence
    model_frame = planning_to_model_frame(task, cfg)
    base_power = predict_frame(model_frame, cfg)
    base_energy = base_power * cfg.resample_seconds / 3600.0
    residual_std = float(json.loads(cfg.uncertainty_json.read_text(encoding="utf-8")).get("base_residual_std_w", 20.0)) if cfg.uncertainty_json.exists() else 20.0
    z = _normal_z(confidence)
    n = len(model_frame)
    variance_total = np.maximum((residual_std * cfg.resample_seconds / 3600.0) ** 2 * n, 1e-10)
    step_radius = z * residual_std * cfg.resample_seconds / 3600.0
    output = task.copy().sort_values("time_s").reset_index(drop=True)
    output["predicted_power_w"] = base_power
    output["predicted_energy_wh"] = base_energy
    output["predicted_energy_lower_wh"] = np.maximum(base_energy - step_radius, 0.0)
    output["predicted_energy_upper_wh"] = base_energy + step_radius
    output["cumulative_energy_wh"] = np.cumsum(base_energy)
    cumulative_radius = z * np.sqrt(variance_total)
    output["cumulative_energy_lower_wh"] = np.maximum(output["cumulative_energy_wh"] - cumulative_radius, 0.0)
    output["cumulative_energy_upper_wh"] = output["cumulative_energy_wh"] + cumulative_radius
    total = float(base_energy.sum())
    lower, upper = max(0.0, total - cumulative_radius), total + cumulative_radius
    reserve = float(safety_reserve_wh if safety_reserve_wh is not None else max(0.1 * upper, 1.0))
    required = upper + float(return_energy_wh) + reserve
    feasible = available_energy_wh is None or float(available_energy_wh) >= required
    summary = {"version": "4.0", "task_id": str(task.task_id.iloc[0]), "step_count": n,
               "total_energy_wh": total, "total_energy_lower_wh": lower, "total_energy_upper_wh": upper,
               "confidence": confidence, "confidence_interval_method": "任务级残差方差合成",
               "available_energy_wh": available_energy_wh, "return_energy_upper_wh": return_energy_wh,
               "safety_reserve_wh": reserve, "required_energy_wh": required,
               "battery_feasible": bool(feasible), "charge_or_replace_required": bool(not feasible),
               "decision": "continue" if feasible else "charge_or_replace"}
    return output, summary


class OnlineTaskSession:
    """功能: 维护任务前基线并在窗口结束后修正剩余路线。
    参数: task为任务规划表，cfg为4.0配置，rls_params为选定RLS参数。
    返回: update后输出剩余路线和调度决策。
    调用位置: main.py的online-demo。
    """

    def __init__(self, task: pd.DataFrame, cfg: ExperimentConfig, rls_params: tuple[float, float] = (0.96, 0.5)) -> None:
        self.task = task.sort_values("time_s").reset_index(drop=True)
        self.cfg = cfg
        self.base, self.summary = predict_task_before(self.task, cfg)
        self.model_frame = planning_to_model_frame(self.task, cfg)
        self.rls = DynamicEnergyRLS(*rls_params)
        self.executed_energy_wh = 0.0
        self.last_window = -1

    def update(self, window_index: int, actual_energy_wh: float, available_energy_wh: float | None = None,
               return_energy_upper_wh: float = 0.0, safety_reserve_wh: float | None = None) -> tuple[pd.DataFrame, dict]:
        """功能: 窗口结束后更新RLS并重新计算当前窗口之后的整条剩余路线。
        参数: window_index为已结束窗口编号，actual_energy_wh为外部计量实际能耗。
        返回: 剩余路线输出和充电/返航决策摘要。
        调用位置: 在线飞行控制器。
        """

        if window_index <= self.last_window:
            raise ValueError("RLS窗口必须按时间递增且每个完整窗口只能更新一次")
        end = min(len(self.base), (window_index + 1) * max(1, int(round(self.cfg.rls_window_seconds / self.cfg.resample_seconds))))
        start = self.last_window + 1
        baseline_window = float(self.base.loc[start:end - 1, "predicted_energy_wh"].sum())
        self.rls.update_window(baseline_window, float(actual_energy_wh))
        self.executed_energy_wh += float(actual_energy_wh)
        self.last_window = end - 1
        remain = self.base.iloc[end:].copy()
        if len(remain):
            remain["predicted_energy_wh"] = remain["predicted_energy_wh"].map(self.rls.predict_energy)
            remain["predicted_power_w"] = remain["predicted_energy_wh"] * 3600.0 / self.cfg.resample_seconds
            remain["cumulative_energy_wh"] = self.executed_energy_wh + remain["predicted_energy_wh"].cumsum()
            lower, upper = self.rls.interval(float(remain.predicted_energy_wh.sum()), self.cfg.default_confidence, len(remain))
        else:
            lower = upper = self.executed_energy_wh
        reserve = float(safety_reserve_wh if safety_reserve_wh is not None else max(1.0, 0.1 * upper))
        required = upper + float(return_energy_upper_wh) + reserve
        feasible = available_energy_wh is None or float(available_energy_wh) >= required
        summary = {"version": "4.0", "updated_window": int(window_index), "executed_energy_wh": self.executed_energy_wh,
                   "remaining_energy_lower_wh": lower, "remaining_energy_upper_wh": upper,
                   "interval_update_count": self.rls.update_count, "recent_residual_wh": self.rls.residual_ema,
                   "parameter_state": self.rls.state(), "available_energy_wh": available_energy_wh,
                   "required_energy_wh": required, "battery_feasible": bool(feasible),
                   "decision": "continue" if feasible else "return_or_charge"}
        return remain, summary
