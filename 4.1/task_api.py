# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: task_api.py
# 开发时间: 2026-09-17
# 文件名: task_api.py
# 功能说明: 提供实验4.1任务前基础动力能耗预测和飞行中剩余路线RLS修正接口
# 版本号：4.1

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data_utils import FEATURE_COLUMNS
from model import DynamicEnergyRLS
from train import predict_frame


REQUIRED_PLANNING_COLUMNS = ["task_id", *FEATURE_COLUMNS]
FORBIDDEN_TASK_COLUMNS = {
    "battery_voltage", "battery_current", "battery_current_discharge_a",
    "power_w", "base_power_w", "energy_interval_wh", "base_energy_interval_wh",
    "actual_energy_wh", "actual_speed_mps", "actual_acceleration_mps2",
    "actual_angular_rate", "actual_imu", "thermal_load_proxy",
    "vision_energy_proxy_w", "communication_energy_proxy_w",
    "velocity_x", "velocity_y", "velocity_z", "speed",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_x", "angular_y", "angular_z",
}


def validate_task_input(task: pd.DataFrame, cfg: ExperimentConfig) -> None:
    """功能: 检查任务前规划器输入是否满足4.1的物理字段与0.2秒契约。
    参数: task为一个或多个完整候选路线，cfg为实验配置对象。
    返回: None；不符合契约时抛出ValueError。
    调用位置: planning_to_model_frame、predict_task_before。
    """

    if task.empty:
        raise ValueError("任务前输入不能为空")
    missing = sorted(set(REQUIRED_PLANNING_COLUMNS) - set(task.columns))
    if missing:
        raise ValueError(f"任务前输入缺少字段: {missing}")
    present = sorted(FORBIDDEN_TASK_COLUMNS.intersection(task.columns))
    if present:
        raise ValueError(f"任务前输入禁止包含实测、标签或伪代理字段: {present}")
    extra = sorted(set(task.columns) - set(REQUIRED_PLANNING_COLUMNS))
    if extra:
        raise ValueError(f"任务前接口只接受task_id和固定10维输入，检测到额外字段: {extra}")
    numeric_columns = FEATURE_COLUMNS
    numeric = task[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"任务前数值字段含空值或非有限值: {bad}")
    if task["task_id"].isna().any():
        raise ValueError("task_id不能为空")
    if task["task_id"].astype(str).str.strip().eq("").any():
        raise ValueError("task_id不能为空字符串")
    if (numeric["payload_g"] < 0).any():
        raise ValueError("payload_g不得为负")
    for column in ("planned_motor_on", "planned_airborne"):
        values = set(numeric[column].unique().tolist())
        if not values.issubset({0.0, 1.0}):
            raise ValueError(f"{column}只能为0或1")
    if (numeric["planned_airborne"] > numeric["planned_motor_on"]).any():
        raise ValueError("必须满足planned_airborne<=planned_motor_on")
    for task_id, group in task.groupby("task_id", sort=False):
        times = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(float)
        if not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"任务{task_id}的time_s必须从0开始")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"任务{task_id}的time_s必须严格递增")
        if len(times) > 1 and not np.allclose(
            np.diff(times), cfg.resample_seconds, rtol=0.0, atol=1e-6,
        ):
            raise ValueError(f"任务{task_id}的相邻time_s必须间隔{cfg.resample_seconds:g}s")
        durations = pd.to_numeric(group["task_duration_s"], errors="coerce").to_numpy(float)
        if not np.allclose(durations, durations[0], rtol=0.0, atol=1e-6):
            raise ValueError(f"任务{task_id}的task_duration_s必须保持一致")
        if durations[0] <= 0.0 or not (times[-1] <= durations[0] <= times[-1] + cfg.resample_seconds + 1e-6):
            raise ValueError(f"任务{task_id}的task_duration_s必须覆盖完整规划时间轴")


def planning_to_model_frame(task: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 校验并按固定顺序生成当前10维TCN输入。
    参数: task为完整候选路线，cfg为实验配置对象。
    返回: 不含电池测量、真实功率和真实能耗的模型输入表。
    调用位置: predict_task_before、OnlineTaskSession。
    """

    validate_task_input(task, cfg)
    result = task.copy()
    result["task_id"] = result["task_id"].astype(str)
    result = result.sort_values(["task_id", "time_s"], kind="stable").reset_index(drop=True)
    result["flight"] = result["task_id"]
    result["route"] = "PLANNER_ROUTE"
    result["time"] = pd.to_numeric(result["time_s"], errors="raise").astype(float)
    result["dt_seconds"] = float(cfg.resample_seconds)
    result["auxiliary_power_w"] = 0.0
    missing_features = sorted(set(FEATURE_COLUMNS) - set(result.columns))
    if missing_features:
        raise ValueError(f"规划器字段无法派生以下TCN输入: {missing_features}")
    columns = ["task_id", "flight", "route", "time", "dt_seconds", "auxiliary_power_w", *FEATURE_COLUMNS]
    return result[columns].copy()


def _normal_z(confidence: float) -> float:
    """功能: 将常用置信度转换为标准正态分位数。
    参数: confidence为置信度。
    返回: 双侧区间使用的近似Z值。
    调用位置: predict_task_before。
    """

    return 1.96 if confidence >= 0.95 else 1.645


def _conditional_residual_std(model_frame: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    """功能: 按4.1任务前风场、载荷、时间和计划状态生成逐时刻残差标准差。
    参数: model_frame为按当前特征契约排列的规划特征，cfg提供不确定性模型路径。
    返回: 与规划时间步一一对应的功率标准差数组。
    调用位置: predict_task_before。
    """

    if not cfg.uncertainty_json.exists():
        raise FileNotFoundError(f"缺少任务前不确定性模型，请先完成4.1训练: {cfg.uncertainty_json}")
    model = json.loads(cfg.uncertainty_json.read_text(encoding="utf-8"))
    fallback = max(float(model.get("base_residual_std_w", 20.0)), 1e-3)
    required = {"feature_mean", "feature_std", "log_abs_error_coefficients", "sigma_scale"}
    if not required.issubset(model):
        missing = sorted(required - set(model))
        raise RuntimeError(f"任务前不确定性模型字段不完整，请重新训练: {missing}")
    design = np.column_stack([
        model_frame["wind_east_mps"].to_numpy(float),
        model_frame["wind_north_mps"].to_numpy(float),
        model_frame["payload_g"].to_numpy(float),
        model_frame["task_duration_s"].to_numpy(float),
        model_frame["time_s"].to_numpy(float),
        model_frame["planned_motor_on"].to_numpy(float),
        model_frame["planned_airborne"].to_numpy(float),
    ])
    mean = np.asarray(model["feature_mean"], dtype=float)
    std = np.asarray(model["feature_std"], dtype=float)
    coefficients = np.asarray(model["log_abs_error_coefficients"], dtype=float)
    matrix = np.column_stack([np.ones(len(design)), (design - mean) / np.maximum(std, 1e-8)])
    sigma = (np.exp(np.clip(matrix @ coefficients, -20.0, 20.0))
             * np.sqrt(np.pi / 2.0) * float(model["sigma_scale"]))
    floor = float(model.get("sigma_floor_w", fallback * 0.1))
    return np.maximum(sigma, floor)


def _task_energy_multiplier(cfg: ExperimentConfig, confidence: float) -> float:
    """功能: 读取验证集完整flight残差得到的任务级区间倍率。
    参数: cfg提供校准文件，confidence为目标置信度。
    返回: 相对于逐步条件能耗方差平方和的半径倍率。
    调用位置: _task_energy_radius、predict_task_before。
    """

    if not cfg.uncertainty_json.exists():
        raise FileNotFoundError(f"缺少任务级区间校准模型，请先完成4.1训练: {cfg.uncertainty_json}")
    model = json.loads(cfg.uncertainty_json.read_text(encoding="utf-8"))
    required = {"task_energy_radius_multiplier", "confidence"}
    if not required.issubset(model):
        missing = sorted(required - set(model))
        raise RuntimeError(f"任务级区间校准字段不完整，请重新训练: {missing}")
    multiplier = float(model["task_energy_radius_multiplier"])
    calibration_confidence = float(model["confidence"])
    multiplier *= _normal_z(confidence) / max(_normal_z(calibration_confidence), 1e-9)
    return multiplier


def _task_energy_radius(sigma_energy_wh: np.ndarray, cfg: ExperimentConfig,
                        confidence: float) -> float:
    """功能: 用验证集完整flight残差校准任务级能耗区间半径。
    参数: sigma_energy_wh为逐步条件能耗标准差，cfg提供校准文件，confidence为目标置信度。
    返回: 不把逐步固定半径线性相加的任务级半径，单位Wh。
    调用位置: predict_task_before、evaluate._conditional_task_before_interval。
    """

    base_std = float(np.sqrt(np.sum(np.asarray(sigma_energy_wh, dtype=float) ** 2)))
    return _task_energy_multiplier(cfg, confidence) * base_std


def predict_task_before(task: pd.DataFrame, cfg: ExperimentConfig, confidence: float | None = None,
                        available_energy_wh: float | None = None, return_energy_wh: float = 0.0,
                        safety_reserve_wh: float | None = None) -> tuple[pd.DataFrame, dict]:
    """功能: 输出单条候选路线的功率、能耗、任务级区间和电池决策。
    参数: task为单条完整规划路线，cfg为配置，其他参数用于置信度与电池安全判断。
    返回: 每0.2秒预测表和可供调度器读取的任务摘要。
    调用位置: main.py、OnlineTaskSession、predict_candidate_tasks。
    """

    if task["task_id"].astype(str).nunique() != 1:
        raise ValueError("predict_task_before一次只处理一个task_id；批量候选请调用predict_candidate_tasks")
    confidence = float(confidence or cfg.default_confidence)
    model_frame = planning_to_model_frame(task, cfg)
    base_power = predict_frame(model_frame, cfg)
    dt = model_frame["dt_seconds"].to_numpy(float)
    auxiliary_power = model_frame["auxiliary_power_w"].to_numpy(float)
    total_power = base_power + auxiliary_power
    base_energy = base_power * dt / 3600.0
    auxiliary_energy = auxiliary_power * dt / 3600.0
    total_energy = base_energy + auxiliary_energy
    sigma_power = _conditional_residual_std(model_frame, cfg)
    sigma_energy = sigma_power * dt / 3600.0
    z = _normal_z(confidence)
    step_radius = z * sigma_energy
    task_multiplier = _task_energy_multiplier(cfg, confidence)
    cumulative_radius = task_multiplier * np.sqrt(np.cumsum(sigma_energy ** 2))
    output = task.copy().sort_values("time_s", kind="stable").reset_index(drop=True)
    output["dt_seconds"] = float(cfg.resample_seconds)
    output["predicted_base_power_w"] = base_power
    output["auxiliary_power_w"] = auxiliary_power
    output["predicted_power_w"] = total_power
    output["predicted_base_energy_wh"] = base_energy
    output["auxiliary_energy_wh"] = auxiliary_energy
    output["predicted_energy_wh"] = total_energy
    output["predicted_energy_lower_wh"] = np.maximum(base_energy - step_radius, 0.0) + auxiliary_energy
    output["predicted_energy_upper_wh"] = base_energy + step_radius + auxiliary_energy
    output["cumulative_energy_wh"] = np.cumsum(total_energy)
    output["cumulative_energy_lower_wh"] = np.maximum(np.cumsum(base_energy) - cumulative_radius, 0.0) + np.cumsum(auxiliary_energy)
    output["cumulative_energy_upper_wh"] = np.cumsum(base_energy) + cumulative_radius + np.cumsum(auxiliary_energy)
    total = float(total_energy.sum())
    lower = float(output["cumulative_energy_lower_wh"].iloc[-1])
    upper = float(output["cumulative_energy_upper_wh"].iloc[-1])
    reserve = float(safety_reserve_wh if safety_reserve_wh is not None else max(0.1 * upper, 1.0))
    required = upper + float(return_energy_wh) + reserve
    feasible = available_energy_wh is None or float(available_energy_wh) >= required
    summary = {
        "version": "4.1", "task_id": str(task["task_id"].iloc[0]), "step_count": int(len(output)),
        "predicted_base_energy_wh": float(base_energy.sum()),
        "known_auxiliary_energy_wh": float(auxiliary_energy.sum()),
        "total_energy_wh": total, "total_energy_lower_wh": lower, "total_energy_upper_wh": upper,
        "confidence": confidence,
        "confidence_interval_method": "条件残差方差合成并按验证集完整flight残差进行共形校准，不对固定半径线性累加",
        "available_energy_wh": available_energy_wh, "return_energy_upper_wh": float(return_energy_wh),
        "safety_reserve_wh": reserve, "required_energy_wh": required,
        "battery_feasible": bool(feasible), "charge_or_replace_required": bool(not feasible),
        "decision": "continue" if feasible else "charge_or_replace",
        "model_input_columns": FEATURE_COLUMNS,
    }
    return output, summary


def predict_candidate_tasks(tasks: pd.DataFrame, cfg: ExperimentConfig, **kwargs) -> tuple[pd.DataFrame, list[dict]]:
    """功能: 批量预测多个候选任务，供调度器比较任务分配和充电时机。
    参数: tasks为包含多个task_id的规划表，cfg为配置，其余参数透传给单任务接口。
    返回: 合并后的逐步预测表和每个候选任务摘要列表。
    调用位置: 上层任务分配器。
    """

    validate_task_input(tasks, cfg)
    outputs, summaries = [], []
    for _, group in tasks.groupby("task_id", sort=False):
        output, summary = predict_task_before(group.copy(), cfg, **kwargs)
        outputs.append(output)
        summaries.append(summary)
    return pd.concat(outputs, ignore_index=True), summaries


def _selected_rls_configuration(cfg: ExperimentConfig) -> tuple[float, float, float, float, float]:
    """功能: 读取验证集选择出的RLS参数和交叉拟合在线区间倍率。
    参数: cfg为实验配置对象。
    返回: 遗忘因子、初始协方差、剩余任务倍率、窗口倍率与初始窗口残差标准差。
    调用位置: OnlineTaskSession.__init__。
    """

    if not cfg.rls_tuning_csv.exists():
        raise FileNotFoundError(f"缺少RLS参数搜索结果，请先运行tune-rls: {cfg.rls_tuning_csv}")
    tuning = pd.read_csv(cfg.rls_tuning_csv)
    if tuning.empty:
        raise RuntimeError("RLS参数搜索结果为空，请重新运行tune-rls")
    if not cfg.rls_interval_calibration_json.exists():
        raise FileNotFoundError(
            f"缺少RLS在线区间校准文件，请先运行tune-rls: {cfg.rls_interval_calibration_json}"
        )
    calibration = json.loads(cfg.rls_interval_calibration_json.read_text(encoding="utf-8"))
    interval_scale = float(calibration.get("interval_scale", np.nan))
    window_interval_scale = float(calibration.get("window_interval_scale", np.nan))
    initial_residual_std_wh = float(calibration.get("initial_residual_std_wh", np.nan))
    if not np.isfinite(interval_scale) or interval_scale < 1.0:
        raise RuntimeError("RLS在线区间校准倍率无效，请重新运行tune-rls")
    if not np.isfinite(initial_residual_std_wh) or initial_residual_std_wh <= 0.0:
        raise RuntimeError("RLS初始窗口残差标准差无效，请重新运行tune-rls")
    if not np.isfinite(window_interval_scale) or window_interval_scale < 1.0:
        raise RuntimeError("RLS窗口区间校准倍率无效，请重新运行tune-rls")
    best = tuning.sort_values("flight_energy_wape_percent").iloc[0]
    return (
        float(best["forgetting_factor"]),
        float(best["initial_covariance"]),
        interval_scale,
        window_interval_scale,
        initial_residual_std_wh,
    )


class OnlineTaskSession:
    """功能: 仅在完整窗口结束后更新RLS，并重算当前窗口之后的全部剩余路线。
    参数: task为单条任务规划表，cfg为4.1配置，rls_params为可选RLS参数。
    返回: update后输出剩余路线、动态区间和调度决策。
    调用位置: main.py的online-demo及在线飞行控制器。
    """

    def __init__(self, task: pd.DataFrame, cfg: ExperimentConfig,
                 rls_params: tuple[float, float] | None = None) -> None:
        self.task = task.sort_values("time_s", kind="stable").reset_index(drop=True)
        self.cfg = cfg
        self.base, self.summary = predict_task_before(self.task, cfg)
        self.model_frame = planning_to_model_frame(self.task, cfg)
        self.window_steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
        (selected_factor, selected_covariance, interval_scale, window_interval_scale,
         initial_residual_std_wh) = _selected_rls_configuration(cfg)
        factor, covariance = rls_params or (selected_factor, selected_covariance)
        self.rls = DynamicEnergyRLS(
            factor, covariance, interval_scale, initial_residual_std_wh,
            window_interval_scale,
        )
        self.executed_energy_wh = 0.0
        self.last_window_index = -1
        self.executed_end_row = 0

    def _correct_remaining(self, start_row: int) -> tuple[pd.DataFrame, float, float, float]:
        remain = self.base.iloc[start_row:].copy().reset_index(drop=True)
        if remain.empty:
            return remain, 0.0, 0.0, 0.0
        corrected_base = np.zeros(len(remain), dtype=float)
        lower_steps = np.zeros(len(remain), dtype=float)
        upper_steps = np.zeros(len(remain), dtype=float)
        for start in range(0, len(remain), self.window_steps):
            positions = np.arange(start, min(start + self.window_steps, len(remain)))
            baseline_steps = remain.loc[positions, "predicted_base_energy_wh"].to_numpy(float)
            baseline_window = float(baseline_steps.sum())
            window_fraction = len(positions) / self.window_steps
            corrected_window = self.rls.predict_energy(baseline_window, window_fraction)
            window_lower, window_upper = self.rls.interval(
                baseline_window, self.cfg.default_confidence, window_fraction,
                scale=self.rls.window_interval_scale,
            )
            weights = baseline_steps / baseline_window if baseline_window > 1e-12 else np.full(len(positions), 1.0 / len(positions))
            corrected_base[positions] = corrected_window * weights
            lower_steps[positions] = window_lower * weights
            upper_steps[positions] = window_upper * weights
        auxiliary = remain["auxiliary_energy_wh"].to_numpy(float)
        baseline_cumulative = remain["predicted_base_energy_wh"].to_numpy(float).cumsum()
        baseline_total = float(baseline_cumulative[-1])
        horizon_windows = len(remain) / self.window_steps
        corrected_total = self.rls.predict_total_energy(baseline_total, horizon_windows)
        distributed_total = float(corrected_base.sum())
        if distributed_total > 1e-12:
            corrected_base *= corrected_total / distributed_total
        elif corrected_total > 0.0:
            corrected_base[:] = corrected_total / len(corrected_base)
        remain["predicted_base_energy_wh"] = corrected_base
        remain["predicted_energy_wh"] = corrected_base + auxiliary
        remain["predicted_power_w"] = remain["predicted_energy_wh"] * 3600.0 / remain["dt_seconds"].to_numpy(float)
        remain["predicted_energy_lower_wh"] = lower_steps + auxiliary
        remain["predicted_energy_upper_wh"] = upper_steps + auxiliary
        remain["cumulative_energy_wh"] = self.executed_energy_wh + remain["predicted_energy_wh"].cumsum()
        auxiliary_cumulative = auxiliary.cumsum()
        cumulative_intervals = np.asarray([
            self.rls.interval(
                base_wh, self.cfg.default_confidence, row_count / self.window_steps,
            )
            for row_count, base_wh in enumerate(baseline_cumulative, start=1)
        ])
        remain["cumulative_energy_lower_wh"] = (
            self.executed_energy_wh + cumulative_intervals[:, 0] + auxiliary_cumulative
        )
        remain["cumulative_energy_upper_wh"] = (
            self.executed_energy_wh + cumulative_intervals[:, 1] + auxiliary_cumulative
        )
        auxiliary_total = float(self.base.iloc[start_row:]["auxiliary_energy_wh"].sum())
        remaining_total = corrected_total + auxiliary_total
        lower, upper = self.rls.interval(baseline_total, self.cfg.default_confidence, horizon_windows)
        return remain, remaining_total, lower + auxiliary_total, upper + auxiliary_total

    def update(self, window_index: int, actual_energy_wh: float,
               predicted_energy_wh: float | None = None, available_energy_wh: float | None = None,
               return_energy_upper_wh: float = 0.0,
               safety_reserve_wh: float | None = None) -> tuple[pd.DataFrame, dict]:
        """功能: 用刚结束窗口的预测—实际能耗配对更新RLS并重算全部未来窗口。
        参数: window_index为窗口编号，actual_energy_wh为外部计量值，predicted_energy_wh可用于校验任务前基线，其余参数用于电池决策。
        返回: 只含未执行时间步的更新预测表和在线决策摘要。
        调用位置: 在线飞行控制器。
        """

        expected = self.last_window_index + 1
        if int(window_index) != expected:
            raise ValueError(f"RLS窗口必须连续更新；当前应输入窗口{expected}，实际收到{window_index}")
        if not np.isfinite(actual_energy_wh) or float(actual_energy_wh) < 0:
            raise ValueError("actual_energy_wh必须是窗口结束后获得的非负有限值")
        start = int(window_index) * self.window_steps
        end = min(len(self.base), start + self.window_steps)
        if start >= len(self.base):
            raise IndexError("window_index超过任务规划范围")
        if end - start != self.window_steps:
            raise ValueError(
                f"窗口{window_index}仅包含{end - start}个离散步，不足完整的"
                f"{self.window_steps}步RLS窗口；截断尾窗只能计入任务能耗，不能更新RLS"
            )
        baseline_total_window = float(self.base.iloc[start:end]["predicted_energy_wh"].sum())
        auxiliary_window = float(self.base.iloc[start:end]["auxiliary_energy_wh"].sum())
        baseline_base_window = float(self.base.iloc[start:end]["predicted_base_energy_wh"].sum())
        if predicted_energy_wh is not None and not np.isclose(float(predicted_energy_wh), baseline_total_window, rtol=1e-5, atol=1e-8):
            raise ValueError("传入的窗口预测能耗与任务前固定基线不一致")
        future_base_before = float(self.base.iloc[end:]["predicted_base_energy_wh"].sum())
        future_auxiliary = float(self.base.iloc[end:]["auxiliary_energy_wh"].sum())
        future_windows = max(len(self.base) - end, 0) / self.window_steps
        before_remaining = self.rls.predict_total_energy(future_base_before, future_windows) + future_auxiliary
        before_lower, before_upper = self.rls.interval(future_base_before, self.cfg.default_confidence, future_windows)
        actual_base_window = max(float(actual_energy_wh) - auxiliary_window, 0.0)
        state = self.rls.update_window(baseline_base_window, actual_base_window)
        self.executed_energy_wh += float(actual_energy_wh)
        self.last_window_index = int(window_index)
        self.executed_end_row = end
        remain, remaining_total, remaining_lower, remaining_upper = self._correct_remaining(end)
        predicted_final = self.executed_energy_wh + remaining_total
        final_lower = self.executed_energy_wh + remaining_lower
        final_upper = self.executed_energy_wh + remaining_upper
        reserve = float(safety_reserve_wh if safety_reserve_wh is not None else max(1.0, 0.1 * remaining_upper))
        required = remaining_upper + float(return_energy_upper_wh) + reserve
        feasible = available_energy_wh is None or float(available_energy_wh) >= required
        deviation_wh = float(actual_energy_wh) - baseline_total_window
        relative_deviation = abs(deviation_wh) / max(abs(baseline_total_window), 1e-9)
        replan_required = relative_deviation >= self.cfg.online_replan_relative_deviation
        degrade_required = relative_deviation >= self.cfg.online_degrade_relative_deviation
        pause_required = relative_deviation >= self.cfg.online_pause_relative_deviation
        if not feasible:
            decision = "return_or_charge"
        elif pause_required:
            decision = "pause"
        elif degrade_required:
            decision = "degrade"
        elif replan_required:
            decision = "replan"
        else:
            decision = "continue"
        summary = {
            "version": "4.1", "updated_window": int(window_index), "executed_end_row": int(end),
            "window_baseline_energy_wh": baseline_total_window, "window_actual_energy_wh": float(actual_energy_wh),
            "window_actual_base_energy_wh": actual_base_window, "executed_energy_wh": self.executed_energy_wh,
            "remaining_energy_wh": remaining_total, "remaining_energy_lower_wh": remaining_lower,
            "remaining_energy_upper_wh": remaining_upper, "predicted_final_energy_wh": predicted_final,
            "predicted_final_lower_wh": final_lower, "predicted_final_upper_wh": final_upper,
            "confidence": self.cfg.default_confidence, "interval_width_wh": remaining_upper - remaining_lower,
            "interval_update_count": self.rls.update_count, "recent_residual_wh": self.rls.last_residual_wh,
            "last_observation_impact": {
                "remaining_prediction_change_wh": remaining_total - before_remaining,
                "remaining_interval_width_before_wh": before_upper - before_lower,
                "remaining_interval_width_after_wh": remaining_upper - remaining_lower,
            },
            "parameter_state": state, "available_energy_wh": available_energy_wh,
            "return_energy_upper_wh": float(return_energy_upper_wh), "safety_reserve_wh": reserve,
            "required_energy_wh": required, "battery_feasible": bool(feasible),
            "window_energy_deviation_wh": deviation_wh,
            "window_energy_relative_deviation_percent": relative_deviation * 100.0,
            "decision_thresholds": {
                "replan_relative_deviation": self.cfg.online_replan_relative_deviation,
                "degrade_relative_deviation": self.cfg.online_degrade_relative_deviation,
                "pause_relative_deviation": self.cfg.online_pause_relative_deviation,
            },
            "replan_required": bool(replan_required),
            "degraded_execution_recommended": bool(degrade_required),
            "pause_required": bool(pause_required),
            "return_or_charge_required": bool(not feasible),
            "decision": decision,
        }
        return remain, summary
