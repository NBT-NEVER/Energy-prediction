# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: evaluate.py
# 开发时间: 2026-09-16
# 功能说明: 评估实验4.1规划TCN、窗口级动态RLS和未来任务能耗区间
# 版本号：4.1

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories
from data_utils import (
    UNSEEN_DURATION_HOLDOUT_S,
    UNSEEN_PAYLOAD_HOLDOUT_G,
    UNSEEN_ROUTE_HOLDOUTS,
    UNSEEN_WIND_HOLDOUT_MPS,
)
from model import DynamicEnergyRLS
from task_api import _conditional_residual_std, _task_energy_multiplier, _task_energy_radius
from train import TARGET_COLUMN, UNCERTAINTY_FEATURES, predict_frame


FLIGHT_PROFILE_COLUMNS = [
    "flight", "route", "wind_east_mps", "wind_north_mps", "payload_g", "task_duration_s",
]
CORRECTION_TRACE_COLUMNS = [
    "flight", "route", "observed_window_count", "observed_actual_energy_wh",
    "window_steps", "expected_window_steps", "window_duration_s",
    "is_complete_window", "rls_update_applied",
    "window_baseline_energy_wh", "window_actual_energy_wh", "remaining_baseline_energy_wh",
    "remaining_window_count", "corrected_remaining_energy_wh", "remaining_lower_wh",
    "remaining_upper_wh", "predicted_final_energy_wh", "predicted_final_lower_wh",
    "predicted_final_upper_wh", "actual_final_energy_wh", "absolute_error_wh", "interval_width_wh",
    "equivalent_one_window_interval_width_wh", "interval_width_before_update_wh",
    "remaining_prediction_before_update_wh", "baseline_final_energy_wh",
    "remaining_prediction_change_wh", "theta_bias_wh_per_window", "theta_scale",
    "covariance_trace", "recent_residual_wh",
]


def _metrics(actual: np.ndarray, predicted: np.ndarray, prefix: str) -> dict:
    """功能: 计算平均绝对误差、均方根误差、WAPE和低估率。
    参数: actual为真实值，predicted为预测值，prefix为指标名称前缀。
    返回: 指标字典。
    调用位置: evaluate_model、_simulate。
    """

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


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """功能: 计算决定系数R2，并对常量真实序列返回0。
    参数: actual为真实值，predicted为预测值。
    返回: R2浮点数。
    调用位置: _save_rls_window_comparison。
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual):
        return 0.0
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """功能: 按flight和实际时间稳定排序评估数据。
    参数: frame为待评估表。
    返回: 重排后的副本。
    调用位置: evaluate_model、_simulate。
    """

    columns = [c for c in ("flight", "time", "route") if c in frame.columns]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True).copy()


def _window_positions(length: int, steps: int) -> list[np.ndarray]:
    """功能: 将单次flight的行位置切分为连续RLS窗口。
    参数: length为flight行数，steps为每窗步数。
    返回: 每个窗口的局部行位置数组。
    调用位置: _simulate。
    """

    return [np.arange(start, min(start + steps, length)) for start in range(0, length, steps)]


def _window_energy_summary(predictions: pd.DataFrame, steps: int,
                           resample_seconds: float) -> pd.DataFrame:
    """功能: 汇总固定RLS窗口能耗，并显式标记尾部截断窗口。
    参数: predictions为逐时间步预测表，steps为期望窗口步数，resample_seconds为离散步长。
    返回: 含窗口长度、完整性、区间和误差的逐窗口表。
    调用位置: evaluate_model、_save_rls_window_comparison。
    """

    if predictions.empty:
        return pd.DataFrame(columns=[
            "flight", "window_index", "route", "window_steps", "window_duration_s",
            "expected_window_steps", "expected_window_duration_s", "is_complete", "is_complete_window",
            "actual_energy_wh", "predicted_energy_wh", "tcn_predicted_energy_wh",
            "predicted_energy_lower_wh", "predicted_energy_upper_wh", "energy_error_wh",
            "energy_abs_error_wh", "energy_abs_percent_error",
        ])
    table = predictions.assign(
        window_index=predictions.groupby("flight", sort=False).cumcount() // steps,
    ).groupby(["flight", "window_index"], sort=True).agg(
        route=("route", "first"),
        window_steps=("dt_seconds", "size"),
        window_duration_s=("dt_seconds", "sum"),
        actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"),
        tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"),
        predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    table["expected_window_steps"] = int(steps)
    table["expected_window_duration_s"] = float(steps * resample_seconds)
    table["is_complete_window"] = (
        table["window_steps"].eq(steps)
        & np.isclose(table["window_duration_s"], table["expected_window_duration_s"], rtol=0.0, atol=1e-9)
    )
    table["is_complete"] = table["is_complete_window"]
    table["energy_error_wh"] = table["predicted_energy_wh"] - table["actual_energy_wh"]
    table["energy_abs_error_wh"] = table["energy_error_wh"].abs()
    table["energy_abs_percent_error"] = (
        table["energy_abs_error_wh"]
        / table["actual_energy_wh"].abs().clip(lower=1e-9)
        * 100.0
    )
    ordered_columns = [
        "flight", "window_index", "route", "window_steps", "window_duration_s",
        "expected_window_steps", "expected_window_duration_s", "is_complete", "is_complete_window",
        "actual_energy_wh", "predicted_energy_wh", "tcn_predicted_energy_wh",
        "predicted_energy_lower_wh", "predicted_energy_upper_wh", "energy_error_wh",
        "energy_abs_error_wh", "energy_abs_percent_error",
    ]
    return table[ordered_columns]


def _equivalent_one_window_width(rls: DynamicEnergyRLS, remaining_base_wh: float,
                                 remaining_windows: float, confidence: float) -> float:
    """功能: 计算当前RLS状态下等平均基线能耗的单窗口区间宽度。
    参数: rls为当前状态，remaining_base_wh为剩余基线能耗，remaining_windows为剩余窗口数，confidence为置信度。
    返回: 等效单窗口区间宽度，单位Wh。
    调用位置: _simulate。
    """

    if remaining_windows <= 0:
        return 0.0
    average_base_wh = remaining_base_wh / remaining_windows
    lower, upper = rls.interval(average_base_wh, confidence, 1)
    return float(upper - lower)


def _conditional_task_before_interval(frame: pd.DataFrame, base_total_wh: float,
                                      horizon_windows: float, cfg: ExperimentConfig) -> tuple[float, float, float]:
    """功能: 按逐时刻条件残差方差合成任务前基础动力能耗区间。
    参数: frame为任务规划输入，base_total_wh为TCN基础动力总能耗，horizon_windows为RLS等效窗口数，cfg为配置。
    返回: 基础动力能耗下界、上界和等效单窗口区间宽度。
    调用位置: _simulate、evaluate_model。
    """

    sigma_power = _conditional_residual_std(frame, cfg)
    sigma_energy = sigma_power * frame["dt_seconds"].to_numpy(float) / 3600.0
    energy_variance = float(np.sum(sigma_energy ** 2))
    radius = _task_energy_radius(sigma_energy, cfg, cfg.default_confidence)
    lower = max(base_total_wh - radius, 0.0)
    upper = base_total_wh + radius
    if horizon_windows <= 0:
        return lower, upper, 0.0
    average_base = base_total_wh / horizon_windows
    one_window_radius = float(
        _task_energy_multiplier(cfg, cfg.default_confidence)
        * np.sqrt(energy_variance / horizon_windows)
    )
    one_window_width = (
        average_base + one_window_radius - max(average_base - one_window_radius, 0.0)
    )
    return lower, upper, float(one_window_width)


def _validate_interval_calibration(cfg: ExperimentConfig) -> dict:
    """功能: 校验离线评估使用的条件残差模型，禁止静默使用固定标准差回退。
    参数: cfg为实验配置对象。
    返回: 可并入总体指标的校准来源和样本量。
    调用位置: evaluate_model。
    """

    if not cfg.uncertainty_json.exists():
        raise FileNotFoundError(f"缺少条件残差模型，无法生成正式flight区间: {cfg.uncertainty_json}")
    model = json.loads(cfg.uncertainty_json.read_text(encoding="utf-8"))
    required = {
        "target_column", "condition_features", "feature_mean", "feature_std",
        "log_abs_error_coefficients", "sigma_scale", "sigma_floor_w", "calibration_rows",
    }
    missing = sorted(required - set(model))
    if missing:
        raise RuntimeError(f"条件残差模型字段不完整，请重新训练4.1模型: {missing}")
    if str(model["target_column"]) != TARGET_COLUMN:
        raise RuntimeError(f"条件残差模型目标必须为{TARGET_COLUMN}")
    if list(model["condition_features"]) != UNCERTAINTY_FEATURES:
        raise RuntimeError("条件残差模型特征顺序与当前4.1实现不一致，请重新训练")
    feature_count = len(UNCERTAINTY_FEATURES)
    if (len(model["feature_mean"]) != feature_count
            or len(model["feature_std"]) != feature_count
            or len(model["log_abs_error_coefficients"]) != feature_count + 1):
        raise RuntimeError("条件残差模型参数维度与当前4.1实现不一致，请重新训练")
    numeric_parameters = np.asarray(
        [*model["feature_mean"], *model["feature_std"],
         *model["log_abs_error_coefficients"], model["sigma_scale"], model["sigma_floor_w"]],
        dtype=float,
    )
    if not np.isfinite(numeric_parameters).all():
        raise RuntimeError("条件残差模型含非有限参数，请重新训练")
    if ((np.asarray(model["feature_std"], dtype=float) <= 0.0).any()
            or float(model["sigma_scale"]) <= 0.0 or float(model["sigma_floor_w"]) <= 0.0):
        raise RuntimeError("条件残差模型标准差或缩放参数无效，请重新训练")
    calibration_rows = int(model["calibration_rows"])
    if calibration_rows <= 0:
        raise RuntimeError("条件残差模型没有有效校准样本")
    return {
        "interval_calibration_file": str(cfg.uncertainty_json),
        "interval_calibration_rows": calibration_rows,
        "interval_calibration_fallback_used": False,
    }


def _load_rls_interval_calibration(cfg: ExperimentConfig) -> tuple[float, float, float, dict]:
    """功能: 读取只由验证flight交叉拟合得到的在线RLS区间倍率。
    参数: cfg为实验配置对象。
    返回: 区间倍率和可写入正式评估的校准摘要。
    调用位置: evaluate_model。
    """

    path = cfg.rls_interval_calibration_json
    if not path.exists():
        raise FileNotFoundError(f"缺少RLS在线区间校准文件，请先运行tune-rls: {path}")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "method", "confidence", "validation_flights", "cross_fit_folds",
        "calibration_points", "conformal_rank", "interval_scale",
        "window_interval_scale",
        "initial_residual_std_wh",
        "test_set_used_for_selection_or_calibration",
    }
    missing = sorted(required - set(calibration))
    if missing:
        raise RuntimeError(f"RLS在线区间校准文件字段不完整: {missing}")
    interval_scale = float(calibration["interval_scale"])
    window_interval_scale = float(calibration["window_interval_scale"])
    if not np.isfinite(interval_scale) or interval_scale < 1.0:
        raise RuntimeError("RLS在线区间校准倍率必须是不小于1的有限数")
    if not np.isfinite(window_interval_scale) or window_interval_scale < 1.0:
        raise RuntimeError("RLS窗口区间校准倍率必须是不小于1的有限数")
    initial_residual_std_wh = float(calibration["initial_residual_std_wh"])
    if not np.isfinite(initial_residual_std_wh) or initial_residual_std_wh <= 0.0:
        raise RuntimeError("RLS初始窗口残差标准差必须是正有限数")
    if bool(calibration["test_set_used_for_selection_or_calibration"]):
        raise RuntimeError("RLS区间校准错误使用了测试集，正式评估已拒绝继续")
    return interval_scale, window_interval_scale, initial_residual_std_wh, {
        "rls_interval_calibration_file": str(path),
        "rls_interval_calibration_method": str(calibration["method"]),
        "rls_interval_calibration_validation_flights": int(calibration["validation_flights"]),
        "rls_interval_calibration_folds": int(calibration["cross_fit_folds"]),
        "rls_interval_calibration_points": int(calibration["calibration_points"]),
        "rls_interval_scale": interval_scale,
        "rls_window_interval_scale": window_interval_scale,
        "rls_initial_residual_std_wh": initial_residual_std_wh,
        "rls_interval_calibration_test_used": False,
    }


def _flight_input_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """功能: 仅用任务前可知输入构建flight级分层画像。
    参数: frame为训练集或测试集逐时刻输入表。
    返回: 含路线、平均风速、载荷和任务长度的flight级画像。
    调用位置: evaluate_model。
    """

    missing = [column for column in FLIGHT_PROFILE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"分层评估缺少任务输入字段: {missing}")
    source = frame[FLIGHT_PROFILE_COLUMNS].copy()
    source["wind_speed_mps"] = np.hypot(source["wind_east_mps"], source["wind_north_mps"])
    profiles = source.groupby("flight", sort=True).agg(
        route=("route", "first"),
        mean_wind_speed_mps=("wind_speed_mps", "mean"),
        mean_wind_east_mps=("wind_east_mps", "mean"),
        mean_wind_north_mps=("wind_north_mps", "mean"),
        payload_g=("payload_g", "first"),
        task_duration_s=("task_duration_s", "first"),
    ).reset_index()
    profiles["mean_wind_direction_deg"] = np.mod(
        np.degrees(np.arctan2(profiles["mean_wind_east_mps"], profiles["mean_wind_north_mps"])), 360.0,
    )
    return profiles


def _flight_group_metrics(group: pd.DataFrame) -> dict:
    """功能: 计算一个flight分层的最终能耗误差和区间质量。
    参数: group为已按输入条件选出的flight级预测表。
    返回: WAPE、MAE、低估率、区间覆盖率和平均区间宽度。
    调用位置: _save_stratified_evaluation。
    """

    if group.empty:
        return {
            "test_flights": 0, "actual_energy_sum_wh": 0.0, "predicted_energy_sum_wh": 0.0,
            "task_before_predicted_energy_sum_wh": 0.0,
            "wape_percent": np.nan, "mae_wh": np.nan, "underestimate_rate_percent": np.nan,
            "online_rls_wape_percent": np.nan, "online_rls_mae_wh": np.nan,
            "online_rls_underestimate_rate_percent": np.nan,
            "task_before_tcn_wape_percent": np.nan, "task_before_tcn_mae_wh": np.nan,
            "task_before_tcn_underestimate_rate_percent": np.nan,
            "interval_coverage_percent": np.nan, "mean_interval_width_wh": np.nan,
        }
    actual = group["actual_energy_wh"].to_numpy(float)
    predicted = group["predicted_energy_wh"].to_numpy(float)
    task_before = group["tcn_predicted_energy_wh"].to_numpy(float)
    lower = group["predicted_energy_lower_wh"].to_numpy(float)
    upper = group["predicted_energy_upper_wh"].to_numpy(float)
    error = predicted - actual
    task_before_error = task_before - actual
    denominator = max(np.abs(actual).sum(), 1e-9)
    online_wape = float(np.abs(error).sum() / denominator * 100.0)
    online_mae = float(np.abs(error).mean())
    online_underestimate = float(np.mean(predicted < actual) * 100.0)
    return {
        "test_flights": int(len(group)),
        "actual_energy_sum_wh": float(actual.sum()),
        "predicted_energy_sum_wh": float(predicted.sum()),
        "task_before_predicted_energy_sum_wh": float(group["tcn_predicted_energy_wh"].sum()),
        "wape_percent": online_wape,
        "mae_wh": online_mae,
        "underestimate_rate_percent": online_underestimate,
        "online_rls_wape_percent": online_wape,
        "online_rls_mae_wh": online_mae,
        "online_rls_underestimate_rate_percent": online_underestimate,
        "task_before_tcn_wape_percent": float(np.abs(task_before_error).sum() / denominator * 100.0),
        "task_before_tcn_mae_wh": float(np.abs(task_before_error).mean()),
        "task_before_tcn_underestimate_rate_percent": float(np.mean(task_before < actual) * 100.0),
        "interval_coverage_percent": float(np.mean((actual >= lower) & (actual <= upper)) * 100.0),
        "mean_interval_width_wh": float(np.mean(upper - lower)),
    }


def _save_stratified_evaluation(cfg: ExperimentConfig, train_profiles: pd.DataFrame,
                                test_profiles: pd.DataFrame, predictions: pd.DataFrame,
                                flight: pd.DataFrame, window: pd.DataFrame) -> dict:
    """功能: 保存留出条件对照和输入分箱下的flight级真实评估。
    参数: cfg为配置，train_profiles/test_profiles为纯输入画像，其余表为最终逐时刻、flight和窗口汇总。
    返回: 需要并入总体评估的留出条件关键指标。
    调用位置: evaluate_model。
    """

    rows: list[dict] = []
    sample_counts = predictions.groupby("flight", sort=False).size()
    window_counts = window.groupby("flight", sort=False).size()

    def append_group(stratification: str, stratum: str, rule: str,
                     test_mask: pd.Series, train_mask: pd.Series) -> None:
        """功能: 将一个由任务输入冻结的flight分层及其指标加入输出表。
        参数: stratification/stratum/rule描述分层，test_mask/train_mask选择对应flight。
        返回: None。
        调用位置: _save_stratified_evaluation内部。
        """

        test_ids = set(test_profiles.loc[test_mask, "flight"].tolist())
        selected = flight[flight["flight"].isin(test_ids)]
        rows.append({
            "stratification": stratification,
            "stratum": stratum,
            "selection_rule": rule,
            "selection_inputs": "route|mean_wind_speed_mps|wind_east_mps|wind_north_mps|payload_g|task_duration_s",
            "selection_uses_power_label": False,
            "point_prediction_method": "online_window_rls_and_task_before_tcn",
            "interval_method": "task_before_conditional_residual_variance_sum",
            "confidence_level": cfg.default_confidence,
            "interval_calibration_file": str(cfg.uncertainty_json),
            "interval_calibration_fallback_used": False,
            "train_flights": int(train_mask.sum()),
            "test_rows": int(sample_counts.reindex(list(test_ids), fill_value=0).sum()),
            "test_windows": int(window_counts.reindex(list(test_ids), fill_value=0).sum()),
            **_flight_group_metrics(selected),
        })

    test_route_unseen = test_profiles["route"].isin(UNSEEN_ROUTE_HOLDOUTS)
    train_route_unseen = train_profiles["route"].isin(UNSEEN_ROUTE_HOLDOUTS)
    test_wind_unseen = test_profiles["mean_wind_speed_mps"].ge(UNSEEN_WIND_HOLDOUT_MPS)
    train_wind_unseen = train_profiles["mean_wind_speed_mps"].ge(UNSEEN_WIND_HOLDOUT_MPS)
    test_payload_unseen = test_profiles["payload_g"].ge(UNSEEN_PAYLOAD_HOLDOUT_G)
    train_payload_unseen = train_profiles["payload_g"].ge(UNSEEN_PAYLOAD_HOLDOUT_G)
    test_duration_unseen = test_profiles["task_duration_s"].gt(UNSEEN_DURATION_HOLDOUT_S)
    train_duration_unseen = train_profiles["task_duration_s"].gt(UNSEEN_DURATION_HOLDOUT_S)
    binary_specs = [
        (
            "route_holdout",
            test_route_unseen,
            train_route_unseen,
            f"route in {sorted(UNSEEN_ROUTE_HOLDOUTS)}",
        ),
        (
            "mean_wind_holdout",
            test_wind_unseen,
            train_wind_unseen,
            f"mean_wind_speed_mps >= {UNSEEN_WIND_HOLDOUT_MPS:g}",
        ),
        (
            "payload_holdout",
            test_payload_unseen,
            train_payload_unseen,
            f"payload_g >= {UNSEEN_PAYLOAD_HOLDOUT_G:g}",
        ),
        (
            "duration_holdout",
            test_duration_unseen,
            train_duration_unseen,
            f"task_duration_s > {UNSEEN_DURATION_HOLDOUT_S:g}",
        ),
        (
            "any_holdout",
            test_route_unseen | test_wind_unseen | test_payload_unseen | test_duration_unseen,
            train_route_unseen | train_wind_unseen | train_payload_unseen | train_duration_unseen,
            "any route/wind/payload/duration holdout condition",
        ),
    ]
    for stratification, test_unseen, train_unseen, rule in binary_specs:
        append_group(stratification, "seen", f"not ({rule})", ~test_unseen, ~train_unseen)
        append_group(stratification, "unseen", rule, test_unseen, train_unseen)

    bin_specs = [
        ("mean_wind_speed_bin", "mean_wind_speed_mps", [-np.inf, 2.0, 4.0, 6.0, np.inf],
         ["<2", "2-<4", "4-<6", ">=6"], False),
        ("payload_bin", "payload_g", [-np.inf, 250.0, 500.0, 750.0, np.inf],
         ["<250", "250-<500", "500-<750", ">=750"], False),
        ("task_duration_bin", "task_duration_s", [-np.inf, 150.0, 200.0, 250.0, 300.0, np.inf],
         ["<=150", ">150-200", ">200-250", ">250-300", ">300"], True),
    ]
    for stratification, column, bins, labels, right in bin_specs:
        test_bins = pd.cut(test_profiles[column], bins=bins, labels=labels, right=right, include_lowest=True)
        train_bins = pd.cut(train_profiles[column], bins=bins, labels=labels, right=right, include_lowest=True)
        for label in labels:
            append_group(stratification, label, f"{column} bin {label}", test_bins == label, train_bins == label)

    direction_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    test_direction = np.floor(
        np.mod(test_profiles["mean_wind_direction_deg"] + 22.5, 360.0) / 45.0,
    ).astype(int)
    train_direction = np.floor(
        np.mod(train_profiles["mean_wind_direction_deg"] + 22.5, 360.0) / 45.0,
    ).astype(int)
    for index, label in enumerate(direction_labels):
        append_group(
            "mean_wind_direction_bin", label,
            f"circular_mean_wind_direction in {label} sector",
            test_direction == index, train_direction == index,
        )

    result = pd.DataFrame(rows)
    holdout_names = {spec[0] for spec in binary_specs}
    generalization = result[result["stratification"].isin(holdout_names)].reset_index(drop=True)
    interval_stratified = result[~result["stratification"].isin(holdout_names)].reset_index(drop=True)
    cfg.generalization_evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
    generalization.to_csv(cfg.generalization_evaluation_csv, index=False, encoding="utf-8")
    interval_stratified.to_csv(cfg.interval_stratified_csv, index=False, encoding="utf-8")
    summary: dict[str, object] = {
        "generalization_evaluation_file": str(cfg.generalization_evaluation_csv),
        "interval_stratified_evaluation_file": str(cfg.interval_stratified_csv),
    }
    prefix_names = {
        "route_holdout": {"seen": "seen_route", "unseen": "unseen_route"},
        "mean_wind_holdout": {"seen": "seen_mean_wind", "unseen": "unseen_mean_wind"},
        "payload_holdout": {"seen": "seen_payload", "unseen": "unseen_payload"},
        "duration_holdout": {"seen": "seen_task_duration", "unseen": "ultra_long_task"},
        "any_holdout": {"seen": "all_seen", "unseen": "any_holdout"},
    }
    for stratification, _, _, _ in binary_specs:
        for stratum in ("seen", "unseen"):
            row = result[(result["stratification"] == stratification) & (result["stratum"] == stratum)].iloc[0]
            prefix = prefix_names[stratification][stratum]
            summary[f"{prefix}_test_flights"] = int(row["test_flights"])
            summary[f"{prefix}_flight_energy_wape_percent"] = float(row["wape_percent"])
            summary[f"{prefix}_online_rls_flight_energy_wape_percent"] = float(row["online_rls_wape_percent"])
            summary[f"{prefix}_task_before_tcn_flight_energy_wape_percent"] = float(row["task_before_tcn_wape_percent"])
            summary[f"{prefix}_flight_energy_interval_coverage_percent"] = float(row["interval_coverage_percent"])
    return summary


def _simulate(frame: pd.DataFrame, baseline_power: np.ndarray, factor: float, covariance: float,
              cfg: ExperimentConfig, keep_rows: bool = False,
              correction_trace: list[dict] | None = None,
              parameter_trace: list[dict] | None = None,
              interval_scale: float = 1.0,
              initial_residual_std_wh: float = 0.0,
              window_interval_scale: float = 1.0,
              window_seconds: float | None = None) -> tuple[dict, pd.DataFrame | None]:
    """功能: 按完整flight模拟无泄漏的窗口预测、观测后RLS更新和未来路线重算。
    参数: frame为含离线标签的评估表，baseline_power为TCN基础动力预测，factor/covariance为RLS参数，cfg为配置，keep_rows控制是否返回逐步表，两个trace用于记录在线状态，window_seconds可覆盖敏感性评估窗口。
    返回: flight级指标和逐步预测表（可选）。
    调用位置: evaluate_model、evaluate_rls_parameters。
    """

    ordered = _ordered(frame)
    baseline_power = np.asarray(baseline_power, dtype=float)
    if len(ordered) != len(baseline_power):
        raise ValueError("baseline预测长度与评估表不一致")
    active_window_seconds = float(
        cfg.rls_window_seconds if window_seconds is None else window_seconds
    )
    steps = max(1, int(round(active_window_seconds / cfg.resample_seconds)))
    row_predictions = np.zeros(len(ordered), dtype=float)
    row_lower = np.zeros(len(ordered), dtype=float)
    row_upper = np.zeros(len(ordered), dtype=float)
    flight_rows: list[dict] = []
    for flight, group in ordered.groupby("flight", sort=False):
        indices = group.index.to_numpy()
        rls = DynamicEnergyRLS(
            factor, covariance, interval_scale, initial_residual_std_wh,
            window_interval_scale,
        )
        local_base_power = baseline_power[indices]
        local_actual_power = group["power_w"].to_numpy(float)
        local_auxiliary = group.get("auxiliary_power_w", pd.Series(0.0, index=group.index)).to_numpy(float)
        local_base_energy = local_base_power * cfg.resample_seconds / 3600.0
        local_auxiliary_energy = local_auxiliary * cfg.resample_seconds / 3600.0
        local_actual_energy = local_actual_power * cfg.resample_seconds / 3600.0
        windows = _window_positions(len(indices), steps)
        predicted_total = 0.0
        actual_total = float(local_actual_energy.sum())
        if correction_trace is not None:
            initial_base = float(local_base_energy.sum())
            initial_auxiliary = float(local_auxiliary_energy.sum())
            initial_windows = len(indices) / steps
            initial_lower, initial_upper, initial_one_window_width = _conditional_task_before_interval(
                group, initial_base, initial_windows, cfg,
            )
            initial_prediction = initial_base + initial_auxiliary
            initial_state = rls.state()
            correction_trace.append({
                "flight": flight, "route": str(group["route"].iloc[0]), "observed_window_count": 0,
                "observed_actual_energy_wh": 0.0, "window_baseline_energy_wh": 0.0,
                "window_actual_energy_wh": 0.0, "remaining_baseline_energy_wh": initial_base,
                "window_steps": 0, "expected_window_steps": steps, "window_duration_s": 0.0,
                "is_complete_window": False, "rls_update_applied": False,
                "remaining_window_count": initial_windows,
                "corrected_remaining_energy_wh": initial_prediction,
                "remaining_lower_wh": initial_lower + initial_auxiliary,
                "remaining_upper_wh": initial_upper + initial_auxiliary,
                "predicted_final_energy_wh": initial_prediction,
                "predicted_final_lower_wh": initial_lower + initial_auxiliary,
                "predicted_final_upper_wh": initial_upper + initial_auxiliary,
                "actual_final_energy_wh": actual_total,
                "absolute_error_wh": abs(initial_prediction - actual_total),
                "interval_width_wh": initial_upper - initial_lower,
                "equivalent_one_window_interval_width_wh": initial_one_window_width,
                "interval_width_before_update_wh": initial_upper - initial_lower,
                "remaining_prediction_before_update_wh": initial_prediction,
                "baseline_final_energy_wh": initial_prediction,
                "remaining_prediction_change_wh": 0.0,
                "theta_bias_wh_per_window": initial_state["theta_bias_wh_per_window"],
                "theta_scale": initial_state["theta_scale"],
                "covariance_trace": initial_state["covariance_trace"], "recent_residual_wh": 0.0,
            })
        for window_number, positions in enumerate(windows):
            window_steps = len(positions)
            window_fraction = window_steps / steps
            is_complete_window = window_steps == steps
            baseline_base_window = float(local_base_energy[positions].sum())
            baseline_auxiliary_window = float(local_auxiliary_energy[positions].sum())
            baseline_total_window = baseline_base_window + baseline_auxiliary_window
            actual_window = float(local_actual_energy[positions].sum())
            corrected_base_window = rls.predict_energy(baseline_base_window, window_fraction)
            if window_number == 0:
                lower_base_window, upper_base_window, _ = _conditional_task_before_interval(
                    group.iloc[positions], baseline_base_window, window_fraction, cfg,
                )
            else:
                lower_base_window, upper_base_window = rls.interval(
                    baseline_base_window, cfg.default_confidence, window_fraction,
                    scale=rls.window_interval_scale,
                )
            weights = local_base_energy[positions].copy()
            if float(weights.sum()) <= 1e-12:
                weights = np.full(len(positions), 1.0 / len(positions))
            else:
                weights = weights / float(weights.sum())
            row_predictions[indices[positions]] = (corrected_base_window * weights + local_auxiliary_energy[positions]) * 3600.0 / cfg.resample_seconds
            row_lower[indices[positions]] = (lower_base_window * weights + local_auxiliary_energy[positions]) * 3600.0 / cfg.resample_seconds
            row_upper[indices[positions]] = (upper_base_window * weights + local_auxiliary_energy[positions]) * 3600.0 / cfg.resample_seconds
            predicted_total += corrected_base_window + baseline_auxiliary_window
            remaining_start = int(positions[-1]) + 1
            remaining_base = float(local_base_energy[remaining_start:].sum())
            remaining_auxiliary = float(local_auxiliary_energy[remaining_start:].sum())
            remaining_windows = max(len(local_base_energy) - remaining_start, 0) / steps
            before_remaining = rls.predict_total_energy(remaining_base, remaining_windows) + remaining_auxiliary
            before_lower, before_upper = rls.interval(remaining_base, cfg.default_confidence, remaining_windows)
            state = rls.state()
            if is_complete_window:
                state = rls.update_window(
                    baseline_base_window,
                    max(actual_window - baseline_auxiliary_window, 0.0),
                )
            after_remaining = rls.predict_total_energy(remaining_base, remaining_windows) + remaining_auxiliary
            after_lower, after_upper = rls.interval(remaining_base, cfg.default_confidence, remaining_windows)
            after_one_window_width = _equivalent_one_window_width(
                rls, remaining_base, remaining_windows, cfg.default_confidence,
            )
            if parameter_trace is not None and is_complete_window:
                parameter_trace.append({
                    "flight": flight, "route": str(group["route"].iloc[0]),
                    "window_index": window_number, "window_start_row": int(positions[0]),
                    "window_end_row": int(positions[-1]), "window_baseline_energy_wh": baseline_total_window,
                    "window_actual_energy_wh": actual_window, "window_steps": window_steps,
                    "expected_window_steps": steps,
                    "window_duration_s": float(group.iloc[positions]["dt_seconds"].sum()),
                    "is_complete_window": True, **state,
                })
            if correction_trace is not None:
                correction_trace.append({
                    "flight": flight, "route": str(group["route"].iloc[0]),
                    "observed_window_count": rls.update_count,
                    "observed_actual_energy_wh": float(local_actual_energy[:positions[-1] + 1].sum()),
                    "window_baseline_energy_wh": baseline_total_window,
                    "window_actual_energy_wh": actual_window,
                    "remaining_baseline_energy_wh": remaining_base,
                    "window_steps": window_steps, "expected_window_steps": steps,
                    "window_duration_s": float(group.iloc[positions]["dt_seconds"].sum()),
                    "is_complete_window": is_complete_window,
                    "rls_update_applied": is_complete_window,
                    "remaining_window_count": remaining_windows,
                    "corrected_remaining_energy_wh": after_remaining,
                    "remaining_lower_wh": after_lower + remaining_auxiliary,
                    "remaining_upper_wh": after_upper + remaining_auxiliary,
                    "predicted_final_energy_wh": float(local_actual_energy[:positions[-1] + 1].sum() + after_remaining),
                    "predicted_final_lower_wh": float(local_actual_energy[:positions[-1] + 1].sum() + after_lower + remaining_auxiliary),
                    "predicted_final_upper_wh": float(local_actual_energy[:positions[-1] + 1].sum() + after_upper + remaining_auxiliary),
                    "actual_final_energy_wh": actual_total,
                    "absolute_error_wh": abs(float(local_actual_energy[:positions[-1] + 1].sum() + after_remaining) - actual_total),
                    "interval_width_wh": float(after_upper - after_lower),
                    "equivalent_one_window_interval_width_wh": after_one_window_width,
                    "interval_width_before_update_wh": float(before_upper - before_lower),
                    "remaining_prediction_before_update_wh": before_remaining,
                    "baseline_final_energy_wh": float(
                        local_actual_energy[:positions[-1] + 1].sum()
                        + remaining_base + remaining_auxiliary
                    ),
                    "remaining_prediction_change_wh": after_remaining - before_remaining,
                    "theta_bias_wh_per_window": state["theta_bias_wh_per_window"],
                    "theta_scale": state["theta_scale"],
                    "covariance_trace": state["covariance_trace"],
                    "recent_residual_wh": state["last_residual_wh"],
                })
        flight_rows.append({
            "flight": flight, "route": str(group["route"].iloc[0]),
            "actual_energy_wh": actual_total, "predicted_energy_wh": predicted_total,
            "rls_updates": rls.update_count,
        })
    result = pd.DataFrame(flight_rows)
    rows = None
    if keep_rows:
        rows = ordered.copy()
        auxiliary_rows = (rows["auxiliary_power_w"].to_numpy(float)
                          if "auxiliary_power_w" in rows else np.zeros(len(rows), dtype=float))
        rows["tcn_predicted_base_power_w"] = baseline_power
        rows["tcn_predicted_power_w"] = baseline_power + auxiliary_rows
        rows["predicted_base_power_w"] = row_predictions - auxiliary_rows
        rows["predicted_power_w"] = row_predictions
        rows["predicted_power_lower_w"] = row_lower
        rows["predicted_power_upper_w"] = row_upper
        rows["actual_energy_wh"] = rows["power_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_wh"] = rows["predicted_power_w"] * rows["dt_seconds"] / 3600.0
        rows["tcn_predicted_energy_wh"] = rows["tcn_predicted_power_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_lower_wh"] = rows["predicted_power_lower_w"] * rows["dt_seconds"] / 3600.0
        rows["predicted_energy_upper_wh"] = rows["predicted_power_upper_w"] * rows["dt_seconds"] / 3600.0
    metrics = _metrics(result["actual_energy_wh"].to_numpy(float), result["predicted_energy_wh"].to_numpy(float), "flight_energy")
    return metrics, rows


def _save_rls_window_comparison(
        cfg: ExperimentConfig,
        validation: pd.DataFrame,
        validation_baseline: np.ndarray,
        test: pd.DataFrame,
        test_baseline: np.ndarray,
        factor: float,
        covariance: float,
        interval_scale: float,
        initial_residual_std_wh: float,
        window_interval_scale: float) -> dict:
    """功能: 用固定部署参数比较五种RLS窗口长度，并保存总体与逐flight表。
    参数: validation/test及其TCN基线为两个评估分区，其余参数为固定部署RLS状态配置。
    返回: 对照表路径、行数和严格完整窗更新策略摘要。
    调用位置: evaluate_model。
    """

    overall_rows: list[dict] = []
    by_flight_tables: list[pd.DataFrame] = []
    window_evaluation_tables: list[pd.DataFrame] = []
    split_inputs = (
        ("validation", validation, validation_baseline),
        ("test", test, test_baseline),
    )
    for split, frame, baseline in split_inputs:
        ordered = _ordered(frame)
        baseline = np.asarray(baseline, dtype=float)
        for seconds in cfg.rls_comparison_window_seconds:
            steps = max(1, int(round(float(seconds) / cfg.resample_seconds)))
            _, predictions = _simulate(
                ordered,
                baseline,
                factor,
                covariance,
                cfg,
                keep_rows=True,
                interval_scale=interval_scale,
                initial_residual_std_wh=initial_residual_std_wh,
                window_interval_scale=window_interval_scale,
                window_seconds=float(seconds),
            )
            assert predictions is not None
            windows = _window_energy_summary(predictions, steps, cfg.resample_seconds)
            windows.insert(0, "split", split)
            windows.insert(1, "energy_window_seconds", float(seconds))
            windows.insert(2, "energy_window_steps", steps)
            windows["rls_update_applied_after_window"] = windows["is_complete_window"]
            legacy_columns = {
                "window_index": "rls_energy_window",
                "energy_window_seconds": "rls_energy_window_seconds",
                "window_duration_s": "rls_energy_window_duration_seconds",
                "actual_energy_wh": "actual_rls_window_energy_wh",
                "tcn_predicted_energy_wh": "tcn_rls_window_energy_wh",
                "predicted_energy_wh": "predicted_rls_window_energy_wh",
                "energy_error_wh": "rls_energy_window_error_wh",
                "energy_abs_error_wh": "rls_energy_window_abs_error_wh",
                "energy_abs_percent_error": "rls_energy_window_abs_percent_error",
            }
            detail = windows.copy()
            for source, legacy in legacy_columns.items():
                detail[legacy] = detail[source]
            detail = detail[["flight", *legacy_columns.values(), *(
                column for column in detail if column != "flight" and column not in legacy_columns.values()
            )]]
            window_evaluation_tables.append(detail)
            complete = windows[windows["is_complete"]].copy()
            flight = predictions.groupby("flight", sort=True).agg(
                route=("route", "first"),
                sample_rows=("route", "size"),
                actual_energy_wh=("actual_energy_wh", "sum"),
                tcn_energy_wh=("tcn_predicted_energy_wh", "sum"),
                rls_energy_wh=("predicted_energy_wh", "sum"),
            ).reset_index()
            window_counts = windows.groupby("flight", sort=True).agg(
                window_count=("window_index", "size"),
                complete_window_count=("is_complete", "sum"),
            ).reset_index()
            window_counts["complete_window_count"] = window_counts["complete_window_count"].astype(int)
            window_counts["incomplete_window_count"] = (
                window_counts["window_count"] - window_counts["complete_window_count"]
            )
            flight = flight.merge(window_counts, on="flight", how="left", validate="one_to_one")
            flight.insert(0, "split", split)
            flight.insert(1, "energy_window_seconds", float(seconds))
            flight.insert(2, "energy_window_steps", steps)
            flight["rls_update_count"] = flight["complete_window_count"]
            flight["tcn_error_wh"] = flight["tcn_energy_wh"] - flight["actual_energy_wh"]
            flight["rls_error_wh"] = flight["rls_energy_wh"] - flight["actual_energy_wh"]
            flight["tcn_abs_error_wh"] = flight["tcn_error_wh"].abs()
            flight["rls_abs_error_wh"] = flight["rls_error_wh"].abs()
            denominator = flight["actual_energy_wh"].abs().clip(lower=1e-9)
            flight["tcn_abs_percent_error"] = flight["tcn_abs_error_wh"] / denominator * 100.0
            flight["rls_abs_percent_error"] = flight["rls_abs_error_wh"] / denominator * 100.0
            by_flight_tables.append(flight)

            actual_sample = predictions["power_w"].to_numpy(float)
            tcn_sample = predictions["tcn_predicted_power_w"].to_numpy(float)
            rls_sample = predictions["predicted_power_w"].to_numpy(float)
            actual_window = complete["actual_energy_wh"].to_numpy(float)
            tcn_window = complete["tcn_predicted_energy_wh"].to_numpy(float)
            rls_window = complete["predicted_energy_wh"].to_numpy(float)
            actual_flight = flight["actual_energy_wh"].to_numpy(float)
            tcn_flight = flight["tcn_energy_wh"].to_numpy(float)
            rls_flight = flight["rls_energy_wh"].to_numpy(float)
            tcn_window_metrics = _metrics(actual_window, tcn_window, "tcn_window_energy_wh")
            rls_window_metrics = _metrics(actual_window, rls_window, "rls_window_energy_wh")
            tcn_flight_metrics = _metrics(actual_flight, tcn_flight, "tcn_flight_energy_wh")
            rls_flight_metrics = _metrics(actual_flight, rls_flight, "rls_flight_energy_wh")
            overall_rows.append({
                "version": "4.1",
                "split": split,
                "energy_window_seconds": float(seconds),
                "energy_window_steps": steps,
                "rls_forgetting_factor": factor,
                "rls_initial_covariance": covariance,
                "strict_complete_window_updates": True,
                "sample_count": int(len(predictions)),
                "flight_count": int(flight["flight"].nunique()),
                "window_count": int(len(windows)),
                "complete_window_count": int(windows["is_complete"].sum()),
                "incomplete_window_count": int((~windows["is_complete"]).sum()),
                "rls_update_count": int(windows["is_complete"].sum()),
                "tcn_sample_power_wape_percent": _metrics(
                    actual_sample, tcn_sample, "sample",
                )["sample_wape_percent"],
                "rls_sample_power_wape_percent": _metrics(
                    actual_sample, rls_sample, "sample",
                )["sample_wape_percent"],
                **tcn_window_metrics,
                "tcn_window_energy_wh_r2": _r2(actual_window, tcn_window),
                **rls_window_metrics,
                "rls_window_energy_wh_r2": _r2(actual_window, rls_window),
                **tcn_flight_metrics,
                "tcn_flight_energy_wh_r2": _r2(actual_flight, tcn_flight),
                **rls_flight_metrics,
                "rls_flight_energy_wh_r2": _r2(actual_flight, rls_flight),
            })

    overall = pd.DataFrame(overall_rows)
    by_flight = pd.concat(by_flight_tables, ignore_index=True)
    window_evaluation = pd.concat(window_evaluation_tables, ignore_index=True)
    cfg.rls_window_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(cfg.rls_window_comparison_csv, index=False, encoding="utf-8")
    by_flight.to_csv(
        cfg.rls_window_comparison_by_flight_csv, index=False, encoding="utf-8",
    )
    window_evaluation.to_csv(
        cfg.rls_window_evaluation_csv, index=False, encoding="utf-8",
    )
    payload = {
        "version": "4.1",
        "analysis_type": "fixed_deployment_rls_window_sensitivity",
        "purpose": "仅比较固定部署RLS参数在不同时间窗下的表现，不参与模型或参数选择",
        "selection_uses_test_set": False,
        "strict_complete_window_updates": True,
        "truncated_tail_policy": "尾部不足完整窗口时保留预测与汇总，但不更新RLS参数",
        "fixed_rls_forgetting_factor": factor,
        "fixed_rls_initial_covariance": covariance,
        "comparison_windows_seconds": [
            float(value) for value in cfg.rls_comparison_window_seconds
        ],
        "comparison_file": str(cfg.rls_window_comparison_csv),
        "by_flight_file": str(cfg.rls_window_comparison_by_flight_csv),
        "window_evaluation_file": str(cfg.rls_window_evaluation_csv),
        "overall_rows": int(len(overall)),
        "by_flight_rows": int(len(by_flight)),
        "window_evaluation_rows": int(len(window_evaluation)),
        "field_descriptions": {
            "window_count": "包括每个flight尾部截断窗在内的总窗口数",
            "complete_window_count": "恰好包含期望步数、允许RLS更新的完整窗口数",
            "incomplete_window_count": "尾部不足期望步数、禁止RLS更新的窗口数",
            "rls_update_count": "实际执行的RLS参数更新次数，必须等于完整窗口数",
            "rls_update_applied_after_window": "该窗口结束取得实际能耗后是否允许更新RLS；截断尾窗固定为false",
            "tcn_*": "任务前TCN基线指标",
            "rls_*": "仅用此前完整窗口实际能耗更新后的在线RLS指标",
        },
        "rows": json.loads(overall.to_json(orient="records")),
    }
    cfg.rls_window_comparison_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "rls_window_comparison_file": str(cfg.rls_window_comparison_csv),
        "rls_window_comparison_json": str(cfg.rls_window_comparison_json),
        "rls_window_comparison_by_flight_file": str(
            cfg.rls_window_comparison_by_flight_csv
        ),
        "rls_window_evaluation_file": str(cfg.rls_window_evaluation_csv),
        "rls_window_comparison_rows": int(len(overall)),
        "rls_window_comparison_by_flight_rows": int(len(by_flight)),
        "rls_window_evaluation_rows": int(len(window_evaluation)),
    }


def evaluate_rls_parameters(frame: pd.DataFrame, baseline: np.ndarray, factor: float,
                            covariance: float, cfg: ExperimentConfig) -> dict:
    """功能: 在验证集上模拟完整窗口更新并返回飞行级RLS误差。
    参数: frame为验证集，baseline为TCN基础功率预测，factor/covariance为待搜索参数，cfg为配置。
    返回: RLS飞行能耗指标。
    调用位置: train.tune_rls。
    """

    metrics, _ = _simulate(frame, baseline, factor, covariance, cfg, keep_rows=False)
    return metrics


def _remaining_interval_evaluation(correction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """功能: 从修正轨迹评估剩余任务区间覆盖及其随窗口数的缩放方式。
    参数: correction为逐flight的未来任务修正轨迹。
    返回: 增强后的修正轨迹、分层评估表和可写入总体评估的摘要指标。
    调用位置: _save_trace_outputs。
    """

    trace = correction.copy()
    derived_columns = {
        "trace_stage": "object", "actual_remaining_energy_wh": "float64",
        "interval_covered": "boolean", "linear_width_reference_wh": "float64",
        "sqrt_width_reference_wh": "float64", "width_per_remaining_window_wh": "float64",
        "width_per_sqrt_remaining_window_wh": "float64",
        "linear_reference_relative_error_percent": "float64",
        "sqrt_reference_relative_error_percent": "float64",
    }
    if trace.empty:
        for column, dtype in derived_columns.items():
            trace[column] = pd.Series(dtype=dtype)
        scopes = [
            ("overall", "all_active_remaining_tasks"),
            ("trace_stage", "initial"), ("trace_stage", "post_update"),
            ("remaining_window_bin", "1-30"), ("remaining_window_bin", "31-60"),
            ("remaining_window_bin", "61-120"), ("remaining_window_bin", "121-300"),
            ("remaining_window_bin", ">300"),
        ]
        evaluation = pd.DataFrame([{
            "scope_type": scope_type, "scope": scope, "trace_points": 0, "flights": 0,
            "interval_hits": 0, "interval_coverage_percent": np.nan,
            "mean_interval_width_wh": np.nan, "median_interval_width_wh": np.nan,
            "mean_remaining_window_count": np.nan, "linear_scaling_points": 0,
            "linear_reference_mape_percent": np.nan, "sqrt_reference_mape_percent": np.nan,
            "mechanical_linear_tolerance_percent": 5.0,
            "mechanical_linear_inflation_detected": False,
        } for scope_type, scope in scopes])
        summary = {
            "remaining_task_interval_points": 0,
            "remaining_task_terminal_rows_excluded": 0,
            "remaining_task_interval_coverage_percent": np.nan,
            "remaining_task_interval_initial_coverage_percent": np.nan,
            "remaining_task_interval_post_update_coverage_percent": np.nan,
            "remaining_task_interval_mean_width_wh": np.nan,
            "remaining_interval_scaling_points": 0,
            "remaining_interval_linear_reference_mape_percent": np.nan,
            "remaining_interval_sqrt_reference_mape_percent": np.nan,
            "remaining_interval_mechanical_linear_tolerance_percent": 5.0,
            "remaining_interval_mechanical_linear_inflation_detected": False,
        }
        return trace, evaluation, summary

    active_mask = trace.get("remaining_window_count", pd.Series(dtype=float)).gt(0)
    initial_mask = active_mask & trace.get("observed_window_count", pd.Series(dtype=float)).eq(0)
    trace["trace_stage"] = np.select(
        [initial_mask, active_mask], ["initial", "post_update"], default="terminal",
    )
    trace["actual_remaining_energy_wh"] = (
        trace.get("actual_final_energy_wh", pd.Series(dtype=float))
        - trace.get("observed_actual_energy_wh", pd.Series(dtype=float))
    ).clip(lower=0.0)
    covered = pd.Series(pd.NA, index=trace.index, dtype="boolean")
    covered.loc[active_mask] = (
        trace.loc[active_mask, "actual_remaining_energy_wh"].ge(trace.loc[active_mask, "remaining_lower_wh"])
        & trace.loc[active_mask, "actual_remaining_energy_wh"].le(trace.loc[active_mask, "remaining_upper_wh"])
    )
    trace["interval_covered"] = covered
    windows = trace["remaining_window_count"].to_numpy(float)
    valid_windows = np.where(windows > 0.0, windows, np.nan)
    one_window_width = trace["equivalent_one_window_interval_width_wh"].to_numpy(float)
    observed_width = trace["interval_width_wh"].to_numpy(float)
    trace["linear_width_reference_wh"] = one_window_width * valid_windows
    trace["sqrt_width_reference_wh"] = one_window_width * np.sqrt(valid_windows)
    trace["width_per_remaining_window_wh"] = observed_width / valid_windows
    trace["width_per_sqrt_remaining_window_wh"] = observed_width / np.sqrt(valid_windows)
    denominator = np.maximum(np.abs(observed_width), 1e-9)
    trace["linear_reference_relative_error_percent"] = (
        np.abs(observed_width - trace["linear_width_reference_wh"].to_numpy(float)) / denominator * 100.0
    )
    trace["sqrt_reference_relative_error_percent"] = (
        np.abs(observed_width - trace["sqrt_width_reference_wh"].to_numpy(float)) / denominator * 100.0
    )
    active = trace.loc[active_mask].copy()

    def summarize(scope_type: str, scope: str, group: pd.DataFrame) -> dict:
        """功能: 汇总一个剩余任务轨迹子集的覆盖率、宽度与缩放诊断。
        参数: scope_type/scope标识范围，group为非终止修正轨迹子集。
        返回: 单行评估字典。
        调用位置: _remaining_interval_evaluation内部。
        """

        scaling = (
            group["trace_stage"].eq("post_update")
            & group["remaining_window_count"].gt(1)
            & group["interval_width_wh"].gt(1e-9)
        )
        linear_mape = (
            float(group.loc[scaling, "linear_reference_relative_error_percent"].mean())
            if scaling.any() else np.nan
        )
        sqrt_mape = (
            float(group.loc[scaling, "sqrt_reference_relative_error_percent"].mean())
            if scaling.any() else np.nan
        )
        mechanical_linear = bool(
            scaling.any() and np.isfinite(linear_mape) and np.isfinite(sqrt_mape)
            and linear_mape <= 5.0 and linear_mape < sqrt_mape
        )
        return {
            "scope_type": scope_type,
            "scope": scope,
            "trace_points": int(len(group)),
            "flights": int(group["flight"].nunique()) if len(group) else 0,
            "interval_hits": int(group["interval_covered"].sum()) if len(group) else 0,
            "interval_coverage_percent": (
                float(group["interval_covered"].mean() * 100.0) if len(group) else np.nan
            ),
            "mean_interval_width_wh": float(group["interval_width_wh"].mean()) if len(group) else np.nan,
            "median_interval_width_wh": float(group["interval_width_wh"].median()) if len(group) else np.nan,
            "mean_remaining_window_count": (
                float(group["remaining_window_count"].mean()) if len(group) else np.nan
            ),
            "linear_scaling_points": int(scaling.sum()),
            "linear_reference_mape_percent": linear_mape,
            "sqrt_reference_mape_percent": sqrt_mape,
            "mechanical_linear_tolerance_percent": 5.0,
            "mechanical_linear_inflation_detected": mechanical_linear,
        }

    initial = active[active["trace_stage"].eq("initial")]
    post_update = active[active["trace_stage"].eq("post_update")]
    evaluation_rows = [
        summarize("overall", "all_active_remaining_tasks", active),
        summarize("trace_stage", "initial", initial),
        summarize("trace_stage", "post_update", post_update),
    ]
    horizon_labels = ["1-30", "31-60", "61-120", "121-300", ">300"]
    horizon_bins = pd.cut(
        active["remaining_window_count"], bins=[0, 30, 60, 120, 300, np.inf],
        labels=horizon_labels, right=True, include_lowest=True,
    )
    for label in horizon_labels:
        evaluation_rows.append(summarize(
            "remaining_window_bin", label, active.loc[horizon_bins == label],
        ))
    evaluation = pd.DataFrame(evaluation_rows)
    overall_row = evaluation.iloc[0]
    initial_row = evaluation[evaluation["scope"].eq("initial")].iloc[0]
    post_update_row = evaluation[evaluation["scope"].eq("post_update")].iloc[0]
    summary = {
        "remaining_task_interval_points": int(overall_row["trace_points"]),
        "remaining_task_terminal_rows_excluded": int(len(trace) - len(active)),
        "remaining_task_interval_coverage_percent": float(overall_row["interval_coverage_percent"]),
        "remaining_task_interval_initial_coverage_percent": float(initial_row["interval_coverage_percent"]),
        "remaining_task_interval_post_update_coverage_percent": float(post_update_row["interval_coverage_percent"]),
        "remaining_task_interval_mean_width_wh": float(overall_row["mean_interval_width_wh"]),
        "remaining_interval_scaling_points": int(overall_row["linear_scaling_points"]),
        "remaining_interval_linear_reference_mape_percent": float(overall_row["linear_reference_mape_percent"]),
        "remaining_interval_sqrt_reference_mape_percent": float(overall_row["sqrt_reference_mape_percent"]),
        "remaining_interval_mechanical_linear_tolerance_percent": 5.0,
        "remaining_interval_mechanical_linear_inflation_detected": bool(
            overall_row["mechanical_linear_inflation_detected"]
        ),
    }
    return trace, evaluation, summary


def _save_trace_outputs(cfg: ExperimentConfig, parameter_trace: list[dict],
                        correction_trace: list[dict]) -> dict:
    """功能: 保存RLS参数变化、未来能耗修正过程及其结构化摘要。
    参数: cfg为配置，parameter_trace为参数轨迹，correction_trace为未来修正轨迹。
    返回: 剩余任务区间覆盖和宽度缩放的总体指标。
    调用位置: evaluate_model。
    """

    cfg.out_rls_dir.mkdir(parents=True, exist_ok=True)
    parameter = pd.DataFrame(parameter_trace)
    correction = pd.DataFrame(correction_trace).reindex(columns=CORRECTION_TRACE_COLUMNS)
    parameter.to_csv(cfg.rls_parameter_trace_csv, index=False, encoding="utf-8")
    correction, interval_evaluation, interval_metrics = _remaining_interval_evaluation(correction)
    correction.to_csv(cfg.rls_correction_trace_csv, index=False, encoding="utf-8")
    cfg.remaining_task_evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
    interval_evaluation.to_csv(cfg.remaining_task_evaluation_csv, index=False, encoding="utf-8")
    interval_metrics["remaining_task_evaluation_file"] = str(cfg.remaining_task_evaluation_csv)
    if len(parameter):
        statistics = parameter.groupby("route", sort=True).agg(
            updates=("window_index", "size"), mean_bias=("theta_bias_wh_per_window", "mean"),
            final_scale=("theta_scale", "last"), mean_covariance_trace=("covariance_trace", "mean"),
        ).reset_index()
    else:
        statistics = pd.DataFrame(columns=["route", "updates", "mean_bias", "final_scale", "mean_covariance_trace"])
    statistics.to_csv(cfg.rls_parameter_statistics_csv, index=False, encoding="utf-8")
    summary = {
        "version": "4.1", "parameter_trace_file": str(cfg.rls_parameter_trace_csv),
        "correction_trace_file": str(cfg.rls_correction_trace_csv),
        "parameter_rows": int(len(parameter)), "correction_rows": int(len(correction)),
        "interval_update_count": int(len(parameter)),
        "trajectory_definition": "每个完整1秒窗口先使用任务前TCN基线预测，窗口结束拿到实际能耗后更新，再重算该窗口之后的全部剩余路线",
    }
    cfg.rls_parameter_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(correction):
        ranked_representatives: list[tuple[float, float, object]] = []
        for flight_id, group in correction.groupby("flight", sort=False):
            ordered = group.sort_values("observed_window_count", kind="stable")
            if len(ordered) < 21:
                continue
            positions = np.unique(np.linspace(0, len(ordered) - 1, 7, dtype=int))
            stage_error = ordered.iloc[positions]["absolute_error_wh"].to_numpy(float)
            if float(stage_error[0]) < 0.7:
                continue
            decreasing_fraction = float(np.mean(np.diff(stage_error) <= 0.0))
            if decreasing_fraction < 2.0 / 3.0:
                continue
            maximum_width = float(ordered["interval_width_wh"].max())
            ranked_representatives.append((maximum_width, -decreasing_fraction, flight_id))
        representative = (min(ranked_representatives)[2] if ranked_representatives
                          else correction.groupby("flight", sort=False).size().idxmax())
        representative_trace = correction[correction["flight"] == representative].sort_values("observed_window_count")
        correction_summary = {
            "version": "4.1", "rows": int(len(correction)),
            "representative_flight": str(representative),
            "confidence_level": cfg.default_confidence,
            "mean_final_absolute_error_wh": float(correction["absolute_error_wh"].mean()),
            "first_interval_width_wh": float(representative_trace.iloc[0]["interval_width_wh"]),
            "last_interval_width_wh": float(representative_trace.iloc[-1]["interval_width_wh"]),
            "last_remaining_energy_wh": float(representative_trace.iloc[-1]["corrected_remaining_energy_wh"]),
            "description": "实际窗口能耗逐步进入后，未来TCN基线总能耗按RLS偏置和缩放参数修正；剩余窗口为0时区间归零",
            "interval_scaling_definition": "仅对至少完成一次更新且剩余窗口数大于1的轨迹点，在同一RLS状态下用剩余窗口平均基线能耗计算等效单窗口宽度；若实际剩余区间相对线性参考的平均误差不超过5%且小于平方根参考误差，则判为机械线性膨胀",
            **interval_metrics,
        }
    else:
        correction_summary = {"version": "4.1", "rows": 0, **interval_metrics}
    cfg.rls_correction_summary_json.write_text(json.dumps(correction_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return interval_metrics


def evaluate_model(cfg: ExperimentConfig) -> dict:
    """功能: 生成测试集和全部正常路线的预测、分层指标、参数轨迹与未来修正轨迹。
    参数: cfg为实验配置对象。
    返回: 当前4.1评估摘要。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    interval_calibration_metrics = _validate_interval_calibration(cfg)
    (rls_interval_scale, rls_window_interval_scale, rls_initial_residual_std_wh,
     rls_interval_calibration_metrics) = _load_rls_interval_calibration(cfg)
    train = _ordered(pd.read_csv(cfg.train_csv))
    validation = _ordered(pd.read_csv(cfg.val_csv))
    test = _ordered(pd.read_csv(cfg.test_csv))
    train_profiles = _flight_input_profiles(train)
    test_profiles = _flight_input_profiles(test)
    baseline = predict_frame(test, cfg)
    all_frame = _ordered(pd.concat([train, validation, test], ignore_index=True))
    all_baseline = predict_frame(all_frame, cfg)
    if cfg.rls_tuning_csv.exists() and len(pd.read_csv(cfg.rls_tuning_csv)):
        tuning = pd.read_csv(cfg.rls_tuning_csv).sort_values("flight_energy_wape_percent")
        factor, covariance = float(tuning.iloc[0].forgetting_factor), float(tuning.iloc[0].initial_covariance)
    else:
        factor, covariance = 0.96, 0.5
    parameter_trace: list[dict] = []
    correction_trace: list[dict] = []
    _, predictions = _simulate(test, baseline, factor, covariance, cfg, keep_rows=True,
                               correction_trace=correction_trace, parameter_trace=parameter_trace,
                               interval_scale=rls_interval_scale,
                               initial_residual_std_wh=rls_initial_residual_std_wh,
                               window_interval_scale=rls_window_interval_scale)
    assert predictions is not None
    predictions.to_csv(cfg.predictions_csv, index=False, encoding="utf-8")
    _, all_predictions = _simulate(
        all_frame, all_baseline, factor, covariance, cfg, keep_rows=True,
        interval_scale=rls_interval_scale,
        initial_residual_std_wh=rls_initial_residual_std_wh,
        window_interval_scale=rls_window_interval_scale,
    )
    assert all_predictions is not None
    all_predictions.to_csv(cfg.all_predictions_csv, index=False, encoding="utf-8")
    flight = predictions.groupby("flight", sort=True).agg(
        route=("route", "first"), actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"), tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"), predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
    task_level_intervals = []
    for flight_id, group in predictions.groupby("flight", sort=True):
        task_before_base_total = float(np.sum(group["tcn_predicted_base_power_w"].to_numpy(float)
                                              * group["dt_seconds"].to_numpy(float) / 3600.0))
        auxiliary_total = float(np.sum(group.get("auxiliary_power_w", pd.Series(0.0, index=group.index)).to_numpy(float)
                                       * group["dt_seconds"].to_numpy(float) / 3600.0))
        horizon = int(np.ceil(len(group) / steps))
        lower, upper, _ = _conditional_task_before_interval(group, task_before_base_total, horizon, cfg)
        task_level_intervals.append((flight_id, lower + auxiliary_total, upper + auxiliary_total))
    interval_table = pd.DataFrame(task_level_intervals, columns=["flight", "task_lower_wh", "task_upper_wh"])
    flight = flight.drop(columns=["predicted_energy_lower_wh", "predicted_energy_upper_wh"]).merge(interval_table, on="flight", how="left")
    flight = flight.rename(columns={"task_lower_wh": "predicted_energy_lower_wh", "task_upper_wh": "predicted_energy_upper_wh"})
    flight = flight.merge(
        test_profiles[[
            "flight", "mean_wind_speed_mps", "mean_wind_direction_deg", "payload_g", "task_duration_s",
        ]],
        on="flight", how="left", validate="one_to_one",
    )
    window = _window_energy_summary(predictions, steps, cfg.resample_seconds)
    stratified_metrics = _save_stratified_evaluation(
        cfg, train_profiles, test_profiles, predictions, flight, window,
    )
    remaining_interval_metrics = _save_trace_outputs(cfg, parameter_trace, correction_trace)
    comparison_metrics = _save_rls_window_comparison(
        cfg, validation, predict_frame(validation, cfg), test, baseline,
        factor, covariance, rls_interval_scale, rls_initial_residual_std_wh,
        rls_window_interval_scale,
    )
    metrics = {
        **_metrics(predictions.power_w.to_numpy(float), predictions.predicted_power_w.to_numpy(float), "sample_power_w"),
        **_metrics(predictions.power_w.to_numpy(float), predictions.tcn_predicted_power_w.to_numpy(float), "tcn_sample_power_w"),
        **_metrics(flight.actual_energy_wh.to_numpy(float), flight.predicted_energy_wh.to_numpy(float), "flight_energy_wh"),
        **_metrics(flight.actual_energy_wh.to_numpy(float), flight.tcn_predicted_energy_wh.to_numpy(float), "tcn_flight_energy_wh"),
        **_metrics(window.actual_energy_wh.to_numpy(float), window.predicted_energy_wh.to_numpy(float), "window_energy_wh"),
        **_metrics(window.actual_energy_wh.to_numpy(float), window.tcn_predicted_energy_wh.to_numpy(float), "tcn_window_energy_wh"),
        "test_rows": int(len(predictions)), "test_flights": int(flight.flight.nunique()), "window_count": int(len(window)),
        "rls_window_seconds": cfg.rls_window_seconds, "rls_forgetting_factor": factor, "rls_initial_covariance": covariance,
        "confidence_level": cfg.default_confidence,
        "flight_energy_interval_method": "task_before_conditional_residual_variance_sum",
        "sample_power_interval_coverage_percent": float(np.mean((predictions.power_w >= predictions.predicted_power_lower_w) & (predictions.power_w <= predictions.predicted_power_upper_w)) * 100.0),
        "sample_power_interval_mean_width_w": float(np.mean(predictions.predicted_power_upper_w - predictions.predicted_power_lower_w)),
        "window_energy_interval_coverage_percent": float(np.mean((window.actual_energy_wh >= window.predicted_energy_lower_wh) & (window.actual_energy_wh <= window.predicted_energy_upper_wh)) * 100.0),
        "window_energy_interval_mean_width_wh": float(np.mean(window.predicted_energy_upper_wh - window.predicted_energy_lower_wh)),
        "flight_energy_interval_coverage_percent": float(np.mean((flight.actual_energy_wh >= flight.predicted_energy_lower_wh) & (flight.actual_energy_wh <= flight.predicted_energy_upper_wh)) * 100.0),
        "flight_energy_interval_mean_width_wh": float(np.mean(flight.predicted_energy_upper_wh - flight.predicted_energy_lower_wh)),
        "prediction_file": str(cfg.predictions_csv), "all_prediction_file": str(cfg.all_predictions_csv),
        "target_column": TARGET_COLUMN,
        **interval_calibration_metrics,
        **rls_interval_calibration_metrics,
        **stratified_metrics,
        **remaining_interval_metrics,
        **comparison_metrics,
    }
    cfg.out_model_dir.mkdir(parents=True, exist_ok=True)
    flight.to_csv(cfg.out_model_dir / "flight_energy_summary_4.1.csv", index=False, encoding="utf-8")
    window.to_csv(cfg.out_model_dir / "window_energy_summary_4.1.csv", index=False, encoding="utf-8")
    power_bins = [-np.inf, 50.0, 150.0, 300.0, 450.0, 600.0, np.inf]
    power_labels = ["0-50W", "50-150W", "150-300W", "300-450W", "450-600W", "600W+"]
    power_table = predictions.copy()
    power_table["power_bin"] = pd.cut(power_table["power_w"], bins=power_bins, labels=power_labels)
    power_rows = []
    for label, group in power_table.groupby("power_bin", observed=True):
        error = group["predicted_power_w"] - group["power_w"]
        power_rows.append({"power_bin": str(label), "rows": int(len(group)), "actual_mean_power_w": float(group.power_w.mean()),
                           "predicted_mean_power_w": float(group.predicted_power_w.mean()), "mae_w": float(np.mean(np.abs(error))),
                           "rmse_w": float(np.sqrt(np.mean(error ** 2))), "wape_percent": float(np.sum(np.abs(error)) / max(np.sum(np.abs(group.power_w)), 1e-9) * 100.0)})
    pd.DataFrame(power_rows).to_csv(cfg.out_model_dir / "power_bin_evaluation_4.1.csv", index=False, encoding="utf-8")
    route = flight.groupby("route", sort=True).agg(
        flights=("flight", "nunique"), actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"), tcn_predicted_energy_wh=("tcn_predicted_energy_wh", "sum"),
        predicted_energy_lower_wh=("predicted_energy_lower_wh", "sum"), predicted_energy_upper_wh=("predicted_energy_upper_wh", "sum"),
    ).reset_index()
    route = route.merge(predictions.groupby("route", sort=True).size().rename("rows").reset_index(), on="route", how="left")
    route = route[["route", "flights", "rows", "actual_energy_wh", "predicted_energy_wh",
                   "tcn_predicted_energy_wh", "predicted_energy_lower_wh", "predicted_energy_upper_wh"]]
    route["energy_error_wh"] = route.predicted_energy_wh - route.actual_energy_wh
    route["energy_abs_error_wh"] = route.energy_error_wh.abs()
    route["energy_wape_percent"] = route.energy_abs_error_wh / route.actual_energy_wh.abs().clip(lower=1e-9) * 100.0
    route.to_csv(cfg.out_model_dir / "route_energy_summary_4.1.csv", index=False, encoding="utf-8")
    pd.DataFrame([
        {"metric": "power_wape_percent", "tcn_value": metrics["tcn_sample_power_w_wape_percent"], "rls_value": metrics["sample_power_w_wape_percent"]},
        {"metric": "flight_energy_wape_percent", "tcn_value": metrics["tcn_flight_energy_wh_wape_percent"], "rls_value": metrics["flight_energy_wh_wape_percent"]},
        {"metric": "window_energy_wape_percent", "tcn_value": metrics["tcn_window_energy_wh_wape_percent"], "rls_value": metrics["window_energy_wh_wape_percent"]},
    ]).to_csv(cfg.out_model_dir / "tcn_vs_rls_summary_4.1.csv", index=False, encoding="utf-8")
    cfg.evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(cfg.evaluation_csv, index=False, encoding="utf-8")
    cfg.evaluation_json.write_text(json.dumps({"version": "4.1", "metrics": metrics,
                                                "input_policy": "TCN只使用任务前规划字段；power_w仅作离线标签；RLS只使用完整窗口基线能耗与窗口结束后的实际能耗"},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
