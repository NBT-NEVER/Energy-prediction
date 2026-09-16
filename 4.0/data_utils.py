# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-09-16
# 文件名: data_utils.py
# 功能说明: 将原始飞行记录重采样为0.2秒规划特征并生成4.0数据集
# 版本号：4.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories


EXCLUDED_ROUTES = {"A1", "A2", "A3"}
FEATURE_COLUMNS = [
    "time_s", "dt_seconds", "task_progress", "task_duration_s",
    "planned_vx_mps", "planned_vy_mps", "planned_vz_mps",
    "planned_horizontal_speed_mps", "planned_speed_mps",
    "planned_ax_mps2", "planned_ay_mps2", "planned_az_mps2",
    "planned_acceleration_mps2", "planned_altitude_m",
    "segment_distance_m", "cumulative_distance_m", "cumulative_climb_m",
    "wind_speed_mps", "wind_sin", "wind_cos", "relative_air_speed_mps",
    "headwind_mps", "crosswind_mps", "payload_kg",
    "payload_camera_enabled", "payload_communication_enabled",
    "payload_compute_enabled", "payload_delivery_enabled",
    "payload_rated_power_w",
]

FEATURE_UNITS = {
    "time_s": "s", "dt_seconds": "s", "task_progress": "ratio", "task_duration_s": "s",
    "planned_vx_mps": "m/s", "planned_vy_mps": "m/s", "planned_vz_mps": "m/s",
    "planned_horizontal_speed_mps": "m/s", "planned_speed_mps": "m/s",
    "planned_ax_mps2": "m/s^2", "planned_ay_mps2": "m/s^2", "planned_az_mps2": "m/s^2",
    "planned_acceleration_mps2": "m/s^2", "planned_altitude_m": "m",
    "segment_distance_m": "m", "cumulative_distance_m": "m", "cumulative_climb_m": "m",
    "wind_speed_mps": "m/s", "wind_sin": "ratio", "wind_cos": "ratio",
    "relative_air_speed_mps": "m/s", "headwind_mps": "m/s", "crosswind_mps": "m/s",
    "payload_kg": "kg", "payload_camera_enabled": "bool", "payload_communication_enabled": "bool",
    "payload_compute_enabled": "bool", "payload_delivery_enabled": "bool", "payload_rated_power_w": "W",
}

SOURCE_COLUMNS = [
    "flight", "route", "time", "wind_speed", "wind_angle", "battery_voltage",
    "battery_current", "speed", "payload", "altitude", "velocity_x", "velocity_y",
    "velocity_z", "position_x", "position_y", "position_z",
]


def save_json(path: Path, payload: dict) -> None:
    """功能: 以UTF-8保存结构化4.0元数据。
    参数: path为输出路径，payload为待保存对象。
    返回: None。
    调用位置: prepare_dataset、train.py。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_raw_flights(cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 读取原始飞行记录并排除地面辅助测试航线。
    参数: cfg为4.0实验配置对象。
    返回: 未使用电池标签作为输入的原始字段表。
    调用位置: prepare_dataset。
    """

    if not cfg.raw_flights_csv.exists():
        raise FileNotFoundError(f"未找到原始飞行记录: {cfg.raw_flights_csv}")
    header = pd.read_csv(cfg.raw_flights_csv, nrows=0).columns.tolist()
    usecols = [column for column in SOURCE_COLUMNS if column in header]
    frame = pd.read_csv(cfg.raw_flights_csv, usecols=usecols, low_memory=False)
    frame["route"] = frame["route"].astype(str).str.strip()
    frame = frame[~frame["route"].isin(EXCLUDED_ROUTES)].copy()
    frame = _numeric(frame, [c for c in SOURCE_COLUMNS if c not in {"flight", "route"}])
    frame = frame.dropna(subset=["flight", "time", "battery_voltage", "battery_current"]).copy()
    return frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)


def _resample_flight(group: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 将单个flight按真实时间戳重采样到0.2秒固定网格。
    参数: group为单个flight原始记录，cfg包含固定步长。
    返回: 重采样后的记录和原始记录范围映射。
    调用位置: build_planning_frame。
    """

    group = group.sort_values("time").drop_duplicates("time")
    start = float(group["time"].min())
    end = float(group["time"].max())
    times = np.arange(start, end + cfg.resample_seconds * 0.5, cfg.resample_seconds)
    source_time = group["time"].to_numpy(dtype=float)
    out = {"flight": int(group["flight"].iloc[0]), "route": str(group["route"].iloc[0]), "time": times}
    numeric_columns = [c for c in group.columns if c not in {"flight", "route"}]
    for column in numeric_columns:
        values = group[column].to_numpy(dtype=float)
        out[column] = np.interp(times, source_time, values)
    result = pd.DataFrame(out)
    left = np.searchsorted(source_time, times, side="left").clip(0, len(source_time) - 1)
    right = np.searchsorted(source_time, times, side="right").clip(0, len(source_time) - 1)
    mapping = pd.DataFrame({
        "flight": int(group["flight"].iloc[0]), "new_time_s": times,
        "raw_left_index": left, "raw_right_index": right,
        "raw_time_left_s": source_time[left], "raw_time_right_s": source_time[right],
    })
    return result, mapping


def _add_planning_features(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 从规划速度、高度、风预报和载荷字段构造任务前特征。
    参数: frame为重采样后的原始字段表。
    返回: 含监督标签和规划输入特征的表。
    调用位置: build_planning_frame。
    """

    result = []
    for flight, group in frame.groupby("flight", sort=False):
        group = group.sort_values("time").copy()
        dt = float(cfg.resample_seconds)
        group["time_s"] = group["time"] - float(group["time"].iloc[0])
        duration = max(float(group["time_s"].iloc[-1]), dt)
        planned_speed = group["speed"].clip(lower=0).fillna(0).to_numpy(float)
        altitude = group["altitude"].fillna(0).to_numpy(float)
        # 原始数据没有任务规划向量时，把任务速度视为规划切向速度，垂直规划速度由高度轨迹差分得到。
        planned_vx = planned_speed
        planned_vy = np.zeros_like(planned_speed)
        planned_vz = np.gradient(altitude, dt)
        planned_ax = np.gradient(planned_vx, dt)
        planned_ay = np.zeros_like(planned_ax)
        planned_az = np.gradient(planned_vz, dt)
        wind_speed = group["wind_speed"].clip(lower=0).fillna(0).to_numpy(float)
        wind_angle = np.deg2rad(group["wind_angle"].fillna(0).to_numpy(float))
        wind_x = wind_speed * np.cos(wind_angle)
        wind_y = wind_speed * np.sin(wind_angle)
        horizontal = np.sqrt(planned_vx**2 + planned_vy**2)
        planned_accel = np.sqrt(planned_ax**2 + planned_ay**2 + planned_az**2)
        relative_air = np.sqrt((planned_vx - wind_x) ** 2 + (planned_vy - wind_y) ** 2 + planned_vz**2)
        headwind = -(planned_vx * wind_x + planned_vy * wind_y) / np.maximum(horizontal, 1e-6)
        crosswind = np.abs(planned_vx * wind_y - planned_vy * wind_x) / np.maximum(horizontal, 1e-6)
        segment_distance = np.sqrt(planned_vx**2 + planned_vy**2 + planned_vz**2) * dt
        group["dt_seconds"] = dt
        group["task_progress"] = np.clip(group["time_s"] / duration, 0, 1)
        group["task_duration_s"] = duration
        group["planned_vx_mps"] = planned_vx
        group["planned_vy_mps"] = planned_vy
        group["planned_vz_mps"] = planned_vz
        group["planned_horizontal_speed_mps"] = horizontal
        group["planned_speed_mps"] = np.sqrt(planned_vx**2 + planned_vy**2 + planned_vz**2)
        group["planned_ax_mps2"] = planned_ax
        group["planned_ay_mps2"] = planned_ay
        group["planned_az_mps2"] = planned_az
        group["planned_acceleration_mps2"] = planned_accel
        group["planned_altitude_m"] = altitude
        group["segment_distance_m"] = segment_distance
        group["cumulative_distance_m"] = np.cumsum(segment_distance)
        group["cumulative_climb_m"] = np.cumsum(np.maximum(planned_vz, 0) * dt)
        group["wind_speed_mps"] = wind_speed
        group["wind_sin"] = np.sin(wind_angle)
        group["wind_cos"] = np.cos(wind_angle)
        group["relative_air_speed_mps"] = relative_air
        group["headwind_mps"] = headwind
        group["crosswind_mps"] = crosswind
        group["payload_kg"] = group["payload"].fillna(0).clip(lower=0) / 1000.0
        group["payload_camera_enabled"] = 0.0
        group["payload_communication_enabled"] = 0.0
        group["payload_compute_enabled"] = 0.0
        group["payload_delivery_enabled"] = 0.0
        group["payload_rated_power_w"] = 0.0
        group["battery_current_discharge_a"] = group["battery_current"].clip(lower=0)
        group["power_w"] = (group["battery_voltage"] * group["battery_current_discharge_a"]).clip(lower=0)
        group["energy_interval_wh"] = group["power_w"] * dt / 3600.0
        result.append(group)
    return pd.concat(result, ignore_index=True)


def build_planning_frame(raw: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 重采样全部正常飞行并生成规划特征、标签和映射。
    参数: raw为load_raw_flights读取的原始表。
    返回: 4.0建模表和原始采样映射表。
    调用位置: prepare_dataset。
    """

    frames = []
    maps = []
    for _, group in raw.groupby("flight", sort=False):
        resampled, mapping = _resample_flight(group, cfg)
        frames.append(resampled)
        maps.append(mapping)
    frame = _add_planning_features(pd.concat(frames, ignore_index=True), cfg)
    frame = frame[["flight", "route", "time", "power_w", "energy_interval_wh", *FEATURE_COLUMNS]].copy()
    return frame, pd.concat(maps, ignore_index=True)


def split_by_flight(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """功能: 按完整flight划分训练、验证和测试，避免轨迹泄漏。
    参数: frame为规划特征表，cfg包含划分比例和随机种子。
    返回: train、val、test三个数据表。
    调用位置: prepare_dataset。
    """

    rng = np.random.default_rng(cfg.random_seed)
    train_ids, val_ids, test_ids = set(), set(), set()
    for _, group in frame[["flight", "route"]].drop_duplicates().groupby("route", sort=True):
        flights = group.flight.to_numpy(int, copy=True)
        rng.shuffle(flights)
        if len(flights) < 3:
            train_ids.update(int(x) for x in flights)
            continue
        n_test = max(1, int(np.ceil(len(flights) * cfg.test_ratio)))
        n_val = max(1, int(np.ceil(len(flights) * cfg.val_ratio)))
        while n_test + n_val >= len(flights):
            n_test = max(1, n_test - 1)
        test_ids.update(int(x) for x in flights[:n_test])
        val_ids.update(int(x) for x in flights[n_test:n_test + n_val])
        train_ids.update(int(x) for x in flights[n_test + n_val:])
    return {
        "train": frame[frame.flight.isin(train_ids)].copy(),
        "val": frame[frame.flight.isin(val_ids)].copy(),
        "test": frame[frame.flight.isin(test_ids)].copy(),
    }


def prepare_dataset(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 生成4.0独立数据、元数据、划分和0.2秒映射记录。
    参数: cfg为配置对象，force为是否覆盖已有数据。
    返回: 当前数据规模和排除规则摘要。
    调用位置: main.py的prepare和all模式。
    """

    ensure_directories(cfg)
    if cfg.clean_data_csv.exists() and not force:
        return json.loads(cfg.dataset_summary_json.read_text(encoding="utf-8"))
    raw = load_raw_flights(cfg)
    frame, mapping = build_planning_frame(raw, cfg)
    splits = split_by_flight(frame, cfg)
    frame.to_csv(cfg.clean_data_csv, index=False, encoding="utf-8")
    mapping.to_csv(cfg.resample_map_csv, index=False, encoding="utf-8")
    for name, table in splits.items():
        table.to_csv(getattr(cfg, f"{name}_csv"), index=False, encoding="utf-8")
    raw_deltas = raw.sort_values(["flight", "time"]).groupby("flight")["time"].diff().dropna()
    raw_sampling = {
        "median_seconds": float(raw_deltas.median()) if len(raw_deltas) else None,
        "min_seconds": float(raw_deltas.min()) if len(raw_deltas) else None,
        "max_seconds": float(raw_deltas.max()) if len(raw_deltas) else None,
        "sample_count": int(len(raw_deltas)),
    }
    meta = {
        "version": "4.0", "feature_columns": FEATURE_COLUMNS,
        "target_column": "power_w", "excluded_routes": sorted(EXCLUDED_ROUTES),
        "input_contract": "任务前规划输入；不得包含电压、电流、实飞速度、实飞加速度、实飞角速度或真实能耗。",
        "sampling": {"raw_sampling_period_unknown": False, "raw_period_seconds": raw_sampling,
                     "resample_seconds": cfg.resample_seconds,
                     "resample_rule": "按真实time线性插值规划字段，功率标签按0.2s固定间隔积分。",
                     "mapping_file": str(cfg.resample_map_csv)},
        "field_metadata": {column: {"unit": FEATURE_UNITS[column],
                                     "source": "planning input" if column in {"time_s", "dt_seconds", "planned_vx_mps", "planned_vy_mps", "planned_vz_mps", "planned_speed_mps", "planned_ax_mps2", "planned_ay_mps2", "planned_az_mps2", "planned_altitude_m", "wind_speed_mps", "payload_kg", "payload_camera_enabled", "payload_communication_enabled", "payload_compute_enabled", "payload_delivery_enabled", "payload_rated_power_w"} else "derived from planning trajectory",
                                     "task_before_available": True} for column in FEATURE_COLUMNS},
        "label_fields": {"power_w": "battery_voltage * max(battery_current, 0)", "energy_interval_wh": "power_w * 0.2 / 3600", "task_before_available": False},
        "route_policy": "route保留用于查询和分组，但不进入feature_columns；支持新路线任务输入。",
    }
    save_json(cfg.feature_meta_json, meta)
    summary = {
        "version": "4.0", "resample_seconds": cfg.resample_seconds,
        "raw_rows_after_excluding_ground_tests": int(len(raw)), "processed_rows": int(len(frame)),
        "flights": int(frame.flight.nunique()), "train_rows": int(len(splits["train"])),
        "val_rows": int(len(splits["val"])), "test_rows": int(len(splits["test"])),
        "train_flights": int(splits["train"].flight.nunique()), "val_flights": int(splits["val"].flight.nunique()),
        "test_flights": int(splits["test"].flight.nunique()), "excluded_routes": sorted(EXCLUDED_ROUTES),
        "routes": sorted(frame.route.unique().tolist()), "feature_count": len(FEATURE_COLUMNS),
        "raw_sampling_period_seconds": raw_sampling,
    }
    save_json(cfg.dataset_summary_json, summary)
    return summary
