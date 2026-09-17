# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-09-16
# 文件名: data_utils.py
# 功能说明: 将原始飞行记录转换为0.2秒任务规划特征并生成实验4.0独立数据集
# 版本号：4.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories


GRAVITY_MPS2 = 9.80665
METERS_PER_DEGREE_LAT = 110_540.0
METERS_PER_DEGREE_LON = 111_320.0
EXCLUDED_ROUTES = {"A1", "A2", "A3"}
# flight数量不足3条的正常路线无法同时进行路线内训练、验证和测试，完整留作未见路线测试。
UNSEEN_ROUTE_HOLDOUTS = {"R2", "R3", "R4", "R7"}
UNSEEN_WIND_HOLDOUT_MPS = 6.0
UNSEEN_PAYLOAD_HOLDOUT_KG = 0.75
UNSEEN_DURATION_HOLDOUT_S = 300.0

# 只保留规划器能够给出的物理状态；速度模长及其固定步长距离不重复进入模型。
FEATURE_COLUMNS = [
    "time_s", "dt_seconds", "task_progress", "task_duration_s",
    "planned_position_x_m", "planned_position_y_m", "planned_altitude_m",
    "planned_vx_mps", "planned_vy_mps", "planned_vz_mps",
    "planned_ax_mps2", "planned_ay_mps2", "planned_az_mps2",
    "cumulative_distance_m", "cumulative_climb_m", "cumulative_descent_m",
    "wind_speed_mps", "wind_sin", "wind_cos", "relative_air_speed_mps",
    "headwind_mps", "crosswind_mps", "payload_kg",
]

SOURCE_COLUMNS = [
    "flight", "route", "time", "wind_speed", "wind_angle", "battery_voltage", "battery_current",
    "position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z", "speed", "payload", "altitude",
]

FIELD_METADATA = {
    "time_s": {"unit": "s", "meaning": "任务开始后的相对时间", "historical_source": "原始time减去本flight首个time", "runtime_source": "任务规划器时间轴"},
    "dt_seconds": {"unit": "s", "meaning": "固定离散时间步长", "historical_source": "统一重采样为0.2 s", "runtime_source": "任务规划器固定为0.2 s"},
    "task_progress": {"unit": "1", "meaning": "当前规划时刻占任务总时长的比例", "historical_source": "time_s/task_duration_s", "runtime_source": "由完整候选路线时长计算"},
    "task_duration_s": {"unit": "s", "meaning": "完整候选任务的计划总时长", "historical_source": "本flight重采样后的末时刻", "runtime_source": "任务规划器给出的完整路线时长"},
    "planned_position_x_m": {"unit": "m", "meaning": "以任务起点为原点的东向局部位置", "historical_source": "原始经度position_x换算为局部米制坐标", "runtime_source": "规划器ENU东向局部坐标"},
    "planned_position_y_m": {"unit": "m", "meaning": "以任务起点为原点的北向局部位置", "historical_source": "原始纬度position_y换算为局部米制坐标", "runtime_source": "规划器ENU北向局部坐标"},
    "planned_altitude_m": {"unit": "m", "meaning": "计划飞行高度", "historical_source": "原始position_z海拔高度", "runtime_source": "规划器高度轨迹"},
    "planned_heading_deg": {"unit": "deg", "meaning": "机体系X轴相对全局规划坐标的航向；只用于风场坐标统一，不进入TCN", "historical_source": "原始姿态四元数提取航向", "runtime_source": "中层轨迹规划器提供的航向角"},
    "planned_vx_mps": {"unit": "m/s", "meaning": "规划轨迹机体系X轴速度分量", "historical_source": "直接使用原始velocity_x，不使用原始speed设定值", "runtime_source": "中层轨迹规划器逐时刻机体系X轴速度"},
    "planned_vy_mps": {"unit": "m/s", "meaning": "规划轨迹机体系Y轴速度分量", "historical_source": "直接使用原始velocity_y，不使用原始speed设定值", "runtime_source": "中层轨迹规划器逐时刻机体系Y轴速度"},
    "planned_vz_mps": {"unit": "m/s", "meaning": "规划轨迹机体系Z轴速度分量", "historical_source": "直接使用原始velocity_z", "runtime_source": "中层轨迹规划器逐时刻机体系Z轴速度"},
    "planned_ax_mps2": {"unit": "m/s^2", "meaning": "规划轨迹机体系X轴线加速度", "historical_source": "直接使用原始linear_acceleration_x", "runtime_source": "中层轨迹规划器逐时刻机体系X轴无重力加速度"},
    "planned_ay_mps2": {"unit": "m/s^2", "meaning": "规划轨迹机体系Y轴线加速度", "historical_source": "直接使用原始linear_acceleration_y", "runtime_source": "中层轨迹规划器逐时刻机体系Y轴无重力加速度"},
    "planned_az_mps2": {"unit": "m/s^2", "meaning": "规划轨迹机体系剔除重力后的Z轴线加速度", "historical_source": "原始linear_acceleration_z静止时接近-9.80665，逐点加9.80665剔除重力", "runtime_source": "中层轨迹规划器逐时刻机体系Z轴无重力加速度"},
    "segment_distance_m": {"unit": "m", "meaning": "当前0.2 s时间步计划飞行距离", "historical_source": "三轴速度模长乘dt，仅作为轨迹量", "runtime_source": "由规划器三轴速度派生"},
    "cumulative_distance_m": {"unit": "m", "meaning": "从任务起点累计的计划飞行距离", "historical_source": "segment_distance_m累计", "runtime_source": "由完整规划轨迹派生"},
    "cumulative_climb_m": {"unit": "m", "meaning": "任务开始后累计上升高度", "historical_source": "高度正向差分累计", "runtime_source": "由规划高度轨迹派生"},
    "cumulative_descent_m": {"unit": "m", "meaning": "任务开始后累计下降高度", "historical_source": "高度负向差分绝对值累计", "runtime_source": "由规划高度轨迹派生"},
    "wind_speed_mps": {"unit": "m/s", "meaning": "规划时可获得的风速预报", "historical_source": "原始wind_speed传感器记录", "runtime_source": "任务时空位置对应的风预报"},
    "wind_sin": {"unit": "1", "meaning": "机体系横向风单位分量，正值指向机体左侧", "historical_source": "姿态四元数航向减全局wind_angle后取正弦", "runtime_source": "规划航向减全局风预报方向后派生"},
    "wind_cos": {"unit": "1", "meaning": "机体系纵向风单位分量，正值指向机头前方", "historical_source": "姿态四元数航向减全局wind_angle后取余弦", "runtime_source": "规划航向减全局风预报方向后派生"},
    "relative_air_speed_mps": {"unit": "m/s", "meaning": "三轴规划速度与水平风矢量之差的模长", "historical_source": "原始速度和风场派生", "runtime_source": "由规划速度和风预报派生"},
    "headwind_mps": {"unit": "m/s", "meaning": "沿水平规划速度反方向的风分量，正值表示逆风", "historical_source": "速度与风矢量点积派生", "runtime_source": "由规划速度和风预报派生"},
    "crosswind_mps": {"unit": "m/s", "meaning": "垂直于水平规划速度的风分量绝对值", "historical_source": "速度与风矢量叉积派生", "runtime_source": "由规划速度和风预报派生"},
    "payload_kg": {"unit": "kg", "meaning": "任务携带的有效载荷质量", "historical_source": "原始payload由g换算为kg", "runtime_source": "任务要求中的载荷质量"},
}


def save_json(path: Path, payload: dict) -> None:
    """功能: 以UTF-8无BOM保存结构化元数据。
    参数: path为输出路径，payload为待保存字典。
    返回: None。
    调用位置: prepare_dataset及其他4.0模块。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_raw_flights(cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 读取原始记录并排除A1、A2、A3地面辅助测试。
    参数: cfg为实验配置。
    返回: 按flight和真实时间排序的原始字段表。
    调用位置: prepare_dataset。
    """

    if not cfg.raw_flights_csv.exists():
        raise FileNotFoundError(f"未找到原始飞行记录: {cfg.raw_flights_csv}")
    header = pd.read_csv(cfg.raw_flights_csv, nrows=0).columns.tolist()
    missing = sorted(set(SOURCE_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"原始数据缺少4.0所需字段: {missing}")
    frame = pd.read_csv(cfg.raw_flights_csv, usecols=SOURCE_COLUMNS, low_memory=False)
    frame["route"] = frame["route"].astype(str).str.strip()
    frame = frame[~frame["route"].isin(EXCLUDED_ROUTES)].copy()
    frame = _numeric(frame, [c for c in SOURCE_COLUMNS if c not in {"flight", "route"}])
    frame = frame.dropna(subset=[
        "flight", "time", "battery_voltage", "battery_current",
        "velocity_x", "velocity_y", "velocity_z",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    ]).copy()
    return frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)


def _resample_flight(group: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 按真实时间戳把一个flight重采样到0.2 s网格。
    参数: group为单flight原始记录，cfg提供步长。
    返回: 重采样记录和原始记录索引范围映射。
    调用位置: build_planning_frame。
    """

    group = group.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    quaternion_columns = ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
    quaternion = group[quaternion_columns].to_numpy(float, copy=True)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(quaternion)):
        if float(quaternion[index - 1] @ quaternion[index]) < 0.0:
            quaternion[index] *= -1.0
    group.loc[:, quaternion_columns] = quaternion
    start, end = float(group.time.min()), float(group.time.max())
    times = np.arange(start, end + cfg.resample_seconds * 0.5, cfg.resample_seconds)
    source_time = group.time.to_numpy(float)
    output = {"flight": int(group.flight.iloc[0]), "route": str(group.route.iloc[0]), "time": times}
    circular_columns = {"wind_angle"}
    for column in [c for c in group.columns if c not in {"flight", "route", *circular_columns}]:
        output[column] = np.interp(times, source_time, group[column].to_numpy(float))
    # 风向跨越0度时沿最短圆弧插值，避免359度与1度被线性插成180度。
    unwrapped_wind = np.unwrap(np.deg2rad(np.mod(group["wind_angle"].to_numpy(float), 360.0)))
    output["wind_angle"] = np.mod(np.rad2deg(np.interp(times, source_time, unwrapped_wind)), 360.0)
    left = np.maximum(np.searchsorted(source_time, times - cfg.resample_seconds / 2, side="left"), 0)
    right = np.minimum(np.searchsorted(source_time, times + cfg.resample_seconds / 2, side="right") - 1, len(source_time) - 1)
    mapping = pd.DataFrame({"flight": int(group.flight.iloc[0]), "new_time_s": times,
                            "raw_left_index": left, "raw_right_index": right,
                            "raw_time_left_s": source_time[left], "raw_time_right_s": source_time[right],
                            "energy_aggregation": "interpolate voltage/current by timestamp; calculate nonnegative power; integrate over fixed 0.2 s"})
    return pd.DataFrame(output), mapping


def _heading_from_quaternion(group: pd.DataFrame) -> np.ndarray:
    """功能: 从重采样姿态四元数提取以北向为零、顺时针为正的连续航向角。
    参数: group为含orientation_x/y/z/w的单flight表。
    返回: 连续航向角数组，单位rad，仅用于把全局风转换到机体系。
    调用位置: _add_planning_features。
    """

    quaternion = group[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy(float)
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1e-12)
    x, y, z, w = quaternion.T
    yaw_enu = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))
    return np.unwrap(np.pi / 2.0 - yaw_enu)


def _add_planning_features(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 用历史实飞轨迹模拟未来规划器可给出的三轴状态和风场输入。
    参数: frame为0.2 s重采样表，cfg提供固定步长。
    返回: 含规划特征、基础功率标签和总功率标签的数据表。
    调用位置: build_planning_frame。
    """

    outputs = []
    for _, group in frame.groupby("flight", sort=False):
        group = group.sort_values("time").copy()
        dt = float(cfg.resample_seconds)
        group["time_s"] = group.time - float(group.time.iloc[0])
        duration = max(float(group.time_s.iloc[-1]), dt)
        latitude0 = float(group.position_y.iloc[0])
        longitude0 = float(group.position_x.iloc[0])
        longitude_scale = METERS_PER_DEGREE_LON * np.cos(np.deg2rad(latitude0))
        group["planned_position_x_m"] = (group.position_x - longitude0) * longitude_scale
        group["planned_position_y_m"] = (group.position_y - latitude0) * METERS_PER_DEGREE_LAT
        group["planned_altitude_m"] = group.position_z
        # 按用户约定直接使用原始机体系速度和加速度；航向只参与风场坐标统一。
        group["planned_vx_mps"] = group["velocity_x"].to_numpy(float)
        group["planned_vy_mps"] = group["velocity_y"].to_numpy(float)
        group["planned_vz_mps"] = group["velocity_z"].to_numpy(float)
        group["planned_ax_mps2"] = group["linear_acceleration_x"].to_numpy(float)
        group["planned_ay_mps2"] = group["linear_acceleration_y"].to_numpy(float)
        group["planned_az_mps2"] = group["linear_acceleration_z"].to_numpy(float) + GRAVITY_MPS2
        speed_norm = np.sqrt(group.planned_vx_mps**2 + group.planned_vy_mps**2 + group.planned_vz_mps**2)
        group["segment_distance_m"] = speed_norm * dt
        group["cumulative_distance_m"] = group.segment_distance_m.cumsum()
        altitude_delta = group.planned_altitude_m.diff().fillna(0.0)
        group["cumulative_climb_m"] = altitude_delta.clip(lower=0).cumsum()
        group["cumulative_descent_m"] = (-altitude_delta.clip(upper=0)).cumsum()
        group["dt_seconds"] = dt
        group["task_progress"] = np.clip(group.time_s / duration, 0.0, 1.0)
        group["task_duration_s"] = duration
        wind_global_rad = np.unwrap(np.deg2rad(group.wind_angle.fillna(0).to_numpy(float)))
        planned_heading_rad = _heading_from_quaternion(group)
        group["planned_heading_deg"] = np.mod(np.rad2deg(planned_heading_rad), 360.0)
        # 机体系Y轴按ROS FLU约定指向左侧，因此相对角使用“航向-风向”。
        wind_relative_rad = planned_heading_rad - wind_global_rad
        wind_speed = group.wind_speed.clip(lower=0).fillna(0).to_numpy(float)
        # 机体系X轴沿机头、Y轴横向；角度0表示风沿机头方向吹向。
        wind_x = wind_speed * np.cos(wind_relative_rad)
        wind_y = wind_speed * np.sin(wind_relative_rad)
        vx, vy, vz = (group[c].to_numpy(float) for c in ("planned_vx_mps", "planned_vy_mps", "planned_vz_mps"))
        horizontal = np.sqrt(vx**2 + vy**2)
        group["wind_speed_mps"] = wind_speed
        group["wind_sin"] = np.sin(wind_relative_rad)
        group["wind_cos"] = np.cos(wind_relative_rad)
        group["relative_air_speed_mps"] = np.sqrt((vx - wind_x)**2 + (vy - wind_y)**2 + vz**2)
        group["headwind_mps"] = -(vx * wind_x + vy * wind_y) / np.maximum(horizontal, 1e-6)
        group["crosswind_mps"] = np.abs(vx * wind_y - vy * wind_x) / np.maximum(horizontal, 1e-6)
        group["payload_kg"] = group.payload.clip(lower=0).fillna(0) / 1000.0
        # 公开数据没有相机、通信、计算设备的额定功率字段，历史样本只能记录为0。
        group["auxiliary_power_w"] = 0.0
        group["battery_current_discharge_a"] = group.battery_current.clip(lower=0)
        group["power_w"] = (group.battery_voltage * group.battery_current_discharge_a).clip(lower=0)
        group["base_power_w"] = (group.power_w - group.auxiliary_power_w).clip(lower=0)
        group["energy_interval_wh"] = group.power_w * dt / 3600.0
        group["base_energy_interval_wh"] = group.base_power_w * dt / 3600.0
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def build_planning_frame(raw: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 重采样全部正常flight并构造最终规划特征表。
    参数: raw为原始数据，cfg为实验配置。
    返回: 建模表和重采样映射表。
    调用位置: prepare_dataset。
    """

    frames, maps = [], []
    for _, group in raw.groupby("flight", sort=False):
        sampled, mapping = _resample_flight(group, cfg)
        frames.append(sampled)
        maps.append(mapping)
    frame = _add_planning_features(pd.concat(frames, ignore_index=True), cfg)
    columns = ["flight", "route", "time", "power_w", "base_power_w", "auxiliary_power_w",
               "energy_interval_wh", "base_energy_interval_wh", "planned_heading_deg",
               "segment_distance_m", *FEATURE_COLUMNS]
    return frame[columns].copy(), pd.concat(maps, ignore_index=True)


def split_by_flight(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """功能: 按完整flight和route分层划分训练、验证、测试集。
    参数: frame为建模表，cfg提供比例和随机种子。
    返回: train、val、test数据表。
    调用位置: prepare_dataset。
    """

    rng = np.random.default_rng(cfg.random_seed)
    ids = {"train": set(), "val": set(), "test": set()}
    profiles = frame.groupby(["flight", "route"], sort=True).agg(
        mean_wind_speed_mps=("wind_speed_mps", "mean"),
        payload_kg=("payload_kg", "first"),
        task_duration_s=("task_duration_s", "first"),
    ).reset_index()
    forced_test = profiles.loc[
        profiles["route"].isin(UNSEEN_ROUTE_HOLDOUTS)
        | profiles["mean_wind_speed_mps"].ge(UNSEEN_WIND_HOLDOUT_MPS)
        | profiles["payload_kg"].ge(UNSEEN_PAYLOAD_HOLDOUT_KG)
        | profiles["task_duration_s"].gt(UNSEEN_DURATION_HOLDOUT_S),
        "flight",
    ].astype(int)
    ids["test"].update(forced_test.tolist())
    for route, group in frame[["flight", "route"]].drop_duplicates().groupby("route", sort=True):
        flights = group.loc[~group.flight.isin(ids["test"]), "flight"].to_numpy(int, copy=True)
        rng.shuffle(flights)
        if str(route) in UNSEEN_ROUTE_HOLDOUTS:
            continue
        if len(flights) < 3:
            ids["train"].update(map(int, flights)); continue
        n_test = max(1, int(np.ceil(len(flights) * cfg.test_ratio)))
        n_val = max(1, int(np.ceil(len(flights) * cfg.val_ratio)))
        while n_test + n_val >= len(flights):
            n_test = max(1, n_test - 1)
        ids["test"].update(map(int, flights[:n_test]))
        ids["val"].update(map(int, flights[n_test:n_test + n_val]))
        ids["train"].update(map(int, flights[n_test + n_val:]))
    return {name: frame[frame.flight.isin(values)].copy() for name, values in ids.items()}


def _write_feature_analysis(frame: pd.DataFrame, cfg: ExperimentConfig) -> None:
    rows = []
    for column in FEATURE_COLUMNS + ["planned_heading_deg", "segment_distance_m", "auxiliary_power_w"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if column == "planned_heading_deg":
            decision = "仅用于将全局风向转换到机体系；相对风特征派生后不重复进入TCN"
        elif column == "segment_distance_m":
            decision = "保留为路线派生量；固定0.2 s下等于三轴速度模长乘常数，不进入TCN"
        elif column == "auxiliary_power_w":
            decision = "不进入TCN，作为确定附加功率在输出端相加"
        else:
            decision = "保留为规划物理输入"
        rows.append({"field": column, "selected_for_tcn": column in FEATURE_COLUMNS,
                     "unit": FIELD_METADATA.get(column, {"unit": "W"})["unit"],
                     "non_null_count": int(values.notna().sum()), "unique_count": int(values.nunique(dropna=True)),
                     "zero_fraction": float(np.mean(np.isclose(values.fillna(0).to_numpy(float), 0.0))),
                     "mean": float(values.mean()), "std": float(values.std()),
                     "decision": decision})
    rows.extend([
        {"field": "speed", "selected_for_tcn": False, "unit": "m/s", "decision": "删除；它是巡航设定速度且与逐时刻三轴速度不是同一物理量"},
        {"field": "planned_speed_mps", "selected_for_tcn": False, "unit": "m/s", "decision": "删除；可由三轴速度计算，属于重复输入"},
        {"field": "planned_horizontal_speed_mps", "selected_for_tcn": False, "unit": "m/s", "decision": "删除；可由vx和vy计算，属于重复输入"},
        {"field": "planned_acceleration_mps2", "selected_for_tcn": False, "unit": "m/s^2", "decision": "删除；可由三轴加速度计算，属于重复输入"},
        {"field": "four_device_state_fields", "selected_for_tcn": False, "unit": "bool", "decision": "删除；公开数据无对应记录，历史列全为0"},
        {"field": "thermal/vision/communication proxies", "selected_for_tcn": False, "unit": "mixed", "decision": "删除；经验代理不是原始设备额定功率，不能参与精度评价"},
    ])
    pd.DataFrame(rows).to_csv(cfg.feature_analysis_csv, index=False, encoding="utf-8")


def prepare_dataset(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 生成实验4.0独立数据、逐维元数据、关系分析和flight切分。
    参数: cfg为配置，force控制是否强制重建。
    返回: 数据规模和输入维度摘要。
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
    _write_feature_analysis(frame, cfg)
    raw_deltas = raw.groupby("flight", sort=False).time.diff().dropna()
    raw_sampling = {"median_seconds": float(raw_deltas.median()), "min_seconds": float(raw_deltas.min()),
                    "max_seconds": float(raw_deltas.max()), "sample_count": int(len(raw_deltas))}
    metadata = {
        "version": "4.0", "feature_count": len(FEATURE_COLUMNS), "feature_columns": FEATURE_COLUMNS,
        "target_fields": {"base_power_w": "TCN监督目标；历史auxiliary_power_w为0",
                          "power_w": "仅用于离线总功率标签和评估",
                          "energy_interval_wh": "power_w * 0.2 / 3600 Wh"},
        "interface_only_fields": {
            "task_id": {"unit": "string", "meaning": "上层调度器分配的任务标识", "enters_tcn": False},
            "route": {"unit": "string", "meaning": "来源追踪和分路线统计字段", "enters_tcn": False},
            "wind_direction_deg": {"unit": "deg", "meaning": "全局规划坐标中的风矢量指向角；气象学来向必须先加180度", "enters_tcn": False},
            "planned_heading_deg": {"unit": "deg", "meaning": "机体系X轴相对全局规划坐标的航向；只用于把风矢量转换到机体系，不进入TCN", "enters_tcn": False},
            "wind_forecast_source": {"unit": "string", "meaning": "风预报发布机构或服务标识", "enters_tcn": False},
            "wind_forecast_timestamp": {"unit": "ISO-8601", "meaning": "风预报发布时间或有效时刻", "enters_tcn": False},
            "wind_speed_error_mps": {"unit": "m/s", "meaning": "风速预报误差范围的非负半宽", "enters_tcn": False},
            "wind_direction_error_deg": {"unit": "deg", "meaning": "风向预报误差范围的非负半宽", "enters_tcn": False},
            "wind_coordinate_system": {"unit": "enum", "meaning": "风向和规划航向共同采用的坐标基准", "enters_tcn": False},
            "wind_direction_convention": {"unit": "enum", "meaning": "VECTOR_TO或METEOROLOGICAL_FROM；接口统一转换为风矢量吹向", "enters_tcn": False},
            "segment_distance_m": {"unit": "m", "meaning": "三轴规划速度模长乘固定0.2 s；保留为路线距离派生量，与速度分量重复，不进入TCN", "enters_tcn": False},
            "auxiliary_power_w": {"unit": "W", "meaning": "规划阶段已知设备额定功率之和；不进入TCN，确定性加到基础动力功率预测", "enters_tcn": False},
        },
        "feature_dimensions": [
            {"dimension": index, "field": name, **FIELD_METADATA[name],
             "task_before_available": True, "flight_only_observation": False,
             "enters_tcn": True}
            for index, name in enumerate(FEATURE_COLUMNS, start=1)
        ],
        "field_metadata": {name: {**FIELD_METADATA[name], "task_before_available": True,
                                  "flight_only_observation": False, "enters_tcn": True}
                           for name in FEATURE_COLUMNS},
        "coordinate_system": {"position": "位置采用任务起点局部ENU坐标：X向东、Y向北、Z为高度",
                              "motion": "速度和加速度按用户要求直接使用原始机体系X/Y/Z分量；历史Z加速度加9.80665剔除重力",
                              "wind_direction_convention": "接口角度表示风矢量吹向；若预报源给出气象学风来向，接入层必须加180度后再归一化到0至360度",
                              "historical_wind_reference_limit": "公开数据只说明wind_angle相对north，未说明真北或磁北；部署接入时必须先把预报风向转换到规划器ENU坐标，不能直接混用地理、磁北或机体系角度",
                              "runtime_requirement": "规划器须提供全局风向和planned_heading_deg；接口先求相对风角，再与机体系速度计算相对空速、逆风和侧风"},
        "sampling": {"raw_period_seconds": raw_sampling, "resample_seconds": cfg.resample_seconds,
                     "mapping_file": str(cfg.resample_map_csv),
                     "energy_aggregation": "电压和电流先按时间戳插值；电流负值截为0后计算非负功率，再乘0.2/3600得到Wh",
                     "wind_angle_interpolation": "对原始风向角先做2π圆周展开，再按真实时间戳插值并映射回0至360度"},
        "excluded_routes": sorted(EXCLUDED_ROUTES),
        "unseen_route_test_routes": sorted(UNSEEN_ROUTE_HOLDOUTS),
        "unseen_condition_holdouts": {
            "mean_wind_speed_mps_greater_or_equal": UNSEEN_WIND_HOLDOUT_MPS,
            "payload_kg_greater_or_equal": UNSEEN_PAYLOAD_HOLDOUT_KG,
            "task_duration_s_greater_than": UNSEEN_DURATION_HOLDOUT_S,
            "selection_uses_power_label": False,
        },
        "removed_inputs": ["speed", "planned_speed_mps", "planned_horizontal_speed_mps", "planned_acceleration_mps2", "segment_distance_m",
                           "payload_camera_enabled", "payload_communication_enabled", "payload_compute_enabled",
                           "payload_delivery_enabled", "payload_rated_power_w", "thermal_load_proxy",
                           "vision_energy_proxy_w", "communication_energy_proxy_w", "route one-hot"],
        "historical_to_planning_assumption": "离线训练把历史飞行的原始机体系速度、去重力加速度、位置、姿态航向和环境记录视为当时规划轨迹的可观测替代。部署时由上层调度和中层轨迹规划器提供同字段。",
    }
    save_json(cfg.feature_meta_json, metadata)
    summary = {"version": "4.0", "resample_seconds": cfg.resample_seconds, "processed_rows": int(len(frame)),
               "flights": int(frame.flight.nunique()), "train_rows": int(len(splits["train"])),
               "val_rows": int(len(splits["val"])), "test_rows": int(len(splits["test"])),
               "train_flights": int(splits["train"].flight.nunique()), "val_flights": int(splits["val"].flight.nunique()),
               "test_flights": int(splits["test"].flight.nunique()), "feature_count": len(FEATURE_COLUMNS),
               "routes": sorted(frame.route.unique().tolist()), "excluded_routes": sorted(EXCLUDED_ROUTES),
               "unseen_route_test_routes": sorted(UNSEEN_ROUTE_HOLDOUTS),
               "unseen_route_test_flights": int(splits["test"].loc[splits["test"].route.isin(UNSEEN_ROUTE_HOLDOUTS), "flight"].nunique()),
               "unseen_wind_test_flights": int(splits["test"].loc[
                   splits["test"].groupby("flight")["wind_speed_mps"].transform("mean").ge(UNSEEN_WIND_HOLDOUT_MPS), "flight"].nunique()),
               "unseen_payload_test_flights": int(splits["test"].loc[
                   splits["test"]["payload_kg"].ge(UNSEEN_PAYLOAD_HOLDOUT_KG), "flight"].nunique()),
               "unseen_duration_test_flights": int(splits["test"].loc[
                   splits["test"]["task_duration_s"].gt(UNSEEN_DURATION_HOLDOUT_S), "flight"].nunique()),
               "raw_sampling_period_seconds": raw_sampling, "auxiliary_power_historical_value_w": 0.0}
    save_json(cfg.dataset_summary_json, summary)
    return summary
