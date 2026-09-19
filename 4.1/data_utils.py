# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-09-17
# 文件名: data_utils.py
# 功能说明: 从Source压缩包构造实验4.1的10维任务前输入、监督目标和审计字段
# 版本号：4.1

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories


METERS_PER_DEGREE_LAT = 110_540.0
METERS_PER_DEGREE_LON = 111_320.0
EXCLUDED_ROUTES = {"A1", "A2", "A3"}
UNSEEN_ROUTE_HOLDOUTS = {"R2", "R3", "R4", "R7"}
UNSEEN_WIND_HOLDOUT_MPS = 6.0
UNSEEN_PAYLOAD_HOLDOUT_G = 750.0
UNSEEN_PAYLOAD_HOLDOUT_KG = UNSEEN_PAYLOAD_HOLDOUT_G / 1000.0
UNSEEN_DURATION_HOLDOUT_S = 300.0
MOTOR_ON_POWER_THRESHOLD_W = 100.0
AIRBORNE_HORIZONTAL_THRESHOLD_M = 1.0
AIRBORNE_VERTICAL_THRESHOLD_M = 0.5
ENERGY_TOLERANCE_PERCENT = 0.1
STATUS_LABEL_METHOD = (
    "motor_on:原始非负电功率>=100W的首末证据间连续标注;"
    "airborne:相对起点水平位移>=1m或垂直位移绝对值>=0.5m的首末证据间连续标注;"
    "强制airborne<=motor_on"
)

FEATURE_COLUMNS = [
    "time_s",
    "task_duration_s",
    "planned_position_east_m",
    "planned_position_north_m",
    "planned_position_up_m",
    "wind_east_mps",
    "wind_north_mps",
    "payload_g",
    "planned_motor_on",
    "planned_airborne",
]

TARGET_COLUMNS = ["power_w", "energy_interval_wh"]
AUDIT_COLUMNS = [
    "flight",
    "route",
    "split",
    "split_role",
    "time",
    "dt_seconds",
    "status_label_method",
    "status_quality_flag",
    "source_sample_count",
    "raw_energy_wh",
    "resampled_energy_wh",
    "resample_energy_error_percent",
]

SOURCE_COLUMNS = [
    "flight",
    "route",
    "time",
    "wind_speed",
    "wind_angle",
    "battery_voltage",
    "battery_current",
    "position_x",
    "position_y",
    "position_z",
    "payload",
]

FIELD_METADATA = {
    "time_s": {
        "unit": "s",
        "meaning": "任务开始后的相对时间",
        "historical_source": "原始time减flight首时刻",
        "runtime_source": "规划时间轴",
    },
    "task_duration_s": {
        "unit": "s",
        "meaning": "完整任务计划时长",
        "historical_source": "当前flight最后真实时刻减首时刻",
        "runtime_source": "规划轨迹末时刻",
    },
    "planned_position_east_m": {
        "unit": "m",
        "meaning": "相对任务起点的ENU东向位置",
        "historical_source": "原始经度position_x按起点纬度换算",
        "runtime_source": "轨迹规划器",
    },
    "planned_position_north_m": {
        "unit": "m",
        "meaning": "相对任务起点的ENU北向位置",
        "historical_source": "原始纬度position_y换算",
        "runtime_source": "轨迹规划器",
    },
    "planned_position_up_m": {
        "unit": "m",
        "meaning": "相对任务起点的ENU上向位置",
        "historical_source": "原始position_z减flight起点高度",
        "runtime_source": "规划高度减任务起点高度",
    },
    "wind_east_mps": {
        "unit": "m/s",
        "meaning": "风矢量的ENU东向分量",
        "historical_source": "wind_speed乘wind_angle正弦",
        "runtime_source": "风场预报转换",
    },
    "wind_north_mps": {
        "unit": "m/s",
        "meaning": "风矢量的ENU北向分量",
        "historical_source": "wind_speed乘wind_angle余弦",
        "runtime_source": "风场预报转换",
    },
    "payload_g": {
        "unit": "g",
        "meaning": "任务有效载荷质量",
        "historical_source": "原始payload",
        "runtime_source": "调度器任务信息",
    },
    "planned_motor_on": {
        "unit": "0/1",
        "meaning": "计划电机运行状态",
        "historical_source": "原始功率阈值和连续阶段分析",
        "runtime_source": "飞控任务状态或起降计划",
    },
    "planned_airborne": {
        "unit": "0/1",
        "meaning": "计划离地状态",
        "historical_source": "相对起点位移阈值和连续阶段分析",
        "runtime_source": "起飞、降落和航段状态",
    },
}


def save_json(path: Path, payload: dict) -> None:
    """功能: 以UTF-8无BOM保存结构化数据。
    参数: path为输出路径，payload为待保存字典。
    返回: None。
    调用位置: prepare_dataset及评估模块。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """功能: 将指定原始字段转为数值并保留无法转换项为缺失值。
    参数: frame为原始表，columns为字段名。
    返回: 转换后的副本。
    调用位置: load_raw_flights。
    """

    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_raw_flights(cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 直接读取Source归档中的flights.csv并完成排序去重。
    参数: cfg为实验配置。
    返回: 排除地面辅助路线后按flight和真实time排序的原始表。
    调用位置: prepare_dataset。
    """

    if not cfg.source_archive.exists():
        raise FileNotFoundError(f"未找到Source原始归档: {cfg.source_archive}")
    with zipfile.ZipFile(cfg.source_archive) as archive:
        if "flights.csv" not in archive.namelist():
            raise FileNotFoundError(f"Source归档内缺少flights.csv: {cfg.source_archive}")
        with archive.open("flights.csv") as stream:
            header = pd.read_csv(stream, nrows=0).columns.tolist()
        missing = sorted(set(SOURCE_COLUMNS) - set(header))
        if missing:
            raise ValueError(f"原始数据缺少4.1所需字段: {missing}")
        with archive.open("flights.csv") as stream:
            frame = pd.read_csv(stream, usecols=SOURCE_COLUMNS, low_memory=False)
    frame["route"] = frame["route"].astype(str).str.strip()
    frame = frame.loc[~frame["route"].isin(EXCLUDED_ROUTES)].copy()
    numeric = [column for column in SOURCE_COLUMNS if column != "route"]
    frame = _numeric(frame, numeric).dropna(subset=numeric)
    frame = frame.sort_values(["flight", "time"], kind="stable")
    frame = frame.drop_duplicates(["flight", "time"], keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Source原始数据清洗后为空")
    return frame


def _continuous_stage(evidence: np.ndarray) -> np.ndarray:
    """功能: 将离散证据转换为首末证据之间的连续任务阶段。
    参数: evidence为布尔证据数组。
    返回: 仅含0和1的整型状态数组。
    调用位置: _resample_flight。
    """

    result = np.zeros(len(evidence), dtype=np.int8)
    indices = np.flatnonzero(evidence)
    if len(indices):
        result[indices[0]:indices[-1] + 1] = 1
    return result


def _raw_energy_wh(time_s: np.ndarray, power_w: np.ndarray) -> float:
    """功能: 按真实时间用梯形积分计算单flight原始总能耗。
    参数: time_s为真实时间，power_w为非负功率。
    返回: 总能耗，单位Wh。
    调用位置: _resample_flight。
    """

    if len(time_s) < 2:
        return 0.0
    return float(np.trapezoid(power_w, time_s) / 3600.0)


def _resample_flight(group: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 将单flight映射到固定0.2秒网格并保持总能耗。
    参数: group为已排序去重的原始flight，cfg提供采样周期。
    返回: 10维输入、监督目标、审计字段和重采样映射。
    调用位置: build_planning_frame。
    """

    group = group.sort_values("time", kind="stable").drop_duplicates("time", keep="last").reset_index(drop=True)
    source_time = group["time"].to_numpy(float)
    start = float(source_time[0])
    relative_source_time = source_time - start
    duration = float(relative_source_time[-1])
    if duration <= 0.0:
        raise ValueError(f"flight {group.flight.iloc[0]} 的真实时间跨度必须大于0")
    dt = float(cfg.resample_seconds)
    row_count = max(1, int(np.ceil(duration / dt)))
    time_s = np.arange(row_count, dtype=float) * dt
    absolute_time = start + time_s

    longitude0 = float(group["position_x"].iloc[0])
    latitude0 = float(group["position_y"].iloc[0])
    altitude0 = float(group["position_z"].iloc[0])
    longitude_scale = METERS_PER_DEGREE_LON * np.cos(np.deg2rad(latitude0))
    east_source = (group["position_x"].to_numpy(float) - longitude0) * longitude_scale
    north_source = (group["position_y"].to_numpy(float) - latitude0) * METERS_PER_DEGREE_LAT
    up_source = group["position_z"].to_numpy(float) - altitude0

    wind_speed = np.clip(group["wind_speed"].to_numpy(float), 0.0, None)
    wind_angle = np.unwrap(np.deg2rad(np.mod(group["wind_angle"].to_numpy(float), 360.0)))
    wind_east_source = wind_speed * np.sin(wind_angle)
    wind_north_source = wind_speed * np.cos(wind_angle)
    raw_power = np.clip(
        group["battery_voltage"].to_numpy(float) * group["battery_current"].to_numpy(float),
        0.0,
        None,
    )

    motor_source = _continuous_stage(raw_power >= MOTOR_ON_POWER_THRESHOLD_W)
    airborne_evidence = (
        np.hypot(east_source, north_source) >= AIRBORNE_HORIZONTAL_THRESHOLD_M
    ) | (np.abs(up_source) >= AIRBORNE_VERTICAL_THRESHOLD_M)
    airborne_source = _continuous_stage(airborne_evidence) * motor_source

    output = pd.DataFrame({
        "flight": int(group["flight"].iloc[0]),
        "route": str(group["route"].iloc[0]),
        "time": absolute_time,
        "time_s": time_s,
        "task_duration_s": duration,
        "planned_position_east_m": np.interp(time_s, relative_source_time, east_source),
        "planned_position_north_m": np.interp(time_s, relative_source_time, north_source),
        "planned_position_up_m": np.interp(time_s, relative_source_time, up_source),
        "wind_east_mps": np.interp(time_s, relative_source_time, wind_east_source),
        "wind_north_mps": np.interp(time_s, relative_source_time, wind_north_source),
        "payload_g": np.interp(time_s, relative_source_time, np.clip(group["payload"], 0.0, None)),
        "planned_motor_on": np.rint(np.interp(time_s, relative_source_time, motor_source)).astype(np.int8),
        "planned_airborne": np.rint(np.interp(time_s, relative_source_time, airborne_source)).astype(np.int8),
    })
    output["planned_airborne"] = np.minimum(output["planned_airborne"], output["planned_motor_on"])
    interpolated_power = np.interp(time_s, relative_source_time, raw_power)
    raw_energy = _raw_energy_wh(relative_source_time, raw_power)
    preliminary_energy = float(interpolated_power.sum() * dt / 3600.0)
    scale = raw_energy / preliminary_energy if preliminary_energy > 0.0 else 0.0
    output["power_w"] = np.clip(interpolated_power * scale, 0.0, None)
    output["dt_seconds"] = dt
    output["energy_interval_wh"] = output["power_w"] * dt / 3600.0
    resampled_energy = float(output["energy_interval_wh"].sum())
    error_percent = abs(resampled_energy - raw_energy) / max(abs(raw_energy), 1e-12) * 100.0

    if not motor_source.any():
        quality = "no_motor_on_evidence"
    elif not airborne_source.any():
        quality = "no_airborne_evidence"
    else:
        quality = "ok"
    output["status_label_method"] = STATUS_LABEL_METHOD
    output["status_quality_flag"] = quality
    output["source_sample_count"] = len(group)
    output["raw_energy_wh"] = raw_energy
    output["resampled_energy_wh"] = resampled_energy
    output["resample_energy_error_percent"] = error_percent

    left = np.maximum(np.searchsorted(relative_source_time, time_s - dt / 2.0, side="left"), 0)
    right = np.minimum(
        np.searchsorted(relative_source_time, time_s + dt / 2.0, side="right") - 1,
        len(relative_source_time) - 1,
    )
    mapping = pd.DataFrame({
        "flight": int(group["flight"].iloc[0]),
        "new_time_s": time_s,
        "raw_left_index": left,
        "raw_right_index": right,
        "raw_time_left_s": relative_source_time[left],
        "raw_time_right_s": relative_source_time[right],
        "energy_aggregation": "真实时间梯形积分后按flight缩放0.2s功率，使区间能耗总和守恒",
    })
    return output, mapping


def build_planning_frame(raw: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能: 独立重采样每个flight并合并4.1加工表。
    参数: raw为原始数据，cfg为实验配置。
    返回: 未切分的加工表和原始索引映射表。
    调用位置: prepare_dataset。
    """

    frames: list[pd.DataFrame] = []
    mappings: list[pd.DataFrame] = []
    for _, group in raw.groupby("flight", sort=False):
        sampled, mapping = _resample_flight(group, cfg)
        frames.append(sampled)
        mappings.append(mapping)
    return pd.concat(frames, ignore_index=True), pd.concat(mappings, ignore_index=True)


def _split_assignments(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 仅按完整flight和任务前条件生成训练、验证、测试归属。
    参数: frame为4.1加工表，cfg提供比例和随机种子。
    返回: 每个flight的split与split_role表。
    调用位置: split_by_flight。
    """

    profiles = frame.groupby(["flight", "route"], sort=True).agg(
        mean_wind_mps=("wind_east_mps", lambda values: 0.0),
        payload_g=("payload_g", "first"),
        task_duration_s=("task_duration_s", "first"),
    ).reset_index()
    wind = frame.assign(
        wind_mps=np.hypot(frame["wind_east_mps"], frame["wind_north_mps"]),
    ).groupby("flight", sort=True)["wind_mps"].mean()
    profiles["mean_wind_mps"] = profiles["flight"].map(wind)
    forced = (
        profiles["route"].isin(UNSEEN_ROUTE_HOLDOUTS)
        | profiles["mean_wind_mps"].ge(UNSEEN_WIND_HOLDOUT_MPS)
        | profiles["payload_g"].ge(UNSEEN_PAYLOAD_HOLDOUT_G)
        | profiles["task_duration_s"].gt(UNSEEN_DURATION_HOLDOUT_S)
    )
    assignments: dict[int, tuple[str, str]] = {
        int(row.flight): ("test", "forced_unseen_condition")
        for row in profiles.loc[forced].itertuples()
    }
    rng = np.random.default_rng(cfg.random_seed)
    remaining = profiles.loc[~profiles["flight"].isin(assignments)]
    for route, group in remaining.groupby("route", sort=True):
        flights = group["flight"].to_numpy(int, copy=True)
        rng.shuffle(flights)
        if len(flights) < 3:
            for flight in flights:
                assignments[int(flight)] = ("train", "route_small_train")
            continue
        n_test = max(1, int(np.ceil(len(flights) * cfg.test_ratio)))
        n_val = max(1, int(np.ceil(len(flights) * cfg.val_ratio)))
        while n_test + n_val >= len(flights):
            n_test = max(1, n_test - 1)
        for flight in flights[:n_test]:
            assignments[int(flight)] = ("test", "random_route_test")
        for flight in flights[n_test:n_test + n_val]:
            assignments[int(flight)] = ("val", "random_route_validation")
        for flight in flights[n_test + n_val:]:
            assignments[int(flight)] = ("train", "random_route_train")
    rows = [
        {"flight": flight, "split": split, "split_role": role}
        for flight, (split, role) in sorted(assignments.items())
    ]
    return pd.DataFrame(rows)


def split_by_flight(frame: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """功能: 为全量表写入追踪字段并按完整flight输出三个集合。
    参数: frame为未切分加工表，cfg为实验配置。
    返回: 含切分审计字段的全量表及train、val、test子表。
    调用位置: prepare_dataset。
    """

    assignments = _split_assignments(frame, cfg)
    tagged = frame.merge(assignments, on="flight", how="left", validate="many_to_one")
    if tagged[["split", "split_role"]].isna().any().any():
        raise RuntimeError("存在未分配到训练、验证或测试集的flight")
    ordered = AUDIT_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
    tagged = tagged[ordered].sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    splits = {name: tagged.loc[tagged["split"].eq(name)].copy() for name in ("train", "val", "test")}
    return tagged, splits


def validate_processed_data(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 检查4.1字段、数值、时间、状态、能耗和flight边界。
    参数: frame为完整加工表，cfg提供采样周期。
    返回: 可写入数据摘要的一致性检查结果。
    调用位置: prepare_dataset及测试。
    """

    required = set(AUDIT_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"4.1加工数据缺少字段: {missing}")
    numeric_columns = FEATURE_COLUMNS + TARGET_COLUMNS + [
        "time", "dt_seconds", "source_sample_count", "raw_energy_wh",
        "resampled_energy_wh", "resample_energy_error_percent",
    ]
    numeric = frame[numeric_columns].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("4.1加工数据包含NaN或Inf")
    if frame.duplicated(["flight", "time_s"]).any():
        raise ValueError("4.1加工数据存在重复flight/time_s")
    deltas = frame.groupby("flight", sort=False)["time_s"].diff().dropna().to_numpy(float)
    if len(deltas) and not np.allclose(deltas, cfg.resample_seconds, rtol=0.0, atol=1e-9):
        raise ValueError("4.1加工数据存在非0.2秒间隔或跨flight差分")
    for column in ("planned_motor_on", "planned_airborne"):
        values = set(frame[column].astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column}必须严格为0或1")
    if (frame["planned_airborne"] > frame["planned_motor_on"]).any():
        raise ValueError("planned_airborne不能大于planned_motor_on")
    interval_expected = frame["power_w"] * frame["dt_seconds"] / 3600.0
    if not np.allclose(frame["energy_interval_wh"], interval_expected, rtol=1e-10, atol=1e-12):
        raise ValueError("energy_interval_wh与power_w、dt_seconds不一致")
    flight_energy = frame.groupby("flight", sort=True).agg(
        raw_energy_wh=("raw_energy_wh", "first"),
        resampled_energy_wh=("energy_interval_wh", "sum"),
    )
    errors = (
        (flight_energy["resampled_energy_wh"] - flight_energy["raw_energy_wh"]).abs()
        / flight_energy["raw_energy_wh"].abs().clip(lower=1e-12)
        * 100.0
    )
    maximum_error = float(errors.max())
    if maximum_error > ENERGY_TOLERANCE_PERCENT:
        raise ValueError(
            f"重采样前后flight总能耗最大误差{maximum_error:.6f}%超过"
            f"{ENERGY_TOLERANCE_PERCENT:.3f}%容差"
        )
    if "route" in FEATURE_COLUMNS:
        raise ValueError("route不得进入TCN输入")
    return {
        "finite_values": True,
        "duplicate_flight_time_rows": 0,
        "fixed_step_seconds": float(cfg.resample_seconds),
        "cross_flight_difference_used": False,
        "binary_state_constraints_passed": True,
        "airborne_le_motor_on": True,
        "route_enters_model": False,
        "energy_tolerance_percent": ENERGY_TOLERANCE_PERCENT,
        "maximum_flight_energy_error_percent": maximum_error,
    }


def _write_feature_analysis(frame: pd.DataFrame, cfg: ExperimentConfig) -> None:
    """功能: 记录10维输入、监督目标、审计字段和禁用字段的选择理由。
    参数: frame为4.1完整加工表，cfg为实验配置。
    返回: None。
    调用位置: prepare_dataset。
    """

    rows: list[dict] = []
    for column in FEATURE_COLUMNS + TARGET_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append({
            "field": column,
            "category": "model_input" if column in FEATURE_COLUMNS else "supervision_target",
            "selected_for_tcn": column in FEATURE_COLUMNS,
            "unit": FIELD_METADATA.get(column, {"unit": "W" if column == "power_w" else "Wh"})["unit"],
            "non_null_count": int(values.notna().sum()),
            "unique_count": int(values.nunique(dropna=True)),
            "zero_fraction": float(np.mean(np.isclose(values.to_numpy(float), 0.0))),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "decision": "保留为TCN任务前输入" if column in FEATURE_COLUMNS else "只作监督和评估，不进入TCN",
        })
    forbidden = [
        "battery_voltage", "battery_current", "velocity_x/y/z", "linear_acceleration_x/y/z",
        "orientation_x/y/z/w", "speed", "legacy_planning_derivatives",
        "legacy_wind_derivatives", "route",
    ]
    rows.extend({
        "field": field,
        "category": "forbidden_or_audit",
        "selected_for_tcn": False,
        "decision": "真实飞行观测、旧派生字段或追踪标识，禁止进入4.1任务前模型",
    } for field in forbidden)
    pd.DataFrame(rows).to_csv(cfg.feature_analysis_csv, index=False, encoding="utf-8")


def prepare_dataset(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 从Source生成实验4.1独立加工数据、切分、元数据和一致性摘要。
    参数: cfg为实验配置，force控制是否强制重建。
    返回: 数据规模、状态分布和一致性检查摘要。
    调用位置: main.py的prepare和all模式。
    """

    ensure_directories(cfg)
    if cfg.clean_data_csv.exists() and cfg.dataset_summary_json.exists() and not force:
        return json.loads(cfg.dataset_summary_json.read_text(encoding="utf-8"))
    raw = load_raw_flights(cfg)
    untagged, mapping = build_planning_frame(raw, cfg)
    frame, splits = split_by_flight(untagged, cfg)
    checks = validate_processed_data(frame, cfg)
    frame.to_csv(cfg.clean_data_csv, index=False, encoding="utf-8")
    mapping.to_csv(cfg.resample_map_csv, index=False, encoding="utf-8")
    for name, table in splits.items():
        table.to_csv(getattr(cfg, f"{name}_csv"), index=False, encoding="utf-8")
    _write_feature_analysis(frame, cfg)

    metadata = {
        "version": "4.1",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_fields": {
            "power_w": "重采样后按flight能耗守恒校准的监督功率",
            "energy_interval_wh": "power_w乘0.2秒再除以3600的区间能耗",
        },
        "audit_fields": AUDIT_COLUMNS,
        "feature_dimensions": [
            {"dimension": index, "field": name, **FIELD_METADATA[name], "enters_tcn": True}
            for index, name in enumerate(FEATURE_COLUMNS, start=1)
        ],
        "field_metadata": FIELD_METADATA,
        "source": {
            "directory": str(cfg.source_dir),
            "archive": str(cfg.source_archive),
            "archive_member": "flights.csv",
        },
        "coordinate_system": {
            "position": "任务起点局部ENU：east向东、north向北、up向上",
            "wind": "wind_angle按相对北向顺时针角解释；east=speed*sin(angle)，north=speed*cos(angle)",
        },
        "status_labeling": {
            "method": STATUS_LABEL_METHOD,
            "motor_on_power_threshold_w": MOTOR_ON_POWER_THRESHOLD_W,
            "airborne_horizontal_threshold_m": AIRBORNE_HORIZONTAL_THRESHOLD_M,
            "airborne_vertical_threshold_m": AIRBORNE_VERTICAL_THRESHOLD_M,
            "runtime_requirement": "任务接口直接接收飞控任务状态或起降计划，不接收真实功率",
        },
        "resampling": {
            "step_seconds": cfg.resample_seconds,
            "energy_method": "原始真实时间梯形积分；重采样功率按flight缩放后生成区间能耗",
            "flight_energy_tolerance_percent": ENERGY_TOLERANCE_PERCENT,
        },
        "forbidden_task_inputs": [
            "power_w", "energy_interval_wh", "battery_voltage", "battery_current",
            "velocity_x", "velocity_y", "velocity_z", "linear_acceleration_x",
            "linear_acceleration_y", "linear_acceleration_z", "orientation_x",
            "orientation_y", "orientation_z", "orientation_w", "speed", "route",
        ],
        "split_policy": {
            "unit": "完整flight",
            "random_seed": cfg.random_seed,
            "unseen_route_test_routes": sorted(UNSEEN_ROUTE_HOLDOUTS),
            "unseen_wind_threshold_mps": UNSEEN_WIND_HOLDOUT_MPS,
            "unseen_payload_threshold_g": UNSEEN_PAYLOAD_HOLDOUT_G,
            "unseen_duration_threshold_s": UNSEEN_DURATION_HOLDOUT_S,
            "selection_uses_supervision_target": False,
        },
    }
    save_json(cfg.feature_meta_json, metadata)

    raw_deltas = raw.groupby("flight", sort=False)["time"].diff().dropna()
    status_counts = frame.groupby(["route", "planned_motor_on", "planned_airborne"], sort=True).size()
    summary = {
        "version": "4.1",
        "resample_seconds": cfg.resample_seconds,
        "processed_rows": int(len(frame)),
        "flights": int(frame["flight"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "routes": sorted(frame["route"].unique().tolist()),
        "excluded_routes": sorted(EXCLUDED_ROUTES),
        "train_rows": int(len(splits["train"])),
        "val_rows": int(len(splits["val"])),
        "test_rows": int(len(splits["test"])),
        "train_flights": int(splits["train"]["flight"].nunique()),
        "val_flights": int(splits["val"]["flight"].nunique()),
        "test_flights": int(splits["test"]["flight"].nunique()),
        "raw_sampling": {
            "median_seconds": float(raw_deltas.median()),
            "min_seconds": float(raw_deltas.min()),
            "max_seconds": float(raw_deltas.max()),
            "sample_count": int(len(raw_deltas)),
        },
        "status_quality_flights": frame.groupby("status_quality_flag")["flight"].nunique().to_dict(),
        "status_combination_rows": {
            f"{route}|motor={motor}|airborne={airborne}": int(count)
            for (route, motor, airborne), count in status_counts.items()
        },
        "consistency_checks": checks,
    }
    save_json(cfg.dataset_summary_json, summary)
    return summary
