# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-07-07
# 文件名: data_utils.py
# 功能说明: 下载无人机公开数据集并完成实验2.0的能耗特征工程、数据清洗和数据切分
# 版本号：2.0

import json
import math
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ensure_directories
from progress import TerminalProgress


RAW_COLUMNS = [
    "flight",
    "time",
    "wind_speed",
    "wind_angle",
    "battery_voltage",
    "battery_current",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_x",
    "angular_y",
    "angular_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "speed",
    "payload",
    "altitude",
    "route",
]

NUMERIC_SOURCE_COLUMNS = [
    "time",
    "wind_speed",
    "wind_angle",
    "battery_voltage",
    "battery_current",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_x",
    "angular_y",
    "angular_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "speed",
    "payload",
    "altitude",
]

ENGINEERED_NUMERIC_FEATURES = [
    "time",
    "flight_progress",
    "wind_speed",
    "wind_sin",
    "wind_cos",
    "programmed_speed_mps",
    "actual_speed_mps",
    "horizontal_speed_mps",
    "vertical_speed_mps",
    "vertical_speed_abs_mps",
    "relative_air_speed_mps",
    "wind_alignment",
    "wind_cross_component_mps",
    "payload_kg",
    "altitude_m",
    "dynamic_accel_norm",
    "angular_rate_norm",
    "obstacle_agility_index",
    "thermal_load_proxy",
    "vision_energy_proxy_w",
    "communication_energy_proxy_w",
]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """功能: 执行外部命令并在失败时抛出带上下文的异常。
    参数: command为命令列表，cwd为执行目录。
    返回: None。
    调用位置: download_source_dataset。
    """

    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"命令执行失败: {' '.join(command)}\n{message}")


def download_source_dataset(cfg: ExperimentConfig) -> None:
    """功能: 从ModelScope公开镜像下载DJI Matrice 100能耗数据集。
    参数: cfg为实验配置对象。
    返回: None。
    调用位置: prepare_dataset。
    """

    ensure_directories(cfg)
    if cfg.raw_zip_file.exists():
        return
    if cfg.source_repo_dir.exists() and not cfg.raw_zip_file.exists():
        raise FileNotFoundError(f"数据仓库已存在但未找到压缩包: {cfg.raw_zip_file}")
    run_command(
        ["git", "clone", "--depth", "1", cfg.source_repo_url, str(cfg.source_repo_dir)],
        cwd=cfg.data_dir,
    )


def extract_source_zip(cfg: ExperimentConfig) -> None:
    """功能: 从源压缩包中抽取建模所需的CSV和说明文件。
    参数: cfg为实验配置对象。
    返回: None。
    调用位置: prepare_dataset。
    """

    ensure_directories(cfg)
    if cfg.raw_flights_csv.exists() and cfg.raw_parameters_csv.exists() and cfg.raw_readme_file.exists():
        return
    if not cfg.raw_zip_file.exists():
        raise FileNotFoundError(f"未找到源数据压缩包: {cfg.raw_zip_file}")
    with zipfile.ZipFile(cfg.raw_zip_file) as archive:
        for name in ("flights.csv", "parameters.csv", "README.txt"):
            archive.extract(name, cfg.raw_data_dir)


def validate_raw_files(cfg: ExperimentConfig) -> None:
    """功能: 校验原始CSV文件是否齐全。
    参数: cfg为实验配置对象。
    返回: None。
    调用位置: prepare_dataset。
    """

    missing = [str(path) for path in (cfg.raw_flights_csv, cfg.raw_parameters_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少原始数据文件: " + ", ".join(missing))


def load_raw_flights(cfg: ExperimentConfig) -> pd.DataFrame:
    """功能: 读取原始飞行数据并保留建模所需字段。
    参数: cfg为实验配置对象。
    返回: 包含原始飞行记录的DataFrame。
    调用位置: prepare_dataset。
    """

    validate_raw_files(cfg)
    frame = pd.read_csv(cfg.raw_flights_csv, low_memory=False)
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"原始数据缺少字段: {missing}")
    return frame[RAW_COLUMNS].copy()


def calculate_time_delta(frame: pd.DataFrame) -> pd.Series:
    """功能: 按flight计算相邻采样点时间间隔。
    参数: frame为已按flight和time排序的数据表。
    返回: dt_seconds时间间隔序列。
    调用位置: add_derived_features。
    """

    dt = frame.groupby("flight", sort=False)["time"].diff()
    valid_dt = dt[(dt > 0) & (dt < 5)]
    fallback = float(valid_dt.median()) if len(valid_dt) else 0.2
    return dt.where((dt > 0) & (dt < 5), fallback).fillna(fallback)


def add_derived_features(raw_frame: pd.DataFrame, training_route: str = "R1") -> tuple[pd.DataFrame, list[str]]:
    """功能: 构造能耗预测所需的实测工况和代理工况特征。
    参数: raw_frame为原始飞行数据，training_route为本实验保留的航线编号。
    返回: 处理后的DataFrame和特征列名列表。
    调用位置: prepare_dataset、prepare_prediction_frame。
    """

    frame = raw_frame.copy()
    for column in NUMERIC_SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["flight"] = pd.to_numeric(frame["flight"], errors="coerce").astype("Int64")
    frame["route"] = frame["route"].astype(str).str.strip()
    frame = frame.dropna(subset=["flight", "route", *NUMERIC_SOURCE_COLUMNS])

    # 2.0 固定只使用 R1 航线，其他航线另存为排除数据，不进入任何数据集。
    frame = frame[frame["route"].eq(str(training_route).strip())].copy()
    frame["flight"] = frame["flight"].astype(int)
    frame = frame.sort_values(["flight", "time"]).reset_index(drop=True)

    frame["dt_seconds"] = calculate_time_delta(frame)
    frame["battery_current_discharge_a"] = frame["battery_current"].clip(lower=0)
    frame["power_w"] = (frame["battery_voltage"] * frame["battery_current_discharge_a"]).clip(lower=0)
    frame["energy_interval_wh"] = frame["power_w"] * frame["dt_seconds"] / 3600.0

    wind_rad = np.deg2rad(frame["wind_angle"].to_numpy(dtype=float))
    wind_x = frame["wind_speed"].to_numpy(dtype=float) * np.cos(wind_rad)
    wind_y = frame["wind_speed"].to_numpy(dtype=float) * np.sin(wind_rad)
    velocity_x = frame["velocity_x"].to_numpy(dtype=float)
    velocity_y = frame["velocity_y"].to_numpy(dtype=float)
    velocity_z = frame["velocity_z"].to_numpy(dtype=float)
    horizontal_speed = np.sqrt(velocity_x**2 + velocity_y**2)
    actual_speed = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
    wind_speed = frame["wind_speed"].to_numpy(dtype=float)
    relative_air_speed = np.sqrt((velocity_x - wind_x) ** 2 + (velocity_y - wind_y) ** 2 + velocity_z**2)
    dot = velocity_x * wind_x + velocity_y * wind_y
    wind_alignment = dot / np.maximum(horizontal_speed * wind_speed, 1e-6)
    wind_cross = np.abs(velocity_x * wind_y - velocity_y * wind_x) / np.maximum(horizontal_speed, 1e-6)

    dynamic_accel_norm = np.sqrt(
        frame["linear_acceleration_x"].to_numpy(dtype=float) ** 2
        + frame["linear_acceleration_y"].to_numpy(dtype=float) ** 2
        + (frame["linear_acceleration_z"].to_numpy(dtype=float) + 9.80665) ** 2
    )
    angular_rate_norm = np.sqrt(
        frame["angular_x"].to_numpy(dtype=float) ** 2
        + frame["angular_y"].to_numpy(dtype=float) ** 2
        + frame["angular_z"].to_numpy(dtype=float) ** 2
    )

    payload_kg = frame["payload"].to_numpy(dtype=float) / 1000.0
    altitude_m = frame["altitude"].to_numpy(dtype=float)
    obstacle_agility = dynamic_accel_norm * (1.0 + angular_rate_norm) + np.abs(velocity_z) * 0.2
    thermal_load = (
        0.5 * np.power(np.maximum(relative_air_speed, 0), 3)
        + payload_kg * dynamic_accel_norm * np.maximum(actual_speed, 0)
        + np.maximum(velocity_z, 0) * (3.6 + payload_kg) * 9.80665
    )

    frame["wind_sin"] = np.sin(wind_rad)
    frame["wind_cos"] = np.cos(wind_rad)
    frame["programmed_speed_mps"] = frame["speed"]
    frame["actual_speed_mps"] = actual_speed
    frame["horizontal_speed_mps"] = horizontal_speed
    frame["vertical_speed_mps"] = velocity_z
    frame["vertical_speed_abs_mps"] = np.abs(velocity_z)
    frame["relative_air_speed_mps"] = relative_air_speed
    frame["wind_alignment"] = np.clip(wind_alignment, -1.0, 1.0)
    frame["wind_cross_component_mps"] = wind_cross
    frame["payload_kg"] = payload_kg
    frame["altitude_m"] = altitude_m
    frame["dynamic_accel_norm"] = dynamic_accel_norm
    frame["angular_rate_norm"] = angular_rate_norm
    frame["obstacle_agility_index"] = obstacle_agility
    frame["thermal_load_proxy"] = thermal_load
    frame["vision_energy_proxy_w"] = 2.5 + 0.15 * actual_speed + 0.30 * obstacle_agility + 0.002 * altitude_m
    frame["communication_energy_proxy_w"] = 0.8 + 0.004 * altitude_m + 0.04 * wind_speed + 0.01 * actual_speed

    max_time = frame.groupby("flight", sort=False)["time"].transform("max").replace(0, np.nan)
    frame["flight_progress"] = (frame["time"] / max_time).fillna(0.0).clip(0, 1)

    route_dummies = pd.get_dummies(frame["route"], prefix="route", dtype=float)
    frame = pd.concat([frame, route_dummies], axis=1)
    route_features = sorted(route_dummies.columns.tolist())
    feature_columns = ENGINEERED_NUMERIC_FEATURES + route_features

    useful_columns = [
        "flight",
        "route",
        "dt_seconds",
        "power_w",
        "energy_interval_wh",
        "battery_voltage",
        "battery_current",
        *feature_columns,
    ]
    frame = frame[useful_columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame, feature_columns


def split_by_flight(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """功能: 按飞行编号划分训练、验证和测试集。
    参数: frame为完整特征数据表，cfg为实验配置对象。
    返回: train、val、test三个DataFrame字典。
    调用位置: prepare_dataset。
    """

    rng = np.random.default_rng(cfg.random_seed)
    flights = frame["flight"].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(flights)
    test_count = max(1, math.ceil(len(flights) * cfg.test_ratio))
    val_count = max(1, math.ceil(len(flights) * cfg.val_ratio))
    test_flights = set(flights[:test_count])
    val_flights = set(flights[test_count : test_count + val_count])
    train_flights = set(flights[test_count + val_count :])
    if not train_flights:
        raise ValueError("训练集flight为空，请降低验证集或测试集比例。")
    return {
        "train": frame[frame["flight"].isin(train_flights)].copy(),
        "val": frame[frame["flight"].isin(val_flights)].copy(),
        "test": frame[frame["flight"].isin(test_flights)].copy(),
    }


def save_json(path: Path, payload: dict) -> None:
    """功能: 以UTF-8保存JSON文件。
    参数: path为保存路径，payload为待保存字典。
    返回: None。
    调用位置: prepare_dataset、train.py、evaluate.py。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    """功能: 以UTF-8读取JSON文件。
    参数: path为JSON文件路径。
    返回: JSON反序列化后的字典。
    调用位置: train.py、predict.py、evaluate.py。
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_dataset(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 完成数据下载、解压、特征工程和数据切分。
    参数: cfg为实验配置对象，force表示是否强制重建处理后数据。
    返回: 数据集摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    expected = [
        cfg.clean_data_csv,
        cfg.excluded_routes_csv,
        cfg.train_csv,
        cfg.val_csv,
        cfg.test_csv,
        cfg.feature_meta_json,
    ]
    if not force and all(path.exists() for path in expected):
        progress = TerminalProgress("数据准备", 1)
        summary = load_json(cfg.dataset_summary_json) if cfg.dataset_summary_json.exists() else {}
        progress.finish(f"已检测到处理后数据，跳过重建（{summary.get('processed_rows', '未知')} 条）")
        return summary

    progress = TerminalProgress("数据准备", 6)
    download_source_dataset(cfg)
    progress.update(1, "原始数据已就绪")
    extract_source_zip(cfg)
    progress.update(2, "原始文件已解压")
    raw = load_raw_flights(cfg)
    progress.update(3, f"已读取 {len(raw)} 条原始记录")
    route_series = raw["route"].astype(str).str.strip()
    route_distribution_before = route_series.value_counts().sort_index().to_dict()
    excluded = raw[route_series.ne(cfg.training_route)].copy()
    excluded.to_csv(cfg.excluded_routes_csv, index=False, encoding="utf-8")
    progress.update(4, f"已筛选 {cfg.training_route} 航线，排除 {len(excluded)} 条其他航线记录")
    feature_frame, feature_columns = add_derived_features(raw, cfg.training_route)
    progress.update(5, f"已构造 {len(feature_columns)} 个特征")
    splits = split_by_flight(feature_frame, cfg)
    progress.update(6, "训练集、验证集和测试集已划分")

    feature_frame.to_csv(cfg.clean_data_csv, index=False, encoding="utf-8")
    splits["train"].to_csv(cfg.train_csv, index=False, encoding="utf-8")
    splits["val"].to_csv(cfg.val_csv, index=False, encoding="utf-8")
    splits["test"].to_csv(cfg.test_csv, index=False, encoding="utf-8")

    meta = {
        "feature_columns": feature_columns,
        "training_route": cfg.training_route,
        "excluded_routes_file": str(cfg.excluded_routes_csv),
        "route_distribution_before_filter": {str(k): int(v) for k, v in route_distribution_before.items()},
        "route_distribution_after_filter": {
            str(k): int(v) for k, v in feature_frame["route"].value_counts().sort_index().items()
        },
        "field_descriptions": {
            "flight": "飞行任务编号；同一编号的连续采样点属于同一次飞行，用于分组切分。",
            "route": "航线编号；本版本仅保留 R1。",
            "dt_seconds": "相邻采样点时间间隔，单位为秒；用于把功率积分为能耗。",
            "power_w": "监督学习目标；电池电压乘以非负放电电流得到的瞬时功率，单位为瓦特。",
            "energy_interval_wh": "当前采样间隔内的能耗，等于 power_w * dt_seconds / 3600，单位为瓦时。",
            "battery_voltage": "电池端电压，单位为伏特。",
            "battery_current": "电池电流原始值，单位为安培；放电功率计算时仅使用非负部分。",
            "wind_angle": "风向角，单位为度；通过正弦和余弦转换为连续特征。",
            "velocity_x": "机体坐标系 X 轴速度，单位为米每秒。",
            "velocity_y": "机体坐标系 Y 轴速度，单位为米每秒。",
            "velocity_z": "机体坐标系 Z 轴速度，单位为米每秒。",
            "angular_x": "绕 X 轴角速度，单位为弧度每秒。",
            "angular_y": "绕 Y 轴角速度，单位为弧度每秒。",
            "angular_z": "绕 Z 轴角速度，单位为弧度每秒。",
            "linear_acceleration_x": "X 轴线性加速度，单位为米每二次方秒。",
            "linear_acceleration_y": "Y 轴线性加速度，单位为米每二次方秒。",
            "linear_acceleration_z": "Z 轴线性加速度，单位为米每二次方秒，包含重力影响的原始测量值。",
            "speed": "任务记录的规划飞行速度，单位为米每秒。",
            "payload": "原始载荷质量，单位为克；payload_kg为其换算值。",
            "altitude": "原始飞行高度，单位为米；altitude_m为模型使用的同值派生字段。",
            "time": "飞行内相对时间戳，单位为秒。",
            "flight_progress": "当前时间除以本次飞行最大时间，范围 0 到 1，表示飞行进度。",
            "wind_speed": "风速，单位为米每秒。",
            "wind_sin": "风向角的正弦分量，用于连续表达风向。",
            "wind_cos": "风向角的余弦分量，用于连续表达风向。",
            "programmed_speed_mps": "任务规划速度，单位为米每秒。",
            "actual_speed_mps": "三轴速度合成后的实际速度，单位为米每秒。",
            "horizontal_speed_mps": "水平面速度合成值，单位为米每秒。",
            "vertical_speed_mps": "垂直速度分量，单位为米每秒；正负表示升降方向。",
            "vertical_speed_abs_mps": "垂直速度绝对值，单位为米每秒，表示垂直机动强度。",
            "relative_air_speed_mps": "相对空气速度，结合机体速度和风速计算，单位为米每秒。",
            "wind_alignment": "水平速度与风速方向的余弦相似度，范围 -1 到 1。",
            "wind_cross_component_mps": "风速在水平速度横向上的分量，单位为米每秒。",
            "payload_kg": "载荷质量，单位为千克。",
            "altitude_m": "飞行高度，单位为米。",
            "dynamic_accel_norm": "去除重力影响后的三轴线性加速度合成强度，单位为米每二次方秒。",
            "angular_rate_norm": "三轴角速度合成强度，单位为弧度每秒。",
            "obstacle_agility_index": "由加速度、角速度和垂直速度构成的机动性代理指标，无量纲。",
            "thermal_load_proxy": "由相对气流、载荷和爬升状态估计的热负荷代理指标，用于表征附加能耗。",
            "vision_energy_proxy_w": "由速度、机动性和高度估计的视觉计算附加功率代理值，单位为瓦特。",
            "communication_energy_proxy_w": "由高度、风速和速度估计的通信附加功率代理值，单位为瓦特。",
            "route_R1": "R1 航线独热编码；R1 样本取 1。",
        },
        "target_column": cfg.target_column,
        "measured_features": [
            "wind_speed",
            "wind_angle",
            "velocity_x/y/z",
            "angular_x/y/z",
            "linear_acceleration_x/y/z",
            "speed",
            "payload",
            "altitude",
            "route",
        ],
        "proxy_features": [
            "obstacle_agility_index",
            "thermal_load_proxy",
            "vision_energy_proxy_w",
            "communication_energy_proxy_w",
        ],
        "target_definition": "power_w = max(battery_voltage * battery_current, 0)",
        "split_method": "按flight分组随机切分，避免同一飞行泄漏到多个集合。",
        "training_route": cfg.training_route,
        "excluded_routes_file": str(cfg.excluded_routes_csv),
    }
    save_json(cfg.feature_meta_json, meta)

    summary = {
        "_文件说明": (
            "本文件汇总实验2.0的数据来源、清洗结果、数据集划分规模、模型输入维度和预测目标。"
            "rows表示采样记录数，flights表示按flight字段统计的不重复飞行次数。"
        ),
        "_字段说明": {
            "source_repo_url": "原始无人机飞行数据集的公开仓库地址。",
            "raw_rows": (
                "从原始flights.csv读取并保留建模所需字段后的采样记录总数；"
                "此时尚未执行缺失值清理和航线筛选，单位为行。"
            ),
            "processed_rows": (
                "完成数值转换、缺失值删除、仅保留R1航线、特征工程及无穷值清理后"
                "保留的有效采样记录总数，单位为行。"
            ),
            "processed_flights": (
                "清洗后有效数据中不同flight编号的数量，即参与后续划分的有效飞行任务总数，单位为次。"
            ),
            "train_rows": "训练集包含的采样记录数。模型使用这些记录拟合网络参数，单位为行。",
            "val_rows": (
                "验证集包含的采样记录数。该集合用于候选模型比较、学习率调整和早停判断，"
                "不直接更新网络参数，单位为行。"
            ),
            "test_rows": "测试集包含的采样记录数。该集合仅用于最终预测和性能评估，单位为行。",
            "train_flights": (
                "训练集包含的不重复飞行任务数量。数据按完整flight分组划分，"
                "同一次飞行不会同时出现在其他数据集中，单位为次。"
            ),
            "val_flights": "验证集包含的不重复飞行任务数量，单位为次。",
            "test_flights": "测试集包含的不重复飞行任务数量，单位为次。",
            "feature_count": (
                "每条样本输入模型的特征数量，包括实测工况、派生特征、代理特征、"
                "飞行进度和航线独热编码；具体维度以feature_count为准。"
            ),
            "target_column": (
                "监督学习的目标字段名。power_w表示由非负放电电流与电池电压相乘得到的"
                "瞬时功率，单位为瓦特（W）。"
            ),
        },
        "source_repo_url": cfg.source_repo_url,
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(feature_frame)),
        "processed_flights": int(feature_frame["flight"].nunique()),
        "train_rows": int(len(splits["train"])),
        "val_rows": int(len(splits["val"])),
        "test_rows": int(len(splits["test"])),
        "train_flights": int(splits["train"]["flight"].nunique()),
        "val_flights": int(splits["val"]["flight"].nunique()),
        "test_flights": int(splits["test"]["flight"].nunique()),
        "feature_count": int(len(feature_columns)),
        "target_column": cfg.target_column,
        "training_route": cfg.training_route,
        "excluded_rows": int(len(excluded)),
        "excluded_routes_file": str(cfg.excluded_routes_csv),
        "route_distribution_before_filter": {str(k): int(v) for k, v in route_distribution_before.items()},
        "route_distribution_after_filter": {
            str(k): int(v) for k, v in feature_frame["route"].value_counts().sort_index().items()
        },
    }
    save_json(cfg.dataset_summary_json, summary)
    progress.finish(f"已保存 {summary['processed_rows']} 条有效记录")
    return summary


def prepare_prediction_frame(input_csv: Path, cfg: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    """功能: 将待预测CSV转换为模型需要的特征表。
    参数: input_csv为待预测CSV，cfg为实验配置对象。
    返回: 预测用DataFrame和特征列名列表。
    调用位置: predict.py。
    """

    meta = load_json(cfg.feature_meta_json)
    feature_columns = meta["feature_columns"]
    frame = pd.read_csv(input_csv)
    if "route" in frame.columns:
        routes = frame["route"].astype(str).str.strip()
        invalid_routes = sorted(set(routes) - {cfg.training_route})
        if invalid_routes:
            raise ValueError(f"实验2.0仅支持{cfg.training_route}航线，输入文件包含: {invalid_routes}")
    else:
        dummy_columns = [column for column in frame.columns if column.startswith("route_") and column != f"route_{cfg.training_route}"]
        active_dummies = [column for column in dummy_columns if pd.to_numeric(frame[column], errors="coerce").fillna(0).abs().gt(1e-8).any()]
        if active_dummies:
            raise ValueError(f"实验2.0仅支持{cfg.training_route}航线，输入特征包含其他航线独热编码: {active_dummies}")
    if all(column in frame.columns for column in feature_columns):
        return frame, feature_columns
    missing_raw = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing_raw:
        raise ValueError(f"预测文件既不是处理后特征表，也缺少原始字段: {missing_raw}")
    feature_frame, _ = add_derived_features(frame, cfg.training_route)
    for column in feature_columns:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0
    return feature_frame, feature_columns
