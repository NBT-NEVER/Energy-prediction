# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-10-30
# 文件名: data_utils.py
# 功能说明: 构造五组TCN实验的原始特征和10步因果SMA特征
# 版本号：1.0

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import (RAW_FEATURE_COLUMNS, SMA_FEATURE_COLUMNS, TARGET_COLUMN,
                    SCHEMA_VERSION, ExperimentConfig, ensure_directories, table_path)


REQUIRED_COLUMNS = ["flight", "time", "dt_seconds", TARGET_COLUMN, "energy_interval_wh", *RAW_FEATURE_COLUMNS]


def prepare_planned_data(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 将4.1字段转换为与AMOVFLY一致的planned数据契约。
    参数: cfg为实验配置，force为是否覆盖既有输出。
    返回: D盘数据文件和划分摘要。
    调用位置: main.py。
    """
    ensure_directories(cfg)
    if cfg.manifest_json.exists() and not force:
        summary = json.loads(cfg.manifest_json.read_text(encoding="utf-8"))
        if summary.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("加工数据契约已过期，请显式重新prepare")
        for name, expected in summary["sha256"].items():
            if hashlib.sha256((cfg.data_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"加工数据指纹已改变: {name}")
        return summary
    names = [("train_4.1.csv", "train_TCN-comparison.csv"),
             ("val_4.1.csv", "val_TCN-comparison.csv"),
             ("test_4.1.csv", "test_TCN-comparison.csv")]
    checksums, converted = {}, []
    for source_name, target_name in names:
        source = cfg.source_processed_dir / source_name
        target = cfg.data_dir / target_name
        if not source.exists():
            raise FileNotFoundError(f"缺少4.1加工数据文件: {source}")
        if force or not target.exists():
            frame = pd.read_csv(source)
            required = ["flight", "route", "split", "split_role", "time", "dt_seconds",
                        "time_s", "task_duration_s", "planned_position_east_m",
                        "planned_position_north_m", "planned_position_up_m", "wind_east_mps",
                        "wind_north_mps", "payload_g", "planned_motor_on", "planned_airborne",
                        "power_w", "energy_interval_wh"]
            missing = sorted(set(required) - set(frame.columns))
            if missing:
                raise ValueError(f"4.1加工表缺少字段: {missing}")
            frame["position_x_m"] = frame["planned_position_east_m"].astype(float)
            frame["position_y_m"] = frame["planned_position_north_m"].astype(float)
            frame["position_z_m"] = frame["planned_position_up_m"].astype(float)
            east, north = frame["wind_east_mps"].astype(float), frame["wind_north_mps"].astype(float)
            frame["wind_speed"] = np.hypot(east, north)
            frame["wind_angle"] = np.arctan2(north, east)
            frame["uav_type"] = 4
            if not frame["planned_motor_on"].isin([0, 1]).all() or not frame["planned_airborne"].isin([0, 1]).all():
                raise ValueError("来源状态不是合法二进制编码")
            if (frame["planned_airborne"] > frame["planned_motor_on"]).any():
                raise ValueError("来源状态不满足airborne<=motor_on")
            frame["planned_motor_on"] = frame["planned_motor_on"].astype(np.int8)
            frame["planned_airborne"] = frame["planned_airborne"].astype(np.int8)
            frame["planned"] = frame["planned_motor_on"] + frame["planned_airborne"]
            columns = ["flight", "route", "split", "split_role", "time", "time_s", "task_duration_s",
                       "position_x_m", "position_y_m", "position_z_m", "wind_speed", "wind_angle",
                       "payload_g", "uav_type", "planned", "planned_motor_on", "planned_airborne",
                       "dt_seconds", "power_w", "energy_interval_wh"]
            output = frame[columns].sort_values(["flight", "time"], kind="stable")
            if not np.isfinite(output.select_dtypes(include=[np.number]).to_numpy(float)).all():
                raise ValueError(f"转换后的数据含非有限值: {source}")
            if not np.allclose(output["energy_interval_wh"], output["power_w"] * output["dt_seconds"] / 3600):
                raise ValueError(f"功率、时间和区间能耗不一致: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(target, index=False, encoding="utf-8")
            converted.append({"name": target.name, "rows": int(len(output)), "flights": int(output.flight.nunique())})
        checksums[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {"version": SCHEMA_VERSION, "source_version": "4.1",
                "uav_type_mapping": {"DJI Matrice 100": 4},
                "planned_mapping": {"0": "停机", "1": "原地启动", "2": "飞行"},
                "feature_columns": list(RAW_FEATURE_COLUMNS), "sma_feature_columns": list(SMA_FEATURE_COLUMNS),
                "wind_angle_sma": "atan2(mean(sin(angle)), mean(cos(angle)))，按flight计算10步圆均值",
                "status_definition": "planned=planned_motor_on+planned_airborne，沿用4.1的100W/1m/0.5m阈值",
                "wind_definition": "wind_speed=hypot(wind_east_mps,wind_north_mps)，wind_angle=atan2(north,east)"}
    metadata_path = cfg.data_dir / "feature_metadata_TCN-comparison.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    checksums[metadata_path.name] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    train, val, test = (pd.read_csv(cfg.data_dir / name, usecols=["flight"]) for name in ("train_TCN-comparison.csv", "val_TCN-comparison.csv", "test_TCN-comparison.csv"))
    sets = [set(table.flight) for table in (train, val, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("train、val、test存在flight交叉")
    source_hashes = {name: hashlib.sha256((cfg.source_processed_dir / name).read_bytes()).hexdigest() for name, _ in names}
    state_counts = {split: {str(int(key)): int(value) for key, value in pd.read_csv(table_path(cfg, split))["planned"].value_counts().sort_index().items()} for split in ("train", "val", "test")}
    summary = {"source_processed_dir": str(cfg.source_processed_dir), "source_sha256": source_hashes,
               "state_counts": state_counts, "resample_seconds": cfg.resample_seconds,
               "split_flights": {split: sorted(int(value) for value in ids) for split, ids in zip(("train", "val", "test"), sets)},
               "data_dir": str(cfg.data_dir), "converted_files": converted, "sha256": checksums,
               "schema_version": SCHEMA_VERSION, "feature_columns": list(RAW_FEATURE_COLUMNS),
               "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
               "train_flights": int(train.flight.nunique()), "val_flights": int(val.flight.nunique()),
               "test_flights": int(test.flight.nunique()), "split_overlap": False}
    cfg.manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_tables(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """功能: 读取本实验独立的训练、验证和测试表并检查字段。
    参数: cfg为实验配置。
    返回: train、validation、test三个数据表。
    调用位置: train.py、evaluate.py。
    """

    tables = tuple(pd.read_csv(table_path(cfg, name))
                   for name in ("train", "val", "test"))
    for frame in tables:
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"加工数据缺少字段: {missing}")
        if not frame.uav_type.eq(4).all() or not frame.planned.isin([0, 1, 2]).all():
            raise ValueError("DJI机型或planned状态编码不合法")
        if not np.array_equal(frame.planned, frame.planned_motor_on + frame.planned_airborne):
            raise ValueError("planned与审计状态不一致")
    return tables


def feature_columns(use_sma: bool) -> list[str]:
    """功能: 返回实验组的确定性模型输入顺序。
    参数: use_sma为是否拼接10步SMA。
    返回: 原始10维或原始10维加5维SMA列表。
    调用位置: prepare_features、train.py。
    """

    if not use_sma:
        return list(RAW_FEATURE_COLUMNS)
    return list(RAW_FEATURE_COLUMNS) + [f"{column}_sma10" for column in SMA_FEATURE_COLUMNS]


def prepare_features(frame: pd.DataFrame, use_sma: bool, sma_window: int) -> pd.DataFrame:
    """功能: 按flight计算只使用当前和历史值的SMA并保留原始字段。
    参数: frame为加工数据表，use_sma为开关，sma_window为窗口点数。
    返回: 按flight和time排序、包含模型输入列的数据表。
    调用位置: train.py、evaluate.py。
    """

    ordered = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True).copy()
    if not use_sma:
        return ordered
    if sma_window < 1:
        raise ValueError("SMA窗口必须为正整数")
    grouped = ordered.groupby("flight", sort=False)
    for column in SMA_FEATURE_COLUMNS:
        target = f"{column}_sma10"
        if column == "wind_angle":
            sin_mean = grouped[column].transform(
                lambda values: np.sin(values).rolling(sma_window, min_periods=1).mean()
            )
            cos_mean = grouped[column].transform(
                lambda values: np.cos(values).rolling(sma_window, min_periods=1).mean()
            )
            ordered[target] = np.arctan2(sin_mean, cos_mean)
        else:
            ordered[target] = grouped[column].transform(
                lambda values: values.rolling(sma_window, min_periods=1).mean()
            )
    return ordered


def validate_features(frame: pd.DataFrame, columns: list[str]) -> None:
    """功能: 验证输入特征有限、时间有序且标签字段存在。
    参数: frame为待检查数据，columns为模型输入列。
    返回: None，异常时抛出ValueError。
    调用位置: train.py、evaluate.py。
    """

    missing = sorted(set(columns + [TARGET_COLUMN]) - set(frame.columns))
    if missing:
        raise ValueError(f"模型输入缺少字段: {missing}")
    if not np.isfinite(frame[columns + [TARGET_COLUMN]].to_numpy(float)).all():
        raise ValueError("模型输入或功率标签存在非有限值")
    if frame.duplicated(["flight", "time"]).any():
        raise ValueError("加工数据含重复flight/time")
    if not np.allclose(frame["energy_interval_wh"], frame["power_w"] * frame["dt_seconds"] / 3600):
        raise ValueError("功率、时间和区间能耗不一致")
    for _, group in frame.groupby("flight", sort=False):
        if not group["time"].is_monotonic_increasing:
            raise ValueError("单个flight时间未按升序排列")
