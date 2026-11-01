# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-10-30
# 文件名: data_utils.py
# 功能说明: 构造四组TCN实验的原始特征和10步因果SMA特征
# 版本号：1.0

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import (RAW_FEATURE_COLUMNS, SMA_FEATURE_COLUMNS, TARGET_COLUMN,
                    ExperimentConfig, ensure_directories, table_path)


REQUIRED_COLUMNS = ["flight", "time", "dt_seconds", TARGET_COLUMN, "energy_interval_wh", *RAW_FEATURE_COLUMNS]


def copy_processed_data(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 将4.1已核验加工表复制到本实验独立数据目录。
    参数: cfg为实验配置，force为是否覆盖已有表。
    返回: 复制文件和数据规模摘要。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    if cfg.manifest_json.exists() and not force:
        previous = json.loads(cfg.manifest_json.read_text(encoding="utf-8"))
        checksums = previous.get("sha256", {})
        if checksums and all((cfg.data_dir / name).exists() for name in checksums):
            for name, expected in checksums.items():
                if hashlib.sha256((cfg.data_dir / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"实验数据副本已改变: {name}，请使用新目录或明确重建")
            return previous
    if not cfg.source_processed_dir.exists():
        raise FileNotFoundError(f"找不到4.1加工数据目录: {cfg.source_processed_dir}")
    names = ["train_4.1.csv", "val_4.1.csv", "test_4.1.csv", "feature_metadata_4.1.json"]
    copied, checksums = [], {}
    for name in names:
        source = cfg.source_processed_dir / name
        target = cfg.data_dir / name.replace("4.1", "TCN-comparison")
        if not source.exists():
            raise FileNotFoundError(f"缺少加工数据文件: {source}")
        if force or not target.exists():
            shutil.copy2(source, target)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if source_hash != target_hash:
            raise ValueError(f"实验副本与来源不一致，需确认后使用--force-prepare重建: {target}")
        checksums[target.name] = target_hash
        copied.append(str(target))
    train = pd.read_csv(table_path(cfg, "train"), usecols=["flight"])
    val = pd.read_csv(table_path(cfg, "val"), usecols=["flight"])
    test = pd.read_csv(table_path(cfg, "test"), usecols=["flight"])
    ids = [set(frame.flight) for frame in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("train、val、test之间出现重复flight")
    summary = {"data_dir": str(cfg.data_dir), "copied_files": copied,
               "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
               "train_flights": int(train.flight.nunique()), "val_flights": int(val.flight.nunique()),
               "test_flights": int(test.flight.nunique()), "sha256": checksums,
               "source_processed_dir": str(cfg.source_processed_dir), "split_overlap": False,
               "split_flights": {name: sorted(int(x) for x in values)
                                 for name, values in zip(("train", "val", "test"), ids)}}
    cfg.manifest_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
