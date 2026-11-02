# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: data_utils.py
# 开发时间: 2026-11-01
# 文件名: data_utils.py
# 功能说明: 构造五组TCN实验的原始特征和10步因果SMA特征
# 版本号：1.0

from __future__ import annotations


import numpy as np
import pandas as pd

from config import (RAW_FEATURE_COLUMNS, SMA_FEATURE_COLUMNS, TARGET_COLUMN,
                    ExperimentConfig, table_path)


REQUIRED_COLUMNS = ["flight", "time", "dt_seconds", TARGET_COLUMN, "energy_interval_wh", *RAW_FEATURE_COLUMNS]


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
