# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-11-01
# 文件名: train.py
# 功能说明: 加载冻结TCN模型、构造历史序列和分批预测功率
# 版本号：1.0

from __future__ import annotations


import numpy as np
import pandas as pd
import torch
from torch import nn

from config import ExperimentConfig
from model import build_model


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    """功能: 检查并返回实验使用的CUDA设备。
    参数: cfg为实验配置对象。
    返回: 可用的torch设备。
    调用位置: train_model、predict_array。
    """

    if str(cfg.device).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.device)
    if str(cfg.device).lower() == "cpu":
        return torch.device("cpu")
    raise RuntimeError("本实验默认要求CUDA；请使用--device cpu显式选择CPU")


def build_sequences(frame: pd.DataFrame, scaler: dict, window_steps: int,
                    require_target: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """功能: 按完整flight构造只使用当前及历史输入的TCN序列。
    参数: frame为特征表，scaler为标准化参数，window_steps为TCN窗口步数，require_target为是否读取标签。
    返回: 序列数组和标准化目标数组。
    调用位置: tune_group、train_group、predict_frame。
    """

    columns = list(scaler["feature_columns"])
    mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    std = np.asarray(scaler["x_std"], dtype=np.float32)
    values = (frame[columns].to_numpy(np.float32) - mean) / std
    target = frame["power_w"].to_numpy(np.float32) if require_target else np.zeros(len(frame), dtype=np.float32)
    sequences = np.empty((len(frame), window_steps, len(columns)), dtype=np.float32)
    targets = np.empty(len(frame), dtype=np.float32)
    for indices in frame.groupby("flight", sort=False).indices.values():
        indices = np.asarray(indices, dtype=np.int64)
        group_values = values[indices]
        group_target = target[indices]
        padded = np.concatenate([np.repeat(group_values[:1], window_steps - 1, axis=0), group_values])
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_steps, axis=0)
        sequences[indices] = windows.transpose(0, 2, 1)
        targets[indices] = (group_target - scaler["y_mean"]) / scaler["y_std"]
    return sequences, targets


def predict_array(model: nn.Module, sequences: np.ndarray, scaler: dict, cfg: ExperimentConfig) -> np.ndarray:
    """功能: 分批预测并把标准化功率还原为瓦特。
    参数: model为TCN，sequences为输入序列，scaler为标准化参数，cfg为配置。
    返回: 非负功率数组。
    调用位置: tune_group、train_group、evaluate.py。
    """

    device = resolve_device(cfg)
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), cfg.batch_size):
            batch = torch.from_numpy(sequences[start:start + cfg.batch_size]).to(device, non_blocking=True)
            outputs.append(model(batch).cpu().numpy())
    values = np.concatenate(outputs) * scaler["y_std"] + scaler["y_mean"]
    return np.maximum(values, 0.0)


def _model_from_checkpoint(checkpoint: dict, input_dim: int) -> nn.Module:
    """功能: 按检查点元数据重建TCN模型。
    参数: checkpoint为权重和结构元数据，input_dim为输入维度。
    返回: 已加载权重的模型。
    调用位置: train_group、evaluate.py。
    """

    model = build_model(input_dim, tuple(checkpoint["channels"]), checkpoint["kernel_size"],
                        checkpoint["dropout"], checkpoint["activation"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
