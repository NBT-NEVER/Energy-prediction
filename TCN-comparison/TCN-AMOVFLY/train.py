# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-11-01
# 文件名: train.py
# 功能说明: 完成五组TCN窗口搜索、正式训练和离线功率预测
# 版本号：1.0

from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import GROUPS, RAW_FEATURE_COLUMNS, ExperimentConfig, group_paths
from data_utils import load_tables, prepare_features, validate_features, feature_columns
from model import build_model


def seed_everything(seed: int) -> None:
    """功能: 固定Python、NumPy和PyTorch的随机种子。
    参数: seed为随机种子。
    返回: None。
    调用位置: tune_group、train_group。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def fit_scaler(frame: pd.DataFrame, columns: list[str], cfg: ExperimentConfig, group: str) -> dict:
    """功能: 只用训练集拟合模型输入和功率目标的标准化参数。
    参数: frame为训练表，columns为输入列，cfg为配置，group为实验组。
    返回: 可序列化的标准化参数。
    调用位置: train_group。
    """

    x = frame[columns].to_numpy(float)
    y = frame["power_w"].to_numpy(float)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
    scaler = {"group": group, "raw_feature_columns": RAW_FEATURE_COLUMNS,
              "feature_columns": columns, "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
              "y_mean": y_mean, "y_std": y_std, "target_column": "power_w",
              "sma_window_steps": cfg.sma_window_steps, "resample_seconds": cfg.resample_seconds,
              "schema_version": "AMOVFLY-TCN-comparison-1.0"}
    path = group_paths(cfg, group)["scaler"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    return scaler


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


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> float:
    """功能: 执行一轮训练或验证并返回平均SmoothL1损失。
    参数: model为TCN，loader为数据加载器，optimizer为训练优化器或None，device为设备。
    返回: 平均损失。
    调用位置: train_model。
    """

    model.train(optimizer is not None)
    criterion = nn.SmoothL1Loss()
    losses = []
    for features, target in loader:
        features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(features)
            loss = criterion(output, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("训练或验证损失出现非有限值")
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(train_seq: np.ndarray, train_y: np.ndarray, val_seq: np.ndarray, val_y: np.ndarray,
                cfg: ExperimentConfig, input_dim: int, activation: str,
                window_steps: int, epochs: int) -> tuple[dict, list[dict]]:
    """功能: 对单个实验组和TCN窗口执行训练并保存最佳验证状态。
    参数: train_seq/train_y/val_seq/val_y为标准化数据，cfg为配置，input_dim为输入维度，activation为激活模式，window_steps和epochs为训练参数。
    返回: 最佳检查点字典和逐轮日志。
    调用位置: tune_group、train_group。
    """

    device = resolve_device(cfg)
    model = build_model(input_dim, cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout, activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_y)),
                              batch_size=cfg.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_y)),
                            batch_size=cfg.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    best_loss, best_state, best_epoch, logs = float("inf"), None, 0, []
    started = time.perf_counter()
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "learning_rate": float(optimizer.param_groups[0]["lr"])})
        print(f"\r窗口{window_steps}步 epoch {epoch}/{epochs} train={train_loss:.5f} val={val_loss:.5f}", end="", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.early_stopping_patience:
            print(f"\n验证损失连续{stale_epochs}轮未改善，提前停止")
            break
    print()
    return {"state_dict": best_state, "best_val_loss": best_loss,
            "window_steps": window_steps, "window_seconds": window_steps * cfg.resample_seconds,
            "channels": list(cfg.tcn_channels), "kernel_size": cfg.tcn_kernel_size,
            "dropout": cfg.tcn_dropout, "activation": activation, "best_epoch": best_epoch,
            "epochs_run": len(logs), "training_seconds": time.perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters())}, logs


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


def _metrics(frame: pd.DataFrame, predicted: np.ndarray, cfg: ExperimentConfig) -> dict:
    """功能: 计算功率、1秒窗口能耗和flight总能耗指标。
    参数: frame为按flight/time排序的数据，predicted为功率预测，cfg为时间配置。
    返回: 指标字典。
    调用位置: tune_group、evaluate.py。
    """

    actual = frame["power_w"].to_numpy(float)
    dt = frame["dt_seconds"].to_numpy(float)
    def wape(a, p):
        """功能: 计算绝对误差和与实际量绝对值和之比。
        参数: a为实际数组，p为预测数组。
        返回: 百分数WAPE。
        调用位置: _metrics。
        """
        return float(np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9) * 100.0)
    row = pd.DataFrame({"flight": frame["flight"].to_numpy(), "actual_wh": actual * dt / 3600.0,
                        "predicted_wh": predicted * dt / 3600.0})
    row["window"] = row.groupby("flight", sort=False).cumcount() // max(1, int(round(1.0 / cfg.resample_seconds)))
    one = row.groupby(["flight", "window"], sort=False)[["actual_wh", "predicted_wh"]].sum()
    flight = row.groupby("flight", sort=False)[["actual_wh", "predicted_wh"]].sum()
    return {
        "power_mae_w": float(np.mean(np.abs(actual - predicted))),
        "power_rmse_w": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "power_wape_percent": wape(actual, predicted),
        "power_underestimate_rate_percent": float(np.mean(predicted < actual) * 100.0),
        "window_energy_mae_wh": float(np.mean(np.abs(one.actual_wh - one.predicted_wh))),
        "window_energy_rmse_wh": float(np.sqrt(np.mean((one.actual_wh - one.predicted_wh) ** 2))),
        "window_energy_wape_percent": wape(one.actual_wh.to_numpy(), one.predicted_wh.to_numpy()),
        "flight_energy_mae_wh": float(np.mean(np.abs(flight.actual_wh - flight.predicted_wh))),
        "flight_energy_rmse_wh": float(np.sqrt(np.mean((flight.actual_wh - flight.predicted_wh) ** 2))),
        "flight_energy_wape_percent": wape(flight.actual_wh.to_numpy(), flight.predicted_wh.to_numpy()),
        "flight_energy_underestimate_rate_percent": float(np.mean(flight.predicted_wh < flight.actual_wh) * 100.0),
        "rows": int(len(frame)), "flights": int(frame.flight.nunique()),
    }


def _select_score(metrics: dict) -> float:
    """功能: 按统一验证指标口径计算TCN窗口验证选择分数。
    参数: metrics为功率、窗口能耗和flight能耗指标。
    返回: 综合选择分数。
    调用位置: tune_group。
    """

    return metrics["window_energy_wape_percent"] + 0.2 * metrics["power_wape_percent"] + 0.5 * metrics["flight_energy_wape_percent"]


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


def tune_group(group: str, cfg: ExperimentConfig) -> dict:
    """功能: 按配置搜索指定实验组的TCN时间窗和候选训练轮数。
    参数: group为A、B、C、D或E，cfg为配置。
    返回: 最优窗口和调参结果路径。
    调用位置: main.py。
    """

    seed_everything(cfg.random_seed)
    train, val, _ = load_tables(cfg)
    spec = GROUPS[group]
    train = prepare_features(train, spec["use_sma"], cfg.sma_window_steps)
    val = prepare_features(val, spec["use_sma"], cfg.sma_window_steps)
    columns = feature_columns(spec["use_sma"])
    validate_features(train, columns)
    validate_features(val, columns)
    scaler = fit_scaler(train, columns, cfg, group)
    rows, epoch_rows = [], []
    for seconds in cfg.tcn_window_seconds:
        seed_everything(cfg.random_seed)
        steps = max(1, int(round(seconds / cfg.resample_seconds)))
        train_seq, train_y = build_sequences(train, scaler, steps)
        val_seq, val_y = build_sequences(val, scaler, steps)
        checkpoint, logs = train_model(train_seq, train_y, val_seq, val_y, cfg, len(columns), spec["activation"], steps, cfg.tune_epochs)
        model = _model_from_checkpoint(checkpoint, len(columns))
        predicted = predict_array(model, val_seq, scaler, cfg)
        metrics = _metrics(val, predicted, cfg)
        rows.append({"group": group, "window_seconds": seconds, "window_steps": steps,
                     "activation": spec["activation"], "feature_count": len(columns),
                     "best_val_loss": checkpoint["best_val_loss"], "epochs_run": len(logs),
                     "best_epoch": checkpoint["best_epoch"], "training_seconds": checkpoint["training_seconds"],
                     "selection_score": _select_score(metrics), **metrics})
        epoch_rows.extend([{**item, "window_seconds": seconds, "group": group} for item in logs])
        pd.DataFrame(rows).to_csv(group_paths(cfg, group)["tuning"], index=False, encoding="utf-8")
        pd.DataFrame(epoch_rows).to_csv(group_paths(cfg, group)["tuning_log"], index=False, encoding="utf-8")
        print(f"组{group}窗口{seconds:g}s: 选择分数={rows[-1]['selection_score']:.4f}")
    result = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)
    path = group_paths(cfg, group)["tuning"]
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8")
    return {"group": group, "best_window_seconds": float(result.iloc[0].window_seconds),
            "best_window_steps": int(result.iloc[0].window_steps), "tuning_file": str(path)}


def train_group(group: str, cfg: ExperimentConfig) -> dict:
    """功能: 使用已选择窗口按配置正式训练并保存最佳验证模型。
    参数: group为A、B、C、D或E，cfg为配置。
    返回: 模型路径和正式训练摘要。
    调用位置: main.py。
    """

    seed_everything(cfg.random_seed)
    train, val, _ = load_tables(cfg)
    spec = GROUPS[group]
    train = prepare_features(train, spec["use_sma"], cfg.sma_window_steps)
    val = prepare_features(val, spec["use_sma"], cfg.sma_window_steps)
    columns = feature_columns(spec["use_sma"])
    validate_features(train, columns)
    validate_features(val, columns)
    scaler = fit_scaler(train, columns, cfg, group)
    tuning_path = group_paths(cfg, group)["tuning"]
    if not tuning_path.exists():
        tune_group(group, cfg)
    tuning = pd.read_csv(tuning_path).sort_values("selection_score")
    seconds = float(tuning.iloc[0].window_seconds)
    steps = max(1, int(round(seconds / cfg.resample_seconds)))
    seed_everything(cfg.random_seed)
    train_seq, train_y = build_sequences(train, scaler, steps)
    val_seq, val_y = build_sequences(val, scaler, steps)
    checkpoint, logs = train_model(train_seq, train_y, val_seq, val_y, cfg, len(columns), spec["activation"], steps, cfg.epochs)
    checkpoint.update({"group": group, "feature_columns": columns, "raw_feature_columns": RAW_FEATURE_COLUMNS,
                       "sma_window_steps": cfg.sma_window_steps, "device_used": str(resolve_device(cfg)),
                       "use_sma": spec["use_sma"], "seed": cfg.random_seed,
                       "resample_seconds": cfg.resample_seconds, "tune_epochs": cfg.tune_epochs,
                       "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
                       "batch_size": cfg.batch_size, "target_column": "power_w",
                       "schema_version": "AMOVFLY-TCN-comparison-1.0"})
    paths = group_paths(cfg, group)
    torch.save(checkpoint, paths["checkpoint"])
    pd.DataFrame(logs).assign(group=group, window_seconds=seconds, activation=spec["activation"]).to_csv(
        paths["training_log"], index=False, encoding="utf-8"
    )
    return {"group": group, "window_seconds": seconds, "window_steps": steps,
            "feature_count": len(columns), "model_file": str(paths["checkpoint"]),
            "training_log": str(paths["training_log"]), "epochs_run": len(logs),
            "best_val_loss": float(checkpoint["best_val_loss"])}
