# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-09-16
# 文件名: train.py
# 功能说明: 执行实验4.0规划TCN的8窗搜索、正式训练、RLS 7乘7搜索和预测
# 版本号：4.0

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig, ensure_directories
from data_utils import FEATURE_COLUMNS, prepare_dataset, save_json
from model import DynamicEnergyRLS, build_model


def seed_everything(seed: int) -> None:
    """功能: 固定4.0训练过程的随机种子。
    参数: seed为随机种子。
    返回: None。
    调用位置: train_tcn。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tables(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """功能: 读取4.0独立训练、验证和测试表。
    参数: cfg为实验配置对象。
    返回: train、val、test数据表。
    调用位置: train_tcn、evaluate.py。
    """

    return tuple(pd.read_csv(path) for path in (cfg.train_csv, cfg.val_csv, cfg.test_csv))


def fit_scaler(train: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 只用训练集估计规划特征和功率目标的标准化参数。
    参数: train为训练集，cfg为配置对象。
    返回: 可序列化的标准化参数。
    调用位置: train_tcn。
    """

    x = train[FEATURE_COLUMNS].to_numpy(float)
    y = train["power_w"].to_numpy(float)
    x_mean, x_std = x.mean(0), x.std(0)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
    scaler = {"feature_columns": FEATURE_COLUMNS, "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
              "y_mean": y_mean, "y_std": y_std, "resample_seconds": cfg.resample_seconds,
              "version": "4.0", "task_before_only": True}
    cfg.scaler_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.scaler_json.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    return scaler


def load_scaler(cfg: ExperimentConfig) -> dict:
    """功能: 读取4.0独立标准化文件。
    参数: cfg为实验配置对象。
    返回: 标准化参数字典。
    调用位置: predict_frame、evaluate、task_api。
    """

    return json.loads(cfg.scaler_json.read_text(encoding="utf-8"))


def build_sequences(frame: pd.DataFrame, scaler: dict, window_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """功能: 按完整flight构造只包含当前及历史规划状态的TCN序列。
    参数: frame为规划特征表，scaler为标准化参数，window_steps为时间窗步数。
    返回: 序列张量、目标功率和原始行索引。
    调用位置: train_tcn、predict_frame。
    """

    mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    std = np.asarray(scaler["x_std"], dtype=np.float32)
    sequences, targets, indices = [], [], []
    for _, group in frame.sort_values(["flight", "time"], kind="stable").groupby("flight", sort=False):
        values = (group[scaler["feature_columns"]].to_numpy(np.float32) - mean) / std
        target = group["power_w"].to_numpy(np.float32)
        rows = group.index.to_numpy()
        for position in range(len(group)):
            start = max(0, position - window_steps + 1)
            window = values[start:position + 1]
            if len(window) < window_steps:
                window = np.vstack([np.repeat(window[:1], window_steps - len(window), axis=0), window])
            sequences.append(window)
            targets.append((target[position] - scaler["y_mean"]) / scaler["y_std"])
            indices.append(rows[position])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(indices)


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train(optimizer is not None)
    losses = []
    criterion = nn.SmoothL1Loss()
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        output = model(features)
        loss = criterion(output, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _predict_array(model: nn.Module, sequences: np.ndarray, scaler: dict, cfg: ExperimentConfig) -> np.ndarray:
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), cfg.batch_size):
            batch = torch.from_numpy(sequences[start:start + cfg.batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    values = np.concatenate(outputs) * float(scaler["y_std"]) + float(scaler["y_mean"])
    return np.maximum(values, 0.0)


def predict_frame(frame: pd.DataFrame, cfg: ExperimentConfig, checkpoint_path: Path | None = None) -> np.ndarray:
    """功能: 使用4.0最终模型对规划特征表进行逐0.2秒功率预测。
    参数: frame为只含规划字段的表，cfg为配置对象，checkpoint_path为可选权重路径。
    返回: 非负功率预测数组，顺序与frame按flight/time排序后的顺序一致。
    调用位置: evaluate.py、task_api.py。
    """

    scaler = load_scaler(cfg)
    path = checkpoint_path or cfg.final_model_file
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ordered = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True).copy()
    sequences, _, _ = build_sequences(ordered, scaler, int(checkpoint["window_steps"]))
    model = build_model(len(scaler["feature_columns"]), tuple(checkpoint["channels"]), checkpoint["kernel_size"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return _predict_array(model, sequences, scaler, cfg)


def _train_one(train_seq: np.ndarray, train_y: np.ndarray, val_seq: np.ndarray, val_y: np.ndarray,
               cfg: ExperimentConfig, window_steps: int, epochs: int) -> tuple[dict, list[dict]]:
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(len(FEATURE_COLUMNS), cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_y)), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_y)), batch_size=cfg.batch_size, shuffle=False)
    best_state, best_loss, wait, log = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        log.append({"window_steps": window_steps, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    return {"state_dict": best_state, "best_val_loss": best_loss, "window_steps": window_steps,
            "channels": list(cfg.tcn_channels), "kernel_size": cfg.tcn_kernel_size, "dropout": cfg.tcn_dropout}, log


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), 1e-6) * 100.0)


def tune_tcn(cfg: ExperimentConfig) -> dict:
    """功能: 搜索8个TCN时间窗候选并保存选择表。
    参数: cfg为实验配置对象。
    返回: 8个候选的验证指标和最优窗口摘要。
    调用位置: main.py。
    """

    train, val, _ = load_tables(cfg)
    train = train.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val = val.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = load_scaler(cfg)
    rows, logs = [], []
    for seconds in cfg.tcn_window_seconds:
        steps = max(1, int(round(seconds / cfg.resample_seconds)))
        train_seq, train_y, _ = build_sequences(train, scaler, steps)
        val_seq, val_y, _ = build_sequences(val, scaler, steps)
        checkpoint, epoch_log = _train_one(train_seq, train_y, val_seq, val_y, cfg, steps, cfg.tune_epochs)
        model = build_model(len(FEATURE_COLUMNS), cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout)
        model.load_state_dict(checkpoint["state_dict"])
        predicted = _predict_array(model, val_seq, scaler, cfg)
        actual = val["power_w"].to_numpy(float)
        rows.append({"window_seconds": seconds, "window_steps": steps, "val_wape_percent": _wape(actual, predicted),
                     "val_mae_w": float(np.mean(np.abs(actual - predicted))), "best_val_loss": checkpoint["best_val_loss"], "epochs_run": len(epoch_log)})
        logs.extend(epoch_log)
    result = pd.DataFrame(rows).sort_values("val_wape_percent").reset_index(drop=True)
    cfg.tcn_tuning_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.tcn_tuning_csv, index=False, encoding="utf-8")
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    return {"candidate_count": len(result), "best_window_seconds": float(result.iloc[0].window_seconds),
            "best_window_steps": int(result.iloc[0].window_steps), "tuning_file": str(cfg.tcn_tuning_csv)}


def train_tcn(cfg: ExperimentConfig) -> dict:
    """功能: 按最优TCN窗口完成4.0正式训练并保存独立权重。
    参数: cfg为实验配置对象。
    返回: 最终窗口、训练状态和权重路径摘要。
    调用位置: main.py。
    """

    seed_everything(cfg.random_seed)
    train, val, _ = load_tables(cfg)
    scaler = load_scaler(cfg)
    if cfg.tcn_tuning_csv.exists():
        tuning = pd.read_csv(cfg.tcn_tuning_csv).sort_values("val_wape_percent")
        seconds = float(tuning.iloc[0].window_seconds)
    else:
        seconds = cfg.tcn_window_seconds[0]
    steps = max(1, int(round(seconds / cfg.resample_seconds)))
    train_seq, train_y, _ = build_sequences(train, scaler, steps)
    val_seq, val_y, _ = build_sequences(val, scaler, steps)
    checkpoint, logs = _train_one(train_seq, train_y, val_seq, val_y, cfg, steps, cfg.epochs)
    checkpoint["feature_columns"] = FEATURE_COLUMNS
    checkpoint["resample_seconds"] = cfg.resample_seconds
    checkpoint["window_seconds"] = seconds
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    save_json(cfg.uncertainty_json, {"version": "4.0", "base_residual_std_w": float(np.std(val_y) * scaler["y_std"]),
                                     "interval_method": "任务级方差合成，不按时间步固定半径线性累加"})
    return {"window_seconds": seconds, "window_steps": steps, "epochs_run": len(logs),
            "best_val_loss": float(checkpoint["best_val_loss"]), "model_file": str(cfg.final_model_file)}


def tune_rls(cfg: ExperimentConfig) -> dict:
    """功能: 在验证集上搜索7乘7共49组RLS参数。
    参数: cfg为实验配置对象。
    返回: RLS最优参数和完整搜索表路径。
    调用位置: main.py。
    """

    from evaluate import evaluate_rls_parameters

    train, val, _ = load_tables(cfg)
    val = val.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    baseline = predict_frame(val, cfg)
    rows = []
    for factor in cfg.rls_forgetting_factors:
        for covariance in cfg.rls_initial_covariances:
            metric = evaluate_rls_parameters(val, baseline, factor, covariance, cfg)
            rows.append({"forgetting_factor": factor, "initial_covariance": covariance, **metric})
    result = pd.DataFrame(rows).sort_values("flight_energy_wape_percent").reset_index(drop=True)
    cfg.rls_tuning_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.rls_tuning_csv, index=False, encoding="utf-8")
    best = result.iloc[0]
    return {"candidate_count": len(result), "best_forgetting_factor": float(best.forgetting_factor),
            "best_initial_covariance": float(best.initial_covariance), "tuning_file": str(cfg.rls_tuning_csv)}
