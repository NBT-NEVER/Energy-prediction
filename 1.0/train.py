# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-07-07
# 文件名: train.py
# 功能说明: 使用GPU训练四轴无人机电能能耗预测MLP模型并记录调参结果
# 版本号：1.0

import random
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig, ensure_directories
from data_utils import load_json, save_json
from model import build_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def set_random_seed(seed: int) -> None:
    """功能: 固定Python、NumPy和PyTorch随机种子。
    参数: seed为随机种子。
    返回: None。
    调用位置: train_model。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str, require_gpu: bool) -> torch.device:
    """功能: 选择训练设备并按要求检查GPU可用性。
    参数: device_name为命令行指定设备，require_gpu表示是否强制CUDA。
    返回: torch.device设备对象。
    调用位置: train_model。
    """

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if require_gpu and device.type != "cuda":
        raise RuntimeError("当前配置要求GPU训练，但PyTorch未检测到CUDA设备。")
    return device


def load_split_data(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """功能: 读取训练集、验证集和特征元数据。
    参数: cfg为实验配置对象。
    返回: train_df、val_df和特征列名列表。
    调用位置: train_model。
    """

    if not cfg.train_csv.exists() or not cfg.val_csv.exists() or not cfg.feature_meta_json.exists():
        raise FileNotFoundError("未找到处理后训练数据，请先运行 prepare 或 all。")
    meta = load_json(cfg.feature_meta_json)
    return pd.read_csv(cfg.train_csv), pd.read_csv(cfg.val_csv), meta["feature_columns"]


def apply_target_transform(values: np.ndarray, target_transform: str) -> np.ndarray:
    """功能: 对功率目标做训练前变换。
    参数: values为原始功率数组，target_transform为变换名称。
    返回: 变换后的目标数组。
    调用位置: build_scaler、transform_frame。
    """

    values = np.asarray(values, dtype=np.float32)
    if target_transform == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if target_transform == "none":
        return values.astype(np.float32)
    raise ValueError(f"不支持的目标变换: {target_transform}")


def build_scaler(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_transform: str,
) -> dict:
    """功能: 根据训练集计算特征和目标标准化参数。
    参数: train_df为训练集，feature_columns为特征列，target_column为目标列。
    返回: 可序列化的标准化参数字典。
    调用位置: train_model。
    """

    x = train_df[feature_columns].to_numpy(dtype=np.float32)
    y = apply_target_transform(train_df[target_column].to_numpy(dtype=np.float32), target_transform)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y.mean())
    y_std = float(y.std() if y.std() >= 1e-6 else 1.0)
    return {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_transform": target_transform,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def transform_frame(frame: pd.DataFrame, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    """功能: 按训练集标准化参数转换特征和目标。
    参数: frame为数据表，scaler为标准化参数。
    返回: 标准化后的X和y数组。
    调用位置: train_model、evaluate.py。
    """

    feature_columns = scaler["feature_columns"]
    target_column = scaler["target_column"]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = apply_target_transform(frame[target_column].to_numpy(dtype=np.float32), scaler.get("target_transform", "none"))
    x = (x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(scaler["x_std"], dtype=np.float32)
    y = (y - float(scaler["y_mean"])) / float(scaler["y_std"])
    return x.astype(np.float32), y.astype(np.float32)


def inverse_target_transform(values: np.ndarray, target_transform: str) -> np.ndarray:
    """功能: 将训练目标空间的数值还原为原始功率。
    参数: values为反标准化后的目标数组，target_transform为目标变换名称。
    返回: 原始功率数组。
    调用位置: predict_original_power。
    """

    if target_transform == "log1p":
        return np.expm1(values)
    if target_transform == "none":
        return values
    raise ValueError(f"不支持的目标反变换: {target_transform}")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """功能: 将NumPy数组封装为PyTorch DataLoader。
    参数: x为特征，y为目标，batch_size为批量大小，shuffle表示是否打乱，device为训练设备。
    返回: PyTorch DataLoader。
    调用位置: run_training_loop。
    """

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """功能: 执行一个训练或验证epoch。
    参数: model为模型，loader为数据加载器，criterion为损失函数，optimizer为空时执行验证。
    返回: 当前epoch的平均损失。
    调用位置: run_training_loop。
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = criterion(prediction, target)
        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += float(loss.item()) * len(target)
        total_count += len(target)
    return total_loss / max(total_count, 1)


def predict_original_power(
    model: nn.Module,
    raw_x: np.ndarray,
    scaler: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """功能: 将候选模型输出还原为原始功率预测。
    参数: model为候选模型，raw_x为未标准化特征，scaler为标准化参数，device为设备，batch_size为批量大小。
    返回: 原始功率预测数组。
    调用位置: validation_selection_metrics。
    """

    x = (raw_x.astype(np.float32) - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(
        scaler["x_std"], dtype=np.float32
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred_scaled = model(batch).detach().cpu().numpy()
            pred_target = pred_scaled * float(scaler["y_std"]) + float(scaler["y_mean"])
            predictions.append(inverse_target_transform(pred_target, scaler.get("target_transform", "none")))
    return np.maximum(np.concatenate(predictions), 0.0)


def validation_selection_metrics(
    model: nn.Module,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    scaler: dict,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    """功能: 计算候选模型的验证集样本功率和flight级能耗指标。
    参数: model为候选模型，val_df为验证集，feature_columns为特征列，scaler为标准化参数，cfg为配置，device为设备。
    返回: 验证集选择指标字典。
    调用位置: train_model。
    """

    raw_x = val_df[feature_columns].to_numpy(dtype=np.float32)
    pred_power = predict_original_power(model, raw_x, scaler, device, cfg.batch_size)
    actual_power = val_df[cfg.target_column].to_numpy(dtype=float)
    sample_wape = float(np.sum(np.abs(pred_power - actual_power)) / np.maximum(np.sum(np.abs(actual_power)), 1e-6) * 100.0)

    eval_frame = val_df[["flight", "dt_seconds", cfg.target_column]].copy()
    eval_frame["predicted_power_w"] = pred_power
    eval_frame["predicted_energy_wh"] = eval_frame["predicted_power_w"] * eval_frame["dt_seconds"] / 3600.0
    eval_frame["actual_energy_wh"] = eval_frame[cfg.target_column] * eval_frame["dt_seconds"] / 3600.0
    grouped = eval_frame.groupby("flight", sort=True).agg(
        actual_energy_wh=("actual_energy_wh", "sum"),
        predicted_energy_wh=("predicted_energy_wh", "sum"),
    )
    energy_error = grouped["predicted_energy_wh"].to_numpy() - grouped["actual_energy_wh"].to_numpy()
    actual_energy = grouped["actual_energy_wh"].to_numpy()
    flight_wape = float(np.sum(np.abs(energy_error)) / np.maximum(np.sum(np.abs(actual_energy)), 1e-6) * 100.0)
    selection_score = float(flight_wape + 0.2 * sample_wape)
    return {
        "val_sample_power_wape": sample_wape,
        "val_flight_energy_wape": flight_wape,
        "selection_score": selection_score,
    }


def run_training_loop(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: ExperimentConfig,
    params: dict,
    device: torch.device,
    epochs: int,
) -> tuple[nn.Module, list[dict], float]:
    """功能: 按给定超参数训练模型并返回最佳验证损失。
    参数: train_x/train_y/val_x/val_y为标准化数据，cfg为配置，params为超参数，device为设备，epochs为轮数。
    返回: 最佳模型、日志列表和最佳验证损失。
    调用位置: train_model。
    """

    model = build_model(
        input_dim=train_x.shape[1],
        hidden_dims=tuple(params["hidden_dims"]),
        dropout=float(params["dropout"]),
        model_type=str(params["model_type"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.HuberLoss(delta=float(params.get("huber_delta", 0.65)))
    train_loader = make_loader(train_x, train_y, cfg.batch_size, True, device)
    val_loader = make_loader(val_x, val_y, cfg.batch_size, False, device)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    logs: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr_now,
                "candidate": params["name"],
                "model_type": params["model_type"],
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, logs, best_val


def candidate_grid(cfg: ExperimentConfig) -> list[dict]:
    """功能: 生成实验1.0的小规模调参候选。
    参数: cfg为实验配置对象。
    返回: 超参数候选列表。
    调用位置: train_model。
    """

    return [
        {
            "name": "plain_192_96_48",
            "model_type": "plain",
            "hidden_dims": [192, 96, 48],
            "dropout": 0.08,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "huber_delta": 0.65,
        },
        {
            "name": "plain_256_128_64",
            "model_type": "plain",
            "hidden_dims": [256, 128, 64],
            "dropout": 0.08,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "huber_delta": 0.65,
        },
        {
            "name": "residual_128_64_32",
            "model_type": "residual",
            "hidden_dims": [128, 64, 32],
            "dropout": 0.05,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "huber_delta": 0.65,
        },
        {
            "name": "residual_192_96_48",
            "model_type": "residual",
            "hidden_dims": [192, 96, 48],
            "dropout": 0.06,
            "learning_rate": cfg.learning_rate * 0.8,
            "weight_decay": cfg.weight_decay,
            "huber_delta": 0.65,
        },
        {
            "name": "residual_256_128_64",
            "model_type": "residual",
            "hidden_dims": [256, 128, 64],
            "dropout": 0.08,
            "learning_rate": cfg.learning_rate * 0.8,
            "weight_decay": cfg.weight_decay * 0.5,
            "huber_delta": 0.65,
        },
    ]


def save_loss_curve(logs: list[dict], path: Path) -> None:
    """功能: 保存训练和验证损失曲线。
    参数: logs为训练日志，path为图片路径。
    返回: None。
    调用位置: train_model。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    log_frame = pd.DataFrame(logs)
    plt.figure(figsize=(8, 5))
    plt.plot(log_frame["epoch"], log_frame["train_loss"], label="train_loss")
    plt.plot(log_frame["epoch"], log_frame["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("UAV Energy MLP Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_model(cfg: ExperimentConfig) -> dict:
    """功能: 执行调参、GPU训练并保存模型权重和训练记录。
    参数: cfg为实验配置对象。
    返回: 训练摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    set_random_seed(cfg.random_seed)
    device = select_device(cfg.device, cfg.require_gpu)
    train_df, val_df, feature_columns = load_split_data(cfg)
    scaler = build_scaler(train_df, feature_columns, cfg.target_column, cfg.target_transform)
    save_json(cfg.scaler_json, scaler)

    train_x, train_y = transform_frame(train_df, scaler)
    val_x, val_y = transform_frame(val_df, scaler)
    tuning_rows = []
    best_params = None
    best_selection_score = float("inf")
    for params in candidate_grid(cfg):
        tune_model, tune_logs, tune_loss = run_training_loop(
            train_x,
            train_y,
            val_x,
            val_y,
            cfg,
            params,
            device,
            max(1, cfg.tune_epochs),
        )
        selection_metrics = validation_selection_metrics(tune_model, val_df, feature_columns, scaler, cfg, device)
        tuning_rows.append(
            {
                "candidate": params["name"],
                "model_type": params["model_type"],
                "hidden_dims": str(params["hidden_dims"]),
                "dropout": params["dropout"],
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "huber_delta": params["huber_delta"],
                "best_val_loss": tune_loss,
                "val_sample_power_wape": selection_metrics["val_sample_power_wape"],
                "val_flight_energy_wape": selection_metrics["val_flight_energy_wape"],
                "selection_score": selection_metrics["selection_score"],
                "epochs_run": len(tune_logs),
            }
        )
        if selection_metrics["selection_score"] < best_selection_score:
            best_selection_score = selection_metrics["selection_score"]
            best_params = params
    if best_params is None:
        raise RuntimeError("调参未得到可用候选模型。")

    final_model, logs, best_val_loss = run_training_loop(
        train_x,
        train_y,
        val_x,
        val_y,
        cfg,
        best_params,
        device,
        max(1, cfg.epochs),
    )
    pd.DataFrame(tuning_rows).to_csv(cfg.tuning_results_csv, index=False, encoding="utf-8")
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    save_loss_curve(logs, cfg.loss_curve_file)

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "input_dim": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": cfg.target_column,
        "target_transform": cfg.target_transform,
        "model_type": best_params["model_type"],
        "hidden_dims": best_params["hidden_dims"],
        "dropout": best_params["dropout"],
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
        "huber_delta": best_params["huber_delta"],
        "scaler_path": str(cfg.scaler_json),
        "device_used": str(device),
        "best_val_loss": best_val_loss,
        "selection_score": best_selection_score,
    }
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    summary = {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "best_candidate": best_params["name"],
        "best_model_type": best_params["model_type"],
        "best_hidden_dims": best_params["hidden_dims"],
        "best_dropout": best_params["dropout"],
        "best_learning_rate": best_params["learning_rate"],
        "best_weight_decay": best_params["weight_decay"],
        "target_transform": cfg.target_transform,
        "best_val_loss_standardized": best_val_loss,
        "best_selection_score": best_selection_score,
        "epochs_run": len(logs),
        "model_file": str(cfg.best_model_file),
        "scaler_file": str(cfg.scaler_json),
    }
    return summary
