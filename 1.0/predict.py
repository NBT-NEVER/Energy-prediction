# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: predict.py
# 开发时间: 2026-07-08
# 文件名: predict.py
# 功能说明: 加载训练好的MLP模型并预测四轴无人机飞行电功率和区间电能
# 版本号：1.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig, ensure_directories
from data_utils import load_json
from data_utils import prepare_prediction_frame
from model import build_model


def resolve_prediction_device(device_name: str) -> torch.device:
    """功能: 选择预测设备。
    参数: device_name为命令行指定设备。
    返回: torch.device设备对象。
    调用位置: predict_from_csv、visualize.py。
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_scaler_path(cfg: ExperimentConfig, checkpoint: dict) -> Path:
    """功能: 查找当前或旧版标准化参数文件。
    参数: cfg为实验配置对象，checkpoint为模型权重字典。
    返回: 可读取的scaler路径。
    调用位置: load_trained_model。
    """

    candidates = [cfg.scaler_json]
    checkpoint_path = checkpoint.get("scaler_path")
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    candidates.append(cfg.save_dir / "scaler_1.0.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到标准化参数: {cfg.scaler_json}")


def load_trained_model(cfg: ExperimentConfig, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    """功能: 加载模型权重和标准化参数。
    参数: cfg为实验配置对象，device为加载设备。
    返回: 模型、checkpoint字典和scaler字典。
    调用位置: predict_from_csv、evaluate.py、visualize.py。
    """

    if not cfg.best_model_file.exists():
        raise FileNotFoundError(f"未找到模型权重: {cfg.best_model_file}")
    checkpoint = torch.load(cfg.best_model_file, map_location=device, weights_only=False)
    model = build_model(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
        model_type=str(checkpoint.get("model_type", "plain")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = load_json(resolve_scaler_path(cfg, checkpoint))
    return model, checkpoint, scaler


def inverse_target_transform(values: np.ndarray, target_transform: str) -> np.ndarray:
    """功能: 将模型输出从训练目标空间还原为功率值。
    参数: values为反标准化后的目标数组，target_transform为训练目标变换名称。
    返回: 还原后的功率数组。
    调用位置: predict_array。
    """

    if target_transform == "log1p":
        return np.expm1(values)
    if target_transform == "none":
        return values
    raise ValueError(f"不支持的目标反变换: {target_transform}")


def predict_array(model: torch.nn.Module, x: np.ndarray, scaler: dict, device: torch.device, batch_size: int) -> np.ndarray:
    """功能: 对标准化前的特征矩阵进行批量预测并反标准化。
    参数: model为模型，x为原始特征矩阵，scaler为标准化参数，device为设备，batch_size为批量大小。
    返回: 反标准化后的功率预测数组。
    调用位置: predict_from_csv、evaluate.py、visualize.py。
    """

    x_mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    x_std = np.asarray(scaler["x_std"], dtype=np.float32)
    x_scaled = ((x.astype(np.float32) - x_mean) / x_std).astype(np.float32)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start : start + batch_size]).to(device)
            pred_scaled = model(batch).detach().cpu().numpy()
            pred_target = pred_scaled * float(scaler["y_std"]) + float(scaler["y_mean"])
            pred = inverse_target_transform(pred_target, scaler.get("target_transform", "none"))
            predictions.append(pred)
    return np.maximum(np.concatenate(predictions), 0.0)


def predict_from_csv(cfg: ExperimentConfig, input_csv: Path | None = None, output_csv: Path | None = None) -> Path:
    """功能: 读取CSV并输出能耗预测结果。
    参数: cfg为实验配置对象，input_csv为输入CSV，output_csv为预测结果路径。
    返回: 预测结果CSV路径。
    调用位置: main.py、evaluate.py。
    """

    ensure_directories(cfg)
    input_csv = input_csv or cfg.test_csv
    output_csv = output_csv or cfg.prediction_csv
    device = resolve_prediction_device(cfg.device)
    model, _, scaler = load_trained_model(cfg, device)
    frame, feature_columns = prepare_prediction_frame(input_csv, cfg)
    for column in scaler["feature_columns"]:
        if column not in frame.columns:
            frame[column] = 0.0
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    pred_power = predict_array(model, x, scaler, device, cfg.batch_size)

    output = frame.copy()
    output["predicted_power_w"] = pred_power
    if "dt_seconds" in output.columns:
        output["predicted_energy_wh"] = output["predicted_power_w"] * output["dt_seconds"] / 3600.0
    if cfg.target_column in output.columns and "dt_seconds" in output.columns:
        output["actual_energy_wh"] = output[cfg.target_column] * output["dt_seconds"] / 3600.0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv
