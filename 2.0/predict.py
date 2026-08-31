# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: predict.py
# 开发时间: 2026-08-31
# 文件名: predict.py
# 功能说明: 执行实验2.0的TCN前向推理和RLS实时功率校正
# 版本号：2.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig, ensure_directories
from data_utils import load_json, prepare_prediction_frame
from device_utils import describe_cuda_device, select_cuda_device
from model import RLSCorrector, build_model
from progress import TerminalProgress
from train import apply_rls_correction, build_sequence_arrays, inverse_target_transform


def resolve_prediction_device(device_name: str) -> torch.device:
    """功能: 选择预测设备。
    参数: device_name为命令行指定设备。
    返回: torch.device设备对象。
    调用位置: predict_from_csv。
    """

    return select_cuda_device(device_name)


def resolve_scaler_path(cfg: ExperimentConfig, checkpoint: dict) -> Path:
    """功能: 查找实验2.0标准化参数文件。
    参数: cfg为实验配置对象，checkpoint为模型权重字典。
    返回: 可读取的标准化参数路径。
    调用位置: load_trained_model。
    """

    candidates = [cfg.scaler_json]
    if checkpoint.get("scaler_path"):
        candidates.append(Path(checkpoint["scaler_path"]))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到标准化参数: {cfg.scaler_json}")


def load_trained_model(cfg: ExperimentConfig, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    """功能: 加载TCN权重、模型元数据和标准化参数。
    参数: cfg为实验配置对象，device为加载设备。
    返回: 模型、checkpoint字典和scaler字典。
    调用位置: predict_from_csv、evaluate.py、visualize.py。
    """

    if not cfg.best_model_file.exists():
        raise FileNotFoundError(f"未找到模型权重: {cfg.best_model_file}")
    checkpoint = torch.load(cfg.best_model_file, map_location=device, weights_only=False)
    model = build_model(int(checkpoint["input_dim"]), tuple(checkpoint["channels"]), float(checkpoint["dropout"]), int(checkpoint.get("kernel_size", 3))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = load_json(resolve_scaler_path(cfg, checkpoint))
    return model, checkpoint, scaler


def predict_array(model: torch.nn.Module, sequences: np.ndarray, scaler: dict, device: torch.device, batch_size: int, progress_label: str = "TCN前向推理") -> np.ndarray:
    """功能: 批量执行TCN前向推理并还原功率。
    参数: model为TCN模型，sequences为标准化短序列，scaler为标准化参数，device为设备，batch_size为批大小，progress_label为进度标签。
    返回: TCN原始功率预测数组。
    调用位置: predict_from_csv、visualize.py。
    """

    predictions: list[np.ndarray] = []
    progress = TerminalProgress(progress_label, max((len(sequences) + batch_size - 1) // batch_size, 1))
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(sequences), batch_size), start=1):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            pred_scaled = model(batch).cpu().numpy()
            pred_target = pred_scaled * float(scaler["y_std"]) + float(scaler["y_mean"])
            predictions.append(inverse_target_transform(pred_target, scaler.get("target_transform", "none")))
            progress.update(batch_index, f"已处理 {min(start + batch_size, len(sequences))}/{len(sequences)} 条记录")
    progress.finish("TCN前向完成")
    return np.maximum(np.concatenate(predictions), 0.0)


def predict_from_csv(cfg: ExperimentConfig, input_csv: Path | None = None, output_csv: Path | None = None, online_update: bool | None = None) -> Path:
    """功能: 读取CSV，执行TCN前向和按flight顺序的RLS在线校正并输出结果。
    参数: cfg为配置，input_csv为输入CSV，output_csv为输出CSV，online_update表示是否用当前实测功率更新RLS。
    返回: 预测结果CSV路径。
    调用位置: main.py、evaluate.py。
    """

    ensure_directories(cfg)
    input_csv = input_csv or cfg.test_csv
    output_csv = output_csv or cfg.prediction_csv
    device = resolve_prediction_device(cfg.device)
    print(f"推理设备: {describe_cuda_device(device)}")
    model, checkpoint, scaler = load_trained_model(cfg, device)
    frame, _ = prepare_prediction_frame(input_csv, cfg)
    frame = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    for column in scaler["feature_columns"]:
        if column not in frame.columns:
            frame[column] = 0.0
    window_steps = int(checkpoint["window_steps"])
    sequences, _ = build_sequence_arrays(frame, scaler, window_steps)
    base_power = predict_array(model, sequences, scaler, device, cfg.batch_size)
    if online_update is None:
        online_update = cfg.target_column in frame.columns
    initial_theta = None if online_update else checkpoint.get("rls_theta")
    corrected_power, theta = apply_rls_correction(base_power, frame, scaler, cfg, initial_theta, update=bool(online_update))
    output = frame.copy()
    output["tcn_predicted_power_w"] = base_power
    output["rls_corrected_power_w"] = corrected_power
    output["predicted_power_w"] = corrected_power
    output["rls_correction_w"] = corrected_power - base_power
    output["rls_update_enabled"] = bool(online_update)
    if "dt_seconds" in output.columns:
        output["tcn_predicted_energy_wh"] = output["tcn_predicted_power_w"] * output["dt_seconds"] / 3600.0
        output["predicted_energy_wh"] = output["predicted_power_w"] * output["dt_seconds"] / 3600.0
    if cfg.target_column in output.columns and "dt_seconds" in output.columns:
        output["actual_energy_wh"] = output[cfg.target_column] * output["dt_seconds"] / 3600.0
    output["rls_final_theta_0"] = float(theta[0])
    output["rls_final_theta_1"] = float(theta[1])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv
