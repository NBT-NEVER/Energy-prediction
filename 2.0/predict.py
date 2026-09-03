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
from uncertainty import (
    add_prediction_intervals,
    interval_summary,
    load_calibration_scores,
    save_calibration,
)


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


def predict_from_csv(
    cfg: ExperimentConfig,
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    online_update: bool | None = None,
    confidence: float | None = None,
) -> Path:
    """功能: 读取CSV，执行TCN前向和按flight顺序的RLS在线校正并输出结果。
    参数: cfg为配置，input_csv为输入CSV，output_csv为输出CSV，online_update表示是否用当前实测功率更新RLS，confidence为置信度。
    返回: 预测结果CSV路径。
    调用位置: main.py、evaluate.py。
    """

    ensure_directories(cfg)
    input_csv = input_csv or cfg.test_csv
    output_csv = output_csv or cfg.prediction_csv
    device = resolve_prediction_device(cfg.device)
    print(f"推理设备: {describe_cuda_device(device)}")
    workflow_progress = TerminalProgress("批量预测总流程", 5)
    model, checkpoint, scaler = load_trained_model(cfg, device)
    workflow_progress.update(1, f"已加载 {len(checkpoint['channels'])}块TCN，窗口 {checkpoint['window_seconds']:g}s")
    frame, _ = prepare_prediction_frame(input_csv, cfg)
    frame = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    workflow_progress.update(2, f"已读取 {len(frame)} 条记录、{frame['flight'].nunique()} 个flight")
    for column in scaler["feature_columns"]:
        if column not in frame.columns:
            frame[column] = 0.0
    window_steps = int(checkpoint["window_steps"])
    sequences, _ = build_sequence_arrays(frame, scaler, window_steps, "预测序列构造")
    workflow_progress.update(3, f"已构造 {len(sequences)} 条 {window_steps} 步序列")
    base_power = predict_array(model, sequences, scaler, device, cfg.batch_size)
    if online_update is None:
        online_update = cfg.target_column in frame.columns
    initial_theta = None if online_update else checkpoint.get("rls_theta")
    corrected_power, theta = apply_rls_correction(base_power, frame, scaler, cfg, initial_theta, update=bool(online_update), progress_label="RLS在线校正")
    workflow_progress.update(4, f"RLS校正完成，在线更新={'开启' if online_update else '关闭'}")
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
    confidence = cfg.default_confidence if confidence is None else float(confidence)
    calibration_scores = load_calibration_scores(cfg.uncertainty_calibration_npz, bool(online_update))
    output, radius = add_prediction_intervals(output, calibration_scores, confidence)
    workflow_progress.update(5, f"置信区间已计算：置信度={confidence:.1%}，功率半径={radius:.2f}W")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")
    workflow_progress.finish(f"预测文件已保存：{output_csv.name}")
    return output_csv


def recalculate_prediction_intervals(
    cfg: ExperimentConfig,
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    confidence: float | None = None,
) -> tuple[Path, dict]:
    """功能: 只读取已有点预测并即时重算指定置信度的功率和能耗区间。
    参数: cfg为配置，input_csv为已有预测CSV，output_csv为更新后的CSV，confidence为新的置信度。
    返回: 输出CSV路径和区间摘要字典。
    调用位置: main.py的interval模式。
    """

    ensure_directories(cfg)
    input_csv = input_csv or cfg.prediction_csv
    output_csv = output_csv or input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到已有预测CSV: {input_csv}，请先运行 predict 或 evaluate。")
    frame = pd.read_csv(input_csv)
    if "rls_update_enabled" not in frame.columns:
        raise ValueError("预测CSV缺少rls_update_enabled字段，无法选择对应校准残差。")
    flags = frame["rls_update_enabled"].astype(str).str.lower().isin({"true", "1", "yes"})
    if len(flags) and flags.nunique() > 1:
        raise ValueError("同一个预测CSV同时包含在线和固定RLS状态，不能使用单一校准区间。")
    online_update = bool(flags.iloc[0]) if len(flags) else False
    confidence = cfg.default_confidence if confidence is None else float(confidence)
    scores = load_calibration_scores(cfg.uncertainty_calibration_npz, online_update)
    updated, radius = add_prediction_intervals(frame, scores, confidence)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_csv, index=False, encoding="utf-8")
    summary = interval_summary(updated, "online" if online_update else "static", radius)
    summary["prediction_file"] = str(output_csv)
    print(
        f"即时区间更新完成：置信度={confidence:.1%}，功率区间半径={radius:.2f}W，"
        f"累计能耗区间={summary['total_predicted_energy_lower_wh']:.5f}~{summary['total_predicted_energy_upper_wh']:.5f}Wh"
    )
    return output_csv, summary


def calibrate_uncertainty(cfg: ExperimentConfig) -> dict:
    """功能: 使用验证集预测残差生成在线和固定RLS置信区间校准文件。
    参数: cfg为实验配置对象。
    返回: 校准摘要字典。
    调用位置: main.py的calibrate模式。
    """

    ensure_directories(cfg)
    device = resolve_prediction_device(cfg.device)
    model, checkpoint, scaler = load_trained_model(cfg, device)
    val_frame, _ = prepare_prediction_frame(cfg.val_csv, cfg)
    val_frame = val_frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    for column in scaler["feature_columns"]:
        if column not in val_frame.columns:
            val_frame[column] = 0.0
    progress = TerminalProgress("置信区间校准总流程", 5)
    progress.update(1, f"已读取验证集 {len(val_frame)} 条记录")
    sequences, _ = build_sequence_arrays(val_frame, scaler, int(checkpoint["window_steps"]), "校准序列构造")
    progress.update(2, "验证序列已构造")
    base_power = predict_array(model, sequences, scaler, device, cfg.batch_size, "校准TCN前向")
    progress.update(3, "TCN校准预测已完成")
    actual = val_frame[cfg.target_column].to_numpy(dtype=float)
    online, _ = apply_rls_correction(base_power, val_frame, scaler, cfg, None, update=True, progress_label="在线RLS校准")
    static, _ = apply_rls_correction(base_power, val_frame, scaler, cfg, checkpoint.get("rls_theta"), update=False, progress_label="固定RLS校准")
    progress.update(4, "在线和固定RLS残差已计算")
    summary = save_calibration(
        cfg.uncertainty_calibration_npz,
        cfg.uncertainty_calibration_json,
        np.abs(online - actual),
        np.abs(static - actual),
        cfg.default_confidence,
    )
    progress.finish(f"校准文件已保存，默认置信度={cfg.default_confidence:.1%}")
    return summary
