# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-09-16
# 文件名: train.py
# 功能说明: 执行实验4.1规划TCN的8窗搜索、正式训练、RLS 7乘7搜索和预测
# 版本号：4.1

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from data_utils import FEATURE_COLUMNS, save_json
from model import build_model
from terminal_logger import log_result


TARGET_COLUMN = "power_w"
UNCERTAINTY_FEATURES = [
    "wind_east_mps", "wind_north_mps", "payload_g", "task_duration_s",
    "time_s", "planned_motor_on", "planned_airborne",
]


def seed_everything(seed: int) -> None:
    """功能: 固定4.1训练过程的随机种子。
    参数: seed为随机种子。
    返回: None。
    调用位置: train_tcn。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cuda_device(cfg: ExperimentConfig) -> torch.device:
    """功能: 强制实验4.1训练和推理使用用户指定的CUDA设备。
    参数: cfg为实验配置对象。
    返回: 已验证可用的torch CUDA设备。
    调用位置: _train_one、_predict_array。
    """

    requested = str(cfg.device).strip().lower()
    if not requested.startswith("cuda"):
        raise RuntimeError(f"实验4.1按约定只允许GPU训练和推理，当前device={cfg.device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("实验4.1要求使用GPU，但当前PyTorch未检测到可用CUDA设备")
    device = torch.device(requested)
    try:
        torch.cuda.get_device_properties(device)
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(f"无法使用指定CUDA设备 {requested}: {exc}") from exc
    return device


def load_tables(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """功能: 读取4.1独立训练、验证和测试表。
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

    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(train.columns))
    if missing:
        raise ValueError(f"训练集缺少标准化所需字段: {missing}")
    x = train[FEATURE_COLUMNS].to_numpy(float)
    y = train[TARGET_COLUMN].to_numpy(float)
    x_mean, x_std = x.mean(0), x.std(0)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
    scaler = {"feature_columns": FEATURE_COLUMNS, "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
              "y_mean": y_mean, "y_std": y_std, "target_column": TARGET_COLUMN,
              "resample_seconds": cfg.resample_seconds,
              "version": "4.1", "task_before_only": True,
              "input_dimension": len(FEATURE_COLUMNS)}
    cfg.scaler_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.scaler_json.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    return scaler


def load_scaler(cfg: ExperimentConfig) -> dict:
    """功能: 读取4.1独立标准化文件。
    参数: cfg为实验配置对象。
    返回: 标准化参数字典。
    调用位置: predict_frame、evaluate、task_api。
    """

    return json.loads(cfg.scaler_json.read_text(encoding="utf-8"))


def build_sequences(frame: pd.DataFrame, scaler: dict, window_steps: int,
                    require_target: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """功能: 按完整flight构造只包含当前及历史规划状态的TCN序列。
    参数: frame为规划特征表，scaler为标准化参数，window_steps为时间窗步数，require_target控制是否读取监督标签。
    返回: 序列张量、目标功率和原始行索引。
    调用位置: train_tcn、predict_frame。
    """

    mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    std = np.asarray(scaler["x_std"], dtype=np.float32)
    feature_columns = list(scaler["feature_columns"])
    missing_features = sorted(set(feature_columns) - set(frame.columns))
    if missing_features:
        raise ValueError(f"模型输入缺少规划特征: {missing_features}")
    target_column = str(scaler.get("target_column", TARGET_COLUMN))
    if require_target and target_column not in frame:
        raise ValueError(f"监督序列缺少目标字段: {target_column}")
    sequences, targets, indices = [], [], []
    for _, group in frame.sort_values(["flight", "time"], kind="stable").groupby("flight", sort=False):
        values = (group[feature_columns].to_numpy(np.float32) - mean) / std
        target = (group[target_column].to_numpy(np.float32) if require_target
                  else np.zeros(len(group), dtype=np.float32))
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
    """功能: 在GPU上执行一轮TCN训练或验证。
    参数: model为TCN，loader为批数据，optimizer为训练优化器或None，device为CUDA设备。
    返回: 当前轮平均SmoothL1损失。
    调用位置: _train_one。
    """

    model.train(optimizer is not None)
    losses = []
    criterion = nn.SmoothL1Loss()
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
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
    """功能: 在GPU上批量推理并还原基础动力功率量纲。
    参数: model为TCN，sequences为标准化前序列，scaler为目标缩放参数，cfg为配置。
    返回: 非负基础动力功率预测数组。
    调用位置: predict_frame、tune_tcn、train_tcn。
    """

    device = resolve_cuda_device(cfg)
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), cfg.batch_size):
            batch = torch.from_numpy(sequences[start:start + cfg.batch_size]).to(device, non_blocking=True)
            outputs.append(model(batch).cpu().numpy())
    values = np.concatenate(outputs) * float(scaler["y_std"]) + float(scaler["y_mean"])
    return np.maximum(values, 0.0)


def predict_frame(frame: pd.DataFrame, cfg: ExperimentConfig, checkpoint_path: Path | None = None) -> np.ndarray:
    """功能: 使用4.1最终模型对规划特征表进行逐0.2秒功率预测。
    参数: frame为只含规划字段的表，cfg为配置对象，checkpoint_path为可选权重路径。
    返回: 非负功率预测数组，顺序与frame按flight/time排序后的顺序一致。
    调用位置: evaluate.py、task_api.py。
    """

    scaler = load_scaler(cfg)
    path = checkpoint_path or cfg.final_model_file
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if scaler.get("target_column") != TARGET_COLUMN or checkpoint.get("target_column") != TARGET_COLUMN:
        raise RuntimeError("当前scaler或模型权重不是以power_w为目标训练的，请重新执行4.1 GPU训练")
    if list(scaler.get("feature_columns", [])) != FEATURE_COLUMNS or list(checkpoint.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise RuntimeError(f"当前scaler或模型权重与4.1的{len(FEATURE_COLUMNS)}维规划特征不一致，请重新训练")
    ordered = frame.sort_values(["flight", "time"], kind="stable").reset_index(drop=True).copy()
    sequences, _, _ = build_sequences(ordered, scaler, int(checkpoint["window_steps"]), require_target=False)
    model = build_model(len(scaler["feature_columns"]), tuple(checkpoint["channels"]), checkpoint["kernel_size"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return _predict_array(model, sequences, scaler, cfg)


def _train_one(train_seq: np.ndarray, train_y: np.ndarray, val_seq: np.ndarray, val_y: np.ndarray,
               cfg: ExperimentConfig, window_steps: int, epochs: int) -> tuple[dict, list[dict]]:
    """功能: 针对一个TCN时间窗完成指定轮数的GPU训练并保留最佳验证权重。
    参数: train_seq/train_y为训练数据，val_seq/val_y为验证数据，cfg为配置，window_steps为窗口步数，epochs为轮数。
    返回: 最佳检查点和逐轮训练日志。
    调用位置: tune_tcn、train_tcn。
    """

    device = resolve_cuda_device(cfg)
    model = build_model(len(FEATURE_COLUMNS), cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_y)),
        batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_y)),
        batch_size=cfg.batch_size, shuffle=False, pin_memory=True,
    )
    best_state, best_loss, wait, log = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        log.append({"window_steps": window_steps, "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        print(
            f"\rTCN训练窗口={window_steps}步 | epoch {epoch}/{epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f}",
            end="",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    print()
    return {"state_dict": best_state, "best_val_loss": best_loss, "window_steps": window_steps,
            "channels": list(cfg.tcn_channels), "kernel_size": cfg.tcn_kernel_size, "dropout": cfg.tcn_dropout}, log


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """功能: 计算加权绝对百分比误差。
    参数: actual为真实数组，predicted为预测数组。
    返回: 百分数形式的WAPE。
    调用位置: tune_tcn。
    """

    return float(np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), 1e-6) * 100.0)


def _tcn_selection_metrics(frame: pd.DataFrame, predicted_power: np.ndarray,
                           cfg: ExperimentConfig) -> dict:
    """功能: 用时间步功率、1秒窗口能耗和整条flight能耗共同评价TCN时间窗。
    参数: frame为验证集，predicted_power为候选预测，cfg提供0.2秒离散步长。
    返回: 与3.2同用途的三类WAPE和综合选择分数。
    调用位置: tune_tcn。
    """

    actual_power = frame[TARGET_COLUMN].to_numpy(float)
    dt = frame["dt_seconds"].to_numpy(float)
    metrics = frame[["flight"]].copy()
    metrics["actual_energy_wh"] = actual_power * dt / 3600.0
    metrics["predicted_energy_wh"] = np.asarray(predicted_power, dtype=float) * dt / 3600.0
    one_second_steps = max(1, int(round(1.0 / cfg.resample_seconds)))
    metrics["one_second_window"] = metrics.groupby("flight", sort=False).cumcount() // one_second_steps
    one_second = metrics.groupby(["flight", "one_second_window"], sort=False)[
        ["actual_energy_wh", "predicted_energy_wh"]
    ].sum()
    flight = metrics.groupby("flight", sort=False)[
        ["actual_energy_wh", "predicted_energy_wh"]
    ].sum()
    sample_wape = _wape(actual_power, predicted_power)
    second_wape = _wape(
        one_second["actual_energy_wh"].to_numpy(float),
        one_second["predicted_energy_wh"].to_numpy(float),
    )
    flight_wape = _wape(
        flight["actual_energy_wh"].to_numpy(float),
        flight["predicted_energy_wh"].to_numpy(float),
    )
    selection_score = second_wape + 0.2 * sample_wape + 0.5 * flight_wape
    return {
        "val_wape_percent": sample_wape,
        "val_tcn_sample_power_wape": sample_wape,
        "val_tcn_second_energy_wape": second_wape,
        "val_tcn_flight_energy_wape": flight_wape,
        "tcn_selection_score": selection_score,
    }


def _training_scaler(train: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 读取与当前规划特征及基础动力功率目标一致的标准化器，必要时重建。
    参数: train为训练集，cfg为实验配置对象。
    返回: 当前实验可用的标准化参数。
    调用位置: tune_tcn、train_tcn。
    """

    if cfg.scaler_json.exists():
        scaler = load_scaler(cfg)
        if (scaler.get("target_column") == TARGET_COLUMN
                and list(scaler.get("feature_columns", [])) == FEATURE_COLUMNS):
            return scaler
    return fit_scaler(train, cfg)


def _uncertainty_design(frame: pd.DataFrame) -> np.ndarray:
    """功能: 构造动态不确定性模型使用的风场、载荷、任务状态和速度设计矩阵。
    参数: frame为验证集或任务前规划特征表。
    返回: 七列条件变量矩阵。
    调用位置: _fit_uncertainty_model。
    """

    return np.column_stack([
        frame["wind_east_mps"].to_numpy(float),
        frame["wind_north_mps"].to_numpy(float),
        frame["payload_g"].to_numpy(float),
        frame["task_duration_s"].to_numpy(float),
        frame["time_s"].to_numpy(float),
        frame["planned_motor_on"].to_numpy(float),
        frame["planned_airborne"].to_numpy(float),
    ])


def _fit_uncertainty_model(frame: pd.DataFrame, residual_w: np.ndarray, cfg: ExperimentConfig) -> dict:
    """功能: 用验证集残差拟合受风场、载荷、任务长度、进度和速度影响的动态功率标准差。
    参数: frame为验证集规划特征，residual_w为基础动力功率残差，cfg为配置对象。
    返回: 可由任务前接口直接读取的异方差模型参数。
    调用位置: train_tcn。
    """

    residual = np.asarray(residual_w, dtype=float)
    design = _uncertainty_design(frame)
    mean = design.mean(axis=0)
    std = design.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (design - mean) / std
    matrix = np.column_stack([np.ones(len(normalized)), normalized])
    target = np.log(np.maximum(np.abs(residual), 1e-3))
    ridge = np.eye(matrix.shape[1], dtype=float) * 1e-3
    ridge[0, 0] = 0.0
    coefficients = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target)
    raw_sigma = np.exp(np.clip(matrix @ coefficients, -20.0, 20.0)) * np.sqrt(np.pi / 2.0)
    global_std = max(float(np.sqrt(np.mean(residual ** 2))), 1e-3)
    scale = global_std / max(float(np.sqrt(np.mean(raw_sigma ** 2))), 1e-9)
    floor = max(float(np.quantile(np.abs(residual), 0.1)), global_std * 0.1, 1e-3)
    calibrated_sigma = np.maximum(raw_sigma * scale, floor)
    calibration = pd.DataFrame({
        "flight": frame["flight"].to_numpy(),
        "residual_energy_wh": residual * frame["dt_seconds"].to_numpy(float) / 3600.0,
        "sigma_energy_wh": calibrated_sigma * frame["dt_seconds"].to_numpy(float) / 3600.0,
    })
    task_scores = []
    for _, group in calibration.groupby("flight", sort=False):
        independent_std = float(np.sqrt(np.sum(group["sigma_energy_wh"].to_numpy(float) ** 2)))
        task_error = abs(float(group["residual_energy_wh"].sum()))
        task_scores.append(task_error / max(independent_std, 1e-12))
    ordered_scores = np.sort(np.asarray(task_scores, dtype=float))
    conformal_rank = min(
        max(int(np.ceil((len(ordered_scores) + 1) * cfg.default_confidence)), 1),
        len(ordered_scores),
    )
    normal_radius = 1.96 if cfg.default_confidence >= 0.95 else 1.645
    task_multiplier = max(float(ordered_scores[conformal_rank - 1]), normal_radius)
    calibrated_coverage = float(np.mean(ordered_scores <= task_multiplier) * 100.0)
    return {
        "version": "4.1",
        "target_column": TARGET_COLUMN,
        "base_residual_std_w": global_std,
        "confidence": cfg.default_confidence,
        "interval_method": "逐时刻条件残差方差合成；任务级标准差按方差平方和计算，不线性累加固定半径",
        "condition_features": UNCERTAINTY_FEATURES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "log_abs_error_coefficients": coefficients.tolist(),
        "sigma_scale": float(scale),
        "sigma_floor_w": float(floor),
        "calibration_rows": int(len(frame)),
        "task_energy_interval_method": "按验证集完整flight总能耗残差进行有限样本共形校准",
        "task_energy_radius_multiplier": task_multiplier,
        "task_energy_calibration_flights": int(len(ordered_scores)),
        "task_energy_conformal_rank": int(conformal_rank),
        "task_energy_calibration_coverage_percent": calibrated_coverage,
    }


def tune_tcn(cfg: ExperimentConfig) -> dict:
    """功能: 搜索8个TCN时间窗候选并保存选择表。
    参数: cfg为实验配置对象。
    返回: 8个候选的验证指标和最优窗口摘要。
    调用位置: main.py。
    """

    seed_everything(cfg.random_seed)
    train, val, _ = load_tables(cfg)
    train = train.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val = val.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = _training_scaler(train, cfg)
    rows, logs = [], []
    for seconds in cfg.tcn_window_seconds:
        steps = max(1, int(round(seconds / cfg.resample_seconds)))
        train_seq, train_y, _ = build_sequences(train, scaler, steps)
        val_seq, val_y, _ = build_sequences(val, scaler, steps)
        checkpoint, epoch_log = _train_one(train_seq, train_y, val_seq, val_y, cfg, steps, cfg.tune_epochs)
        model = build_model(len(FEATURE_COLUMNS), cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout)
        model.load_state_dict(checkpoint["state_dict"])
        predicted = _predict_array(model, val_seq, scaler, cfg)
        actual = val[TARGET_COLUMN].to_numpy(float)
        candidate_name = f"tcn_window_{seconds:g}s"
        selection = _tcn_selection_metrics(val, predicted, cfg)
        candidate_result = {
            "candidate": candidate_name, "window_seconds": seconds, "window_steps": steps,
            **selection, "val_mae_w": float(np.mean(np.abs(actual - predicted))),
            "best_val_loss": checkpoint["best_val_loss"], "epochs_run": len(epoch_log),
        }
        rows.append(candidate_result)
        print(
            f"TCN候选 {candidate_name}：样本WAPE={selection['val_tcn_sample_power_wape']:.4f}% | "
            f"1秒能耗WAPE={selection['val_tcn_second_energy_wape']:.4f}% | "
            f"flight能耗WAPE={selection['val_tcn_flight_energy_wape']:.4f}% | "
            f"选择分数={selection['tcn_selection_score']:.4f}"
        )
        log_result(f"TCN候选结果 {candidate_name}", candidate_result)
        for item in epoch_log:
            item.update({"phase": "tcn_tuning", "candidate": candidate_name,
                         "window_seconds": float(seconds)})
        logs.extend(epoch_log)
    result = pd.DataFrame(rows).sort_values("tcn_selection_score").reset_index(drop=True)
    cfg.tcn_tuning_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.tcn_tuning_csv, index=False, encoding="utf-8")
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    return {"candidate_count": len(result), "best_window_seconds": float(result.iloc[0].window_seconds),
            "best_window_steps": int(result.iloc[0].window_steps),
            "best_tcn_selection_score": float(result.iloc[0].tcn_selection_score),
            "tuning_file": str(cfg.tcn_tuning_csv)}


def train_tcn(cfg: ExperimentConfig) -> dict:
    """功能: 按最优TCN窗口完成4.1正式训练并保存独立权重。
    参数: cfg为实验配置对象。
    返回: 最终窗口、训练状态和权重路径摘要。
    调用位置: main.py。
    """

    seed_everything(cfg.random_seed)
    train, val, _ = load_tables(cfg)
    train = train.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val = val.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = _training_scaler(train, cfg)
    if cfg.tcn_tuning_csv.exists():
        tuning = pd.read_csv(cfg.tcn_tuning_csv)
        score_column = "tcn_selection_score" if "tcn_selection_score" in tuning else "val_wape_percent"
        tuning = tuning.sort_values(score_column)
        seconds = float(tuning.iloc[0].window_seconds)
    else:
        seconds = cfg.tcn_window_seconds[0]
    steps = max(1, int(round(seconds / cfg.resample_seconds)))
    train_seq, train_y, _ = build_sequences(train, scaler, steps)
    val_seq, val_y, _ = build_sequences(val, scaler, steps)
    checkpoint, logs = _train_one(train_seq, train_y, val_seq, val_y, cfg, steps, cfg.epochs)
    checkpoint["feature_columns"] = FEATURE_COLUMNS
    checkpoint["target_column"] = TARGET_COLUMN
    checkpoint["resample_seconds"] = cfg.resample_seconds
    checkpoint["window_seconds"] = seconds
    checkpoint["device_used"] = str(resolve_cuda_device(cfg))
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    for item in logs:
        item.update({"phase": "final_tcn", "candidate": "final_tcn",
                     "window_seconds": float(seconds)})
    if cfg.training_log_csv.exists():
        previous = pd.read_csv(cfg.training_log_csv)
        if "phase" in previous:
            previous = previous[previous["phase"] != "final_tcn"]
        else:
            previous = previous.iloc[0:0]
        combined = pd.concat([previous, pd.DataFrame(logs)], ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(logs)
    combined.to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    model = build_model(len(FEATURE_COLUMNS), cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dropout)
    model.load_state_dict(checkpoint["state_dict"])
    val_predicted = _predict_array(model, val_seq, scaler, cfg)
    uncertainty = _fit_uncertainty_model(val, val[TARGET_COLUMN].to_numpy(float) - val_predicted, cfg)
    save_json(cfg.uncertainty_json, uncertainty)
    return {"window_seconds": seconds, "window_steps": steps, "epochs_run": len(logs),
            "best_val_loss": float(checkpoint["best_val_loss"]), "model_file": str(cfg.final_model_file)}


def tune_rls(cfg: ExperimentConfig) -> dict:
    """功能: 在验证集上搜索7乘7共49组RLS参数。
    参数: cfg为实验配置对象。
    返回: RLS最优参数和完整搜索表路径。
    调用位置: main.py。
    """

    from evaluate import _simulate, evaluate_rls_parameters

    train, val, _ = load_tables(cfg)
    val = val.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    baseline = predict_frame(val, cfg)

    def window_residual_std(frame: pd.DataFrame, prediction: np.ndarray) -> float:
        """按完整1秒窗口计算验证集基础动力能耗残差标准差。"""

        step_count = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
        residuals: list[float] = []
        for _, group in frame.groupby("flight", sort=False):
            indices = group.index.to_numpy()
            row_residual = (
                group[TARGET_COLUMN].to_numpy(float) - prediction[indices]
            ) * group["dt_seconds"].to_numpy(float) / 3600.0
            for start in range(0, len(group), step_count):
                window = row_residual[start:start + step_count]
                if len(window) == step_count:
                    residuals.append(float(window.sum()))
        if not residuals:
            raise RuntimeError("验证集没有可用于RLS先验残差估计的完整窗口")
        return max(float(np.sqrt(np.mean(np.square(residuals)))), 1e-6)

    deployment_initial_residual_std_wh = window_residual_std(val, baseline)
    rows = []
    for factor in cfg.rls_forgetting_factors:
        for covariance in cfg.rls_initial_covariances:
            metric = evaluate_rls_parameters(val, baseline, factor, covariance, cfg)
            candidate_result = {
                "forgetting_factor": factor,
                "initial_covariance": covariance,
                "selection_scope": "all_validation_flights",
                "validation_flights": int(val["flight"].nunique()),
                **metric,
            }
            rows.append(candidate_result)
            print(
                f"RLS候选 ff={factor:g}, cov={covariance:g} | "
                f"flight能耗WAPE={metric['flight_energy_wape_percent']:.4f}%"
            )
            log_result(f"RLS候选结果 ff={factor:g}, cov={covariance:g}", candidate_result)
    result = pd.DataFrame(rows).sort_values("flight_energy_wape_percent").reset_index(drop=True)
    cfg.rls_tuning_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.rls_tuning_csv, index=False, encoding="utf-8")
    best = result.iloc[0]

    # 区间倍率使用完整flight交叉拟合，保证每个校准flight未参与对应折的RLS参数选择。
    flight_ids = sorted(val["flight"].drop_duplicates().tolist(), key=lambda value: str(value))
    shuffled_ids = np.asarray(flight_ids, dtype=object)
    np.random.default_rng(cfg.random_seed).shuffle(shuffled_ids)
    fold_count = min(5, len(shuffled_ids))
    calibration_traces: list[dict] = []
    window_calibration_rows: list[pd.DataFrame] = []
    fold_records: list[dict] = []
    for fold_index, calibration_ids in enumerate(np.array_split(shuffled_ids, fold_count), start=1):
        calibration_set = set(calibration_ids.tolist())
        calibration_mask = val["flight"].isin(calibration_set).to_numpy()
        selection_mask = ~calibration_mask
        selection_frame = val.loc[selection_mask].reset_index(drop=True)
        selection_baseline = baseline[selection_mask]
        fold_initial_residual_std_wh = window_residual_std(
            selection_frame, selection_baseline,
        )
        fold_candidates = []
        for factor in cfg.rls_forgetting_factors:
            for covariance in cfg.rls_initial_covariances:
                metric = evaluate_rls_parameters(
                    selection_frame, selection_baseline, factor, covariance, cfg,
                )
                fold_candidates.append({
                    "forgetting_factor": factor,
                    "initial_covariance": covariance,
                    **metric,
                })
        fold_best = min(fold_candidates, key=lambda item: item["flight_energy_wape_percent"])
        fold_trace: list[dict] = []
        _, fold_predictions = _simulate(
            val.loc[calibration_mask].reset_index(drop=True),
            baseline[calibration_mask],
            float(fold_best["forgetting_factor"]),
            float(fold_best["initial_covariance"]),
            cfg,
            keep_rows=True,
            correction_trace=fold_trace,
            interval_scale=1.0,
            initial_residual_std_wh=fold_initial_residual_std_wh,
            window_interval_scale=1.0,
        )
        assert fold_predictions is not None
        step_count = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
        fold_windows = fold_predictions.assign(
            window_index=fold_predictions.groupby("flight").cumcount() // step_count,
            calibration_fold=fold_index,
        ).groupby(["flight", "window_index", "calibration_fold"], sort=False).agg(
            window_steps=("dt_seconds", "size"),
            actual_energy_wh=("actual_energy_wh", "sum"),
            predicted_energy_wh=("predicted_energy_wh", "sum"),
            lower_energy_wh=("predicted_energy_lower_wh", "sum"),
            upper_energy_wh=("predicted_energy_upper_wh", "sum"),
        ).reset_index()
        window_calibration_rows.append(
            fold_windows[
                fold_windows["window_index"].gt(0)
                & fold_windows["window_steps"].eq(step_count)
            ]
        )
        for item in fold_trace:
            item["calibration_fold"] = fold_index
        calibration_traces.extend(fold_trace)
        fold_records.append({
            "fold": fold_index,
            "selection_flights": [str(value) for value in shuffled_ids if value not in calibration_set],
            "calibration_flights": [str(value) for value in calibration_ids.tolist()],
            "selected_forgetting_factor": float(fold_best["forgetting_factor"]),
            "selected_initial_covariance": float(fold_best["initial_covariance"]),
            "selection_flight_energy_wape_percent": float(fold_best["flight_energy_wape_percent"]),
            "initial_residual_std_wh": fold_initial_residual_std_wh,
        })

    trace = pd.DataFrame(calibration_traces)
    active = trace[
        trace["observed_window_count"].gt(0)
        & trace["remaining_window_count"].gt(0)
    ].copy()
    actual_remaining = (
        active["actual_final_energy_wh"] - active["observed_actual_energy_wh"]
    ).clip(lower=0.0)
    center = active["corrected_remaining_energy_wh"].to_numpy(float)
    raw_radius = np.maximum(
        active["remaining_upper_wh"].to_numpy(float) - center,
        center - active["remaining_lower_wh"].to_numpy(float),
    )
    scores = np.abs(actual_remaining.to_numpy(float) - center) / np.maximum(raw_radius, 1e-12)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise RuntimeError("RLS区间校准没有可用的验证集post-update轨迹点")
    def finite_sample_scale(values: np.ndarray) -> tuple[np.ndarray, int, float]:
        ordered = np.sort(values[np.isfinite(values)])
        if not len(ordered):
            raise RuntimeError("RLS区间校准没有可用的有限非一致性分数")
        rank = min(
            max(int(np.ceil((len(ordered) + 1) * cfg.default_confidence)), 1),
            len(ordered),
        )
        return ordered, rank, max(float(ordered[rank - 1]), 1.0)

    ordered_scores, conformal_rank, interval_scale = finite_sample_scale(scores)
    window_calibration = pd.concat(window_calibration_rows, ignore_index=True)
    window_center = window_calibration["predicted_energy_wh"].to_numpy(float)
    window_radius = np.maximum(
        window_calibration["upper_energy_wh"].to_numpy(float) - window_center,
        window_center - window_calibration["lower_energy_wh"].to_numpy(float),
    )
    window_scores = (
        np.abs(window_calibration["actual_energy_wh"].to_numpy(float) - window_center)
        / np.maximum(window_radius, 1e-12)
    )
    ordered_window_scores, window_conformal_rank, window_interval_scale = finite_sample_scale(
        window_scores,
    )
    calibration = {
        "version": "4.1",
        "method": "validation_flight_cross_fitted_post_update_conformal_scale",
        "confidence": cfg.default_confidence,
        "score_definition": "abs(actual_remaining_wh - corrected_remaining_wh) / unscaled_interval_radius_wh",
        "validation_flights": int(len(shuffled_ids)),
        "cross_fit_folds": int(fold_count),
        "calibration_points": int(len(ordered_scores)),
        "conformal_rank": int(conformal_rank),
        "interval_scale": interval_scale,
        "window_interval_scale": window_interval_scale,
        "initial_residual_std_wh": deployment_initial_residual_std_wh,
        "initial_residual_source": "validation_complete_1s_window_base_energy_residual_rmse",
        "raw_coverage_percent": float(np.mean(ordered_scores <= 1.0) * 100.0),
        "calibrated_coverage_percent": float(np.mean(ordered_scores <= interval_scale) * 100.0),
        "window_calibration_points": int(len(ordered_window_scores)),
        "window_conformal_rank": int(window_conformal_rank),
        "window_raw_coverage_percent": float(np.mean(ordered_window_scores <= 1.0) * 100.0),
        "window_calibrated_coverage_percent": float(
            np.mean(ordered_window_scores <= window_interval_scale) * 100.0
        ),
        "median_nonconformity_score": float(np.median(ordered_scores)),
        "maximum_nonconformity_score": float(np.max(ordered_scores)),
        "deployment_forgetting_factor": float(best.forgetting_factor),
        "deployment_initial_covariance": float(best.initial_covariance),
        "test_set_used_for_selection_or_calibration": False,
        "folds": fold_records,
    }
    save_json(cfg.rls_interval_calibration_json, calibration)
    return {
        "candidate_count": len(result),
        "best_forgetting_factor": float(best.forgetting_factor),
        "best_initial_covariance": float(best.initial_covariance),
        "interval_scale": interval_scale,
        "window_interval_scale": window_interval_scale,
        "initial_residual_std_wh": deployment_initial_residual_std_wh,
        "interval_calibration_points": int(len(ordered_scores)),
        "tuning_file": str(cfg.rls_tuning_csv),
        "interval_calibration_file": str(cfg.rls_interval_calibration_json),
    }
