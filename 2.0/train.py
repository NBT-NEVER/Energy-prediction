# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-08-31
# 文件名: train.py
# 功能说明: 训练实验2.0的TCN模型并联合验证短时间窗与RLS实时校正效果
# 版本号：2.0

import random
from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig, ensure_directories
from data_utils import load_json, save_json
from device_utils import describe_cuda_device, select_cuda_device
from model import RLSCorrector, build_model
from progress import TerminalProgress

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
    调用位置: build_scaler、transform_target。
    """

    values = np.asarray(values, dtype=np.float32)
    if target_transform == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if target_transform == "none":
        return values
    raise ValueError(f"不支持的目标变换: {target_transform}")


def inverse_target_transform(values: np.ndarray, target_transform: str) -> np.ndarray:
    """功能: 将训练目标空间数值还原为原始功率。
    参数: values为反标准化目标数组，target_transform为目标变换名称。
    返回: 原始功率数组。
    调用位置: predict_original_power。
    """

    if target_transform == "log1p":
        return np.expm1(values)
    if target_transform == "none":
        return values
    raise ValueError(f"不支持的目标反变换: {target_transform}")


def build_scaler(train_df: pd.DataFrame, feature_columns: list[str], cfg: ExperimentConfig) -> dict:
    """功能: 根据训练集计算特征、目标和RLS功率尺度。
    参数: train_df为训练集，feature_columns为特征列，cfg为实验配置。
    返回: 可序列化的标准化参数字典。
    调用位置: train_model。
    """

    x = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_raw = train_df[cfg.target_column].to_numpy(dtype=np.float32)
    y = apply_target_transform(y_raw, cfg.target_transform)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    return {
        "feature_columns": feature_columns,
        "target_column": cfg.target_column,
        "target_transform": cfg.target_transform,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y.mean()),
        "y_std": float(y.std() if y.std() >= 1e-6 else 1.0),
        "power_scale": float(max(np.median(y_raw[y_raw > 0]) if np.any(y_raw > 0) else 100.0, 1.0)),
    }


def estimate_sample_interval(frame: pd.DataFrame) -> float:
    """功能: 从有效采样间隔估算数据集的典型采样周期。
    参数: frame为训练数据表。
    返回: 典型采样周期，单位s。
    调用位置: train_model。
    """

    values = frame.loc[(frame["dt_seconds"] > 0) & (frame["dt_seconds"] < 5), "dt_seconds"]
    return float(values.median()) if len(values) else 0.2


def window_seconds_to_steps(window_seconds: float, sample_interval: float) -> int:
    """功能: 将秒级时间窗折算为至少2步的采样长度。
    参数: window_seconds为时间窗秒数，sample_interval为典型采样周期。
    返回: 时间窗采样步数。
    调用位置: train_model。
    """

    return max(2, int(round(float(window_seconds) / max(sample_interval, 1e-6))))


def build_sequence_arrays(frame: pd.DataFrame, scaler: dict, window_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """功能: 按flight构造只含当前及历史信息的左填充TCN短序列。
    参数: frame为已按flight组织的数据表，scaler为标准化参数，window_steps为窗口步数。
    返回: 三维输入序列和标准化目标数组。
    调用位置: train_model、predict.py。
    """

    feature_columns = scaler["feature_columns"]
    ordered = frame.sort_values(["flight", "time"], kind="stable") if "time" in frame.columns else frame.copy()
    raw_x = ordered[feature_columns].to_numpy(dtype=np.float32)
    scaled_x = (raw_x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(scaler["x_std"], dtype=np.float32)
    sequences = np.empty((len(ordered), window_steps, len(feature_columns)), dtype=np.float32)
    for indices in ordered.groupby("flight", sort=False).indices.values():
        group_indices = np.asarray(indices, dtype=int)
        group_x = scaled_x[group_indices]
        padded = np.pad(group_x, ((window_steps - 1, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_steps, axis=0).transpose(0, 2, 1)
        sequences[group_indices] = windows
    if scaler["target_column"] in ordered.columns:
        y = apply_target_transform(ordered[scaler["target_column"]].to_numpy(dtype=np.float32), scaler["target_transform"])
        y = (y - float(scaler["y_mean"])) / float(scaler["y_std"])
    else:
        y = np.zeros(len(ordered), dtype=np.float32)
    return sequences, y.astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """功能: 将TCN序列和目标封装为DataLoader。
    参数: x为序列特征，y为目标，batch_size为批量大小，shuffle表示是否打乱，device为训练设备。
    返回: PyTorch DataLoader。
    调用位置: run_training_loop。
    """

    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")


class BalancedHuberLoss(nn.Module):
    """功能: 对低功率和高功率样本提高损失权重，减轻预测向主流平台收缩。
    参数: target_mean为目标均值，target_std为目标标准差，delta为Huber阈值。
    返回: 每个样本加权后的平均Huber损失。
    调用位置: run_training_loop。
    """

    def __init__(self, target_mean: float, target_std: float, delta: float) -> None:
        super().__init__()
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        self.delta = float(delta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """功能: 计算按功率区间平衡的Huber损失。
        参数: prediction为模型输出，target为标准化目标。
        返回: 标量损失。
        调用位置: run_epoch。
        """

        error = prediction - target
        absolute = error.abs()
        huber = torch.where(absolute <= self.delta, 0.5 * error.square(), self.delta * (absolute - 0.5 * self.delta))
        raw_power = target * self.target_std + self.target_mean
        weights = torch.ones_like(raw_power)
        weights = torch.where(raw_power < 100.0, weights * 1.8, weights)
        weights = torch.where(raw_power > 650.0, weights * 1.5, weights)
        return (huber * weights).mean()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    """功能: 执行一个TCN训练或验证epoch。
    参数: model为模型，loader为数据加载器，criterion为损失函数，optimizer为空时执行验证，device为设备。
    返回: 当前epoch平均损失。
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
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * len(target)
        total_count += len(target)
    return total_loss / max(total_count, 1)


def run_training_loop(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, cfg: ExperimentConfig, params: dict, device: torch.device, epochs: int) -> tuple[nn.Module, list[dict], float]:
    """功能: 按给定TCN和窗口超参数训练模型。
    参数: train_x/train_y/val_x/val_y为序列数据，cfg为配置，params为超参数，device为设备，epochs为轮数。
    返回: 最佳模型、训练日志和最佳验证损失。
    调用位置: train_model。
    """

    model = build_model(train_x.shape[2], tuple(params["channels"]), float(params["dropout"]), int(params["kernel_size"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = BalancedHuberLoss(float(params.get("target_mean", 0.0)), float(params.get("target_std", 1.0)), float(params["huber_delta"]))
    train_loader = make_loader(train_x, train_y, cfg.batch_size, True, device)
    val_loader = make_loader(val_x, val_y, cfg.batch_size, False, device)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    logs: list[dict] = []
    progress = TerminalProgress(f"训练 {params['name']}", epochs)
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr_now, "candidate": params["name"], "model_type": "tcn_rls", "window_seconds": params["window_seconds"], "window_steps": params["window_steps"]})
        progress.update(epoch, f"train={train_loss:.5f}, val={val_loss:.5f}, lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            progress.finish(f"早停，最佳验证损失={best_val:.5f}", completed=False)
            break
    else:
        progress.finish(f"最佳验证损失={best_val:.5f}")
    model.load_state_dict(best_state)
    return model, logs, best_val


def predict_original_power(model: nn.Module, sequences: np.ndarray, scaler: dict, device: torch.device, batch_size: int) -> np.ndarray:
    """功能: 批量执行TCN前向并还原原始功率。
    参数: model为TCN，sequences为标准化短序列，scaler为标准化参数，device为设备，batch_size为批量大小。
    返回: TCN原始功率预测数组。
    调用位置: train_model。
    """

    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            scaled = model(batch).cpu().numpy()
            target = scaled * float(scaler["y_std"]) + float(scaler["y_mean"])
            outputs.append(inverse_target_transform(target, scaler["target_transform"]))
    return np.maximum(np.concatenate(outputs), 0.0)


def apply_rls_correction(base_power: np.ndarray, frame: pd.DataFrame, scaler: dict, cfg: ExperimentConfig, initial_theta: list[float] | None = None, update: bool = True) -> tuple[np.ndarray, list[float]]:
    """功能: 按flight时间顺序执行先预测后更新的RLS实时校正。
    参数: base_power为TCN功率，frame为对应数据，scaler为尺度参数，cfg为配置，initial_theta为初始状态，update表示是否使用实测值更新。
    返回: 校正功率数组和最终RLS参数。
    调用位置: train_model、predict.py。
    """

    corrector = RLSCorrector(cfg.rls_forgetting_factor, cfg.rls_initial_covariance, scaler["power_scale"], initial_theta)
    corrected = np.empty(len(frame), dtype=np.float64)
    previous_flight = None
    actual_values = frame[cfg.target_column].to_numpy(dtype=float) if update and cfg.target_column in frame.columns else None
    flights = frame["flight"].to_numpy() if "flight" in frame.columns else np.zeros(len(frame), dtype=int)
    for index, (flight, base) in enumerate(zip(flights, base_power)):
        if previous_flight is None or flight != previous_flight:
            corrector.reset()
            previous_flight = flight
        corrected[index] = corrector.predict(float(base))
        if actual_values is not None:
            corrector.update(float(base), float(actual_values[index]))
    return corrected.astype(np.float32), corrector.theta.tolist()


def estimate_rls_initial_theta(base_power: np.ndarray, actual_power: np.ndarray, power_scale: float) -> list[float]:
    """功能: 用训练集TCN输出和实测功率估计RLS的全局仿射初值。
    参数: base_power为TCN功率数组，actual_power为实测功率数组，power_scale为功率尺度。
    返回: [偏置, 比例]形式的初始参数。
    调用位置: train_model。
    """

    phi = np.column_stack([np.ones(len(base_power)), base_power / max(float(power_scale), 1.0)])
    theta, *_ = np.linalg.lstsq(phi, actual_power / max(float(power_scale), 1.0), rcond=None)
    return [float(np.clip(theta[0], -1.0, 1.0)), float(np.clip(theta[1], 0.0, 2.0))]


def selection_metrics(base_power: np.ndarray, corrected_power: np.ndarray, frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 计算验证集TCN基线与RLS校正后的功率和飞行能耗WAPE。
    参数: base_power为TCN结果，corrected_power为RLS结果，frame为验证集，cfg为配置。
    返回: 用于候选排序的指标字典。
    调用位置: train_model。
    """

    actual = frame[cfg.target_column].to_numpy(dtype=float)
    sample_base = float(np.sum(np.abs(base_power - actual)) / max(np.sum(np.abs(actual)), 1e-6) * 100.0)
    sample_corrected = float(np.sum(np.abs(corrected_power - actual)) / max(np.sum(np.abs(actual)), 1e-6) * 100.0)
    eval_frame = frame[["flight", "dt_seconds", cfg.target_column]].copy()
    eval_frame["base"] = base_power
    eval_frame["corrected"] = corrected_power
    for column in (cfg.target_column, "base", "corrected"):
        eval_frame[f"{column}_energy"] = eval_frame[column] * eval_frame["dt_seconds"] / 3600.0
    grouped = eval_frame.groupby("flight", sort=True)[[f"{cfg.target_column}_energy", "base_energy", "corrected_energy"]].sum()
    actual_energy = grouped[f"{cfg.target_column}_energy"].to_numpy()
    base_energy_wape = float(np.sum(np.abs(grouped["base_energy"].to_numpy() - actual_energy)) / max(np.sum(np.abs(actual_energy)), 1e-6) * 100.0)
    corrected_energy_wape = float(np.sum(np.abs(grouped["corrected_energy"].to_numpy() - actual_energy)) / max(np.sum(np.abs(actual_energy)), 1e-6) * 100.0)
    return {"val_tcn_sample_power_wape": sample_base, "val_sample_power_wape": sample_corrected, "val_tcn_flight_energy_wape": base_energy_wape, "val_flight_energy_wape": corrected_energy_wape, "selection_score": corrected_energy_wape + 0.2 * sample_corrected}


def candidate_grid(cfg: ExperimentConfig, sample_interval: float) -> list[dict]:
    """功能: 为每个短时间窗生成一个可比较的TCN超参数候选。
    参数: cfg为实验配置，sample_interval为典型采样周期。
    返回: 包含秒级窗口和采样步数的候选列表。
    调用位置: train_model。
    """

    return [{"name": f"tcn_rls_window_{seconds:g}s", "model_type": "tcn_rls", "window_seconds": float(seconds), "window_steps": window_seconds_to_steps(seconds, sample_interval), "channels": [64, 64, 32], "kernel_size": 3, "dropout": 0.08, "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay, "huber_delta": 0.65} for seconds in cfg.window_seconds_candidates]


def save_loss_curve(logs: list[dict], path) -> None:
    """功能: 保存最终TCN训练和验证损失曲线。
    参数: logs为训练日志，path为图片路径。
    返回: None。
    调用位置: train_model。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(logs)
    plt.figure(figsize=(8, 5))
    plt.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    plt.plot(frame["epoch"], frame["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("UAV Energy TCN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_model(cfg: ExperimentConfig) -> dict:
    """功能: 调优短时间窗、训练TCN、标定RLS并保存实验2.0产物。
    参数: cfg为实验配置对象。
    返回: 训练摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    set_random_seed(cfg.random_seed)
    device = select_cuda_device(cfg.device)
    print(f"训练设备: {describe_cuda_device(device)}")
    train_df, val_df, feature_columns = load_split_data(cfg)
    train_df = train_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = build_scaler(train_df, feature_columns, cfg)
    sample_interval = estimate_sample_interval(train_df)
    scaler["sample_interval_seconds"] = sample_interval
    save_json(cfg.scaler_json, scaler)

    tuning_rows: list[dict] = []
    best_params = None
    best_score = float("inf")
    candidates = candidate_grid(cfg, sample_interval)
    for candidate in candidates:
        candidate["target_mean"] = scaler["y_mean"]
        candidate["target_std"] = scaler["y_std"]
    print(f"开始短时间窗调参：共 {len(candidates)} 个候选，每个候选最多训练 {cfg.tune_epochs} 轮。")
    candidate_progress = TerminalProgress("TCN时间窗调参", len(candidates))
    for index, params in enumerate(candidates, start=1):
        train_x, train_y = build_sequence_arrays(train_df, scaler, params["window_steps"])
        val_x, val_y = build_sequence_arrays(val_df, scaler, params["window_steps"])
        model, logs, val_loss = run_training_loop(train_x, train_y, val_x, val_y, cfg, params, device, max(1, cfg.tune_epochs))
        train_base = predict_original_power(model, train_x, scaler, device, cfg.batch_size)
        rls_theta = estimate_rls_initial_theta(train_base, train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"])
        val_base = predict_original_power(model, val_x, scaler, device, cfg.batch_size)
        val_corrected, _ = apply_rls_correction(val_base, val_df, scaler, cfg, rls_theta, update=True)
        metrics = selection_metrics(val_base, val_corrected, val_df, cfg)
        tuning_rows.append({"candidate": params["name"], **params, "channels": str(params["channels"]), "best_val_loss": val_loss, **metrics, "epochs_run": len(logs)})
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_params = params.copy()
        candidate_progress.update(index, f"{params['window_seconds']:g}s，综合分数={metrics['selection_score']:.4f}")
        del train_x, train_y, val_x, val_y, model
        torch.cuda.empty_cache()
    candidate_progress.finish(f"最优窗口={best_params['window_seconds'] if best_params else '无'}s")
    if best_params is None:
        raise RuntimeError("时间窗调参未得到可用候选。")

    train_x, train_y = build_sequence_arrays(train_df, scaler, best_params["window_steps"])
    val_x, val_y = build_sequence_arrays(val_df, scaler, best_params["window_steps"])
    print(f"开始最终训练：窗口 {best_params['window_seconds']:g}s（{best_params['window_steps']}步），最多 {cfg.epochs} 轮。")
    final_model, logs, best_val_loss = run_training_loop(train_x, train_y, val_x, val_y, cfg, best_params, device, max(1, cfg.epochs))
    train_base = predict_original_power(final_model, train_x, scaler, device, cfg.batch_size)
    rls_theta = estimate_rls_initial_theta(train_base, train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"])

    pd.DataFrame(tuning_rows).to_csv(cfg.tuning_results_csv, index=False, encoding="utf-8")
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    save_loss_curve(logs, cfg.loss_curve_file)
    checkpoint = {"model_state_dict": final_model.state_dict(), "input_dim": len(feature_columns), "feature_columns": feature_columns, "target_column": cfg.target_column, "target_transform": cfg.target_transform, "model_type": "tcn_rls", "channels": best_params["channels"], "kernel_size": best_params["kernel_size"], "dropout": best_params["dropout"], "learning_rate": best_params["learning_rate"], "weight_decay": best_params["weight_decay"], "huber_delta": best_params["huber_delta"], "window_seconds": best_params["window_seconds"], "window_steps": best_params["window_steps"], "sample_interval_seconds": sample_interval, "rls_theta": rls_theta, "rls_forgetting_factor": cfg.rls_forgetting_factor, "rls_initial_covariance": cfg.rls_initial_covariance, "power_scale": scaler["power_scale"], "scaler_path": str(cfg.scaler_json), "device_used": str(device), "best_val_loss": best_val_loss, "selection_score": best_score}
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    return {"device": str(device), "cuda_device_name": torch.cuda.get_device_name(device.index or 0), "best_candidate": best_params["name"], "best_model_type": "tcn_rls", "best_window_seconds": best_params["window_seconds"], "best_window_steps": best_params["window_steps"], "sample_interval_seconds": sample_interval, "best_channels": best_params["channels"], "best_dropout": best_params["dropout"], "best_learning_rate": best_params["learning_rate"], "best_weight_decay": best_params["weight_decay"], "rls_forgetting_factor": cfg.rls_forgetting_factor, "rls_initial_theta": rls_theta, "target_transform": cfg.target_transform, "best_val_loss_standardized": best_val_loss, "best_selection_score": best_score, "epochs_run": len(logs), "model_file": str(cfg.best_model_file), "scaler_file": str(cfg.scaler_json)}
