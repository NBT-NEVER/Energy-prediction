# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: train.py
# 开发时间: 2026-09-07
# 文件名: train.py
# 功能说明: 执行实验3.0的TCN窗口选择、秒级能量监督RLS网格选择和最终训练
# 版本号：3.0

import random
import time
from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

from config import ExperimentConfig, ensure_directories
from data_utils import load_json, save_json
from device_utils import describe_cuda_device, select_cuda_device
from model import RLSCorrector, build_model
from progress import TerminalProgress
from terminal_logger import log_result
from uncertainty import save_calibration

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# CUDA训练启用TF32，降低矩阵计算耗时；不影响CPU运行路径。
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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


def build_sequence_arrays(frame: pd.DataFrame, scaler: dict, window_steps: int, progress_label: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """功能: 按flight构造只含当前及历史信息的左填充TCN短序列。
    参数: frame为已按flight组织的数据表，scaler为标准化参数，window_steps为窗口步数。
    返回: 三维输入序列和标准化目标数组。
    progress_label为可选的序列构造进度标题。
    调用位置: train_model、predict.py。
    """

    feature_columns = scaler["feature_columns"]
    ordered = frame.sort_values(["flight", "time"], kind="stable") if "time" in frame.columns else frame.copy()
    raw_x = ordered[feature_columns].to_numpy(dtype=np.float32)
    scaled_x = (raw_x - np.asarray(scaler["x_mean"], dtype=np.float32)) / np.asarray(scaler["x_std"], dtype=np.float32)
    sequences = np.empty((len(ordered), window_steps, len(feature_columns)), dtype=np.float32)
    groups = list(ordered.groupby("flight", sort=False).indices.values())
    progress = TerminalProgress(progress_label, len(groups)) if progress_label else None
    for group_number, indices in enumerate(groups, start=1):
        group_indices = np.asarray(indices, dtype=int)
        group_x = scaled_x[group_indices]
        padded = np.pad(group_x, ((window_steps - 1, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_steps, axis=0).transpose(0, 2, 1)
        sequences[group_indices] = windows
        if progress:
            progress.update(group_number, f"flight {group_number}/{len(groups)}，样本 {len(group_indices)}")
    if progress:
        progress.finish("序列构造完成")
    if scaler["target_column"] in ordered.columns:
        y = apply_target_transform(ordered[scaler["target_column"]].to_numpy(dtype=np.float32), scaler["target_transform"])
        y = (y - float(scaler["y_mean"])) / float(scaler["y_std"])
    else:
        y = np.zeros(len(ordered), dtype=np.float32)
    return sequences, y.astype(np.float32)


class CompleteWindowBatchSampler(Sampler[list[int]]):
    """功能: 按完整秒窗口打包样本，确保能量积分窗口不会被批次拆开。
    参数: group_ids为秒窗口编号，batch_size为批次目标样本数，shuffle表示是否打乱窗口。
    返回: 每次迭代返回一个样本索引列表。
    调用位置: make_loader。
    """

    def __init__(self, group_ids: np.ndarray, batch_size: int, shuffle: bool) -> None:
        self.groups = [np.asarray(indices, dtype=int).tolist() for indices in pd.Series(np.arange(len(group_ids))).groupby(group_ids, sort=False).apply(list)]
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.shuffle(order)
        batch: list[int] = []
        for group_index in order:
            group = self.groups[int(group_index)]
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, int(np.ceil(sum(len(group) for group in self.groups) / self.batch_size)))


def build_energy_group_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """功能: 为训练样本生成秒窗口编号、真实时间间隔和完整窗口标记。
    参数: frame为按flight和time排序的数据表。
    返回: group_ids、dt_seconds和complete_mask数组。
    调用位置: train_model、run_training_loop。
    """

    second_window = np.floor(frame["time"].to_numpy(dtype=float)).astype(np.int64)
    keys = pd.MultiIndex.from_arrays([frame["flight"].to_numpy(), second_window])
    group_ids, _ = pd.factorize(keys, sort=False)
    dt = frame["dt_seconds"].to_numpy(dtype=np.float32)
    duration = pd.Series(dt).groupby(group_ids).transform("sum").to_numpy(dtype=np.float32)
    complete = duration >= 0.95
    return group_ids.astype(np.int64), dt, complete.astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, group_ids: np.ndarray, dt: np.ndarray, complete: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """功能: 将TCN序列和目标封装为DataLoader。
    参数: x为序列特征，y为目标，batch_size为批量大小，shuffle表示是否打乱，device为训练设备。
    返回: PyTorch DataLoader。
    调用位置: run_training_loop。
    """

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(group_ids), torch.from_numpy(dt), torch.from_numpy(complete))
    sampler = CompleteWindowBatchSampler(group_ids, batch_size, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")


class BalancedHuberLoss(nn.Module):
    """功能: 对低功率和高功率样本提高损失权重，减轻预测向主流平台收缩。
    参数: target_mean为目标均值，target_std为目标标准差，delta为Huber阈值。
    返回: 每个样本加权后的平均Huber损失。
    调用位置: run_training_loop。
    """

    def __init__(self, target_mean: float, target_std: float, delta: float, target_transform: str, power_scale: float, energy_weight: float = 0.2) -> None:
        super().__init__()
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        self.delta = float(delta)
        self.target_transform = target_transform
        self.energy_scale = max(float(power_scale) / 3600.0, 1e-6)
        self.energy_weight = float(energy_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor, dt: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
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
        point_loss = (huber * weights).mean()
        if self.energy_weight <= 0 or not torch.any(complete > 0.5):
            return point_loss
        prediction_target = prediction * self.target_std + self.target_mean
        actual_target = target * self.target_std + self.target_mean
        if self.target_transform == "log1p":
            prediction_power = torch.expm1(prediction_target).clamp_min(0.0)
            actual_power = torch.expm1(actual_target).clamp_min(0.0)
        else:
            prediction_power = prediction_target.clamp_min(0.0)
            actual_power = actual_target.clamp_min(0.0)
        _, inverse_groups = torch.unique(group_ids, sorted=False, return_inverse=True)
        group_count = int(inverse_groups.max().item()) + 1
        integration_dt = dt.to(dtype=prediction_power.dtype)
        predicted_energy = torch.zeros(group_count, device=prediction.device, dtype=prediction_power.dtype)
        actual_energy = torch.zeros_like(predicted_energy)
        group_complete = torch.zeros_like(predicted_energy)
        predicted_energy.scatter_add_(0, inverse_groups, prediction_power * integration_dt / 3600.0)
        actual_energy.scatter_add_(0, inverse_groups, actual_power.to(prediction_power.dtype) * integration_dt / 3600.0)
        group_complete.scatter_add_(0, inverse_groups, complete.to(prediction_power.dtype))
        valid = group_complete > 0.5
        predicted_energy = predicted_energy[valid]
        actual_energy = actual_energy[valid]
        energy_error = (predicted_energy - actual_energy) / self.energy_scale
        energy_loss = torch.nn.functional.smooth_l1_loss(energy_error, torch.zeros_like(energy_error), beta=self.delta)
        return point_loss + self.energy_weight * energy_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler | None = None,
    progress_label: str | None = None,
) -> float:
    """功能: 执行一个TCN训练或验证epoch。
    参数: model为模型，loader为数据加载器，criterion为损失函数，optimizer为空时执行验证，device为设备，grad_scaler为CUDA混合精度梯度缩放器。
    返回: 当前epoch平均损失。
    progress_label为可选的批次进度标题。
    调用位置: run_training_loop。
    """

    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = device.type == "cuda"
    total_loss = torch.zeros((), device=device)
    total_count = 0
    progress = TerminalProgress(progress_label, len(loader), width=24) if progress_label else None
    for batch_index, (features, target, group_ids, dt, complete) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        group_ids = group_ids.to(device, non_blocking=True)
        dt = dt.to(device, non_blocking=True)
        complete = complete.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            prediction = model(features)
            loss = criterion(prediction, target, group_ids, dt, complete)
        if is_train:
            if grad_scaler is not None and amp_enabled:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.detach() * len(target)
        total_count += len(target)
        # 减少每个批次的CPU-GPU同步和终端刷新，最后一批仍然强制刷新。
        if progress and (batch_index % 10 == 0 or batch_index == len(loader)):
            progress.update(batch_index, f"loss={loss.detach().item():.5f}，样本 {total_count}")
    average_loss = float(total_loss.item()) / max(total_count, 1)
    if progress:
        progress.finish(f"平均损失={average_loss:.5f}")
    return average_loss


def run_training_loop(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, train_groups: tuple[np.ndarray, np.ndarray, np.ndarray], val_groups: tuple[np.ndarray, np.ndarray, np.ndarray], cfg: ExperimentConfig, params: dict, device: torch.device, epochs: int, early_stopping: bool = True) -> tuple[nn.Module, list[dict], float]:
    """功能: 按给定TCN和窗口超参数训练模型。
    参数: train_x/train_y/val_x/val_y为序列数据，cfg为配置，params为超参数，device为设备，epochs为轮数，early_stopping表示是否启用早停。
    返回: 最佳模型、训练日志和最佳验证损失。
    调用位置: train_model。
    """

    model = build_model(train_x.shape[2], tuple(params["channels"]), float(params["dropout"]), int(params["kernel_size"]), params.get("dt_feature_index")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = BalancedHuberLoss(float(params.get("target_mean", 0.0)), float(params.get("target_std", 1.0)), float(params["huber_delta"]), str(params.get("target_transform", "none")), float(params.get("power_scale", 100.0)), float(params.get("energy_loss_weight", 0.2)))
    train_loader = make_loader(train_x, train_y, *train_groups, cfg.batch_size, True, device)
    val_loader = make_loader(val_x, val_y, *val_groups, cfg.batch_size, False, device)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    logs: list[dict] = []
    progress = TerminalProgress(f"训练 {params['name']}", epochs)
    for epoch in range(1, epochs + 1):
        # 调参和最终训练均只显示epoch级进度，避免训练批次与候选进度重复刷新。
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, grad_scaler)
        val_loss = run_epoch(model, val_loader, criterion, None, device, grad_scaler)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr_now,
            "candidate": params["name"],
            "model_type": "tcn",
            "window_seconds": params["window_seconds"],
            "window_steps": params["window_steps"],
            "phase": "training",
        })
        progress.update(epoch, f"train={train_loss:.5f}, val={val_loss:.5f}, lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping and stale_epochs >= cfg.patience:
            progress.finish(f"早停，最佳验证损失={best_val:.5f}", completed=False)
            break
    else:
        progress.finish(f"最佳验证损失={best_val:.5f}")
    best_epoch = min(logs, key=lambda item: item["val_loss"])["epoch"]
    for log in logs:
        log["is_best_epoch"] = int(log["epoch"] == best_epoch)
    model.load_state_dict(best_state)
    return model, logs, best_val


def predict_original_power(model: nn.Module, sequences: np.ndarray, scaler: dict, device: torch.device, batch_size: int, progress_label: str | None = None) -> np.ndarray:
    """功能: 批量执行TCN前向并还原原始功率。
    参数: model为TCN，sequences为标准化短序列，scaler为标准化参数，device为设备，batch_size为批量大小。
    返回: TCN原始功率预测数组。
    progress_label为可选的推理批次进度标题。
    调用位置: train_model。
    """

    outputs: list[np.ndarray] = []
    model.eval()
    progress = TerminalProgress(progress_label, max((len(sequences) + batch_size - 1) // batch_size, 1)) if progress_label else None
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(sequences), batch_size), start=1):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            scaled = model(batch).cpu().numpy()
            target = scaled * float(scaler["y_std"]) + float(scaler["y_mean"])
            outputs.append(inverse_target_transform(target, scaler["target_transform"]))
            if progress:
                progress.update(batch_index, f"已处理 {min(start + batch_size, len(sequences))}/{len(sequences)} 条")
    if progress:
        progress.finish("TCN批量推理完成")
    return np.maximum(np.concatenate(outputs), 0.0)


def apply_rls_correction(base_power: np.ndarray, frame: pd.DataFrame, scaler: dict, cfg: ExperimentConfig, initial_theta: list[float] | None = None, update: bool = True, progress_label: str | None = None, rls_params: dict | None = None, trace_path=None) -> tuple[np.ndarray, list[float]]:
    """功能: 按flight时间顺序执行先预测后更新的RLS实时校正。
    参数: base_power为TCN功率，frame为对应数据，scaler为尺度参数，cfg为配置，initial_theta为初始状态，update表示是否使用实测值更新，progress_label为可选进度标题，rls_params为RLS超参数。
    返回: 校正功率数组和最终RLS参数。
    调用位置: train_model、predict.py。
    """

    rls_params = rls_params or {"forgetting_factor": cfg.rls_forgetting_factor, "initial_covariance": cfg.rls_initial_covariance}
    corrector = RLSCorrector(float(rls_params["forgetting_factor"]), float(rls_params["initial_covariance"]), scaler["power_scale"], initial_theta)
    corrected = np.empty(len(frame), dtype=np.float64)
    actual_values = frame[cfg.target_column].to_numpy(dtype=float) if update and cfg.target_column in frame.columns else None
    progress = TerminalProgress(progress_label, max(len(frame), 1)) if progress_label else None
    ordered = frame.reset_index(drop=True)
    window_ids = np.floor(ordered["time"].to_numpy(dtype=float)).astype(int)
    trace_rows: list[dict] = []
    processed_windows = 0
    for flight, flight_frame in ordered.groupby("flight", sort=False):
        if update:
            corrector.reset_neutral()
        else:
            corrector.reset()
        previous_state = None
        flight_positions = flight_frame.index.to_numpy(dtype=int)
        marked = flight_frame.assign(_window=window_ids[flight_positions])
        for window_number, (_, positions_frame) in enumerate(marked.groupby("_window", sort=True)):
            positions = positions_frame.index.to_numpy(dtype=int)
            state = int(float(np.mean(base_power[positions])) >= cfg.flight_state_threshold_w)
            state_changed = previous_state is not None and state != previous_state
            if state_changed:
                corrector.reset_neutral()
            theta_before = corrector.theta.tolist()
            for index in positions:
                corrected[index] = corrector.predict(float(base_power[index]))
            dt = ordered.iloc[positions]["dt_seconds"].to_numpy(dtype=float)
            predicted_energy = float(np.sum(corrected[positions] * dt) / 3600.0)
            base_energy = float(np.sum(base_power[positions] * dt) / 3600.0)
            if actual_values is not None:
                actual_energy = float(np.sum(actual_values[positions] * dt) / 3600.0)
                if float(np.sum(dt)) >= 0.95:
                    # 当前窗口结束后才更新，更新结果只会影响下一个窗口。
                    corrector.update_window(base_energy, actual_energy, float(np.sum(dt)))
            trace_rows.append({"flight": int(flight), "second_window": int(window_ids[positions[0]]), "flight_state": state, "state_changed": int(state_changed), "theta_bias_before": float(theta_before[0]), "theta_scale_before": float(theta_before[1]), "theta_bias_after": float(corrector.theta[0]), "theta_scale_after": float(corrector.theta[1]), "tcn_energy_wh": base_energy, "predicted_energy_wh": predicted_energy, "actual_energy_wh": actual_energy if actual_values is not None else np.nan})
            previous_state = state
            processed_windows += 1
            if progress and (processed_windows % 100 == 0 or int(positions[-1]) + 1 == len(frame)):
                progress.update(int(positions[-1]) + 1, f"已完成flight={flight}的窗口{int(window_ids[positions[0]])}")
    if progress:
        progress.finish("RLS在线校正完成")
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_frame = pd.DataFrame(trace_rows)
        trace_frame.to_csv(trace_path, index=False, encoding="utf-8")
        statistics = trace_frame.groupby("flight", sort=True).agg(
            window_count=("second_window", "count"),
            state_switch_count=("state_changed", "sum"),
            bias_mean=("theta_bias_after", "mean"),
            bias_std=("theta_bias_after", "std"),
            bias_min=("theta_bias_after", "min"),
            bias_max=("theta_bias_after", "max"),
            scale_mean=("theta_scale_after", "mean"),
            scale_std=("theta_scale_after", "std"),
            scale_min=("theta_scale_after", "min"),
            scale_max=("theta_scale_after", "max"),
        ).reset_index()
        statistics.to_csv(cfg.rls_parameter_statistics_csv, index=False, encoding="utf-8")
        save_json(cfg.rls_parameter_summary_json, {
            "version": "3.0",
            "window_count": int(len(trace_frame)),
            "flight_count": int(trace_frame["flight"].nunique()),
            "state_switch_count": int(trace_frame["state_changed"].sum()),
            "bias_mean": float(trace_frame["theta_bias_after"].mean()),
            "bias_std": float(trace_frame["theta_bias_after"].std(ddof=0)),
            "bias_min": float(trace_frame["theta_bias_after"].min()),
            "bias_max": float(trace_frame["theta_bias_after"].max()),
            "scale_mean": float(trace_frame["theta_scale_after"].mean()),
            "scale_std": float(trace_frame["theta_scale_after"].std(ddof=0)),
            "scale_min": float(trace_frame["theta_scale_after"].min()),
            "scale_max": float(trace_frame["theta_scale_after"].max()),
        })
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


def energy_window_selection_metrics(base_power: np.ndarray, corrected_power: np.ndarray, frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 按不规则采样间隔计算所有flight的秒级能量价值函数。
    参数: base_power和corrected_power为TCN及RLS逐点预测，frame为验证数据表，cfg为配置。
    返回: 按flight等权汇总的逐点、秒级和整flight误差指标。
    调用位置: train_model的RLS超参数搜索。
    """

    work = frame[["flight", "time", "dt_seconds", cfg.target_column]].copy()
    work["window"] = np.floor(work["time"].to_numpy(dtype=float)).astype(int)
    work["base_energy"] = base_power * work["dt_seconds"] / 3600.0
    work["corrected_energy"] = corrected_power * work["dt_seconds"] / 3600.0
    work["actual_energy"] = work[cfg.target_column] * work["dt_seconds"] / 3600.0
    windows = work.groupby(["flight", "window"], sort=True)[["base_energy", "corrected_energy", "actual_energy"]].sum()
    per_flight = windows.groupby(level=0).sum()
    second_errors = windows.groupby(level=0).apply(lambda item: np.sum(np.abs(item["corrected_energy"] - item["actual_energy"])) / max(np.sum(np.abs(item["actual_energy"])), 1e-6) * 100.0)
    flight_errors = np.abs(per_flight["corrected_energy"] - per_flight["actual_energy"]) / per_flight["actual_energy"].abs().clip(lower=1e-6) * 100.0
    power_errors = work.assign(error=np.abs(corrected_power - work[cfg.target_column])).groupby("flight")["error"].sum() / work.groupby("flight")[cfg.target_column].apply(lambda x: np.abs(x).sum()).clip(lower=1e-6) * 100.0
    score = second_errors + 0.2 * power_errors + 0.5 * flight_errors
    tail_penalty = 0.25 * float(np.quantile(flight_errors, 0.95)) + 0.1 * float(score.std(ddof=0))
    return {
        "val_second_energy_wape_mean": float(second_errors.mean()),
        "val_flight_energy_wape_mean": float(flight_errors.mean()),
        "val_sample_power_wape_mean": float(power_errors.mean()),
        "selection_score": float(score.mean() + tail_penalty),
        "selection_score_std": float(score.std(ddof=0)),
        "flight_energy_error_p95": float(np.quantile(flight_errors, 0.95)),
        "tail_penalty": tail_penalty,
        "flight_count": int(score.size),
    }


def tcn_selection_metrics(base_power: np.ndarray, frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 仅根据TCN原始验证预测计算窗口选择指标。
    参数: base_power为TCN验证预测，frame为验证数据表，cfg为实验配置。
    返回: TCN样本功率和flight能耗WAPE及窗口选择分数。
    调用位置: train_model。
    """

    actual = frame[cfg.target_column].to_numpy(dtype=float)
    sample_wape = float(np.sum(np.abs(base_power - actual)) / max(np.sum(np.abs(actual)), 1e-6) * 100.0)
    metrics_frame = frame[["flight", "time", "dt_seconds", cfg.target_column]].copy()
    metrics_frame["tcn_energy"] = base_power * metrics_frame["dt_seconds"] / 3600.0
    metrics_frame["actual_energy"] = metrics_frame[cfg.target_column] * metrics_frame["dt_seconds"] / 3600.0
    grouped = metrics_frame.groupby("flight", sort=True)[["tcn_energy", "actual_energy"]].sum()
    flight_wape = float(np.sum(np.abs(grouped["tcn_energy"] - grouped["actual_energy"])) / max(np.sum(np.abs(grouped["actual_energy"])), 1e-6) * 100.0)
    metrics_frame["second_window"] = np.floor(metrics_frame["time"].to_numpy(dtype=float)).astype(int) if "time" in metrics_frame else 0
    second = metrics_frame.groupby(["flight", "second_window"])[["tcn_energy", "actual_energy"]].sum()
    second_wape = float(np.sum(np.abs(second["tcn_energy"] - second["actual_energy"])) / max(np.sum(np.abs(second["actual_energy"])), 1e-6) * 100.0)
    return {"val_tcn_sample_power_wape": sample_wape, "val_tcn_second_energy_wape": second_wape, "val_tcn_flight_energy_wape": flight_wape, "tcn_selection_score": second_wape + 0.2 * sample_wape + 0.5 * flight_wape}


def rls_candidate_grid(cfg: ExperimentConfig) -> list[dict]:
    """功能: 生成RLS验证候选参数组合。
    参数: cfg为实验配置。
    返回: RLS遗忘因子和初始协方差的组合列表，所有候选均在窗口结束后立即更新。
    调用位置: train_model。
    """

    return [
        {"candidate": f"rls_ff{forgetting:g}_cov{cov:g}", "forgetting_factor": forgetting, "initial_covariance": cov, "warmup_windows": 0}
        for forgetting in cfg.rls_forgetting_factors
        for cov in cfg.rls_initial_covariances
    ]


def candidate_grid(cfg: ExperimentConfig, sample_interval: float) -> list[dict]:
    """功能: 为每个短时间窗生成一个可比较的TCN超参数候选。
    参数: cfg为实验配置，sample_interval为典型采样周期。
    返回: 包含秒级窗口和采样步数的候选列表。
    调用位置: train_model。
    """

    return [{"name": f"tcn_window_{seconds:g}s", "model_type": "tcn", "window_seconds": float(seconds), "window_steps": window_seconds_to_steps(seconds, sample_interval), "channels": list(cfg.tcn_channels), "kernel_size": 3, "dropout": 0.08, "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay, "huber_delta": 0.65, "dt_feature_index": 1} for seconds in cfg.window_seconds_candidates]


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


def tune_rls_only(cfg: ExperimentConfig) -> dict:
    """功能: 加载已有TCN权重，仅重新搜索RLS超参数并写回最优参数。
    参数: cfg为实验配置对象。
    返回: RLS搜索摘要字典。
    调用位置: main.py的tune-rls模式。
    """

    ensure_directories(cfg)
    if not cfg.best_model_file.exists():
        raise FileNotFoundError(f"未找到已训练TCN模型: {cfg.best_model_file}")
    device = select_cuda_device(cfg.device)
    checkpoint = torch.load(cfg.best_model_file, map_location=device, weights_only=False)
    scaler = load_json(cfg.scaler_json)
    train_df, val_df, feature_columns = load_split_data(cfg)
    train_df = train_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    window_steps = int(checkpoint["window_steps"])
    model = build_model(int(checkpoint["input_dim"]), tuple(checkpoint["channels"]), float(checkpoint["dropout"]), int(checkpoint.get("kernel_size", 3)), checkpoint.get("dt_feature_index")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_x, _ = build_sequence_arrays(train_df, scaler, window_steps, "RLS初值训练序列")
    val_x, _ = build_sequence_arrays(val_df, scaler, window_steps, "RLS搜索验证序列")
    train_base = predict_original_power(model, train_x, scaler, device, cfg.batch_size, "RLS初值TCN推理")
    val_base = predict_original_power(model, val_x, scaler, device, cfg.batch_size, "RLS搜索TCN推理")
    initial_theta = estimate_rls_initial_theta(train_base, train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"])
    rows: list[dict] = []
    best_params = None
    best_score = float("inf")
    candidates = rls_candidate_grid(cfg)
    print(f"仅搜索RLS：固定TCN={checkpoint.get('best_tcn_candidate', checkpoint.get('window_seconds'))}，候选数={len(candidates)}")
    for index, params in enumerate(candidates, start=1):
        corrected, _ = apply_rls_correction(val_base, val_df, scaler, cfg, initial_theta, update=True, rls_params=params)
        metrics = energy_window_selection_metrics(val_base, corrected, val_df, cfg)
        rows.append({"phase": "rls_only", "tcn_window_seconds": checkpoint["window_seconds"], "tcn_window_steps": window_steps, **params, **metrics})
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_params = params.copy()
        if index % max(1, len(candidates) // 10) == 0 or index == len(candidates):
            print(f"RLS候选 {index}/{len(candidates)}，当前最优分数={best_score:.6f}")
    if best_params is None:
        raise RuntimeError("RLS参数搜索未得到可用结果。")
    pd.DataFrame(rows).to_csv(cfg.rls_tuning_results_csv, index=False, encoding="utf-8")
    checkpoint.update({
        "rls_theta": initial_theta,
        "rls_forgetting_factor": best_params["forgetting_factor"],
        "rls_initial_covariance": best_params["initial_covariance"],
        "rls_warmup_windows": 0,
        "best_rls_candidate": best_params["candidate"],
        "rls_selection_score": best_score,
        "rls_candidate_count": len(rows),
    })
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    print(f"RLS最优结果：{best_params['candidate']}，选择分数={best_score:.6f}")
    return {"version": "3.0", "mode": "tune-rls", "fixed_tcn_candidate": checkpoint.get("best_tcn_candidate"), "best_rls_candidate": best_params["candidate"], "rls_forgetting_factor": best_params["forgetting_factor"], "rls_initial_covariance": best_params["initial_covariance"], "rls_warmup_windows": 0, "best_rls_selection_score": best_score, "rls_candidate_count": len(rows)}


def train_fixed_tcn(cfg: ExperimentConfig) -> dict:
    """功能: 使用当前调参表中的最优TCN窗口完成一次正式训练，不重复搜索TCN或RLS。
    参数: cfg为实验配置对象。
    返回: 固定超参数训练摘要字典。
    调用位置: main.py的train-fixed模式。
    """

    ensure_directories(cfg)
    set_random_seed(cfg.random_seed)
    device = select_cuda_device(cfg.device)
    train_df, val_df, feature_columns = load_split_data(cfg)
    train_df = train_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = build_scaler(train_df, feature_columns, cfg)
    sample_interval = estimate_sample_interval(train_df)
    scaler["sample_interval_seconds"] = sample_interval
    save_json(cfg.scaler_json, scaler)
    if cfg.tuning_results_csv.exists():
        tuning = pd.read_csv(cfg.tuning_results_csv).sort_values("tcn_selection_score")
        selected = tuning.iloc[0]
        window_seconds = float(selected["window_seconds"])
    elif cfg.best_model_file.exists():
        previous = torch.load(cfg.best_model_file, map_location="cpu", weights_only=False)
        window_seconds = float(previous.get("window_seconds", cfg.default_window_seconds))
    else:
        window_seconds = float(cfg.default_window_seconds)
    params = candidate_grid(cfg, sample_interval)[0]
    params.update({
        "name": f"tcn_window_{window_seconds:g}s",
        "window_seconds": window_seconds,
        "window_steps": window_seconds_to_steps(window_seconds, sample_interval),
        "target_mean": scaler["y_mean"],
        "target_std": scaler["y_std"],
        "target_transform": scaler["target_transform"],
        "power_scale": scaler["power_scale"],
        "dt_feature_index": feature_columns.index("dt_seconds"),
        "energy_loss_weight": 0.2,
    })
    print(f"固定TCN超参数：窗口={window_seconds:g}s/{params['window_steps']}步，通道={params['channels']}，卷积核={params['kernel_size']}，dropout={params['dropout']}，学习率={params['learning_rate']:.2e}，权重衰减={params['weight_decay']:.2e}，epoch={cfg.epochs}")
    train_x, train_y = build_sequence_arrays(train_df, scaler, params["window_steps"], "固定参数训练序列")
    val_x, val_y = build_sequence_arrays(val_df, scaler, params["window_steps"], "固定参数验证序列")
    model, logs, best_val = run_training_loop(train_x, train_y, val_x, val_y, build_energy_group_arrays(train_df), build_energy_group_arrays(val_df), cfg, params, device, cfg.epochs, early_stopping=True)
    for log in logs:
        log["phase"] = "final_tcn"
    train_base = predict_original_power(model, train_x, scaler, device, cfg.batch_size, "固定参数训练集推理")
    initial_theta = estimate_rls_initial_theta(train_base, train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"])
    previous = torch.load(cfg.best_model_file, map_location="cpu", weights_only=False) if cfg.best_model_file.exists() else {}
    checkpoint = {
        **previous,
        "model_state_dict": model.state_dict(), "input_dim": len(feature_columns), "feature_columns": feature_columns,
        "dt_feature_index": params["dt_feature_index"], "target_column": cfg.target_column, "target_transform": cfg.target_transform,
        "model_type": "time_aware_tcn", "channels": params["channels"], "kernel_size": params["kernel_size"],
        "dropout": params["dropout"], "learning_rate": params["learning_rate"], "weight_decay": params["weight_decay"],
        "huber_delta": params["huber_delta"], "window_seconds": window_seconds, "window_steps": params["window_steps"],
        "sample_interval_seconds": sample_interval, "rls_theta": initial_theta, "power_scale": scaler["power_scale"],
        "scaler_path": str(cfg.scaler_json), "best_val_loss": best_val,
        "best_epoch": min(logs, key=lambda item: item["val_loss"])["epoch"], "best_tcn_candidate": params["name"],
        "best_tcn_window_seconds": window_seconds, "best_tcn_window_steps": params["window_steps"],
    }
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    pd.DataFrame(logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    save_loss_curve(logs, cfg.loss_curve_file)
    return {"version": "3.0", "mode": "train-fixed", "best_tcn_candidate": params["name"], "window_seconds": window_seconds, "window_steps": params["window_steps"], "epochs_run": len(logs), "best_epoch": checkpoint["best_epoch"], "best_val_loss": best_val, "model_file": str(cfg.best_model_file)}


def train_model(cfg: ExperimentConfig, stop_after_tcn: bool = False) -> dict:
    """功能: 按两阶段规则选择TCN窗口和秒级能量监督RLS参数并保存实验3.0产物。
    参数: cfg为实验配置对象。
    返回: 训练摘要字典。
    调用位置: main.py。
    """

    ensure_directories(cfg)
    set_random_seed(cfg.random_seed)
    device = select_cuda_device(cfg.device)
    print(f"训练设备: {describe_cuda_device(device)}")
    workflow_progress = TerminalProgress("训练总流程", 5)
    train_df, val_df, feature_columns = load_split_data(cfg)
    train_df = train_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    scaler = build_scaler(train_df, feature_columns, cfg)
    sample_interval = estimate_sample_interval(train_df)
    scaler["sample_interval_seconds"] = sample_interval
    save_json(cfg.scaler_json, scaler)
    train_groups = build_energy_group_arrays(train_df)
    val_groups = build_energy_group_arrays(val_df)
    workflow_progress.update(1, f"训练集 {len(train_df)} 条，验证集 {len(val_df)} 条，特征 {len(feature_columns)} 维")

    window_rows: list[dict] = []
    all_training_logs: list[dict] = []
    candidates = candidate_grid(cfg, sample_interval)
    dt_feature_index = feature_columns.index("dt_seconds")
    for candidate in candidates:
        candidate["target_mean"] = scaler["y_mean"]
        candidate["target_std"] = scaler["y_std"]
        candidate["dt_feature_index"] = dt_feature_index
        candidate["dt_mean"] = scaler["x_mean"][dt_feature_index]
        candidate["dt_std"] = scaler["x_std"][dt_feature_index]
        candidate["energy_loss_weight"] = 0.2
        candidate["target_transform"] = scaler["target_transform"]
        candidate["power_scale"] = scaler["power_scale"]
    print(f"阶段一：TCN窗口选择，共 {len(candidates)} 个候选，每个候选固定训练 {cfg.tune_epochs} 轮。")
    candidate_progress = TerminalProgress("TCN时间窗调参", len(candidates))
    for index, params in enumerate(candidates, start=1):
        candidate_started = time.perf_counter()
        print(f"\nTCN时间窗候选 {index}/{len(candidates)} 开始：{params['window_seconds']:g}s/{params['window_steps']}步")
        train_x, train_y = build_sequence_arrays(train_df, scaler, params["window_steps"], f"{params['window_seconds']:g}s 训练序列")
        val_x, val_y = build_sequence_arrays(val_df, scaler, params["window_steps"], f"{params['window_seconds']:g}s 验证序列")
        model, logs, val_loss = run_training_loop(train_x, train_y, val_x, val_y, train_groups, val_groups, cfg, params, device, cfg.tune_epochs, early_stopping=False)
        for log in logs:
            log["phase"] = "stage1_tcn_window"
        all_training_logs.extend(logs)
        val_base = predict_original_power(model, val_x, scaler, device, cfg.batch_size, f"{params['window_seconds']:g}s 验证集推理")
        metrics = tcn_selection_metrics(val_base, val_df, cfg)
        row = {"phase": "stage1_tcn_window", "candidate": params["name"], **params, "channels": str(params["channels"]), "best_val_loss": val_loss, **metrics, "epochs_run": len(logs)}
        window_rows.append(row)
        elapsed = time.perf_counter() - candidate_started
        print(
            f"TCN时间窗 {params['window_seconds']:g}s 测试结果："
            f"验证损失={val_loss:.6f}，选择分数={metrics['tcn_selection_score']:.6f}，耗时={elapsed:.2f}s"
        )
        log_result(
            "tune-tcn候选结果",
            {
                "candidate": params["name"],
                "window_seconds": params["window_seconds"],
                "window_steps": params["window_steps"],
                "best_val_loss": val_loss,
                "tcn_selection_score": metrics["tcn_selection_score"],
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        candidate_progress.update(index, f"{params['window_seconds']:g}s，TCN分数={metrics['tcn_selection_score']:.4f}")
        if index == 1 or metrics["tcn_selection_score"] < min(item["tcn_selection_score"] for item in window_rows[:-1]):
            best_params = params.copy()
            best_window_row = row
            best_window_val_base = val_base.copy()
            best_window_train_x = train_x.copy()
            best_window_train_y = train_y.copy()
            best_window_state = deepcopy(model.state_dict())
        del train_x, train_y, val_x, val_y, model
        torch.cuda.empty_cache()
    candidate_progress.finish(f"TCN最优窗口={best_params['window_seconds'] if best_params else '无'}s")
    if best_params is not None and best_window_row is not None:
        print(f"TCN最优结果：候选={best_params['name']}，时间窗={best_params['window_seconds']:g}s/{best_params['window_steps']}步，选择分数={best_window_row['tcn_selection_score']:.6f}")
    workflow_progress.update(2, f"已完成 {len(candidates)} 个时间窗候选比较")
    if best_params is None:
        raise RuntimeError("时间窗调参未得到可用候选。")

    if best_params is None:
        raise RuntimeError("TCN窗口调参未得到可用候选。")

    pd.DataFrame(window_rows).to_csv(cfg.tuning_results_csv, index=False, encoding="utf-8")
    if stop_after_tcn:
        stage1_checkpoint = {"model_state_dict": best_window_state, "input_dim": len(feature_columns), "feature_columns": feature_columns, "dt_feature_index": best_params.get("dt_feature_index"), "target_column": cfg.target_column, "target_transform": cfg.target_transform, "model_type": "time_aware_tcn", "channels": best_params["channels"], "kernel_size": best_params["kernel_size"], "dropout": best_params["dropout"], "learning_rate": best_params["learning_rate"], "weight_decay": best_params["weight_decay"], "huber_delta": best_params["huber_delta"], "window_seconds": best_params["window_seconds"], "window_steps": best_params["window_steps"], "sample_interval_seconds": sample_interval, "power_scale": scaler["power_scale"], "scaler_path": str(cfg.scaler_json), "best_tcn_candidate": best_params["name"], "best_tcn_window_seconds": best_params["window_seconds"], "best_tcn_window_steps": best_params["window_steps"], "tcn_selection_score": best_window_row["tcn_selection_score"], "tcn_candidate_count": len(window_rows)}
        torch.save(stage1_checkpoint, cfg.best_model_file)
        pd.DataFrame(all_training_logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
        workflow_progress.finish("TCN候选搜索完成，已保存最优TCN；未执行RLS搜索和最终训练")
        return {"version": "3.0", "mode": "tune-tcn", "best_tcn_candidate": best_params["name"], "best_tcn_window_seconds": best_params["window_seconds"], "best_tcn_window_steps": best_params["window_steps"], "best_tcn_selection_score": best_window_row["tcn_selection_score"], "tcn_candidate_count": len(window_rows), "model_file": str(cfg.best_model_file)}

    train_x, train_y = best_window_train_x, best_window_train_y
    stage1_model = build_model(train_x.shape[2], tuple(best_params["channels"]), best_params["dropout"], best_params["kernel_size"], best_params.get("dt_feature_index")).to(device)
    stage1_model.load_state_dict(best_window_state)
    rls_theta = estimate_rls_initial_theta(
        predict_original_power(stage1_model, train_x, scaler, device, cfg.batch_size, "阶段二RLS初值训练集推理"),
        train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"],
    )
    del stage1_model
    torch.cuda.empty_cache()
    rls_rows: list[dict] = []
    best_rls = None
    best_rls_score = float("inf")
    rls_candidates = rls_candidate_grid(cfg)
    rls_candidate_count = len(rls_candidates)
    print(f"阶段二：固定最优TCN窗口，使用同一份验证预测搜索{rls_candidate_count}组RLS参数。")
    for rls_index, rls_params in enumerate(rls_candidates, start=1):
        corrected, _ = apply_rls_correction(best_window_val_base, val_df, scaler, cfg, rls_theta, update=True, rls_params=rls_params)
        metrics = energy_window_selection_metrics(best_window_val_base, corrected, val_df, cfg)
        row = {"phase": "stage2_rls", "tcn_window_seconds": best_params["window_seconds"], "tcn_window_steps": best_params["window_steps"], **rls_params, **metrics}
        rls_rows.append(row)
        if metrics["selection_score"] < best_rls_score:
            best_rls_score, best_rls = metrics["selection_score"], rls_params.copy()
        if rls_index % max(1, rls_candidate_count // 10) == 0 or rls_index == rls_candidate_count:
            print(f"RLS候选 {rls_index}/{rls_candidate_count}，当前最优分数={best_rls_score:.4f}")
    if best_rls is None:
        raise RuntimeError("RLS参数搜索未得到可用候选。")
    print(f"RLS最优结果：候选={best_rls['candidate']}，遗忘因子={best_rls['forgetting_factor']:g}，初始协方差={best_rls['initial_covariance']:g}，warmup固定为0个完整窗口，选择分数={best_rls_score:.6f}")
    workflow_progress.update(3, f"窗口={best_params['window_seconds']:g}s，RLS共完成{rls_candidate_count}组")

    train_x, train_y = build_sequence_arrays(train_df, scaler, best_params["window_steps"], "最终训练序列")
    val_x, val_y = build_sequence_arrays(val_df, scaler, best_params["window_steps"], "最终验证序列")
    print(f"开始最终训练：窗口 {best_params['window_seconds']:g}s（{best_params['window_steps']}步），最多 {cfg.epochs} 轮。")
    final_model, logs, best_val_loss = run_training_loop(train_x, train_y, val_x, val_y, train_groups, val_groups, cfg, best_params, device, cfg.epochs, early_stopping=True)
    for log in logs:
        log["phase"] = "final_tcn"
    all_training_logs.extend(logs)
    train_base = predict_original_power(final_model, train_x, scaler, device, cfg.batch_size, "最终训练集推理")
    rls_theta = estimate_rls_initial_theta(train_base, train_df[cfg.target_column].to_numpy(dtype=float), scaler["power_scale"])
    workflow_progress.update(4, f"最终模型训练完成，最佳验证损失={best_val_loss:.5f}")

    # 用独立验证集残差校准两种RLS运行状态，后续修改置信度时无需重新执行TCN。
    val_base = predict_original_power(final_model, val_x, scaler, device, cfg.batch_size, "置信区间校准推理")
    val_online, _ = apply_rls_correction(val_base, val_df, scaler, cfg, None, update=True, rls_params=best_rls, progress_label="在线RLS校准残差")
    val_static, _ = apply_rls_correction(val_base, val_df, scaler, cfg, rls_theta, update=False, rls_params=best_rls, progress_label="固定RLS校准残差")
    actual_val = val_df[cfg.target_column].to_numpy(dtype=float)
    calibration = save_calibration(
        cfg.uncertainty_calibration_npz,
        cfg.uncertainty_calibration_json,
        np.abs(val_online - actual_val),
        np.abs(val_static - actual_val),
        cfg.default_confidence,
    )
    workflow_progress.update(5, f"置信区间校准完成，默认置信度={cfg.default_confidence:.0%}，在线半径={calibration['online_default_radius_w']:.2f}W")

    pd.DataFrame(window_rows).to_csv(cfg.tuning_results_csv, index=False, encoding="utf-8")
    pd.DataFrame(rls_rows).to_csv(cfg.rls_tuning_results_csv, index=False, encoding="utf-8")
    pd.DataFrame(all_training_logs).to_csv(cfg.training_log_csv, index=False, encoding="utf-8")
    save_loss_curve(logs, cfg.loss_curve_file)
    checkpoint = {"model_state_dict": final_model.state_dict(), "input_dim": len(feature_columns), "feature_columns": feature_columns, "dt_feature_index": best_params.get("dt_feature_index"), "target_column": cfg.target_column, "target_transform": cfg.target_transform, "model_type": "time_aware_tcn", "channels": best_params["channels"], "kernel_size": best_params["kernel_size"], "dropout": best_params["dropout"], "learning_rate": best_params["learning_rate"], "weight_decay": best_params["weight_decay"], "huber_delta": best_params["huber_delta"], "window_seconds": best_params["window_seconds"], "window_steps": best_params["window_steps"], "sample_interval_seconds": sample_interval, "rls_theta": rls_theta, "rls_forgetting_factor": best_rls["forgetting_factor"], "rls_initial_covariance": best_rls["initial_covariance"], "rls_warmup_windows": 0, "power_scale": scaler["power_scale"], "scaler_path": str(cfg.scaler_json), "uncertainty_calibration_npz": str(cfg.uncertainty_calibration_npz), "uncertainty_calibration_json": str(cfg.uncertainty_calibration_json), "device_used": str(device), "best_val_loss": best_val_loss, "best_epoch": min(logs, key=lambda item: item["val_loss"])["epoch"], "best_tcn_candidate": best_params["name"], "best_tcn_window_seconds": best_params["window_seconds"], "best_tcn_window_steps": best_params["window_steps"], "tcn_selection_score": best_window_row["tcn_selection_score"], "best_rls_candidate": best_rls["candidate"], "rls_selection_score": best_rls_score, "tcn_candidate_count": len(window_rows), "rls_candidate_count": len(rls_rows)}
    torch.save(checkpoint, cfg.best_model_file)
    torch.save(checkpoint, cfg.final_model_file)
    workflow_progress.finish(f"权重、日志和置信区间校准已保存；深度={len(best_params['channels'])}块")
    return {"version": "3.0", "device": str(device), "cuda_device_name": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else "CPU", "best_candidate": best_params["name"], "best_model_type": "tcn_second_energy_rls", "best_window_seconds": best_params["window_seconds"], "best_window_steps": best_params["window_steps"], "best_tcn_candidate": best_params["name"], "best_tcn_window_seconds": best_params["window_seconds"], "best_tcn_window_steps": best_params["window_steps"], "sample_interval_seconds": sample_interval, "best_channels": best_params["channels"], "best_dropout": best_params["dropout"], "best_learning_rate": best_params["learning_rate"], "best_weight_decay": best_params["weight_decay"], "rls_forgetting_factor": best_rls["forgetting_factor"], "rls_initial_covariance": best_rls["initial_covariance"], "rls_warmup_windows": 0, "best_rls_candidate": best_rls["candidate"], "rls_initial_theta": rls_theta, "target_transform": cfg.target_transform, "best_val_loss_standardized": best_val_loss, "best_epoch": min(logs, key=lambda item: item["val_loss"])["epoch"], "tcn_selection_score": best_window_row["tcn_selection_score"], "best_tcn_selection_score": best_window_row["tcn_selection_score"], "rls_selection_score": best_rls_score, "best_rls_selection_score": best_rls_score, "tcn_candidate_count": len(window_rows), "rls_candidate_count": len(rls_rows), "epochs_run": len(logs), "model_file": str(cfg.best_model_file), "scaler_file": str(cfg.scaler_json), "uncertainty_calibration_file": str(cfg.uncertainty_calibration_npz)}
