# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: vehicle_pipeline.py
# 开发时间: 2026-10-19
# 文件名: vehicle_pipeline.py
# 功能说明: 完成5.0车辆数据清洗、TCN超参搜索、RLS校正、评估和结果展示
# 版本号：5.0

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


# 只使用预测时刻已经发生或可测得的车辆状态；不使用由整段行程终点反推的进度。
FEATURES = config.FEATURE_GROUPS["full_planning"]
TARGET = "power_w"
ARCHITECTURE_VERSION = "causal_dilated_residual_v3"


def _log_event(event: str, **payload: object) -> None:
    """功能: 以JSONL记录一次可复现运行事件。
    参数: event为事件名称，payload为事件字段。
    返回: None。
    调用位置: 数据准备、训练、预测和评估流程。
    """
    config.ensure_directories()
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with config.RUN_LOG_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_dataset_statistics(all_data: pd.DataFrame) -> None:
    """功能: 输出数据集、划分、特征和行程统计表。
    参数: all_data为清洗后含split字段的全量数据。
    返回: None。
    调用位置: prepare_data。
    """
    split_rows = []
    for split, part in all_data.groupby("split", sort=True):
        split_rows.append({"split": split, "trip_count": int(part.trip_id.nunique()), "row_count": len(part), "duration_s_sum": float(part.groupby("trip_id").time_s.max().sum()), "power_mean_w": float(part.power_w.mean()), "power_std_w": float(part.power_w.std()), "power_min_w": float(part.power_w.min()), "power_max_w": float(part.power_w.max()), "energy_sum_wh": float(part.energy_wh.sum()), "distance_sum_km": float((part.speed_kmh.clip(lower=0) / 3.6).sum() / 1000)})
    pd.DataFrame(split_rows).to_csv(config.SPLIT_STATS_CSV, index=False, encoding="utf-8")
    pd.DataFrame([
        {"dataset": "GECR_real_world_data", "row_count_1hz": len(all_data), "trip_count": int(all_data.trip_id.nunique()), "source_file_count": int(all_data.trip_id.nunique()), "sampling_seconds": config.SAMPLE_SECONDS, "power_mean_w": float(all_data.power_w.mean()), "power_std_w": float(all_data.power_w.std()), "energy_sum_wh": float(all_data.energy_wh.sum()), "distance_sum_km": float((all_data.speed_kmh.clip(lower=0) / 3.6).sum() / 1000), "negative_power_ratio": float((all_data.power_w < 0).mean())}
    ]).to_csv(config.DATASET_STATS_CSV, index=False, encoding="utf-8")
    feature_rows = []
    for column in FEATURES + [TARGET]:
        series = all_data[column].astype(float)
        feature_rows.append({"feature": column, "role": "model_input" if column in FEATURES else "target", "count": int(series.count()), "missing_count": int(series.isna().sum()), "mean": float(series.mean()), "std": float(series.std()), "min": float(series.min()), "p01": float(series.quantile(.01)), "median": float(series.median()), "p99": float(series.quantile(.99)), "max": float(series.max())})
    pd.DataFrame(feature_rows).to_csv(config.FEATURE_STATS_CSV, index=False, encoding="utf-8")
    trip_rows = []
    for trip, part in all_data.groupby("trip_id", sort=True):
        trip_rows.append({"trip_id": trip, "split": part.split.iloc[0], "row_count": len(part), "duration_s": float(part.time_s.max()), "distance_km": float((part.speed_kmh.clip(lower=0) / 3.6).sum() / 1000), "energy_wh": float(part.energy_wh.sum()), "power_mean_w": float(part.power_w.mean()), "power_std_w": float(part.power_w.std()), "latitude_start": float(part.latitude.iloc[0]), "longitude_start": float(part.longitude.iloc[0]), "latitude_end": float(part.latitude.iloc[-1]), "longitude_end": float(part.longitude.iloc[-1])})
    pd.DataFrame(trip_rows).to_csv(config.TRIP_STATS_CSV, index=False, encoding="utf-8")


def _set_seed(seed: int) -> None:
    """功能: 固定Python、NumPy和PyTorch的随机状态。
    参数: seed为随机种子。
    返回: None。
    调用位置: train_model、search_tcn。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_trip(path: Path) -> pd.DataFrame:
    """功能: 读取一个GECR道路行程并聚合为1秒样本。
    参数: path为Excel文件路径。
    返回: 已排序、去重、聚合并派生特征的行程表。
    调用位置: prepare_data。
    """
    raw = pd.read_excel(path, sheet_name="Continuous20Hz", engine="openpyxl")
    required = {"PhaseTime", "DAActualSpeed", "REESSCurrent", "REESSVoltage", "tempAmb", "elevation", "lat", "long", "gear"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name}缺少字段: {missing}")
    cols = sorted(required)
    df = raw[cols].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["PhaseTime", "DAActualSpeed", "REESSVoltage", "REESSCurrent"])
    df = df.sort_values("PhaseTime", kind="stable").drop_duplicates("PhaseTime").reset_index(drop=True)
    if len(df) < 30 or df["PhaseTime"].iloc[-1] <= df["PhaseTime"].iloc[0]:
        raise ValueError(f"{path.name}有效记录不足")
    df["bin"] = np.floor((df["PhaseTime"] - df["PhaseTime"].iloc[0]) / config.SAMPLE_SECONDS).astype(int)
    # 作者MATLAB代码使用pow=-sum(vol.*cur,2)；当前道路文件的第二通道恒为零。
    df["power_w"] = -df["REESSVoltage"] * df["REESSCurrent"]
    agg = df.groupby("bin", as_index=False).agg(
        time_s=("PhaseTime", "mean"), speed_kmh=("DAActualSpeed", "mean"),
        temperature_c=("tempAmb", "mean"), elevation_m=("elevation", "mean"), latitude=("lat", "mean"), longitude=("long", "mean"),
        gear=("gear", "median"), voltage_v=("REESSVoltage", "mean"), current_a=("REESSCurrent", "mean"),
        power_w=("power_w", "mean"), source_rows=("power_w", "size"),
    )
    agg["time_s"] -= agg["time_s"].iloc[0]
    agg["trip_id"] = path.stem
    agg["task_duration_s"] = float(agg["time_s"].iloc[-1])
    agg["planned_speed_kmh"] = agg["speed_kmh"]
    agg["planned_acceleration_kmhps"] = agg["planned_speed_kmh"].diff().fillna(0.0) / config.SAMPLE_SECONDS
    agg["planned_distance_m"] = (agg["planned_speed_kmh"].clip(lower=0) / 3.6 * config.SAMPLE_SECONDS).cumsum()
    agg["planned_elevation_m"] = agg["elevation_m"] - agg["elevation_m"].iloc[0]
    elevation_delta = agg["planned_elevation_m"].diff().fillna(0.0)
    distance_delta = agg["planned_speed_kmh"].clip(lower=0) / 3.6 * config.SAMPLE_SECONDS
    agg["planned_grade_percent"] = (100 * elevation_delta / distance_delta.where(distance_delta > 0.5)).fillna(0.0).clip(-30, 30)
    north = np.gradient(agg["latitude"].to_numpy(float)) * 111_320
    east = np.gradient(agg["longitude"].to_numpy(float)) * 111_320 * np.cos(np.deg2rad(agg["latitude"].to_numpy(float)))
    agg["planned_heading_deg"] = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    agg["ambient_temperature_c"] = agg["temperature_c"]
    agg["planned_stop_flag"] = (agg["planned_speed_kmh"] < 1.0).astype(float)
    agg["energy_wh"] = agg["power_w"] * config.SAMPLE_SECONDS / 3600.0
    return agg.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def prepare_data(force: bool = False) -> dict:
    """功能: 清洗GECR道路行程并按完整行程生成训练、验证和测试表。
    参数: force为是否覆盖已有清洗结果。
    返回: 数据规模和切分摘要。
    调用位置: main的prepare和all模式。
    """
    config.ensure_directories()
    if config.DATA_SUMMARY_JSON.exists() and not force:
        cached = json.loads(config.DATA_SUMMARY_JSON.read_text(encoding="utf-8"))
        if cached.get("data_contract_version") == config.DATA_CONTRACT_VERSION and cached.get("feature_columns") == FEATURES and config.SPLIT_STATS_CSV.exists() and config.FEATURE_STATS_CSV.exists():
            _log_event("prepare_cache_hit", summary=str(config.DATA_SUMMARY_JSON), trip_count=cached.get("trip_count"), row_count_1hz=cached.get("row_count_1hz"))
            return cached
    source = config.SOURCE_DIR / "real_world_data"
    paths = sorted(source.glob("*.xlsx"))
    frames, errors = [], []
    for path in paths:
        try:
            frames.append(_read_trip(path))
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
    if len(frames) < 20:
        raise RuntimeError(f"有效行程数过少: {len(frames)}")
    all_data = pd.concat(frames, ignore_index=True)
    trip_ids = sorted(all_data.trip_id.unique())
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(trip_ids)
    n_test = max(1, round(len(trip_ids) * 0.20)); n_cal = max(20, round(len(trip_ids) * 0.20)); n_val = max(1, round(len(trip_ids) * 0.10))
    test_ids = set(trip_ids[:n_test]); cal_ids = set(trip_ids[n_test:n_test + n_cal]); val_ids = set(trip_ids[n_test + n_cal:n_test + n_cal + n_val]); train_ids = set(trip_ids[n_test + n_cal + n_val:])
    all_data["split"] = all_data.trip_id.map(lambda x: "test" if x in test_ids else "calibration" if x in cal_ids else "val" if x in val_ids else "train")
    _write_dataset_statistics(all_data)
    all_data.to_csv(config.CLEAN_CSV, index=False, encoding="utf-8")
    for split, ids in (("train", train_ids), ("val", val_ids), ("calibration", cal_ids), ("test", test_ids)):
        all_data[all_data.trip_id.isin(ids)].to_csv(getattr(config, f"{split.upper()}_CSV"), index=False, encoding="utf-8")
    summary = {
        "dataset": "MOTIVES-LAB GECR real_world_data", "source": str(source),
        "trip_count": len(trip_ids), "row_count_1hz": int(len(all_data)),
        "train_trips": sorted(train_ids), "val_trips": sorted(val_ids), "calibration_trips": sorted(cal_ids), "test_trips": sorted(test_ids),
        "train_rows": int((all_data.split == "train").sum()), "val_rows": int((all_data.split == "val").sum()), "calibration_rows": int((all_data.split == "calibration").sum()), "test_rows": int((all_data.split == "test").sum()),
        "feature_columns": FEATURES, "target": "power_w=-REESSVoltage*REESSCurrent; signed battery-side power",
        "data_contract_version": config.DATA_CONTRACT_VERSION,
        "plan_signal_note": "GECR无调度器计划轨迹；planned_*由已执行行程实测轨迹代理构造，仅用于离线回放验证，不代表真实出发前泛化。",
        "invalid_files": errors, "units_warning": "tot_disc/tot_char的公开表头未注明物理单位，不作为监督标签。",
        "speed_unit": "DAActualSpeed按作者MATLAB代码为km/h", "sampling_seconds": config.SAMPLE_SECONDS,
    }
    config.DATA_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_event("prepare_complete", trip_count=len(trip_ids), row_count_1hz=len(all_data), train_trips=len(train_ids), val_trips=len(val_ids), test_trips=len(test_ids), invalid_file_count=len(errors), summary=str(config.DATA_SUMMARY_JSON))
    return summary


class TCN(nn.Module):
    """轻量因果卷积网络，输入形状为(batch, window, feature)。"""

    def __init__(self, feature_count: int, channels: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        input_channels = feature_count
        for dilation in (1, 2, 4):
            self.blocks.append(TCNBlock(input_channels, channels, kernel_size, dilation, dropout))
            input_channels = channels
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.transpose(1, 2)
        for block in self.blocks:
            signal = block(signal)
        return self.head(signal)[:, :, -1].squeeze(1)


class CausalConv1d(nn.Module):
    """仅使用卷积位置及其历史输入的左填充一维卷积。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    """双卷积膨胀因果残差块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.skip(x))


def _windows(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, target_mean: float, target_std: float, steps: int, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """功能: 为每条行程构造不跨行程的历史窗口。
    参数: df为分区数据，mean/std为输入标准化参数，target_mean/std为目标标准化参数，steps为窗口长度。
    返回: 窗口输入、标准化目标、行程编号和原始目标。
    调用位置: train_model、search_tcn。
    """
    features = features or FEATURES
    xs, ys, trips, originals = [], [], [], []
    for trip, part in df.groupby("trip_id", sort=False):
        arr = ((part[features].to_numpy(np.float32) - mean) / std).astype(np.float32)
        target = part[TARGET].to_numpy(np.float32)
        padded = np.pad(arr, ((steps - 1, 0), (0, 0)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, steps, axis=0).transpose(0, 2, 1).copy()
        xs.extend(windows); ys.extend((target - target_mean) / target_std); originals.extend(target); trips.extend([trip] * len(part))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), trips, np.asarray(originals, np.float32)


def _validate_window_coverage(frame: pd.DataFrame, steps: int, count: int) -> None:
    """功能: 确认窗口构造覆盖分区内所有合格行程而非单条行程。
    参数: frame为原始分区，steps为上下文长度，count为生成窗口数。
    返回: None；数量不一致时抛出异常。
    调用位置: search_tcn、train_model、search_rls。
    """
    expected = sum(len(part) for _, part in frame.groupby("trip_id", sort=False))
    if count != expected:
        raise RuntimeError(f"窗口覆盖检查失败: 实际{count}，预期{expected}，窗口长度{steps}")


def _fit_scaler(train: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """功能: 只用训练行程拟合输入和功率目标标准化参数。
    参数: train为训练数据表。
    返回: 输入均值、输入标准差、目标均值和目标标准差。
    调用位置: search_tcn、train_model。
    """
    features = features or FEATURES
    mean = train[features].mean().to_numpy(float); std = train[features].std().replace(0, 1).fillna(1).to_numpy(float)
    target_mean = float(train[TARGET].mean()); target_std = max(float(train[TARGET].std()), 1.0)
    config.SCALER_FILE.write_text(json.dumps({"features": features, "mean": mean.tolist(), "std": std.tolist(), "target_mean": target_mean, "target_std": target_std}, ensure_ascii=False, indent=2), encoding="utf-8")
    return mean, std, target_mean, target_std


def _device() -> torch.device:
    return torch.device(config.DEVICE if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")


def _fit_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, params: dict, seed: int, epochs: int, patience: int) -> tuple[dict, list[dict]]:
    """功能: 训练单个TCN候选并保留验证集最优状态。
    参数: x/y为训练和验证窗口，params为网络超参数，seed为随机种子，epochs/patience为训练控制参数。
    返回: checkpoint和训练日志。
    调用位置: search_tcn、train_model。
    """
    _set_seed(seed); device = _device()
    model = TCN(x_train.shape[-1], params["channels"], params["kernel_size"], params["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=config.BATCH_SIZE, shuffle=True)
    best, state, wait, logs = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = nn.functional.smooth_l1_loss(model(xb.to(device)), yb.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); total += float(loss.item()) * len(xb)
        model.eval()
        model.eval(); val_total = 0.0
        val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=config.BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for xb, yb in val_loader:
                val_total += float(nn.functional.smooth_l1_loss(model(xb.to(device)), yb.to(device), reduction="sum").item())
        val_loss = val_total / len(x_val)
        logs.append({"epoch": epoch, "train_loss": total / len(x_train), "val_loss": val_loss, "seed": seed})
        if val_loss < best: best, state, wait = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else: wait += 1
        if wait >= patience: break
    return {"state_dict": state, "best_val_loss": best, "params": params, "seed": seed}, logs


def _predict(model: TCN, x: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    model = model.to(_device()); model.eval()
    with torch.no_grad(): return model(torch.from_numpy(x).to(_device())).cpu().numpy() * target_std + target_mean


def _seed_predictions(frame: pd.DataFrame, checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    """功能: 对给定行程表生成逐秒多种子功率预测和实测功率。
    参数: frame为完整行程表，checkpoint为正式TCN权重与预处理参数。
    返回: 种子预测矩阵和实测功率向量。
    调用位置: predict_test、功率区间校准。
    """
    features = checkpoint["features"]
    x, _, _, actual = _windows(frame, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), checkpoint["target_mean"], checkpoint["target_std"], int(checkpoint["window_steps"]), features)
    params = checkpoint["params"]
    predictions = []
    for state in checkpoint["state_dicts"]:
        model = TCN(len(features), int(params["channels"]), int(params["kernel_size"]), float(params["dropout"]))
        model.load_state_dict(state)
        predictions.append(_predict(model, x, checkpoint["target_mean"], checkpoint["target_std"]))
    return np.stack(predictions, axis=0), actual


def _split_conformal_quantile(scores: list[float], coverage: float) -> float:
    """功能: 计算有限样本split-conformal次序统计量。
    参数: scores为独立校准单元分数，coverage为目标覆盖率。
    返回: 有限样本校准分位数；样本不足时返回无穷大。
    调用位置: predict_test、RLS区间校准。
    """
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * coverage))
    return float(ordered[rank - 1]) if rank <= len(ordered) else float("inf")


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), 1e-6) * 100)


def _selection(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> dict:
    temp = frame.iloc[-len(predicted):].copy().reset_index(drop=True)
    temp["actual"] = actual; temp["predicted"] = predicted; temp["energy"] = temp.power_w * config.SAMPLE_SECONDS / 3600; temp["pred_energy"] = temp.predicted * config.SAMPLE_SECONDS / 3600
    temp["window"] = temp.groupby("trip_id").cumcount() // 5
    window = temp.groupby(["trip_id", "window"], as_index=False)[["energy", "pred_energy"]].sum()
    trip = temp.groupby("trip_id", as_index=False)[["energy", "pred_energy"]].sum()
    power = _wape(temp.actual.to_numpy(), temp.predicted.to_numpy()); energy = _wape(window.energy.to_numpy(), window.pred_energy.to_numpy()); trip_wape = _wape(trip.energy.to_numpy(), trip.pred_energy.to_numpy())
    return {"power_wape_percent": power, "window_5s_wape_percent": energy, "trip_energy_wape_percent": trip_wape, "selection_score": 0.2 * power + energy + 0.5 * trip_wape}


def search_tcn() -> dict:
    """功能: 对TCN窗口、网络参数和三个随机种子执行验证集搜索。
    参数: 无，读取配置和已清洗数据。
    返回: 最优窗口、参数、种子和搜索表路径。
    调用位置: main的search或all模式。
    """
    train = pd.read_csv(config.TRAIN_CSV); val = pd.read_csv(config.VAL_CSV)
    trial_rows = []; ablation_rows = []
    config.SEARCH_RESULTS_CSV.unlink(missing_ok=True); config.SEARCH_TRIALS_CSV.unlink(missing_ok=True)
    config.FEATURE_ABLATION_CSV.unlink(missing_ok=True)

    def evaluate_candidate(features: list[str], steps: int, params: dict, seeds: tuple[int, ...], stage: str) -> tuple[dict, list[dict]]:
        mean, std, tm, ts = _fit_scaler(train, features)
        xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps, features)
        xv, yv, _, orig = _windows(val, mean, std, tm, ts, steps, features)
        _validate_window_coverage(train, steps, len(xtr)); _validate_window_coverage(val, steps, len(xv))
        stride = config.REFINE_TRAIN_STRIDE if stage == "复核" else config.SEARCH_TRAIN_STRIDE
        xtr, ytr = xtr[::stride], ytr[::stride]
        val_frame = val.copy().reset_index(drop=True)
        metrics_by_seed = []; details = []
        for seed in seeds:
            ckpt, log = _fit_model(xtr, ytr, xv, yv, params, seed, config.SEARCH_EPOCHS if stage != "复核" else config.REFINE_EPOCHS, config.SEARCH_PATIENCE + (5 if stage == "复核" else 0))
            model = TCN(len(features), params["channels"], params["kernel_size"], params["dropout"]); model.load_state_dict(ckpt["state_dict"])
            predicted = _predict(model, xv, tm, ts); metric = _selection(val_frame, orig, predicted); metrics_by_seed.append(metric)
            details.append({"feature_group": stage, "features": ",".join(features), "window_steps": steps, **params, "seed": seed, "best_val_loss": ckpt["best_val_loss"], **metric, "epochs_run": len(log)})
            print(f"TCN候选 阶段={stage} | 窗口={steps}s | seed={seed} | 功率WAPE={metric['power_wape_percent']:.4f}% | 5秒能耗WAPE={metric['window_5s_wape_percent']:.4f}% | 整程能耗WAPE={metric['trip_energy_wape_percent']:.4f}% | 选择分数={metric['selection_score']:.4f}")
        averaged = {key: float(np.mean([m[key] for m in metrics_by_seed])) for key in metrics_by_seed[0]}
        return {"features": features, "window_steps": steps, **params, **averaged, "seed_count": len(seeds), "search_stage": stage}, details

    for group_name, features in config.FEATURE_GROUPS.items():
        candidates = []
        for steps in config.WINDOW_CANDIDATES:
            result, details = evaluate_candidate(features, steps, config.ABLATION_PARAMS, config.SEARCH_SEEDS, "特征消融:" + group_name)
            candidates.append(result); trial_rows.extend(details)
        best_group = min(candidates, key=lambda item: item["selection_score"])
        ablation_rows.append({"feature_group": group_name, "features": ",".join(features), "feature_count": len(features), **{key: best_group[key] for key in ("window_steps", "power_wape_percent", "window_5s_wape_percent", "trip_energy_wape_percent", "selection_score")}})
        pd.DataFrame(trial_rows).to_csv(config.SEARCH_TRIALS_CSV, index=False, encoding="utf-8")
        pd.DataFrame(ablation_rows).to_csv(config.FEATURE_ABLATION_CSV, index=False, encoding="utf-8")

    selected_group = min(ablation_rows, key=lambda item: item["selection_score"])["feature_group"]
    features = config.FEATURE_GROUPS[selected_group]; rows = []
    for steps in config.WINDOW_CANDIDATES:
        for params in config.MODEL_CANDIDATES:
            result, details = evaluate_candidate(features, steps, params, config.SEARCH_SEEDS, "超参数:" + selected_group)
            rows.append(result); trial_rows.extend(details)
            pd.DataFrame(trial_rows).to_csv(config.SEARCH_TRIALS_CSV, index=False, encoding="utf-8")
            pd.DataFrame(rows).to_csv(config.SEARCH_RESULTS_CSV, index=False, encoding="utf-8")
    shortlist = sorted(rows, key=lambda item: item["selection_score"])[:config.REFINE_TOP_CANDIDATES]
    refined_rows = []
    for candidate in shortlist:
        params = {key: candidate[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}
        result, details = evaluate_candidate(features, int(candidate["window_steps"]), params, config.REFINE_SEEDS, "复核")
        refined_rows.append(result); trial_rows.extend(details)
        pd.DataFrame(trial_rows).to_csv(config.SEARCH_TRIALS_CSV, index=False, encoding="utf-8")
    result = pd.DataFrame(refined_rows).sort_values("selection_score").reset_index(drop=True)
    result.to_csv(config.SEARCH_RESULTS_CSV, index=False, encoding="utf-8")
    best = result.iloc[0].to_dict(); config.SELECTED_CONFIG_JSON.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_event("search_tcn_complete", candidate_count=len(result), trial_count=len(trial_rows), selected=best, search_file=str(config.SEARCH_RESULTS_CSV))
    return {"candidate_count": len(result), "trial_count": len(trial_rows), "feature_ablation_file": str(config.FEATURE_ABLATION_CSV), "selected": best, "search_file": str(config.SEARCH_RESULTS_CSV)}


def train_model() -> dict:
    """功能: 使用验证集最优配置进行多种子集成训练并生成测试功率预测。
    参数: 无，读取搜索结果和数据表。
    返回: 训练和预测摘要。
    调用位置: main的train或all模式。
    """
    train = pd.read_csv(config.TRAIN_CSV); val = pd.read_csv(config.VAL_CSV); test = pd.read_csv(config.TEST_CSV)
    selected = json.loads(config.SELECTED_CONFIG_JSON.read_text(encoding="utf-8")); features = list(selected["features"]); steps = int(selected["window_steps"]); params = {key: selected[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}; params["channels"] = int(params["channels"]); params["kernel_size"] = int(params["kernel_size"])
    mean, std, tm, ts = _fit_scaler(train, features)
    xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps, features); xv, yv, _, _ = _windows(val, mean, std, tm, ts, steps, features); xt, _, _, _ = _windows(test, mean, std, tm, ts, steps, features)
    _validate_window_coverage(train, steps, len(xtr)); _validate_window_coverage(val, steps, len(xv)); _validate_window_coverage(test, steps, len(xt))
    logs = []; checkpoints = []
    for seed in config.REFINE_SEEDS:
        ckpt, log = _fit_model(xtr, ytr, xv, yv, params, seed, config.EPOCHS, config.SEARCH_PATIENCE + 3); checkpoints.append(ckpt["state_dict"]); logs.extend([{**row, "seed": seed, "phase": "final"} for row in log])
        print(f"正式训练 seed={seed} | epochs={len(log)} | best_val_smooth_l1={ckpt['best_val_loss']:.6f} | train_samples={len(xtr)} | val_samples={len(xv)}")
    log_frame = pd.DataFrame(logs)
    log_frame.to_csv(config.TRAIN_LOG_CSV, index=False, encoding="utf-8")
    log_frame.to_json(config.TRAINING_EPOCH_LOG_JSONL, orient="records", lines=True, force_ascii=False)
    torch.save({"architecture_version": ARCHITECTURE_VERSION, "features": features, "window_steps": steps, "params": params, "target_mean": tm, "target_std": ts, "mean": mean.tolist(), "std": std.tolist(), "seed_count": len(checkpoints), "state_dicts": checkpoints}, config.MODEL_FILE)
    _log_event("train_complete", architecture=ARCHITECTURE_VERSION, selected_config=selected, seeds=list(config.REFINE_SEEDS), device=str(_device()), best_val_loss_by_seed=log_frame.groupby("seed").val_loss.min().to_dict(), model=str(config.MODEL_FILE))
    prediction_summary = predict_test()
    return {"device": str(_device()), "window_steps": steps, "features": features, "selected_params": params, "seed_count": len(checkpoints), "seeds": list(config.REFINE_SEEDS), "test_rows": prediction_summary["prediction_rows"], "model": str(config.MODEL_FILE)}


def predict_test() -> dict:
    """功能: 从已保存的三种子模型载入权重并重新生成冻结测试集预测。
    参数: 无，读取模型文件和测试数据。
    返回: 测试行程数、逐秒样本数及预测表路径。
    调用位置: main的predict模式。
    """
    checkpoint = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("模型权重不属于当前因果TCN架构，请先重新搜索并训练")
    features = checkpoint.get("features", [])
    if not features or not set(features).issubset(FEATURES):
        raise ValueError("模型输入字段与当前计划信号契约不一致，请重新搜索并训练")
    test = pd.read_csv(config.TEST_CSV); calibration = pd.read_csv(config.CALIBRATION_CSV); steps = int(checkpoint["window_steps"])
    seed_predictions, actual = _seed_predictions(test, checkpoint)
    calibration_predictions, calibration_actual = _seed_predictions(calibration, checkpoint)
    _validate_window_coverage(test, steps, len(actual)); _validate_window_coverage(calibration, steps, len(calibration_actual))
    predicted = np.mean(seed_predictions, axis=0)
    seed_std = np.std(seed_predictions, axis=0, ddof=1) if len(seed_predictions) > 1 else np.zeros_like(predicted)
    power_scores = []
    cursor = 0
    for _, part in calibration.groupby("trip_id", sort=False):
        end = cursor + len(part)
        calibration_mean = calibration_predictions[:, cursor:end].mean(axis=0)
        power_scores.append(float(np.quantile(np.abs(calibration_actual[cursor:end] - calibration_mean), 0.95)))
        cursor = end
    power_quantile = _split_conformal_quantile(power_scores, config.CALIBRATION_COVERAGE)
    ci_half = np.full_like(predicted, power_quantile, dtype=float)
    output = test.copy().reset_index(drop=True)
    output["actual_speed_kmh"] = output["speed_kmh"].to_numpy()
    output["input_time_s"] = output.time_s.to_numpy()
    output["predicted_power_w"] = predicted; output["actual_power_w"] = actual
    output["power_seed_std_w"] = seed_std
    output["power_interval_half_width_w"] = ci_half
    output["power_prediction_interval_low_w"] = predicted - ci_half
    output["power_prediction_interval_high_w"] = predicted + ci_half
    output["power_ci95_low_w"] = output.power_prediction_interval_low_w
    output["power_ci95_high_w"] = output.power_prediction_interval_high_w
    for seed_index, seed_power in enumerate(seed_predictions):
        output[f"seed_{seed_index + 1}_predicted_power_w"] = seed_power
    output["predicted_energy_wh"] = predicted * config.SAMPLE_SECONDS / 3600; output["actual_energy_wh"] = actual * config.SAMPLE_SECONDS / 3600
    output["energy_ci95_low_wh"] = output.power_ci95_low_w * config.SAMPLE_SECONDS / 3600
    output["energy_ci95_high_wh"] = output.power_ci95_high_w * config.SAMPLE_SECONDS / 3600
    output["cumulative_energy_wh"] = output.groupby("trip_id").predicted_energy_wh.cumsum()
    for seed_index in range(len(seed_predictions)):
        seed_energy_column = f"seed_{seed_index + 1}_predicted_energy_wh"
        output[seed_energy_column] = output[f"seed_{seed_index + 1}_predicted_power_w"] * config.SAMPLE_SECONDS / 3600
    output["cumulative_energy_ci95_low_wh"] = output.groupby("trip_id").energy_ci95_low_wh.cumsum()
    output["cumulative_energy_ci95_high_wh"] = output.groupby("trip_id").energy_ci95_high_wh.cumsum()
    elapsed_s = output.groupby("trip_id").cumcount().add(1) * config.SAMPLE_SECONDS
    output["mean_energy_rate_wh_per_s"] = output.cumulative_energy_wh / elapsed_s
    output["mean_energy_rate_ci95_low_wh_per_s"] = output.cumulative_energy_ci95_low_wh / elapsed_s
    output["mean_energy_rate_ci95_high_wh_per_s"] = output.cumulative_energy_ci95_high_wh / elapsed_s
    output.to_csv(config.PREDICTIONS_CSV, index=False, encoding="utf-8")
    calibration_output = calibration.copy().reset_index(drop=True)
    calibration_output["actual_power_w"] = calibration_actual
    calibration_output["predicted_power_w"] = calibration_predictions.mean(axis=0)
    calibration_output["actual_energy_wh"] = calibration_output.actual_power_w * config.SAMPLE_SECONDS / 3600
    calibration_output["predicted_energy_wh"] = calibration_output.predicted_power_w * config.SAMPLE_SECONDS / 3600
    calibration_output.to_csv(config.CALIBRATION_PREDICTIONS_CSV, index=False, encoding="utf-8")
    calibration_info = {"method": "trip-calibrated pointwise power interval", "target_coverage": config.CALIBRATION_COVERAGE, "calibration_trip_count": len(power_scores), "quantile_w": power_quantile, "score": "each calibration trip's 95th percentile of absolute per-second power residual; finite-sample quantile across trips", "interpretation": "calibrated per-second power error band; not a whole-trip simultaneous band", "finite_sample_rank": int(np.ceil((len(power_scores) + 1) * config.CALIBRATION_COVERAGE))}
    config.POWER_INTERVAL_CALIBRATION_JSON.write_text(json.dumps(calibration_info, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_event("predict_test_complete", test_trips=int(output.trip_id.nunique()), calibration_trips=len(power_scores), prediction_rows=len(output), prediction_file=str(config.PREDICTIONS_CSV), power_interval_method=calibration_info["method"], power_interval_coverage=config.CALIBRATION_COVERAGE, power_interval_half_width_w=power_quantile)
    return {"test_trips": int(output.trip_id.nunique()), "calibration_trips": len(power_scores), "prediction_rows": len(output), "power_interval_half_width_w": power_quantile, "predictions_file": str(config.PREDICTIONS_CSV)}


def _rls_predict(actual: np.ndarray, baseline: np.ndarray, window_size: int, forgetting: float, covariance: float) -> np.ndarray:
    """功能: 对窗口能耗执行先预测、后更新的RLS校正。
    参数: actual/baseline为逐秒能耗，window_size为窗口长度，forgetting/covariance为RLS参数。
    返回: 逐窗口校正后的能耗。
    调用位置: search_rls、evaluate.
    """
    theta = np.array([0.0, 1.0]); P = covariance * np.eye(2); output = []
    for start in range(0, len(actual), window_size):
        a = float(actual[start:start + window_size].sum()); b = float(baseline[start:start + window_size].sum()); phi = np.array([1.0, b]); output.append(float(phi @ theta)); gain = P @ phi / (forgetting + phi @ P @ phi); theta = theta + gain * (a - phi @ theta); P = (P - np.outer(gain, phi) @ P) / forgetting
    return np.asarray(output)


def _rls_final_trajectory(trip_id: str, actual: np.ndarray, baseline: np.ndarray, window_size: int, forgetting: float, covariance: float, residual_variance_prior: float = 0.0) -> list[dict]:
    """功能: 回放逐窗口RLS反馈并重算整段最终能耗预测。
    参数: trip_id为行程标识，actual/baseline为秒级能量，window_size为反馈窗口点数，forgetting/covariance为递推参数。
    返回: 每次反馈后的已观测能量、最终能耗估计、剩余参数方差和真实终值记录。
    调用位置: calibration/evaluate流程。
    """
    actual = np.asarray(actual, dtype=float); baseline = np.asarray(baseline, dtype=float)
    actual_windows = np.array([actual[i:i + window_size].sum() for i in range(0, len(actual), window_size)])
    baseline_windows = np.array([baseline[i:i + window_size].sum() for i in range(0, len(baseline), window_size)])
    theta = np.array([0.0, 1.0]); covariance_matrix = covariance * np.eye(2)
    observed_energy = 0.0; records = []; count = len(actual_windows)
    residual_variance = max(float(residual_variance_prior), 0.0)
    for index, (actual_window, baseline_window) in enumerate(zip(actual_windows, baseline_windows)):
        phi = np.array([1.0, baseline_window])
        prediction_before_feedback = float(phi @ theta)
        innovation = float(actual_window - prediction_before_feedback)
        gain = covariance_matrix @ phi / (forgetting + phi @ covariance_matrix @ phi)
        theta = theta + gain * (actual_window - prediction_before_feedback)
        covariance_matrix = (covariance_matrix - np.outer(gain, phi) @ covariance_matrix) / forgetting
        residual_variance = 0.9 * residual_variance + 0.1 * innovation ** 2
        observed_energy += float(actual_window)
        remaining = baseline_windows[index + 1:]
        remaining_features = np.column_stack([np.ones(len(remaining)), remaining]) if len(remaining) else np.empty((0, 2))
        remaining_prediction = remaining_features @ theta if len(remaining) else np.empty(0)
        sum_features = remaining_features.sum(axis=0)
        parameter_variance = float(sum_features @ covariance_matrix @ sum_features) if len(remaining) else 0.0
        residual_prediction_variance = len(remaining) * residual_variance
        final_prediction = observed_energy + float(remaining_prediction.sum())
        records.append({"trip_id": str(trip_id), "completed_windows": index + 1, "total_windows": count, "remaining_windows": len(remaining), "progress": (index + 1) / count, "observed_energy_wh": observed_energy, "predicted_remaining_energy_wh": float(remaining_prediction.sum()), "predicted_final_energy_wh": final_prediction, "actual_final_energy_wh": float(actual_windows.sum()), "parameter_variance_wh2": max(parameter_variance, 0.0), "remaining_residual_variance_wh2": max(residual_prediction_variance, 0.0), "baseline_final_energy_wh": float(baseline_windows.sum())})
    return records


def _rls_trajectory_for_frame(frame: pd.DataFrame, window_size: int, forgetting: float, covariance: float, residual_variance_prior: float = 0.0) -> pd.DataFrame:
    """功能: 将秒级实测和预测能量逐行程聚合并执行在线RLS回放。
    参数: frame含trip_id、actual_energy_wh和predicted_energy_wh，其他参数为RLS配置。
    返回: 所有行程每个反馈点的整段能耗估计轨迹。
    调用位置: evaluate_and_plot。
    """
    records = []
    for trip_id, part in frame.groupby("trip_id", sort=False):
        records.extend(_rls_final_trajectory(trip_id, part.actual_energy_wh.to_numpy(), part.predicted_energy_wh.to_numpy(), window_size, forgetting, covariance, residual_variance_prior))
    return pd.DataFrame(records)


def search_rls() -> dict:
    """功能: 在验证集按完整行程搜索RLS窗口和递推参数。
    参数: 无。
    返回: 最优RLS组合和搜索表路径。
    调用位置: main的search或all模式。
    """
    val = pd.read_csv(config.VAL_CSV); selected = json.loads(config.SELECTED_CONFIG_JSON.read_text(encoding="utf-8")); features = list(selected["features"]); steps = int(selected["window_steps"]); train = pd.read_csv(config.TRAIN_CSV); mean, std, tm, ts = _fit_scaler(train, features); xv, yv, _, orig = _windows(val, mean, std, tm, ts, steps, features)
    _validate_window_coverage(val, steps, len(xv))
    # 用最终选定配置在验证集重新拟合一个基线，搜索只读取其预测，不接触测试集。
    params = {key: selected[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}; params["channels"] = int(params["channels"]); params["kernel_size"] = int(params["kernel_size"]); xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps, features); ckpt, _ = _fit_model(xtr, ytr, xv, yv, params, config.RANDOM_SEED, config.EPOCHS, config.SEARCH_PATIENCE + 3); model = TCN(len(features), params["channels"], params["kernel_size"], float(params["dropout"])); model.load_state_dict(ckpt["state_dict"]); base = _predict(model, xv, tm, ts) * config.SAMPLE_SECONDS / 3600
    val_frame = val.copy().reset_index(drop=True); base_frame = val_frame.copy(); base_frame["actual_energy"] = orig * config.SAMPLE_SECONDS / 3600; base_frame["baseline_energy"] = base
    rows = []
    for seconds in config.RLS_WINDOWS_SECONDS:
        for forgetting in config.RLS_FORGETTING_FACTORS:
            for covariance in config.RLS_INITIAL_COVARIANCES:
                all_actual, all_base, corrected = [], [], []; actual_trip, base_trip, corrected_trip = [], [], []
                for _, part in base_frame.groupby("trip_id", sort=False):
                    a = part.actual_energy.to_numpy(); b = part.baseline_energy.to_numpy(); grouped = pd.DataFrame({"a": a, "b": b}).groupby(np.arange(len(a)) // seconds, sort=True).sum(); aw = grouped.a.to_numpy(); bw = grouped.b.to_numpy(); r = _rls_predict(aw, bw, 1, forgetting, covariance); all_actual.append(aw); all_base.append(bw); corrected.append(r); actual_trip.append(float(aw.sum())); base_trip.append(float(bw.sum())); corrected_trip.append(float(r.sum()))
                a = np.concatenate(all_actual); b = np.concatenate(all_base); r = np.concatenate(corrected); trip_wape = _wape(np.asarray(actual_trip), np.asarray(corrected_trip)); rows.append({"window_seconds": seconds, "forgetting_factor": forgetting, "initial_covariance": covariance, "tcn_window_wape_percent": _wape(a, b), "rls_window_wape_percent": _wape(a, r), "tcn_trip_wape_percent": _wape(np.asarray(actual_trip), np.asarray(base_trip)), "rls_trip_wape_percent": trip_wape, "selection_score": trip_wape})
                print(f"RLS候选 window={seconds}s | forgetting={forgetting:.2f} | covariance={covariance:g} | 秒级能耗窗口WAPE={rows[-1]['rls_window_wape_percent']:.4f}% | 整程能耗WAPE={trip_wape:.4f}%")
    result = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True); result.to_csv(config.RLS_SEARCH_CSV, index=False, encoding="utf-8"); best = result.iloc[0].to_dict(); selected.update({"rls_window_seconds": best["window_seconds"], "rls_forgetting_factor": best["forgetting_factor"], "rls_initial_covariance": best["initial_covariance"]}); config.SELECTED_CONFIG_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"); _log_event("search_rls_complete", candidate_count=len(result), selected=best, search_file=str(config.RLS_SEARCH_CSV)); return {"candidate_count": len(result), "selected": best, "search_file": str(config.RLS_SEARCH_CSV)}


def evaluate_and_plot() -> dict:
    """功能: 依据冻结的模型预测和RLS配置生成测试指标、图表与窗口结果。
    参数: 无。
    返回: TCN、RLS和逐行程指标。
    调用位置: main的evaluate或all模式。
    """
    pred = pd.read_csv(config.PREDICTIONS_CSV); selected = json.loads(config.SELECTED_CONFIG_JSON.read_text(encoding="utf-8")); window = int(selected["rls_window_seconds"]); ff = float(selected["rls_forgetting_factor"]); cov = float(selected["rls_initial_covariance"]); pred["window"] = pred.groupby("trip_id").cumcount() // window
    windows = pred.groupby(["trip_id", "window"], as_index=False).agg(actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum")); corrected = []
    for _, part in windows.groupby("trip_id", sort=False): corrected.extend(_rls_predict(part.actual_energy_wh.to_numpy(), part.predicted_energy_wh.to_numpy(), 1, ff, cov).tolist())
    windows["rls_energy_wh"] = corrected; windows.to_csv(config.RLS_WINDOWS_CSV, index=False, encoding="utf-8")
    validation = pd.read_csv(config.VAL_CSV)
    checkpoint = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    validation_seed_predictions, validation_actual = _seed_predictions(validation, checkpoint)
    validation_frame = validation.copy().reset_index(drop=True)
    validation_frame["actual_energy_wh"] = validation_actual * config.SAMPLE_SECONDS / 3600
    validation_frame["predicted_energy_wh"] = validation_seed_predictions.mean(axis=0) * config.SAMPLE_SECONDS / 3600
    validation_frame["window"] = validation_frame.groupby("trip_id").cumcount() // window
    validation_windows = validation_frame.groupby(["trip_id", "window"], as_index=False).agg(actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"))
    validation_innovations = []
    for _, part in validation_windows.groupby("trip_id", sort=False):
        corrected_validation = _rls_predict(part.actual_energy_wh.to_numpy(), part.predicted_energy_wh.to_numpy(), 1, ff, cov)
        validation_innovations.extend((part.actual_energy_wh.to_numpy() - corrected_validation).tolist())
    residual_variance_prior = float(np.mean(np.square(validation_innovations))) if validation_innovations else 0.0
    test_trajectory = _rls_trajectory_for_frame(pred, window, ff, cov, residual_variance_prior)
    calibration = pd.read_csv(config.CALIBRATION_PREDICTIONS_CSV)
    calibration_trajectory = _rls_trajectory_for_frame(calibration, window, ff, cov, residual_variance_prior)
    calibration_trajectory["total_prediction_variance_wh2"] = calibration_trajectory.parameter_variance_wh2 + calibration_trajectory.remaining_residual_variance_wh2
    calibration_trajectory["normalized_final_error"] = np.abs(calibration_trajectory.actual_final_energy_wh - calibration_trajectory.predicted_final_energy_wh) / np.sqrt(np.maximum(calibration_trajectory.total_prediction_variance_wh2, 1.0))
    calibration_trip_scores = calibration_trajectory.groupby("trip_id").normalized_final_error.max().tolist()
    rls_quantile = _split_conformal_quantile(calibration_trip_scores, config.CALIBRATION_COVERAGE)
    test_trajectory["total_prediction_variance_wh2"] = test_trajectory.parameter_variance_wh2 + test_trajectory.remaining_residual_variance_wh2
    test_trajectory["interval_radius_wh"] = np.where(test_trajectory.remaining_windows > 0, rls_quantile * np.sqrt(np.maximum(test_trajectory.total_prediction_variance_wh2, 1.0)), 0.0)
    test_trajectory["predicted_final_lower_wh"] = test_trajectory.predicted_final_energy_wh - test_trajectory.interval_radius_wh
    test_trajectory["predicted_final_upper_wh"] = test_trajectory.predicted_final_energy_wh + test_trajectory.interval_radius_wh
    test_trajectory.to_csv(config.RLS_TRAJECTORY_CSV, index=False, encoding="utf-8")
    calibration_trajectory.to_csv(config.RLS_CALIBRATION_REPLAY_CSV, index=False, encoding="utf-8")
    trips = windows.groupby("trip_id", as_index=False)[["actual_energy_wh", "predicted_energy_wh", "rls_energy_wh"]].sum(); pred["distance_km"] = np.maximum(pred.actual_speed_kmh, 0) * config.SAMPLE_SECONDS / 3600; distance = pred.groupby("trip_id", as_index=False).distance_km.sum(); trips = trips.merge(distance, on="trip_id"); trips["actual_wh_per_km"] = trips.actual_energy_wh / trips.distance_km.clip(lower=.01); trips["tcn_wh_per_km"] = trips.predicted_energy_wh / trips.distance_km.clip(lower=.01); trips["rls_wh_per_km"] = trips.rls_energy_wh / trips.distance_km.clip(lower=.01); trips.to_csv(config.TRIP_METRICS_CSV, index=False, encoding="utf-8")
    def metric(a, b):
        e = a - b; return {"mae_wh": float(np.mean(np.abs(e))), "rmse_wh": float(np.sqrt(np.mean(e ** 2))), "mape_percent": float(np.mean(np.abs(e) / np.maximum(np.abs(a), 1e-3)) * 100), "wape_percent": _wape(a, b)}
    power_wape = _wape(pred.actual_power_w.to_numpy(), pred.predicted_power_w.to_numpy())
    power_point_coverage = float(((pred.actual_power_w >= pred.power_prediction_interval_low_w) & (pred.actual_power_w <= pred.power_prediction_interval_high_w)).mean() * 100)
    power_trip_coverage = float(pred.groupby("trip_id").apply(lambda part: ((part.actual_power_w >= part.power_prediction_interval_low_w) & (part.actual_power_w <= part.power_prediction_interval_high_w)).all()).mean() * 100)
    online_rls = test_trajectory[test_trajectory.progress < 1.0]
    rls_coverage = float(((online_rls.actual_final_energy_wh >= online_rls.predicted_final_lower_wh) & (online_rls.actual_final_energy_wh <= online_rls.predicted_final_upper_wh)).mean() * 100) if len(online_rls) else float("nan")
    windows["tcn_abs_error"] = np.abs(windows.actual_energy_wh - windows.predicted_energy_wh)
    windows["rls_abs_error"] = np.abs(windows.actual_energy_wh - windows.rls_energy_wh)
    metrics = {"tcn": metric(trips.actual_energy_wh.to_numpy(), trips.predicted_energy_wh.to_numpy()), "tcn_rls": metric(trips.actual_energy_wh.to_numpy(), trips.rls_energy_wh.to_numpy()), "trip_count": len(trips), "window_count": len(windows), "prediction_rows": len(pred), "power_wape_percent": power_wape, "window_energy_wape_percent": _wape(windows.actual_energy_wh.to_numpy(), windows.predicted_energy_wh.to_numpy()), "rls_window_energy_wape_percent": _wape(windows.actual_energy_wh.to_numpy(), windows.rls_energy_wh.to_numpy()), "prediction_intervals": {"nominal_coverage": config.CALIBRATION_COVERAGE, "power_method": "trip-calibrated pointwise interval from each calibration trip's 95th-percentile absolute per-second residual", "power_interval_half_width_w": float(pred.power_interval_half_width_w.iloc[0]), "power_point_empirical_coverage_percent": power_point_coverage, "power_whole_trip_empirical_coverage_percent": power_trip_coverage, "power_interval_unit": "逐秒功率误差；不是整条行程同时覆盖区间", "rls_method": "independent-trip conformal calibration of online final-energy errors using remaining-error variance plus RLS parameter covariance", "rls_online_final_energy_empirical_coverage_percent": rls_coverage, "rls_online_snapshot_count": int(len(online_rls)), "rls_calibration_trip_count": int(calibration_trajectory.trip_id.nunique()), "rls_calibration_quantile": rls_quantile, "rls_validation_residual_variance_prior_wh2": residual_variance_prior, "rls_terminal_feedback_excluded": True}, "selected_config": selected, "wh_per_km": {"actual_mean": float(trips.actual_wh_per_km.mean()), "tcn_mean": float(trips.tcn_wh_per_km.mean()), "rls_mean": float(trips.rls_wh_per_km.mean())}}
    config.METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_summary = {"nominal_coverage": config.CALIBRATION_COVERAGE, "power_point_coverage_percent": power_point_coverage, "power_whole_trip_coverage_percent": power_trip_coverage, "rls_online_final_energy_coverage_percent": rls_coverage, "power_calibration_trips": int(calibration.trip_id.nunique()), "rls_calibration_trips": int(calibration_trajectory.trip_id.nunique()), "rls_normalized_trip_quantile": rls_quantile, "rls_residual_variance_prior_wh2": residual_variance_prior, "rls_interval_components": ["剩余窗口预测误差方差", "RLS参数协方差"]}
    config.CALIBRATION_METRICS_JSON.write_text(json.dumps(calibration_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot = pred[pred.trip_id == pred.trip_id.iloc[0]].copy(); plot["actual_cum"] = plot.actual_energy_wh.cumsum(); plot["pred_cum"] = plot.predicted_energy_wh.cumsum(); _plot_all(pred, plot, trips, test_trajectory)
    _log_event("evaluate_complete", trip_count=len(trips), prediction_rows=len(pred), power_wape_percent=power_wape, power_interval_coverage_percent=power_point_coverage, power_trip_coverage_percent=power_trip_coverage, rls_online_interval_coverage_percent=rls_coverage, trip_energy_wape_percent=metrics["tcn"]["wape_percent"], rls_trip_energy_wape_percent=metrics["tcn_rls"]["wape_percent"], output=str(config.METRICS_JSON))
    return metrics


def _save_rls_gifs(trajectory: pd.DataFrame) -> None:
    """功能: 保存RLS整段最终能耗预测及区间宽度的反馈动画。
    参数: trajectory为测试行程逐反馈点预测与校准区间表。
    返回: None。
    调用位置: _plot_all。
    """
    if trajectory.empty:
        return
    counts = trajectory.groupby("trip_id").size()
    target_count = float(counts.median())
    trip_id = (counts - target_count).abs().idxmin()
    part = trajectory[trajectory.trip_id == trip_id].sort_values("completed_windows").reset_index(drop=True)
    frame_ids = np.unique(np.linspace(0, len(part) - 1, min(60, len(part))).round().astype(int))
    baseline = float(part.baseline_final_energy_wh.iloc[0]); actual = float(part.actual_final_energy_wh.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def update_final(frame_index: int):
        ax.clear(); end = frame_ids[frame_index] + 1; shown = part.iloc[:end]
        ax.axhline(baseline, color="tab:gray", linestyle="--", label="TCN基准总能耗")
        ax.axhline(actual, color="black", linestyle=":", label="真实整段能耗")
        ax.plot(shown.progress, shown.predicted_final_energy_wh, color="tab:blue", marker="o", markersize=2, label="RLS最终能耗估计")
        ax.fill_between(shown.progress.to_numpy(float), shown.predicted_final_lower_wh.to_numpy(float), shown.predicted_final_upper_wh.to_numpy(float), color="tab:blue", alpha=.2, label="校准预测区间")
        ax.set(xlim=(0, 1), xlabel="已观测行程比例", ylabel="整段最终能耗估计 (Wh)", title=f"RLS反馈更新 | 测试行程 {trip_id} | 已完成 {int(shown.completed_windows.iloc[-1])}/{int(shown.total_windows.iloc[-1])} 窗口")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=.2)
        return ax.lines

    FuncAnimation(fig, update_final, frames=len(frame_ids), interval=120, blit=False).save(config.GIFS["rls_final_energy"], writer=PillowWriter(fps=8))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def update_width(frame_index: int):
        ax.clear(); end = frame_ids[frame_index] + 1; shown = part.iloc[:end]
        ax.plot(shown.progress, shown.interval_radius_wh * 2, color="tab:red", marker="o", markersize=2, label="预测区间全宽")
        ax.set(xlim=(0, 1), xlabel="已观测行程比例", ylabel="整段能耗区间宽度 (Wh)", title=f"RLS剩余预测不确定性 | 测试行程 {trip_id}")
        ax.legend(loc="best"); ax.grid(alpha=.2)
        return ax.lines

    FuncAnimation(fig, update_width, frames=len(frame_ids), interval=120, blit=False).save(config.GIFS["rls_interval_width"], writer=PillowWriter(fps=8))
    plt.close(fig)


def _plot_all(pred: pd.DataFrame, sample: pd.DataFrame, trips: pd.DataFrame, rls_trajectory: pd.DataFrame) -> None:
    """功能: 生成损失、搜索、散点、残差、行程指标和代表行程图。
    参数: pred为测试逐秒预测，sample为代表行程，trips为逐行程汇总。
    返回: None。
    调用位置: evaluate_and_plot。
    """
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scales = {}

    def save_figure(key: str, x_label: str, y_label: str, units: str, x_limits: tuple[float, float] | None = None, y_limits: tuple[float, float] | None = None) -> None:
        path = config.FIGURES[key]
        if x_limits is not None:
            plt.xlim(*x_limits)
        if y_limits is not None:
            plt.ylim(*y_limits)
        plt.xlabel(x_label); plt.ylabel(y_label); plt.tight_layout()
        width_in, height_in = plt.gcf().get_size_inches()
        plt.savefig(path, dpi=180); plt.close()
        scales[path.name] = {"width_in": float(width_in), "height_in": float(height_in), "dpi": 180, "x_label": x_label, "y_label": y_label, "units": units, "x_limits": x_limits, "y_limits": y_limits}

    if config.TRAIN_LOG_CSV.exists():
        log = pd.read_csv(config.TRAIN_LOG_CSV).groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean(); plt.figure(figsize=(10, 5.5)); plt.plot(log.epoch, log.train_loss, label="三种子平均训练损失"); plt.plot(log.epoch, log.val_loss, label="三种子平均验证损失"); plt.legend(); save_figure("loss", "轮次", "标准化 SmoothL1", "loss")
    if config.SEARCH_RESULTS_CSV.exists():
        search = pd.read_csv(config.SEARCH_RESULTS_CSV).sort_values("selection_score"); plt.figure(figsize=(10, 5.5)); plt.bar(np.arange(len(search)), search.selection_score); save_figure("search", "候选配置（按综合分数排序）", "验证综合分数", "score")
    if config.RLS_SEARCH_CSV.exists():
        search = pd.read_csv(config.RLS_SEARCH_CSV).sort_values("selection_score").reset_index(drop=True); plt.figure(figsize=(10, 5.5)); plt.bar(np.arange(len(search)), search.selection_score); save_figure("rls_search", "RLS候选组合（按验证行程误差排序）", "验证行程能耗 WAPE (%)", "percent")
    lo, hi = float(min(pred.actual_power_w.min(), pred.predicted_power_w.min())), float(max(pred.actual_power_w.max(), pred.predicted_power_w.max()))
    plt.figure(figsize=(10, 5.5)); plt.scatter(pred.actual_power_w, pred.predicted_power_w, s=3, alpha=.18); plt.plot([lo, hi], [lo, hi], "r--"); save_figure("power_scatter", "实测功率 (W)", "预测功率 (W)", "W", (lo, hi), (lo, hi))
    plt.figure(figsize=(10, 5.5)); plt.hist(pred.predicted_power_w - pred.actual_power_w, bins=100); save_figure("residual", "预测误差 (W)", "样本数", "W/count")
    plt.figure(figsize=(10, 5.5)); x = np.arange(len(trips)); plt.plot(x, trips.actual_energy_wh, "o-", label="实测"); plt.plot(x, trips.predicted_energy_wh, "o-", label="TCN"); plt.plot(x, trips.rls_energy_wh, "o-", label="TCN+RLS"); plt.legend(); save_figure("trip_energy", "测试行程序号", "净能耗 (Wh)", "Wh")
    plt.figure(figsize=(10, 5.5)); plt.plot(sample.time_s, sample.actual_cum, label="实测累计能耗"); plt.plot(sample.time_s, sample.pred_cum, label="TCN累计预测"); plt.fill_between(sample.time_s, sample.cumulative_energy_ci95_low_wh, sample.cumulative_energy_ci95_high_wh, alpha=.2, label="逐秒校准功率边界积分范围"); plt.title(f"GECR真实道路行程 {sample.trip_id.iloc[0]}"); plt.legend(); save_figure("trip_trace", "时间 (s)", "累计电池侧净能耗 (Wh)", "Wh")

    route_summary = pred.groupby("trip_id").agg(distance_km=("distance_km", "sum"), duration_s=("time_s", "max")).sort_values("distance_km")
    route_ids = route_summary.iloc[np.linspace(0, len(route_summary) - 1, min(8, len(route_summary))).round().astype(int)].index.tolist()
    routes = pred[pred.trip_id.isin(route_ids)].copy()
    routes["latitude"] = routes["latitude"].interpolate().bfill()
    routes["longitude"] = routes["longitude"].interpolate().bfill()
    plt.figure(figsize=(10, 7));
    for trip_id, part in routes.groupby("trip_id", sort=True):
        plt.plot(part.longitude, part.latitude, linewidth=1.0, alpha=.8, label=str(trip_id))
    plt.legend(title="trip", ncol=2); save_figure("multi_route_map", "经度 (degree)", "纬度 (degree)", "degree")

    plt.figure(figsize=(12, 6));
    for trip_id, part in routes.groupby("trip_id", sort=True):
        rel = part.time_s / max(float(part.time_s.max()), 1.0)
        plt.plot(rel, part.predicted_power_w, linewidth=.8, alpha=.65, label=str(trip_id))
    plt.legend(title="trip", ncol=4); save_figure("multi_route_power", "行程相对进度", "预测电池功率 (W)", "W", (0, 1))

    plt.figure(figsize=(12, 6));
    for trip_id, part in routes.groupby("trip_id", sort=True):
        rel = part.time_s / max(float(part.time_s.max()), 1.0)
        actual_cum = part.actual_energy_wh.cumsum(); predicted_cum = part.predicted_energy_wh.cumsum()
        plt.plot(rel, actual_cum, linewidth=.8, alpha=.55, label=f"{trip_id} 实测")
        plt.plot(rel, predicted_cum, linewidth=1.0, alpha=.8, linestyle="--", label=f"{trip_id} TCN")
    plt.legend(title="trip / series", ncol=4, fontsize=7); save_figure("multi_route_energy", "行程相对进度", "累计净能耗 (Wh)", "Wh", (0, 1))

    sample = sample.copy()
    plt.figure(figsize=(11, 5.5)); plt.plot(sample.time_s, sample.actual_power_w, color="black", linewidth=.7, alpha=.65, label="实测功率"); plt.plot(sample.time_s, sample.predicted_power_w, color="tab:blue", linewidth=1.0, label="TCN预测均值"); plt.fill_between(sample.time_s, sample.power_ci95_low_w, sample.power_ci95_high_w, color="tab:blue", alpha=.2, label="95%种子区间"); plt.legend(); save_figure("power_confidence", "时间 (s)", "功率 (W)", "W")

    plt.figure(figsize=(11, 5.5)); plt.plot(sample.time_s, sample.mean_energy_rate_wh_per_s * 3600, label="运行平均功率估计"); plt.fill_between(sample.time_s, sample.mean_energy_rate_ci95_low_wh_per_s * 3600, sample.mean_energy_rate_ci95_high_wh_per_s * 3600, alpha=.22, label="95%运行均值种子区间"); plt.legend(); save_figure("energy_confidence", "时间 (s)", "运行平均功率等效值 (W)", "W")

    plt.figure(figsize=(11, 5.5)); plt.plot(sample.time_s, sample.pred_cum, label="累计能耗预测"); plt.plot(sample.time_s, sample.actual_cum, color="black", alpha=.65, label="实测累计能耗"); plt.fill_between(sample.time_s, sample.cumulative_energy_ci95_low_wh, sample.cumulative_energy_ci95_high_wh, alpha=.2, label="逐秒校准功率边界积分范围"); plt.legend(); save_figure("cumulative_energy_confidence", "时间 (s)", "累计净能耗 (Wh)", "Wh")

    width_data = rls_trajectory.assign(interval_width_wh=rls_trajectory.predicted_final_upper_wh - rls_trajectory.predicted_final_lower_wh).copy()
    width_data["progress_bin"] = np.minimum((width_data.progress * 10).astype(int), 9)
    width_summary = width_data.groupby("progress_bin", as_index=False).agg(median_width_wh=("interval_width_wh", "median"), q25_width_wh=("interval_width_wh", lambda values: values.quantile(.25)), q75_width_wh=("interval_width_wh", lambda values: values.quantile(.75)))
    width_summary["progress"] = (width_summary.progress_bin + .5) / 10
    plt.figure(figsize=(10, 5.5)); plt.plot(width_summary.progress, width_summary.median_width_wh, marker="o", label="测试行程区间宽度中位数"); plt.fill_between(width_summary.progress, width_summary.q25_width_wh, width_summary.q75_width_wh, alpha=.2, label="行程间四分位范围"); plt.legend(); save_figure("rls_interval_width", "已观测行程比例（10%分箱）", "整段最终能耗区间宽度 (Wh)", "Wh", (0, 1))
    _save_rls_gifs(rls_trajectory)

    split_stats = pd.read_csv(config.SPLIT_STATS_CSV) if config.SPLIT_STATS_CSV.exists() else pd.DataFrame()
    if not split_stats.empty:
        plt.figure(figsize=(10, 5.5)); xpos = np.arange(len(split_stats)); width=.38; plt.bar(xpos-width/2, split_stats.trip_count, width, label="行程数"); plt.bar(xpos+width/2, split_stats.row_count / 1000, width, label="样本数（千）"); plt.xticks(xpos, split_stats.split); plt.legend(); save_figure("dataset_split", "数据划分", "数量（样本数按千计）", "count")
    plt.figure(figsize=(10, 5.5)); plt.hist(pred.actual_power_w, bins=100, alpha=.55, density=True, label="实测功率"); plt.hist(pred.predicted_power_w, bins=100, alpha=.55, density=True, label="预测功率"); plt.legend(); save_figure("data_distribution", "功率 (W)", "概率密度 (1/W)", "W / 1/W")

    config.FIGURE_SCALES_JSON.write_text(json.dumps(scales, ensure_ascii=False, indent=2), encoding="utf-8")
    trip_stats = trips.merge(route_summary, left_on="trip_id", right_index=True, how="left") if "trip_id" in trips else trips.copy()
    trip_stats.to_csv(config.TEST_ROUTE_STATS_CSV, index=False, encoding="utf-8")
