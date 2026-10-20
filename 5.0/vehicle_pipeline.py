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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config


# 只使用预测时刻已经发生或可测得的车辆状态；不使用由整段行程终点反推的进度。
FEATURES = [
    "speed_kmh", "speed_delta_kmh", "acceleration_kmhps", "speed_mean_5s_kmh",
    "speed_std_5s_kmh", "temperature_c", "elevation_rate_mps", "gear", "time_s",
    "voltage_v", "current_a", "power_observed_w", "power_mean_5s_w", "power_std_5s_w",
]
TARGET = "power_w"


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
    required = {"PhaseTime", "DAActualSpeed", "REESSCurrent", "REESSVoltage", "tempAmb", "elevation", "gear"}
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
        temperature_c=("tempAmb", "mean"), elevation_m=("elevation", "mean"),
        gear=("gear", "median"), voltage_v=("REESSVoltage", "mean"), current_a=("REESSCurrent", "mean"),
        power_w=("power_w", "mean"), source_rows=("power_w", "size"),
    )
    agg["time_s"] -= agg["time_s"].iloc[0]
    agg["trip_id"] = path.stem
    agg["speed_delta_kmh"] = agg["speed_kmh"].diff().fillna(0.0)
    agg["acceleration_kmhps"] = agg["speed_delta_kmh"] / config.SAMPLE_SECONDS
    agg["speed_mean_5s_kmh"] = agg["speed_kmh"].rolling(5, min_periods=1).mean()
    agg["speed_std_5s_kmh"] = agg["speed_kmh"].rolling(5, min_periods=1).std().fillna(0.0)
    agg["power_observed_w"] = agg["power_w"]
    agg["power_mean_5s_w"] = agg["power_w"].rolling(5, min_periods=1).mean()
    agg["power_std_5s_w"] = agg["power_w"].rolling(5, min_periods=1).std().fillna(0.0)
    agg["elevation_rate_mps"] = agg["elevation_m"].diff().fillna(0.0) / config.SAMPLE_SECONDS
    agg["elevation_rate_mps"] = agg["elevation_rate_mps"].clip(-20, 20)
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
        return json.loads(config.DATA_SUMMARY_JSON.read_text(encoding="utf-8"))
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
    n_test = max(1, round(len(trip_ids) * 0.2)); n_val = max(1, round(len(trip_ids) * 0.15))
    test_ids = set(trip_ids[:n_test]); val_ids = set(trip_ids[n_test:n_test + n_val]); train_ids = set(trip_ids[n_test + n_val:])
    all_data["split"] = all_data.trip_id.map(lambda x: "test" if x in test_ids else "val" if x in val_ids else "train")
    all_data.to_csv(config.CLEAN_CSV, index=False, encoding="utf-8")
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        all_data[all_data.trip_id.isin(ids)].to_csv(getattr(config, f"{split.upper()}_CSV"), index=False, encoding="utf-8")
    summary = {
        "dataset": "MOTIVES-LAB GECR real_world_data", "source": str(source),
        "trip_count": len(trip_ids), "row_count_1hz": int(len(all_data)),
        "train_trips": sorted(train_ids), "val_trips": sorted(val_ids), "test_trips": sorted(test_ids),
        "train_rows": int((all_data.split == "train").sum()), "val_rows": int((all_data.split == "val").sum()), "test_rows": int((all_data.split == "test").sum()),
        "feature_columns": FEATURES, "target": "power_w=-REESSVoltage*REESSCurrent; signed battery-side power",
        "invalid_files": errors, "units_warning": "tot_disc/tot_char的公开表头未注明物理单位，不作为监督标签。",
        "speed_unit": "DAActualSpeed按作者MATLAB代码为km/h", "sampling_seconds": config.SAMPLE_SECONDS,
    }
    config.DATA_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class TCN(nn.Module):
    """轻量因果卷积网络，输入形状为(batch, window, feature)。"""

    def __init__(self, feature_count: int, channels: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size - 1
        self.net = nn.Sequential(
            nn.Conv1d(feature_count, channels, kernel_size, padding=pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.ReLU(), nn.Conv1d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))[:, :, -1].squeeze(1)


def _windows(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, target_mean: float, target_std: float, steps: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """功能: 为每条行程构造不跨行程的历史窗口。
    参数: df为分区数据，mean/std为输入标准化参数，target_mean/std为目标标准化参数，steps为窗口长度。
    返回: 窗口输入、标准化目标、行程编号和原始目标。
    调用位置: train_model、search_tcn。
    """
    xs, ys, trips, originals = [], [], [], []
    for trip, part in df.groupby("trip_id", sort=False):
        arr = ((part[FEATURES].to_numpy(np.float32) - mean) / std).astype(np.float32)
        target = part[TARGET].to_numpy(np.float32)
        for end in range(steps - 1, len(part) - 1):
            xs.append(arr[end - steps + 1:end + 1]); ys.append((target[end + 1] - target_mean) / target_std); originals.append(target[end + 1]); trips.append(trip)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), trips, np.asarray(originals, np.float32)


def _validate_window_coverage(frame: pd.DataFrame, steps: int, count: int) -> None:
    """功能: 确认窗口构造覆盖分区内所有合格行程而非单条行程。
    参数: frame为原始分区，steps为上下文长度，count为生成窗口数。
    返回: None；数量不一致时抛出异常。
    调用位置: search_tcn、train_model、search_rls。
    """
    expected = sum(max(len(part) - steps, 0) for _, part in frame.groupby("trip_id", sort=False))
    if count != expected:
        raise RuntimeError(f"窗口覆盖检查失败: 实际{count}，预期{expected}，窗口长度{steps}")


def _fit_scaler(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """功能: 只用训练行程拟合输入和功率目标标准化参数。
    参数: train为训练数据表。
    返回: 输入均值、输入标准差、目标均值和目标标准差。
    调用位置: search_tcn、train_model。
    """
    mean = train[FEATURES].mean().to_numpy(float); std = train[FEATURES].std().replace(0, 1).fillna(1).to_numpy(float)
    target_mean = float(train[TARGET].mean()); target_std = max(float(train[TARGET].std()), 1.0)
    config.SCALER_FILE.write_text(json.dumps({"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "target_mean": target_mean, "target_std": target_std}, ensure_ascii=False, indent=2), encoding="utf-8")
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
    model = TCN(len(FEATURES), params["channels"], params["kernel_size"], params["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=config.BATCH_SIZE, shuffle=True)
    best, state, wait, logs = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = nn.functional.smooth_l1_loss(model(xb.to(device)), yb.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); total += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad(): val_loss = float(nn.functional.smooth_l1_loss(model(torch.from_numpy(x_val).to(device)), torch.from_numpy(y_val).to(device)).item())
        logs.append({"epoch": epoch, "train_loss": total / len(x_train), "val_loss": val_loss, "seed": seed})
        if val_loss < best: best, state, wait = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else: wait += 1
        if wait >= patience: break
    return {"state_dict": state, "best_val_loss": best, "params": params, "seed": seed}, logs


def _predict(model: TCN, x: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    model = model.to(_device()); model.eval()
    with torch.no_grad(): return model(torch.from_numpy(x).to(_device())).cpu().numpy() * target_std + target_mean


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
    train = pd.read_csv(config.TRAIN_CSV); val = pd.read_csv(config.VAL_CSV); mean, std, tm, ts = _fit_scaler(train)
    rows, trial_rows = [], []
    config.SEARCH_RESULTS_CSV.unlink(missing_ok=True); config.SEARCH_TRIALS_CSV.unlink(missing_ok=True)
    for steps in config.WINDOW_CANDIDATES:
        xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps); xv, yv, _, orig = _windows(val, mean, std, tm, ts, steps)
        _validate_window_coverage(train, steps, len(xtr)); _validate_window_coverage(val, steps, len(xv))
        val_frame = pd.concat([part.iloc[steps:] for _, part in val.groupby("trip_id", sort=False)], ignore_index=True)
        for params in config.MODEL_CANDIDATES:
            seed_metrics = []
            for seed in config.SEARCH_SEEDS:
                ckpt, log = _fit_model(xtr, ytr, xv, yv, params, seed, config.SEARCH_EPOCHS, config.SEARCH_PATIENCE)
                model = TCN(len(FEATURES), params["channels"], params["kernel_size"], params["dropout"]); model.load_state_dict(ckpt["state_dict"]); pred = _predict(model, xv, tm, ts); metrics = _selection(val_frame, orig, pred)
                trial = {"window_steps": steps, **params, "seed": seed, "best_val_loss": ckpt["best_val_loss"], **metrics, "epochs_run": len(log)}
                trial_rows.append(trial)
                pd.DataFrame(trial_rows).to_csv(config.SEARCH_TRIALS_CSV, index=False, encoding="utf-8")
                seed_metrics.append(metrics)
            averaged = {key: float(np.mean([m[key] for m in seed_metrics])) for key in seed_metrics[0]}
            row = {"window_steps": steps, **params, **averaged, "seed_count": len(seed_metrics), "search_stage": "筛选"}
            rows.append(row)
            pd.DataFrame(rows).sort_values("selection_score").to_csv(config.SEARCH_RESULTS_CSV, index=False, encoding="utf-8")
    shortlist = sorted(rows, key=lambda item: item["selection_score"])[:config.REFINE_TOP_CANDIDATES]
    refined_rows = []
    for candidate in shortlist:
        params = {key: candidate[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}
        steps = int(candidate["window_steps"]); xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps); xv, yv, _, orig = _windows(val, mean, std, tm, ts, steps)
        _validate_window_coverage(train, steps, len(xtr)); _validate_window_coverage(val, steps, len(xv))
        val_frame = pd.concat([part.iloc[steps:] for _, part in val.groupby("trip_id", sort=False)], ignore_index=True); metrics_by_seed = []
        for seed in config.REFINE_SEEDS:
            ckpt, log = _fit_model(xtr, ytr, xv, yv, params, seed, config.REFINE_EPOCHS, config.SEARCH_PATIENCE + 5)
            model = TCN(len(FEATURES), params["channels"], params["kernel_size"], params["dropout"]); model.load_state_dict(ckpt["state_dict"]); predicted = _predict(model, xv, tm, ts); metric = _selection(val_frame, orig, predicted); metrics_by_seed.append(metric)
            trial = {"window_steps": steps, **params, "seed": seed, "best_val_loss": ckpt["best_val_loss"], **metric, "epochs_run": len(log), "stage": "前三名复核"}; trial_rows.append(trial); pd.DataFrame(trial_rows).to_csv(config.SEARCH_TRIALS_CSV, index=False, encoding="utf-8")
        averaged = {key: float(np.mean([m[key] for m in metrics_by_seed])) for key in metrics_by_seed[0]}
        refined_rows.append({"window_steps": steps, **params, **averaged, "seed_count": len(config.REFINE_SEEDS), "search_stage": "三种子复核"})
    result = pd.DataFrame(refined_rows).sort_values("selection_score").reset_index(drop=True)
    result.to_csv(config.SEARCH_RESULTS_CSV, index=False, encoding="utf-8")
    best = result.iloc[0].to_dict(); config.SELECTED_CONFIG_JSON.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"candidate_count": len(result), "trial_count": len(trial_rows), "selected": best, "search_file": str(config.SEARCH_RESULTS_CSV)}


def train_model() -> dict:
    """功能: 使用验证集最优配置进行多种子集成训练并生成测试功率预测。
    参数: 无，读取搜索结果和数据表。
    返回: 训练和预测摘要。
    调用位置: main的train或all模式。
    """
    train = pd.read_csv(config.TRAIN_CSV); val = pd.read_csv(config.VAL_CSV); test = pd.read_csv(config.TEST_CSV); mean, std, tm, ts = _fit_scaler(train)
    selected = json.loads(config.SELECTED_CONFIG_JSON.read_text(encoding="utf-8")); steps = int(selected["window_steps"]); params = {key: selected[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}; params["channels"] = int(params["channels"]); params["kernel_size"] = int(params["kernel_size"])
    xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps); xv, yv, _, _ = _windows(val, mean, std, tm, ts, steps); xt, _, _, _ = _windows(test, mean, std, tm, ts, steps)
    _validate_window_coverage(train, steps, len(xtr)); _validate_window_coverage(val, steps, len(xv)); _validate_window_coverage(test, steps, len(xt))
    logs = []; checkpoints = []
    for seed in config.REFINE_SEEDS:
        ckpt, log = _fit_model(xtr, ytr, xv, yv, params, seed, config.EPOCHS, config.SEARCH_PATIENCE + 3); checkpoints.append(ckpt["state_dict"]); logs.extend([{**row, "seed": seed, "phase": "final"} for row in log])
    config.TRAIN_LOG_CSV.write_text(pd.DataFrame(logs).to_csv(index=False), encoding="utf-8")
    torch.save({"features": FEATURES, "window_steps": steps, "params": params, "target_mean": tm, "target_std": ts, "mean": mean.tolist(), "std": std.tolist(), "seed_count": len(checkpoints), "state_dicts": checkpoints}, config.MODEL_FILE)
    prediction_summary = predict_test()
    return {"device": str(_device()), "window_steps": steps, "selected_params": params, "seed_count": len(checkpoints), "seeds": list(config.REFINE_SEEDS), "test_rows": prediction_summary["prediction_rows"], "model": str(config.MODEL_FILE)}


def predict_test() -> dict:
    """功能: 从已保存的三种子模型载入权重并重新生成冻结测试集预测。
    参数: 无，读取模型文件和测试数据。
    返回: 测试行程数、逐秒样本数及预测表路径。
    调用位置: main的predict模式。
    """
    checkpoint = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    if checkpoint.get("features") != FEATURES:
        raise ValueError("模型输入字段与当前代码不一致，请重新训练模型")
    test = pd.read_csv(config.TEST_CSV); steps = int(checkpoint["window_steps"]); params = checkpoint["params"]
    x, _, _, actual = _windows(test, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), checkpoint["target_mean"], checkpoint["target_std"], steps)
    _validate_window_coverage(test, steps, len(x)); preds = []
    for state in checkpoint["state_dicts"]:
        model = TCN(len(FEATURES), int(params["channels"]), int(params["kernel_size"]), float(params["dropout"])); model.load_state_dict(state); preds.append(_predict(model, x, checkpoint["target_mean"], checkpoint["target_std"]))
    predicted = np.mean(preds, axis=0)
    input_rows = pd.concat([part.iloc[steps - 1:-1] for _, part in test.groupby("trip_id", sort=False)], ignore_index=True)
    output = pd.concat([part.iloc[steps:] for _, part in test.groupby("trip_id", sort=False)], ignore_index=True)
    output["actual_speed_kmh"] = output["speed_kmh"].to_numpy()
    output[FEATURES] = input_rows[FEATURES].to_numpy(); output["input_time_s"] = input_rows.time_s.to_numpy()
    output["predicted_power_w"] = predicted; output["actual_power_w"] = actual
    output["predicted_energy_wh"] = predicted * config.SAMPLE_SECONDS / 3600; output["actual_energy_wh"] = actual * config.SAMPLE_SECONDS / 3600
    output.to_csv(config.PREDICTIONS_CSV, index=False, encoding="utf-8")
    return {"test_trips": int(output.trip_id.nunique()), "prediction_rows": len(output), "predictions_file": str(config.PREDICTIONS_CSV)}


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


def search_rls() -> dict:
    """功能: 在验证集按完整行程搜索RLS窗口和递推参数。
    参数: 无。
    返回: 最优RLS组合和搜索表路径。
    调用位置: main的search或all模式。
    """
    val = pd.read_csv(config.VAL_CSV); selected = json.loads(config.SELECTED_CONFIG_JSON.read_text(encoding="utf-8")); steps = int(selected["window_steps"]); train = pd.read_csv(config.TRAIN_CSV); mean, std, tm, ts = _fit_scaler(train); xv, yv, _, orig = _windows(val, mean, std, tm, ts, steps)
    _validate_window_coverage(val, steps, len(xv))
    # 用最终选定配置在验证集重新拟合一个基线，搜索只读取其预测，不接触测试集。
    params = {key: selected[key] for key in ("channels", "kernel_size", "dropout", "learning_rate", "weight_decay")}; params["channels"] = int(params["channels"]); params["kernel_size"] = int(params["kernel_size"]); xtr, ytr, _, _ = _windows(train, mean, std, tm, ts, steps); ckpt, _ = _fit_model(xtr, ytr, xv, yv, params, config.RANDOM_SEED, config.EPOCHS, config.SEARCH_PATIENCE + 3); model = TCN(len(FEATURES), params["channels"], params["kernel_size"], float(params["dropout"])); model.load_state_dict(ckpt["state_dict"]); base = _predict(model, xv, tm, ts) * config.SAMPLE_SECONDS / 3600
    val_frame = pd.concat([part.iloc[steps:] for _, part in val.groupby("trip_id", sort=False)], ignore_index=True); base_frame = val_frame.copy(); base_frame["actual_energy"] = orig * config.SAMPLE_SECONDS / 3600; base_frame["baseline_energy"] = base
    rows = []
    for seconds in config.RLS_WINDOWS_SECONDS:
        for forgetting in config.RLS_FORGETTING_FACTORS:
            for covariance in config.RLS_INITIAL_COVARIANCES:
                all_actual, all_base, corrected = [], [], []; actual_trip, base_trip, corrected_trip = [], [], []
                for _, part in base_frame.groupby("trip_id", sort=False):
                    a = part.actual_energy.to_numpy(); b = part.baseline_energy.to_numpy(); grouped = pd.DataFrame({"a": a, "b": b}).groupby(np.arange(len(a)) // seconds, sort=True).sum(); aw = grouped.a.to_numpy(); bw = grouped.b.to_numpy(); r = _rls_predict(aw, bw, 1, forgetting, covariance); all_actual.append(aw); all_base.append(bw); corrected.append(r); actual_trip.append(float(aw.sum())); base_trip.append(float(bw.sum())); corrected_trip.append(float(r.sum()))
                a = np.concatenate(all_actual); b = np.concatenate(all_base); r = np.concatenate(corrected); trip_wape = _wape(np.asarray(actual_trip), np.asarray(corrected_trip)); rows.append({"window_seconds": seconds, "forgetting_factor": forgetting, "initial_covariance": covariance, "tcn_window_wape_percent": _wape(a, b), "rls_window_wape_percent": _wape(a, r), "tcn_trip_wape_percent": _wape(np.asarray(actual_trip), np.asarray(base_trip)), "rls_trip_wape_percent": trip_wape, "selection_score": trip_wape})
    result = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True); result.to_csv(config.RLS_SEARCH_CSV, index=False, encoding="utf-8"); best = result.iloc[0].to_dict(); selected.update({"rls_window_seconds": best["window_seconds"], "rls_forgetting_factor": best["forgetting_factor"], "rls_initial_covariance": best["initial_covariance"]}); config.SELECTED_CONFIG_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"); return {"candidate_count": len(result), "selected": best, "search_file": str(config.RLS_SEARCH_CSV)}


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
    trips = windows.groupby("trip_id", as_index=False)[["actual_energy_wh", "predicted_energy_wh", "rls_energy_wh"]].sum(); pred["distance_km"] = np.maximum(pred.actual_speed_kmh, 0) * config.SAMPLE_SECONDS / 3600; distance = pred.groupby("trip_id", as_index=False).distance_km.sum(); trips = trips.merge(distance, on="trip_id"); trips["actual_wh_per_km"] = trips.actual_energy_wh / trips.distance_km.clip(lower=.01); trips["tcn_wh_per_km"] = trips.predicted_energy_wh / trips.distance_km.clip(lower=.01); trips["rls_wh_per_km"] = trips.rls_energy_wh / trips.distance_km.clip(lower=.01); trips.to_csv(config.TRIP_METRICS_CSV, index=False, encoding="utf-8")
    def metric(a, b):
        e = a - b; return {"mae_wh": float(np.mean(np.abs(e))), "rmse_wh": float(np.sqrt(np.mean(e ** 2))), "mape_percent": float(np.mean(np.abs(e) / np.maximum(np.abs(a), 1e-3)) * 100), "wape_percent": _wape(a, b)}
    power_wape = _wape(pred.actual_power_w.to_numpy(), pred.predicted_power_w.to_numpy())
    windows["tcn_abs_error"] = np.abs(windows.actual_energy_wh - windows.predicted_energy_wh)
    windows["rls_abs_error"] = np.abs(windows.actual_energy_wh - windows.rls_energy_wh)
    metrics = {"tcn": metric(trips.actual_energy_wh.to_numpy(), trips.predicted_energy_wh.to_numpy()), "tcn_rls": metric(trips.actual_energy_wh.to_numpy(), trips.rls_energy_wh.to_numpy()), "trip_count": len(trips), "window_count": len(windows), "power_wape_percent": power_wape, "window_energy_wape_percent": _wape(windows.actual_energy_wh.to_numpy(), windows.predicted_energy_wh.to_numpy()), "rls_window_energy_wape_percent": _wape(windows.actual_energy_wh.to_numpy(), windows.rls_energy_wh.to_numpy()), "selected_config": selected, "wh_per_km": {"actual_mean": float(trips.actual_wh_per_km.mean()), "tcn_mean": float(trips.tcn_wh_per_km.mean()), "rls_mean": float(trips.rls_wh_per_km.mean())}}
    config.METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot = pred[pred.trip_id == pred.trip_id.iloc[0]].copy(); plot["actual_cum"] = plot.actual_energy_wh.cumsum(); plot["pred_cum"] = plot.predicted_energy_wh.cumsum(); _plot_all(pred, plot, trips)
    return metrics


def _plot_all(pred: pd.DataFrame, sample: pd.DataFrame, trips: pd.DataFrame) -> None:
    """功能: 生成损失、搜索、散点、残差、行程指标和代表行程图。
    参数: pred为测试逐秒预测，sample为代表行程，trips为逐行程汇总。
    返回: None。
    调用位置: evaluate_and_plot。
    """
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if config.TRAIN_LOG_CSV.exists():
        log = pd.read_csv(config.TRAIN_LOG_CSV).groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean(); plt.figure(figsize=(8, 4)); plt.plot(log.epoch, log.train_loss, label="三种子平均训练损失"); plt.plot(log.epoch, log.val_loss, label="三种子平均验证损失"); plt.xlabel("轮次"); plt.ylabel("标准化 SmoothL1"); plt.legend(); plt.tight_layout(); plt.savefig(config.FIGURES["loss"], dpi=160); plt.close()
    if config.SEARCH_RESULTS_CSV.exists():
        search = pd.read_csv(config.SEARCH_RESULTS_CSV).sort_values("selection_score"); plt.figure(figsize=(9, 4)); plt.bar(np.arange(len(search)), search.selection_score); plt.xlabel("候选配置（按综合分数排序）"); plt.ylabel("验证综合分数"); plt.tight_layout(); plt.savefig(config.FIGURES["search"], dpi=160); plt.close()
    if config.RLS_SEARCH_CSV.exists():
        search = pd.read_csv(config.RLS_SEARCH_CSV).sort_values("selection_score").reset_index(drop=True); plt.figure(figsize=(8, 4)); plt.bar(np.arange(len(search)), search.selection_score); plt.xlabel("RLS候选组合（按验证行程误差排序）"); plt.ylabel("验证行程能耗 WAPE (%)"); plt.tight_layout(); plt.savefig(config.FIGURES["rls_search"], dpi=160); plt.close()
    plt.figure(figsize=(6, 5)); plt.scatter(pred.actual_power_w, pred.predicted_power_w, s=2, alpha=.2); lo, hi = pred.actual_power_w.min(), pred.actual_power_w.max(); plt.plot([lo, hi], [lo, hi], "r--"); plt.xlabel("实测功率 (W)"); plt.ylabel("预测功率 (W)"); plt.tight_layout(); plt.savefig(config.FIGURES["power_scatter"], dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); plt.hist(pred.predicted_power_w - pred.actual_power_w, bins=80); plt.xlabel("预测误差 (W)"); plt.ylabel("样本数"); plt.tight_layout(); plt.savefig(config.FIGURES["residual"], dpi=160); plt.close()
    plt.figure(figsize=(9, 4)); x = np.arange(len(trips)); plt.plot(x, trips.actual_energy_wh, "o-", label="实测"); plt.plot(x, trips.predicted_energy_wh, "o-", label="TCN"); plt.plot(x, trips.rls_energy_wh, "o-", label="TCN+RLS"); plt.xlabel("测试行程编号"); plt.ylabel("净能耗 (Wh)"); plt.legend(); plt.tight_layout(); plt.savefig(config.FIGURES["trip_energy"], dpi=160); plt.close()
    plt.figure(figsize=(10, 4.5)); plt.plot(sample.time_s, sample.actual_cum, label="实测累计能耗"); plt.plot(sample.time_s, sample.pred_cum, label="TCN累计预测"); plt.xlabel("时间 (s)"); plt.ylabel("电池侧净能耗 (Wh)"); plt.title(f"GECR真实道路行程 {sample.trip_id.iloc[0]} 能耗预测"); plt.legend(); plt.tight_layout(); plt.savefig(config.FIGURES["trip_trace"], dpi=160); plt.close()
