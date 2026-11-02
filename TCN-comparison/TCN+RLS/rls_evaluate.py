# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: rls_evaluate.py
# 开发时间: 2026-10-30
# 文件名: rls_evaluate.py
# 功能说明: 对五组既有TCN分别搜索RLS参数、校准区间并评价测试覆盖率
# 版本号：1.0

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import GROUPS, RAW_FEATURE_COLUMNS, SCHEMA_VERSION, ExperimentConfig, ensure_directories, group_paths, table_path
from data_utils import feature_columns, load_tables, prepare_features, validate_features
from rls import SimulationOutput, calibrate_scales, coverage, interval_score, simulate_rls
from train import _model_from_checkpoint, build_sequences, predict_array


def prepare_data(cfg: ExperimentConfig, force: bool = False) -> dict:
    """功能: 校验共享加工表的源指纹和flight划分。
    参数: cfg为实验配置，force为独立路径时是否重建副本。
    返回: 数据行数、flight数和SHA-256摘要。
    调用位置: main.py。
    """
    ensure_directories(cfg)
    names = ["train_TCN-comparison.csv", "val_TCN-comparison.csv",
             "test_TCN-comparison.csv", "feature_metadata_TCN-comparison.json"]
    for name in names:
        source = cfg.source_data_dir / name
        target = cfg.data_dir / name
        if not source.exists():
            raise FileNotFoundError(f"缺少TCN数据副本: {source}")
        if source.resolve() != target.resolve() and (force or not target.exists()):
            shutil.copy2(source, cfg.data_dir / name)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise ValueError(f"数据副本与来源不一致: {name}，请检查或显式使用--force-prepare重建")
    missing = [name for name in names if not (cfg.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"TCN+RLS数据目录缺少文件: {missing}")
    checksums = {name: hashlib.sha256((cfg.data_dir / name).read_bytes()).hexdigest() for name in names}
    canonical_path = cfg.source_manifest_json
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if canonical.get("schema_version", canonical.get("version")) != SCHEMA_VERSION:
        raise ValueError("共享数据的版本契约不匹配")
    for name, actual in checksums.items():
        if canonical["sha256"].get(name) != actual:
            raise ValueError(f"DJI planned共享加工表与预处理指纹不一致: {name}")
    train, val, test = (pd.read_csv(table_path(cfg, split), usecols=["flight"])
                        for split in ("train", "val", "test"))
    sets = [set(train.flight), set(val.flight), set(test.flight)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("TCN+RLS的train、val、test存在flight交叉")
    summary = {
        "data_dir": str(cfg.data_dir), "source_data_dir": str(cfg.source_data_dir),
        "sha256": checksums, "split_overlap": False, "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
        "train_flights": int(train.flight.nunique()), "val_flights": int(val.flight.nunique()),
        "test_flights": int(test.flight.nunique()),
        "split_flights": {name: sorted(int(x) for x in values)
                          for name, values in zip(("train", "val", "test"), sets)},
    }
    cfg.manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _copy_tcn_assets(cfg: ExperimentConfig, group: str) -> tuple[Path, Path]:
    """功能: 将TCN权重和标准化器复制为TCN+RLS统一命名产物。
    参数: cfg为实验配置，group为A、B、C、D或E。
    返回: 本实验目录中的checkpoint和scaler路径。
    调用位置: load_baseline。
    """
    paths = group_paths(cfg, group)
    source_checkpoint = paths["tcn_source_checkpoint"]
    source_scaler = cfg.tcn_source_model_dir / f"TCN-comparison_{group}_scaler.json"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"找不到TCN权重: {source_checkpoint}")
    if not source_scaler.exists():
        raise FileNotFoundError(f"找不到TCN标准化器: {source_scaler}")
    if not paths["tcn_checkpoint"].exists() or hashlib.sha256(source_checkpoint.read_bytes()).digest() != hashlib.sha256(paths["tcn_checkpoint"].read_bytes()).digest():
        shutil.copy2(source_checkpoint, paths["tcn_checkpoint"])
    if not paths["scaler"].exists() or hashlib.sha256(source_scaler.read_bytes()).digest() != hashlib.sha256(paths["scaler"].read_bytes()).digest():
        shutil.copy2(source_scaler, paths["scaler"])
    return paths["tcn_checkpoint"], paths["scaler"]


def load_baselines(cfg: ExperimentConfig, group: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    """功能: 读取单组TCN权重并生成验证、测试功率基线。
    参数: cfg为实验配置，group为A、B、C、D或E。
    返回: 验证表、测试表、验证功率、测试功率、checkpoint和scaler。
    调用位置: tune_rls、evaluate_group。
    """
    checkpoint_path, scaler_path = _copy_tcn_assets(cfg, group)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    expected = feature_columns(GROUPS[group]["use_sma"])
    if checkpoint.get("group") != group or checkpoint.get("feature_columns") != expected:
        raise ValueError(f"{group}组TCN checkpoint特征契约不匹配")
    if checkpoint.get("activation") != GROUPS[group]["activation"]:
        raise ValueError(f"{group}组TCN checkpoint激活契约不匹配")
    if scaler.get("group") != group or scaler.get("feature_columns") != expected:
        raise ValueError(f"{group}组TCN scaler特征契约不匹配")
    if checkpoint.get("channels") != [32, 32] or checkpoint.get("kernel_size") != 3 or checkpoint.get("dropout") != 0.05:
        raise ValueError("冻结TCN结构与参考架构不一致")
    if checkpoint.get("weight_decay") != 1e-4 or checkpoint.get("batch_size") != 2048 or checkpoint.get("epochs_run") != 80:
        raise ValueError("冻结TCN训练预算与本实验不一致")
    fingerprint = hashlib.sha256(cfg.source_manifest_json.read_bytes()).hexdigest()
    for asset in (checkpoint, scaler):
        if asset.get("schema_version") != SCHEMA_VERSION or asset.get("data_manifest_sha256") != fingerprint:
            raise ValueError("冻结TCN或scaler不属于当前数据清单")
        if asset.get("sma_window_steps") != cfg.sma_window_steps or asset.get("resample_seconds") != cfg.resample_seconds:
            raise ValueError("冻结TCN的采样与SMA配置不一致")
    _, validation, test = load_tables(cfg)
    validation = prepare_features(validation, GROUPS[group]["use_sma"], cfg.sma_window_steps)
    test = prepare_features(test, GROUPS[group]["use_sma"], cfg.sma_window_steps)
    validate_features(validation, expected)
    validate_features(test, expected)
    model = _model_from_checkpoint(checkpoint, len(expected))
    val_seq, _ = build_sequences(validation, scaler, int(checkpoint["window_steps"]), require_target=False)
    test_seq, _ = build_sequences(test, scaler, int(checkpoint["window_steps"]), require_target=False)
    return validation, test, predict_array(model, val_seq, scaler, cfg), predict_array(model, test_seq, scaler, cfg), checkpoint, scaler


def _flight_wape(flights: pd.DataFrame, actual: str = "actual_energy_wh", predicted: str = "predicted_energy_wh") -> float:
    """功能: 计算完整flight总能耗WAPE。
    参数: flights为逐flight表，actual和predicted为列名。
    返回: 百分比WAPE。
    调用位置: tune_rls、summarise_output。
    """
    return float(np.abs(flights[actual] - flights[predicted]).sum() /
                 max(np.abs(flights[actual]).sum(), 1e-9) * 100.0)


def _sample_metrics(predictions: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """功能: 计算功率和固定1秒窗口能耗的点指标。
    参数: predictions为RLS逐采样预测表，cfg为采样配置。
    返回: 功率与1秒能耗指标。
    调用位置: summarise_output。
    """
    actual = predictions.power_w.to_numpy(float)
    pred = predictions.predicted_power_w.to_numpy(float)
    tcn = predictions.tcn_predicted_power_w.to_numpy(float)
    row = predictions[["flight", "actual_energy_wh", "predicted_energy_wh", "tcn_predicted_power_w"]].copy()
    row["tcn_energy_wh"] = tcn * predictions.dt_seconds.to_numpy(float) / 3600.0
    row["window"] = row.groupby("flight", sort=False).cumcount() // max(1, int(round(1.0 / cfg.resample_seconds)))
    one = row.groupby(["flight", "window"], sort=False).agg(
        actual_energy_wh=("actual_energy_wh", "sum"), predicted_energy_wh=("predicted_energy_wh", "sum"),
        tcn_energy_wh=("tcn_energy_wh", "sum"),
    ).reset_index()
    wape = lambda a, p: float(np.abs(a - p).sum() / max(np.abs(a).sum(), 1e-9) * 100.0)
    return {
        "power_mae_w": float(np.mean(np.abs(actual - pred))),
        "power_rmse_w": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "power_wape_percent": wape(actual, pred),
        "tcn_power_wape_percent": wape(actual, tcn),
        "window_energy_mae_wh": float(np.mean(np.abs(one.actual_energy_wh - one.predicted_energy_wh))),
        "window_energy_wape_percent": wape(one.actual_energy_wh.to_numpy(), one.predicted_energy_wh.to_numpy()),
        "tcn_window_energy_wape_percent": wape(one.actual_energy_wh.to_numpy(), one.tcn_energy_wh.to_numpy()),
        "sample_rows": int(len(predictions)), "one_second_windows": int(len(one)),
    }


def summarise_output(output: SimulationOutput, cfg: ExperimentConfig, group: str,
                     checkpoint: dict, best: dict, calibration: dict) -> dict:
    """功能: 汇总RLS点预测、能耗误差、区间覆盖率和区间评分。
    参数: output为测试集模拟结果，cfg为配置，group为实验组，checkpoint为TCN元数据，best为RLS最优参数，calibration为区间倍率。
    返回: 可写入汇总CSV和JSON的指标字典。
    调用位置: evaluate_group。
    """
    flight = output.flights
    window = output.windows
    predictions = output.predictions
    active_trace = output.trace[output.trace.remaining_window_count > 0]
    sample = _sample_metrics(predictions, cfg)
    flight_wape = _flight_wape(flight)
    tcn_flight_wape = _flight_wape(flight, "actual_energy_wh", "tcn_energy_wh")
    window_wape = float(np.abs(window.actual_energy_wh - window.predicted_energy_wh).sum() /
                       max(np.abs(window.actual_energy_wh).sum(), 1e-9) * 100.0)
    metrics = {
        "group": group, "name": GROUPS[group]["name"], "activation": GROUPS[group]["activation"],
        "use_sma": GROUPS[group]["use_sma"], "feature_count": int(checkpoint["feature_count"]),
        "tcn_window_seconds": float(checkpoint["window_seconds"]), "tcn_window_steps": int(checkpoint["window_steps"]),
        "rls_forgetting_factor": float(best["forgetting_factor"]),
        "rls_initial_covariance": float(best["initial_covariance"]),
        "rls_window_seconds": float(best["window_seconds"]),
        "rls_window_steps": int(round(best["window_seconds"] / cfg.resample_seconds)),
        "confidence_level": cfg.confidence_level,
        "tcn_flight_energy_wape_percent": tcn_flight_wape,
        "flight_energy_wape_percent": flight_wape,
        "flight_energy_mae_wh": float(np.mean(np.abs(flight.actual_energy_wh - flight.predicted_energy_wh))),
        "window_energy_wape_percent": window_wape,
        "tcn_window_energy_wape_percent": sample["tcn_window_energy_wape_percent"],
        "power_wape_percent": sample["power_wape_percent"],
        "tcn_power_wape_percent": sample["tcn_power_wape_percent"],
        "flight_interval_coverage_percent": coverage(flight.actual_energy_wh, flight.predicted_lower_wh, flight.predicted_upper_wh),
        "flight_interval_mean_width_wh": float(np.mean(flight.predicted_upper_wh - flight.predicted_lower_wh)),
        "flight_interval_score_wh": interval_score(flight.actual_energy_wh.to_numpy(), flight.predicted_lower_wh.to_numpy(), flight.predicted_upper_wh.to_numpy(), cfg.confidence_level),
        "window_interval_coverage_percent": coverage(window.actual_energy_wh, window.predicted_lower_wh, window.predicted_upper_wh),
        "window_interval_mean_width_wh": float(np.mean(window.predicted_upper_wh - window.predicted_lower_wh)),
        "window_interval_score_wh": interval_score(window.actual_energy_wh.to_numpy(), window.predicted_lower_wh.to_numpy(), window.predicted_upper_wh.to_numpy(), cfg.confidence_level),
        "sample_power_interval_coverage_percent": coverage(predictions.power_w, predictions.predicted_power_lower_w, predictions.predicted_power_upper_w),
        "sample_power_interval_mean_width_w": float(np.mean(predictions.predicted_power_upper_w - predictions.predicted_power_lower_w)),
        "remaining_trace_interval_coverage_percent": coverage(active_trace.actual_final_energy_wh, active_trace.predicted_final_lower_wh, active_trace.predicted_final_upper_wh),
        "remaining_trace_interval_mean_width_wh": float(np.mean(active_trace.predicted_final_upper_wh - active_trace.predicted_final_lower_wh)),
        "remaining_trace_points": int(len(active_trace)),
        "task_interval_scale": float(calibration["task_interval_scale"]),
        "window_interval_scale": float(calibration["window_interval_scale"]),
        "test_flights": int(len(flight)), "test_rows": int(len(predictions)),
        **sample,
    }
    metrics["rls_window_energy_wape_percent"] = window_wape
    metrics["task_start_interval_coverage_percent"] = metrics["flight_interval_coverage_percent"]
    metrics["task_start_interval_mean_width_wh"] = metrics["flight_interval_mean_width_wh"]
    metrics["task_start_interval_score_wh"] = metrics["flight_interval_score_wh"]
    online = active_trace[active_trace.observed_window_count > 0]
    metrics["post_feedback_interval_coverage_percent"] = coverage(online.actual_final_energy_wh, online.predicted_final_lower_wh, online.predicted_final_upper_wh)
    metrics["post_feedback_interval_mean_width_wh"] = float(np.mean(online.predicted_final_upper_wh - online.predicted_final_lower_wh))
    metrics["post_feedback_interval_score_wh"] = interval_score(online.actual_final_energy_wh.to_numpy(), online.predicted_final_lower_wh.to_numpy(), online.predicted_final_upper_wh.to_numpy(), cfg.confidence_level)
    metrics["post_feedback_points"] = int(len(online))
    return metrics


def export_interval_stages(output: SimulationOutput, cfg: ExperimentConfig, group: str) -> pd.DataFrame:
    """功能: 按任务开始及已观察25%、50%、75%阶段评价任务总能耗区间。
    参数: output为测试模拟结果，cfg为配置，group为实验组。
    返回: 每个阶段按flight等权计算的区间指标。
    调用位置: evaluate_group。
    """
    rows = []
    for fraction in (0.0, 0.25, 0.5, 0.75):
        selected = []
        for _, trace in output.trace.groupby("flight", sort=False):
            total = float(trace.iloc[0].remaining_window_count)
            progress = 1.0 - trace.remaining_window_count / total
            candidates = trace[(progress >= fraction) & (trace.remaining_window_count > 0)]
            if not candidates.empty:
                selected.append(candidates.iloc[0])
        frame = pd.DataFrame(selected)
        actual = frame.actual_final_energy_wh.to_numpy()
        predicted = frame.predicted_final_energy_wh.to_numpy()
        lower, upper = frame.predicted_final_lower_wh.to_numpy(), frame.predicted_final_upper_wh.to_numpy()
        rows.append({"group": group, "observed_fraction": fraction, "flights": len(frame),
                     "coverage_percent": coverage(actual, lower, upper),
                     "mean_width_wh": float(np.mean(upper - lower)),
                     "interval_score_wh": interval_score(actual, lower, upper, cfg.confidence_level),
                     "energy_wape_percent": float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1e-9) * 100)})
    result = pd.DataFrame(rows)
    result.to_csv(group_paths(cfg, group)["stage_intervals"], index=False, encoding="utf-8")
    return result


def tune_group(group: str, cfg: ExperimentConfig) -> dict:
    """功能: 在验证子集上搜索RLS遗忘因子、初始协方差和反馈时间窗。
    参数: group为实验组，cfg为实验配置。
    返回: 最优参数和调参文件路径。
    调用位置: main.py。
    """
    validation, _, val_baseline, _, checkpoint, _ = load_baselines(cfg, group)
    flights = sorted(validation.flight.unique())
    tune_ids = set(flights[::2])
    subset = validation[validation.flight.isin(tune_ids)].reset_index(drop=True)
    baseline = val_baseline[validation.flight.isin(tune_ids).to_numpy()]
    rows: list[dict] = []
    for window_seconds in cfg.rls_window_seconds:
        for forgetting_factor in cfg.rls_forgetting_factors:
            for initial_covariance in cfg.rls_initial_covariances:
                output = simulate_rls(subset, baseline, forgetting_factor, initial_covariance,
                                      window_seconds, cfg.resample_seconds, cfg.confidence_level)
                score = _flight_wape(output.flights)
                rows.append({
                    "group": group, "window_seconds": window_seconds, "window_steps": int(round(window_seconds / cfg.resample_seconds)),
                    "forgetting_factor": forgetting_factor, "initial_covariance": initial_covariance,
                    "flight_energy_wape_percent": score,
                    "flight_energy_mae_wh": float(np.mean(np.abs(output.flights.actual_energy_wh - output.flights.predicted_energy_wh))),
                    "flights": int(len(output.flights)), "selection_split": "validation_even_flights",
                })
        pd.DataFrame(rows).to_csv(group_paths(cfg, group)["rls_tuning"], index=False, encoding="utf-8")
        best_so_far = min(row["flight_energy_wape_percent"] for row in rows)
        print(f"组{group}已完成RLS窗口{window_seconds:g}s，累计{len(rows)}个候选，"
              f"当前最优验证任务能耗WAPE={best_so_far:.6f}%", flush=True)
    result = pd.DataFrame(rows).sort_values(["flight_energy_wape_percent", "window_seconds"]).reset_index(drop=True)
    result["data_manifest_sha256"] = checkpoint["data_manifest_sha256"]
    result["tcn_sha256"] = hashlib.sha256(group_paths(cfg, group)["tcn_checkpoint"].read_bytes()).hexdigest()
    result.to_csv(group_paths(cfg, group)["rls_tuning"], index=False, encoding="utf-8")
    best = result.iloc[0].to_dict()
    group_paths(cfg, group)["tuning_log"].write_text(
        "TCN+RLS超参搜索\n"
        f"group={group}\n候选数={len(result)}\n"
        f"best_window_seconds={best['window_seconds']}\n"
        f"best_forgetting_factor={best['forgetting_factor']}\n"
        f"best_initial_covariance={best['initial_covariance']}\n"
        f"validation_flight_energy_wape_percent={best['flight_energy_wape_percent']}\n",
        encoding="utf-8",
    )
    return {"group": group, **{key: best[key] for key in ("window_seconds", "forgetting_factor", "initial_covariance", "flight_energy_wape_percent")},
            "candidate_count": len(result), "tuning_file": str(group_paths(cfg, group)["rls_tuning"])}


def calibrate_group(group: str, cfg: ExperimentConfig, best: dict) -> dict:
    """功能: 用验证奇数flight残差校准任务和窗口区间倍率。
    参数: group为实验组，cfg为配置，best为RLS最优参数。
    返回: 区间校准摘要和校准结果路径。
    调用位置: main.py。
    """
    validation, _, val_baseline, _, checkpoint, _ = load_baselines(cfg, group)
    flights = sorted(validation.flight.unique())
    calibration_ids = set(flights[1::2])
    subset = validation[validation.flight.isin(calibration_ids)].reset_index(drop=True)
    baseline = val_baseline[validation.flight.isin(calibration_ids).to_numpy()]
    output = simulate_rls(subset, baseline, float(best["forgetting_factor"]), float(best["initial_covariance"]),
                          float(best["window_seconds"]), cfg.resample_seconds, cfg.confidence_level)
    calibration = calibrate_scales(output, cfg.confidence_level)
    calibration.update({"data_manifest_sha256": checkpoint["data_manifest_sha256"],
                        "tcn_sha256": hashlib.sha256(group_paths(cfg, group)["tcn_checkpoint"].read_bytes()).hexdigest(),
                        "group": group, "calibration_split": "validation_odd_flights",
                        "rls_forgetting_factor": best["forgetting_factor"],
                        "rls_initial_covariance": best["initial_covariance"],
                        "rls_window_seconds": best["window_seconds"],
                        "calibration_flights": int(len(calibration_ids)),
                        "source_trace_rows": int(len(output.trace))})
    paths = group_paths(cfg, group)
    paths["interval_calibration"].write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    output.trace.to_csv(paths["validation_trace"], index=False, encoding="utf-8")
    return calibration


def evaluate_group(group: str, cfg: ExperimentConfig) -> dict:
    """功能: 冻结验证集选择的RLS参数和区间倍率并评价测试集。
    参数: group为实验组，cfg为实验配置。
    返回: RLS和置信区间测试指标。
    调用位置: main.py。
    """
    tuning_path = group_paths(cfg, group)["rls_tuning"]
    if not tuning_path.exists():
        tune_group(group, cfg)
    tuning = pd.read_csv(tuning_path).sort_values("flight_energy_wape_percent").iloc[0].to_dict()
    calibration_path = group_paths(cfg, group)["interval_calibration"]
    if not calibration_path.exists():
        calibrate_group(group, cfg, tuning)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    for calibration_key, tuning_key in (("rls_forgetting_factor", "forgetting_factor"),
                                         ("rls_initial_covariance", "initial_covariance"),
                                         ("rls_window_seconds", "window_seconds")):
        if not np.isclose(float(calibration[calibration_key]), float(tuning[tuning_key])):
            raise ValueError(f"{group}组区间校准参数已过期，请重新运行calibrate")
    if not np.isclose(calibration["confidence_level"], cfg.confidence_level):
        raise ValueError("置信水平已改变，请重新运行calibrate")
    _, test, _, test_baseline, checkpoint, _ = load_baselines(cfg, group)
    fingerprint = checkpoint["data_manifest_sha256"]
    tcn_hash = hashlib.sha256(group_paths(cfg, group)["tcn_checkpoint"].read_bytes()).hexdigest()
    for asset in (tuning, calibration):
        if asset.get("data_manifest_sha256") != fingerprint or asset.get("tcn_sha256") != tcn_hash:
            raise ValueError("搜索或校准产物已过期，请重新tune和calibrate")
    output = simulate_rls(test, test_baseline, float(tuning["forgetting_factor"]), float(tuning["initial_covariance"]),
                          float(tuning["window_seconds"]), cfg.resample_seconds, cfg.confidence_level,
                          float(calibration["task_interval_scale"]), float(calibration["window_interval_scale"]))
    paths = group_paths(cfg, group)
    output.predictions.to_csv(paths["test_predictions"], index=False, encoding="utf-8")
    output.trace.to_csv(paths["test_trace"], index=False, encoding="utf-8")
    output.flights.to_csv(paths["flight_metrics"], index=False, encoding="utf-8")
    output.windows.to_csv(paths["window_metrics"], index=False, encoding="utf-8")
    export_interval_stages(output, cfg, group)
    recipe = {"group": group, "forgetting_factor": float(tuning["forgetting_factor"]),
              "initial_covariance": float(tuning["initial_covariance"]),
              "window_seconds": float(tuning["window_seconds"]), "calibration": calibration,
              "theta_initial": [0.0, 1.0], "reset_per_flight": True,
              "initial_residual_std_wh": 0.01,
              "tcn_checkpoint": str(paths["tcn_checkpoint"]), "scaler": str(paths["scaler"])}
    paths["rls_model"].write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_summary = {"group": group, "feature_count": len(feature_columns(GROUPS[group]["use_sma"])),
                          "window_seconds": checkpoint["window_seconds"], "window_steps": checkpoint["window_steps"]}
    metrics = summarise_output(output, cfg, group, checkpoint_summary, tuning, calibration)
    metrics.update({"schema_version": SCHEMA_VERSION, "data_manifest_sha256": fingerprint, "tcn_sha256": tcn_hash,
                    "tuning_candidates": int(len(pd.read_csv(tuning_path))), "validation_calibration_flights": int(calibration["calibration_flights"]),
                    "model_file": str(paths["tcn_checkpoint"]), "prediction_file": str(paths["test_predictions"]),
                    "trace_file": str(paths["test_trace"])})
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_all(cfg: ExperimentConfig, groups=None) -> pd.DataFrame:
    """功能: 依次评价五组TCN加RLS并写入汇总表。
    参数: cfg为实验配置，groups为待评价组别列表。
    返回: 五组指标表。
    调用位置: main.py。
    """
    selected = list(groups or GROUPS)
    rows = [evaluate_group(group, cfg) for group in selected]
    for group in GROUPS:
        path = group_paths(cfg, group)["metrics"]
        if group not in selected and path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    result = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    result.to_csv(cfg.summary_csv, index=False, encoding="utf-8")
    return result
