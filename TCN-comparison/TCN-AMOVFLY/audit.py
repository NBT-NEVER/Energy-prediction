# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: audit.py
# 开发时间: 2026-11-02
# 文件名: audit.py
# 功能说明: 独立核验planned源标签、能量、划分、训练预算和预测指标
# 版本号：1.0

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from config import GROUPS, RAW_FEATURE_COLUMNS, SCHEMA_VERSION, build_config, group_paths, table_path
from data_utils import copy_processed_data, prepare_features, feature_columns
from train import _metrics


def audit(data_only=False):
    """功能: 从源记录独立重建阶段并核对保存的实验产物。
    参数: data_only为是否只核验数据而不要求模型完成。
    返回: 可保存为JSON的核验记录。
    调用位置: 命令行及最终交付检查。
    """
    cfg = build_config()
    summary = copy_processed_data(cfg)
    tables = {split: pd.read_csv(table_path(cfg, split), encoding="utf-8") for split in ("train", "val", "test")}
    audits = pd.read_csv(cfg.data_dir / "amovfly_flight_audit.csv", encoding="utf-8")
    merged = pd.concat(tables.values(), ignore_index=True)
    assert len(audits) == summary["accepted_flights"]
    assert audits.groupby("simultaneous_group").split.nunique().max() == 1
    assert audits.groupby("simultaneous_group").split_role.nunique().max() == 1
    assert merged.groupby("flight").split.nunique().max() == 1
    assert set(merged.uav_type) == {1, 2, 3} and set(merged.planned) == {0, 1, 2}
    assert np.isfinite(merged.select_dtypes(include="number").to_numpy()).all()
    assert not merged.duplicated(["flight", "time"]).any()
    assert (merged.dt_seconds > 0).all() and (merged.dt_seconds <= cfg.resample_seconds + 1e-9).all()
    np.testing.assert_allclose(merged.energy_interval_wh, merged.power_w * merged.dt_seconds / 3600, atol=1e-12)
    for row in audits.itertuples(index=False):
        path = Path(row.source_file)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.source_sha256
        columns = ["time", "gps_x", "gps_y", "gps_z", "wind_speed", "wind_angle", "v_x", "v_y", "v_z", "battery_voltage", "battery_current"]
        raw = pd.read_csv(path, encoding="utf-8-sig")[columns].apply(pd.to_numeric, errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna().sort_values("time", kind="stable").drop_duplicates("time")
        raw_time = raw.time.to_numpy() - raw.time.iloc[0]
        power = np.maximum(raw.battery_voltage.to_numpy() * raw.battery_current.to_numpy(), 0)
        xyz = raw[["gps_x", "gps_y", "gps_z"]].to_numpy(float, copy=True)
        xyz -= xyz[0].copy()
        motor = np.zeros(len(raw), dtype=np.int8)
        flight = np.zeros(len(raw), dtype=np.int8)
        indices = np.flatnonzero(power >= 100.)
        if len(indices):
            motor[indices[0]:indices[-1]+1] = 1
        indices = np.flatnonzero((np.hypot(xyz[:, 0], xyz[:, 1]) >= 1.) | (np.abs(xyz[:, 2]) >= .5))
        if len(indices):
            flight[indices[0]:indices[-1]+1] = 1
        flight *= motor
        sampled = merged.loc[merged.flight.eq(row.flight)].sort_values("time")
        motor_grid = np.rint(np.interp(sampled.time_s, raw_time, motor)).astype(np.int8)
        flight_grid = np.minimum(np.rint(np.interp(sampled.time_s, raw_time, flight)).astype(np.int8), motor_grid)
        np.testing.assert_array_equal(sampled.planned_motor_on, motor_grid)
        np.testing.assert_array_equal(sampled.planned_airborne, flight_grid)
        np.testing.assert_array_equal(sampled.planned, motor_grid + flight_grid)
        assert sampled.uav_type.eq({"Y": 1, "R": 2, "G": 3}[path.stem[3]]).all()
        np.testing.assert_allclose(sampled.energy_interval_wh.sum(), np.trapezoid(power, raw_time) / 3600, atol=1e-10)
        np.testing.assert_allclose(sampled.dt_seconds.sum(), raw_time[-1], atol=1e-9)
    checks = ["all source and output SHA256 match", "complete-flight and simultaneous-group split/role isolation",
              f"{len(audits)} source records independently reconstructed with DJI 100W/1m/0.5m thresholds",
              "planned discrete merge and Y/R/G mapping; finite values; real duration and energy conservation"]
    models = {}
    if not data_only:
        fingerprint = hashlib.sha256(cfg.manifest_json.read_bytes()).hexdigest()
        metrics = pd.read_csv(cfg.summary_csv)
        assert set(metrics.group) == set(GROUPS)
        for group, spec in GROUPS.items():
            paths = group_paths(cfg, group)
            checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
            scaler = json.loads(paths["scaler"].read_text(encoding="utf-8"))
            columns = feature_columns(spec["use_sma"])
            for obj in (checkpoint, scaler):
                assert obj["schema_version"] == SCHEMA_VERSION
                assert obj["data_manifest_sha256"] == fingerprint
                assert obj["feature_columns"] == columns and obj["raw_feature_columns"] == RAW_FEATURE_COLUMNS
            assert checkpoint["channels"] == [32, 32] and checkpoint["dropout"] == .05
            assert checkpoint["weight_decay"] == 1e-4 and checkpoint["batch_size"] == 2048
            assert checkpoint["epochs_run"] == 80 and checkpoint["activation"] == spec["activation"]
            logs = pd.read_csv(paths["training_log"])
            assert logs.epoch.tolist() == list(range(1, 81))
            assert int(logs.loc[logs.val_loss.idxmin(), "epoch"]) == checkpoint["best_epoch"]
            tune = pd.read_csv(paths["tuning"])
            tune_logs = pd.read_csv(paths["tuning_log"])
            assert set(tune.window_seconds) == set(cfg.tcn_window_seconds)
            assert tune.epochs_run.eq(cfg.tune_epochs).all() and len(tune_logs) == 4 * cfg.tune_epochs
            train = prepare_features(tables["train"], spec["use_sma"], cfg.sma_window_steps)
            x, y = train[columns].to_numpy(float), train.power_w.to_numpy(float)
            std = x.std(axis=0)
            std[std < 1e-6] = 1.
            np.testing.assert_allclose(scaler["x_mean"], x.mean(axis=0))
            np.testing.assert_allclose(scaler["x_std"], std)
            np.testing.assert_allclose(scaler["y_mean"], y.mean())
            np.testing.assert_allclose(scaler["y_std"], y.std())
            predictions = pd.read_csv(paths["predictions"])
            np.testing.assert_array_equal(predictions.flight, tables["test"].flight)
            np.testing.assert_allclose(predictions.power_w, tables["test"].power_w)
            calculated = _metrics(tables["test"], predictions.predicted_power_w.to_numpy(), cfg)
            saved = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            aggregate = metrics.loc[metrics.group.eq(group)].iloc[0]
            for key, value in calculated.items():
                np.testing.assert_allclose(saved[key], value, atol=1e-8)
                np.testing.assert_allclose(aggregate[key], value, atol=1e-8)
            models[group] = {"checkpoint_sha256": hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest(),
                             "epochs": 80, "best_epoch": checkpoint["best_epoch"],
                             "window_seconds": checkpoint["window_seconds"],
                             "flight_wape_percent": calculated["flight_energy_wape_percent"]}
        checks += ["5 groups x 4 candidates x 8 epochs; each formal model complete 80 epochs",
                   "all checkpoints use requested 32/32 channels, .05 dropout, 1e-4 decay and 2048 batch",
                   "full-train-only scalers and best validation epoch; prediction CSV metrics independently recomputed"]
    record = {"status": "passed", "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "data_only": data_only, "checks": checks, "counts": summary["counts"],
              "state_counts": summary["state_counts"], "models": models}
    path = cfg.log_dir / ("planned_data_audit.json" if data_only else "planned_artifact_audit.json")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    print(json.dumps(audit(parser.parse_args().data_only), ensure_ascii=False, indent=2))
