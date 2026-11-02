# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: audit_amovfly.py
# 开发时间: 2026-11-01
# 文件名: audit_amovfly.py
# 功能说明: 核验AMOVFLY数据、冻结模型、训练预算、指标和文档链接
# 版本号：1.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch


def load_config(name):
    """功能: 隔离载入两组实验配置，避免同名模块冲突。
    参数: name为实验目录名。
    返回: 配置模块。
    调用位置: main。
    """
    path = Path(__file__).resolve().parent / name / "config.py"
    key = name.replace("+", "_").replace("-", "_") + "_audit_config"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    """功能: 计算文件的SHA-256。
    参数: path为文件路径。
    返回: 十六进制摘要。
    调用位置: main。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    """功能: 读取UTF-8结构化记录。
    参数: path为JSON路径。
    返回: 记录字典。
    调用位置: main。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def check_links(path):
    """功能: 核对本次文档中的本地文件和目录锚点。
    参数: path为Markdown文档路径。
    返回: 已核对链接数量。
    调用位置: main。
    """
    content = path.read_text(encoding="utf-8")
    count = 0
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", content):
        target = unquote(target.strip().strip("<>"))
        if target.startswith(("http:", "https:", "mailto:", "codex:")):
            continue
        file_part, _, anchor = target.partition("#")
        destination = path.parent / file_part if file_part else path
        assert destination.exists(), (path, target)
        if anchor and destination.suffix == ".md":
            headings = re.findall(r"^#{1,6}\s+(.+)$", destination.read_text(encoding="utf-8"), re.M)
            slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
            assert anchor in slugs, (path, target, slugs)
        count += 1
    return count


def main():
    """功能: 执行最终产物核验并写入两个实验的审计日志。
    参数: 无。
    返回: None，核验失败时以异常退出。
    调用位置: 脚本直接运行。
    """
    offline = load_config("TCN-AMOVFLY")
    online = load_config("TCN+RLS-AMOVFLY")
    cfg, rcfg = offline.build_config(), online.build_config()
    for config in (cfg, rcfg):
        (config.log_dir / "AMOVFLY_artifact_audit.json").write_text(
            json.dumps({"status": "running", "checked_at": datetime.now().astimezone().isoformat()}), encoding="utf-8")
    assert cfg.data_dir.resolve() == rcfg.data_dir.resolve()
    summary = read_json(cfg.data_dir / "dataset_summary_TCN-comparison.json")
    checks = []
    for manifest in (summary, read_json(rcfg.manifest_json)):
        for name, expected in manifest["sha256"].items():
            assert digest(cfg.data_dir / name) == expected, name
    checks.append("all preprocessing and RLS manifest SHA256 values match")
    tables = {s: pd.read_csv(offline.table_path(cfg, s)) for s in ("train", "val", "test")}
    audit = pd.read_csv(cfg.data_dir / "amovfly_flight_audit.csv")
    assert audit.groupby("simultaneous_group").split_role.nunique().max() == 1
    assert len(audit) == summary["accepted_flights"] == 256
    sets = [set(frame.flight) for frame in tables.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    for split, frame in tables.items():
        assert len(frame) == summary[f"{split}_rows"]
        assert frame.flight.nunique() == summary[f"{split}_flights"]
        assert not frame.duplicated(["flight", "time"]).any()
        assert np.isfinite(frame[[*offline.RAW_FEATURE_COLUMNS, "power_w", "dt_seconds"]]).all().all()
        assert (frame.dt_seconds > 0).all() and (frame.dt_seconds <= .2 + 1e-9).all()
        assert (frame.groupby("flight").time.diff().dropna() > 0).all()
        np.testing.assert_allclose(frame.energy_interval_wh, frame.power_w * frame.dt_seconds / 3600., atol=1e-12)
        energies = frame.groupby("flight").energy_interval_wh.sum()
        durations = frame.groupby("flight").dt_seconds.sum()
        truth = audit.set_index("flight").loc[energies.index]
        np.testing.assert_allclose(energies, truth.raw_energy_wh, rtol=1e-11)
        np.testing.assert_allclose(durations, truth.duration_s, rtol=1e-11)
    for row in audit.itertuples():
        assert digest(Path(row.source_file)) == row.source_sha256
    assert set(tables["val"].split_role) == {"val", "calibration"}
    assert tables["val"].groupby("split_role").flight.nunique().to_dict() == {"calibration": 18, "val": 18}
    assert not set(offline.RAW_FEATURE_COLUMNS) & {"power_w", "energy_interval_wh", "task_duration_s", "flight", "route"}
    checks.append("256 source hashes; flight/group/role isolation; finite inputs; time and energy conservation")

    group_results = {}
    for g, spec in offline.GROUPS.items():
        paths, rpaths = offline.group_paths(cfg, g), online.group_paths(rcfg, g)
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        scaler = read_json(paths["scaler"])
        expected = offline.RAW_FEATURE_COLUMNS + ([f"{c}_sma10" for c in offline.SMA_FEATURE_COLUMNS] if spec["use_sma"] else [])
        assert checkpoint["schema_version"] == "AMOVFLY-TCN-comparison-1.0"
        assert checkpoint["feature_columns"] == scaler["feature_columns"] == expected
        assert checkpoint["activation"] == spec["activation"]
        assert checkpoint["epochs_run"] == cfg.epochs == 20
        assert checkpoint["tune_epochs"] == cfg.tune_epochs == 5
        assert digest(paths["checkpoint"]) == digest(rpaths["tcn_checkpoint"])
        assert digest(paths["scaler"]) == digest(rpaths["scaler"])
        features = tables["train"].sort_values(["flight", "time"], kind="stable").copy()
        if spec["use_sma"]:
            for c in offline.SMA_FEATURE_COLUMNS:
                features[f"{c}_sma10"] = features.groupby("flight")[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
        x = features[expected].to_numpy(float)
        np.testing.assert_allclose(x.mean(axis=0), scaler["x_mean"], rtol=1e-10, atol=1e-10)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.
        np.testing.assert_allclose(std, scaler["x_std"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(features.power_w.mean(), scaler["y_mean"], rtol=1e-10)
        formal, tuning, epochs = (pd.read_csv(paths[k]) for k in ("training_log", "tuning", "tuning_log"))
        assert formal.epoch.tolist() == list(range(1, 21))
        assert set(tuning.window_seconds) == set(cfg.tcn_window_seconds)
        assert (tuning.epochs_run == 5).all() and (tuning.flights == 18).all()
        assert epochs.groupby("window_seconds").epoch.apply(list).tolist() == [list(range(1, 6))] * 4
        for p, prefix in ((paths, "TCN"), (rpaths, "RLS")):
            flights = pd.read_csv(p["flight_metrics"], float_precision="round_trip")
            metrics = read_json(p["metrics"])
            wape = float((flights.predicted_energy_wh - flights.actual_energy_wh).abs().sum() / flights.actual_energy_wh.abs().sum() * 100.)
            np.testing.assert_allclose(wape, metrics["flight_energy_wape_percent"], atol=1e-10)
            assert set(flights.flight) == sets[2]
            group_results.setdefault(g, {})[prefix + "_flight_wape_percent"] = wape
        rtune = pd.read_csv(rpaths["rls_tuning"])
        assert len(rtune) == 245 and (rtune.flights == 18).all()
        calibration = read_json(rpaths["interval_calibration"])
        assert calibration["calibration_flights"] == 18
        caltrace = pd.read_csv(rpaths["validation_trace"])
        assert set(caltrace.flight) == set(tables["val"].loc[tables["val"].split_role == "calibration", "flight"])
        trace = pd.read_csv(rpaths["test_trace"], float_precision="round_trip")
        metrics = read_json(rpaths["metrics"])
        for stage, prefix in (("initial", "task_start"), ("post_update", "post_feedback")):
            selected = trace.loc[trace.stage == stage]
            if stage == "post_update":
                selected = selected.loc[selected.remaining_window_count > 0]
            a = selected.actual_final_energy_wh.to_numpy()
            lo, hi = selected.predicted_final_lower_wh.to_numpy(), selected.predicted_final_upper_wh.to_numpy()
            coverage = float(((a >= lo) & (a <= hi)).mean() * 100.)
            score = (hi - lo + 40. * np.maximum(lo - a, 0) + 40. * np.maximum(a - hi, 0)).mean()
            np.testing.assert_allclose(coverage, metrics[prefix + "_interval_coverage_percent"], atol=1e-10)
            np.testing.assert_allclose(score, metrics[prefix + "_interval_score_wh"], atol=1e-9)
        baseline = pd.read_csv(paths["predictions"], float_precision="round_trip")
        prediction = pd.read_csv(rpaths["test_predictions"], float_precision="round_trip")
        np.testing.assert_allclose(baseline.predicted_power_w, prediction.tcn_predicted_power_w, rtol=1e-6, atol=5e-5)
        checks.append(f"{g}: full-train scaler; 4x5 tuning/20 formal epochs; frozen TCN/scaler SHA; 245 RLS candidates; WAPE/coverage/IS recomputed")

    tests = {}
    for config in (cfg, rcfg):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=config.project_root, env=env, capture_output=True, text=True, encoding="utf-8")
        (config.log_dir / "AMOVFLY_validation_tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stderr
        tests[config.project_root.name] = result.stderr.strip().splitlines()[-3:]
        for path in config.project_root.rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    checks.append("TCN 8 and RLS 4 behavior tests; Python syntax compilation")
    docs = [p for config in (cfg, rcfg) for p in config.project_root.rglob("*.md")]
    link_count = sum(check_links(path) for path in docs)
    for path in docs + [cfg.project_root.parent / "prepare_amovfly.py", cfg.project_root.parent / "amovfly_reporting.py"]:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and "\ufffd" not in raw.decode("utf-8"), str(path)
    checks.append(f"UTF-8 documents; {link_count} local file and heading links")
    record = {"checked_at": datetime.now().astimezone().isoformat(), "status": "passed",
              "checks": checks, "data_counts": summary["counts"], "data_sha256": summary["sha256"],
              "groups": group_results, "tests": tests,
              "limits": "Single seed; 20 formal epochs; measured-trajectory replay; empirical interval coverage below 95%."}
    for config in (cfg, rcfg):
        (config.log_dir / "AMOVFLY_artifact_audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
