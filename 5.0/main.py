# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-09-16
# 功能说明: 调度实验4.1数据准备、TCN/RLS搜索、评估和任务规划接口演示
# 版本号：4.1

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import build_config, ensure_directories
from data_utils import prepare_dataset
from evaluate import evaluate_model
from task_api import OnlineTaskSession, predict_task_before
from terminal_logger import TerminalLogCapture, log_result
from train import fit_scaler, resolve_cuda_device, train_tcn, tune_rls, tune_tcn
from visualize import generate_all_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实验4.1：面向任务规划和充电决策的无人机能耗预测")
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["prepare", "tune-tcn", "train", "tune-rls", "evaluate", "task-demo", "online-demo", "visualize", "all"])
    parser.add_argument("--data-dir", type=Path, default=None, help="覆盖实验4.1版本数据目录。")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-prepare", action="store_true")
    return parser


def _config(args):
    overrides = {"device": args.device, "epochs": args.epochs, "tune_epochs": args.tune_epochs,
                 "batch_size": args.batch_size, "save_dir": args.save_dir, "out_dir": args.out_dir,
                 "data_dir": args.data_dir}
    cfg = build_config(**{k: v for k, v in overrides.items() if v is not None})
    ensure_directories(cfg)
    return cfg


def _reset_generated_outputs(cfg) -> list[str]:
    """功能: 在完整重跑前清除4.1中会被本轮重新生成的旧产物目录。
    参数: cfg为实验配置对象。
    返回: 已清理目录名称列表。
    调用位置: main。
    """

    out_root = cfg.out_dir.resolve()
    targets = [cfg.out_model_dir, cfg.out_prediction_dir, cfg.out_task_dir,
               cfg.out_figure_dir, cfg.out_route_dir, cfg.out_rls_dir]
    removed = []
    for path in targets:
        resolved = path.resolve()
        if resolved.parent != out_root:
            raise RuntimeError(f"拒绝清理OUT_DIR边界之外的目录: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
            removed.append(resolved.name)
    if cfg.visualization_summary_json.exists():
        cfg.visualization_summary_json.unlink()
    ensure_directories(cfg)
    return removed


def _stage(name: str, action):
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n>>> {name} 开始: {started_at}")
    try:
        result = action()
    except Exception as exc:
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        log_result(f"{name}阶段", {
            "started_at": started_at, "finished_at": finished_at,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": "失败", "exception_type": type(exc).__name__, "message": str(exc),
        })
        raise
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed = time.perf_counter() - started
    log_result(f"{name}结果", result)
    log_result(f"{name}阶段", {
        "started_at": started_at, "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 3), "status": "完成",
    })
    print(f">>> {name} 完成，耗时 {elapsed:.2f}s")
    return result


def _ensure_prepared(cfg, force=False):
    summary = _stage("prepare：0.2秒数据准备", lambda: prepare_dataset(cfg, force=force))
    train = pd.read_csv(cfg.train_csv)
    if not cfg.scaler_json.exists() or force:
        fit_scaler(train, cfg)
    return summary


def _demo_task(cfg) -> pd.DataFrame:
    dt = cfg.resample_seconds
    times = np.arange(0.0, 40.0, dt)
    progress = times / 40.0
    motor_on = ((times >= 2.0) & (times <= 38.0)).astype(int)
    airborne = ((times >= 4.0) & (times <= 36.0)).astype(int)
    up = np.where(times < 4.0, 0.0, np.where(times <= 8.0, (times - 4.0) * 5.0, 20.0))
    up = np.where(times > 32.0, np.maximum((36.0 - times) * 5.0, 0.0), up)
    task = pd.DataFrame({
        "task_id": "demo_new_route_001",
        "time_s": times,
        "task_duration_s": 40.0,
        "planned_position_east_m": 120.0 * progress,
        "planned_position_north_m": 25.0 * np.sin(np.pi * progress),
        "planned_position_up_m": up,
        "wind_east_mps": 1.5 + 0.2 * np.sin(times / 10.0),
        "wind_north_mps": 2.6 + 0.2 * np.cos(times / 10.0),
        "payload_g": 220.0,
        "planned_motor_on": motor_on,
        "planned_airborne": airborne,
    })
    return task


def task_demo(cfg) -> dict:
    task = _demo_task(cfg)
    task.to_csv(cfg.task_input_csv, index=False, encoding="utf-8")
    output, summary = predict_task_before(task, cfg, available_energy_wh=80.0, return_energy_wh=8.0)
    output.to_csv(cfg.task_output_csv, index=False, encoding="utf-8")
    cfg.task_output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def online_demo(cfg) -> dict:
    task = pd.read_csv(cfg.task_input_csv) if cfg.task_input_csv.exists() else _demo_task(cfg)
    session = OnlineTaskSession(task, cfg)
    steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
    baseline_window = float(session.base.iloc[:steps]["predicted_energy_wh"].sum())
    remaining, summary = session.update(0, baseline_window * 1.05, available_energy_wh=80.0, return_energy_upper_wh=8.0)
    remaining.to_csv(cfg.online_output_csv, index=False, encoding="utf-8")
    cfg.online_output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def visualize(cfg) -> dict:
    return generate_all_visualizations(cfg)


def run(mode: str, cfg, force_prepare: bool = False) -> dict:
    if mode == "prepare": return _ensure_prepared(cfg, force_prepare)
    if mode == "tune-tcn": _ensure_prepared(cfg, False); return _stage("tune-tcn：扩展时间窗搜索", lambda: tune_tcn(cfg))
    if mode == "train": _ensure_prepared(cfg, False); return _stage("train：正式TCN", lambda: train_tcn(cfg))
    if mode == "tune-rls": return _stage("tune-rls：扩展参数网格", lambda: tune_rls(cfg))
    if mode == "evaluate": return _stage("evaluate：测试评估", lambda: evaluate_model(cfg))
    if mode == "task-demo": return _stage("task-demo：任务前接口", lambda: task_demo(cfg))
    if mode == "online-demo": return _stage("online-demo：在线RLS接口", lambda: online_demo(cfg))
    if mode == "visualize": return visualize(cfg)
    if mode == "all":
        summary = {"prepare": _ensure_prepared(cfg, force_prepare),
                   "tune_tcn": _stage("tune-tcn：扩展时间窗搜索", lambda: tune_tcn(cfg)),
                   "train": _stage("train：正式TCN", lambda: train_tcn(cfg)),
                   "tune_rls": _stage("tune-rls：扩展参数网格", lambda: tune_rls(cfg)),
                   "evaluate": _stage("evaluate：测试评估", lambda: evaluate_model(cfg)),
                   "task_demo": _stage("task-demo：任务前接口", lambda: task_demo(cfg)),
                   "online_demo": _stage("online-demo：在线RLS接口", lambda: online_demo(cfg)),
                   "visualize": _stage("visualize：SVG图表", lambda: visualize(cfg))}
        return summary
    raise ValueError(mode)


def main() -> None:
    args = build_parser().parse_args()
    cfg = _config(args)
    removed = _reset_generated_outputs(cfg) if args.mode == "all" else []
    reset = args.mode in {"all", "tune-tcn"}
    with TerminalLogCapture(cfg.terminal_log_file, args.mode, reset_log=reset):
        print(f"实验4.1 | mode={args.mode} | device={cfg.device} | resample={cfg.resample_seconds}s")
        if removed:
            print(f"已清理并重建旧产物目录: {', '.join(removed)}")
        if args.mode != "prepare":
            device = resolve_cuda_device(cfg)
            properties = torch.cuda.get_device_properties(device)
            print(f"CUDA设备={properties.name} | 显存={properties.total_memory / 1024 ** 3:.2f} GiB")
        print(f"TCN候选={list(cfg.tcn_window_seconds)}s | RLS组合={len(cfg.rls_forgetting_factors) * len(cfg.rls_initial_covariances)}")
        result = run(args.mode, cfg, args.force_prepare)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
