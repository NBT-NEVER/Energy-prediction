# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-10-19
# 文件名: main.py
# 功能说明: 调度实验5.0无人车真实行程数据准备、训练、评估与展示
# 版本号：5.0

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
import pandas as pd
from vehicle_pipeline import _log_event, evaluate_and_plot, predict_test, prepare_data, search_rls, search_tcn, train_model
from terminal_logger import TerminalLogCapture, log_result


def _stage(name: str, action):
    """功能: 输出阶段开始、结果、耗时并写入终端日志。
    参数: name为阶段名称，action为无参执行函数。
    返回: 阶段结果对象。
    调用位置: main。
    """
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n>>> {name} 开始: {started_at}")
    try:
        result = action()
    except Exception as exc:
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        log_result(f"{name}阶段", {"started_at": started_at, "finished_at": finished_at, "elapsed_seconds": round(time.perf_counter() - started, 3), "status": "失败", "exception_type": type(exc).__name__, "message": str(exc)})
        raise
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed = time.perf_counter() - started
    log_result(f"{name}结果", result)
    log_result(f"{name}阶段", {"started_at": started_at, "finished_at": finished_at, "elapsed_seconds": round(elapsed, 3), "status": "完成"})
    print(f">>> {name} 完成，耗时 {elapsed:.2f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="实验5.0：GECR真实道路BEV双层能耗预测")
    parser.add_argument("mode", nargs="?", default="all", choices=("prepare", "search", "train", "predict", "predict-task", "evaluate", "all"))
    parser.add_argument("--force-prepare", action="store_true", help="重新清洗并切分Excel行程。")
    parser.add_argument("--task-input", help="调度器逐秒计划CSV路径，仅用于predict-task。")
    parser.add_argument("--task-output", help="逐秒功率/能耗预测CSV输出路径。")
    args = parser.parse_args()
    config.ensure_directories()
    reset_log = args.mode in {"all", "search"}
    with TerminalLogCapture(config.TERMINAL_LOG_FILE, args.mode, reset_log=reset_log):
        print(f"实验5.0 | mode={args.mode} | device={config.DEVICE} | resample={config.SAMPLE_SECONDS}s")
        print(f"TCN窗口候选={list(config.WINDOW_CANDIDATES)}s | TCN参数候选={len(config.MODEL_CANDIDATES)} | RLS组合={len(config.RLS_WINDOWS_SECONDS) * len(config.RLS_FORGETTING_FACTORS) * len(config.RLS_INITIAL_COVARIANCES)}")
        print("TCN目标=逐秒功率；调参选择分数=0.2×逐秒功率WAPE+5秒能耗WAPE+0.5×整程能耗WAPE")
        print("RLS目标=逐秒功率积分后的窗口能耗；每个窗口先预测，获得该窗口真实能量后更新参数")
        results = {}
        if args.mode in ("prepare", "all"):
            results["prepare"] = _stage("prepare：1秒数据准备", lambda: prepare_data(force=args.force_prepare))
        if args.mode in ("search", "all"):
            results["tcn_search"] = _stage("search-tcn：功率预测、能耗优先调参", search_tcn)
            results["rls_search"] = _stage("search-rls：秒级能耗窗口递推调参", search_rls)
        if args.mode in ("train", "all"):
            results["train"] = _stage("train：正式TCN功率模型", train_model)
        if args.mode == "predict":
            results["predict"] = _stage("predict：冻结测试集功率与能耗曲线", predict_test)
        if args.mode == "predict-task":
            if not args.task_input:
                parser.error("predict-task必须指定 --task-input")
            from task_api import predict_scheduled_task
            def run_task():
                task_output, summary = predict_scheduled_task(pd.read_csv(args.task_input))
                output_path = Path(args.task_output) if args.task_output else config.TASK_PREDICTIONS_CSV
                output_path.parent.mkdir(parents=True, exist_ok=True)
                task_output.to_csv(output_path, index=False, encoding="utf-8")
                _log_event("predict_task_complete", task_id=summary["task_id"], step_count=summary["step_count"], predicted_total_energy_wh=summary["predicted_total_energy_wh"], prediction_file=str(output_path), confidence_interval=summary["confidence_interval"])
                return {**summary, "prediction_file": str(output_path)}
            results["predict_task"] = _stage("predict-task：任务前功率与能耗曲线", run_task)
        if args.mode in ("evaluate", "all"):
            results["evaluate"] = _stage("evaluate：独立行程功率与能耗评估", evaluate_and_plot)
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
