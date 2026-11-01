# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-10-30
# 文件名: main.py
# 功能说明: 调度TCN加RLS五组调参、区间校准、测试评价和绘图
# 版本号：1.0

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import pandas as pd

from config import GROUPS, build_config, ensure_directories, group_paths  # 实验组、配置与统一输出路径
from rls_evaluate import calibrate_group, evaluate_all, prepare_data, tune_group  # 校准、评价、数据准备与RLS搜索


def build_parser() -> argparse.ArgumentParser:
    """功能: 构造TCN加RLS命令行解析器。
    参数: 无。
    返回: 参数解析器。
    调用位置: main。
    """
    parser = argparse.ArgumentParser(description="五组TCN加RLS和置信区间评估")
    parser.add_argument("mode", nargs="?", default="all", choices=["prepare", "tune", "calibrate", "evaluate", "plot", "report", "all"])
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser


def main() -> None:
    """功能: 执行TCN加RLS完整实验流程。
    参数: 命令行参数。
    返回: None。
    调用位置: 脚本直接运行。
    """
    args = build_parser().parse_args()
    cfg = build_config(device=args.device, out_dir=args.out_dir, save_dir=args.save_dir, data_dir=args.data_dir)
    ensure_directories(cfg)
    runtime = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               "platform": platform.platform(), "config": asdict(cfg),
               "scope": "使用TCN已训练权重，新增RLS调参和区间校准；不重新训练TCN",
               "mode": args.mode, "groups": [args.group] if args.group else list(GROUPS)}
    if args.mode == "all" and args.group:
        group_runtime = cfg.runtime_json.with_name(f"{cfg.runtime_json.stem}_{args.group}.json")
        group_runtime.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if (args.mode == "all" and not args.group) or not cfg.runtime_json.exists():
        cfg.runtime_json.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with cfg.terminal_log.open("a", encoding="utf-8") as log:
        class Capture:
            """功能: 同步保存终端输出。
            参数: stream和file为输出对象。
            返回: 可替代标准输出的对象。
            调用位置: main。
            """
            def __init__(self, stream, file): self.stream, self.file = stream, file
            def write(self, value): self.stream.write(value); self.file.write(value); self.file.flush(); return len(value)
            def flush(self): self.stream.flush(); self.file.flush()
        sys.stdout, sys.stderr = Capture(original_stdout, log), Capture(original_stderr, log)
        try:
            print(f"\n[{runtime['started_at']}] START mode={args.mode} group={args.group or 'ABCDE'}")
            run_pipeline(args, cfg)
            print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] COMPLETE mode={args.mode}")
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def run_pipeline(args, cfg) -> None:
    """功能: 根据模式依次执行准备、调参、校准、评价和报告。
    参数: args为命令行参数，cfg为实验配置。
    返回: None。
    调用位置: main。
    """
    groups = [args.group] if args.group else list(GROUPS)
    if args.mode in {"prepare", "all"}:
        print(json.dumps(prepare_data(cfg, args.force_prepare), ensure_ascii=False, indent=2))
        if args.mode == "prepare": return
    if args.mode in {"tune", "all"}:
        for group in groups:
            print(f"\n===== 组{group} RLS超参搜索 =====")
            print(json.dumps(tune_group(group, cfg), ensure_ascii=False, indent=2))
        if args.mode == "tune": return
    if args.mode in {"calibrate", "all"}:
        for group in groups:
            tuning = pd.read_csv(group_paths(cfg, group)["rls_tuning"]).sort_values("flight_energy_wape_percent").iloc[0].to_dict()
            print(json.dumps(calibrate_group(group, cfg, tuning), ensure_ascii=False, indent=2))
        if args.mode == "calibrate": return
    if args.mode in {"evaluate", "all"}:
        result = evaluate_all(cfg, groups)
        print(result[["group", "rls_window_seconds", "rls_forgetting_factor", "rls_initial_covariance",
                      "flight_energy_wape_percent", "task_start_interval_coverage_percent",
                      "post_feedback_interval_coverage_percent"]].to_string(index=False))
        if args.mode == "evaluate":
            from report import write_report
            write_report(cfg)
            return
    if args.mode in {"plot", "all"}:
        from visualize_rls import plot_all
        print(json.dumps(plot_all(cfg), ensure_ascii=False, indent=2))
    if args.mode in {"report", "plot", "all"}:
        from report import write_report
        print(json.dumps(write_report(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
