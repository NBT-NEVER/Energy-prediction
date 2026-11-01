# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-10-30
# 文件名: main.py
# 功能说明: 调度四组TCN离线功率与能耗预测实验
# 版本号：1.0

from __future__ import annotations

import argparse
import json
import sys
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from config import GROUPS, build_config, ensure_directories  # 实验组、集中路径和目录初始化
from data_utils import copy_processed_data  # 将已划分的4.1加工表导入独立D盘目录
from evaluate import evaluate_all  # 汇总离线功率与能耗测试指标
from train import train_group, tune_group  # 正式训练与TCN窗口搜索
from visualize import plot_results  # 根据CSV重建结果图表


def build_parser() -> argparse.ArgumentParser:
    """功能: 构造四组TCN实验命令行参数。
    参数: 无。
    返回: 参数解析器。
    调用位置: main。
    """

    parser = argparse.ArgumentParser(description="四组TCN离线功率和能耗预测实验")
    parser.add_argument("mode", nargs="?", default="all", choices=["prepare", "tune", "train", "evaluate", "plot", "report", "all"])
    parser.add_argument("--group", choices=["A", "B", "C", "D"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


class TerminalCapture:
    """功能: 同时将控制台文本写入UTF-8日志。
    参数: stream为原控制台，file为日志文件。
    返回: 可替代stdout或stderr的对象。
    调用位置: main。
    """

    def __init__(self, stream, file):
        """功能: 绑定控制台与日志文件。
        参数: stream为原始流，file为日志文件。
        返回: None。
        调用位置: main。
        """
        self.stream, self.file = stream, file

    def write(self, value):
        """功能: 同步写入控制台和日志。
        参数: value为文本。
        返回: 写入字符数。
        调用位置: print及异常输出。
        """
        self.stream.write(value)
        self.file.write(value.replace("\r", "\n"))
        self.file.flush()
        return len(value)

    def flush(self):
        """功能: 刷新两个输出流。
        参数: 无。
        返回: None。
        调用位置: print及程序退出。
        """
        self.stream.flush()
        self.file.flush()


def main() -> None:
    """功能: 执行数据准备、窗口搜索、正式训练、测试评价和绘图。
    参数: 命令行参数。
    返回: None。
    调用位置: 脚本直接运行。
    """

    args = build_parser().parse_args()
    cfg = build_config(device=args.device, tune_epochs=args.tune_epochs, epochs=args.epochs,
                       batch_size=args.batch_size, random_seed=args.seed, data_dir=args.data_dir,
                       save_dir=args.save_dir, out_dir=args.out_dir)
    ensure_directories(cfg)
    runtime = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               "platform": platform.platform(), "config": asdict(cfg)}
    cfg.runtime_json.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with cfg.terminal_log.open("a", encoding="utf-8") as log:
        sys.stdout, sys.stderr = TerminalCapture(original_stdout, log), TerminalCapture(original_stderr, log)
        try:
            run_pipeline(args, cfg)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def run_pipeline(args, cfg):
    """功能: 根据命令行模式依次完成各实验阶段。
    参数: args为命令行参数，cfg为已派生路径的配置。
    返回: None。
    调用位置: main。
    """

    groups = [args.group] if args.group else list(GROUPS)
    if args.mode in {"prepare", "all"}:
        prepared = copy_processed_data(cfg, force=args.force_prepare)
        print(json.dumps({key: value for key, value in prepared.items()
                          if key not in {"sha256", "split_flights", "copied_files"}}, ensure_ascii=False, indent=2))
        if args.mode == "prepare":
            return
    if args.mode in {"tune", "all"}:
        for group in groups:
            print(f"\n===== 组 {group} 窗口搜索 =====")
            print(json.dumps(tune_group(group, cfg), ensure_ascii=False, indent=2))
        if args.mode == "tune":
            return
    if args.mode in {"train", "all"}:
        for group in groups:
            print(f"\n===== 组 {group} 正式训练 =====")
            print(json.dumps(train_group(group, cfg), ensure_ascii=False, indent=2))
        if args.mode == "train":
            return
    if args.mode in {"evaluate", "all"}:
        result = evaluate_all(cfg, groups)
        print(result[["group", "window_seconds", "feature_count", "power_wape_percent",
                      "window_energy_wape_percent", "flight_energy_wape_percent"]].to_string(index=False))
        if args.mode == "evaluate":
            from report import write_reports
            print(json.dumps(write_reports(cfg), ensure_ascii=False, indent=2))
            return
    if args.mode in {"plot", "all"}:
        # 使用已保存指标和预测表生成图形，避免绘图时重算测试预测。
        print(json.dumps({"figures": plot_results(cfg)}, ensure_ascii=False, indent=2))
    if args.mode in {"report", "plot", "all", "evaluate"}:
        from report import write_reports
        print(json.dumps(write_reports(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
