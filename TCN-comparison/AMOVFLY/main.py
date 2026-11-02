# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-11-02
# 文件名: main.py
# 功能说明: 调度AMOVFLY原始数据统计、图表及详细分析报告
# 版本号：1.0

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from analysis import analyze_dataset  # 汇总原始字段、飞行和异常统计
from config import build_config, ensure_directories  # 加载统计路径并创建输出目录
from report import write_report  # 生成含表格、解释和图表索引的Markdown报告
from visualize import create_figures  # 保存统计图与代表性航线图


def build_parser() -> argparse.ArgumentParser:
    """功能: 构造AMOVFLY统计分析命令行参数。
    参数: 无。
    返回: 参数解析器。
    调用位置: main。
    """
    parser = argparse.ArgumentParser(description="AMOVFLY数据统计与异常分析")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-gap", type=float, default=None)
    return parser


def main() -> None:
    """功能: 执行AMOVFLY原始数据统计、可视化及报告生成。
    参数: 命令行参数。
    返回: None。
    调用位置: 脚本直接运行。
    """
    args = build_parser().parse_args()
    overrides = {"data_dir": args.data_dir, "out_dir": args.out_dir}
    cfg = build_config(**overrides)
    if args.max_gap is not None:
        from dataclasses import replace
        cfg = replace(cfg, max_gap_seconds=args.max_gap)
    ensure_directories(cfg)
    runtime = {"started_at": datetime.now().astimezone().isoformat(),
               "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "source": str(cfg.data_dir),
               "output": str(cfg.out_dir), "max_gap_seconds": cfg.max_gap_seconds}
    cfg.runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    result = analyze_dataset(cfg)
    figures = create_figures(cfg, result)
    report = write_report(cfg, result, figures)
    print(json.dumps({"summary": result["record"], "figures": figures, "report": str(report)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
