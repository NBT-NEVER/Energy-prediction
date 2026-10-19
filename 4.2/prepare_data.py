# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: prepare_data.py
# 开发时间: 2026-09-18
# 文件名: prepare_data.py
# 功能说明: 独立执行实验4.1原始飞行数据整理、重采样、状态标注和数据集切分
# 版本号：4.1

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import build_config
from data_utils import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    """功能: 构造4.1独立数据处理命令行参数。
    参数: 无。
    返回: argparse参数解析器。
    调用位置: main。
    """

    parser = argparse.ArgumentParser(description="实验4.1独立数据处理")
    parser.add_argument("--data-dir", type=Path, default=None, help="4.1加工数据输出目录。")
    parser.add_argument("--source-archive", type=Path, default=None, help="Source中的原始ZIP归档。")
    parser.add_argument("--force", action="store_true", help="覆盖并重新生成现有4.1加工数据。")
    return parser


def main() -> None:
    """功能: 从Source生成4.1完整加工数据并打印结构化摘要。
    参数: 命令行参数。
    返回: None。
    调用位置: 脚本直接运行。
    """

    args = build_parser().parse_args()
    overrides = {
        key: value
        for key, value in {"data_dir": args.data_dir, "source_archive": args.source_archive}.items()
        if value is not None
    }
    summary = prepare_dataset(build_config(**overrides), force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
