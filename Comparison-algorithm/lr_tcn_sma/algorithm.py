# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: algorithm.py
# 开发时间: 2026-09-20
# 文件名: algorithm.py
# 功能说明: 提供 LR-TCN-SMA（LeakyReLU Temporal Convolutional Network with Simple Moving Average，带简单移动平均的LeakyReLU时间卷积网络）的独立运行入口
# 版本号：4.1

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comparison_engine import main as run_algorithm_experiment  # 运行当前目录对应的单算法实验

if __name__ == "__main__":
    run_algorithm_experiment(["lr_tcn_sma"])
