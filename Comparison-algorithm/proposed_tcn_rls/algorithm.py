# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: algorithm.py
# 开发时间: 2026-09-20
# 文件名: algorithm.py
# 功能说明: 提供 4.1 TCN（Temporal Convolutional Network，时间卷积网络）+
#          RLS（Recursive Least Squares，递推最小二乘）的独立运行入口
# 版本号：4.1

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comparison_engine import main as run_algorithm_experiment  # 运行当前目录对应的单算法实验

if __name__ == "__main__":
    run_algorithm_experiment(["proposed_tcn_rls"])
