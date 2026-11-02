# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-11-01
# 文件名: report.py
# 功能说明: 生成TCN加RLS实验README和当前输出详细报告
# 版本号：1.0

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amovfly_reporting import write_reports

def write_report(cfg):
    return write_reports(cfg, online=True)
