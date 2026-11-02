# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_analysis.py
# 开发时间: 2026-11-02
# 文件名: test_analysis.py
# 功能说明: 验证AMOVFLY文件发现和数据质量异常识别阈值
# 版本号：1.0

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import NUMERIC_METADATA, _physical_anomalies, ready_files
from config import build_config


class AnalysisTests(unittest.TestCase):
    """功能: 检查分析项目中关键异常事件与文件范围。
    参数: unittest自动提供。
    返回: 测试结果。
    调用位置: unittest discover。
    """

    def test_quality_events_are_counted_without_changing_source(self):
        """功能: 核对重复时间、长断点、缺失和越界事件计数。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        columns = list(NUMERIC_METADATA)
        frame = pd.DataFrame({column: np.ones(5) for column in columns})
        frame["time"] = [0., .2, .2, 3., 3.2]
        frame["wind_speed"] = [1., 1., np.nan, 1., 1.]
        frame["wind_angle"] = 10.
        frame["battery_voltage"] = 15.
        frame["battery_current"] = 10.
        frame["power"] = 150.
        frame["battery_remain"] = [.5, .4, .3, .2, 1.2]
        frame["gps_z"] = [0., 1., -1., 3., 4.]
        frame[["o_x", "o_y", "o_z", "o_w"]] = [0., 0., 0., 1.]
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FAFS" / "UavY_test.csv"
            events = _physical_anomalies(frame, path, build_config())
        counts = {row["code"]: row["count"] for row in events}
        self.assertEqual(counts["duplicate_time"], 1)
        self.assertEqual(counts["non_increasing_time"], 1)
        self.assertEqual(counts["long_sampling_gap"], 1)
        self.assertEqual(counts["nonfinite_value"], 1)
        self.assertEqual(counts["physical_range_battery_remain"], 1)
        self.assertEqual(counts["physical_range_gps_z"], 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_only_ready_csv_files_are_discovered(self):
        """功能: 确认只扫描五个场景的ready同名子目录。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "FAFS" / "FAFS" / "flight.csv"
            valid.parent.mkdir(parents=True)
            valid.touch()
            (root / "FAFS" / "raw_data").mkdir()
            (root / "FAFS" / "raw_data" / "raw.csv").touch()
            cfg = replace(build_config(), data_dir=root)
            self.assertEqual(ready_files(cfg), [valid])

    def test_output_override_derives_all_artifact_directories(self):
        """功能: 确认覆盖输出根目录时表格与图表也跟随变更。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        with tempfile.TemporaryDirectory() as directory:
            cfg = build_config(out_dir=directory)
            root = Path(directory)
            self.assertEqual(cfg.result_dir, root / "results")
            self.assertEqual(cfg.figure_dir, root / "figures")
            self.assertEqual(cfg.log_dir, root / "logs")


if __name__ == "__main__":
    unittest.main()
