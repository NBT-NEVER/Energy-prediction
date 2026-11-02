# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_prepare_amovfly.py
# 开发时间: 2026-11-01
# 文件名: test_prepare_amovfly.py
# 功能说明: 验证原始飞行裁剪、风角处理、能量守恒及长断点拒收
# 版本号：1.0

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import _read_ready_file, _continuous_stage


class PrepareTests(unittest.TestCase):
    """功能: 覆盖AMOVFLY清洗中的物理量和缺测边界。
    参数: unittest自动提供。
    返回: 测试结果。
    调用位置: unittest discover。
    """

    def fixture(self):
        """功能: 构造包含离地边界与跨周风角的短飞行。
        参数: 无。
        返回: 原始格式数据表。
        调用位置: 本测试类。
        """
        n = 40
        return pd.DataFrame({
            "time": np.arange(n) * .21, "gps_x": np.arange(n), "gps_y": 0.,
            "gps_z": [0., *([3.] * (n - 2)), 0.], "wind_speed": 1.,
            "wind_angle": np.where(np.arange(n) % 2, 179., -179.),
            "v_x": 1., "v_y": 0., "v_z": 0., "battery_voltage": 10.,
            "battery_current": np.linspace(10., 20., n),
        })

    def test_clean_energy_tail_wind_and_label_isolation(self):
        """功能: 核对去重裁剪、短尾时间箱、跨周风角和标签隔离。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        source = self.fixture()
        dirty = pd.concat([source, source.iloc[[5]], source.iloc[[7]].assign(battery_current=np.nan)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UavR_P200_example.csv"
            dirty.to_csv(path, index=False, encoding="utf-8")
            result, audit = _read_ready_file(path)
            self.assertEqual(audit["rows_invalid"], 1)
            self.assertEqual(audit["rows_duplicate_time"], 1)
            self.assertEqual(audit["rows_trimmed"], 0)
            expected = np.trapezoid(source.battery_voltage * source.battery_current, source.time) / 3600.
            self.assertAlmostEqual(result.energy_interval_wh.sum(), expected, places=12)
            self.assertAlmostEqual(result.dt_seconds.sum(), 8.19, places=12)
            self.assertLess(result.dt_seconds.iloc[-1], .2)
            self.assertTrue((result.wind_sensor_cos_mps < -.99).all())
            np.testing.assert_allclose(result[["position_x_m", "position_y_m", "position_z_m"]].iloc[0], 0.)
            dirty["battery_current"] *= 2.
            dirty.to_csv(path, index=False, encoding="utf-8")
            changed, _ = _read_ready_file(path)
            inputs = [c for c in result if c not in {"power_w", "energy_interval_wh"}]
            np.testing.assert_allclose(result[inputs], changed[inputs])

    def test_dji_status_thresholds_and_combined_state(self):
        """功能: 核对100W、1m、0.5m边界、阶段填充及三级标签。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        np.testing.assert_array_equal(_continuous_stage(np.array([0, 1, 0, 1, 0], dtype=bool)), [0, 1, 1, 1, 0])
        source = self.fixture()
        source["time"] = np.arange(len(source)) * .2
        source["gps_x"] = 0.
        source["gps_z"] = 0.
        source["battery_current"] = 0.
        source.loc[5:34, "battery_current"] = 10.
        source.loc[10:29, "gps_z"] = .5
        source.loc[15, "battery_current"] = 0.
        with tempfile.TemporaryDirectory() as directory:
            for name, code in (("Y", 1), ("R", 2), ("G", 3)):
                path = Path(directory) / f"Uav{name}_P200_example.csv"
                source.to_csv(path, index=False, encoding="utf-8")
                result, _ = _read_ready_file(path)
                self.assertTrue(result.uav_type.eq(code).all())
                np.testing.assert_array_equal(result.planned, [0]*5 + [1]*5 + [2]*20 + [1]*5 + [0]*4)
                self.assertTrue((result.planned_airborne <= result.planned_motor_on).all())
                self.assertEqual(result.planned.iloc[15], 2)
            source["gps_z"] = 0.
            source.loc[10:29, "gps_x"] = 1.
            source.to_csv(path, index=False, encoding="utf-8")
            horizontal, _ = _read_ready_file(path)
            np.testing.assert_array_equal(horizontal.planned, result.planned)

    def test_all_zero_power_is_finite(self):
        """功能: 检查全停机记录不会出现守恒除零。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        source = self.fixture().assign(battery_current=0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UavY_P0_example.csv"
            source.to_csv(path, index=False, encoding="utf-8")
            result, _ = _read_ready_file(path)
            self.assertTrue(result.planned.eq(0).all())
            self.assertTrue(result.power_w.eq(0).all())

    def test_long_gap_rejected(self):
        """功能: 确认离地航段内超过2秒的缺口被拒收。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        source = self.fixture()
        source.loc[20:, "time"] += 3.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UavR_P200_example.csv"
            source.to_csv(path, index=False, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "长断点"):
                _read_ready_file(path)
