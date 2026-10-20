# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_vehicle_pipeline.py
# 开发时间: 2026-10-19
# 文件名: test_vehicle_pipeline.py
# 功能说明: 检查5.0多行程窗口、下一秒标签和RLS因果更新契约
# 版本号：5.0

import unittest

import numpy as np
import pandas as pd

from vehicle_pipeline import FEATURES, _rls_predict, _validate_window_coverage, _windows


class VehiclePipelineContractTests(unittest.TestCase):
    """检查5.0模型输入与在线校正的关键数据契约。"""

    def test_windows_cover_each_trip_and_predict_next_power(self):
        frame = pd.DataFrame({
            "trip_id": ["a"] * 5 + ["b"] * 4,
            "power_w": np.arange(9, dtype=float),
        })
        for feature in FEATURES:
            frame[feature] = np.arange(9, dtype=float)
        x, y, trips, actual = _windows(frame, np.zeros(len(FEATURES)), np.ones(len(FEATURES)), 0.0, 1.0, 2)
        _validate_window_coverage(frame, 2, len(x))
        self.assertEqual(len(x), 5)
        self.assertEqual(set(trips), {"a", "b"})
        np.testing.assert_array_equal(actual, [2, 3, 4, 7, 8])
        np.testing.assert_array_equal(y, actual)

    def test_rls_first_window_is_predicted_before_observation_update(self):
        result = _rls_predict(np.array([10.0, 20.0]), np.array([8.0, 16.0]), 1, 0.9, 1.0)
        np.testing.assert_allclose(result, [8.0, 19.91502298], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
