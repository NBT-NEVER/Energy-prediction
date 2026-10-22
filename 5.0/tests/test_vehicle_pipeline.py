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
import torch

from vehicle_pipeline import ARCHITECTURE_VERSION, CausalConv1d, FEATURES, TCN, _rls_final_trajectory, _rls_predict, _split_conformal_quantile, _validate_window_coverage, _windows
from task_api import validate_scheduled_task


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
        self.assertEqual(len(x), 9)
        self.assertEqual(set(trips), {"a", "b"})
        np.testing.assert_array_equal(actual, np.arange(9))
        np.testing.assert_array_equal(y, actual)
        np.testing.assert_array_equal(x[0, 0], np.zeros(len(FEATURES)))
        np.testing.assert_array_equal(x[0, 1], np.zeros(len(FEATURES)))
        np.testing.assert_array_equal(x[1, 0], np.zeros(len(FEATURES)))

    def test_rls_first_window_is_predicted_before_observation_update(self):
        result = _rls_predict(np.array([10.0, 20.0]), np.array([8.0, 16.0]), 1, 0.9, 1.0)
        np.testing.assert_allclose(result, [8.0, 19.91502298], rtol=1e-6)

    def test_rls_final_energy_trajectory_adds_observed_and_remaining_forecast(self):
        rows = _rls_final_trajectory("trip", np.array([10.0, 20.0, 30.0]), np.array([8.0, 16.0, 24.0]), 1, 0.96, 1.0, 4.0)
        self.assertEqual(rows[0]["observed_energy_wh"], 10.0)
        self.assertEqual(rows[0]["remaining_windows"], 2)
        self.assertAlmostEqual(rows[0]["predicted_final_energy_wh"], rows[0]["observed_energy_wh"] + rows[0]["predicted_remaining_energy_wh"])
        self.assertGreaterEqual(rows[0]["parameter_variance_wh2"], 0.0)
        self.assertGreater(rows[0]["remaining_residual_variance_wh2"], 0.0)
        self.assertEqual(rows[-1]["remaining_windows"], 0)

    def test_split_conformal_uses_finite_sample_rank(self):
        self.assertEqual(_split_conformal_quantile(list(range(1, 21)), 0.95), 20.0)
        self.assertTrue(np.isinf(_split_conformal_quantile(list(range(1, 10)), 0.95)))

    def test_tcn_uses_earlier_history_with_causal_receptive_field(self):
        model = TCN(feature_count=1, channels=2, kernel_size=3, dropout=0.0).eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, torch.nn.Conv1d):
                    layer.weight.fill_(0.1)
                    layer.bias.fill_(0.1)
        x = torch.ones((1, 40, 1), requires_grad=True)
        output = model(x)
        output.sum().backward()
        self.assertGreater(float(x.grad[0, -29, 0].abs()), 0.0)
        self.assertEqual(float(x.grad[0, -30, 0].abs()), 0.0)
        self.assertEqual(ARCHITECTURE_VERSION, "causal_dilated_residual_v3")

    def test_scheduler_contract_accepts_shuffled_rows_and_numeric_strings(self):
        frame = pd.DataFrame({"task_id": ["t1", "t1"], "payload_kg": ["10", "10"]})
        frame["time_s"] = [1, 0]
        frame["task_duration_s"] = [2, 2]
        for feature in FEATURES:
            if feature not in frame:
                frame[feature] = [0, 0]
        frame["planned_speed_kmh"] = [20, 10]
        validate_scheduled_task(frame)

    def test_scheduler_contract_rejects_observed_power(self):
        frame = pd.DataFrame({"task_id": ["t1"], "payload_kg": [0]})
        for feature in FEATURES:
            frame[feature] = [0]
        frame["task_duration_s"] = 1
        frame["power_w"] = 100
        with self.assertRaisesRegex(ValueError, "禁止包含实测状态或监督标签"):
            validate_scheduled_task(frame)

    def test_scheduler_contract_rejects_truncated_task_horizon(self):
        frame = pd.DataFrame({"task_id": ["t1", "t1"], "payload_kg": [0, 0], "time_s": [0, 1], "task_duration_s": [3, 3]})
        for feature in FEATURES:
            if feature not in frame:
                frame[feature] = [0, 0]
        with self.assertRaisesRegex(ValueError, "完整计划轨迹"):
            validate_scheduled_task(frame)


if __name__ == "__main__":
    unittest.main()
