# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_rls_pipeline.py
# 开发时间: 2026-10-30
# 文件名: test_rls_pipeline.py
# 功能说明: 验证RLS更新顺序、区间边界和四组搜索网格
# 版本号：1.0

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GROUPS, RLS_FORGETTING_FACTORS, RLS_INITIAL_COVARIANCES, RLS_WINDOW_SECONDS
from rls import DynamicEnergyRLS, simulate_rls


def small_frame(rows=10):
    """功能: 构造短flight模拟表。
    参数: rows为采样点数。
    返回: 含能耗标签的测试表。
    调用位置: RlsTests。
    """
    return pd.DataFrame({
        "flight": [1] * rows, "route": ["R"] * rows,
        "time": np.arange(rows) * .2, "time_s": np.arange(rows) * .2,
        "dt_seconds": [.2] * rows, "power_w": [100.] * rows,
    })


class RlsTests(unittest.TestCase):
    """功能: 验证RLS加区间实验的关键行为。
    参数: unittest自动提供。
    返回: 测试结果。
    调用位置: unittest discover。
    """

    def test_grid_has_window_dimension(self):
        """功能: 检查RLS网格包含遗忘因子、协方差和时间窗。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        self.assertEqual(len(RLS_FORGETTING_FACTORS) * len(RLS_INITIAL_COVARIANCES) * len(RLS_WINDOW_SECONDS), 245)
        self.assertEqual(set(GROUPS), {"A", "B", "C", "D"})

    def test_update_changes_state_only_when_called(self):
        """功能: 检查RLS更新前后的参数状态变化。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        rls = DynamicEnergyRLS(.96, .5, initial_residual_std_wh=.01)
        before = rls.state().copy()
        rls.update(.02, .03)
        after = rls.state()
        self.assertEqual(before["update_count"], 0)
        self.assertEqual(after["update_count"], 1)
        self.assertNotEqual(before["theta_bias_wh_per_window"], after["theta_bias_wh_per_window"])

    def test_simulation_has_causal_trace_and_nonnegative_interval(self):
        """功能: 检查模拟输出的任务边界、反馈顺序和非负区间。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        frame = small_frame()
        output = simulate_rls(frame, np.full(len(frame), 100.), .88, 1.0, 1.0, .2)
        self.assertEqual(len(output.predictions), len(frame))
        self.assertEqual(output.trace.iloc[0].observed_window_count, 0)
        self.assertTrue((output.predictions.predicted_power_lower_w >= 0).all())
        self.assertTrue((output.predictions.predicted_power_upper_w >= output.predictions.predicted_power_lower_w).all())
        self.assertTrue((output.flights.predicted_lower_wh >= 0).all())
        self.assertTrue((output.flights.predicted_upper_wh >= output.flights.predicted_lower_wh).all())

    def test_current_and_future_labels_cannot_change_current_prediction(self):
        """功能: 扰动当前和未来实际功率并检查窗口预测不受未到达标签影响。
        参数: 无。
        返回: 因果顺序断言结果。
        调用位置: unittest。
        """
        frame = small_frame(17)
        changed = frame.copy()
        changed.loc[5:, "power_w"] = 500.0
        baseline = np.full(len(frame), 90.0)
        original = simulate_rls(frame, baseline, .88, 1.0, 1.0, .2)
        perturbed = simulate_rls(changed, baseline, .88, 1.0, 1.0, .2)
        np.testing.assert_allclose(original.predictions.predicted_power_w[:10],
                                   perturbed.predictions.predicted_power_w[:10])
        self.assertNotAlmostEqual(original.windows.predicted_energy_wh.iloc[2],
                                  perturbed.windows.predicted_energy_wh.iloc[2])
        self.assertEqual(original.flights.rls_updates.iloc[0], 3)


if __name__ == "__main__":
    unittest.main()
