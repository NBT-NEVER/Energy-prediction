# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_pipeline.py
# 开发时间: 2026-11-01
# 文件名: test_pipeline.py
# 功能说明: 验证SMA任务隔离、历史序列、能量单位及模型特征契约
# 版本号：1.0

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_FEATURE_COLUMNS, build_config, group_paths
from data_utils import prepare_features, feature_columns
from model import build_model
from train import build_sequences, _metrics


def fixture_frame():
    """功能: 构造两个数值差异较大的短任务。
    参数: 无。
    返回: 包含原始10维字段的测试表。
    调用位置: PipelineTests。
    """

    frame = pd.DataFrame({column: [1., 3., 8., 100., 200.] for column in RAW_FEATURE_COLUMNS})
    frame["flight"] = [1, 1, 1, 2, 2]
    frame["time"] = [0., .2, .4, 0., .2]
    frame["dt_seconds"] = .2
    frame["power_w"] = 360.
    frame["energy_interval_wh"] = .02
    return frame


class PipelineTests(unittest.TestCase):
    """功能: 验证预处理、模型与评价的关键行为。
    参数: unittest自动提供。
    返回: 测试结果。
    调用位置: unittest discover。
    """

    def test_sma_prefix_task_boundary_and_future(self):
        """功能: 检查前缀均值、任务边界和未来值隔离。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        frame = fixture_frame()
        ordered = prepare_features(frame, True, 10)
        name = "position_x_m_sma10"
        np.testing.assert_allclose(ordered[name], [1., 2., 4., 100., 150.])
        np.testing.assert_array_equal(ordered.position_x_m, frame.position_x_m)
        frame.loc[2, "position_x_m"] = 99999
        changed = prepare_features(frame, True, 10)
        np.testing.assert_array_equal(ordered.loc[:1, name], changed.loc[:1, name])

    def test_sequence_padding_alignment_and_task_reset(self):
        """功能: 检查窗口补齐与不同任务的历史重置。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        frame = fixture_frame()
        scaler = {"feature_columns": RAW_FEATURE_COLUMNS, "x_mean": [0.] * 10,
                  "x_std": [1.] * 10, "y_mean": 0., "y_std": 1.}
        x, y = build_sequences(frame, scaler, 3)
        np.testing.assert_array_equal(x[2, :, 0], [1., 3., 8.])
        np.testing.assert_array_equal(x[3, :, 0], [100., 100., 100.])
        np.testing.assert_array_equal(y, [360.] * 5)

    def test_labels_do_not_enter_inference_sequences(self):
        """功能: 检查移除标签后推理输入不变。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        frame = fixture_frame()
        scaler = {"feature_columns": RAW_FEATURE_COLUMNS, "x_mean": [0.] * 10,
                  "x_std": [1.] * 10, "y_mean": 0., "y_std": 1.}
        first, _ = build_sequences(frame, scaler, 3, False)
        frame = frame.drop(columns="power_w")
        second, _ = build_sequences(frame, scaler, 3, False)
        np.testing.assert_array_equal(first, second)

    def test_energy_units_and_perfect_prediction(self):
        """功能: 检查功率积分量纲与已知误差数值。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        frame = fixture_frame()
        result = _metrics(frame, frame.power_w.to_numpy(), build_config())
        self.assertEqual(result["flight_energy_wape_percent"], 0.)
        half = _metrics(frame, frame.power_w.to_numpy() / 2, build_config())
        self.assertAlmostEqual(half["flight_energy_wape_percent"], 50.)
        self.assertAlmostEqual(half["flight_energy_mae_wh"], .025)

    def test_model_shape_and_hidden_activation(self):
        """功能: 检查两种特征维度和线性预测头。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        for dim, activation in ((10, "mixed"), (15, "silu")):
            model = build_model(dim, (32, 32), 3, .05, activation).eval()
            self.assertEqual(model(torch.zeros(2, 3, dim)).shape, (2,))
            self.assertIsInstance(model.head[-1], torch.nn.Linear)
        leaky = build_model(15, (32, 32), 3, .05, "leaky_relu").eval()
        self.assertIsInstance(leaky.tcn[0].body[2], torch.nn.LeakyReLU)
        self.assertAlmostEqual(leaky.tcn[0].body[2].negative_slope, .05)
        self.assertIsInstance(leaky.tcn[1].body[6], torch.nn.LeakyReLU)
        self.assertIsInstance(leaky.head[1], torch.nn.LeakyReLU)
        self.assertFalse(any(isinstance(layer, (torch.nn.SiLU, torch.nn.ReLU)) for layer in leaky.modules()))
        for layer in leaky.modules():
            if isinstance(layer, torch.nn.LeakyReLU):
                self.assertAlmostEqual(layer.negative_slope, .05)
        # 主路与跳连各传递同一负数，融合后为两倍输入，检查残差出口斜率。
        for block in leaky.tcn:
            block.body = torch.nn.Identity()
            block.skip = torch.nn.Identity()
            negative = -torch.ones(1, 32, 3)
            torch.testing.assert_close(block(negative), negative * .1)
        self.assertEqual(len(feature_columns(True)), 15)

    def test_model_path_override(self):
        """功能: 检查模型和日志路径随根目录正确派生。
        参数: 无。
        返回: 断言结果。
        调用位置: unittest。
        """
        with tempfile.TemporaryDirectory() as directory:
            cfg = build_config(save_dir=Path(directory), out_dir=Path(directory) / "out")
            paths = group_paths(cfg, "D")
            self.assertEqual(paths["checkpoint"].parent, Path(directory))
            self.assertEqual(paths["training_log"].parent, Path(directory) / "out" / "logs")


if __name__ == "__main__":
    unittest.main()
