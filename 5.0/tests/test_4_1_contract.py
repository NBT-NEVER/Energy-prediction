# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: test_4_1_contract.py
# 开发时间: 2026-09-18
# 文件名: test_4_1_contract.py
# 功能说明: 验证实验4.1输入契约、加工数据约束和模型产物字段一致性
# 版本号：4.1

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch


VERSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_DIR))

from config import build_config  # noqa: E402
from data_utils import FEATURE_COLUMNS, validate_processed_data  # noqa: E402
from main import _demo_task  # noqa: E402
from task_api import REQUIRED_PLANNING_COLUMNS, validate_task_input  # noqa: E402


EXPECTED_FEATURES = [
    "time_s",
    "task_duration_s",
    "planned_position_east_m",
    "planned_position_north_m",
    "planned_position_up_m",
    "wind_east_mps",
    "wind_north_mps",
    "payload_g",
    "planned_motor_on",
    "planned_airborne",
]


class ContractTests(unittest.TestCase):
    """功能: 集中验证4.1数据、接口、scaler和checkpoint契约。
    参数: unittest自动注入。
    返回: unittest断言结果。
    调用位置: unittest discover。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = build_config()

    def test_feature_order_is_exact(self) -> None:
        self.assertEqual(FEATURE_COLUMNS, EXPECTED_FEATURES)
        self.assertEqual(REQUIRED_PLANNING_COLUMNS, ["task_id", *EXPECTED_FEATURES])
        self.assertNotIn("route", FEATURE_COLUMNS)

    def test_demo_uses_only_interface_columns(self) -> None:
        task = _demo_task(self.cfg)
        self.assertEqual(task.columns.tolist(), REQUIRED_PLANNING_COLUMNS)
        validate_task_input(task, self.cfg)

    def test_binary_and_logical_constraints_are_enforced(self) -> None:
        task = _demo_task(self.cfg)
        task.loc[0, "planned_airborne"] = 1
        task.loc[0, "planned_motor_on"] = 0
        with self.assertRaisesRegex(ValueError, "planned_airborne"):
            validate_task_input(task, self.cfg)

    def test_forbidden_observation_is_rejected(self) -> None:
        task = _demo_task(self.cfg)
        task["battery_voltage"] = 24.0
        with self.assertRaisesRegex(ValueError, "禁止"):
            validate_task_input(task, self.cfg)

    def test_processed_data_consistency(self) -> None:
        frame = pd.read_csv(self.cfg.clean_data_csv)
        checks = validate_processed_data(frame, self.cfg)
        self.assertTrue(checks["finite_values"])
        self.assertTrue(checks["airborne_le_motor_on"])
        self.assertFalse(checks["route_enters_model"])

    def test_artifact_feature_contract(self) -> None:
        metadata = json.loads(self.cfg.feature_meta_json.read_text(encoding="utf-8"))
        scaler = json.loads(self.cfg.scaler_json.read_text(encoding="utf-8"))
        checkpoint = torch.load(self.cfg.final_model_file, map_location="cpu", weights_only=False)
        self.assertEqual(metadata["feature_columns"], EXPECTED_FEATURES)
        self.assertEqual(scaler["feature_columns"], EXPECTED_FEATURES)
        self.assertEqual(checkpoint["feature_columns"], EXPECTED_FEATURES)
        self.assertEqual(scaler["target_column"], "power_w")
        self.assertEqual(checkpoint["target_column"], "power_w")


if __name__ == "__main__":
    unittest.main()
