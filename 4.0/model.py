# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: model.py
# 开发时间: 2026-09-16
# 文件名: model.py
# 功能说明: 定义实验4.0规划特征TCN和只接收窗口能耗的动态RLS
# 版本号：4.0

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """功能: 构造只使用当前及历史规划状态的因果卷积。
    参数: in_channels、out_channels、kernel_size和dilation为卷积结构参数。
    返回: 因果卷积特征。
    调用位置: TCNBlock。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    """功能: 提取一个规划时间窗中的多尺度时序特征。
    参数: in_channels、out_channels、kernel_size、dilation和dropout为网络参数。
    返回: 残差融合后的时序特征。
    调用位置: PlanningTCN。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels), nn.SiLU(), nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels), nn.SiLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.skip(x))


class PlanningTCN(nn.Module):
    """功能: 用任务前规划状态序列预测当前0.2秒功率。
    参数: input_dim为规划特征数，channels、kernel_size和dropout为网络参数。
    返回: 标准化功率预测值。
    调用位置: train.py、task_api.py。
    """

    def __init__(self, input_dim: int, channels: tuple[int, ...], kernel_size: int, dropout: float) -> None:
        super().__init__()
        layers = []
        last = input_dim
        for index, channel in enumerate(channels):
            layers.append(TCNBlock(last, channel, kernel_size, 2 ** index, dropout))
            last = channel
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(last, last), nn.SiLU(), nn.Linear(last, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(x.transpose(1, 2))[:, :, -1]
        return self.head(encoded).squeeze(-1)


def build_model(input_dim: int, channels: tuple[int, ...], kernel_size: int, dropout: float) -> nn.Module:
    """功能: 构建4.0规划TCN。
    参数: input_dim、channels、kernel_size和dropout为模型结构参数。
    返回: PlanningTCN实例。
    调用位置: train.py、task_api.py。
    """

    return PlanningTCN(input_dim, channels, kernel_size, dropout)


class DynamicEnergyRLS:
    """功能: 只用完整窗口预测能耗和实际能耗更新剩余路线校正状态。
    参数: forgetting_factor为遗忘因子，initial_covariance为初始协方差。
    返回: 通过predict_energy、update_window和interval提供在线状态。
    调用位置: task_api.py、evaluate.py。
    """

    def __init__(self, forgetting_factor: float = 0.96, initial_covariance: float = 0.5) -> None:
        self.forgetting_factor = float(forgetting_factor)
        self.initial_covariance = float(initial_covariance)
        self.theta = np.array([0.0, 1.0], dtype=float)
        self.covariance = np.eye(2, dtype=float) * self.initial_covariance
        self.residual_ema = 0.0
        self.residual_var = 0.0
        self.update_count = 0

    def predict_energy(self, baseline_energy_wh: float) -> float:
        """功能: 用当前偏置和缩放状态修正一个完整时间窗的任务前能耗。
        参数: baseline_energy_wh为任务前模型窗口预测能耗。
        返回: 当前RLS状态下的窗口能耗预测。
        调用位置: update_window、task_api。
        """

        phi = np.array([1.0, float(baseline_energy_wh)], dtype=float)
        return max(0.0, float(phi @ self.theta))

    def update_window(self, baseline_energy_wh: float, actual_energy_wh: float) -> dict:
        """功能: 在窗口结束后使用同一窗口预测和实际能耗更新RLS。
        参数: baseline_energy_wh为窗口开始前的任务前预测，actual_energy_wh为外部计量实际值。
        返回: 更新后的参数、协方差和残差统计。
        调用位置: task_api.OnlineTaskSession.update。
        """

        phi = np.array([1.0, float(baseline_energy_wh)], dtype=float)
        denominator = self.forgetting_factor + phi @ self.covariance @ phi
        gain = self.covariance @ phi / max(float(denominator), 1e-12)
        residual = float(actual_energy_wh) - float(phi @ self.theta)
        self.theta = self.theta + gain * residual
        self.theta[0] = np.clip(self.theta[0], -max(abs(actual_energy_wh), 1e-3), max(abs(actual_energy_wh), 1e-3))
        self.theta[1] = np.clip(self.theta[1], 0.25, 1.75)
        self.covariance = (self.covariance - np.outer(gain, phi) @ self.covariance) / self.forgetting_factor
        self.residual_ema = 0.85 * self.residual_ema + 0.15 * residual
        self.residual_var = 0.85 * self.residual_var + 0.15 * (residual - self.residual_ema) ** 2
        self.update_count += 1
        return self.state()

    def interval(self, baseline_total_wh: float, confidence: float = 0.95, horizon_windows: int = 1) -> tuple[float, float]:
        """功能: 根据参数协方差、最近残差和剩余窗口数计算动态任务级区间。
        参数: baseline_total_wh为剩余路线基线能耗，confidence为置信度，horizon_windows为剩余窗口数。
        返回: 剩余任务能耗下界和上界。
        调用位置: task_api和evaluate。
        """

        z = 1.96 if confidence >= 0.95 else 1.645
        mean = self.predict_energy(baseline_total_wh)
        phi = np.array([1.0, float(baseline_total_wh)], dtype=float)
        parameter_var = max(0.0, float(phi @ self.covariance @ phi))
        residual_var = max(self.residual_var, 1e-8) * max(1.0, math.sqrt(max(horizon_windows, 1)))
        radius = z * math.sqrt(parameter_var + residual_var)
        return max(0.0, mean - radius), max(0.0, mean + radius)

    def state(self) -> dict:
        """功能: 导出在线RLS可序列化状态。
        参数: 无。
        返回: 参数、协方差和残差统计字典。
        调用位置: task_api、evaluate。
        """

        return {"theta_bias": float(self.theta[0]), "theta_scale": float(self.theta[1]),
                "covariance": self.covariance.tolist(), "residual_ema": float(self.residual_ema),
                "residual_std": float(math.sqrt(max(self.residual_var, 0.0))),
                "update_count": int(self.update_count)}
