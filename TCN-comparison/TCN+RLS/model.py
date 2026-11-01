# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: model.py
# 开发时间: 2026-10-30
# 文件名: model.py
# 功能说明: 定义四组离线功率预测使用的因果TCN和激活函数变体
# 版本号：1.0

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """功能: 构造只访问当前及历史时刻的因果一维卷积。
    参数: in_channels、out_channels、kernel_size和dilation为卷积参数。
    返回: 因果卷积特征。
    调用位置: TCNBlock。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        """功能: 初始化左侧补齐卷积。
        参数: 输入输出通道、卷积核和膨胀率。
        返回: None。
        调用位置: TCNBlock初始化。
        """

        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能: 对输入左侧补齐后执行卷积。
        参数: x为批次、通道、时间张量。
        返回: 相同时间长度的卷积输出。
        调用位置: TCNBlock.forward。
        """

        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    """功能: 提取一个规划时间窗的多尺度时序特征。
    参数: in_channels、out_channels、kernel_size、dilation、dropout和activation为网络参数。
    返回: 残差融合后的时序特征。
    调用位置: PlanningTCN。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float, activation: str) -> None:
        """功能: 初始化双卷积主路与残差跳连。
        参数: 通道、卷积核、膨胀率、丢弃率和出口激活模式。
        返回: None。
        调用位置: PlanningTCN初始化。
        """

        super().__init__()
        self.activation_name = activation
        hidden_activation = nn.SiLU()
        self.body = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels), hidden_activation, nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels), nn.SiLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能: 融合主路与跳连并应用指定出口激活。
        参数: x为历史窗口张量。
        返回: 残差编码。
        调用位置: PlanningTCN.forward。
        """

        residual = self.body(x) + self.skip(x)
        return F.silu(residual) if self.activation_name == "silu" else torch.relu(residual)


class PlanningTCN(nn.Module):
    """功能: 根据历史规划特征预测当前离散时刻的标准化功率。
    参数: input_dim、channels、kernel_size、dropout和activation为模型参数。
    返回: 标准化功率预测值。
    调用位置: train.py、evaluate.py。
    """

    def __init__(self, input_dim: int, channels: tuple[int, ...], kernel_size: int,
                 dropout: float, activation: str) -> None:
        """功能: 初始化残差块序列与线性功率回归头。
        参数: 输入维度、通道、卷积核、丢弃率和激活模式。
        返回: None。
        调用位置: build_model。
        """

        super().__init__()
        layers = []
        last = input_dim
        for index, channel in enumerate(channels):
            layers.append(TCNBlock(last, channel, kernel_size, 2 ** index, dropout, activation))
            last = channel
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(last, last), nn.SiLU(), nn.Linear(last, 1))
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能: 编码历史窗口并预测窗口末端功率。
        参数: x为批次、时间、特征张量。
        返回: 每个样本的标准化功率。
        调用位置: train.py训练与预测。
        """

        encoded = self.tcn(x.transpose(1, 2))[:, :, -1]
        return self.head(encoded).squeeze(-1)


def build_model(input_dim: int, channels: tuple[int, ...], kernel_size: int,
                dropout: float, activation: str) -> nn.Module:
    """功能: 构建指定输入维度和激活函数的PlanningTCN。
    参数: input_dim、channels、kernel_size、dropout和activation为结构参数。
    返回: PlanningTCN实例。
    调用位置: train.py、evaluate.py。
    """

    if activation not in {"mixed", "silu"}:
        raise ValueError(f"不支持的激活模式: {activation}")
    return PlanningTCN(input_dim, channels, kernel_size, dropout, activation)
