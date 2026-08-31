# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: model.py
# 开发时间: 2026-08-31
# 文件名: model.py
# 功能说明: 定义实验2.0的因果TCN瞬时功率预测模型
# 版本号：2.0

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """功能: 构造只读取当前及历史样本的一维因果卷积。
    参数: in_channels为输入通道数，out_channels为输出通道数，kernel_size为卷积核宽度，dilation为膨胀率。
    返回: 左侧补零后的卷积特征。
    调用位置: TCNBlock。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能: 对时间序列执行因果卷积。
        参数: x为形状(batch, channels, time)的特征张量。
        返回: 保持时间长度且不使用未来信息的卷积结果。
        调用位置: TCNBlock.forward。
        """

        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    """功能: 堆叠两层因果卷积、归一化、激活和Dropout。
    参数: in_channels为输入通道数，out_channels为输出通道数，kernel_size为卷积核宽度，dilation为膨胀率，dropout为丢弃率。
    返回: TCN时间特征。
    调用位置: TemporalConvNet。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """功能: 提取一个时间尺度上的局部和膨胀时序特征。
        参数: x为时间序列特征张量。
        返回: 残差融合后的时间特征。
        调用位置: TemporalConvNet.forward。
        """

        return torch.relu(self.net(x) + self.shortcut(x))


class TemporalConvNet(nn.Module):
    """功能: 使用多层因果TCN从短时间窗预测当前时刻标准化功率。
    参数: input_dim为每个采样点的特征数，channels为卷积通道序列，kernel_size为卷积核宽度，dropout为丢弃率。
    返回: 当前窗口末端的标准化功率预测值。
    调用位置: train.py、predict.py、evaluate.py。
    """

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (64, 64, 32), kernel_size: int = 3, dropout: float = 0.08) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for index, channel in enumerate(channels):
            layers.append(TCNBlock(last, channel, kernel_size, 2**index, dropout))
            last = channel
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(last, last // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(last // 2, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能: 执行一批短时间窗的TCN前向推理。
        参数: features为形状(batch, time, feature)的标准化序列。
        返回: 窗口末端的标准化功率预测张量。
        调用位置: train.py、predict.py、evaluate.py。
        """

        encoded = self.network(features.transpose(1, 2))
        return self.head(encoded[:, :, -1]).squeeze(-1)


def build_model(input_dim: int, channels: tuple[int, ...], dropout: float, kernel_size: int = 3) -> nn.Module:
    """功能: 构建实验2.0的TCN模型。
    参数: input_dim为输入特征数，channels为TCN通道宽度，dropout为丢弃率，kernel_size为卷积核宽度。
    返回: TemporalConvNet模型实例。
    调用位置: train.py、predict.py。
    """

    return TemporalConvNet(input_dim=input_dim, channels=channels, kernel_size=kernel_size, dropout=dropout)


class RLSCorrector:
    """功能: 用递推最小二乘在线估计TCN功率残差的仿射校正参数。
    参数: forgetting_factor为遗忘因子，initial_covariance为初始协方差大小，power_scale为功率归一化尺度。
    返回: 无，调用predict和update获得校正值并更新状态。
    调用位置: train.py、predict.py、evaluate.py。
    """

    def __init__(self, forgetting_factor: float = 0.995, initial_covariance: float = 1000.0, power_scale: float = 100.0) -> None:
        self.forgetting_factor = float(forgetting_factor)
        self.initial_covariance = float(initial_covariance)
        self.power_scale = max(float(power_scale), 1.0)
        self.bias_limit = self.power_scale
        self.theta = torch.zeros(2, dtype=torch.float64)
        self.covariance = torch.eye(2, dtype=torch.float64) * self.initial_covariance

    def reset_covariance(self) -> None:
        """功能: 为一段新的独立飞行重置参数协方差。
        参数: 无。
        返回: None。
        调用位置: apply_rls_correction。
        """

        self.covariance = torch.eye(2, dtype=torch.float64) * self.initial_covariance

    def predict(self, base_power: float) -> float:
        """功能: 根据当前TCN功率和历史校正参数给出实时校正功率。
        参数: base_power为TCN前向推理得到的功率。
        返回: RLS校正后的功率。
        调用位置: apply_rls_correction。
        """

        phi = torch.tensor([1.0, float(base_power) / self.power_scale], dtype=torch.float64)
        correction = float(torch.dot(phi, self.theta).clamp(-0.25 * self.power_scale, 0.25 * self.power_scale))
        return max(float(base_power) + correction, 0.0)

    def update(self, base_power: float, actual_power: float) -> None:
        """功能: 用当前已观测功率更新递推最小二乘状态。
        参数: base_power为TCN功率，actual_power为当前采样点实测功率。
        返回: None。
        调用位置: apply_rls_correction。
        """

        phi = torch.tensor([1.0, float(base_power) / self.power_scale], dtype=torch.float64)
        gain_denominator = self.forgetting_factor + phi @ self.covariance @ phi
        gain = self.covariance @ phi / max(float(gain_denominator), 1e-12)
        residual = float(actual_power) - float(base_power) - float(phi @ self.theta)
        self.theta = self.theta + gain * residual
        self.theta[0] = self.theta[0].clamp(-self.bias_limit, self.bias_limit)
        self.theta[1] = self.theta[1].clamp(-1.0, 1.0)
        self.covariance = (self.covariance - torch.outer(gain, phi) @ self.covariance) / self.forgetting_factor
