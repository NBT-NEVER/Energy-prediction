# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: model.py
# 开发时间: 2026-07-07
# 文件名: model.py
# 功能说明: 定义四轴无人机能耗预测实验1.0的自建MLP回归模型
# 版本号：1.0

from __future__ import annotations

import torch
from torch import nn


class EnergyMLP(nn.Module):
    """功能: 使用多层感知机拟合飞行工况到电功率的非线性映射。
    参数: input_dim为输入特征数，hidden_dims为隐藏层宽度，dropout为丢弃率。
    返回: 前向传播输出标准化后的功率预测值。
    调用位置: train.py、predict.py、evaluate.py。
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64, 32), dropout: float = 0.08) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能: 执行一次模型前向计算。
        参数: features为标准化后的输入特征张量。
        返回: 标准化后的功率预测张量。
        调用位置: train.py、predict.py、evaluate.py。
        """

        return self.network(features).squeeze(-1)


class ResidualBlock(nn.Module):
    """功能: 构建带投影捷径的残差全连接块。
    参数: input_dim为输入维度，output_dim为输出维度，dropout为丢弃率。
    返回: 前向传播输出张量。
    调用位置: ResidualEnergyMLP。
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.shortcut = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能: 执行残差块前向计算。
        参数: features为输入特征张量。
        返回: 残差融合后的特征张量。
        调用位置: ResidualEnergyMLP.forward。
        """

        return self.activation(self.main(features) + self.shortcut(features))


class ResidualEnergyMLP(nn.Module):
    """功能: 使用残差全连接网络拟合复杂飞行工况和电功率之间的映射。
    参数: input_dim为输入特征数，hidden_dims为残差块宽度序列，dropout为丢弃率。
    返回: 前向传播输出标准化后的功率预测值。
    调用位置: train.py、predict.py、evaluate.py。
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (192, 128, 64), dropout: float = 0.06) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(last_dim, hidden_dim, dropout))
            last_dim = hidden_dim
        layers.extend([nn.Dropout(dropout), nn.Linear(last_dim, 1)])
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """功能: 执行残差模型前向计算。
        参数: features为标准化后的输入特征张量。
        返回: 标准化后的功率预测张量。
        调用位置: train.py、predict.py、evaluate.py。
        """

        return self.network(features).squeeze(-1)


def build_model(input_dim: int, hidden_dims: tuple[int, ...], dropout: float, model_type: str = "plain") -> nn.Module:
    """功能: 根据模型类型构建能耗预测模型实例。
    参数: input_dim为输入特征数，hidden_dims为隐藏层宽度，dropout为丢弃率，model_type为模型类型。
    返回: EnergyMLP或ResidualEnergyMLP模型。
    调用位置: train.py、predict.py、evaluate.py。
    """

    if model_type == "residual":
        return ResidualEnergyMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    if model_type == "plain":
        return EnergyMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"不支持的模型类型: {model_type}")
