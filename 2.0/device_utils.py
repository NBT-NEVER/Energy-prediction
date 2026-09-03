# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: device_utils.py
# 开发时间: 2026-07-17
# 文件名: device_utils.py
# 功能说明: 统一校验并选择实验2.1的TCN和RLS CUDA训练推理设备
# 版本号：2.1

"""GPU 设备选择与显示工具。"""

from __future__ import annotations

import torch


def select_cuda_device(device_name: str) -> torch.device:
    """功能: 校验 CUDA 可用性并返回指定的 GPU 设备。
    参数: device_name 为命令行指定的 CUDA 设备名称或 auto。
    返回: 可用于模型训练和推理的 CUDA 设备对象。
    调用位置: train.py、predict.py 和 visualize.py。
    """

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA GPU，训练和模型推理不能继续执行。")
    device = torch.device("cuda" if device_name == "auto" else device_name)
    if device.type != "cuda":
        raise ValueError(f"仅支持 CUDA 设备，当前设备设置为: {device_name}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA 设备编号超出范围: {device_name}")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    return device


def describe_cuda_device(device: torch.device) -> str:
    """功能: 返回 CUDA 设备编号和名称，供终端日志显示。
    参数: device 为已校验的 CUDA 设备对象。
    返回: GPU 设备描述字符串。
    调用位置: train.py、predict.py 和 visualize.py。
    """

    device_index = torch.cuda.current_device() if device.index is None else device.index
    return f"{device} ({torch.cuda.get_device_name(device_index)})"
