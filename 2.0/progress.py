# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: progress.py
# 开发时间: 2026-07-17
# 文件名: progress.py
# 功能说明: 在终端中显示实验2.0的TCN和RLS流程迭代进度
# 版本号：2.0

"""轻量级终端进度条，不依赖第三方库。"""

from __future__ import annotations

import sys
import time


class TerminalProgress:
    """功能: 在单行终端输出中展示已完成比例和阶段详情。
    参数: label 为进度条名称，total 为总步骤数，width 为条形宽度。
    返回: 无，调用 update 和 finish 更新显示。
    调用位置: 数据准备、模型训练、预测、评估和可视化流程。
    """

    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.width = max(int(width), 10)
        self.current = 0
        self._finished = False
        self.started_at = time.perf_counter()
        self.last_render_at = self.started_at
        self._render("")

    def update(self, current: int, detail: str = "") -> None:
        """功能: 更新当前完成数量并刷新终端进度条。
        参数: current 为已完成步骤数，detail 为当前步骤说明。
        返回: 无。
        调用位置: 各处理阶段和批次循环内部。
        """

        self.current = min(max(int(current), 0), self.total)
        self._render(detail)

    def finish(self, detail: str = "完成", completed: bool = True) -> None:
        """功能: 将进度条设为完成状态并换行结束输出。
        参数: detail 为完成后的补充说明，completed 表示是否正常完成全部步骤。
        返回: 无。
        调用位置: 各流程的最后一步。
        """

        if self._finished:
            return
        if completed:
            self.current = self.total
        self._render(detail)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._finished = True

    def _render(self, detail: str) -> None:
        ratio = self.current / self.total
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        rate = self.current / elapsed
        remaining = (self.total - self.current) / rate if rate > 1e-9 else float("inf")
        eta = "--:--" if not remaining < 86400 else time.strftime("%M:%S", time.gmtime(max(remaining, 0)))
        elapsed_text = time.strftime("%M:%S", time.gmtime(elapsed))
        suffix = f" | {detail}" if detail else ""
        sys.stdout.write(
            f"\r{self.label} [{bar}] {ratio:>6.1%} ({self.current}/{self.total}) "
            f"耗时 {elapsed_text} ETA {eta}{suffix}"
        )
        sys.stdout.flush()
