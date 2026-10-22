# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: terminal_logger.py
# 开发时间: 2026-10-20
# 文件名: terminal_logger.py
# 功能说明: 将实验5.0终端标准输出和异常信息同步记录到独立日志文件
# 版本号：5.0

from __future__ import annotations

import sys
import traceback as traceback_module
from datetime import datetime
from pathlib import Path
from typing import TextIO


_ACTIVE_CAPTURE: "TerminalLogCapture | None" = None


def log_result(title: str, payload: object) -> None:
    """功能: 将阶段摘要写入当前终端日志。
    参数: title为摘要标题，payload为字符串、字典或其他结果。
    返回: None。
    调用位置: main.py和vehicle_pipeline.py。
    """
    if _ACTIVE_CAPTURE is not None:
        _ACTIVE_CAPTURE.write_result(title, payload)


class TeeStream:
    """功能: 将标准输出和标准错误同时写入原终端与UTF-8日志。
    参数: terminal为原始流，log_file为日志文件，exclude_progress是否过滤回车进度刷新。
    返回: 兼容标准流的代理对象。
    调用位置: TerminalLogCapture。
    """

    def __init__(self, terminal: TextIO, log_file: TextIO | None) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.skip_progress_newline = False

    def write(self, message: str) -> int:
        self.terminal.write(message)
        if message.startswith("\r"):
            self.skip_progress_newline = True
            return len(message)
        if self.skip_progress_newline and message in {"\n", "\r\n"}:
            self.skip_progress_newline = False
            return len(message)
        self.skip_progress_newline = False
        if self.log_file is not None:
            self.log_file.write(message)
            self.log_file.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        if self.log_file is not None:
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding


class TerminalLogCapture:
    """功能: 捕获单次运行的终端输出并追加阶段结果和异常信息。
    参数: log_path为日志路径，mode为运行模式，reset_log是否清空旧日志。
    返回: 上下文管理器对象。
    调用位置: main.py。
    """

    def __init__(self, log_path: Path, mode: str, reset_log: bool = False) -> None:
        self.log_path = Path(log_path)
        self.mode = mode
        self.reset_log = reset_log
        self.log_file: TextIO | None = None
        self.original_stdout: TextIO | None = None
        self.original_stderr: TextIO | None = None
        self.started_at = 0.0

    def __enter__(self) -> "TerminalLogCapture":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w" if self.reset_log else "a", encoding="utf-8", newline="")
        self.original_stdout, self.original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(self.original_stdout, self.log_file)
        sys.stderr = TeeStream(self.original_stderr, self.log_file)
        global _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = self
        self.started_at = datetime.now().timestamp()
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        action = "清空后记录" if self.reset_log else "追加记录"
        print(f"\n{'=' * 72}\n运行开始: {started} | 模式: {self.mode} | 日志: {action}\n结果日志: {self.log_path}")
        return self

    def write_result(self, title: str, payload: object) -> None:
        if self.log_file is None:
            return
        self.log_file.write(f"\n[{title}]\n")
        if isinstance(payload, dict):
            for key, value in payload.items():
                self.log_file.write(f"{key}: {value}\n")
        else:
            self.log_file.write(f"{payload}\n")
        self.log_file.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            traceback_module.print_exception(exc_type, exc_value, traceback, file=sys.stderr)
            log_result("异常结果", {"exception_type": exc_type.__name__, "message": str(exc_value)})
        status = "失败" if exc_type else "完成"
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = max(datetime.now().timestamp() - self.started_at, 0.0)
        log_result("运行结果", {"finished_at": finished, "status": status, "elapsed_seconds": round(elapsed, 3)})
        print(f"运行结束: {finished} | 状态: {status} | 总耗时: {elapsed:.2f}s\n{'=' * 72}")
        sys.stdout = self.original_stdout or sys.__stdout__
        sys.stderr = self.original_stderr or sys.__stderr__
        global _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = None
        if self.log_file is not None:
            self.log_file.close()
        return False
