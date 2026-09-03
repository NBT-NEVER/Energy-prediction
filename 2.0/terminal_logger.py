# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: terminal_logger.py
# 开发时间: 2026-09-03
# 文件名: terminal_logger.py
# 功能说明: 将实验2.0的终端标准输出和异常信息同步追加到日志文件
# 版本号：2.0

from __future__ import annotations

import sys
import traceback as traceback_module
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TeeStream:
    """功能: 将终端字符流同时写入原终端和UTF-8日志文件。
    参数: terminal为原始字符流，log_file为日志文件，exclude_progress控制是否排除动态进度刷新。
    返回: 兼容标准输出流的代理对象。
    调用位置: TerminalLogCapture。
    """

    def __init__(self, terminal: TextIO, log_file: TextIO, exclude_progress: bool = True) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.exclude_progress = exclude_progress
        self._skip_progress_newline = False

    def write(self, message: str) -> int:
        """功能: 同步写入终端和日志文件。
        参数: message为待输出文本。
        返回: 原始消息的字符数。
        调用位置: print、进度条和异常输出。
        """

        self.terminal.write(message)
        if self.exclude_progress and message.startswith("\r"):
            self._skip_progress_newline = True
            return len(message)
        if self._skip_progress_newline and message in {"\n", "\r\n"}:
            self._skip_progress_newline = False
            return len(message)
        self._skip_progress_newline = False
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self) -> None:
        """功能: 刷新终端和日志缓冲区。
        参数: 无。
        返回: None。
        调用位置: print、进度条及Python运行时。
        """

        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding


class TerminalLogCapture:
    """功能: 在一次程序运行期间捕获标准输出和标准错误并追加到日志。
    参数: log_path为终端日志保存路径，mode为当前命令行运行模式。
    返回: 上下文管理器对象。
    调用位置: main。
    """

    def __init__(self, log_path: Path, mode: str) -> None:
        self.log_path = Path(log_path)
        self.mode = mode
        self.log_file: TextIO | None = None
        self.original_stdout: TextIO | None = None
        self.original_stderr: TextIO | None = None
        self.failed = False

    def __enter__(self) -> "TerminalLogCapture":
        """功能: 打开日志文件并接管标准输出和标准错误。
        参数: 无。
        返回: 当前上下文管理器。
        调用位置: main。
        """

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("a", encoding="utf-8", newline="")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeStream(self.original_stdout, self.log_file)
        sys.stderr = TeeStream(self.original_stderr, self.log_file)
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"\n{'=' * 72}\n终端记录开始: {started_at} | 模式: {self.mode}\n日志文件: {self.log_path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """功能: 记录运行结束状态并恢复原始输出流。
        参数: exc_type、exc_value和traceback为上下文中的异常信息。
        返回: 发生异常时返回True，回溯已记录并由main设置非零退出状态。
        调用位置: main。
        """

        if exc_type is not None:
            self.failed = True
            traceback_module.print_exception(exc_type, exc_value, traceback, file=sys.stderr)
        status = "失败" if exc_type else "完成"
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"终端记录结束: {finished_at} | 状态: {status}\n{'=' * 72}")
        sys.stdout = self.original_stdout or sys.__stdout__
        sys.stderr = self.original_stderr or sys.__stderr__
        if self.log_file is not None:
            self.log_file.close()
        return exc_type is not None
