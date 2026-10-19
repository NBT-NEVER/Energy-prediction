# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: terminal_logger.py
# 开发时间: 2026-09-16
# 文件名: terminal_logger.py
# 功能说明: 将实验4.1的终端标准输出和异常信息同步追加到独立日志文件
# 版本号：4.1

from __future__ import annotations

import sys
import traceback as traceback_module
from datetime import datetime
from pathlib import Path
from typing import TextIO


_ACTIVE_CAPTURE: "TerminalLogCapture | None" = None


def log_result(title: str, payload: object) -> None:
    """功能: 将阶段结果写入当前运行日志，不写入过程进度。
    参数: title为结果标题，payload为字符串、字典或其他结果内容。
    返回: None。
    调用位置: main.py的阶段摘要输出。
    """

    if _ACTIVE_CAPTURE is not None:
        _ACTIVE_CAPTURE.write_result(title, payload)


class TeeStream:
    """功能: 将终端字符流同时写入原终端和UTF-8日志文件。
    参数: terminal为原始字符流，log_file为可选日志文件，exclude_progress控制是否排除动态进度刷新。
    返回: 兼容标准输出流的代理对象。
    调用位置: TerminalLogCapture。
    """

    def __init__(self, terminal: TextIO, log_file: TextIO | None = None, exclude_progress: bool = True) -> None:
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
        if self.log_file is not None:
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
        if self.log_file is not None:
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding


class TerminalLogCapture:
    """功能: 在一次程序运行期间保留终端输出并单独保存阶段结果日志。
    参数: log_path为终端日志保存路径，mode为当前命令行运行模式，reset_log控制本次运行前是否清空旧日志。
    返回: 上下文管理器对象。
    调用位置: main。
    """

    def __init__(self, log_path: Path, mode: str, reset_log: bool = False) -> None:
        self.log_path = Path(log_path)
        self.mode = mode
        self.reset_log = reset_log
        self.log_file: TextIO | None = None
        self.original_stdout: TextIO | None = None
        self.original_stderr: TextIO | None = None
        self.failed = False
        self.started_at = 0.0

    def __enter__(self) -> "TerminalLogCapture":
        """功能: 打开日志文件并接管标准输出和标准错误。
        参数: 无。
        返回: 当前上下文管理器。
        调用位置: main。
        """

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "w" if self.reset_log else "a"
        self.log_file = self.log_path.open(file_mode, encoding="utf-8", newline="")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeStream(self.original_stdout, self.log_file)
        sys.stderr = TeeStream(self.original_stderr, self.log_file)
        global _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = self
        self.started_at = datetime.now().timestamp()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        log_action = "清空后记录" if self.reset_log else "追加记录"
        print(f"\n{'=' * 72}\n运行开始: {started_at} | 模式: {self.mode} | 日志: {log_action}\n结果日志: {self.log_path}")
        return self

    def write_result(self, title: str, payload: object) -> None:
        """功能: 将摘要结果按可读文本写入日志文件。
        参数: title为结果标题，payload为待记录的结果内容。
        返回: None。
        调用位置: log_result。
        """

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
        """功能: 记录运行结束状态并恢复原始输出流。
        参数: exc_type、exc_value和traceback为上下文中的异常信息。
        返回: False，使异常在记录后继续向上传播并产生非零退出状态。
        调用位置: main。
        """

        if exc_type is not None:
            self.failed = True
            traceback_module.print_exception(exc_type, exc_value, traceback, file=sys.stderr)
            log_result("异常结果", {"exception_type": exc_type.__name__, "message": str(exc_value)})
        status = "失败" if exc_type else "完成"
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = max(datetime.now().timestamp() - self.started_at, 0.0)
        log_result("运行结果", {"finished_at": finished_at, "status": status, "elapsed_seconds": round(elapsed, 3)})
        print(f"运行结束: {finished_at} | 状态: {status} | 总耗时: {elapsed:.2f}s\n{'=' * 72}")
        sys.stdout = self.original_stdout or sys.__stdout__
        sys.stderr = self.original_stderr or sys.__stderr__
        global _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = None
        if self.log_file is not None:
            self.log_file.close()
        return False
