# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-11-02
# 文件名: config.py
# 功能说明: 集中管理AMOVFLY数据统计项目路径与分析口径
# 版本号：1.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent  # 统计代码与项目文档
DATA_DIR = Path(r"D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset/sourse")  # 原始AMOVFLY数据
SAVE_DIR = PROJECT_ROOT / "out"  # 统计图表与表格输出目录
OUT_DIR = SAVE_DIR
LOG_DIR = OUT_DIR / "logs"
RESULT_DIR = OUT_DIR / "results"
FIGURE_DIR = OUT_DIR / "figures"
SCENES = ("FAFS", "FAVS", "VAFS", "VAVS", "Random")
IQR_FACTOR = 1.5
MAX_GAP_SECONDS = 2.0
EXPECTED_SAMPLE_SECONDS = 0.2


@dataclass(frozen=True)
class AnalysisConfig:
    """功能: 保存AMOVFLY数据统计的输入、输出和异常识别阈值。
    参数: 字段由项目默认路径和常量提供。
    返回: 可复现的分析配置对象。
    调用位置: main.py、analysis.py、visualize.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    out_dir: Path = OUT_DIR
    result_dir: Path = RESULT_DIR
    figure_dir: Path = FIGURE_DIR
    log_dir: Path = LOG_DIR
    scenes: tuple[str, ...] = SCENES
    iqr_factor: float = IQR_FACTOR
    max_gap_seconds: float = MAX_GAP_SECONDS
    expected_sample_seconds: float = EXPECTED_SAMPLE_SECONDS

    @property
    def report_path(self) -> Path:
        """功能: 返回本次统计分析的详细报告路径。
        参数: 无。
        返回: Markdown报告路径。
        调用位置: main.py。
        """
        return self.out_dir / "outreadme.md"

    @property
    def runtime_path(self) -> Path:
        """功能: 返回分析运行环境记录路径。
        参数: 无。
        返回: JSON路径。
        调用位置: main.py。
        """
        return self.out_dir / "runtime.json"


def build_config(**overrides) -> AnalysisConfig:
    """功能: 根据命令行覆盖项构造AMOVFLY统计配置。
    参数: overrides为配置字段覆盖值。
    返回: 分析配置对象。
    调用位置: main.py。
    """
    values = {k: v for k, v in overrides.items() if v is not None}
    for key in ("project_root", "data_dir", "out_dir", "result_dir", "figure_dir", "log_dir"):
        if key in values:
            values[key] = Path(values[key])
    cfg = AnalysisConfig(**values)
    if "out_dir" in values:
        cfg = AnalysisConfig(**{
            **cfg.__dict__, "result_dir": values.get("result_dir", cfg.out_dir / "results"),
            "figure_dir": values.get("figure_dir", cfg.out_dir / "figures"),
            "log_dir": values.get("log_dir", cfg.out_dir / "logs"),
        })
    return cfg


def ensure_directories(cfg: AnalysisConfig) -> None:
    """功能: 创建统计表、日志和图表输出目录。
    参数: cfg为分析配置。
    返回: None。
    调用位置: main.py。
    """
    for path in (cfg.out_dir, cfg.result_dir, cfg.figure_dir, cfg.log_dir):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    config = build_config()
    ensure_directories(config)
    print(f"DATA_DIR={config.data_dir}")
    print(f"OUT_DIR={config.out_dir}")
