# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-10-30
# 文件名: config.py
# 功能说明: 集中管理五组TCN加RLS及置信区间实验路径和参数
# 版本号：1.0

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:/Python-files/Energy-prediction/data/dji_matrice_100_data")
EXPERIMENT_DATA_DIR = DATA_ROOT / "TCN-planned"
DATA_DIR = EXPERIMENT_DATA_DIR
SOURCE_DATA_DIR = EXPERIMENT_DATA_DIR
SAVE_DIR = Path(r"D:/Python-files/Energy-prediction/model/TCN+RLS-DJI-planned")
TCN_SOURCE_MODEL_DIR = Path(r"D:/Python-files/Energy-prediction/model/TCN-planned")
OUT_DIR = PROJECT_ROOT / "out"
MODEL_DIR = SAVE_DIR
LOG_DIR = OUT_DIR / "logs"
FIGURE_DIR = OUT_DIR / "figures"
RESULT_DIR = OUT_DIR / "results"

RAW_FEATURE_COLUMNS = [
    "position_x_m", "position_y_m", "position_z_m", "wind_speed", "wind_angle",
    "time_s", "task_duration_s", "payload_g", "uav_type", "planned",
]
SMA_FEATURE_COLUMNS = [
    "position_x_m", "position_y_m", "position_z_m", "wind_speed", "wind_angle",
]
SCHEMA_VERSION = "DJI-TCN-planned-1.0"
TARGET_COLUMN = "power_w"
GROUPS = {
    "A": {"name": "原始10维_混合激活", "use_sma": False, "activation": "mixed"},
    "B": {"name": "原始10维_统一SiLU", "use_sma": False, "activation": "silu"},
    "C": {"name": "原始10维加5维SMA_混合激活", "use_sma": True, "activation": "mixed"},
    "D": {"name": "原始10维加5维SMA_统一SiLU", "use_sma": True, "activation": "silu"},
    "E": {"name": "原始10维加5维SMA_统一LeakyReLU(0.05)", "use_sma": True, "activation": "leaky_relu"},
}

RESAMPLE_SECONDS = 0.2
SMA_WINDOW_STEPS = 10
RLS_FORGETTING_FACTORS = (0.85, 0.88, 0.90, 0.93, 0.96, 0.98, 0.99)
RLS_INITIAL_COVARIANCES = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
RLS_WINDOW_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0)
CONFIDENCE_LEVEL = 0.95
BATCH_SIZE = 2048
RANDOM_SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存TCN加RLS实验的路径和超参数。
    参数: 字段由默认常量或命令行覆盖项提供。
    返回: 实验配置对象。
    调用位置: main.py、rls_evaluate.py、report.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    source_data_dir: Path = SOURCE_DATA_DIR
    source_processed_dir: Path = SOURCE_DATA_DIR
    save_dir: Path = SAVE_DIR
    tcn_source_model_dir: Path = TCN_SOURCE_MODEL_DIR
    out_dir: Path = OUT_DIR
    model_dir: Path = MODEL_DIR
    log_dir: Path = LOG_DIR
    figure_dir: Path = FIGURE_DIR
    result_dir: Path = RESULT_DIR
    resample_seconds: float = RESAMPLE_SECONDS
    sma_window_steps: int = SMA_WINDOW_STEPS
    rls_forgetting_factors: tuple[float, ...] = RLS_FORGETTING_FACTORS
    rls_initial_covariances: tuple[float, ...] = RLS_INITIAL_COVARIANCES
    rls_window_seconds: tuple[float, ...] = RLS_WINDOW_SECONDS
    confidence_level: float = CONFIDENCE_LEVEL
    batch_size: int = BATCH_SIZE
    random_seed: int = RANDOM_SEED
    device: str = "cuda"

    @property
    def summary_csv(self) -> Path:
        """功能: 返回五组RLS汇总表路径。
        参数: 无。
        返回: CSV路径。
        调用位置: rls_evaluate.py、report.py。
        """
        return self.result_dir / "TCN_RLS_all_metrics.csv"

    @property
    def manifest_json(self) -> Path:
        """功能: 返回数据指纹和规模记录路径。
        参数: 无。
        返回: JSON路径。
        调用位置: main.py、report.py。
        """
        return self.data_dir / "dataset_summary_TCN_RLS.json"

    @property
    def source_manifest_json(self) -> Path:
        """功能: 返回冻结TCN的数据清单路径。
        参数: 无。
        返回: JSON路径。
        调用位置: rls_evaluate.py、audit.py。
        """
        return self.source_data_dir / "dataset_summary_TCN-comparison.json"

    @property
    def audit_json(self) -> Path:
        """功能: 返回本次产物审计路径。
        参数: 无。
        返回: JSON路径。
        调用位置: audit.py、report.py。
        """
        return self.log_dir / "planned_artifact_audit.json"

    @property
    def runtime_json(self) -> Path:
        """功能: 返回运行环境快照路径。
        参数: 无。
        返回: JSON路径。
        调用位置: main.py。
        """
        return self.out_dir / "TCN_RLS_runtime.json"

    @property
    def terminal_log(self) -> Path:
        """功能: 返回终端日志路径。
        参数: 无。
        返回: UTF-8日志路径。
        调用位置: main.py。
        """
        return self.log_dir / "TCN_RLS_terminal.log"

    @property
    def report_md(self) -> Path:
        """功能: 返回详细输出报告路径。
        参数: 无。
        返回: Markdown路径。
        调用位置: report.py。
        """
        return self.out_dir / "outreadme.md"

    @property
    def verification_log(self) -> Path:
        """功能: 返回交付校验日志路径。
        参数: 无。
        返回: UTF-8日志路径。
        调用位置: 交付校验命令。
        """
        return self.log_dir / "TCN_RLS_verification.log"


def group_paths(cfg: ExperimentConfig, group: str) -> dict[str, Path]:
    """功能: 返回单组RLS模型、调参、预测、指标和区间路径。
    参数: cfg为配置，group为A、B、C、D或E。
    返回: 统一命名的路径字典。
    调用位置: main.py、rls_evaluate.py、report.py。
    """
    key = str(group).upper()
    prefix = f"TCN_RLS_{key}"
    return {
        "tcn_checkpoint": cfg.model_dir / f"{prefix}_tcn.pt",
        "tcn_source_checkpoint": cfg.tcn_source_model_dir / f"TCN-comparison_{key}_best.pt",
        "scaler": cfg.model_dir / f"{prefix}_scaler.json",
        "rls_model": cfg.model_dir / f"{prefix}_rls.json",
        "stage_intervals": cfg.result_dir / f"{prefix}_stage_intervals.csv",
        "rls_tuning": cfg.result_dir / f"{prefix}_rls_tuning.csv",
        "interval_calibration": cfg.result_dir / f"{prefix}_interval_calibration.json",
        "validation_trace": cfg.result_dir / f"{prefix}_validation_trace.csv",
        "test_predictions": cfg.result_dir / f"{prefix}_test_predictions.csv",
        "test_trace": cfg.result_dir / f"{prefix}_test_trace.csv",
        "flight_metrics": cfg.result_dir / f"{prefix}_flight_metrics.csv",
        "window_metrics": cfg.result_dir / f"{prefix}_window_metrics.csv",
        "metrics": cfg.result_dir / f"{prefix}_metrics.json",
        "tuning_log": cfg.log_dir / f"{prefix}_rls_tuning.log",
    }


def table_path(cfg: ExperimentConfig, split: str) -> Path:
    """功能: 集中返回训练、验证和测试加工表路径。
    参数: cfg为配置，split为train、val或test。
    返回: CSV路径。
    调用位置: data_utils.py、rls_evaluate.py。
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"不支持的数据划分: {split}")
    return cfg.data_dir / f"{split}_TCN-comparison.csv"


def figure_path(cfg: ExperimentConfig, name: str, suffix: str = "png") -> Path:
    """功能: 集中生成图表路径。
    参数: cfg为配置，name为图表名称，suffix为文件后缀。
    返回: 图表路径。
    调用位置: visualize_rls.py、report.py。
    """
    return cfg.figure_dir / f"TCN_RLS_{name}.{suffix}"


def ensure_directories(cfg: ExperimentConfig) -> None:
    """功能: 创建数据、模型、日志、结果和图表目录。
    参数: cfg为实验配置。
    返回: None。
    调用位置: main.py、rls_evaluate.py。
    """
    for path in (cfg.data_dir, cfg.save_dir, cfg.out_dir, cfg.model_dir,
                 cfg.log_dir, cfg.figure_dir, cfg.result_dir):
        path.mkdir(parents=True, exist_ok=True)


def build_config(**overrides) -> ExperimentConfig:
    """功能: 根据命令行覆盖项构造实验配置。
    参数: overrides为字段名和值组成的字典。
    返回: 实验配置对象。
    调用位置: main.py。
    """
    values = {key: value for key, value in overrides.items() if value is not None}
    cfg = ExperimentConfig(**values)
    if "save_dir" in values and "model_dir" not in values:
        cfg = replace(cfg, model_dir=cfg.save_dir)
    if "out_dir" in values:
        cfg = replace(cfg, log_dir=cfg.out_dir / "logs", figure_dir=cfg.out_dir / "figures",
                      result_dir=cfg.out_dir / "results")
    if not 0.0 < cfg.confidence_level < 1.0:
        raise ValueError("置信水平必须在0和1之间")
    return cfg


if __name__ == "__main__":
    cfg = build_config()
    ensure_directories(cfg)
    print(f"DATA_DIR={cfg.data_dir}")
    print(f"SAVE_DIR={cfg.save_dir}")
    print(f"OUT_DIR={cfg.out_dir}")
