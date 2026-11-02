# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-11-01
# 文件名: config.py
# 功能说明: 集中管理五组TCN离线功率与能耗预测实验路径和参数
# 版本号：1.0

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent  # 代码与说明文档目录
DATA_ROOT = Path(r"D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset")
EXPERIMENT_DATA_DIR = DATA_ROOT / "TCN-AMOVFLY-planned"
DATA_DIR = EXPERIMENT_DATA_DIR  # 本实验独立加工数据
SOURCE_PROCESSED_DIR = DATA_ROOT / "sourse"  # AMOVFLY原始ready data目录
SAVE_DIR = Path(r"D:/Python-files/Energy-prediction/model/TCN-AMOVFLY-planned")  # 新契约权重和标准化器
OUT_DIR = PROJECT_ROOT / "out"  # 本地统计表、日志和图表
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
TARGET_COLUMN = "power_w"
ENERGY_COLUMN = "energy_interval_wh"
GROUPS = {
    "A": {"name": "原始10维_混合激活", "use_sma": False, "activation": "mixed"},
    "B": {"name": "原始10维_统一SiLU", "use_sma": False, "activation": "silu"},
    "C": {"name": "原始10维加5维SMA_混合激活", "use_sma": True, "activation": "mixed"},
    "D": {"name": "原始10维加5维SMA_统一SiLU", "use_sma": True, "activation": "silu"},
    "E": {"name": "原始10维加5维SMA_统一LeakyReLU(0.05)", "use_sma": True, "activation": "leaky_relu"},
}

RESAMPLE_SECONDS = 0.2
SMA_WINDOW_STEPS = 10
TCN_WINDOW_SECONDS = (1.0, 2.0, 5.0, 10.0)
TUNE_EPOCHS = 8
EPOCHS = 80
BATCH_SIZE = 2048
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TCN_CHANNELS = (32, 32)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.05
SCHEMA_VERSION = "AMOVFLY-TCN-planned-1.0"
MAX_GAP_SECONDS = 2.0
MOTOR_ON_POWER_THRESHOLD_W = 100.0
AIRBORNE_HORIZONTAL_THRESHOLD_M = 1.0
AIRBORNE_VERTICAL_THRESHOLD_M = 0.5
RANDOM_SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存五组离线TCN实验的路径和超参数。
    参数: 字段由默认常量或命令行覆盖项提供。
    返回: 实验配置对象。
    调用位置: main.py、train.py、evaluate.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    source_processed_dir: Path = SOURCE_PROCESSED_DIR
    save_dir: Path = SAVE_DIR
    out_dir: Path = OUT_DIR
    model_dir: Path = MODEL_DIR
    log_dir: Path = LOG_DIR
    figure_dir: Path = FIGURE_DIR
    result_dir: Path = RESULT_DIR
    resample_seconds: float = RESAMPLE_SECONDS
    sma_window_steps: int = SMA_WINDOW_STEPS
    tcn_window_seconds: tuple[float, ...] = TCN_WINDOW_SECONDS
    tune_epochs: int = TUNE_EPOCHS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    tcn_channels: tuple[int, ...] = TCN_CHANNELS
    tcn_kernel_size: int = TCN_KERNEL_SIZE
    tcn_dropout: float = TCN_DROPOUT
    random_seed: int = RANDOM_SEED
    device: str = "cuda"

    @property
    def summary_csv(self) -> Path:
        """功能: 返回五组汇总表路径。
        参数: 无。
        返回: CSV路径。
        调用位置: evaluate.py、report.py。
        """
        return self.result_dir / "TCN-comparison_all_metrics.csv"

    @property
    def manifest_json(self) -> Path:
        """功能: 返回数据指纹与规模记录路径。
        参数: 无。
        返回: JSON路径。
        调用位置: data_utils.py、report.py。
        """
        return self.data_dir / "dataset_summary_TCN-comparison.json"

    @property
    def terminal_log(self) -> Path:
        """功能: 返回控制台日志路径。
        参数: 无。
        返回: UTF-8日志路径。
        调用位置: main.py。
        """
        return self.log_dir / "TCN-comparison_terminal.log"

    @property
    def report_md(self) -> Path:
        """功能: 返回当前输出报告路径。
        参数: 无。
        返回: Markdown路径。
        调用位置: report.py。
        """
        return self.out_dir / "outreadme.md"

    @property
    def readme_md(self) -> Path:
        """功能: 返回稳定项目说明路径。
        参数: 无。
        返回: Markdown路径。
        调用位置: report.py。
        """
        return self.project_root / "README.md"

    @property
    def runtime_json(self) -> Path:
        """功能: 返回软件硬件与命令配置记录路径。
        参数: 无。
        返回: JSON路径。
        调用位置: main.py、report.py。
        """
        return self.out_dir / "TCN-comparison_runtime.json"


def group_paths(cfg: ExperimentConfig, group: str) -> dict[str, Path]:
    """功能: 返回单个实验组的模型、调参、预测和指标路径。
    参数: cfg为实验配置，group为A、B、C、D或E。
    返回: 统一命名的路径字典。
    调用位置: train.py、evaluate.py、visualize.py。
    """

    key = str(group).upper()
    return {
        "checkpoint": cfg.model_dir / f"TCN-comparison_{key}_best.pt",
        "scaler": cfg.model_dir / f"TCN-comparison_{key}_scaler.json",
        "tuning": cfg.result_dir / f"TCN-comparison_{key}_window_tuning.csv",
        "training_log": cfg.log_dir / f"TCN-comparison_{key}_training.csv",
        "tuning_log": cfg.log_dir / f"TCN-comparison_{key}_tuning_epochs.csv",
        "predictions": cfg.result_dir / f"TCN-comparison_{key}_test_predictions.csv",
        "metrics": cfg.result_dir / f"TCN-comparison_{key}_metrics.json",
        "flight_metrics": cfg.result_dir / f"TCN-comparison_{key}_flight_energy.csv",
        "window_metrics": cfg.result_dir / f"TCN-comparison_{key}_window_energy.csv",
        "stage_metrics": cfg.result_dir / f"TCN-comparison_{key}_stage_metrics.csv",
        "route_metrics": cfg.result_dir / f"TCN-comparison_{key}_route_metrics.csv",
    }


def table_path(cfg: ExperimentConfig, split: str) -> Path:
    """功能: 集中返回各划分的加工表路径。
    参数: cfg为配置，split为train、val或test。
    返回: 数据表路径。
    调用位置: data_utils.py、report.py。
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(f"不支持的数据划分: {split}")
    return cfg.data_dir / f"{split}_TCN-comparison.csv"


def figure_path(cfg: ExperimentConfig, name: str, suffix: str = "png") -> Path:
    """功能: 集中生成图表路径。
    参数: cfg为配置，name为图表用途，suffix为格式。
    返回: 图表文件路径。
    调用位置: visualize.py、report.py。
    """

    return cfg.figure_dir / f"TCN-comparison_{name}.{suffix}"


def ensure_directories(cfg: ExperimentConfig) -> None:
    """功能: 创建实验数据、模型、日志、结果和图表目录。
    参数: cfg为实验配置对象。
    返回: None。
    调用位置: main.py、prepare_data.py。
    """

    for path in (cfg.data_dir, cfg.save_dir, cfg.out_dir, cfg.model_dir,
                 cfg.log_dir, cfg.figure_dir, cfg.result_dir):
        path.mkdir(parents=True, exist_ok=True)


def build_config(**overrides) -> ExperimentConfig:
    """功能: 按命令行覆盖项构造实验配置。
    参数: overrides为字段名和值组成的覆盖字典。
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
    if cfg.tune_epochs < 1 or cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("训练轮数和批大小必须为正数")
    return cfg


if __name__ == "__main__":
    cfg = build_config()
    ensure_directories(cfg)
    print(f"DATA_DIR={cfg.data_dir}")
    print(f"SAVE_DIR={cfg.save_dir}")
    print(f"OUT_DIR={cfg.out_dir}")
