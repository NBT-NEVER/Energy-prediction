# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-09-09
# 文件名: config.py
# 功能说明: 管理对比算法实验的数据、输出路径与统一训练参数
# 版本号：4.1

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# 对比实验严格复用 4.1 的加工数据，保证输入维度、采样步长、flight 划分和样本数量一致。
_DATA_CANDIDATES = (
    Path(r"D:/Python-files/Energy-prediction/data/dji_matrice_100_data/4.1/processed"),
    PROJECT_ROOT.parent / "4.1" / "processed",
)
DATA_DIR = next((path for path in _DATA_CANDIDATES if path.exists()), _DATA_CANDIDATES[0])
SAVE_DIR = PROJECT_ROOT / "out" / "model"
OUT_DIR = PROJECT_ROOT / "out"
# 4.1 主方法的固定输出目录；对比算法统一读取同一版本的数据契约。
BASELINE_DIR = PROJECT_ROOT.parent / "4.1" / "out"
TRAIN_CSV = DATA_DIR / "train_4.1.csv"
VAL_CSV = DATA_DIR / "val_4.1.csv"
TEST_CSV = DATA_DIR / "test_4.1.csv"
FEATURE_META_JSON = DATA_DIR / "feature_metadata_4.1.json"

FEATURE_COLUMNS = [
    "time_s", "task_duration_s", "planned_position_east_m", "planned_position_north_m",
    "planned_position_up_m", "wind_east_mps", "wind_north_mps", "payload_g",
    "planned_motor_on", "planned_airborne",
]
TARGET_COLUMN = "power_w"
RANDOM_SEED = 20260909
SEQUENCE_WINDOW = 20
# 深度对比模型的最大训练轮数；最终权重由验证集早停选择。
DEEP_EPOCHS = 24
DEEP_PATIENCE = 4
BATCH_SIZE = 4096
# 深度对比模型只在验证集上选择参数；候选数量保持有限，避免算力差异成为结果主因。
MODEL_CANDIDATES = {
    "lstm": (
        {"hidden_size": 64, "layers": 2, "bidirectional": True, "dropout": 0.30,
         "learning_rate": 1e-3, "weight_decay": 1e-4},
        {"hidden_size": 128, "layers": 2, "bidirectional": True, "dropout": 0.50,
         "learning_rate": 1e-3, "weight_decay": 1e-4},
    ),
    "lr_tcn_sma": (
        {"channels": 48, "blocks": 3, "dropout": 0.08, "sma_window": 5,
         "learning_rate": 2e-3, "weight_decay": 1e-4},
        {"channels": 64, "blocks": 3, "dropout": 0.12, "sma_window": 10,
         "learning_rate": 1e-3, "weight_decay": 1e-4},
    ),
    "lstm_transformer": (
        {"model_dim": 32, "lstm_layers": 1, "transformer_layers": 1, "heads": 4,
         "feedforward": 64, "dropout": 0.08, "learning_rate": 2e-3,
         "weight_decay": 1e-4},
        {"model_dim": 64, "lstm_layers": 1, "transformer_layers": 2, "heads": 4,
         "feedforward": 128, "dropout": 0.10, "learning_rate": 1e-3,
         "weight_decay": 1e-4},
    ),
    "cnn_lstm": (
        {"cnn_channels": 48, "lstm_hidden": 48, "dropout": 0.10,
         "learning_rate": 2e-3, "weight_decay": 1e-4},
        {"cnn_channels": 64, "lstm_hidden": 64, "dropout": 0.15,
         "learning_rate": 1e-3, "weight_decay": 1e-4},
    ),
}

# 图表文件名统一配置，避免绘图代码重复散落输出名称。
FIGURE_NAMES = {
    "power_scatter": "power_scatter.png",
    "flight_power_timeseries": "flight_power_timeseries.png",
    "residual_histogram": "residual_histogram.png",
    "residual_vs_actual": "residual_vs_actual.png",
    "power_bin_error": "power_bin_error.png",
    "flight_energy_scatter": "flight_energy_scatter.png",
    "flight_energy_error_ranking": "flight_energy_error_ranking.png",
    "cumulative_energy_timeseries": "cumulative_energy_timeseries.png",
    "phase_error_metrics": "phase_error_metrics.png",
    "absolute_error_cdf": "absolute_error_cdf.png",
    "prediction_zoom": "prediction_zoom.png",
    "flight_energy_error_histogram": "flight_energy_error_histogram.png",
    "phase_error_distribution": "phase_error_distribution.png",
    "comparison_metrics": "comparison_metrics.png",
    "comparison_error_boxplot": "comparison_error_boxplot.png",
    "comparison_phase_wape": "comparison_phase_wape.png",
    "comparison_flight_energy_scatter": "comparison_flight_energy_scatter.png",
    "comparison_metric_heatmap": "comparison_metric_heatmap.png",
    "comparison_error_cdf": "comparison_error_cdf.png",
    "comparison_overall_rank": "comparison_overall_rank.png",
    "comparison_power_timeseries": "comparison_power_timeseries.png",
    "comparison_error_violin": "comparison_error_violin.png",
    "comparison_bland_altman": "comparison_bland_altman.png",
    "comparison_energy_error_distribution": "comparison_energy_error_distribution.png",
    "comparison_phase_metrics": "comparison_phase_metrics.png",
    "comparison_metric_radar": "comparison_metric_radar.png",
}

ALGORITHMS = {
    "proposed_tcn_rls": {
        "name": "4.1 TCN（Temporal Convolutional Network，时间卷积网络）+ RLS（Recursive Least Squares，递推最小二乘）",
        "kind": "baseline",
        "source": "本项目 4.1 主方法；TCN 规划输入与窗口级 RLS 在线校正来自本项目 4.1 实现，图表组织参照 Luo et al. (2025)。",
        "citation_key": "[A0]",
    },
    "rf_tlatt_lite": {
        "name": "RF-TLATT（Random Forest–Temporal Local Attention and Time，随机森林-时序局部注意力与时间）相位岭回归",
        "kind": "rf_tlatt_lite",
        "source": "Luo et al. (2025) 的相位专用建模思想；当前实现为速度阈值分类加相位岭回归，不是完整 RF-TLATT 复现。",
        "citation_key": "[A1]",
    },
    "physical_mlr": {
        "name": "MLR（Multiple Linear Regression，多元线性回归）物理特征基线",
        "kind": "physical_mlr",
        "source": "Jastrzębska et al. (2026) 的物理指标回归思想；当前实现为单一全局回归加 moving 指示变量。",
        "citation_key": "[A2]",
    },
    "lstm": {
        "name": "BiLSTM（Bidirectional Long Short-Term Memory，双向长短期记忆网络）",
        "kind": "lstm",
        "source": "Muli et al. (2022) 的 LSTM 数据驱动能耗预测。",
        "citation_key": "[A3]",
    },
    "lr_tcn_sma": {
        "name": "LR-TCN-SMA（LeakyReLU Temporal Convolutional Network with Simple Moving Average，带简单移动平均的LeakyReLU时间卷积网络）",
        "kind": "lr_tcn_sma",
        "source": "Dudukcu et al. (2024) 的 LeakyReLU TCN 与简单移动平均输入。",
        "citation_key": "[A4]",
    },
    "lstm_transformer": {
        "name": "LSTM-Transformer（Long Short-Term Memory–Transformer，长短期记忆-Transformer混合网络）",
        "kind": "lstm_transformer",
        "source": "Feng et al. (2024) 的 LSTM-Transformer 能耗预测框架；结构采用 LSTM 短期时序编码、位置编码和 Transformer Encoder 长程依赖建模。",
        "citation_key": "[A5]",
    },
    "cnn_lstm": {
        "name": "CNN-LSTM（Convolutional Neural Network–Long Short-Term Memory，卷积神经网络-长短期记忆网络）",
        "kind": "cnn_lstm",
        "source": "Luo et al. (2025) RF-TLATT 论文表 8 的 CNN-LSTM 对比名称（二手来源）；当前实现采用一维卷积提取局部波形，再由 LSTM 聚合时间依赖。",
        "citation_key": "[A1]",
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存单一算法的路径和固定参数。
    参数: algorithm为算法标识。
    返回: 实验配置对象。
    调用位置: main.py与comparison_engine.py。
    """

    algorithm: str

    @property
    def algorithm_dir(self) -> Path:
        return PROJECT_ROOT / self.algorithm

    @property
    def out_dir(self) -> Path:
        return self.algorithm_dir / "out"

    @property
    def figure_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def model_dir(self) -> Path:
        return self.out_dir / "model"

    @property
    def prediction_csv(self) -> Path:
        return self.out_dir / "predictions.csv"

    @property
    def metrics_json(self) -> Path:
        return self.out_dir / "metrics.json"

    @property
    def metrics_csv(self) -> Path:
        return self.out_dir / "metrics.csv"

    @property
    def flight_summary_csv(self) -> Path:
        return self.out_dir / "flight_energy_summary.csv"

    @property
    def training_log_csv(self) -> Path:
        return self.model_dir / "training_log.csv"

    @property
    def tuning_csv(self) -> Path:
        return self.model_dir / "tuning_results.csv"


def ensure_directories(cfg: ExperimentConfig | None = None) -> None:
    """功能: 创建项目或算法所需输出目录。
    参数: cfg为可选的单算法配置对象。
    返回: 无。
    调用位置: 主程序和评估程序。
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        for path in (cfg.algorithm_dir, cfg.out_dir, cfg.figure_dir, cfg.model_dir):
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"OUT_DIR={OUT_DIR}")
