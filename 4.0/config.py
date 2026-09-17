# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-09-16
# 文件名: config.py
# 功能说明: 集中管理实验4.0任务前能耗预测与在线RLS规划接口的路径和参数
# 版本号：4.0

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("D:/Python-files/Energy-prediction/data")
SAVE_DIR = PROJECT_ROOT / "model"
OUT_DIR = PROJECT_ROOT / "out"
OUT_DATA_DIR = OUT_DIR / "data"
OUT_MODEL_DIR = OUT_DIR / "model"
OUT_PREDICTION_DIR = OUT_DIR / "predictions"
OUT_TASK_DIR = OUT_DIR / "tasks"
OUT_RLS_DIR = OUT_DIR / "rls"
OUT_FIGURE_DIR = OUT_DIR / "figures"
OUT_ROUTE_DIR = OUT_DIR / "routes"
LOG_DIR = OUT_DIR / "logs"

RAW_DATA_DIR = DATA_DIR / "dji_matrice_100"
RAW_FLIGHTS_CSV = RAW_DATA_DIR / "flights.csv"
PROCESSED_DIR = OUT_DATA_DIR / "processed_4.0"
CLEAN_DATA_CSV = PROCESSED_DIR / "planning_energy_features_4.0.csv"
TRAIN_CSV = PROCESSED_DIR / "train_4.0.csv"
VAL_CSV = PROCESSED_DIR / "val_4.0.csv"
TEST_CSV = PROCESSED_DIR / "test_4.0.csv"
FEATURE_META_JSON = PROCESSED_DIR / "feature_metadata_4.0.json"
FEATURE_ANALYSIS_CSV = PROCESSED_DIR / "feature_analysis_4.0.csv"
DATASET_SUMMARY_JSON = PROCESSED_DIR / "dataset_summary_4.0.json"
RESAMPLE_MAP_CSV = PROCESSED_DIR / "resample_map_4.0.csv"

BEST_MODEL_FILE = SAVE_DIR / "best_planning_tcn_4.0.pt"
FINAL_MODEL_FILE = SAVE_DIR / "final_planning_tcn_4.0.pt"
SCALER_JSON = SAVE_DIR / "planning_scaler_4.0.json"
TRAINING_LOG_CSV = OUT_MODEL_DIR / "training_log_4.0.csv"
TCN_TUNING_CSV = OUT_MODEL_DIR / "tcn_window_tuning_4.0.csv"
RLS_TUNING_CSV = OUT_RLS_DIR / "rls_tuning_4.0.csv"
RLS_INTERVAL_CALIBRATION_JSON = OUT_RLS_DIR / "rls_interval_calibration_4.0.json"
EVALUATION_JSON = OUT_MODEL_DIR / "evaluation_4.0.json"
EVALUATION_CSV = OUT_MODEL_DIR / "evaluation_4.0.csv"
RLS_WINDOW_COMPARISON_CSV = OUT_MODEL_DIR / "rls_energy_window_comparison_4.0.csv"
RLS_WINDOW_COMPARISON_JSON = OUT_MODEL_DIR / "rls_energy_window_comparison_4.0.json"
RLS_WINDOW_COMPARISON_BY_FLIGHT_CSV = OUT_MODEL_DIR / "rls_energy_window_comparison_by_flight_4.0.csv"
RLS_WINDOW_EVALUATION_CSV = OUT_MODEL_DIR / "rls_energy_window_evaluation_4.0.csv"
GENERALIZATION_EVALUATION_CSV = OUT_MODEL_DIR / "generalization_evaluation_4.0.csv"
INTERVAL_STRATIFIED_CSV = OUT_MODEL_DIR / "interval_stratified_evaluation_4.0.csv"
REMAINING_TASK_EVALUATION_CSV = OUT_MODEL_DIR / "remaining_task_evaluation_4.0.csv"
PREDICTIONS_CSV = OUT_PREDICTION_DIR / "test_predictions_4.0.csv"
ALL_PREDICTIONS_CSV = OUT_PREDICTION_DIR / "all_route_predictions_4.0.csv"
TASK_INPUT_CSV = OUT_TASK_DIR / "demo_candidate_task_4.0.csv"
TASK_OUTPUT_CSV = OUT_TASK_DIR / "task_before_prediction_4.0.csv"
TASK_OUTPUT_JSON = OUT_TASK_DIR / "task_before_summary_4.0.json"
ONLINE_OUTPUT_CSV = OUT_TASK_DIR / "online_remaining_prediction_4.0.csv"
ONLINE_OUTPUT_JSON = OUT_TASK_DIR / "online_decision_4.0.json"
UNCERTAINTY_JSON = OUT_MODEL_DIR / "uncertainty_model_4.0.json"
TERMINAL_LOG_FILE = LOG_DIR / "terminal_4.0.log"
VISUALIZATION_SUMMARY_JSON = OUT_DIR / "visualization_summary_4.0.json"
RLS_PARAMETER_TRACE_CSV = OUT_RLS_DIR / "rls_parameter_trace_4.0.csv"
RLS_PARAMETER_STATISTICS_CSV = OUT_RLS_DIR / "rls_parameter_statistics_4.0.csv"
RLS_PARAMETER_SUMMARY_JSON = OUT_RLS_DIR / "rls_parameter_summary_4.0.json"
RLS_CORRECTION_TRACE_CSV = OUT_RLS_DIR / "rls_future_correction_trace_4.0.csv"
RLS_CORRECTION_SUMMARY_JSON = OUT_RLS_DIR / "rls_future_correction_summary_4.0.json"
RLS_CORRECTION_GIF = OUT_RLS_DIR / "rls_future_energy_correction_4.0.gif"

RESAMPLE_SECONDS = 0.2
TCN_WINDOW_SECONDS = (0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0)
RLS_FORGETTING_FACTORS = (0.85, 0.88, 0.90, 0.93, 0.96, 0.98, 0.99)
RLS_INITIAL_COVARIANCES = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
RLS_WINDOW_SECONDS = 1.0
RLS_COMPARISON_WINDOW_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0)
RANDOM_SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 2048
TUNE_EPOCHS = 20
EPOCHS = 80
# 4.0按用户要求完成80轮；不因验证集短期波动提前终止候选训练。
PATIENCE = 80
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEFAULT_CONFIDENCE = 0.95
ONLINE_REPLAN_RELATIVE_DEVIATION = 0.15
ONLINE_DEGRADE_RELATIVE_DEVIATION = 0.30
ONLINE_PAUSE_RELATIVE_DEVIATION = 0.50
TCN_CHANNELS = (32, 32)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存实验4.0独立路径、任务输入字段和超参数。
    参数: 字段由build_config根据命令行覆盖项生成。
    返回: 实验配置对象。
    调用位置: main.py、train.py、task_api.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    save_dir: Path = SAVE_DIR
    out_dir: Path = OUT_DIR
    out_data_dir: Path = OUT_DATA_DIR
    out_model_dir: Path = OUT_MODEL_DIR
    out_prediction_dir: Path = OUT_PREDICTION_DIR
    out_task_dir: Path = OUT_TASK_DIR
    out_rls_dir: Path = OUT_RLS_DIR
    out_figure_dir: Path = OUT_FIGURE_DIR
    out_route_dir: Path = OUT_ROUTE_DIR
    log_dir: Path = LOG_DIR
    raw_flights_csv: Path = RAW_FLIGHTS_CSV
    processed_dir: Path = PROCESSED_DIR
    clean_data_csv: Path = CLEAN_DATA_CSV
    train_csv: Path = TRAIN_CSV
    val_csv: Path = VAL_CSV
    test_csv: Path = TEST_CSV
    feature_meta_json: Path = FEATURE_META_JSON
    feature_analysis_csv: Path = FEATURE_ANALYSIS_CSV
    dataset_summary_json: Path = DATASET_SUMMARY_JSON
    resample_map_csv: Path = RESAMPLE_MAP_CSV
    best_model_file: Path = BEST_MODEL_FILE
    final_model_file: Path = FINAL_MODEL_FILE
    scaler_json: Path = SCALER_JSON
    training_log_csv: Path = TRAINING_LOG_CSV
    tcn_tuning_csv: Path = TCN_TUNING_CSV
    rls_tuning_csv: Path = RLS_TUNING_CSV
    rls_interval_calibration_json: Path = RLS_INTERVAL_CALIBRATION_JSON
    evaluation_json: Path = EVALUATION_JSON
    evaluation_csv: Path = EVALUATION_CSV
    rls_window_comparison_csv: Path = RLS_WINDOW_COMPARISON_CSV
    rls_window_comparison_json: Path = RLS_WINDOW_COMPARISON_JSON
    rls_window_comparison_by_flight_csv: Path = RLS_WINDOW_COMPARISON_BY_FLIGHT_CSV
    rls_window_evaluation_csv: Path = RLS_WINDOW_EVALUATION_CSV
    generalization_evaluation_csv: Path = GENERALIZATION_EVALUATION_CSV
    interval_stratified_csv: Path = INTERVAL_STRATIFIED_CSV
    remaining_task_evaluation_csv: Path = REMAINING_TASK_EVALUATION_CSV
    predictions_csv: Path = PREDICTIONS_CSV
    all_predictions_csv: Path = ALL_PREDICTIONS_CSV
    task_input_csv: Path = TASK_INPUT_CSV
    task_output_csv: Path = TASK_OUTPUT_CSV
    task_output_json: Path = TASK_OUTPUT_JSON
    online_output_csv: Path = ONLINE_OUTPUT_CSV
    online_output_json: Path = ONLINE_OUTPUT_JSON
    uncertainty_json: Path = UNCERTAINTY_JSON
    terminal_log_file: Path = TERMINAL_LOG_FILE
    visualization_summary_json: Path = VISUALIZATION_SUMMARY_JSON
    rls_parameter_trace_csv: Path = RLS_PARAMETER_TRACE_CSV
    rls_parameter_statistics_csv: Path = RLS_PARAMETER_STATISTICS_CSV
    rls_parameter_summary_json: Path = RLS_PARAMETER_SUMMARY_JSON
    rls_correction_trace_csv: Path = RLS_CORRECTION_TRACE_CSV
    rls_correction_summary_json: Path = RLS_CORRECTION_SUMMARY_JSON
    rls_correction_gif: Path = RLS_CORRECTION_GIF
    resample_seconds: float = RESAMPLE_SECONDS
    tcn_window_seconds: tuple[float, ...] = TCN_WINDOW_SECONDS
    rls_forgetting_factors: tuple[float, ...] = RLS_FORGETTING_FACTORS
    rls_initial_covariances: tuple[float, ...] = RLS_INITIAL_COVARIANCES
    rls_window_seconds: float = RLS_WINDOW_SECONDS
    rls_comparison_window_seconds: tuple[float, ...] = RLS_COMPARISON_WINDOW_SECONDS
    random_seed: int = RANDOM_SEED
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    batch_size: int = BATCH_SIZE
    tune_epochs: int = TUNE_EPOCHS
    epochs: int = EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    default_confidence: float = DEFAULT_CONFIDENCE
    online_replan_relative_deviation: float = ONLINE_REPLAN_RELATIVE_DEVIATION
    online_degrade_relative_deviation: float = ONLINE_DEGRADE_RELATIVE_DEVIATION
    online_pause_relative_deviation: float = ONLINE_PAUSE_RELATIVE_DEVIATION
    tcn_channels: tuple[int, ...] = TCN_CHANNELS
    tcn_kernel_size: int = TCN_KERNEL_SIZE
    tcn_dropout: float = TCN_DROPOUT
    device: str = "cuda"


def build_config(**overrides) -> ExperimentConfig:
    """功能: 构建4.0配置并按覆盖项重新定位独立目录。
    参数: overrides为命令行或测试提供的配置字段。
    返回: 已完成路径派生的ExperimentConfig。
    调用位置: main.py。
    """

    cfg = ExperimentConfig()
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            cfg = replace(cfg, **{key: value})
    root = cfg.out_dir
    return replace(
        cfg,
        out_data_dir=root / "data", out_model_dir=root / "model",
        out_prediction_dir=root / "predictions", out_task_dir=root / "tasks",
        out_rls_dir=root / "rls", out_figure_dir=root / "figures", out_route_dir=root / "routes",
        log_dir=root / "logs", processed_dir=root / "data" / "processed_4.0",
        clean_data_csv=root / "data" / "processed_4.0" / "planning_energy_features_4.0.csv",
        train_csv=root / "data" / "processed_4.0" / "train_4.0.csv",
        val_csv=root / "data" / "processed_4.0" / "val_4.0.csv",
        test_csv=root / "data" / "processed_4.0" / "test_4.0.csv",
        feature_meta_json=root / "data" / "processed_4.0" / "feature_metadata_4.0.json",
        feature_analysis_csv=root / "data" / "processed_4.0" / "feature_analysis_4.0.csv",
        dataset_summary_json=root / "data" / "processed_4.0" / "dataset_summary_4.0.json",
        resample_map_csv=root / "data" / "processed_4.0" / "resample_map_4.0.csv",
        training_log_csv=root / "model" / "training_log_4.0.csv",
        tcn_tuning_csv=root / "model" / "tcn_window_tuning_4.0.csv",
        rls_tuning_csv=root / "rls" / "rls_tuning_4.0.csv",
        rls_interval_calibration_json=root / "rls" / "rls_interval_calibration_4.0.json",
        evaluation_json=root / "model" / "evaluation_4.0.json",
        evaluation_csv=root / "model" / "evaluation_4.0.csv",
        rls_window_comparison_csv=root / "model" / "rls_energy_window_comparison_4.0.csv",
        rls_window_comparison_json=root / "model" / "rls_energy_window_comparison_4.0.json",
        rls_window_comparison_by_flight_csv=root / "model" / "rls_energy_window_comparison_by_flight_4.0.csv",
        rls_window_evaluation_csv=root / "model" / "rls_energy_window_evaluation_4.0.csv",
        generalization_evaluation_csv=root / "model" / "generalization_evaluation_4.0.csv",
        interval_stratified_csv=root / "model" / "interval_stratified_evaluation_4.0.csv",
        remaining_task_evaluation_csv=root / "model" / "remaining_task_evaluation_4.0.csv",
        predictions_csv=root / "predictions" / "test_predictions_4.0.csv",
        all_predictions_csv=root / "predictions" / "all_route_predictions_4.0.csv",
        task_input_csv=root / "tasks" / "demo_candidate_task_4.0.csv",
        task_output_csv=root / "tasks" / "task_before_prediction_4.0.csv",
        task_output_json=root / "tasks" / "task_before_summary_4.0.json",
        online_output_csv=root / "tasks" / "online_remaining_prediction_4.0.csv",
        online_output_json=root / "tasks" / "online_decision_4.0.json",
        uncertainty_json=root / "model" / "uncertainty_model_4.0.json",
        terminal_log_file=root / "logs" / "terminal_4.0.log",
        visualization_summary_json=root / "visualization_summary_4.0.json",
        rls_parameter_trace_csv=root / "rls" / "rls_parameter_trace_4.0.csv",
        rls_parameter_statistics_csv=root / "rls" / "rls_parameter_statistics_4.0.csv",
        rls_parameter_summary_json=root / "rls" / "rls_parameter_summary_4.0.json",
        rls_correction_trace_csv=root / "rls" / "rls_future_correction_trace_4.0.csv",
        rls_correction_summary_json=root / "rls" / "rls_future_correction_summary_4.0.json",
        rls_correction_gif=root / "rls" / "rls_future_energy_correction_4.0.gif",
    )


def ensure_directories(cfg: ExperimentConfig) -> None:
    """功能: 创建4.0运行需要的独立目录。
    参数: cfg为实验配置对象。
    返回: None。
    调用位置: main.py、prepare_dataset。
    """

    for path in (cfg.save_dir, cfg.out_dir, cfg.out_data_dir, cfg.out_model_dir,
                 cfg.out_prediction_dir, cfg.out_task_dir, cfg.out_rls_dir,
                 cfg.out_figure_dir, cfg.out_route_dir, cfg.log_dir, cfg.processed_dir):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = build_config()
    ensure_directories(cfg)
    print(f"PROJECT_ROOT={cfg.project_root}")
    print(f"DATA_DIR={cfg.data_dir}")
    print(f"SAVE_DIR={cfg.save_dir}")
    print(f"OUT_DIR={cfg.out_dir}")
