# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-10-19
# 文件名: config.py
# 功能说明: 集中管理实验5.0车辆能耗预测的数据、模型和结果路径
# 版本号：5.0

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
# GECR原始公开数据保存在项目之外，避免误提交大型原始文件。
DATA_ROOT = Path(r"D:\Python-files\Energy-prediction\data\MOTIVES-LAB-GECR")
SOURCE_DIR = DATA_ROOT / "Source"
DATA_DIR = DATA_ROOT / "5.0"
PROCESSED_DIR = DATA_DIR / "processed"
SAVE_DIR = PROJECT_ROOT / "model"
OUT_DIR = PROJECT_ROOT / "out"
OUT_MODEL_DIR = OUT_DIR / "model"
PREDICTION_DIR = OUT_DIR / "predictions"
TASK_DIR = OUT_DIR / "tasks"
RLS_DIR = OUT_DIR / "rls"
ROUTE_DIR = OUT_DIR / "routes"
FIGURE_DIR = OUT_DIR / "figures"
TRAINING_FIGURE_DIR = FIGURE_DIR / "training"
PREDICTION_FIGURE_DIR = FIGURE_DIR / "prediction"
RESULTS_FIGURE_DIR = FIGURE_DIR / "results"
LOG_DIR = OUT_DIR / "logs"

TRAIN_CSV = PROCESSED_DIR / "train_5.0.csv"
VAL_CSV = PROCESSED_DIR / "val_5.0.csv"
TEST_CSV = PROCESSED_DIR / "test_5.0.csv"
CALIBRATION_CSV = PROCESSED_DIR / "calibration_5.0.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_trips_5.0.csv"
DATA_SUMMARY_JSON = PROCESSED_DIR / "data_summary_5.0.json"
MODEL_FILE = SAVE_DIR / "vehicle_tcn_5.0.pt"
SCALER_FILE = SAVE_DIR / "scaler_5.0.json"
TRAIN_LOG_CSV = OUT_MODEL_DIR / "training_log_5.0.csv"
RUN_LOG_JSONL = LOG_DIR / "run_log_5.0.jsonl"
TERMINAL_LOG_FILE = LOG_DIR / "terminal_5.0.log"
TRAINING_EPOCH_LOG_JSONL = LOG_DIR / "training_epochs_5.0.jsonl"
DATASET_STATS_CSV = OUT_MODEL_DIR / "dataset_statistics_5.0.csv"
SPLIT_STATS_CSV = OUT_MODEL_DIR / "split_statistics_5.0.csv"
FEATURE_STATS_CSV = OUT_MODEL_DIR / "feature_statistics_5.0.csv"
TRIP_STATS_CSV = OUT_MODEL_DIR / "trip_statistics_5.0.csv"
TEST_ROUTE_STATS_CSV = ROUTE_DIR / "test_route_statistics_5.0.csv"
FIGURE_SCALES_JSON = OUT_DIR / "visualization_summary_5.0.json"
SEARCH_RESULTS_CSV = OUT_MODEL_DIR / "hyperparameter_search_5.0.csv"
SEARCH_TRIALS_CSV = OUT_MODEL_DIR / "search_trials_5.0.csv"
FEATURE_ABLATION_CSV = OUT_MODEL_DIR / "feature_ablation_5.0.csv"
RLS_SEARCH_CSV = RLS_DIR / "rls_search_5.0.csv"
SELECTED_CONFIG_JSON = OUT_MODEL_DIR / "selected_config_5.0.json"
PREDICTIONS_CSV = PREDICTION_DIR / "test_predictions_5.0.csv"
TASK_PREDICTIONS_CSV = TASK_DIR / "scheduled_task_predictions_5.0.csv"
RLS_WINDOWS_CSV = RLS_DIR / "test_rls_windows_5.0.csv"
RLS_TRAJECTORY_CSV = RLS_DIR / "test_rls_trajectory_5.0.csv"
CALIBRATION_METRICS_JSON = OUT_MODEL_DIR / "calibration_metrics_5.0.json"
POWER_INTERVAL_CALIBRATION_JSON = OUT_MODEL_DIR / "power_interval_calibration_5.0.json"
CALIBRATION_PREDICTIONS_CSV = PREDICTION_DIR / "calibration_predictions_5.0.csv"
RLS_CALIBRATION_REPLAY_CSV = RLS_DIR / "rls_calibration_replay_5.0.csv"
METRICS_JSON = OUT_MODEL_DIR / "metrics_5.0.json"
TRIP_METRICS_CSV = OUT_MODEL_DIR / "trip_metrics_5.0.csv"
FIGURES = {
    "loss": TRAINING_FIGURE_DIR / "loss_curves_5.0.png",
    "search": TRAINING_FIGURE_DIR / "hyperparameter_ranking_5.0.png",
    "rls_search": TRAINING_FIGURE_DIR / "rls_hyperparameter_ranking_5.0.png",
    "power_scatter": PREDICTION_FIGURE_DIR / "test_power_scatter_5.0.png",
    "residual": PREDICTION_FIGURE_DIR / "test_power_residuals_5.0.png",
    "trip_energy": PREDICTION_FIGURE_DIR / "test_trip_energy_5.0.png",
    "trip_trace": PREDICTION_FIGURE_DIR / "test_trip_prediction_5.0.png",
    "multi_route_map": ROUTE_DIR / "multi_route_map_5.0.png",
    "multi_route_power": ROUTE_DIR / "multi_route_power_5.0.png",
    "multi_route_energy": ROUTE_DIR / "multi_route_energy_5.0.png",
    "power_confidence": PREDICTION_FIGURE_DIR / "power_confidence_interval_5.0.png",
    "tcn_second_energy_interval": PREDICTION_FIGURE_DIR / "tcn_second_energy_interval_5.0.png",
    "tcn_trip_energy_interval": PREDICTION_FIGURE_DIR / "tcn_trip_energy_interval_5.0.png",
    "rls_second_energy_interval": PREDICTION_FIGURE_DIR / "rls_second_energy_interval_5.0.png",
    "rls_final_energy_interval": PREDICTION_FIGURE_DIR / "rls_final_energy_interval_5.0.png",
    "energy_confidence": PREDICTION_FIGURE_DIR / "energy_confidence_interval_5.0.png",
    "cumulative_energy_confidence": PREDICTION_FIGURE_DIR / "cumulative_energy_confidence_interval_5.0.png",
    "dataset_split": RESULTS_FIGURE_DIR / "dataset_split_statistics_5.0.png",
    "data_distribution": RESULTS_FIGURE_DIR / "data_distribution_5.0.png",
    "rls_interval_width": RLS_DIR / "rls_final_energy_interval_width_5.0.png",
}
GIFS = {
    "rls_final_energy": RLS_DIR / "rls_final_energy_updates_5.0.gif",
    "rls_interval_width": RLS_DIR / "rls_interval_width_updates_5.0.gif",
}
REPORT_FILE = OUT_DIR / "outreadme.md"

SAMPLE_SECONDS = 1.0
DATA_CONTRACT_VERSION = "planned_trajectory_proxy_v2_independent_calibration"
WINDOW_CANDIDATES = (10, 30, 60)
FEATURE_GROUPS = {
    "motion_core": ["planned_speed_kmh", "planned_acceleration_kmhps", "planned_grade_percent"],
    "motion_time": ["time_s", "task_duration_s", "planned_speed_kmh", "planned_acceleration_kmhps", "planned_grade_percent"],
    "route_context": ["time_s", "task_duration_s", "planned_speed_kmh", "planned_acceleration_kmhps", "planned_distance_m", "planned_elevation_m", "planned_grade_percent", "planned_heading_deg"],
    "route_environment": ["time_s", "task_duration_s", "planned_speed_kmh", "planned_acceleration_kmhps", "planned_distance_m", "planned_elevation_m", "planned_grade_percent", "planned_heading_deg", "ambient_temperature_c"],
    "full_planning": ["time_s", "task_duration_s", "planned_speed_kmh", "planned_acceleration_kmhps", "planned_distance_m", "planned_elevation_m", "planned_grade_percent", "planned_heading_deg", "ambient_temperature_c", "planned_stop_flag"],
}
ABLATION_PARAMS = {"channels": 16, "kernel_size": 3, "dropout": 0.2, "learning_rate": 0.0005, "weight_decay": 0.001}
MODEL_CANDIDATES = (
    {"channels": 8, "kernel_size": 3, "dropout": 0.2, "learning_rate": 0.0005, "weight_decay": 0.001},
    {"channels": 16, "kernel_size": 3, "dropout": 0.3, "learning_rate": 0.0005, "weight_decay": 0.001},
    {"channels": 16, "kernel_size": 5, "dropout": 0.3, "learning_rate": 0.0003, "weight_decay": 0.003},
    {"channels": 32, "kernel_size": 3, "dropout": 0.2, "learning_rate": 0.0003, "weight_decay": 0.001},
)
SEARCH_SEEDS = (42,)
REFINE_SEEDS = (17, 42, 2026)
SEARCH_EPOCHS = 3
SEARCH_PATIENCE = 2
REFINE_TOP_CANDIDATES = 3
REFINE_EPOCHS = 6
SEARCH_TRAIN_STRIDE = 1
REFINE_TRAIN_STRIDE = 1
WINDOW_STEPS = 20
EPOCHS = 35
BATCH_SIZE = 2048
CALIBRATION_COVERAGE = 0.95
POWER_INTERVAL_COVERAGE = 0.90
TRAJECTORY_INTERVAL_COVERAGE = 0.90
POWER_INTERVAL_BINS_W = (1000.0, 5000.0, 15000.0)
RANDOM_SEED = 42
DEVICE = "cuda"
RLS_WINDOWS_SECONDS = (5, 10, 20)
RLS_FORGETTING_FACTORS = (0.90, 0.96, 0.99)
RLS_INITIAL_COVARIANCES = (1.0, 10.0)


def ensure_directories() -> None:
    """功能: 创建5.0数据、模型和运行结果目录。
    参数: 无。
    返回: None。
    调用位置: main.py和vehicle_pipeline.py。
    """

    for path in (PROCESSED_DIR, SAVE_DIR, OUT_MODEL_DIR, FIGURE_DIR, TRAINING_FIGURE_DIR, PREDICTION_FIGURE_DIR, RESULTS_FIGURE_DIR, PREDICTION_DIR, TASK_DIR, RLS_DIR, ROUTE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    for name in ("SOURCE_DIR", "DATA_DIR", "SAVE_DIR", "OUT_DIR"):
        print(f"{name}: {globals()[name]}")
