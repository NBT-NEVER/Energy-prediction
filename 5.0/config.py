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
FIGURE_DIR = OUT_DIR / "figures"
PREDICTION_DIR = OUT_DIR / "predictions"
LOG_DIR = OUT_DIR / "logs"

TRAIN_CSV = PROCESSED_DIR / "train_5.0.csv"
VAL_CSV = PROCESSED_DIR / "val_5.0.csv"
TEST_CSV = PROCESSED_DIR / "test_5.0.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_trips_5.0.csv"
DATA_SUMMARY_JSON = PROCESSED_DIR / "data_summary_5.0.json"
MODEL_FILE = SAVE_DIR / "vehicle_tcn_5.0.pt"
SCALER_FILE = SAVE_DIR / "scaler_5.0.json"
TRAIN_LOG_CSV = OUT_DIR / "training_log_5.0.csv"
SEARCH_RESULTS_CSV = OUT_DIR / "hyperparameter_search_5.0.csv"
SEARCH_TRIALS_CSV = OUT_DIR / "search_trials_5.0.csv"
RLS_SEARCH_CSV = OUT_DIR / "rls_search_5.0.csv"
SELECTED_CONFIG_JSON = OUT_DIR / "selected_config_5.0.json"
PREDICTIONS_CSV = PREDICTION_DIR / "test_predictions_5.0.csv"
RLS_WINDOWS_CSV = PREDICTION_DIR / "test_rls_windows_5.0.csv"
METRICS_JSON = OUT_DIR / "metrics_5.0.json"
TRIP_METRICS_CSV = OUT_DIR / "trip_metrics_5.0.csv"
FIGURES = {
    "loss": FIGURE_DIR / "loss_curves_5.0.png",
    "search": FIGURE_DIR / "hyperparameter_ranking_5.0.png",
    "rls_search": FIGURE_DIR / "rls_hyperparameter_ranking_5.0.png",
    "power_scatter": FIGURE_DIR / "test_power_scatter_5.0.png",
    "residual": FIGURE_DIR / "test_power_residuals_5.0.png",
    "trip_energy": FIGURE_DIR / "test_trip_energy_5.0.png",
    "trip_trace": FIGURE_DIR / "test_trip_prediction_5.0.png",
}
REPORT_FILE = OUT_DIR / "outreadme.md"

SAMPLE_SECONDS = 1.0
WINDOW_CANDIDATES = (10, 20, 40)
MODEL_CANDIDATES = (
    {"channels": 32, "kernel_size": 3, "dropout": 0.05, "learning_rate": 0.001, "weight_decay": 0.0001},
    {"channels": 32, "kernel_size": 3, "dropout": 0.15, "learning_rate": 0.0005, "weight_decay": 0.0001},
    {"channels": 32, "kernel_size": 5, "dropout": 0.10, "learning_rate": 0.0005, "weight_decay": 0.0001},
    {"channels": 64, "kernel_size": 3, "dropout": 0.10, "learning_rate": 0.0003, "weight_decay": 0.0001},
)
SEARCH_SEEDS = (17, 42)
REFINE_SEEDS = (17, 42, 2026)
SEARCH_EPOCHS = 10
SEARCH_PATIENCE = 3
REFINE_TOP_CANDIDATES = 3
REFINE_EPOCHS = 18
WINDOW_STEPS = 20
EPOCHS = 35
BATCH_SIZE = 512
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

    for path in (PROCESSED_DIR, SAVE_DIR, FIGURE_DIR, PREDICTION_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    for name in ("SOURCE_DIR", "DATA_DIR", "SAVE_DIR", "OUT_DIR"):
        print(f"{name}: {globals()[name]}")
