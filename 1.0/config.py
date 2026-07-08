# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-07-07
# 文件名: config.py
# 功能说明: 集中管理四轴无人机能耗预测实验1.0的路径、参数和输出文件名
# 版本号：1.0

from dataclasses import dataclass, replace
from pathlib import Path


# 项目代码目录
PROJECT_ROOT = Path(__file__).resolve().parent
# 原始数据和处理后数据目录
DATA_DIR = Path("D:/Python-files/Energy-prediction/data")
# 模型、权重、训练日志和评估结果保存目录
SAVE_DIR = Path("D:/Python-files/Energy-prediction/model")
# 预测输出和辅助图表目录
OUT_DIR = PROJECT_ROOT / "output"

# 在线数据源和原始数据文件
SOURCE_REPO_URL = "https://www.modelscope.cn/datasets/OmniData/Data_Collected_with_Package_etc.git"
SOURCE_REPO_DIR = DATA_DIR / "Data_Collected_with_Package_etc"
RAW_ZIP_FILE = SOURCE_REPO_DIR / "raw" / "12683453.zip"
RAW_DATA_DIR = DATA_DIR / "dji_matrice_100"
RAW_FLIGHTS_CSV = RAW_DATA_DIR / "flights.csv"
RAW_PARAMETERS_CSV = RAW_DATA_DIR / "parameters.csv"
RAW_README_FILE = RAW_DATA_DIR / "README.txt"

# 处理后数据和数据切分文件
PROCESSED_DIR = DATA_DIR / "processed_1.0"
CLEAN_DATA_CSV = PROCESSED_DIR / "uav_energy_features.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
FEATURE_META_JSON = PROCESSED_DIR / "feature_metadata.json"
DATASET_SUMMARY_JSON = PROCESSED_DIR / "dataset_summary.json"

# 模型、权重、日志和评估文件
SCALER_JSON = SAVE_DIR / "scaler_1.0.json"
BEST_MODEL_FILE = SAVE_DIR / "best_energy_mlp_1.0.pt"
FINAL_MODEL_FILE = SAVE_DIR / "final_energy_mlp_1.0.pt"
TRAINING_LOG_CSV = SAVE_DIR / "training_log_1.0.csv"
TUNING_RESULTS_CSV = SAVE_DIR / "tuning_results_1.0.csv"
LOSS_CURVE_FILE = SAVE_DIR / "loss_curve_1.0.png"
EVALUATION_JSON = SAVE_DIR / "evaluation_1.0.json"
EVALUATION_CSV = SAVE_DIR / "evaluation_1.0.csv"
FLIGHT_ENERGY_SUMMARY_CSV = SAVE_DIR / "flight_energy_summary_1.0.csv"
POWER_BIN_EVALUATION_CSV = SAVE_DIR / "power_bin_evaluation_1.0.csv"
PREDICTION_CSV = OUT_DIR / "test_predictions_1.0.csv"

# 训练和切分默认参数
TARGET_COLUMN = "power_w"
TARGET_TRANSFORM = "none"
RANDOM_SEED = 42
TEST_RATIO = 0.15
VAL_RATIO = 0.15
BATCH_SIZE = 2048
EPOCHS = 55
TUNE_EPOCHS = 14
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DEFAULT_DEVICE = "auto"
REQUIRE_GPU = True


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存实验1.0的路径和训练参数。
    参数: 无，字段由build_config统一构建。
    返回: 不直接返回，作为配置数据结构使用。
    调用位置: main.py、data_utils.py、train.py、predict.py、evaluate.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    save_dir: Path = SAVE_DIR
    out_dir: Path = OUT_DIR
    source_repo_url: str = SOURCE_REPO_URL
    source_repo_dir: Path = SOURCE_REPO_DIR
    raw_zip_file: Path = RAW_ZIP_FILE
    raw_data_dir: Path = RAW_DATA_DIR
    raw_flights_csv: Path = RAW_FLIGHTS_CSV
    raw_parameters_csv: Path = RAW_PARAMETERS_CSV
    raw_readme_file: Path = RAW_README_FILE
    processed_dir: Path = PROCESSED_DIR
    clean_data_csv: Path = CLEAN_DATA_CSV
    train_csv: Path = TRAIN_CSV
    val_csv: Path = VAL_CSV
    test_csv: Path = TEST_CSV
    feature_meta_json: Path = FEATURE_META_JSON
    dataset_summary_json: Path = DATASET_SUMMARY_JSON
    scaler_json: Path = SCALER_JSON
    best_model_file: Path = BEST_MODEL_FILE
    final_model_file: Path = FINAL_MODEL_FILE
    training_log_csv: Path = TRAINING_LOG_CSV
    tuning_results_csv: Path = TUNING_RESULTS_CSV
    loss_curve_file: Path = LOSS_CURVE_FILE
    evaluation_json: Path = EVALUATION_JSON
    evaluation_csv: Path = EVALUATION_CSV
    flight_energy_summary_csv: Path = FLIGHT_ENERGY_SUMMARY_CSV
    power_bin_evaluation_csv: Path = POWER_BIN_EVALUATION_CSV
    prediction_csv: Path = PREDICTION_CSV
    target_column: str = TARGET_COLUMN
    target_transform: str = TARGET_TRANSFORM
    random_seed: int = RANDOM_SEED
    test_ratio: float = TEST_RATIO
    val_ratio: float = VAL_RATIO
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    tune_epochs: int = TUNE_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    patience: int = PATIENCE
    device: str = DEFAULT_DEVICE
    require_gpu: bool = REQUIRE_GPU


def build_config(**overrides: object) -> ExperimentConfig:
    """功能: 根据命令行覆盖项构建实验配置。
    参数: overrides为需要覆盖的配置字段。
    返回: ExperimentConfig配置对象。
    调用位置: main.py。
    """

    cfg = ExperimentConfig()
    normalized = {}
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        normalized[key] = Path(value) if isinstance(current, Path) else value
    return replace(cfg, **normalized)


def ensure_directories(cfg: ExperimentConfig | None = None) -> None:
    """功能: 创建实验所需的数据、模型和输出目录。
    参数: cfg为实验配置对象，默认使用build_config生成。
    返回: None。
    调用位置: main.py、data_utils.py、train.py、predict.py。
    """

    cfg = cfg or build_config()
    for path in (cfg.data_dir, cfg.save_dir, cfg.out_dir, cfg.raw_data_dir, cfg.processed_dir):
        path.mkdir(parents=True, exist_ok=True)


def print_key_paths(cfg: ExperimentConfig | None = None) -> None:
    """功能: 打印关键路径并列出已有文件。
    参数: cfg为实验配置对象，默认使用build_config生成。
    返回: None。
    调用位置: config.py直接运行时。
    """

    cfg = cfg or build_config()
    ensure_directories(cfg)
    paths = {
        "PROJECT_ROOT": cfg.project_root,
        "DATA_DIR": cfg.data_dir,
        "SAVE_DIR": cfg.save_dir,
        "OUT_DIR": cfg.out_dir,
        "RAW_FLIGHTS_CSV": cfg.raw_flights_csv,
        "TRAIN_CSV": cfg.train_csv,
        "BEST_MODEL_FILE": cfg.best_model_file,
    }
    for name, path in paths.items():
        print(f"{name}: {path}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                print(f"  - {child.name}")


if __name__ == "__main__":
    print_key_paths()
