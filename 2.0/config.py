# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: config.py
# 开发时间: 2026-07-08
# 文件名: config.py
# 功能说明: 集中管理四轴无人机能耗预测实验2.0的路径、参数和输出文件名
# 版本号：2.0

from dataclasses import dataclass, replace
from pathlib import Path


# 项目代码目录
PROJECT_ROOT = Path(__file__).resolve().parent
# 原始公开数据目录，只存放下载或解压的源数据
DATA_DIR = Path("D:/Python-files/Energy-prediction/data")
# 模型训练权重目录，只保存 .pt 权重文件
SAVE_DIR = Path("D:/Python-files/Energy-prediction/model")
# 统一输出目录，处理后数据、日志、评估、预测和可视化都写入这里
OUT_DIR = PROJECT_ROOT / "out"
OUT_DATA_DIR = OUT_DIR / "data"
OUT_MODEL_DIR = OUT_DIR / "model"
FIGURE_DIR = OUT_DIR / "figures"
PREDICTION_DIR = OUT_DIR / "predictions"
CUSTOM_DIR = OUT_DIR / "custom"

# 在线数据源和原始数据文件
SOURCE_REPO_URL = "https://www.modelscope.cn/datasets/OmniData/Data_Collected_with_Package_etc.git"
SOURCE_REPO_DIR = DATA_DIR / "Data_Collected_with_Package_etc"
RAW_ZIP_FILE = SOURCE_REPO_DIR / "raw" / "12683453.zip"
RAW_DATA_DIR = DATA_DIR / "dji_matrice_100"
RAW_FLIGHTS_CSV = RAW_DATA_DIR / "flights.csv"
RAW_PARAMETERS_CSV = RAW_DATA_DIR / "parameters.csv"
RAW_README_FILE = RAW_DATA_DIR / "README.txt"

# 处理后数据和数据切分文件
PROCESSED_DIR = OUT_DATA_DIR / "processed_2.0"
CLEAN_DATA_CSV = PROCESSED_DIR / "uav_energy_features.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
FEATURE_META_JSON = PROCESSED_DIR / "feature_metadata.json"
DATASET_SUMMARY_JSON = PROCESSED_DIR / "dataset_summary.json"

# 模型权重文件，只保存在 SAVE_DIR
BEST_MODEL_FILE = SAVE_DIR / "best_energy_tcn_rls_2.0.pt"
FINAL_MODEL_FILE = SAVE_DIR / "final_energy_tcn_rls_2.0.pt"

# 训练记录、评估结果和预测输出文件
SCALER_JSON = OUT_MODEL_DIR / "scaler_2.0.json"
TRAINING_LOG_CSV = OUT_MODEL_DIR / "training_log_2.0.csv"
TUNING_RESULTS_CSV = OUT_MODEL_DIR / "tuning_results_2.0.csv"
EVALUATION_JSON = OUT_MODEL_DIR / "evaluation_2.0.json"
EVALUATION_CSV = OUT_MODEL_DIR / "evaluation_2.0.csv"
FLIGHT_ENERGY_SUMMARY_CSV = OUT_MODEL_DIR / "flight_energy_summary_2.0.csv"
POWER_BIN_EVALUATION_CSV = OUT_MODEL_DIR / "power_bin_evaluation_2.0.csv"
PREDICTION_CSV = PREDICTION_DIR / "test_predictions_2.0.csv"
UNCERTAINTY_CALIBRATION_NPZ = OUT_MODEL_DIR / "uncertainty_calibration_2.0.npz"
UNCERTAINTY_CALIBRATION_JSON = OUT_MODEL_DIR / "uncertainty_calibration_2.0.json"

# 可视化和自定义工况输出文件
LOSS_CURVE_FILE = FIGURE_DIR / "training" / "loss_curve_2.0.png"
TRAINING_VIS_DIR = FIGURE_DIR / "training"
RESULT_VIS_DIR = FIGURE_DIR / "results"
PREDICTION_VIS_DIR = FIGURE_DIR / "prediction"
CUSTOM_VIS_DIR = FIGURE_DIR / "custom"
CUSTOM_SCENARIO_CSV = CUSTOM_DIR / "custom_scenarios_2.0.csv"
CUSTOM_PREDICTION_CSV = CUSTOM_DIR / "custom_predictions_2.0.csv"
CUSTOM_PREDICTION_SUMMARY_JSON = CUSTOM_DIR / "custom_prediction_summary_2.0.json"
VISUALIZATION_SUMMARY_JSON = OUT_DIR / "visualization_summary_2.0.json"

# 训练和切分默认参数
TARGET_COLUMN = "power_w"
TARGET_TRANSFORM = "none"
RANDOM_SEED = 42
TEST_RATIO = 0.15
VAL_RATIO = 0.15
BATCH_SIZE = 2048
EPOCHS = 80
TUNE_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DEFAULT_DEVICE = "cuda"
# TCN/RLS短时间窗候选值，单位为秒；训练时会折算为采样步数并写入调参结果。
WINDOW_SECONDS_CANDIDATES = (0.6, 1.0, 1.5, 2.0, 3.0)
DEFAULT_WINDOW_SECONDS = 1.5
DEFAULT_CONFIDENCE = 0.95
# TCN残差块通道；4个块对应更深的时间特征提取网络
TCN_CHANNELS = (64, 64, 64, 32)
RLS_FORGETTING_FACTOR = 0.995
RLS_INITIAL_COVARIANCE = 1000.0


@dataclass(frozen=True)
class ExperimentConfig:
    """功能: 保存实验2.0的路径、TCN窗口和RLS参数。
    参数: 无，字段由build_config统一构建。
    返回: 不直接返回，作为配置数据结构使用。
    调用位置: main.py、data_utils.py、train.py、predict.py、evaluate.py、visualize.py。
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    save_dir: Path = SAVE_DIR
    out_dir: Path = OUT_DIR
    out_data_dir: Path = OUT_DATA_DIR
    out_model_dir: Path = OUT_MODEL_DIR
    figure_dir: Path = FIGURE_DIR
    prediction_dir: Path = PREDICTION_DIR
    custom_dir: Path = CUSTOM_DIR
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
    uncertainty_calibration_npz: Path = UNCERTAINTY_CALIBRATION_NPZ
    uncertainty_calibration_json: Path = UNCERTAINTY_CALIBRATION_JSON
    training_vis_dir: Path = TRAINING_VIS_DIR
    result_vis_dir: Path = RESULT_VIS_DIR
    prediction_vis_dir: Path = PREDICTION_VIS_DIR
    custom_vis_dir: Path = CUSTOM_VIS_DIR
    custom_scenario_csv: Path = CUSTOM_SCENARIO_CSV
    custom_prediction_csv: Path = CUSTOM_PREDICTION_CSV
    visualization_summary_json: Path = VISUALIZATION_SUMMARY_JSON
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
    window_seconds_candidates: tuple[float, ...] = WINDOW_SECONDS_CANDIDATES
    default_window_seconds: float = DEFAULT_WINDOW_SECONDS
    default_confidence: float = DEFAULT_CONFIDENCE
    tcn_channels: tuple[int, ...] = TCN_CHANNELS
    rls_forgetting_factor: float = RLS_FORGETTING_FACTOR
    rls_initial_covariance: float = RLS_INITIAL_COVARIANCE
    custom_prediction_summary_json: Path = CUSTOM_PREDICTION_SUMMARY_JSON


def build_config(**overrides: object) -> ExperimentConfig:
    """功能: 根据命令行覆盖项构建实验配置并同步派生路径。
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
    if "data_dir" in normalized:
        data_dir = normalized["data_dir"]
        source_repo_dir = data_dir / "Data_Collected_with_Package_etc"
        raw_data_dir = data_dir / "dji_matrice_100"
        normalized.setdefault("source_repo_dir", source_repo_dir)
        normalized.setdefault("raw_zip_file", source_repo_dir / "raw" / "12683453.zip")
        normalized.setdefault("raw_data_dir", raw_data_dir)
        normalized.setdefault("raw_flights_csv", raw_data_dir / "flights.csv")
        normalized.setdefault("raw_parameters_csv", raw_data_dir / "parameters.csv")
        normalized.setdefault("raw_readme_file", raw_data_dir / "README.txt")
    if "save_dir" in normalized:
        save_dir = normalized["save_dir"]
        normalized.setdefault("best_model_file", save_dir / "best_energy_tcn_rls_2.0.pt")
        normalized.setdefault("final_model_file", save_dir / "final_energy_tcn_rls_2.0.pt")
    if "out_dir" in normalized:
        out_dir = normalized["out_dir"]
        out_data_dir = out_dir / "data"
        out_model_dir = out_dir / "model"
        figure_dir = out_dir / "figures"
        prediction_dir = out_dir / "predictions"
        custom_dir = out_dir / "custom"
        processed_dir = out_data_dir / "processed_2.0"
        normalized.setdefault("out_data_dir", out_data_dir)
        normalized.setdefault("out_model_dir", out_model_dir)
        normalized.setdefault("figure_dir", figure_dir)
        normalized.setdefault("prediction_dir", prediction_dir)
        normalized.setdefault("custom_dir", custom_dir)
        normalized.setdefault("processed_dir", processed_dir)
        normalized.setdefault("clean_data_csv", processed_dir / "uav_energy_features.csv")
        normalized.setdefault("train_csv", processed_dir / "train.csv")
        normalized.setdefault("val_csv", processed_dir / "val.csv")
        normalized.setdefault("test_csv", processed_dir / "test.csv")
        normalized.setdefault("feature_meta_json", processed_dir / "feature_metadata.json")
        normalized.setdefault("dataset_summary_json", processed_dir / "dataset_summary.json")
        normalized.setdefault("scaler_json", out_model_dir / "scaler_2.0.json")
        normalized.setdefault("training_log_csv", out_model_dir / "training_log_2.0.csv")
        normalized.setdefault("tuning_results_csv", out_model_dir / "tuning_results_2.0.csv")
        normalized.setdefault("evaluation_json", out_model_dir / "evaluation_2.0.json")
        normalized.setdefault("evaluation_csv", out_model_dir / "evaluation_2.0.csv")
        normalized.setdefault("flight_energy_summary_csv", out_model_dir / "flight_energy_summary_2.0.csv")
        normalized.setdefault("power_bin_evaluation_csv", out_model_dir / "power_bin_evaluation_2.0.csv")
        normalized.setdefault("prediction_csv", prediction_dir / "test_predictions_2.0.csv")
        normalized.setdefault("uncertainty_calibration_npz", out_model_dir / "uncertainty_calibration_2.0.npz")
        normalized.setdefault("uncertainty_calibration_json", out_model_dir / "uncertainty_calibration_2.0.json")
        normalized.setdefault("loss_curve_file", figure_dir / "training" / "loss_curve_2.0.png")
        normalized.setdefault("training_vis_dir", figure_dir / "training")
        normalized.setdefault("result_vis_dir", figure_dir / "results")
        normalized.setdefault("prediction_vis_dir", figure_dir / "prediction")
        normalized.setdefault("custom_vis_dir", figure_dir / "custom")
        normalized.setdefault("custom_scenario_csv", custom_dir / "custom_scenarios_2.0.csv")
        normalized.setdefault("custom_prediction_csv", custom_dir / "custom_predictions_2.0.csv")
        normalized.setdefault("visualization_summary_json", out_dir / "visualization_summary_2.0.json")
        normalized.setdefault("custom_prediction_summary_json", custom_dir / "custom_prediction_summary_2.0.json")
    return replace(cfg, **normalized)


def ensure_directories(cfg: ExperimentConfig | None = None) -> None:
    """功能: 创建实验所需的数据、模型、预测和图表输出目录。
    参数: cfg为实验配置对象，默认使用build_config生成。
    返回: None。
    调用位置: main.py、data_utils.py、train.py、predict.py、evaluate.py、visualize.py。
    """

    cfg = cfg or build_config()
    paths = (
        cfg.data_dir,
        cfg.save_dir,
        cfg.out_dir,
        cfg.out_data_dir,
        cfg.out_model_dir,
        cfg.figure_dir,
        cfg.prediction_dir,
        cfg.custom_dir,
        cfg.raw_data_dir,
        cfg.processed_dir,
        cfg.training_vis_dir,
        cfg.result_vis_dir,
        cfg.prediction_vis_dir,
        cfg.custom_vis_dir,
    )
    for path in paths:
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
        "OUT_MODEL_DIR": cfg.out_model_dir,
        "RAW_FLIGHTS_CSV": cfg.raw_flights_csv,
        "TRAIN_CSV": cfg.train_csv,
        "BEST_MODEL_FILE": cfg.best_model_file,
        "UNCERTAINTY_CALIBRATION_NPZ": cfg.uncertainty_calibration_npz,
        "PREDICTION_CSV": cfg.prediction_csv,
    }
    for name, path in paths.items():
        print(f"{name}: {path}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                print(f"  - {child.name}")


if __name__ == "__main__":
    print_key_paths()
