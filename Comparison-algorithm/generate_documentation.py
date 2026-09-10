# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: generate_documentation.py
# 开发时间: 2026-09-10
# 文件名: generate_documentation.py
# 功能说明: 根据最终对比实验输出生成论文式总README和各算法README
# 版本号：3.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALGORITHMS, FEATURE_COLUMNS, FIGURE_NAMES, SEQUENCE_WINDOW, TEST_CSV, TRAIN_CSV, VAL_CSV


ROOT = Path(__file__).resolve().parent
METRICS_CSV = ROOT / "out" / "comparison_metrics.csv"
FEATURE_META_JSON = ROOT.parent / "3.0" / "out" / "data" / "processed_3.0" / "feature_metadata.json"
BASELINE_EVALUATION_JSON = ROOT.parent / "3.0" / "out" / "model" / "evaluation_3.0.json"

PHASES = ("idle", "ascent", "descent", "cruise")
PHASE_CN = {"idle": "停机/低速", "ascent": "上升", "descent": "下降", "cruise": "巡航"}
POWER_EDGES = (-np.inf, 50.0, 300.0, 450.0, 600.0, np.inf)
POWER_BINS = ("0–50 W", "50–300 W", "300–450 W", "450–600 W", ">600 W")


ALGORITHM_REFERENCES = {
    "proposed_tcn_rls": "<sup>[A0]</sup><sup>[A6]</sup><sup>[A7]</sup>",
    "rf_tlatt_lite": "<sup>[A1]</sup>",
    "physical_mlr": "<sup>[A2]</sup>",
    "lstm": "<sup>[A3]</sup>",
    "lr_tcn_sma": "<sup>[A4]</sup>",
    "lstm_transformer": "<sup>[A5]</sup>",
    "cnn_lstm": "<sup>[A1]</sup>",
}


BACKGROUND_REFERENCES = """### 10.2 背景与应用参考文献

- **[B1]** Rodrigues, T. A., Patrikar, J., Choudhry, A., et al. *In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery*. Scientific Data, 2021, 8(1): 155. DOI: 10.1038/s41597-021-00930-x.
- **[B2]** Dorling, K., Heinrichs, J., Messier, G. G., Magierowski, S. *Vehicle Routing Problems for Drone Delivery*. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2017, 47(1): 70–85. DOI: 10.1109/TSMC.2016.2582745.
- **[B3]** Cabuk, U. C., Tosun, M., Dagdeviren, O., Ozturk, Y. *Modeling Energy Consumption of Small Drones for Swarm Missions*. IEEE Transactions on Intelligent Transportation Systems, 2024, 25(8): 10176–10189. DOI: 10.1109/TITS.2024.3350042.
- **[B4]** Prasetia, A. S., Wai, R.-J., Wen, Y.-L., Wang, Y.-K. *Mission-Based Energy Consumption Prediction of Multirotor UAV*. IEEE Access, 2019, 7: 33055–33063. DOI: 10.1109/ACCESS.2019.2903644.
- **[B5]** Dietrich, T., Krug, S., Zimmermann, A. *An Empirical Study on Generic Multicopter Energy Consumption Profiles*. 2017 Annual IEEE International Systems Conference (SysCon), 2017. DOI: 10.1109/SYSCON.2017.7934762.
"""


ALGORITHM_REFERENCE_ENTRIES = {
    "A0": "**[A0]** Energy-prediction 3.0 本项目 TCN+RLS 方法及固定模型输出；TCN 与 RLS 的理论依据分别见 [A6]、[A7]，实验结果组织方式参照 [A1]。",
    "A1": "**[A1]** Luo, W., Li, N., Xiong, Z., Chen, W., Li, Y., Tang, C., Li, Y., Dong, C. *Phase-based power prediction for quadrotor UAVs with RF-TLATT*. Energy, 2025, 335: 138208. DOI: 10.1016/j.energy.2025.138208.",
    "A2": "**[A2]** Jastrzębska, A., Lerke, M., Kwiatkowski, W. *Prediction of energy consumption in unmanned aerial vehicles*. Electric Power Systems Research, 2026, 257: 113008. DOI: 10.1016/j.epsr.2026.113008.",
    "A3": "**[A3]** Muli, C., Park, S., Liu, M. *A Comparative Study on Energy Consumption Models for Drones*. In: Internet of Things, GIoTS 2022, Lecture Notes in Computer Science, vol. 13533, Springer, 2022, pp. 199–210. DOI: 10.1007/978-3-031-20936-9_16.",
    "A4": "**[A4]** Dudukcu, H. V., Taskiran, M., Kahraman, N. *UAV instantaneous power consumption prediction using LR-TCN with simple moving average*. Concurrency and Computation: Practice and Experience, 2024, 36(3): e7913. DOI: 10.1002/cpe.7913.",
    "A5": "**[A5]** Feng, Z., Zhang, J., Jiang, H., Yao, X., Qian, Y., Zhang, H. *Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework*. Energy, 2024, 302: 131780. DOI: 10.1016/j.energy.2024.131780.",
    "A6": "**[A6]** Bai, S., Kolter, J. Z., Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271, 2018. DOI: 10.48550/arXiv.1803.01271.",
    "A7": "**[A7]** Sayed, A. H. *Adaptive Filters*. Wiley, 2008. DOI: 10.1002/9780470374122.",
}


ALGORITHM_REFERENCE_KEYS = {
    "proposed_tcn_rls": ("A0", "A1", "A6", "A7"),
    "rf_tlatt_lite": ("A1",),
    "physical_mlr": ("A2",),
    "lstm": ("A3",),
    "lr_tcn_sma": ("A4",),
    "lstm_transformer": ("A5",),
    "cnn_lstm": ("A1",),
}


ALGORITHM_REFERENCE_LIST = """### 10.1 对比算法与方法来源文献

""" + "\n".join(f"- {ALGORITHM_REFERENCE_ENTRIES[key]}" for key in ALGORITHM_REFERENCE_ENTRIES) + "\n"


ALGORITHM_INPUTS = {
    "proposed_tcn_rls": "23 个公共特征，100 步（12 s）历史至当前窗口；末端 TCN 输出再由上一完整秒窗更新的 RLS 参数校正",
    "rf_tlatt_lite": "从公共字段构造 16 个当前时刻物理与交互量；无序列窗口，按预测专用速度阈值选择相位回归头",
    "physical_mlr": "从公共字段构造 7 个当前时刻物理量；无序列窗口，使用单一全局回归及 moving 指示变量",
    "lstm": "23 个公共特征，17 步（约 2.0 s）历史至当前窗口",
    "lr_tcn_sma": "23 个公共特征先按 flight 做 10 点历史 SMA，再组成 17 步（约 2.0 s）窗口",
    "lstm_transformer": "23 个公共特征，17 步（约 2.0 s）历史至当前窗口",
    "cnn_lstm": "23 个公共特征，17 步（约 2.0 s）历史至当前窗口",
}


ALGORITHM_APPLICATIONS = {
    "proposed_tcn_rls": "适合飞行中能够持续取得电压、电流或窗口能量反馈的在线功率跟踪和剩余能量修正。TCN 负责从历史工况提取非线性动态，RLS 用已结束秒窗校正后续窗口，因此更适合传感器闭环部署；若部署端没有真实能量反馈，应只使用 TCN 原始输出，不能预期本实验中的 RLS 能耗优势。",
    "rf_tlatt_lite": "对应相位差异明显的四旋翼功率预测场景，例如上升、下降、低速和巡航分别具有不同功率关系。当前轻量实现只需速度和当前工况即可选择回归头，计算量低；它没有论文中的随机森林相位分类器和 Transformer-LSTM 预测器，适合作为相位建模思想的低成本对照。",
    "physical_mlr": "适合需要系数可解释、算力受限或只掌握当前载荷、速度、风速等物理量的任务前粗估。线性项便于检查质量、垂直运动和风对功率的方向性影响，但当前单一全局回归无法显式记忆状态切换，也没有复现论文的静止/飞行两套子模型。",
    "lstm": "适合速度、姿态和外部负载随时间连续变化的短窗功率估计。双向分支在已缓存的 17 步历史至当前窗口内从两端编码，能够利用窗口内部的上下文；部署时仍需先缓存完整窗口，且本实现不会读取预测时刻之后的样本。",
    "lr_tcn_sma": "适合传感器噪声较明显、又要求卷积推理可并行的在线瞬时功率估计。SMA 抑制高频抖动，因果卷积只读取当前及历史样本；代价是快速起降或负载突变可能被平滑，尖峰位置和幅值需结合局部放大图检查。",
    "lstm_transformer": "原文面向电动汽车能耗，本实验将其短期 LSTM 与窗口内自注意力结构迁移到 UAV。它适合 17 步窗口中既有局部变化又有跨位置关联的工况，但这里的窗口只有约 2.0 s，结果只能说明短窗依赖建模效果，不能外推为整段航程的长期预测能力。",
    "cnn_lstm": "适合先从相邻采样点提取局部波形，再由 LSTM 汇总短时演化的功率估计。当前来源仅为 Luo 等论文表 8 的二手算法名称，所给材料无法核实原始 CNN-LSTM 论文和完整层配置，因此本实现是明确标注的工程适配基线。",
}


REPLACED_ALGORITHM_RESULTS = (
    {
        "name": "任务级多项式 Elastic Net",
        "reference": "<sup>[B4]</sup>",
        "power_mae": 107.736,
        "power_wape": 26.516,
        "energy_wape": 4.982,
        "replacement": "LSTM-Transformer",
        "replacement_power_wape": 7.293,
        "replacement_energy_wape": 2.542,
    },
    {
        "name": "经验多相位能耗剖面",
        "reference": "<sup>[B5]</sup>",
        "power_mae": 73.492,
        "power_wape": 18.088,
        "energy_wape": 6.779,
        "replacement": "CNN-LSTM",
        "replacement_power_wape": 7.770,
        "replacement_energy_wape": 2.997,
    },
)


FEATURE_ENGLISH = {
    "time": ("飞行内相对时间", "Flight-relative time", "s"),
    "dt_seconds": ("采样时间间隔", "Sampling time interval", "s"),
    "flight_progress": ("飞行进度", "Flight progress", "0–1"),
    "wind_speed": ("风速", "Wind speed", "m/s"),
    "wind_sin": ("风向正弦分量", "Sine of wind direction", "—"),
    "wind_cos": ("风向余弦分量", "Cosine of wind direction", "—"),
    "programmed_speed_mps": ("规划速度", "Programmed speed", "m/s"),
    "actual_speed_mps": ("实际合速度", "Actual resultant speed", "m/s"),
    "horizontal_speed_mps": ("水平速度", "Horizontal speed", "m/s"),
    "vertical_speed_mps": ("垂直速度", "Vertical speed", "m/s"),
    "vertical_speed_abs_mps": ("垂直速度绝对值", "Absolute vertical speed", "m/s"),
    "relative_air_speed_mps": ("相对空速", "Relative air speed", "m/s"),
    "wind_alignment": ("风速方向对齐度", "Wind alignment cosine", "−1–1"),
    "wind_cross_component_mps": ("横向风速分量", "Crosswind component", "m/s"),
    "payload_kg": ("载荷质量", "Payload mass", "kg"),
    "altitude_m": ("飞行高度", "Flight altitude", "m"),
    "dynamic_accel_norm": ("动态加速度模", "Dynamic acceleration norm", "m/s²"),
    "angular_rate_norm": ("角速度模", "Angular-rate norm", "rad/s"),
    "obstacle_agility_index": ("障碍机动代理指标", "Obstacle-agility proxy index", "—"),
    "thermal_load_proxy": ("热负荷代理量", "Thermal-load proxy", "—"),
    "vision_energy_proxy_w": ("视觉计算附加功率代理量", "Vision-energy power proxy", "W"),
    "communication_energy_proxy_w": ("通信附加功率代理量", "Communication-energy power proxy", "W"),
    "route_R1": ("R1航线独热变量", "R1 route one-hot variable", "0/1"),
}


def phase_label(frame: pd.DataFrame, vertical_threshold: float = 0.15,
                horizontal_threshold: float = 1.0) -> np.ndarray:
    """功能: 用速度阈值生成与绘图代码一致的四阶段标签。
    参数: frame为含垂直和水平速度的数据表；阈值单位为m/s。
    返回: 每条记录的阶段英文名称数组。
    调用位置: data_summary和README生成函数。
    """
    vertical = frame["vertical_speed_mps"].to_numpy(float)
    horizontal = frame["horizontal_speed_mps"].to_numpy(float)
    labels = np.full(len(frame), "cruise", dtype=object)
    labels[np.abs(vertical) < vertical_threshold] = "idle"
    labels[vertical > vertical_threshold] = "ascent"
    labels[vertical < -vertical_threshold] = "descent"
    labels[(np.abs(vertical) < vertical_threshold) & (horizontal > horizontal_threshold)] = "cruise"
    return labels


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """功能: 读取固定训练、验证、测试划分和特征元数据。
    参数: 无，路径来自config.py。
    返回: train、validation、test和feature metadata表。
    调用位置:文档入口。
    """
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    validation = pd.read_csv(VAL_CSV, encoding="utf-8")
    test = pd.read_csv(TEST_CSV, encoding="utf-8")
    metadata = json.loads(FEATURE_META_JSON.read_text(encoding="utf-8"))
    return train, validation, test, metadata


def read_algorithm_data(folder: str, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """功能:读取单个算法的预测表和flight能耗表并补充诊断字段。
    参数: folder为算法目录名；test为公共测试表。
    返回: 逐点诊断表和flight汇总表。
    调用位置:overall和algorithm_readme。
    """
    prediction_path = ROOT / folder / "out" / "predictions.csv"
    summary_path = ROOT / folder / "out" / "flight_energy_summary.csv"
    prediction = pd.read_csv(prediction_path, encoding="utf-8")
    summary = pd.read_csv(summary_path, encoding="utf-8")
    if len(prediction) != len(test):
        raise ValueError(f"{folder} predictions.csv rows {len(prediction)} != test rows {len(test)}")
    prediction["residual_w"] = prediction["predicted_power_w"] - prediction["power_w"]
    prediction["absolute_error_w"] = prediction["residual_w"].abs()
    prediction["phase_name"] = phase_label(test)
    prediction["power_bin"] = pd.cut(prediction["power_w"], POWER_EDGES, labels=POWER_BINS)
    return prediction, summary


def regression_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """功能:为文档计算一组回归统计量。
    参数: y为真实值；predicted为预测值。
    返回: MAE、RMSE、WAPE和R2。
    调用位置:分析段落生成。
    """
    y = np.asarray(y, float)
    predicted = np.asarray(predicted, float)
    error = predicted - y
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(y)), 1e-12) * 100),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator > 1e-12 else 0.0,
    }


def data_summary(folder: str, test: pd.DataFrame) -> dict[str, object]:
    """功能:提取单算法图表所需的数值和误差分位数。
    参数: folder为算法目录；test为公共测试集。
    返回:可直接插入Markdown的统计字典。
    调用位置:algorithm_readme和overall。
    """
    prediction, flights = read_algorithm_data(folder, test)
    power = regression_metrics(prediction.power_w, prediction.predicted_power_w)
    energy = regression_metrics(flights.actual_energy_wh, flights.predicted_energy_wh)
    phase_rows = []
    for phase in PHASES:
        part = prediction[prediction.phase_name == phase]
        values = regression_metrics(part.power_w, part.predicted_power_w)
        phase_rows.append({"phase": phase, "n": len(part), **values})
    bin_rows = []
    for power_bin in POWER_BINS:
        part = prediction[prediction.power_bin == power_bin]
        values = regression_metrics(part.power_w, part.predicted_power_w) if len(part) else {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "r2": float("nan")}
        bin_rows.append({"bin": power_bin, "n": len(part), **values})
    tuning_path = ROOT / folder / "out" / "model" / "tuning_results.csv"
    tuning = pd.read_csv(tuning_path, encoding="utf-8") if tuning_path.exists() else pd.DataFrame()
    selected = tuning[tuning.get("selected", pd.Series(dtype=bool)).astype(bool)] if not tuning.empty and "selected" in tuning else pd.DataFrame()
    return {
        "prediction": prediction,
        "flights": flights,
        "power": power,
        "energy": energy,
        "phase_rows": phase_rows,
        "bin_rows": bin_rows,
        "tuning": tuning,
        "selected": selected.iloc[0].to_dict() if len(selected) else {},
    }


def fmt(value: object, digits: int = 3) -> str:
    """功能:格式化文档中的数字并处理缺失值。
    参数: value为数值；digits为小数位数。
    返回:适合Markdown的字符串。
    调用位置:全部文档生成函数。
    """
    try:
        number = float(value)
        return "—" if not np.isfinite(number) else f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "—" if value is None else str(value)


def params_text(params: dict[str, object]) -> str:
    """功能:把调参CSV选中行转换为可读参数串。
    参数: params为选中候选字典。
    返回: 参数名和值组成的中文短句。
    调用位置:算法README生成。
    """
    if not params:
        return "未找到候选记录"
    ignored = {"candidate", "selected", "selection_score", "validation_loss", "best_epoch",
               "validation_power_mae_w", "validation_power_wape_percent", "validation_flight_energy_wape_percent"}
    pieces = []
    for key, value in params.items():
        if key in ignored:
            continue
        pieces.append(f"`{key}={fmt(value, 4)}`")
    return "、".join(pieces)


def algorithm_reference_markdown(folder: str) -> str:
    """功能:生成单算法README所需的来源文献列表。
    参数:folder为算法目录名。
    返回:只包含直接来源和必要理论来源的Markdown。
    调用位置:build_algorithm_readme。
    """
    keys = ALGORITHM_REFERENCE_KEYS[folder]
    lines = [f"- {ALGORITHM_REFERENCE_ENTRIES[key]}" for key in keys]
    if folder == "cnn_lstm":
        lines += [
            "",
            "> 来源层级说明：Luo 等只在表 8 报告 CNN-LSTM，并在正文写作未编号的 “Xuebing et al. (2024)”；其参考文献表缺少相应题录。所给材料无法核实该模型的一手作者、题名和层配置，因此 [A1] 在本目录属于二手来源。",
        ]
    return "\n".join(lines)


def algorithm_input_table(metrics: pd.DataFrame) -> str:
    """功能:列出各算法实际使用的特征形式和时间范围。
    参数:metrics为总体结果表。
    返回:Markdown输入对照表。
    调用位置:build_overall_readme。
    """
    lines = [
        "| 算法 | 实际输入与时间范围 |",
        "|---|---|",
    ]
    for folder in metrics.algorithm:
        algorithm_name = metrics.loc[metrics.algorithm == folder, "algorithm_name"].iloc[0]
        lines.append(f"| {algorithm_name} | {ALGORITHM_INPUTS[folder]} |")
    return "\n".join(lines)


def replacement_table() -> str:
    """功能:记录两项淘汰候选及新算法在同一测试集上的结果。
    参数:无，数据来自本轮替换前后已完成的固定实验。
    返回:Markdown候选替换表和量化说明。
    调用位置:build_overall_readme。
    """
    lines = [
        "| 被淘汰候选 | 来源 | 原功率 MAE(W) | 原功率 WAPE(%) | 原能耗 WAPE(%) | 替换算法 | 新功率 WAPE(%) | 新能耗 WAPE(%) |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in REPLACED_ALGORITHM_RESULTS:
        lines.append(
            f"| {row['name']} | {row['reference']} | {row['power_mae']:.3f} | {row['power_wape']:.3f} | "
            f"{row['energy_wape']:.3f} | {row['replacement']} | {row['replacement_power_wape']:.3f} | "
            f"{row['replacement_energy_wape']:.3f} |"
        )
    first, second = REPLACED_ALGORITHM_RESULTS
    return "\n".join(lines) + (
        f"\n\nLSTM-Transformer 相对任务级多项式 Elastic Net 的功率 WAPE 下降 "
        f"{(first['power_wape']-first['replacement_power_wape'])/first['power_wape']*100:.2f}%，能耗 WAPE 下降 "
        f"{(first['energy_wape']-first['replacement_energy_wape'])/first['energy_wape']*100:.2f}%；CNN-LSTM 相对经验多相位能耗剖面的"
        f"功率 WAPE 下降 {(second['power_wape']-second['replacement_power_wape'])/second['power_wape']*100:.2f}%，"
        f"能耗 WAPE 下降 {(second['energy_wape']-second['replacement_energy_wape'])/second['energy_wape']*100:.2f}%。"
        "两项旧候选已从有效算法清单和项目目录中删除，表中只保留替换依据。"
    )


ALGORITHM_SPECS = {
    "proposed_tcn_rls": {
        "title": "3.0 TCN + 秒级能量 RLS（主方法）",
        "source": "本项目 3.0 实现；TCN 与 RLS 理论分别参考 Bai 等和 Sayed，实验图表组织参考 Luo 等人的 RF-TLATT 研究",
        "structure": """输入张量为 $B\\times100\\times23$：$B$ 是批量大小，100 是从历史到当前的时间步数，23 是每步特征数。该 checkpoint 的窗口覆盖 12 s，测试集典型采样间隔约 0.12 s。网络共有 81,049 个可训练参数。

1. **时间门控层**：取 23 维输入中已标准化的 `dt_seconds`，经 `Linear(1,8) → SiLU → Linear(8,1)` 得到形状 $B\\times100\\times1$ 的门值 $g_t=1+0.2\\tanh(\\cdot)$，再广播乘到同一时刻的 23 个特征。门值范围为 0.8–1.2，使不规则采样间隔参与输入缩放。
2. **TCN 残差块 1**：两层 $k=3$、膨胀率 $d=1$ 的左填充因果卷积，通道依次为 $23\\to64\\to64$。每层依次执行 GroupNorm（1 组）、SiLU 和 Dropout(0.08)；旁路用 $1\\times1$ 卷积把 23 维映射到 64 维，主路与旁路相加后执行 ReLU。
3. **TCN 残差块 2**：通道为 $64\\to64\\to64$，两层卷积的膨胀率均为 2。输入输出通道相同，残差旁路为恒等映射。
4. **TCN 残差块 3**：通道保持 64，两层卷积的膨胀率均为 4，用更稀疏的历史位置补充中尺度变化。
5. **TCN 残差块 4**：通道为 $64\\to32\\to32$，膨胀率为 8，并用 $1\\times1$ 旁路降到 32 维。四个残差块共有 8 层时间卷积，理论感受野为 $1+2(k-1)(1+2+4+8)=61$ 步，且每层都只在左侧补零，不读取预测时刻之后的数据。
6. **双支路回归头**：主路取 TCN 最后时刻的 32 维向量，执行 `Linear(32,16) → SiLU → Dropout(0.08) → Linear(16,1)`；旁路只取当前时刻原始 23 维输入，执行 `LayerNorm(23) → Linear(23,16) → SiLU → Linear(16,1)`。两路标量相加后反标准化为 TCN 功率。
7. **在线 RLS 校正层**：每个 flight 从 $[\\theta_0,\\theta_1]=[0,1]$ 和 $0.25I$ 协方差开始。程序先用旧参数预测当前完整秒窗内的所有点；当该窗累计时长不小于 0.95 s 时，才用其真实能量与 TCN 能量更新参数，更新结果从下一秒窗生效。遗忘因子为 0.90，偏置与缩放分别限制在 $[-1,1]$ 和 $[0,2]$。当 TCN 窗口均值跨过 50 W 的飞行状态阈值时恢复中性参数，避免停机与飞行状态共用一组校正量。""",
        "formula": """$$\\hat P_{f,i}^{\\mathrm{RLS}}=\\max\\left(0,\\;s\\theta_{0,f,k}+\\theta_{1,f,k}\\hat P_{f,i}^{\\mathrm{TCN}}\\right)$$
$$K_k=\\frac{P_k\\phi_k}{\\lambda+\\phi_k^\\mathsf{T}P_k\\phi_k},\\quad \\theta_{k+1}=\\theta_k+K_k(y_k-\\phi_k^\\mathsf{T}\\theta_k),\\quad P_{k+1}=\\frac{P_k-K_k\\phi_k^\\mathsf{T}P_k}{\\lambda}$$
其中：

- $\\hat P_{f,i}^{\\mathrm{RLS}}$：flight $f$ 中第 $i$ 个采样点经 RLS 校正的功率，单位 W。
- $\\hat P_{f,i}^{\\mathrm{TCN}}$：同一采样点的 TCN 原始功率，单位 W。
- $k$：当前已经结束的完整秒窗编号；$i$：秒窗内采样点编号；$f$：flight 编号。
- $s$：训练集功率标准差形成的功率尺度，单位 W。
- $\\theta_k=[\\theta_{0,k},\\theta_{1,k}]^\\mathsf{T}$：第 $k$ 个秒窗预测时使用的偏置与缩放参数。
- $\\phi_k=[1,\\bar P_k^{\\mathrm{TCN}}/s]^\\mathsf{T}$：由第 $k$ 个秒窗 TCN 平均功率构成的回归向量。
- $y_k=\\bar P_k^{\\mathrm{true}}/s$：第 $k$ 个秒窗真实平均功率的尺度化目标。
- $P_k$：RLS 参数协方差矩阵；初始值为 $0.25I$。
- $\\lambda$：遗忘因子，本次取 0.90；$K_k$：第 $k$ 次更新的 RLS 增益。
- $I$：二阶单位矩阵；$\\max(0,\\cdot)$：把物理上无意义的负功率截断为 0。""",
    },
    "rf_tlatt_lite": {
        "title": "RF-TLATT 思路的相位感知岭回归（轻量基线）",
        "source": "Luo 等，2025，RF-TLATT 论文表 8 与相位分类流程",
        "structure": """Luo 等的完整 RF-TLATT 先用随机森林识别飞行阶段，再为每个阶段训练包含 LSTM、Transformer 和注意力的专用预测器。本目录只保留“先分相位、再使用相位专用模型”的控制变量思想：相位由速度阈值直接判定，预测器改为岭回归。因此本结果不能写成 RF-TLATT 的复现结果。

1. **预测相位判别层**：验证集比较垂直/水平速度阈值 $(0.10,0.5)$、$(0.15,1.0)$、$(0.25,1.5)$ m/s，最终选中 $(0.25,1.5)$ m/s。其规则为：$v_z>0.25$ m/s 判为 ascent，$v_z<-0.25$ m/s 判为 descent，$|v_z|<0.25$ m/s 且 $v_{xy}\\le1.5$ m/s 判为 idle，$|v_z|<0.25$ m/s 且 $v_{xy}>1.5$ m/s 判为 cruise。README 的统一诊断图仍使用固定的 $(0.15,1.0)$ m/s 阈值划分阶段，两者用途不同。
2. **16 维特征层**：每条记录依次使用实际合速度、水平速度、垂直速度、垂直速度绝对值、载荷、高度、风速、动态加速度模、角速度模、相对空速、风向对齐度、横向风速分量、合速度平方、垂直速度平方、合速度×载荷、合速度×风速。该方法只读当前记录，不构造 17 步序列。
3. **标准化与截距层**：每个回归头都用训练子集的均值和标准差标准化 16 个输入，再在首列加入常数 1。每个相位头有 16 个特征系数和 1 个截距；岭惩罚不作用于截距。
4. **四个相位回归头**：idle、ascent、descent、cruise 分别求闭式岭回归。若某相位训练样本少于 $2\\times16=32$ 条，则回退到用全部训练记录拟合的全局头；输出经 $\\max(0,\\cdot)$ 截断后得到瞬时功率。
5. **验证选择层**：每组阈值与 $\\alpha\\in\\{0.01,0.1,1,10\\}$ 组合均只在验证集打分。最终选中 $\\alpha=10$、垂直阈值 0.25 m/s、水平阈值 1.5 m/s，选择分数为功率 WAPE 加 0.5 倍 flight 能耗 WAPE。""",
        "formula": """$$\\hat P_i=\\max\\left(0,\\;[1,\\mathbf z_i^\\mathsf{T}]\\hat\\beta_{c_i}\\right),\\qquad \\hat\\beta_c=(X_c^\\mathsf{T}X_c+\\alpha I)^{-1}X_c^\\mathsf{T}\\mathbf y_c$$
其中：

- $\\hat P_i$：第 $i$ 条测试记录的非负预测功率，单位 W。
- $c_i$：第 $i$ 条记录由预测阈值判定的相位类别。
- $\\mathbf z_i$：第 $i$ 条记录经训练统计量标准化后的 16 维特征向量。
- $[1,\\mathbf z_i^\\mathsf{T}]$：含截距常数 1 的行向量。
- $\\hat\\beta_c$：相位 $c$ 的 17 维回归系数估计。
- $X_c$：训练集中相位 $c$ 的含截距设计矩阵；$\\mathbf y_c$：对应真实功率列向量。
- $\\alpha$：岭正则强度，本次验证集选中 10；$I$：截距位置为 0、其余对角元素为 1 的惩罚矩阵。
- $i$：测试记录索引；$c$：相位类别索引。""",
    },
    "physical_mlr": {
        "title": "物理特征 MLR（单一全局回归适配版）",
        "source": "Jastrzębska 等，2026，UAV energy consumption prediction",
        "structure": """Jastrzębska 等的原方法分别建立静止与飞行子模型，并使用物理运动量解释能耗。本工程为适配现有 R1 字段，只建立一个全局回归，在输入中加入 `moving` 指示量；同时由于数据没有论文所需的独立垂直加速度字段，代码用带符号垂直速度项代替对应垂直运动项。下面描述的是当前可运行实现，而不是论文两套子模型的逐项复现。

1. **总质量代理层**：按 $m=1.0+payload_{kg}$ 构造总质量，其中 1.0 kg 是项目采用的机体基准质量，`payload_kg` 为载荷质量。
2. **运动状态指示层**：实际合速度大于 0.1 m/s 时令 $I_{move}=1$，否则为 0。该变量只是单一全局模型中的一列，不会切换到另一套回归系数。
3. **7 维物理项展开层**：依次形成 $I_{move}$、$am$、$v_zm$、$v_{xy}^2m^{2/3}$、$v_z^2m^{2/3}$、$m$ 和风速 $w$。其中 $a$ 使用三轴动态加速度模；代码中的 $\\operatorname{sign}(v_z)|v_z|m$ 数值上等于 $v_zm$。每项只使用当前记录，不包含历史序列。
4. **标准化层**：用训练集均值和标准差分别标准化 7 个输入量，零标准差以 1 替代；再加入常数截距列，因此模型共有 8 个回归系数。
5. **线性/岭闭式解层**：验证集比较 $\\alpha\\in\\{0,0.01,0.1,1,10,100\\}$，最终选择 $\\alpha=0$，即普通最小二乘。预测值经 $\\max(0,\\cdot)$ 截断。该模型无记忆单元，不能显式利用阶段切换前后的历史。""",
        "formula": """$$\\mathbf u_i=\\left[I_{move},\\;am,\\;v_zm,\\;v_{xy}^2m^{2/3},\\;v_z^2m^{2/3},\\;m,\\;w\\right]_i^\\mathsf{T},\\qquad z_{i,j}=\\frac{u_{i,j}-\\mu_j}{\\sigma_j}$$
$$\\hat P_i=\\max\\left(0,\\;\\beta_0+\\sum_{j=1}^{7}\\beta_j z_{i,j}\\right)$$
其中：

- $\\mathbf u_i$：第 $i$ 条记录的 7 维物理构造量向量；$u_{i,j}$：该向量的第 $j$ 个分量。
- $I_{move}$：运动状态指示量，实际合速度大于 0.1 m/s 时为 1，否则为 0。
- $a$：三轴动态加速度模，单位 $\\mathrm{m/s^2}$；$m$：机体基准质量与载荷质量之和，单位 kg。
- $v_z$：带符号垂直速度，单位 m/s；$v_{xy}$：水平速度，单位 m/s。
- $w$：风速，单位 m/s。
- $\\mu_j$、$\\sigma_j$：第 $j$ 个构造量在训练集上的均值和标准差；$z_{i,j}$：对应标准化值。
- $\\hat P_i$：第 $i$ 条记录截断为非负值后的预测功率，单位 W。
- $\\beta_0$：截距；$\\beta_j$：第 $j$ 个标准化物理项的回归系数。
- $i$：记录索引；$j$：物理构造量索引；$\\max(0,\\cdot)$：把负预测截断为 0。""",
    },
    "lstm": {
        "title": "两层双向 LSTM",
        "source": "Muli 等，2022，A Comparative Study on Energy Consumption Models for Drones",
        "structure": """Muli 等使用两层堆叠双向 LSTM，每层每方向 128 个 hidden cells，并在后端使用 Dropout 与 Dense(tanh)。本工程保留两层双向结构，在每方向 64 和 128 单元之间用验证集选择；最终选中每方向 64 单元、Dropout 0.30，共 149,057 个可训练参数。

1. **输入与填充层**：每条样本由同一 flight 中当前点及之前最多 16 个点组成，得到 $B\\times17\\times23$ 张量。flight 开头不足 17 步时复制最早记录到左侧；23 个通道均使用训练集统计量标准化。
2. **第 1 个双向 LSTM 层**：正向和反向各有 64 个记忆单元。两方向在 17 步缓存窗口内分别从首端和末端扫描，逐步更新输入门、遗忘门、候选记忆和输出门；沿特征维拼接后，每个时间位置输出 128 维。
3. **第 2 个双向 LSTM 层**：输入和输出均为 $B\\times17\\times128$。PyTorch LSTM 在第 1 层到第 2 层之间使用 0.30 Dropout；第二层正向与反向最终 hidden state 各为 $B\\times64$。
4. **状态拼接层**：将第二层正向与反向最终 hidden state 拼成 $B\\times128$。反向分支只反向读取已经缓存的“历史至当前”窗口，不访问预测时刻之后的记录。
5. **回归头**：执行 `Dropout(0.30) → Linear(128,32) → Tanh → Linear(32,1)`，输出一个标准化功率，再按训练目标均值与标准差还原为 W 并截断为非负值。
6. **训练选择层**：使用 AdamW、SmoothL1 损失和梯度范数 5.0 裁剪；每个候选按验证损失保存最佳 epoch，再用验证功率 WAPE 与 flight 能耗 WAPE 的组合分数比较候选。本次 64 单元候选在第 24 轮取得最终保存状态。""",
        "formula": """$$i_t=\\sigma(W_i x_t+U_i h_{t-1}+b_i),\\quad f_t=\\sigma(W_f x_t+U_f h_{t-1}+b_f)$$
$$\\tilde c_t=\\tanh(W_cx_t+U_ch_{t-1}+b_c),\\quad c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$$
$$o_t=\\sigma(W_o x_t+U_o h_{t-1}+b_o),\\quad h_t=o_t\\odot\\tanh(c_t)$$
其中：

- $x_t$：时间步 $t$ 的 23 维标准化输入；$h_{t-1}$：前一时间步隐状态。
- $i_t$、$f_t$、$o_t$：输入门、遗忘门和输出门向量。
- $\\tilde c_t$：候选记忆；$c_t$：当前记忆状态；$c_{t-1}$：前一时间步记忆状态。
- $h_t$：当前隐状态；双向层分别计算正向与反向 $h_t$ 后再拼接。
- $W_i,W_f,W_c,W_o$：输入到各门或候选记忆的权重矩阵。
- $U_i,U_f,U_c,U_o$：上一隐状态到各门或候选记忆的循环权重矩阵。
- $b_i,b_f,b_c,b_o$：相应偏置向量。
- $\\sigma$：Sigmoid 激活函数；$\\tanh$：双曲正切；$\\odot$：逐元素乘法；$t$：窗口内时间步索引。""",
    },
    "lr_tcn_sma": {
        "title": "LR-TCN-SMA",
        "source": "Dudukcu 等，2024，UAV instantaneous power prediction using LR-TCN with simple moving average",
        "structure": """Dudukcu 等提出以 LeakyReLU 替代常规 TCN 激活并引入简单移动平均（SMA）特征。本工程把 23 个字段全部先做历史 SMA，再送入统一 TCN，没有复现原文原始特征与 SMA 特征的并行接口。验证集在 48/64 通道和 5/10 点 SMA 两组候选中选中 64 通道、10 点 SMA、Dropout 0.12，共 67,841 个可训练参数。

1. **SMA 输入层**：对每个 flight、每个特征在当前点及之前最多 9 个点上求均值；flight 开头使用实际可用点数。平滑后再用训练集统计量标准化，并构造 $B\\times17\\times23$ 的历史至当前窗口。
2. **残差块 1**：第一层为左因果 `Conv1d(23,64,k=3,d=1)`，第二层为 `Conv1d(64,64,k=3,d=1)`；每层后依次执行 LeakyReLU(0.05) 和 Dropout(0.12)。旁路用 $1\\times1$ 卷积把 23 通道投影到 64 通道，主路与旁路相加。
3. **残差块 2**：两层 $64\\to64$ 左因果卷积，膨胀率均为 2；旁路为恒等映射。该块扩大当前输出能够访问的历史间隔。
4. **残差块 3**：两层 $64\\to64$ 左因果卷积，膨胀率均为 4。三块理论感受野为 $1+2(k-1)(1+2+4)=29$ 步，但实际输入只有 17 步，因此真实数据上下文最多为当前点及前 16 点，其余位置来自左填充。
5. **当前时刻读出层**：取第三个残差块最后时间位置的 64 维向量，经 `Linear(64,1)` 输出标准化功率，再还原为 W 并截断为非负值。TCN 卷积不读取预测时刻之后的记录，`dt_seconds` 与其他特征一起参与 SMA 和卷积，原始真实时间间隔仍用于最终能耗积分。
6. **训练选择层**：使用 AdamW、SmoothL1 和梯度裁剪；候选内部按验证损失保留最佳 epoch，再按统一选择分数确定结构。本次最终候选保存于第 24 轮。""",
        "formula": """$$q_t=\\min(q,t+1),\\qquad x_t^{SMA}=\\frac{1}{q_t}\\sum_{j=0}^{q_t-1}x_{t-j},\\qquad h_t=\\operatorname{LeakyReLU}(W*x_{\\le t}+b)$$
其中：

- $x_t$：时间步 $t$ 的 23 维原始输入向量。
- $x_t^{SMA}$：时间步 $t$ 的历史移动平均输入；实际 flight 起点按可用点数调整分母。
- $q$：设定的 SMA 最大窗口长度，本次为 10；$q_t$：时间步 $t$ 实际可用的平均点数；$j$：回看步索引。
- $x_{t-j}$：当前点之前第 $j$ 步的输入；$x_{\\le t}$：不晚于当前点的输入序列。
- $W$：因果卷积核权重；$b$：卷积偏置；$*$：仅左侧填充的膨胀卷积运算。
- $h_t$：当前时间步的卷积特征；$t$：窗口内时间步索引。
- $\\operatorname{LeakyReLU}(u)=\\max(u,0)+0.05\\min(u,0)$：本实现使用的带负半轴斜率激活函数。""",
    },
    "lstm_transformer": {
        "title": "LSTM-Transformer",
        "source": "Feng 等，2024，Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework",
        "structure": """Feng 等的方法先由 LSTM 提取短期变化，再由 Transformer 建模较长依赖，原始对象是电动汽车。当前 UAV 适配版在 $D=32/64$、1/2 个 Encoder 等候选中选中 $D=64$、1 层 LSTM、2 层 Transformer Encoder、4 个注意力头、128 维前馈层和 Dropout 0.10，共 105,089 个可训练参数。

1. **输入与填充层**：同一 flight 的当前点和之前最多 16 点组成 $B\\times17\\times23$ 张量；起点不足 17 步时复制最早记录到左侧，23 个特征按训练集统计量标准化。
2. **线性投影层**：`Linear(23,64)` 独立作用于每个时间步，把不同量纲的 23 维输入映射到统一的 64 维模型空间，输出 $B\\times17\\times64$。
3. **可学习位置编码层**：加入形状 $1\\times17\\times64$ 的参数矩阵，使注意力能够区分窗口首端、内部和当前时刻；位置参数与投影结果逐元素相加。
4. **单层 LSTM**：64 个记忆单元依次处理 17 个位置，输出仍为 $B\\times17\\times64$。该层先汇总相邻点的短期顺序信息，再交给自注意力。
5. **Transformer Encoder 1**：4 个注意力头并行工作，每个头的键/查询维度为 $64/4=16$；多头输出经残差连接和 LayerNorm，再通过 `Linear(64,128) → GELU → Dropout → Linear(128,64)` 前馈网络及第二组残差归一化。
6. **Transformer Encoder 2**：重复同样的 4 头注意力和 128 维前馈结构，在第一层关系表示上再次组合窗口内位置。实现未使用因果 mask，但整个 17 步输入窗只包含当前及过去记录，因此不会读取预测时刻之后的数据。
7. **归一化与回归头**：取第二个 Encoder 最后时间位置的 64 维向量，经 `LayerNorm(64) → Linear(64,32) → GELU → Linear(32,1)` 输出标准化功率，再还原为 W 并截断为非负值。
8. **训练选择层**：使用 AdamW、SmoothL1、梯度裁剪和验证早停。本次选中候选的最佳保存轮次为第 21 轮。""",
        "formula": """$$\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}\\left(\\frac{QK^\\mathsf{T}}{\\sqrt{d_k}}\\right)V$$
$$head_r=\\operatorname{Attention}(XW_r^Q,XW_r^K,XW_r^V),\\qquad \\operatorname{MHA}(X)=\\operatorname{Concat}(head_1,\\ldots,head_h)W^O$$
其中：

- $X$：LSTM 输出的窗口序列矩阵，每条样本为 $17\\times64$。
- $Q$、$K$、$V$：查询、键和值矩阵，用于计算位置间权重并加权汇总信息。
- $d_k$：单个注意力头的键维度，本次为 16；$\\sqrt{d_k}$：抑制点积随维度增大的缩放项。
- $W_r^Q,W_r^K,W_r^V$：第 $r$ 个头的查询、键、值投影矩阵。
- $head_r$：第 $r$ 个注意力头输出；$r$：注意力头索引。
- $h$：注意力头总数，本次为 4；$\\operatorname{Concat}$：沿特征维拼接各头输出。
- $W^O$：把多头拼接结果映射回 64 维的输出投影矩阵。
- $\\operatorname{softmax}$：在键位置维度上把缩放点积转换为和为 1 的注意力权重。""",
    },
    "cnn_lstm": {
        "title": "CNN-LSTM",
        "source": "Luo 等，2025，RF-TLATT 论文表 8 列出的 CNN-LSTM 对比名称（二手来源）",
        "structure": """Luo 等仅在表 8 给出 CNN-LSTM 的对比结果，正文将其归于“Xuebing et al. (2024)”，但所给论文的参考文献表缺少相应完整条目，因而无法核实原模型层数与参数。当前目录实现的是可复现的 CNN-LSTM 工程基线，不冒充该未知原模型的直接复现。验证集在 48/64 通道候选中选中卷积通道 $C=48$、LSTM 隐层 $H=48$、Dropout 0.10，共 29,377 个可训练参数。

1. **输入与填充层**：当前点与同一 flight 的前 16 点组成 $B\\times17\\times23$ 标准化序列；flight 开头不足部分复制最早记录。转置后得到 Conv1d 所需的 $B\\times23\\times17$。
2. **卷积层 1**：`Conv1d(23,48,k=3,padding=1)` 同时汇总 23 个输入通道和相邻 3 个时间位置，输出 $B\\times48\\times17$；随后执行 `BatchNorm1d(48) → LeakyReLU(0.05) → Dropout(0.10)`。
3. **卷积层 2**：`Conv1d(48,48,k=3,padding=1) → BatchNorm1d(48) → LeakyReLU(0.05)`，继续组合第一层局部模式，输出形状仍为 $B\\times48\\times17$。第二层后没有额外 Dropout。
4. **单层 LSTM**：转回 $B\\times17\\times48$，由 48 个记忆单元顺序聚合卷积特征；输出整个序列并取最后时间位置的 $B\\times48$ 向量。
5. **回归头**：对最后时刻向量执行 `Dropout(0.10) → Linear(48,1)`，得到标准化功率，再还原为 W 并截断为非负值。
6. **时间信息边界**：两层卷积使用对称 `padding=1`，中间位置会组合其左右相邻位置；但送入模型的 17 步缓冲区以当前预测时刻结束，所以不会访问该时刻之后的数据。训练使用 AdamW、SmoothL1 和梯度裁剪，本次最佳状态保存于第 20 轮。""",
        "formula": """$$z_{c,t}=\\operatorname{LeakyReLU}\\left(\\operatorname{BN}\\left(\\sum_{r=1}^{C_{in}}\\sum_{\\tau=-1}^{1}W_{c,r,\\tau}x_{r,t+\\tau}+b_c\\right)\\right)$$
其中：

- $x_{r,t+\\tau}$：输入通道 $r$ 在窗口位置 $t+\\tau$ 的值；边界外位置由零填充。
- $C_{in}$：该卷积层的输入通道数，第一层为 23、第二层为 48。
- $W_{c,r,\\tau}$：从输入通道 $r$、相对位置 $\\tau$ 到输出通道 $c$ 的卷积权重。
- $b_c$：输出通道 $c$ 的偏置；$z_{c,t}$：卷积、批归一化和激活后的局部特征。
- $c$：输出通道索引；$r$：输入通道索引；$t$：窗口内位置；$\\tau\\in\\{-1,0,1\\}$：三点卷积核的相对位置。
- $\\operatorname{BN}$：批归一化；$\\operatorname{LeakyReLU}$：负半轴斜率为 0.05 的激活函数。""",
    },
}


COMMON_FORMULAS = r'''### 评价指标与能耗积分

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
$$
$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\qquad
\mathrm{MAPE}=\frac{100\%}{n_0}\sum_{|y_i|>\epsilon}\left|\frac{y_i-\hat y_i}{y_i}\right|
$$
$$
\mathrm{WAPE}=\frac{\sum_i|y_i-\hat y_i|}{\sum_i|y_i|}\times100\%,\qquad
E_f=\sum_{i\in f}P_{f,i}\frac{\Delta t_{f,i}}{3600}
$$

其中：

- $y_i$：第 $i$ 个真实值；在样本级为功率 W，在 flight 级为能耗 Wh。
- $\hat y_i$：与 $y_i$ 对应的预测值；$y_i-\hat y_i$：评价指标采用的误差差值。
- $n$：参与 MAE、RMSE、$R^2$ 或 WAPE 的样本数；$i$：样本索引。
- $n_0$：满足 $|y_i|>\epsilon$ 的 MAPE 有效样本数；$\epsilon=10^{-6}$ 是近零筛选阈值。
- $\bar y$：全部真实值的算术平均数。
- $P_{f,i}$：flight $f$ 第 $i$ 个采样点的功率，单位 W。
- $\Delta t_{f,i}$：对应记录的真实采样间隔，单位 s。
- $E_f$：flight $f$ 的总能耗，单位 Wh；$f$：flight 编号。
- $3600$：从 W·s 换算到 Wh 的秒数因子。

MAE、RMSE、MAPE、WAPE 越小越好，$R^2$ 越接近 1 越好。能耗先按真实 `dt_seconds` 积分，再在 28 个 flight 上计算指标，避免把不等间隔采样误当成等权样本。

本实验的选择分数为：

$$S_{val}=\mathrm{WAPE}_{power,val}+0.5\,\mathrm{WAPE}_{energy,val}$$

其中：

- $S_{val}$：验证集候选选择分数，越小越好。
- $\mathrm{WAPE}_{power,val}$：验证集逐点功率 WAPE，单位 %。
- $\mathrm{WAPE}_{energy,val}$：验证集按 flight 汇总后的能耗 WAPE，单位 %。
- $0.5$：能耗 WAPE 在选择分数中的固定权重。

该分数只在 28 个 validation flight 上计算，用于比较候选结构、正则强度和相位阈值；测试集 28 个 flight 在参数固定后才运行。'''


VARIABLE_GLOSSARY = r'''### 变量、坐标轴和字段词典

| 字段或指标 | 中文全称 | English full name | 单位/范围 | 计算方式和含义 |
|---|---|---|---|---|
| `flight` | 飞行任务编号 | Flight identifier | 无量纲 | 同一编号的连续记录属于同一任务，能耗按该字段分组。 |
| `route` | 航线编号 | Route identifier | — | 本实验仅保留 R1。 |
| `time` | 飞行内相对时间 | Flight-relative time | s | 每个 flight 从 0 s 开始的时间坐标，所有时序图横轴使用它。 |
| `dt_seconds`（$\Delta t_i$） | 采样时间间隔 | Sampling time interval | s | 相邻时间戳之差；能耗积分使用真实值，不用固定步长替代。 |
| `power_w`（$y_i$） | 实测瞬时功率 | Measured instantaneous power | W | `max(battery_voltage × battery_current, 0)`，作为监督目标和散点图横坐标。 |
| `predicted_power_w`（$\hat y_i$） | 预测瞬时功率 | Predicted instantaneous power | W | 算法输出的非负功率，作为时序图预测线和散点图纵坐标。 |
| `residual_w`（$e_i$） | 功率残差 | Power residual | W | $e_i=\hat y_i-y_i$；正值表示高估，负值表示低估。 |
| `absolute_error_w` | 功率绝对误差 | Absolute power error | W | $|e_i|$，用于箱线图、CDF、分箱误差和小提琴图。 |
| `actual_energy_wh` | 实测采样能量 | Measured interval energy | Wh | $P_i\Delta t_i/3600$。 |
| `predicted_energy_wh` | 预测采样能量 | Predicted interval energy | Wh | $\hat P_i\Delta t_i/3600$。 |
| `actual_energy_wh`（flight汇总） | 实测 flight 总能耗 | Measured flight energy | Wh | 同一 flight 的 `actual_energy_wh` 求和。 |
| `predicted_energy_wh`（flight汇总） | 预测 flight 总能耗 | Predicted flight energy | Wh | 同一 flight 的 `predicted_energy_wh` 求和。 |
| `energy_error_wh` | flight 能耗误差 | Flight-energy error | Wh | 预测总能耗减实测总能耗；正值为任务级高估。 |
| `phase_name` | 飞行阶段 | Flight phase | idle/ascent/descent/cruise | 由 $v_z$ 和水平速度阈值推导，不是人工标注。 |
| MAE | 平均绝对误差 | Mean absolute error | W 或 Wh | 所有误差绝对值的算术平均，代表典型偏差。 |
| RMSE | 均方根误差 | Root mean square error | W 或 Wh | 误差平方平均后开方，对尖峰误差更敏感。 |
| $R^2$ | 决定系数 | Coefficient of determination | 无量纲 | 相对于真实均值基线的解释度。 |
| MAPE | 平均绝对百分比误差 | Mean absolute percentage error | % | 逐点相对误差平均；接近 0 的真实功率被排除。 |
| WAPE | 加权绝对百分比误差 | Weighted absolute percentage error | % | 总绝对误差除以真实值绝对和，适合跨功率段总体比较。 |
| CDF | 经验累积分布函数 | Empirical cumulative distribution function | 0–1 | $F(a)=\#\{|e_i|\le a\}/n$，读取某误差阈值以内的样本比例。 |
| P50/P90/P95 | 误差分位数 | Error percentiles | W | 绝对误差排序后第 50%、90%、95% 位置，反映典型和尾部风险。 |
'''


ALGORITHM_CHART_DESCRIPTIONS = {
    "power_scatter.png": "横轴是实测瞬时功率（Measured instantaneous power，W），纵轴是预测瞬时功率（Predicted instantaneous power，W）。绘图从 45,237 条测试记录中每隔 20 条取 1 条，共显示 2,262 个点；黑色虚线 $y=x$ 是理想线，线上方代表高估，点到理想线的垂直距离是该点绝对误差。抽样只用于减轻点云遮挡，指标仍由全部记录计算。",
    "flight_power_timeseries.png": "横轴是飞行时间（Flight time，s），纵轴是功率（Power，W）。同一 flight 的实线是实测功率，虚线是预测功率；蓝、橙、绿分别对应图例中的前三个测试 flight。曲线峰值错位表示响应滞后，预测线持续高于实线表示该时段高估。",
    "residual_histogram.png": "横轴是功率残差 $e=\\hat P-P$（Residual，W），纵轴是记录数（Count）。蓝色柱表示残差频数，红色竖虚线是零误差。中心偏右/偏左分别表示总体高估/低估，分布宽度和长尾对应 RMSE 与 P95。",
    "residual_vs_actual.png": "横轴是实测功率（Measured power，W），纵轴是残差（Residual，W）；颜色分别表示 idle、ascent、descent、cruise。红色水平虚线为零残差。该图每隔 4 条记录取 1 条，显示 11,310 个点；抽样只服务于显示。某个功率区间持续偏离零线说明该区间存在系统偏差，扇形扩散说明误差随功率增大而增大。",
    "power_bin_error.png": "横轴是实测功率区间；左纵轴是区间 MAE（W），橙色柱表示误差；右纵轴是样本数（Samples），蓝色折线和圆点表示样本量。两条纵轴不能混读。柱高用于比较区间难度，折线用于判断该区间对总体指标的权重。",
    "flight_energy_scatter.png": "横轴是实测单 flight 总能耗（Measured flight energy，Wh），纵轴是预测总能耗（Predicted flight energy，Wh）。蓝点代表一个 flight，黑色虚线是 $y=x$；线上方为任务级高估，点距直线越远表示累计能量偏差越大。",
    "flight_energy_error_ranking.png": "横轴是 `predicted_energy_wh−actual_energy_wh`（Flight-energy error，Wh），纵轴是 flight 编号。红柱表示高估，蓝柱表示低估，黑色竖线为零误差。柱长直接给出每个任务的累计偏差。",
    "cumulative_energy_timeseries.png": "横轴是飞行时间（s），纵轴是累计能耗（Cumulative energy，Wh）。同色实线是实测能量累加，虚线是预测能量累加；末端垂直距离是该 flight 总能耗误差，曲线斜率对应当前功率水平。",
    "phase_error_metrics.png": "横轴依次为 idle（停机/低速）、ascent（上升）、descent（下降）、cruise（巡航）；三个子图纵轴分别是 MAE（W）、RMSE（W）和 WAPE（%）。每根柱表示一个阶段的聚合误差，RMSE 明显高于 MAE 时说明阶段内存在尖峰。",
    "absolute_error_cdf.png": "横轴是绝对功率误差（Absolute power error，W），纵轴是经验 CDF（0–1）。蓝线在阈值 $a$ 处的高度表示误差不超过 $a$ W 的比例；红色水平线是 90% 参考线，曲线越靠左上越好。",
    "prediction_zoom.png": "上子图横轴为局部飞行时间（s），纵轴为功率（W）；深蓝实线是实测功率，橙色虚线是预测功率。下子图横轴相同，纵轴为残差（W），紫线为 $\\hat P-P$，黑色虚线为零误差。窗口以该算法最大绝对误差为中心，专门观察突变响应。",
    "flight_energy_error_histogram.png": "横轴是 flight 能耗误差（Wh），纵轴是 flight 数量。蓝色柱表示误差分布，黑色虚线是零误差，红线是平均误差；红线偏离零说明存在任务级系统偏差。",
    "phase_error_distribution.png": "横轴是四个飞行阶段，纵轴是绝对功率误差（W）。箱体为第 25–75 百分位，红线为中位数，须线表示非离群范围；箱体高说明该阶段误差离散程度大。",
    "comparison_metrics.png": "左右两个横向柱图分别以算法为纵轴、样本功率 WAPE（Power WAPE，%）和 flight 能耗 WAPE（Flight-energy WAPE，%）为横轴。蓝柱对应逐点功率，橙柱对应任务累计能耗，柱越短越好。",
    "comparison_error_boxplot.png": "横轴为算法，纵轴为绝对功率误差（W）。箱体从 Q1 到 Q3，红线为 P50，图中隐藏极端离群点以便比较主体分布；中位数低且箱体短表示典型误差小、稳定性高。",
    "comparison_phase_wape.png": "横轴为 idle、ascent、descent、cruise，纵轴为算法；颜色和格内数字都是阶段功率 WAPE（%）。颜色越深表示该算法在该阶段误差越大，适合定位阶段性退化。",
    "comparison_flight_energy_scatter.png": "每个小图对应一种算法，横轴为实测 flight 能耗（Wh），纵轴为预测 flight 能耗（Wh）；蓝点为同一测试 flight，黑色虚线为理想线。小图之间可以直接比较点云离线程度。",
    "comparison_metric_heatmap.png": "横轴为 Power MAE、Power RMSE、Power WAPE、Energy MAE、Energy WAPE，纵轴为算法。底色为按每列最小值和最大值归一化后的误差，绿色较优、红色较差；格内数字保留原始单位值。",
    "comparison_error_cdf.png": "横轴为绝对功率误差（W），纵轴为经验 CDF；每条线由图例标识一种算法。在固定阈值处，线越高表示更多样本达到该误差要求，曲线越靠左上越好。",
    "comparison_overall_rank.png": "横轴为五项误差指标的平均名次，纵轴为算法。每项指标先按从小到大排名，再求平均；柱越短表示功率与能耗折中越好。该图是描述性排序，不是统计显著性检验。",
    "comparison_power_timeseries.png": "横轴为测试集中功率标准差最大的同一 flight 的飞行时间（s），纵轴为功率（W）。黑色粗线是实测功率，其他彩色线分别对应图例算法；若该 flight 超过 700 条记录，图中只显示前 700 条。所有线共享同一时间轴，可比较峰值位置、过冲和稳态偏差。",
    "comparison_error_violin.png": "横轴为算法，纵轴为绝对功率误差（W）。每个算法每隔 5 条测试记录取 1 条，共 9,048 个误差值；小提琴宽度表示该误差附近的核密度，内部两组水平标记分别表示均值和中位数。抽样不参与指标计算。",
    "comparison_bland_altman.png": "每个子图是一种算法，横轴为实测与预测功率的均值（W），纵轴为残差（W）。45,237 条记录按步长 7 抽样，共显示 6,463 个点；红线为该抽样点集的平均偏差，橙色虚线为平均偏差 ±1.96 个样本标准差。若残差随横轴明显倾斜，说明存在功率相关偏差。",
    "comparison_energy_error_distribution.png": "横轴为算法，纵轴为 flight 能耗误差（Wh）；箱体表示任务级误差的主体范围，红线为中位数，黑色水平虚线为零误差。它与功率箱线图不同，关注的是整段任务的累计偏差。",
    "comparison_phase_metrics.png": "三个并列热图分别给出阶段 MAE（W）、RMSE（W）和 WAPE（%），横轴为四阶段，纵轴为算法，格内数字为原始指标。并列观察可以区分低功率分母放大的百分比误差与真实瓦特误差。",
    "comparison_metric_radar.png": "极坐标轴分别为 Power MAE、Power RMSE、Power WAPE、Energy MAE、Energy WAPE；每项先转换为 $1-(x-x_{min})/(x_{max}-x_{min})$ 的相对得分，越靠外表示该列相对误差越低。它只反映当前七种方法之间的相对位置。",
}


def feature_table(metadata: dict[str, object]) -> str:
    """功能:生成公共数据中23个候选变量的中英文、单位和作用表。
    参数: metadata为3.0特征元数据字典。
    返回: Markdown表格文本。
    调用位置:overall_readme和algorithm_readme。
    """
    descriptions = metadata.get("field_descriptions", {}) if isinstance(metadata, dict) else {}
    lines = [
        "| 变量 | 中文全称 | English full name | 单位/范围 | 在模型中的作用 |",
        "|---|---|---|---|---|",
    ]
    for feature in FEATURE_COLUMNS:
        chinese, english, unit = FEATURE_ENGLISH.get(feature, (feature, feature, "—"))
        purpose = descriptions.get(feature, "作为统一输入特征")
        lines.append(f"| `{feature}` | {chinese} | {english} | {unit} | {purpose} |")
    return "\n".join(lines)


def tuning_markdown(folder: str, summary: dict[str, object]) -> str:
    """功能:生成候选参数表和最终选择说明。
    参数: folder为算法目录；summary为data_summary返回值。
    返回: Markdown调参小节。
    调用位置:algorithm_readme和overall_readme。
    """
    tuning = summary["tuning"]
    selected = summary["selected"]
    if not isinstance(tuning, pd.DataFrame) or tuning.empty:
        return "本算法为固定 3.0 checkpoint 或无可调候选，未单独生成候选表。"
    display = tuning.copy()
    # 只展示能解释选择过程的字段，避免把表格横向撑得过宽。
    preferred = [
        "candidate", "best_epoch", "validation_power_wape_percent",
        "validation_flight_energy_wape_percent", "selection_score", "selected",
    ]
    parameter_columns = [
        column for column in display.columns
        if column not in preferred and column not in {"validation_loss", "validation_power_mae_w"}
    ]
    columns = [column for column in preferred if column in display.columns] + parameter_columns
    display = display[columns]
    headers = {"candidate": "候选", "best_epoch": "验证最优轮次", "validation_power_wape_percent": "验证功率WAPE(%)",
               "validation_flight_energy_wape_percent": "验证能耗WAPE(%)", "selection_score": "选择分数", "selected": "最终选择"}
    lines = ["| " + " | ".join(headers.get(column, column) for column in display.columns) + " |",
             "|" + "|".join("---:" if display[column].dtype.kind in "biufc" else "---" for column in display.columns) + "|"]
    for _, row in display.iterrows():
        cells = []
        for column in display.columns:
            value = row[column]
            if column == "selected":
                cells.append("是" if bool(value) else "否")
            elif isinstance(value, (float, np.floating)):
                cells.append(fmt(value, 4))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    selected_text = params_text(selected)
    return "\n".join(lines) + f"\n\n最终选择候选 `{selected.get('candidate', '—')}`：{selected_text}。验证集选择分数为 `{fmt(selected.get('selection_score'), 4)}`；测试集没有参与候选筛选。"


def phase_markdown(summary: dict[str, object]) -> str:
    """功能:生成阶段和功率分箱的真实统计表。
    参数: summary为单算法统计字典。
    返回: Markdown表格和解释文本。
    调用位置:algorithm_readme。
    """
    phase_rows = summary["phase_rows"]
    bin_rows = summary["bin_rows"]
    lines = [
        "### 按飞行阶段的误差",
        "",
        "| 阶段 | 中文名称 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(f"| {row['phase']} | {PHASE_CN[row['phase']]} | {row['n']:,} | {fmt(row['mae'], 2)} | {fmt(row['rmse'], 2)} | {fmt(row['wape'], 2)} |")
    lines += [
        "",
        "统一诊断阶段由速度阈值推导：$v_z>0.15$ m/s 为 ascent，$v_z<-0.15$ m/s 为 descent，$|v_z|<0.15$ m/s 且 $v_{xy}\\le1.0$ m/s 为 idle，$|v_z|<0.15$ m/s 且 $v_{xy}>1.0$ m/s 为 cruise。边界值也保留为 cruise。该规则只用于让七种算法按同一口径生成阶段图；RF-TLATT 轻量基线预测时使用验证集另选的 0.25/1.5 m/s 阈值。idle 的真实功率中位数接近 0，百分比误差分母较小，阅读时应结合 MAE（W）和样本数；ascent/descent 的功率变化更大，RMSE 对少量尖峰更敏感。",
        "",
        "### 按实测功率区间的误差",
        "",
        "| 实测功率区间 | 样本数 | MAE(W) | RMSE(W) | WAPE(%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in bin_rows:
        lines.append(f"| {row['bin']} | {row['n']:,} | {fmt(row['mae'], 2)} | {fmt(row['rmse'], 2)} | {fmt(row['wape'], 2)} |")
    dominant = max(bin_rows, key=lambda row: row["n"])
    hardest = max((row for row in bin_rows if np.isfinite(row["mae"])), key=lambda row: row["mae"])
    lines += [
        "",
        f"测试集中样本最多的是 `{dominant['bin']}`（{dominant['n']:,} 条），它对总体 MAE 的贡献最大；当前算法的绝对误差最高区间是 `{hardest['bin']}`（MAE={fmt(hardest['mae'], 2)} W），通常对应起飞、降落或功率过渡段。",
    ]
    return "\n".join(lines)


def algorithm_chart_analysis(folder: str, summary: dict[str, object]) -> str:
    """功能:为单算法每幅图生成带真实数值的解释段落。
    参数: folder为算法目录；summary为统计字典。
    返回: Markdown图表说明。
    调用位置:algorithm_readme。
    """
    power = summary["power"]
    energy = summary["energy"]
    prediction = summary["prediction"]
    flights = summary["flights"]
    p50 = prediction.absolute_error_w.quantile(.50)
    p90 = prediction.absolute_error_w.quantile(.90)
    p95 = prediction.absolute_error_w.quantile(.95)
    residual_mean = prediction.residual_w.mean()
    residual_std = prediction.residual_w.std()
    max_error_row = prediction.loc[prediction.absolute_error_w.idxmax()]
    first_flights = prediction.flight.drop_duplicates().tolist()[:3]
    chart_order = [
        "power_scatter.png", "flight_power_timeseries.png", "residual_histogram.png",
        "residual_vs_actual.png", "power_bin_error.png", "flight_energy_scatter.png",
        "flight_energy_error_ranking.png", "cumulative_energy_timeseries.png",
        "phase_error_metrics.png", "absolute_error_cdf.png", "prediction_zoom.png",
        "flight_energy_error_histogram.png", "phase_error_distribution.png",
    ]
    lines = ["## 5. 图表与数据分析", "", f"该算法在测试集上的功率 MAE 为 **{fmt(power['mae'], 3)} W**、RMSE 为 **{fmt(power['rmse'], 3)} W**、$R^2={fmt(power['r2'], 4)}$、WAPE 为 **{fmt(power['wape'], 3)}%**；flight 能耗 MAE 为 **{fmt(energy['mae'], 3)} Wh**、RMSE 为 **{fmt(energy['rmse'], 3)} Wh**、WAPE 为 **{fmt(energy['wape'], 3)}%**。残差均值为 {fmt(residual_mean, 3)} W，样本标准差为 {fmt(residual_std, 3)} W，绝对误差 P50/P90/P95 为 {fmt(p50, 2)}/{fmt(p90, 2)}/{fmt(p95, 2)} W。下面逐图说明横纵轴、每条线或颜色、计算方式以及当前图中的数值。", ""]
    for filename in chart_order:
        description = ALGORITHM_CHART_DESCRIPTIONS[filename]
        extra = ""
        if filename == "power_scatter.png":
            extra = f"图中显示等间隔抽样点，$R^2={fmt(power['r2'], 4)}$ 与 WAPE={fmt(power['wape'], 3)}% 则由全部 45,237 条记录计算。WAPE 表示全部绝对误差之和占实测功率绝对值之和的比例。"
        elif filename == "flight_power_timeseries.png":
            details = []
            for flight_id in first_flights:
                part = prediction[prediction.flight == flight_id]
                details.append(f"flight {flight_id} 的区间 MAE={part.absolute_error_w.mean():.2f} W")
            extra = "图中前三个测试 flight 为 " + "、".join(map(str, first_flights)) + "；" + "，".join(details) + "。同色实线/虚线分别是实测/预测，峰值错位会增加尾部误差。"
        elif filename == "residual_histogram.png":
            direction = "高估" if residual_mean > 0 else "低估"
            extra = f"残差均值 {fmt(residual_mean, 3)} W 表明整体存在轻微{direction}倾向；标准差 {fmt(residual_std, 2)} W 与 RMSE 的差异共同反映了尖峰样本。"
        elif filename == "residual_vs_actual.png":
            extra = "该图每隔 4 条记录绘制 1 点，指标不抽样。若点云在高功率端展开更宽，说明残差离散度随功率改变；颜色只表示统一诊断阶段，不是模型一定使用的阶段输入。"
        elif filename == "power_bin_error.png":
            hardest = max((row for row in summary["bin_rows"] if np.isfinite(row["mae"])), key=lambda row: row["mae"])
            extra = f"橙柱最高的区间是 {hardest['bin']}（MAE={fmt(hardest['mae'], 2)} W）；蓝线显示该区间只有 {hardest['n']:,} 条样本，不能仅凭柱高判断其对总体指标的贡献。"
        elif filename == "flight_energy_scatter.png":
            extra = f"28 个蓝点对应 28 个 flight；任务级 WAPE={fmt(energy['wape'], 3)}%，点到 $y=x$ 的垂直距离就是 `energy_error_wh`。"
        elif filename == "flight_energy_error_ranking.png":
            extra = f"任务误差范围为 {fmt(flights.energy_error_wh.min(), 3)}～{fmt(flights.energy_error_wh.max(), 3)} Wh；红柱是高估，蓝柱是低估。"
        elif filename == "cumulative_energy_timeseries.png":
            extra = "每条实线/虚线对使用同一 `dt_seconds` 积分；曲线末端差异与 flight 能耗误差表一致，斜率变化对应当前功率变化。"
        elif filename == "phase_error_metrics.png":
            best_phase = min(summary["phase_rows"], key=lambda row: row["wape"])
            extra = f"按 WAPE 看，当前算法误差最低的阶段是 {best_phase['phase']}（{PHASE_CN[best_phase['phase']]}，{fmt(best_phase['wape'], 2)}%）；若某阶段 RMSE 明显高于 MAE，说明少量突变点主导了平方误差。"
        elif filename == "absolute_error_cdf.png":
            extra = f"CDF 在 {fmt(p50, 2)} W、{fmt(p90, 2)} W 和 {fmt(p95, 2)} W 附近分别达到 0.50、0.90 和 0.95，表示对应比例的测试点误差不超过这些阈值。"
        elif filename == "prediction_zoom.png":
            extra = f"最大绝对误差位于 flight {int(max_error_row.flight)}、时间 {max_error_row.time:.2f} s；实测/预测功率为 {max_error_row.power_w:.2f}/{max_error_row.predicted_power_w:.2f} W，残差为 {max_error_row.residual_w:+.2f} W。上图显示该点前后最多各 100 条记录，下图给出同一窗口的残差符号和持续时间。"
        elif filename == "flight_energy_error_histogram.png":
            extra = f"红色均值线对应平均任务误差 {fmt(flights.energy_error_wh.mean(), 3)} Wh；若它偏离黑色零线，说明误差不是单纯随机波动。"
        elif filename == "phase_error_distribution.png":
            medians = [f"{row['phase']} {prediction.loc[prediction.phase_name == row['phase'], 'absolute_error_w'].median():.2f} W" for row in summary["phase_rows"]]
            extra = "四阶段绝对误差中位数为 " + "、".join(medians) + "。箱体高表示中间 50% 样本跨度大；图中隐藏离群点，完整尾部应结合 CDF 和 P95。"
        lines += [f"### `{filename}`", "", f"![{filename}](./out/figures/{filename})", "", description, extra, ""]
    return "\n".join(lines)


def method_table(metrics: pd.DataFrame) -> str:
    """功能:生成七种方法及文献来源表。
    参数: metrics为comparison_metrics.csv数据。
    返回: Markdown表格。
    调用位置:overall_readme。
    """
    lines = ["| 算法目录 | 方法 | 代码中实际结构 | 算法来源 |", "|---|---|---|---|"]
    short = {
        "proposed_tcn_rls": "TCN 四残差块 + 时间门控 + 秒级能量反馈 RLS",
        "rf_tlatt_lite": "速度相位分类 + 四个相位专用岭回归头",
        "physical_mlr": "物理量展开 + 标准化多元线性/岭回归",
        "lstm": "两层双向 LSTM + Dropout + Tanh 回归头",
        "lr_tcn_sma": "因果 LeakyReLU TCN 残差块 + 历史 SMA",
        "lstm_transformer": "线性投影 + 位置编码 + LSTM + 多头 Transformer Encoder",
        "cnn_lstm": "两层 Conv1d/BatchNorm + LSTM + 线性头",
    }
    for _, row in metrics.iterrows():
        folder = row["algorithm"]
        lines.append(f"| [`{folder}`](./{folder}/README.md) | {row['algorithm_name']} | {short.get(folder, '见算法README')} | {ALGORITHM_REFERENCES.get(folder, '')} |")
    return "\n".join(lines)


def overall_analysis(metrics: pd.DataFrame, summaries: dict[str, dict[str, object]]) -> str:
    """功能:根据最终CSV生成跨算法数值分析。
    参数: metrics为总体指标表；summaries为各算法诊断统计。
    返回:论文式结果分析文本。
    调用位置:overall_readme。
    """
    best_power = metrics.loc[metrics.sample_power_wape_percent.idxmin()]
    best_energy = metrics.loc[metrics.flight_energy_wape_percent.idxmin()]
    strongest = metrics.sort_values("sample_power_wape_percent").iloc[1]
    test = summaries[best_power.algorithm]["prediction"]
    mean_power = float(test.power_w.mean())
    median_power = float(test.power_w.median())
    min_power = float(test.power_w.min())
    max_power = float(test.power_w.max())
    total_energy = float(summaries[best_power.algorithm]["flights"].actual_energy_wh.sum())
    phase_counts = test.phase_name.value_counts().reindex(PHASES, fill_value=0)
    bins = test.power_bin.value_counts().reindex(POWER_BINS, fill_value=0)
    p50 = test.absolute_error_w.quantile(.5); p90 = test.absolute_error_w.quantile(.9); p95 = test.absolute_error_w.quantile(.95)
    phase_lookup = {
        folder: {row["phase"]: row for row in summary["phase_rows"]}
        for folder, summary in summaries.items()
    }
    baseline = json.loads(BASELINE_EVALUATION_JSON.read_text(encoding="utf-8"))
    raw_tcn_power_wape = float(baseline["tcn_sample_power_w_wape_percent"])
    raw_tcn_energy_wape = float(baseline["tcn_flight_energy_wh_wape_percent"])
    corrected_power_wape = float(baseline["sample_power_w_wape_percent"])
    corrected_energy_wape = float(baseline["flight_energy_wh_wape_percent"])
    lines = [
        "### 8.2 结果与讨论",
        "",
        f"测试集共有 **{len(test):,}** 条采样记录、**{test.flight.nunique()}** 个 flight。实测瞬时功率均值为 {mean_power:.2f} W、中位数为 {median_power:.2f} W、范围为 {min_power:.2f}–{max_power:.2f} W；按真实 `dt_seconds` 积分后，28 个 flight 的实测总能耗为 {total_energy:,.3f} Wh。阶段样本数为 " + "、".join(f"{phase}（{PHASE_CN[phase]}）{int(phase_counts[phase]):,}" for phase in PHASES) + "。功率区间样本数为 " + "、".join(f"{name} {int(bins[name]):,}" for name in POWER_BINS) + "。",
        "",
        f"按样本功率 WAPE 排名，{best_power.algorithm_name} 最低（{best_power.sample_power_wape_percent:.3f}%），其次为 {strongest.algorithm_name}（{strongest.sample_power_wape_percent:.3f}%）；按 flight 能耗 WAPE 排名，{best_energy.algorithm_name} 最低（{best_energy.flight_energy_wape_percent:.3f}%）。主方法的绝对误差 P50/P90/P95 为 {p50:.2f}/{p90:.2f}/{p95:.2f} W，说明大多数采样点误差集中在较小范围，但 P95 仍保留少量起降和状态切换尾部。",
        "",
        f"主方法与最强无测试期目标反馈对照 {strongest.algorithm_name} 的功率 MAE 分别为 {best_power.sample_power_mae_w:.3f} W 和 {strongest.sample_power_mae_w:.3f} W，差值为 {strongest.sample_power_mae_w-best_power.sample_power_mae_w:.3f} W；功率 WAPE 相对下降 {(strongest.sample_power_wape_percent-best_power.sample_power_wape_percent)/strongest.sample_power_wape_percent*100:.2f}%。flight 能耗 WAPE 分别为 {best_energy.flight_energy_wape_percent:.3f}% 和 {strongest.flight_energy_wape_percent:.3f}%。",
        "",
        f"同一 3.0 checkpoint 的原始 TCN 功率 WAPE 为 {raw_tcn_power_wape:.3f}%，加入在线 RLS 后为 {corrected_power_wape:.3f}%，相对下降 {(raw_tcn_power_wape-corrected_power_wape)/raw_tcn_power_wape*100:.2f}%；原始 TCN 的 flight 能耗 WAPE 为 {raw_tcn_energy_wape:.3f}%，RLS 后为 {corrected_energy_wape:.3f}%，相对下降 {(raw_tcn_energy_wape-corrected_energy_wape)/raw_tcn_energy_wape*100:.2f}%。这组同模型前后值说明，主方法的任务级优势主要受在线校正影响。RLS 在每个完整秒窗结束后读取该窗真实能量，并把更新参数用于下一窗；其余对比算法没有同样的测试期目标反馈，因此不能把能耗差异全部归因于 TCN 表征能力。",
        "",
        "阶段结果也呈现出不同的误差来源。idle 的功率分母较小，WAPE 往往高于 MAE 所反映的绝对误差；ascent 和 descent 的垂直速度变化使功率尖峰更密集，RMSE 对这些点更敏感；cruise 的输入变化相对连续，通常更适合时序模型。功率分箱中 450–600 W 是样本主体，因而该区间的 MAE 对总体结果贡献最大；50–300 W 样本较少但多出现在过渡段，适合用于检查模型响应延迟。",
        "",
        f"模型之间的差异与结构特点一致：LSTM-Transformer 的 idle MAE 为 {phase_lookup['lstm_transformer']['idle']['mae']:.2f} W，是七种方法中的最低值；LR-TCN-SMA 的 idle RMSE 为 {phase_lookup['lr_tcn_sma']['idle']['rmse']:.2f} W，是该阶段最低值。主方法在 ascent、descent 和 cruise 的 MAE 分别为 {phase_lookup['proposed_tcn_rls']['ascent']['mae']:.2f}、{phase_lookup['proposed_tcn_rls']['descent']['mae']:.2f} 和 {phase_lookup['proposed_tcn_rls']['cruise']['mae']:.2f} W，均低于六个对照。Physical MLR 在 descent 的 MAE 为 {phase_lookup['physical_mlr']['descent']['mae']:.2f} W，相位岭回归在 idle 的 MAE 为 {phase_lookup['rf_tlatt_lite']['idle']['mae']:.2f} W，反映无历史线性模型对低功率混合分布和状态变化的刻画较弱。以上只描述当前固定划分和单次训练结果；未做重复试验、置信区间或消融，不能据此断言结构差异具有统计显著性。",
    ]
    return "\n".join(lines)


def overall_chart_section(metrics: pd.DataFrame, summaries: dict[str, dict[str, object]]) -> str:
    """功能:生成总体图表嵌入、坐标和逐图数据分析。
    参数: metrics为总体指标表；summaries为各算法统计。
    返回: Markdown图表章节。
    调用位置:overall_readme。
    """
    best = metrics.sort_values("sample_power_wape_percent").iloc[0]
    strongest = metrics.sort_values("sample_power_wape_percent").iloc[1]
    names = metrics.algorithm.tolist()
    display_names = dict(zip(metrics.algorithm, metrics.algorithm_name))
    common_test = summaries[names[0]]["prediction"]
    metric_columns = [
        "sample_power_mae_w", "sample_power_rmse_w", "sample_power_wape_percent",
        "flight_energy_mae_wh", "flight_energy_wape_percent",
    ]
    rank_score = metrics[metric_columns].rank(method="min", ascending=True).mean(axis=1)
    ranked = sorted(((metrics.loc[index, "algorithm_name"], score) for index, score in rank_score.items()), key=lambda item: item[1])
    section = [
        "### 8.3 总体图谱与逐图分析",
        "",
        f"13 张总体图均基于同一批 7 种算法、45,237 条测试记录和 28 个 flight；散点密集的图按图注规则等间隔抽样，但所有表格指标仍用完整测试集计算。图例中的每条线或颜色都对应一个算法目录。当前功率 WAPE 最低的是 {best.algorithm_name}（{best.sample_power_wape_percent:.3f}%），第二名为 {strongest.algorithm_name}（{strongest.sample_power_wape_percent:.3f}%）。",
        "",
    ]
    chart_order = [
        "comparison_metrics.png", "comparison_error_boxplot.png", "comparison_phase_wape.png",
        "comparison_flight_energy_scatter.png", "comparison_metric_heatmap.png", "comparison_error_cdf.png",
        "comparison_overall_rank.png", "comparison_power_timeseries.png", "comparison_error_violin.png",
        "comparison_bland_altman.png", "comparison_energy_error_distribution.png", "comparison_phase_metrics.png",
        "comparison_metric_radar.png",
    ]
    for filename in chart_order:
        section += [f"#### `{filename}`", "", f"![{filename}](./out/{filename})", "", ALGORITHM_CHART_DESCRIPTIONS[filename]]
        if filename == "comparison_metrics.png":
            worst_power = metrics.loc[metrics.sample_power_wape_percent.idxmax()]
            worst_energy = metrics.loc[metrics.flight_energy_wape_percent.idxmax()]
            section.append(
                f"蓝色功率柱从主方法的 {best.sample_power_wape_percent:.3f}% 到 {worst_power.algorithm_name} 的 {worst_power.sample_power_wape_percent:.3f}%；"
                f"橙色能耗柱从主方法的 {best.flight_energy_wape_percent:.3f}% 到 {worst_energy.algorithm_name} 的 {worst_energy.flight_energy_wape_percent:.3f}%。"
                "两组柱使用不同评价对象，不能把蓝柱与橙柱直接相减。"
            )
        elif filename == "comparison_error_boxplot.png":
            for row in metrics.itertuples():
                p = summaries[row.algorithm]["prediction"]
                section.append(f"{row.algorithm_name} 的绝对误差中位数为 {p.absolute_error_w.quantile(.5):.2f} W，P95 为 {p.absolute_error_w.quantile(.95):.2f} W。")
        elif filename == "comparison_phase_wape.png":
            phase_sentences = []
            for phase in PHASES:
                rows = [(name, next(item for item in summaries[name]["phase_rows"] if item["phase"] == phase)["wape"]) for name in names]
                phase_best = min(rows, key=lambda item: item[1])
                phase_sentences.append(f"{phase} 最低为 {display_names[phase_best[0]]} {phase_best[1]:.2f}%")
            section.append("每个格子由该阶段全部记录先汇总分子、分母后计算 WAPE。" + "；".join(phase_sentences) + "。")
        elif filename == "comparison_flight_energy_scatter.png":
            details = []
            for name in names:
                flights = summaries[name]["flights"]
                details.append(f"{display_names[name]} 平均偏差 {flights.energy_error_wh.mean():+.3f} Wh、最大绝对偏差 {flights.energy_error_wh.abs().max():.3f} Wh")
            section.append("每个小图有 28 个点，线上方为能耗高估、下方为低估。" + "；".join(details) + "。")
        elif filename == "comparison_metric_heatmap.png":
            section.append(
                f"主方法在五列中均为最小值，因此五格都位于绿色端；最强无测试期目标反馈方法 {strongest.algorithm_name} 的功率 MAE/RMSE/WAPE 为 "
                f"{strongest.sample_power_mae_w:.2f} W、{strongest.sample_power_rmse_w:.2f} W、{strongest.sample_power_wape_percent:.2f}%。"
                "底色按每一列单独做最小-最大归一化，格内数字仍保留 W、Wh 或 % 的原单位。"
            )
        elif filename == "comparison_error_cdf.png":
            quantiles = []
            for name in names:
                values = summaries[name]["prediction"].absolute_error_w
                quantiles.append(f"{display_names[name]} {values.quantile(.50):.2f}/{values.quantile(.90):.2f}/{values.quantile(.95):.2f} W")
            section.append("在纵轴 $F(a)=0.50/0.90/0.95$ 处向下读取横轴，分别得到 P50/P90/P95；当前数据依次为：" + "；".join(quantiles) + "。")
        elif filename == "comparison_overall_rank.png":
            section.append("五列误差分别从小到大排名后求平均，结果为：" + "、".join(f"{name} {score:.1f}" for name, score in ranked) + "。该值只作描述性折中，不是显著性检验。")
        elif filename == "comparison_power_timeseries.png":
            flight_id = common_test.groupby("flight").power_w.std().sort_values(ascending=False).index[0]
            flight_part = common_test[common_test.flight == flight_id].iloc[:700]
            rows = flight_part.index.to_numpy()
            trace_errors = []
            for name in names:
                prediction = summaries[name]["prediction"].loc[rows]
                trace_errors.append(f"{display_names[name]} MAE {prediction.absolute_error_w.mean():.2f} W")
            section.append(
                f"程序选择功率标准差最大的 flight {flight_id}，显示其前 {len(flight_part)} 条记录；黑线实测功率范围为 "
                f"{flight_part.power_w.min():.2f}–{flight_part.power_w.max():.2f} W。该显示区间内，" + "、".join(trace_errors) + "。"
            )
        elif filename == "comparison_error_violin.png":
            details = []
            for name in names:
                sampled = summaries[name]["prediction"].absolute_error_w.iloc[::5]
                details.append(f"{display_names[name]} 抽样均值/中位数 {sampled.mean():.2f}/{sampled.median():.2f} W")
            section.append("图中密度和内部标记按每 5 条取 1 条的显示样本计算：" + "；".join(details) + "。")
        elif filename == "comparison_bland_altman.png":
            details = []
            step = max(1, len(common_test) // 6000)
            for name in names:
                residual = summaries[name]["prediction"].residual_w.iloc[::step]
                mean = residual.mean(); std = residual.std()
                details.append(f"{display_names[name]} {mean:+.2f} W（{mean-1.96*std:.2f}, {mean+1.96*std:.2f} W）")
            section.append("以下为红色平均残差及两条橙色 95% 一致性界限：" + "；".join(details) + "。界限是描述性样本界限，不是预测区间。")
        elif filename == "comparison_energy_error_distribution.png":
            details = []
            for name in names:
                error = summaries[name]["flights"].energy_error_wh
                details.append(f"{display_names[name]} 中位数 {error.median():+.3f} Wh、范围 {error.min():+.3f} 至 {error.max():+.3f} Wh")
            section.append("任务级误差按预测能耗减实测能耗计算：" + "；".join(details) + "。正负逐点误差可能在积分中抵消，因此应与功率箱线图联合阅读。")
        elif filename == "comparison_phase_metrics.png":
            details = []
            for phase in PHASES:
                rows = [(name, next(item for item in summaries[name]["phase_rows"] if item["phase"] == phase)) for name in names]
                best_mae = min(rows, key=lambda item: item[1]["mae"])
                best_rmse = min(rows, key=lambda item: item[1]["rmse"])
                details.append(f"{phase} 的最低 MAE 为 {display_names[best_mae[0]]} {best_mae[1]['mae']:.2f} W，最低 RMSE 为 {display_names[best_rmse[0]]} {best_rmse[1]['rmse']:.2f} W")
            section.append("三张热图分别使用 W、W 和 %。" + "；".join(details) + "。idle 的 WAPE 还会被接近 0 W 的真实值放大。")
        elif filename == "comparison_metric_radar.png":
            section.append(
                "主方法在五项误差中均为当前最小值，因而五个顶点都位于外圈；六种无测试期目标反馈对照中，LSTM-Transformer 的三项功率误差最低，LSTM 的两项能耗误差最低。"
                "雷达值是七种方法内的最小-最大相对得分，不代表绝对精度，也不包含 R² 或计算成本。"
            )
        section.append("")
    return "\n".join(section)


def build_overall_readme(metrics: pd.DataFrame, summaries: dict[str, dict[str, object]], metadata: dict[str, object]) -> str:
    """功能:组合论文式总体README正文。
    参数: metrics为总体结果；summaries为算法统计；metadata为特征元数据。
    返回:完整README字符串。
    调用位置:main。
    """
    train, validation, test, _ = load_data()
    best = metrics.loc[metrics.sample_power_wape_percent.idxmin()]
    strongest = metrics.sort_values("sample_power_wape_percent").iloc[1]
    lines = [
        "# Comparison-algorithm：四轴无人机功率与能耗对比实验",
        "",
        "## 摘要",
        "",
        f"针对 R1 航线上的四轴无人机瞬时功率与单次任务能耗预测，本文在固定 train/validation/test flight 划分上比较 3.0 TCN+RLS 主方法和六种论文来源基线。两项原候选因功率 WAPE 达到 26.516% 和 18.088% 而被替换，新加入 LSTM-Transformer 与 CNN-LSTM。最终主方法在 45,237 条测试记录上取得 {best.sample_power_mae_w:.3f} W 的功率 MAE 和 {best.sample_power_wape_percent:.3f}% 的功率 WAPE，在 28 个测试 flight 上取得 {best.flight_energy_wape_percent:.3f}% 的能耗 WAPE；最强无测试期目标反馈对照 {strongest.algorithm_name} 的功率 WAPE 为 {strongest.sample_power_wape_percent:.3f}%。由于主方法在测试期使用已结束秒窗的真实能量更新下一窗 RLS 参数，任务级能耗差异同时包含在线反馈收益。所有结论限于当前固定划分和单次随机种子。",
        "",
        "**关键词：** 四轴无人机；功率预测；能耗预测；时间卷积网络；递推最小二乘；对比实验",
        "",
        "## 1. 研究背景与实验目的",
        "",
        "无人机逐点功率可由飞行内电压、电流和工况记录建立监督数据<sup>[B1]</sup>；预测结果可为返航阈值、剩余航时和任务调度提供输入。任务总能耗还能进入配送路径成本<sup>[B2]</sup>、编队任务代价<sup>[B3]</sup>和多旋翼任务可行性判断<sup>[B4]</sup>。本实验的目的，是在相同 flight 划分、相同功率目标、相同真实时间积分和相同评价指标下，比较带秒级能量反馈的 TCN+RLS 与六种时序、相位和物理基线。各方法使用同一份源数据，但依据论文结构选择不同输入表示，因此“统一条件”不等于所有模型都读取相同特征子集或相同时间窗。",
        "",
        "## 2. 数据、场景与实验边界",
        "",
        f"数据来自 3.0 的 R1 处理集：训练集 {len(train):,} 条、{train.flight.nunique()} 个 flight；验证集 {len(validation):,} 条、{validation.flight.nunique()} 个 flight；测试集 {len(test):,} 条、{test.flight.nunique()} 个 flight。典型采样间隔约为 {test.dt_seconds.median():.2f} s，功率目标为 `power_w`（W），能耗按每条记录的 `dt_seconds` 积分为 Wh。数据按完整 flight 划分，避免同一飞行同时出现在训练与测试中。",
        "",
        "当前场景是单架四轴无人机在 R1 航线上的逐点功率与任务能耗估计。Luo 等把相位识别与专用时序模型用于四旋翼功率预测<sup>[A1]</sup>；Muli 等比较无人机数据驱动能耗模型<sup>[A3]</sup>；Dudukcu 等研究带 SMA 的 LR-TCN 瞬时功率模型<sup>[A4]</sup>；Jastrzębska 等从物理运动量建立 UAV 能耗模型<sup>[A2]</sup>；Feng 等在电动汽车上验证 LSTM-Transformer 的长短期组合<sup>[A5]</sup>；Cabuk 等把小型无人机能耗用于编队连接恢复、拓扑维护和队形变换的任务代价<sup>[B3]</sup>。本实验只迁移这些建模思想，论文原数据集上的数值不作为本项目结果。",
        "",
        "主方法的部署边界需要单列说明：3.0 在每个完整约 1 s 窗口结束后读取该窗口真实能量，RLS 更新后的偏置和缩放只用于下一窗口；六种对比方法在测试时没有同样的目标反馈。因此主方法的 flight 能耗优势包含在线反馈带来的收益，不能把全部差异归因于 TCN 的前馈特征提取。",
        "",
        "## 3. 方法与文献来源",
        "",
        method_table(metrics),
        "",
        "算法来源文献只在本节及各算法 README 的‘算法来源’中列出；研究背景和应用场景文献放在文末的 B 类列表，避免把方法来源与背景引用混在一起。正文中的 `<sup>[A*]</sup>` 和 `<sup>[B*]</sup>` 是规范上标引用。",
        "",
        "## 4. 统一运行方式",
        "",
        "```powershell",
        "cd I:\\STUDY\\python\\project\\Energy-prediction\\Comparison-algorithm",
        "python main.py",
        "python main.py --algorithm lstm_transformer cnn_lstm",
        "python generate_documentation.py",
        "```",
        "",
        "`main.py` 从 `config.py` 读取路径和算法注册表；`comparison_engine.py` 读取 train/validation/test，先在 validation 上选择候选，再生成测试预测、指标、诊断图和总体图；`generate_documentation.py` 只读取最终 CSV/JSON 并重建本文档。每个算法目录均包含 `algorithm.py`、独立 `README.md`、`out/predictions.csv`、`out/flight_energy_summary.csv`、`out/metrics.csv/json`、`out/model/tuning_results.csv`（适用时）和 `out/figures/`。",
        "",
        "## 5. 统一输入变量",
        "",
        f"公共处理表提供 23 个候选字段：{', '.join(f'`{column}`' for column in FEATURE_COLUMNS)}。TCN+RLS 和四个深度基线读取全部 23 个字段；RF-TLATT 轻量基线和 Physical MLR 从这些字段构造较小的当前时刻特征集。各模型实际输入如下。",
        "",
        algorithm_input_table(metrics),
        "",
        "下表说明公共字段的中英文全称、单位、构造方式和物理含义。字段出现在公共表中不等于每个算法都直接使用它。",
        "",
        feature_table(metadata),
        "",
        "## 6. 算法原理与逐层结构",
        "",
    ]
    for folder in metrics.algorithm:
        spec = ALGORITHM_SPECS[folder]
        lines += [f"### {spec['title']} {ALGORITHM_REFERENCES[folder]}", "", f"**实现来源：** {spec['source']}。", "", spec["structure"], "", spec["formula"], ""]
    lines += [
        "## 7. 调参与公平性设置",
        "",
        "所有需要学习的候选均只使用训练 flight 拟合，在独立 validation flight 上选择，test flight 只在结构与参数固定后用于最终评价。每个深度候选以 SmoothL1 损失最多训练 24 轮，验证损失连续 4 轮不改善则早停，并保存该候选验证损失最低的 epoch；不同候选再按功率 WAPE + 0.5×flight 能耗 WAPE 选择。线性模型只在验证集选择正则项和 RF 轻量基线的相位阈值。候选明细位于各目录 `out/model/tuning_results.csv`。",
        "",
        "主方法 3.0 的固定 TCN checkpoint 使用 12 s/100 步输入窗和 `[64,64,64,32]` 通道；RLS 使用遗忘因子 0.90、初始协方差 0.25、无预热窗口，这些参数来自 3.0 验证集搜索。LSTM、LR-TCN-SMA、LSTM-Transformer、CNN-LSTM 使用统一 17 步窗口并在本项目验证集重新选参；RF 相位基线和 Physical MLR 是当前记录回归，不使用序列窗口。由于主方法另有在线真实能量反馈，本实验在数据划分与指标上保持一致，但并非严格同信息量的消融比较。",
        "",
        "### 7.1 低效候选删除与替换",
        "",
        "先前两项候选在相同测试集上的误差显著高于时序模型，因此不再进入有效算法清单。替换算法均来自所给论文体系：LSTM-Transformer 是 Feng 等提出的混合框架<sup>[A5]</sup>，CNN-LSTM 是 Luo 等表 8 报告的对照名称<sup>[A1]</sup>。替换前后的固定结果如下。",
        "",
        replacement_table(),
        "",
        "## 8. 评价指标与结果",
        "",
        COMMON_FORMULAS,
        "",
        "### 8.1 测试集总体指标",
        "",
        "| 方法 | 功率 MAE(W) | 功率 RMSE(W) | 功率 R² | 功率 WAPE(%) | flight能耗 MAE(Wh) | flight能耗 RMSE(Wh) | flight能耗 WAPE(%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"| {row.algorithm_name} | {row.sample_power_mae_w:.3f} | {row.sample_power_rmse_w:.3f} | {row.sample_power_r2:.4f} | {row.sample_power_wape_percent:.3f} | {row.flight_energy_mae_wh:.3f} | {row.flight_energy_rmse_wh:.3f} | {row.flight_energy_wape_percent:.3f} |")
    lines += ["", overall_analysis(metrics, summaries), "", overall_chart_section(metrics, summaries), "", VARIABLE_GLOSSARY]
    lines += [
        "",
        "## 9. 输出文件与目录",
        "",
        "```text",
        "Comparison-algorithm/",
        "├─ config.py                         # 路径、23维特征、候选参数和图表名称",
        "├─ main.py                           # 全部算法统一入口",
        "├─ comparison_engine.py              # 算法、训练、评估和绘图",
        "├─ generate_documentation.py         # 从最终CSV生成README",
        "├─ proposed_tcn_rls/out/             # 主方法逐点/flight结果和13张图",
        "├─ rf_tlatt_lite/out/                # 相位轻量基线结果和13张图",
        "├─ physical_mlr/out/                 # 物理MLR结果和13张图",
        "├─ lstm/out/                         # 双向LSTM结果和13张图",
        "├─ lr_tcn_sma/out/                   # LR-TCN-SMA结果和13张图",
        "├─ lstm_transformer/out/             # LSTM-Transformer结果和13张图",
        "├─ cnn_lstm/out/                     # CNN-LSTM结果和13张图",
        "└─ out/                              # 跨算法CSV、JSON和13张总体图",
        "```",
        "",
        "每个算法 `out/figures/` 的 13 张图依次覆盖散点、典型时序、残差直方图、残差-功率、功率分箱、flight 能耗散点、flight 误差排序、累计能耗、阶段柱图、CDF、最大误差局部放大、flight 能耗误差直方图和阶段误差箱线图；根目录 `out/` 另有 13 张跨算法图，包括柱图、箱线图、阶段热图、散点矩阵、归一化热图、CDF、综合排名、同 flight 曲线、小提琴图、Bland–Altman 图、能耗误差分布、阶段三指标热图和雷达图。",
        "",
        "## 10. 参考文献",
        "",
        ALGORITHM_REFERENCE_LIST,
        "",
        BACKGROUND_REFERENCES,
        "",
        "PDF 原件目录：`C:\\Users\\18030\\Desktop\\GPT-files\\Energy-prediction-Comparison-algorithm\\pdf`。所收录论文分别采用瞬时功率、任务能耗和固定时长归一化能耗等口径，研究对象也涵盖四旋翼、多旋翼和电动汽车；其原文指标不能与本实验按真实 `dt_seconds` 积分得到的 R1 指标直接换算或相减，相关论文只用于说明算法机制、应用场景和图表组织依据。",
    ]
    return "\n".join(lines) + "\n"


def build_algorithm_readme(folder: str, metrics: pd.DataFrame, summaries: dict[str, dict[str, object]], metadata: dict[str, object]) -> str:
    """功能:生成单个算法目录的完整README。
    参数: folder为算法目录；metrics为总体指标；summaries为统计字典；metadata为特征元数据。
    返回:算法README字符串。
    调用位置:main。
    """
    row = metrics[metrics.algorithm == folder].iloc[0]
    summary = summaries[folder]
    spec = ALGORITHM_SPECS[folder]
    train, validation, test, _ = load_data()
    lines = [
        f"# {spec['title']}",
        "",
        f"**算法来源：** {spec['source']}。正文引用 {ALGORITHM_REFERENCES[folder]}。",
        "",
        "## 1. 方法概述",
        "",
        spec["structure"],
        "",
        spec["formula"],
        "",
        "## 2. 统一实验条件",
        "",
        f"本算法与其他方法使用同一 R1 源数据和同一完整 flight 划分：训练 {len(train):,} 条、{train.flight.nunique()} 个 flight；验证 {len(validation):,} 条、{validation.flight.nunique()} 个 flight；测试 {len(test):,} 条、{test.flight.nunique()} 个 flight。实际模型输入为：{ALGORITHM_INPUTS[folder]}。测试功率目标统一为 `power_w`，能耗统一按每条记录的 `dt_seconds` 积分；测试集只用于最终报告，不参与候选选择。",
        "",
        f"本次最终测试结果：功率 MAE={row.sample_power_mae_w:.3f} W，RMSE={row.sample_power_rmse_w:.3f} W，R²={row.sample_power_r2:.4f}，WAPE={row.sample_power_wape_percent:.3f}%；flight 能耗 MAE={row.flight_energy_mae_wh:.3f} Wh，RMSE={row.flight_energy_rmse_wh:.3f} Wh，R²={row.flight_energy_r2:.4f}，WAPE={row.flight_energy_wape_percent:.3f}%。",
        "",
        "## 3. 验证集调参记录",
        "",
        tuning_markdown(folder, summary),
        "",
        "## 4. 评价指标和变量",
        "",
        COMMON_FORMULAS,
        "",
        VARIABLE_GLOSSARY,
        "",
        "下表列出公共数据中的 23 个候选字段。当前算法是否直接使用某字段，以第 2 节的实际输入和第 1 节逐层结构为准。",
        "",
        feature_table(metadata),
        "",
        phase_markdown(summary),
        "",
        algorithm_chart_analysis(folder, summary),
        "",
        "## 6. 输出文件",
        "",
        "- `out/predictions.csv`：每条测试记录的 flight、时间、真实功率、预测功率和采样能量。",
        "- `out/flight_energy_summary.csv`：每个 flight 的实测能耗、预测能耗、记录数和 `energy_error_wh`。",
        "- `out/metrics.csv`、`out/metrics.json`：功率级和 flight 能耗级 MAE、RMSE、R²、MAPE、WAPE。",
        "- `out/model/tuning_results.csv`：验证集候选参数、验证指标、选择分数和 `selected` 标记（固定 checkpoint 方法没有该文件）。",
        "- `out/model/training_log.csv`：深度模型各候选每轮训练损失、验证损失和设备信息。",
        "- `out/figures/`：本 README 中嵌入的 13 张诊断图。",
        "",
        "## 7. 应用场景与限制",
        "",
        f"{ALGORITHM_APPLICATIONS[folder]} {ALGORITHM_REFERENCES[folder]}",
        "",
        "统一诊断阶段来自速度阈值而非人工标签；所有指标只对应当前 R1 固定划分和一个随机种子。不同论文的原始对象、输入字段、采样条件和预测目标并不相同，其原文指标不能与本目录数值直接横向相减。本实验未做重复训练、置信区间或显著性检验。",
        "",
        "## 8. 参考文献",
        "",
        algorithm_reference_markdown(folder),
        "",
        "背景和应用文献见上级 [README](../README.md) 的 B 类列表。PDF 原件目录：`C:\\Users\\18030\\Desktop\\GPT-files\\Energy-prediction-Comparison-algorithm\\pdf`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """功能:从最终实验CSV生成总README和七个算法README。
    参数:无，输入文件由config.py和项目out目录确定。
    返回:无，写入UTF-8 Markdown文件。
    调用位置:命令行 `python generate_documentation.py`。
    """
    metrics = pd.read_csv(METRICS_CSV, encoding="utf-8")
    _, _, test, metadata = load_data()
    summaries = {folder: data_summary(folder, test) for folder in metrics.algorithm}
    (ROOT / "README.md").write_text(build_overall_readme(metrics, summaries, metadata), encoding="utf-8")
    for folder in metrics.algorithm:
        (ROOT / folder / "README.md").write_text(build_algorithm_readme(folder, metrics, summaries, metadata), encoding="utf-8")
    print(f"generated README for {len(metrics)} algorithms")


if __name__ == "__main__":
    main()
