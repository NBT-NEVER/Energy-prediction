# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-07-08
# 文件名: main.py
# 功能说明: 统一调度实验3.2的TCN前向、秒级能量RLS校正、训练、评估和可视化流程
# 版本号：3.2

import argparse
import time
from datetime import datetime
from pathlib import Path

from config import build_config, ensure_directories  # 构建实验配置并创建路径目录
from data_utils import download_source_dataset, extract_source_zip, prepare_dataset  # 下载、解包和切分数据
from evaluate import evaluate_model  # 生成模型评估指标
from predict import calibrate_uncertainty, predict_from_csv, recalculate_prediction_intervals  # 执行预测和置信区间处理
from train import compare_rls_energy_windows, train_fixed_tcn, train_model, tune_rls_only  # 执行固定超参数训练、TCN调参和RLS调参
from terminal_logger import TerminalLogCapture, log_result  # 保留终端过程并记录阶段结果
from visualize import generate_all_visualizations, predict_custom_scenario  # 生成图表并预测自定义工况
from route_visualization import generate_route_products  # 生成18、23、83航线的IMU轨迹和功率能量产物
from plot_rls_window_analysis import main as plot_rls_window_analysis  # 绘制RLS能量窗变量分析图


def build_parser() -> argparse.ArgumentParser:
    """功能: 构建命令行参数解析器。
    参数: 无。
    返回: argparse.ArgumentParser对象。
    调用位置: main。
    """

    parser = argparse.ArgumentParser(description="四轴无人机飞行能耗预测实验3.2：TCN + 秒级能量监督RLS")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["download", "prepare", "tune-tcn", "train", "train-fixed", "tune-rls", "compare-rls-windows", "route-visualize", "test", "calibrate", "predict", "interval", "evaluate", "visualize", "custom", "all"],
        help="运行模式，all会依次完成数据准备、TCN调参、固定训练、RLS调参、校准、评估和可视化。",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="覆盖原始数据目录。")
    parser.add_argument("--save-dir", type=Path, default=None, help="覆盖模型权重保存目录。")
    parser.add_argument("--out-dir", type=Path, default=None, help="覆盖统一输出目录。")
    parser.add_argument("--epochs", type=int, default=None, help="最终训练轮数。")
    parser.add_argument("--tune-epochs", type=int, default=None, help="每组候选超参的调参训练轮数。")
    parser.add_argument("--batch-size", type=int, default=None, help="训练或预测批量大小。")
    parser.add_argument("--learning-rate", type=float, default=None, help="基础学习率。")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW权重衰减。")
    parser.add_argument("--patience", type=int, default=None, help="早停等待轮数。")
    parser.add_argument("--target-transform", choices=["log1p", "none"], default=None, help="目标功率训练变换。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子。")
    parser.add_argument("--device", default=None, help="训练或预测使用的 CUDA 设备，默认cuda。")
    parser.add_argument("--window-candidates", default=None, help="覆盖时间窗候选秒数，逗号分隔，例如0.6,1.0,1.5。")
    parser.add_argument("--tcn-channels", default=None, help="覆盖TCN通道深度，逗号分隔，例如64,64,64,32。")
    parser.add_argument("--force-prepare", action="store_true", help="强制重新生成处理后数据和切分文件。")
    parser.add_argument("--input-csv", type=Path, default=None, help="predict模式的输入CSV。")
    parser.add_argument("--output-csv", type=Path, default=None, help="predict模式的输出CSV。")
    parser.add_argument("--confidence", type=float, default=None, help="置信度，取0到1之间，例如0.95；interval模式可即时修改。")

    parser.add_argument("--custom-csv", type=Path, default=None, help="custom模式的自定义工况CSV。")
    parser.add_argument("--duration-s", type=float, default=180.0, help="默认自定义工况总时长，单位s。")
    parser.add_argument("--sample-dt", type=float, default=1.0, help="默认自定义工况采样间隔，单位s。")
    parser.add_argument("--wind-speed", type=float, default=4.0, help="默认空气流速，单位m/s。")
    parser.add_argument("--wind-angle", type=float, default=0.0, help="默认风向角，单位deg。")
    parser.add_argument("--flight-speed", type=float, default=8.0, help="默认巡航速度，单位m/s。")
    parser.add_argument("--payload-g", type=float, default=250.0, help="默认载荷，单位g。")
    parser.add_argument("--altitude", type=float, default=50.0, help="默认飞行高度，单位m。")
    parser.add_argument("--route", default="R1", help="默认航线编号，如R1、R2或H。")
    return parser


def print_dict(title: str, payload: dict) -> None:
    """功能: 打印流程摘要。
    参数: title为标题，payload为摘要字典。
    返回: None。
    调用位置: main。
    """

    print(f"\n[{title}]")
    for key, value in payload.items():
        print(f"{key}: {value}")
    log_result(title, payload)


def run_timed_stage(label: str, action):
    """功能: 执行一个阶段并在终端显示起止时间和耗时。
    参数: label为阶段名称，action为无参数阶段函数。
    返回: action的返回值。
    调用位置: run_mode的all流程及独立训练流程。
    """

    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n>>> {label} 开始: {started_at}")
    result = action()
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f">>> {label} 完成: {finished_at} | 耗时 {elapsed:.2f}s")
    log_result(label, {"started_at": started_at, "finished_at": finished_at, "elapsed_seconds": round(elapsed, 3), "status": "完成"})
    return result


def print_process_intro(mode: str) -> None:
    """功能: 输出当前运行模式及各阶段的处理内容。
    参数: mode 为命令行指定的运行模式。
    返回: 无。
    调用位置: main。
    """

    descriptions = {
        "download": "下载公开数据仓库并解压建模所需 CSV 文件。",
        "prepare": "清洗飞行记录，构造特征，并按 flight 划分训练、验证和测试集。",
        "train": "标准化训练数据，比较候选网络，训练并保存最优模型。",
        "train-fixed": "读取当前选定的TCN时间窗，跳过超参数搜索并正式训练TCN。",
        "tune-tcn": "仅搜索TCN时间窗候选并保存当前最优TCN模型。",
        "tune-rls": "加载当前训练模型，仅搜索RLS遗忘因子和初始协方差，能量窗固定为参考值。",
        "compare-rls-windows": "固定最优RLS参数，比较1、2、5、10、20秒能量反馈窗对误差和更新统计的影响。",
        "route-visualize": "读取固定模型测试预测和原始IMU，为18、23、83航线生成独立静态图与轨迹动图。",
        "test": "加载训练好的TCN和RLS参数，运行测试集并生成结果图。",
        "calibrate": "使用验证集残差生成在线RLS和固定RLS的置信区间校准文件。",
        "predict": "加载最优模型，对输入 CSV 逐批预测功率和区间能耗。",
        "interval": "读取已有预测 CSV，只重算指定置信度下的功率和累计能耗区间。",
        "evaluate": "生成测试集预测，统计逐点功率和整次飞行能耗误差。",
        "visualize": "读取训练和评估输出，生成损失、误差及预测曲线。",
        "custom": "构造或读取自定义工况，预测功率与累计能耗。",
        "all": "依次执行数据准备、TCN调参、固定参数训练、RLS调参、校准、测试评估和结果可视化。",
    }
    print("\n" + "=" * 72)
    print("四轴无人机飞行能耗预测实验 3.2：TCN + 秒级能量监督RLS")
    print(f"当前模式: {mode}")
    print(f"运行内容: {descriptions[mode]}")
    if mode == "all":
        print("执行顺序: prepare -> tune-tcn -> train-fixed -> tune-rls -> calibrate -> evaluate -> visualize")
    print("=" * 72)


def run_mode(args: argparse.Namespace, cfg) -> None:
    """功能: 根据命令行模式执行实验3.2的具体流程。
    参数: args为命令行参数，cfg为实验配置对象。
    返回: None。
    调用位置: main。
    """

    print_process_intro(args.mode)
    print(
        f"训练配置: TCN深度={len(cfg.tcn_channels)}块，通道={list(cfg.tcn_channels)}，"
        f"调参轮数={cfg.tune_epochs}，最终轮数={cfg.epochs}，批大小={cfg.batch_size}，默认置信度={cfg.default_confidence:.1%}"
    )
    print(f"TCN时间窗候选: {list(cfg.window_seconds_candidates)}")
    print(f"RLS调参候选: 遗忘因子={list(cfg.rls_forgetting_factors)}，初始协方差={list(cfg.rls_initial_covariances)}，共49组；能量窗变量分析={list(cfg.rls_energy_window_candidates)}s，warmup固定为0")

    if args.mode == "download":
        print("\n[下载数据] 检查公开数据仓库和原始压缩包。")
        download_source_dataset(cfg)
        extract_source_zip(cfg)
        print_dict("download", {"raw_flights_csv": str(cfg.raw_flights_csv), "raw_parameters_csv": str(cfg.raw_parameters_csv)})
    elif args.mode == "prepare":
        print("\n[数据准备] 清洗记录、构造特征并划分数据集。")
        summary = prepare_dataset(cfg, force=args.force_prepare)
        print_dict("prepare", summary)
    elif args.mode == "tune-tcn":
        print("\n[TCN超参数] 仅比较当前TCN时间窗候选。")
        summary = run_timed_stage("tune-tcn：TCN时间窗搜索", lambda: train_model(cfg, stop_after_tcn=True))
        print_dict("tune-tcn", summary)
    elif args.mode == "train":
        print("\n[模型训练] 调参后训练并保存最优网络。")
        summary = train_model(cfg)
        print_dict("train", summary)
    elif args.mode == "train-fixed":
        print("\n[固定参数训练] 使用当前选定TCN参数正式训练，不重复搜索超参数。")
        summary = run_timed_stage("train-fixed：固定TCN参数正式训练", lambda: train_fixed_tcn(cfg))
        print_dict("train-fixed", summary)
    elif args.mode == "tune-rls":
        print("\n[RLS超参数] 加载训练好的TCN，仅搜索遗忘因子和初始协方差；能量窗不参与搜索。")
        summary = run_timed_stage("tune-rls：RLS参数搜索", lambda: tune_rls_only(cfg))
        print_dict("tune-rls", summary)
    elif args.mode == "compare-rls-windows":
        print("\n[RLS能量窗变量分析] 固定RLS最优参数，逐个测试五个能量反馈时间窗。")
        summary = run_timed_stage("compare-rls-windows：RLS能量窗变量分析", lambda: compare_rls_energy_windows(cfg))
        print_dict("compare-rls-windows", summary)
    elif args.mode == "route-visualize":
        print("\n[航线可视化] 为18、23、83航线生成独立功率、能量、IMU轨迹图和GIF。")
        summary = run_timed_stage("route-visualize：航线产物生成", generate_route_products)
        run_timed_stage("route-visualize：RLS能量窗分析图", plot_rls_window_analysis)
        print_dict("route-visualize", {"routes": list(summary), "output_dir": str(cfg.out_dir / "routes")})
    elif args.mode == "test":
        print("\n[独立测试] 加载当前TCN和RLS参数，生成测试指标与图表。")
        metrics = evaluate_model(cfg)
        visual_summary = generate_all_visualizations(cfg)
        print_dict("evaluate", metrics)
        print_dict("visualize", visual_summary)
    elif args.mode == "predict":
        print("\n[批量预测] 加载模型并输出预测 CSV。")
        output_path = predict_from_csv(cfg, args.input_csv, args.output_csv, confidence=args.confidence)
        print_dict("predict", {"prediction_file": str(output_path)})
    elif args.mode == "calibrate":
        print("\n[置信区间校准] 使用验证集误差建立可调置信度校准文件。")
        summary = calibrate_uncertainty(cfg)
        print_dict("calibrate", summary)
    elif args.mode == "interval":
        print("\n[即时区间] 不重新运行模型，只更新已有预测的置信度区间。")
        output_path, summary = recalculate_prediction_intervals(cfg, args.input_csv, args.output_csv, args.confidence)
        print_dict("interval", {"prediction_file": str(output_path), **summary})
    elif args.mode == "evaluate":
        print("\n[模型评估] 生成测试预测并计算误差指标。")
        metrics = evaluate_model(cfg)
        print_dict("evaluate", metrics)
    elif args.mode == "visualize":
        print("\n[结果可视化] 生成训练、评估和预测图表。")
        summary = generate_all_visualizations(cfg)
        print_dict("visualize", summary)
    elif args.mode == "custom":
        print("\n[自定义工况] 预测指定飞行条件下的功率和累计能耗。")
        summary = predict_custom_scenario(
            cfg,
            args.custom_csv,
            args.duration_s,
            args.sample_dt,
            args.wind_speed,
            args.wind_angle,
            args.flight_speed,
            args.payload_g,
            args.altitude,
            args.route,
            confidence=args.confidence,
        )
        print_dict("custom", summary)
    elif args.mode == "all":
        print("\n[1/7 prepare] 清洗记录、构造特征并划分数据集。")
        data_summary = run_timed_stage("prepare：数据准备", lambda: prepare_dataset(cfg, force=args.force_prepare))
        print("\n[2/7 tune-tcn] 搜索TCN时间窗，并测试每个候选窗口。")
        tune_summary = run_timed_stage("tune-tcn：TCN时间窗搜索", lambda: train_model(cfg, stop_after_tcn=True))
        print("\n[3/7 train-fixed] 使用已选TCN参数正式训练，不重复搜索。")
        fixed_summary = run_timed_stage("train-fixed：固定TCN参数正式训练", lambda: train_fixed_tcn(cfg))
        print("\n[4/7 tune-rls] 固定正式TCN模型，搜索RLS参数。")
        rls_summary = run_timed_stage("tune-rls：RLS参数搜索", lambda: tune_rls_only(cfg))
        print("\n[5/8 compare-rls-windows] 固定RLS参数，比较五个能量窗。")
        window_summary = run_timed_stage("compare-rls-windows：RLS能量窗变量分析", lambda: compare_rls_energy_windows(cfg))
        print("\n[6/8 calibrate] 生成预测区间校准结果。")
        calibration_summary = run_timed_stage("calibrate：置信区间校准", lambda: calibrate_uncertainty(cfg))
        print("\n[7/8 evaluate] 生成测试预测并计算误差指标。")
        metrics = run_timed_stage("evaluate：测试集评估", lambda: evaluate_model(cfg))
        print("\n[8/8 visualize] 生成训练、评估和预测图表。")
        visual_summary = run_timed_stage("visualize：结果可视化", lambda: generate_all_visualizations(cfg))
        print_dict("prepare", data_summary)
        print_dict("tune-tcn", tune_summary)
        print_dict("train-fixed", fixed_summary)
        print_dict("tune-rls", rls_summary)
        print_dict("compare-rls-windows", window_summary)
        print_dict("calibrate", calibration_summary)
        print_dict("evaluate", metrics)
        print_dict("visualize", visual_summary)


def main() -> None:
    """功能: 构建配置并在终端日志上下文中调度命令行流程。
    参数: 无。
    返回: None。
    调用位置: 命令行入口。
    """

    args = build_parser().parse_args()
    cfg = build_config(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        tune_epochs=args.tune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        target_transform=args.target_transform,
        random_seed=args.seed,
        device=args.device,
        window_seconds_candidates=tuple(float(item) for item in args.window_candidates.split(",")) if args.window_candidates else None,
        tcn_channels=tuple(int(item) for item in args.tcn_channels.split(",")) if args.tcn_channels else None,
        default_confidence=args.confidence,
    )
    ensure_directories(cfg)
    # 捕获整个业务流程的标准输出、进度刷新和异常信息。
    with TerminalLogCapture(cfg.terminal_log_file, args.mode) as terminal_log:
        run_mode(args, cfg)
    if terminal_log.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
