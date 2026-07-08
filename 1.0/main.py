# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-07-08
# 文件名: main.py
# 功能说明: 统一调度无人机能耗预测实验1.0的数据处理、训练、预测、评估和可视化流程
# 版本号：1.0

import argparse
from pathlib import Path

from config import build_config, ensure_directories  # 构建实验配置并创建路径目录
from data_utils import download_source_dataset, extract_source_zip, prepare_dataset  # 下载、解包和切分数据
from evaluate import evaluate_model  # 生成模型评估指标
from predict import predict_from_csv  # 执行模型预测
from train import train_model  # 执行GPU训练和调参
from visualize import generate_all_visualizations, predict_custom_scenario  # 生成图表并预测自定义工况


def build_parser() -> argparse.ArgumentParser:
    """功能: 构建命令行参数解析器。
    参数: 无。
    返回: argparse.ArgumentParser对象。
    调用位置: main。
    """

    parser = argparse.ArgumentParser(description="四轴无人机飞行能耗预测模型实验1.0")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["download", "prepare", "train", "predict", "evaluate", "visualize", "custom", "all"],
        help="运行模式，all会依次完成数据处理、训练、评估和可视化。",
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
    parser.add_argument("--device", default=None, help="训练或预测设备，默认auto。")
    parser.add_argument("--allow-cpu", action="store_true", help="允许没有CUDA时退回CPU训练。")
    parser.add_argument("--force-prepare", action="store_true", help="强制重新生成处理后数据和切分文件。")
    parser.add_argument("--input-csv", type=Path, default=None, help="predict模式的输入CSV。")
    parser.add_argument("--output-csv", type=Path, default=None, help="predict模式的输出CSV。")

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


def main() -> None:
    """功能: 根据命令行模式执行机器学习和可视化流程。
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
        require_gpu=not args.allow_cpu,
    )
    ensure_directories(cfg)

    if args.mode == "download":
        download_source_dataset(cfg)
        extract_source_zip(cfg)
        print_dict("download", {"raw_flights_csv": str(cfg.raw_flights_csv), "raw_parameters_csv": str(cfg.raw_parameters_csv)})
    elif args.mode == "prepare":
        summary = prepare_dataset(cfg, force=args.force_prepare)
        print_dict("prepare", summary)
    elif args.mode == "train":
        summary = train_model(cfg)
        print_dict("train", summary)
    elif args.mode == "predict":
        output_path = predict_from_csv(cfg, args.input_csv, args.output_csv)
        print_dict("predict", {"prediction_file": str(output_path)})
    elif args.mode == "evaluate":
        metrics = evaluate_model(cfg)
        print_dict("evaluate", metrics)
    elif args.mode == "visualize":
        summary = generate_all_visualizations(cfg)
        print_dict("visualize", summary)
    elif args.mode == "custom":
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
        )
        print_dict("custom", summary)
    elif args.mode == "all":
        data_summary = prepare_dataset(cfg, force=args.force_prepare)
        train_summary = train_model(cfg)
        metrics = evaluate_model(cfg)
        visual_summary = generate_all_visualizations(cfg)
        print_dict("prepare", data_summary)
        print_dict("train", train_summary)
        print_dict("evaluate", metrics)
        print_dict("visualize", visual_summary)


if __name__ == "__main__":
    main()
