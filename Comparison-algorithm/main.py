# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-09-09
# 文件名: main.py
# 功能说明: 调度对比算法实验、重建统一指标汇总并生成实验文档
# 版本号：3.0

import argparse
import json

import pandas as pd

from comparison_engine import main as run_experiments  # 运行指定或全部对比算法
from config import ALGORITHMS  # 算法配置及统一汇总顺序
from config import OUT_DIR  # 对比实验根输出目录
from config import ExperimentConfig  # 单算法输出路径配置


def rebuild_comparison_metrics() -> list[str]:
    """功能: 从各算法指标文件重建根目录统一指标汇总。
    参数: 无。
    返回: 尚未生成指标文件的算法标识列表。
    调用位置: main函数完成算法实验后。
    """

    metric_frames: list[pd.DataFrame] = []
    missing_algorithms: list[str] = []
    for algorithm in ALGORITHMS:
        metrics_path = ExperimentConfig(algorithm).metrics_csv
        if not metrics_path.is_file():
            missing_algorithms.append(algorithm)
            continue

        frame = pd.read_csv(metrics_path, encoding="utf-8")
        if len(frame) != 1:
            raise ValueError(f"算法 {algorithm} 的指标文件应仅包含一行: {metrics_path}")
        if "algorithm" not in frame.columns or str(frame.iloc[0]["algorithm"]) != algorithm:
            raise ValueError(f"算法 {algorithm} 的指标标识与目录不一致: {metrics_path}")
        # 名称和来源属于配置元数据；重建汇总时同步，避免旧运行名称继续出现在README。
        frame.loc[frame.index[0], "algorithm_name"] = ALGORITHMS[algorithm]["name"]
        frame.loc[frame.index[0], "reference"] = ALGORITHMS[algorithm]["source"]
        frame.to_csv(metrics_path, index=False, encoding="utf-8")
        ExperimentConfig(algorithm).metrics_json.write_text(
            json.dumps(frame.iloc[0].to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metric_frames.append(frame)

    if not metric_frames:
        raise FileNotFoundError("未找到任何算法的 metrics.csv，无法重建总体指标。")

    comparison_metrics = pd.concat(metric_frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_metrics.to_csv(OUT_DIR / "comparison_metrics.csv", index=False, encoding="utf-8")
    records = comparison_metrics.to_dict(orient="records")
    (OUT_DIR / "comparison_metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return missing_algorithms


def generate_readmes() -> None:
    """功能: 调用统一文档生成器更新总README和算法README。
    参数: 无。
    返回: 无。
    调用位置: main函数完成全部算法实验及指标校验后。
    """

    from generate_documentation import main as generate_documentation  # 延迟加载文档生成入口

    generate_documentation()


def main() -> None:
    """功能: 解析命令行、运行算法并维护总体实验产物。
    参数: 无，参数来自命令行。
    返回: 无。
    调用位置: Python脚本入口。
    """
    parser = argparse.ArgumentParser(description="3.0固定测试集对比算法实验")
    parser.add_argument("--algorithm", nargs="*", choices=list(ALGORITHMS), help="只运行指定算法，默认运行全部算法")
    args = parser.parse_args()
    selected_algorithms = args.algorithm or None
    if selected_algorithms is not None:
        try:
            run_experiments(selected_algorithms)
        finally:
            # 即使单算法总览绘图失败，也按全部已有算法结果恢复根目录汇总。
            missing_algorithms = rebuild_comparison_metrics()
        if missing_algorithms:
            print(f"总体指标已按现有结果重建；尚缺少算法: {'、'.join(missing_algorithms)}")
        return

    run_experiments()
    missing_algorithms = rebuild_comparison_metrics()
    if missing_algorithms:
        missing_text = "、".join(missing_algorithms)
        raise FileNotFoundError(f"全量实验结束后仍缺少算法指标: {missing_text}")
    # 全量实验通过结果完整性检查后，再更新论文式项目文档。
    generate_readmes()


if __name__ == "__main__":
    main()
