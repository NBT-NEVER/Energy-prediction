# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: main.py
# 开发时间: 2026-10-19
# 文件名: main.py
# 功能说明: 调度实验5.0无人车真实行程数据准备、训练、评估与展示
# 版本号：5.0

from __future__ import annotations

import argparse
import json

import config
from vehicle_pipeline import evaluate_and_plot, predict_test, prepare_data, search_rls, search_tcn, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="实验5.0：GECR真实道路BEV双层能耗预测")
    parser.add_argument("mode", nargs="?", default="all", choices=("prepare", "search", "train", "predict", "evaluate", "all"))
    parser.add_argument("--force-prepare", action="store_true", help="重新清洗并切分Excel行程。")
    args = parser.parse_args()
    config.ensure_directories()
    results = {}
    if args.mode in ("prepare", "all"):
        results["prepare"] = prepare_data(force=args.force_prepare)
    if args.mode in ("search", "all"):
        results["tcn_search"] = search_tcn()
        results["rls_search"] = search_rls()
    if args.mode in ("train", "all"):
        results["train"] = train_model()
    if args.mode == "predict":
        results["predict"] = predict_test()
    if args.mode in ("evaluate", "all"):
        results["evaluate"] = evaluate_and_plot()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
