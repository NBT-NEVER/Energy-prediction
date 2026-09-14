# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: plot_rls_window_analysis.py
# 开发时间: 2026-09-14
# 文件名: plot_rls_window_analysis.py
# 功能说明: 绘制RLS能量窗变量对功率、能量误差和窗口数量的影响
# 版本号：3.2

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "out" / "model" / "rls_energy_window_comparison_3.2.csv"
OUT = ROOT / "out" / "figures" / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    """功能: 读取变量分析汇总并保存三张对比图。
    参数: 无。
    返回: None。
    调用位置: 命令行直接运行。
    """
    frame = pd.read_csv(CSV)
    test = frame[frame["stage"].eq("test")].sort_values("energy_window_seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    x = test["energy_window_seconds"]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x, test["sample_power_w_wape_percent"], "o-", label="功率 WAPE")
    axis.plot(x, test["rls_window_energy_wh_wape_percent"], "o-", label="窗口能量 WAPE")
    axis.plot(x, test["rls_flight_energy_wh_wape_percent"], "o-", label="flight 能量 WAPE")
    axis.set_xlabel("RLS 能量反馈窗 (s)")
    axis.set_ylabel("WAPE (%)")
    axis.set_title("RLS 能量窗对误差的影响")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rls_energy_window_error_comparison.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, test["rls_window_energy_wh_mae"], "o-", label="MAE")
    axes[0].plot(x, test["rls_window_energy_wh_rmse"], "o-", label="RMSE")
    axes[0].set_xlabel("RLS 能量反馈窗 (s)")
    axes[0].set_ylabel("窗口能量误差 (Wh)")
    axes[0].set_title("窗口级 MAE 与 RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, test["rls_window_energy_wh_r2"], "o-", label="窗口 R²")
    axes[1].plot(x, test["rls_flight_energy_wh_r2"], "o-", label="flight R²")
    axes[1].set_xlabel("RLS 能量反馈窗 (s)")
    axes[1].set_ylabel("R²")
    axes[1].set_title("能量决定系数")
    axes[1].set_ylim(0.99, 1.001)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "rls_energy_window_energy_metrics.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x, test["complete_window_count"], "o-", label="完整窗口数")
    axis.plot(x, test["window_count"], "o--", label="总窗口数")
    axis.set_xlabel("RLS 能量反馈窗 (s)")
    axis.set_ylabel("窗口数量")
    axis.set_title("RLS 能量窗对反馈更新密度的影响")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rls_energy_window_count_comparison.svg")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
