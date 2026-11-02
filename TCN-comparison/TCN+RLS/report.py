# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-10-30
# 文件名: report.py
# 功能说明: 生成TCN加RLS实验README和当前输出详细报告
# 版本号：1.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from config import GROUPS, ExperimentConfig, group_paths


def markdown_table(frame: pd.DataFrame) -> str:
    """功能: 将数据表转换为Markdown表格。
    参数: frame为待输出表格。
    返回: Markdown表格文本。
    调用位置: write_report。
    """
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float): values.append(f"{value:.4f}")
            else: values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(cfg: ExperimentConfig) -> dict:
    """功能: 生成TCN加RLS稳定说明和当前输出报告。
    参数: cfg为实验配置。
    返回: README和报告路径。
    调用位置: main.py。
    """
    metrics = pd.read_csv(cfg.summary_csv).sort_values("group")
    manifest = json.loads(cfg.manifest_json.read_text(encoding="utf-8"))
    checkpoints = []
    for group in metrics.group:
        checkpoint = torch.load(group_paths(cfg, group)["tcn_checkpoint"], map_location="cpu", weights_only=False)
        checkpoints.append({"组别": group, "隐藏激活": GROUPS[group]["activation"], "TCN步数": checkpoint["window_steps"],
                            "候选轮数": checkpoint["tune_epochs"], "正式轮数": checkpoint["epochs_run"],
                            "最佳轮次": checkpoint["best_epoch"], "参数量": checkpoint["parameter_count"],
                            "种子": checkpoint["seed"]})
    checkpoint_table = markdown_table(pd.DataFrame(checkpoints))
    interval_columns = ["group", "sample_power_interval_coverage_percent", "window_interval_coverage_percent",
                        "task_start_interval_coverage_percent", "post_feedback_interval_coverage_percent",
                        "post_feedback_interval_mean_width_wh", "post_feedback_interval_score_wh"]
    interval_frame = metrics[interval_columns].copy()
    interval_frame.columns = ["组别", "功率覆盖(%)", "RLS窗口覆盖(%)", "任务前覆盖(%)", "反馈后在线覆盖(%)", "反馈后宽度(Wh)", "反馈后区间评分(Wh)"]
    interval_table = markdown_table(interval_frame)
    point_frame = metrics[["group", "tcn_power_wape_percent", "power_wape_percent", "power_rmse_w",
                           "tcn_window_energy_wape_percent", "window_energy_wape_percent", "rls_window_energy_wape_percent"]].copy()
    point_frame.columns = ["组别", "TCN功率WAPE(%)", "RLS功率WAPE(%)", "RLS功率RMSE(W)", "TCN 1s能耗WAPE(%)", "RLS 1s能耗WAPE(%)", "RLS反馈窗能耗WAPE(%)"]
    point_table = markdown_table(point_frame)
    stages = pd.concat([pd.read_csv(group_paths(cfg, group)["stage_intervals"]) for group in metrics.group], ignore_index=True)
    stage_coverage = stages.pivot(index="group", columns="observed_fraction", values="coverage_percent")
    stage_coverage.columns = [f"进度{fraction:.0%}覆盖(%)" for fraction in stage_coverage.columns]
    stage_coverage.index.name = "组别"
    stage_coverage_table = markdown_table(stage_coverage.reset_index())
    table = metrics[["group", "name", "feature_count", "tcn_window_seconds", "rls_window_seconds",
                     "rls_forgetting_factor", "rls_initial_covariance", "tcn_flight_energy_wape_percent",
                     "flight_energy_wape_percent", "flight_interval_coverage_percent",
                     "flight_interval_mean_width_wh", "flight_interval_score_wh"]].copy()
    table.columns = ["组别", "输入与激活", "输入维度", "TCN窗口(s)", "RLS窗口(s)", "遗忘因子", "初始协方差",
                     "TCN能耗WAPE(%)", "RLS逐窗累计WAPE(%)", "任务前覆盖率(%)", "任务前宽度(Wh)", "任务前区间评分(Wh)"]
    result = markdown_table(table)
    gain = []
    for _, row in metrics.iterrows():
        gain.append(f"{row.group}组：RLS前 {row.tcn_flight_energy_wape_percent:.4f}% → RLS后 {row.flight_energy_wape_percent:.4f}%，"
                    f"任务前覆盖率 {row.flight_interval_coverage_percent:.2f}%、任务前平均宽度 {row.flight_interval_mean_width_wh:.4f} Wh；"
                    f"反馈后在线覆盖率 {row.post_feedback_interval_coverage_percent:.2f}%。")
    best = metrics.loc[metrics.flight_energy_wape_percent.idxmin()]
    best_interval = metrics.loc[metrics.post_feedback_interval_score_wh.idxmin()]
    dynamic_note = (f"当前完成组为 {', '.join(metrics.group.astype(str))}。"
                    f"逐窗累计任务能耗WAPE最低为{best.group}组({best.flight_energy_wape_percent:.4f}%)，"
                    f"反馈后在线区间评分最低为{best_interval.group}组({best_interval.post_feedback_interval_score_wh:.4f}Wh)。"
                    "反馈后的改善使用了任务执行期间取得的实际窗口能耗，不能全部解释为任务开始前预测能力改善。"
                    "区间覆盖率为当前划分上的经验覆盖率，不等同于理论保证；同一flight内部存在相关轨迹点，"
                    "阶段表每flight每阶段只取一个点。RLS测试结果未用于反向选择超参。")
    lookup = metrics.set_index("group")
    activation_comparison = ""
    if {"D", "E"}.issubset(lookup.index):
        d, e = lookup.loc["D"], lookup.loc["E"]
        activation_comparison = (
            f"E与D使用完全相同的15维输入和10步SMA；TCN窗口分别为{e.tcn_window_seconds:g}s与{d.tcn_window_seconds:g}s，"
            f"RLS遗忘因子分别为{e.rls_forgetting_factor:g}与{d.rls_forgetting_factor:g}。"
            f"E的RLS功率WAPE为{e.power_wape_percent:.4f}%（D为{d.power_wape_percent:.4f}%），"
            f"逐窗累计任务能耗WAPE为{e.flight_energy_wape_percent:.4f}%（D为{d.flight_energy_wape_percent:.4f}%）。"
            f"E反馈后在线覆盖率{e.post_feedback_interval_coverage_percent:.2f}%，"
            f"平均宽度{e.post_feedback_interval_mean_width_wh:.4f}Wh，区间评分{e.post_feedback_interval_score_wh:.4f}Wh；"
            f"D对应为{d.post_feedback_interval_coverage_percent:.2f}%、{d.post_feedback_interval_mean_width_wh:.4f}Wh、"
            f"{d.post_feedback_interval_score_wh:.4f}Wh。"
            "本次E点预测误差更高，但区间更窄且综合评分更低；不能仅凭一个指标认定统一激活整体更优。"
            "各组独立选择TCN窗口和RLS参数，因此差异包含选模结果；当前只有种子42单次结果。"
            f"E任务前覆盖{e.task_start_interval_coverage_percent:.2f}%，反馈后覆盖略低于名义95%，"
            f"RLS窗口覆盖{e.window_interval_coverage_percent:.2f}%，功率采样覆盖{e.sample_power_interval_coverage_percent:.2f}%；"
            "在线总能耗区间接近95%不能替代其他层级或各阶段的覆盖率检查。")
    readme = f'''# TCN+RLS 五组离线功率、能耗与区间评估

## 目录

- [实验范围](#实验范围)
- [五组设置](#五组设置)
- [运行命令](#运行命令)
- [结果摘要](#结果摘要)
- [任务阶段覆盖率](#任务阶段覆盖率)
- [图表与核验记录](#图表与核验记录)
- [文件结构](#文件结构)
- [边界与解释](#边界与解释)

## 实验范围

本目录在 `../TCN` 已完成的 A/B/C/D/E 五组 TCN 功率基线上加入因果 RLS。RLS 只在一个完整反馈时间窗结束、取得该窗实际能耗后更新；当前时间窗使用更新前状态预测。RLS 搜索同时包含遗忘因子、初始协方差和时间窗长度。实验不重新训练 TCN，便于把新增差异限定为 RLS 和区间评估。

置信区间采用与在线更新相同的因果模拟顺序。验证 flight 按排序后的序列索引交替拆分：`[::2]` 的15个 flight 用于 RLS 超参搜索，`[1::2]` 的14个 flight 用于区间倍率校准；测试集只用于最终评价。名义置信水平为 95%，同时报告任务前、窗口级和反馈后在线覆盖率、平均区间宽度、区间评分以及按任务阶段抽取的覆盖率。

## 五组设置

| 组别 | 输入 | 激活 | 原始TCN来源 |
| --- | --- | --- | --- |
| A | 原始10维 | 混合 SiLU/ReLU | `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_A_best.pt` |
| B | 原始10维 | 统一 SiLU | `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_B_best.pt` |
| C | 原始10维+5维10步SMA，共15维 | 混合 SiLU/ReLU | `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_C_best.pt` |
| D | 原始10维+5维10步SMA，共15维 | 统一 SiLU | `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_D_best.pt` |
| E | 原始10维+5维10步SMA，共15维 | 统一 LeakyReLU(0.05) | `D:/Python-files/Energy-prediction/model/TCN-planned/TCN-comparison_E_best.pt` |

SMA 通道为三维规划位置和风速、风向，按 flight 分组、只使用当前与历史值计算。原始10维字段为 `position_x_m, position_y_m, position_z_m, wind_speed, wind_angle, time_s, task_duration_s, payload_g, uav_type, planned`；其中 `uav_type` 为机型编号，`planned` 为停机/原地启动/飞行三态。TCN 的两个残差块通道均为32，卷积核3，膨胀率1和2，GroupNorm(1,32)，Dropout=0.05；A-D预测头为 Linear(32,32)、SiLU、Linear(32,1)。

训练采用SmoothL1Loss、AdamW、学习率3e-4、权重衰减1e-4、批大小2048、梯度裁剪2.0。采样间隔0.2s，SMA固定10步（2s）；TCN窗口候选为1、2、5、10s，每候选8轮，正式80轮，种子42。权重保留验证损失最低轮次，标准化器仅由训练集拟合。本轮读取的权重元数据为：

{checkpoint_table}

RLS 搜索网格：遗忘因子 `{list(cfg.rls_forgetting_factors)}`，初始协方差 `{list(cfg.rls_initial_covariances)}`，时间窗 `{list(cfg.rls_window_seconds)}` s，共 {len(cfg.rls_forgetting_factors)*len(cfg.rls_initial_covariances)*len(cfg.rls_window_seconds)} 个候选/组。每个flight重置RLS，初始偏置0、比例1，协方差对角元为初始系数乘以1e-4和初始系数；残差标准差初值0.01Wh，方差更新权重0.15。比例裁剪到0.25～1.75，偏置幅值不超过当前实际窗口能耗。所有候选完整模拟，按验证搜索子集逐窗累计任务能耗WAPE最小选取。

## 运行命令

```powershell
cd I:/STUDY/python/project/Energy-prediction/TCN-comparison/TCN+RLS
python main.py all --device cuda
```

分阶段运行：`python main.py prepare`、`python main.py tune`、`python main.py calibrate`、`python main.py evaluate`、`python main.py plot`、`python main.py report`。没有 CUDA 时可显式使用 `--device cpu`。

仅复现E组使用 `python main.py all --group E --device cuda`，需先完成 `../TCN` 中E组训练。本次沿用A-D已有产物，只新增E的245候选搜索、校准和评价；汇总与图表包含A-E。搜索进度和异常追加到 `out/logs/TCN_RLS_terminal.log`，逐候选数据为 `out/results/TCN_RLS_E_rls_tuning.csv`；每组完整命令另保存运行环境快照，E为 `out/TCN_RLS_runtime_E.json`。

## 结果摘要

本报告由 [汇总CSV](out/results/TCN_RLS_all_metrics.csv) 自动生成。数据规模为训练 {manifest['train_rows']} 行/{manifest['train_flights']} 个 flight，验证 {manifest['val_rows']} 行/{manifest['val_flights']} 个 flight，测试 {manifest['test_rows']} 行/{manifest['test_flights']} 个 flight。验证flight按编号排序后，以从0开始的序列索引划分为15个搜索flight和14个校准flight；不是按flight编号奇偶划分。

{result}

表中RLS逐窗累计WAPE将每个窗口在反馈前的预测相加。任务前区间则在尚无实际反馈时生成，中心等于TCN总能耗；它不是以RLS逐窗累计预测为中心的区间。以下将不同层级的覆盖率分别列出，反馈后在线区间指“已观测实际消耗+剩余预测”的最终总能耗区间，排除已完成整个任务的零宽度点：

{interval_table}

功率、固定1秒能耗与RLS反馈时间窗的点预测指标分别为：

{point_table}

逐组结果：

{chr(10).join('- ' + item for item in gain)}

{dynamic_note}

{activation_comparison}

## 任务阶段覆盖率

每个flight在任务开始和首次达到25%、50%、75%已观察进度时各取一个尚未完成任务的预测，当前每组每阶段均为47个flight。阶段样本按flight等权；反馈后在线汇总覆盖率则包含同一flight的多个相关时点，两者权重不同。

{stage_coverage_table}

E在25%阶段覆盖率为91.49%，50%为93.62%，75%为100.00%。反馈后在线汇总94.64%不能解释为所有阶段均达到名义95%。区间平均宽度和区间评分与覆盖率同时评价，完整逐阶段数据见当前输出报告。

## 图表与核验记录

图表、各阶段覆盖率、完整公式和逐文件说明见 [当前输出报告](out/outreadme.md)。

图表共7套，均保存PNG和SVG：任务能耗WAPE、分层区间覆盖率、任务前覆盖率与宽度、RLS时间窗搜索、代表任务功率、五组在线区间分面和任务阶段覆盖率。在线分面采用2×3布局，显示A-E并隐藏空白格；代表flight按测试任务长度的中位位置选择。

复核日志：[4项行为测试](out/logs/E_validation_tests.log)、[数据与模型产物核验](out/logs/E_artifact_audit.json)。模型复制SHA-256一致，TCN冻结基线的汇总指标一致；少量逐采样CSV有约0.000031W的float32写出精度差异。名义95%区间采用相同校准规则、各组独立倍率，覆盖率按CSV严格边界复算时使用 `float_precision="round_trip"`。

![任务能耗误差](out/figures/TCN_RLS_flight_energy_wape.svg)

![阶段区间覆盖率](out/figures/TCN_RLS_stage_interval_coverage.svg)

## 文件结构

- `config.py`：数据、模型、输出路径和RLS搜索网格。
- `rls.py`：RLS更新、区间和覆盖率计算。
- `rls_evaluate.py`：TCN基线加载、RLS调参、区间校准与测试评价。
- `visualize_rls.py`：误差、覆盖率、宽度、窗口搜索和在线轨迹图。
- `main.py`：统一入口。
- `out/results/`：五组调参、校准、逐窗口、逐flight和汇总CSV/JSON。
- `out/figures/`：PNG和SVG图表，包括误差、覆盖率、时间窗搜索、在线区间和阶段覆盖率。
- `out/logs/`：终端及运行记录。

数据副本位于 `{cfg.data_dir.as_posix()}`。TCN权重、标准化器及每组RLS配方 `TCN_RLS_<组别>_rls.json` 保存于 `{cfg.model_dir.as_posix()}`；配方包含RLS超参、初始化和冻结校准倍率，在线状态按flight重置。校准审计JSON、逐阶段CSV和其余结果保存在 `out/results/`。项目依赖见 [requirements.txt](requirements.txt)，执行 `python -m unittest discover -s tests -v` 可检查反馈因果性。

[README.md](README.md)说明入口和参数，[improve.md](improve.md)追加工程变化，[out/outreadme.md](out/outreadme.md)随结果同步；数据和模型因体积与既有仓库存储约定保留在D盘，`out/`沿用Git忽略策略，完整运行可由 `python main.py all --device cuda` 重建。仅绘图使用 `python main.py plot`，仅同步报告使用 `python main.py report`。首次prepare会复制缺失数据；`--force-prepare`显式重建副本。支持 `--group A`、`--data-dir`、`--save-dir`、`--out-dir`；部分组评价保留其他已有组结果，改变配置时应使用独立输出目录并重新调参、校准。

## 边界与解释

RLS 使用的是完整窗口实际能耗反馈，属于离线回放的在线校正模拟。TCN全部预推理仅使用加工规划特征，未来实际功率只在对应窗口结束后作为反馈；原始规划位置、风和运行状态仍来自实飞代理，不能据此判断独立计划轨迹和预报风场下的部署精度。区间是面向未来观测值的预测区间，不是模型均值参数的置信区间。校准采用同一套残差比率分位数规则，各组倍率不同；任务倍率混合任务起点和多个在线阶段，未单独保证起点或每个阶段95%覆盖。校准flight也曾用于已有TCN的验证选择，因此不是完全独立于整个建模过程的校准集。同一flight内部有时间相关性，无理论覆盖保证或多种子显著性结论。
'''
    (cfg.project_root / "README.md").write_text(readme, encoding="utf-8")
    lines = ["# TCN+RLS 当前输出报告", "", "## 目录", "",
             "- [A. 算法、数据和评价口径](#a-算法数据和评价口径)",
             "- [B. 当前结果](#b-当前结果)", "- [C. 图表与文件说明](#c-图表与文件说明)",
             "- [结果文件](#结果文件)", "- [日志与运行配置](#日志与运行配置)", "",
             "## A. 算法、数据和评价口径", "",
             "### A.1 实验边界与数据流", "",
             "本报告对应 `TCN+RLS` 当前输出。五组 TCN 权重来自 `../TCN`，本目录不重新训练 TCN；新增计算只包括 RLS 超参搜索、因果反馈模拟、区间校准和测试评价。加工表由 D 盘 `TCN-planned` 数据副本生成，训练/验证/测试分别为 "
             f"{manifest['train_rows']} 行/{manifest['train_flights']} 个 flight、{manifest['val_rows']} 行/{manifest['val_flights']} 个 flight、{manifest['test_rows']} 行/{manifest['test_flights']} 个 flight。三部分的 flight 集合没有交叉，数据指纹保存在 `{cfg.manifest_json.as_posix()}`。", "",
             "每个样本的采样间隔为 0.2 s。A/B 输入为原始 10 维字段 `position_x_m, position_y_m, position_z_m, wind_speed, wind_angle, time_s, task_duration_s, payload_g, uav_type, planned`；C/D/E 在原始字段后追加位置、风速和风向的 5 个 10 步因果 SMA 通道，共 15 维。SMA 在单个 flight 内按时间升序计算，只使用当前和历史样本。TCN 输出当前采样点功率，功率乘以 `dt_seconds/3600` 得到采样能耗，再按 RLS 反馈窗口聚合。数据流为：加工表 → 特征构造与训练集标准化 → TCN 功率基线 → 窗口能耗 → RLS 校正 → 任务级/窗口级区间 → 测试指标。", "",
             "### A.2 TCN 基线结构", "",
             "每组沿用冻结 TCN：两个通道数为 32 的残差块，每块两次因果卷积，卷积核为 3，膨胀率为 1 和 2，Dropout=0.05；预测头为 `Linear(32,32)-对应隐藏激活-Linear(32,1)`。训练使用 SmoothL1Loss、AdamW、学习率 `3e-4`、权重衰减 `1e-4`、批大小 2048、梯度范数裁剪2.0、随机种子 42；4个窗口候选（1、2、5、10s）各训练 8 轮，正式模型训练 80 轮，保存验证损失最低轮次。RLS 阶段只加载冻结权重和训练集标准化器，不更新神经网络梯度。GroupNorm使用整个历史输入窗内的统计量，当前逐目标历史窗口推理不能替换成整条flight一次前向或简单卷积缓存。窗口起点用该flight首个输入重复补齐。", "", checkpoint_table, "",
             "### A.3 RLS 回归和因果更新", "",
             "第 $k$ 个完整反馈窗口的 TCN 能耗为 $x_k$，实际能耗为 $y_k$，RLS 用偏置和比例两个参数进行校正：", "",
             "$$\\hat y_k=\\boldsymbol{\\phi}_k^{\\mathsf T}\\boldsymbol{\\theta}_k,\\qquad \\boldsymbol{\\phi}_k=[1,x_k]^{\\mathsf T}$$", "",
             "其中：", "", "$\\hat y_k$：窗口能耗的 RLS 预测值。", "", "$x_k$：TCN 基线窗口能耗，单位为 Wh。", "", "$y_k$：窗口实际能耗，单位为 Wh。", "", "$\\boldsymbol{\\phi}_k$：回归向量。", "", "$\\boldsymbol{\\theta}_k$：偏置和比例参数向量。", "",
             "窗口实际反馈到达后才执行一次更新：", "",
             "$$e_k=y_k-\\boldsymbol{\\phi}_k^{\\mathsf T}\\boldsymbol{\\theta}_k,\\quad \\mathbf K_k=\\frac{\\mathbf P_k\\boldsymbol{\\phi}_k}{\\lambda+\\boldsymbol{\\phi}_k^{\\mathsf T}\\mathbf P_k\\boldsymbol{\\phi}_k}$$", "",
             "$$\\boldsymbol{\\theta}_{k+1}=\\boldsymbol{\\theta}_k+\\mathbf K_ke_k,\\qquad \\mathbf P_{k+1}=\\frac{\\mathbf P_k-\\mathbf K_k\\boldsymbol{\\phi}_k^{\\mathsf T}\\mathbf P_k}{\\lambda}$$", "",
             "其中：", "", "$e_k$：窗口残差。", "", "$\\mathbf K_k$：RLS 增益向量。", "", "$\\mathbf P_k$：参数协方差矩阵。", "", "$\\lambda$：遗忘因子。", "",
             "每个flight初始化参数为 `[0,1]`，初始协方差为 `diag(初始系数×1e-4,初始系数)`。更新后比例裁剪到0.25～1.75，偏置幅值裁剪到当前实际窗口能耗与0.001Wh中的较大值；协方差取对称矩阵，输出能耗裁剪为非负。截断窗口的偏置按实际样本数占完整窗口样本数的比例缩放。残差均值和方差以0.85旧值+0.15新信息更新，方差初值为0.01Wh的平方。", "",
             "实现中的顺序是：先用更新前状态预测当前窗口；窗口完整结束并取得该窗口实际能耗后再更新；更新后的状态只用于后续窗口和剩余能耗预测。最后一个不足完整窗口的片段只预测、不反馈更新。TCN未来规划能耗预先计算，但未来实际功率不会用于更新前预测；真实累计消耗只在已观察部分加入任务总能耗。", "",
             "### A.4 区间构造、校准和覆盖率", "",
             "RLS 根据参数协方差和残差方差估计剩余能耗区间。对剩余窗口回归向量 $\\boldsymbol{\\phi}$，先计算未校准半径：", "",
             "$$v=\\max(\\hat\\sigma^2,(0.25|\\bar e|)^2,10^{-8}),\\qquad r=z_{1-\\alpha/2}\\sqrt{v\\left[\\max(0,\\boldsymbol{\\phi}^{\\mathsf T}\\mathbf P\\boldsymbol{\\phi})+\\max(h,1)\\right]}$$", "",
             "$$[L,U]=[\\max(0,\\hat E-sr),\\hat E+sr]$$", "",
             "其中：", "", "$r$：未校准区间半径，单位为 Wh。", "", "$z_{1-\\alpha/2}$：名义95%双侧预测区间使用的正态近似分位点1.96。", "", "$\\alpha$：名义未覆盖概率0.05。", "", "$\\hat\\sigma^2$：RLS 残差方差的指数滑动估计，单位为Wh平方。", "", "$\\bar e$：窗口残差的指数滑动均值，单位为Wh。", "", "$v$：加上下限与偏差项后的创新方差。", "", "$h$：剩余等效窗口数。", "", "$\\boldsymbol{\\phi}$：剩余能耗回归向量，其两项为剩余等效窗口数和TCN剩余能耗。", "", "$\\mathbf P$：当前RLS参数协方差。", "", "$\\hat E$：RLS 预测的剩余总能耗，按相同回归向量计算并裁剪为非负。", "", "$s$：验证集残差比率分位数得到的区间倍率。", "", "$L,U$：剩余能耗区间下界和上界，单位为Wh；加上已观测消耗得到最终总能耗区间。", "",
             "名义置信水平为95%。29个验证flight按编号排序，以从0开始的序列索引划分：`[::2]`的15个flight用于245个RLS候选搜索，`[1::2]`的14个flight用于倍率校准；不是按flight编号本身奇偶划分。任务倍率由校准flight中任务起点及所有剩余窗口数大于0的反馈时点计算，窗口倍率仅由完整窗口计算；两者均为绝对残差除以未校准半径后的95%经验分位数，并取不小于1。测试集冻结参数和倍率后计算覆盖率，当前策略未进行有限样本分位数修正或按flight独立保形校准。校准flight也参与过冻结TCN的验证选模，不能宣称整个建模过程的严格独立校准覆盖保证。", "",
             "任务前区间中心为TCN全任务基线；反馈后在线区间中心为已观察实际消耗加RLS剩余预测；逐窗累计能耗则累计每个窗口更新前的预测。三者口径不同。旧 `flight_interval_*` 字段为任务前区间，新增 `task_start_interval_*` 为同值显式别名；`remaining_trace_interval_*`含初始点，`post_feedback_interval_*`只取有反馈且任务尚未结束的点。完成任务后实际总量已知形成零宽度点，全部覆盖评价排除这些点。功率采样区间由窗口区间按TCN预测能耗占比分配，未独立按功率残差校准。", "",
             "覆盖率为实际值位于上下界之间的比例；宽度为上界减下界的均值。区间评分等于宽度加上超出上下界的误差乘以40（95%水平），数值越低越好。在宽度和覆盖率之间存在取舍，不能仅按覆盖率选组。相关在线轨迹点汇总使长flight权重更大；补充每flight在任务开始及首次达到25%、50%、75%已观察进度时各取一点的阶段表。阶段未达到或任务已完成则不取，当前各阶段样本数由CSV给出。精确复核CSV覆盖率时使用 `pd.read_csv(path, float_precision=\"round_trip\")`，因为少量区间边界极接近实际值，默认解析误差会改变严格边界比较。", "",
             "### A.5 搜索、停止条件和指标", "",
             "RLS 网格为7个遗忘因子×7个初始协方差×5个反馈时间窗，共245个候选/组。候选全部运行到结束，不使用提前停止；以验证搜索子集逐窗累计总能耗WAPE最小者为最优。测试阶段不再搜索参数。功率、固定1秒窗口能耗、RLS反馈窗口能耗和Flight能耗WAPE均为绝对误差和除以实际绝对值和乘100%；固定1秒指标的尾窗也纳入。`window_energy_*`仍指固定1秒指标，`rls_window_energy_wape_percent`指实际RLS反馈窗指标，避免把两种窗长度混淆。", "",
             "实现链接：[config.py](../config.py) 管理路径与网格；[rls.py](../rls.py) 实现RLS、区间和指标；[rls_evaluate.py](../rls_evaluate.py) 实现基线加载、调参、校准和测试；[visualize_rls.py](../visualize_rls.py) 生成图表；[main.py](../main.py) 调度流程。", "",
             "## B. 当前结果", "", result, "", "### 功率与窗口点预测", "", point_table, "",
             "### 分层区间覆盖率", "", interval_table, "",
             "### 任务阶段覆盖率", "", "每flight每阶段仅取首次达到指定已观察进度且任务未完成的一个预测，样本等权。", ""]
    stage_table = stages.rename(columns={"group":"组别", "observed_fraction":"已观察比例", "flights":"flight数", "coverage_percent":"覆盖率(%)", "mean_width_wh":"平均宽度(Wh)", "interval_score_wh":"区间评分(Wh)", "energy_wape_percent":"总能耗WAPE(%)"})
    lines += [markdown_table(stage_table), "", "### 数值解释", ""]
    lines.extend([f"- {item}" for item in gain])
    lines += ["", activation_comparison, "", "### 校准倍率和搜索审计", ""]
    calibration_rows = []
    for group in metrics.group:
        calibration = json.loads(group_paths(cfg, group)["interval_calibration"].read_text(encoding="utf-8"))
        tuning = pd.read_csv(group_paths(cfg, group)["rls_tuning"])
        calibration_rows.append({"组别":group, "搜索候选":len(tuning), "搜索flight":int(tuning.flights.iloc[0]),
                                 "校准flight":calibration["calibration_flights"],
                                 "任务倍率":calibration["task_interval_scale"], "窗口倍率":calibration["window_interval_scale"],
                                 "任务校准点":calibration["task_calibration_points"], "窗口校准点":calibration["window_calibration_points"]})
    lines += [markdown_table(pd.DataFrame(calibration_rows)), "",
              dynamic_note + "当前没有多种子重复、独立新校准集或部署输入验证。", ""]
    if cfg.runtime_json.exists():
        runtime = json.loads(cfg.runtime_json.read_text(encoding="utf-8"))
        lines += [f"保存的早期完整运行环境：Python `{runtime.get('python', '').splitlines()[0]}`；PyTorch `{runtime.get('torch')}`；CUDA `{runtime.get('cuda')}`；GPU `{runtime.get('gpu')}`；启动时间 `{runtime.get('started_at')}`。分阶段重画或报告刷新不覆盖完整运行环境快照，本次E组独立环境快照见 `TCN_RLS_runtime_E.json`，启动于2026-11-01；后续记录见追加终端日志。", ""]
    lines += ["", "## C. 图表与文件说明", "", "图表由 `visualize_rls.py` 从当前 CSV/JSON 重建，PNG 和 SVG 均保留。", ""]
    figure_notes = {
        "flight_energy_wape": "源文件为五组 `*_metrics.json`；横轴为 A/B/C/D/E，纵轴为 Flight 总能耗 WAPE(%)，比较冻结 TCN 基线和 RLS 后结果。当前各组 RLS 后误差与基线的差异见汇总表。",
        "interval_coverage": "源文件为 `TCN_RLS_all_metrics.csv`；横轴A/B/C/D/E，纵轴为功率采样、RLS窗口、任务前总能耗和在线轨迹(含初始点)覆盖率(%)，虚线为名义95%。仅反馈后覆盖率另见分层表，不能把单一层级结果外推到其他层级。",
        "interval_coverage_width": "源文件为 `TCN_RLS_all_metrics.csv`；横轴为任务前区间平均宽度(Wh)，纵轴为任务前覆盖率(%)。D为97.87%/2.8021Wh，E为93.62%/3.0357Wh；该图属于任务前口径，反馈后宽度分别为4.2100/3.2980Wh，二者不可混用。",
        "rls_window_search": "源文件为五组 `*_rls_tuning.csv`；横轴为 RLS 反馈时间窗(s)，纵轴为该时间窗下最优验证 Flight 能耗 WAPE(%)，用于检查时间窗搜索是否完整。",
        "representative_power": "源文件为五组 `*_test_predictions.csv`；横轴为任务时间(s)，纵轴为功率(W)，代表 flight 按测试集任务长度中位位置选取，不按误差挑选。",
        "remaining_energy_trace": "源文件为五组 `*_test_trace.csv`；同一中位长度代表flight的五组分面图，横轴为已完成反馈窗口数，纵轴为最终总能耗预测(Wh)，曲线为预测，色带为名义95%区间，黑色虚线为实际最终能耗。只显示尚有剩余任务的点，不能用单个代表任务替代全部flight覆盖统计。",
        "stage_interval_coverage": "源文件为五组 `*_stage_intervals.csv`；横轴为已观察任务进度0%、25%、50%、75%，纵轴为最终总能耗区间覆盖率(%)，每flight每阶段只取首次达到阈值的一点。名义95%为水平虚线，各组独立校准，阶段间表现不构成统计独立重复。"
    }
    for name, title in [("flight_energy_wape", "TCN与RLS Flight能耗WAPE"), ("interval_coverage", "置信区间覆盖率"),
                        ("interval_coverage_width", "覆盖率与区间宽度"), ("rls_window_search", "RLS时间窗搜索"),
                        ("representative_power", "代表任务功率"), ("remaining_energy_trace", "剩余能耗在线轨迹"),
                        ("stage_interval_coverage", "任务阶段区间覆盖率")]:
        png = Path("figures") / f"TCN_RLS_{name}.png"
        svg = Path("figures") / f"TCN_RLS_{name}.svg"
        if (cfg.figure_dir / png.name).exists():
            lines += [f"### {title}", "", figure_notes[name], "", f"![{title}]({png.as_posix()})", "", f"[SVG版本]({svg.as_posix()})", ""]
    lines += ["## 结果文件", "", "| 文件 | 说明 |", "| --- | --- |"]
    descriptions = {"_flight_metrics.csv":"每flight实际、TCN基线、逐窗累计能耗和任务前区间，Wh",
                    "_interval_calibration.json":"验证校准方法、冻结倍率和校准样本数",
                    "_metrics.json":"该组全部点预测、覆盖率、宽度和评分指标",
                    "_rls_tuning.csv":"245个RLS候选、验证搜索分数和样本量",
                    "_test_predictions.csv":"逐采样功率(W)、区间分配和采样能耗(Wh)",
                    "_test_trace.csv":"测试任务初始及反馈后预测、区间和RLS状态",
                    "_validation_trace.csv":"校准子集的未校准任务轨迹（倍率估计来源）",
                    "_window_metrics.csv":"RLS反馈窗实际与预测能耗、区间和完整窗标志，Wh",
                    "_stage_intervals.csv":"每flight等权的任务阶段覆盖率、宽度、评分和WAPE",
                    "_all_metrics.csv":"五组完整指标汇总，README及对比图数据源"}
    for path in sorted(cfg.result_dir.glob("*")):
        description = next((value for suffix, value in descriptions.items() if path.name.endswith(suffix)), "实验结果")
        lines.append(f"| [{path.name}](results/{path.name}) | {description} |")
    lines += ["", "## 日志与运行配置", "", "| 文件 | 说明 |", "| --- | --- |"]
    for path in sorted(cfg.log_dir.glob("*")):
        lines.append(f"| [{path.name}](logs/{path.name}) | RLS搜索或终端日志 |")
    if cfg.runtime_json.exists():
        lines.append(f"| [{cfg.runtime_json.name}]({cfg.runtime_json.name}) | Python、CUDA、GPU和配置快照 |")
    for path in sorted(cfg.out_dir.glob("TCN_RLS_runtime_*.json")):
        lines.append(f"| [{path.name}]({path.name}) | 单组完整运行环境快照 |")
    cfg.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"readme": str(cfg.project_root / "README.md"), "report": str(cfg.report_md)}
