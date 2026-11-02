# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: report.py
# 开发时间: 2026-11-02
# 文件名: report.py
# 功能说明: 汇编AMOVFLY字段、统计结果、异常发现及图表说明
# 版本号：1.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import AnalysisConfig


def _md_table(frame: pd.DataFrame, limit: int = 30) -> str:
    """功能: 将DataFrame转换为可读的Markdown表格。
    参数: frame为待展示数据，limit为最多展示行数。
    返回: Markdown表格文本。
    调用位置: write_report。
    """
    view = frame.head(limit).copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_report(cfg: AnalysisConfig, result: dict, figures: list[str]) -> Path:
    """功能: 生成与当前数据统计表和图表一致的详细报告。
    参数: cfg为分析配置，result为统计结果，figures为生成图表路径。
    返回: Markdown报告路径。
    调用位置: main.py。
    """
    record = result["record"]
    tables = result["frames"]
    rel = lambda path: Path(path).relative_to(cfg.out_dir).as_posix()
    scene_leader = tables["scene"].sort_values("total_duration_hours", ascending=False).iloc[0]
    missing_fields = tables["dimensions"].query("缺失数 > 0").sort_values("缺失数", ascending=False)
    long_gap = tables["anomaly_summary"].loc[tables["anomaly_summary"].code == "long_sampling_gap"]
    negative_altitude = tables["anomaly_summary"].loc[tables["anomaly_summary"].code == "physical_range_gps_z"]
    lines = [
        "# AMOVFLY数据集统计与异常分析", "", "## 目录", "",
        "- [数据范围与总体规模](#数据范围与总体规模)", "- [数据维度](#数据维度)",
        "- [描述性统计](#描述性统计)", "- [缺失与异常分析](#缺失与异常分析)",
        "- [代表性航线](#代表性航线)", "- [图表说明](#图表说明)",
        "- [输出文件](#输出文件)", "", "## 数据范围与总体规模", "",
        f"本分析读取 `{record['source_directory']}` 下FAFS、FAVS、VAFS、VAVS、Random目录中与场景目录同名的ready CSV。raw_data采集文件、Route素材和wind独立观测文件不与ready飞行逐点表混合统计。",
        "", f"共{record['ready_files']}个ready文件、{record['total_rows']:,}行记录；飞行时长合计{record['duration_hours']:.2f}小时，单飞行平均{record['mean_flight_duration_s']:.1f}s，中位数{record['median_flight_duration_s']:.1f}s。"
        f"Flight_info记录{record['flight_info_entries']}行；与ready文件名未匹配数为{record['unmatched_flight_info']}。", "",
        _md_table(tables["scene"]), "", "时长为每个文件time最大值减最小值之和，不等同于连续飞行占空时间；文件行数包含源表已有采样记录。",
        f"\n\n{scene_leader.scene}累计时长占比最高，为{scene_leader.duration_share_percent:.1f}%。未匹配Flight_info的7个ready文件为：{', '.join(record['unmatched_flight_names'])}。",
        "", "## 数据维度", "", f"源ready CSV有{record['raw_columns']}列（含一个行序号）；统计表另含文件名、场景两个索引字段及3个派生字段，共{len(tables['dimensions'])}列。派生字段为载荷文件名参数、速度模长和电压电流功率乘积。行序号不计作传感器维度。", "",
        _md_table(tables["dimensions"], 40), "", "## 描述性统计", "",
        "下表包含有效值数、均值、标准差、分位数和极值。IQR候选按全体记录的Q1−1.5×IQR与Q3+1.5×IQR计算，适用于定位长尾或需复核的记录，不能单独据此认定测量错误。", "",
        _md_table(tables["stats"], 40), "", "## 缺失与异常分析", "",
        f"全表共{record['missing_cells']:,}个缺失单元格。逐字段结果见[缺失统计表]({rel(result['tables']['missing'])})。", "",
        "异常检查分为采样/记录结构、量纲物理边界和跨字段一致性。功率一致性按原始说明中的power≈battery_voltage×battery_current检查，容差max(2W,5%)；四元数模长以1±0.05筛查。阈值外样本保留并列入复核，不在探索统计阶段自动删除。", "",
        _md_table(tables["anomaly_summary"], 80), "",
        (f"缺失主要集中在{missing_fields.iloc[0]['字段']}和{missing_fields.iloc[1]['字段']}，分别为{int(missing_fields.iloc[0]['缺失数']):,}和{int(missing_fields.iloc[1]['缺失数']):,}行；"
         if len(missing_fields) >= 2 else "缺失字段较少；")
        + (f"超过2秒的采样间隔分布于{int(long_gap.iloc[0].affected_flights)}个航段，共{int(long_gap.iloc[0].flagged_values)}处；"
           if len(long_gap) else "未发现超过2秒的采样间隔；")
        + (f"gps_z小于0的记录分布于{int(negative_altitude.iloc[0].affected_flights)}个航段，共{int(negative_altitude.iloc[0].flagged_values):,}行，需结合离地零点、起降阶段和测量噪声判断，不宜直接剔除。"
           if len(negative_altitude) else "未发现gps_z小于0的记录。"), "",
        "逐飞行事件明细：[异常事件表](" + rel(result["tables"]["anomalies"]) + ")；逐航段规模、时间和轨迹范围：[飞行统计表](" + rel(result["tables"]["flights"]) + ")。采样断点按排序后的最大时间差描述，实际时间序列不在本分析中插值或裁剪。", "",
        "### 统计极端值候选", "", _md_table(tables["stats"].sort_values("IQR极端候选数", ascending=False)[["字段", "有效数", "IQR下界", "IQR上界", "IQR极端候选数", "IQR极端候选率_percent"]], 30), "",
        "## 代表性航线", "",
        "每个场景选取行数最多的单次飞行作为展示样例。左图将实际real_lat/real_long轨迹及有效aim_lat/aim_long参考点投影到首个GPS点附近的局部东/北平面；右图对比gps_z与aim_z。局部平面投影只用于路线形态展示；原始gps_x/gps_y仍保留在字段统计，不将两种位置表示混为同一坐标系。", "",
        _md_table(pd.DataFrame(record["representatives"])), "",
    ]
    lines += ["## 图表说明", ""]
    for figure in figures:
        path = Path(figure)
        if path.suffix.lower() == ".png":
            key = path.stem.split("_", 1)[-1]
            lines += [f"### {key}", "", f"![{key}]({rel(path)})", ""]
    lines += ["## 输出文件", "", "- [字段维度清单](results/dimension_inventory.csv)",
              "- [数值字段描述统计与IQR候选](results/numeric_statistics.csv)",
              "- [场景统计](results/scene_summary.csv)", "- [逐飞行统计及Flight_info](results/flight_summary.csv)",
              "- [字段缺失率](results/missingness.csv)", "- [异常事件明细](results/anomaly_events.csv)",
              "- [异常事件汇总](results/anomaly_summary.csv)", "- [机器可读摘要](results/analysis_summary.json)",
              "- [运行环境](runtime.json)", "", "输出图表同时提供PNG与SVG，源表不被修改。", ""]
    cfg.report_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg.report_path
