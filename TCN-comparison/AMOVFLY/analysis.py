# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: analysis.py
# 开发时间: 2026-11-02
# 文件名: analysis.py
# 功能说明: 统计AMOVFLY ready数据字段、飞行场景与异常记录
# 版本号：1.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import AnalysisConfig


NUMERIC_METADATA = {
    "time": ("s", "飞行启动后的时间"),
    "wind_speed": ("m/s", "无人机相对风速"),
    "wind_angle": ("deg", "无人机相对风向角"),
    "air_pressure": ("Pa", "机载气压"),
    "battery_voltage": ("V", "电池端电压"),
    "battery_current": ("A", "电池端电流"),
    "battery_remain": ("比例", "电池剩余比例"),
    "gps_x": ("m", "起飞点局部坐标x"),
    "gps_y": ("m", "起飞点局部坐标y"),
    "gps_z": ("m", "离地高度"),
    "real_lat": ("deg", "实际位置纬度"),
    "real_long": ("deg", "实际位置经度"),
    "aim_lat": ("deg", "参考航点纬度"),
    "aim_long": ("deg", "参考航点经度"),
    "aim_z": ("m", "参考航点高度"),
    "o_x": ("四元数", "姿态四元数x分量"),
    "o_y": ("四元数", "姿态四元数y分量"),
    "o_z": ("四元数", "姿态四元数z分量"),
    "o_w": ("四元数", "姿态四元数w分量"),
    "v_x": ("m/s", "局部速度x分量"),
    "v_y": ("m/s", "局部速度y分量"),
    "v_z": ("m/s", "垂直速度分量"),
    "la_x": ("m/s^2", "加速度x分量"),
    "la_y": ("m/s^2", "加速度y分量"),
    "la_z": ("m/s^2", "加速度z分量"),
    "power": ("W", "电压乘电流得到的功率"),
}


def ready_files(cfg: AnalysisConfig) -> list[Path]:
    """功能: 收集五种飞行场景下的ready CSV。
    参数: cfg为分析配置。
    返回: 按场景与文件名排序的CSV路径。
    调用位置: analyze_dataset。
    """
    return sorted(path for scene in cfg.scenes
                  for path in (cfg.data_dir / scene / scene).glob("*.csv"))


def _payload_from_name(name: str) -> float:
    """功能: 从飞行文件名中解析载荷参数。
    参数: name为CSV文件名。
    返回: 文件名P字段克数，未记录时为NaN。
    调用位置: analyze_dataset。
    """
    match = re.search(r"_P(\d+)", name)
    return float(match.group(1)) if match else np.nan


def _physical_anomalies(frame: pd.DataFrame, path: Path, cfg: AnalysisConfig) -> list[dict]:
    """功能: 对单次飞行检查明显的数据质量与物理边界问题。
    参数: frame为原始ready表，path为文件路径，cfg为阈值配置。
    返回: 文件级异常事件列表。
    调用位置: analyze_dataset。
    """
    name = path.stem
    events = []

    def add(code, count, severity, detail):
        if count:
            events.append({"flight_name": name, "scene": path.parent.name, "code": code,
                           "count": int(count), "severity": severity, "detail": detail})

    numeric = frame.select_dtypes(include=[np.number])
    nonfinite = int((~np.isfinite(numeric.to_numpy(float))).sum())
    add("nonfinite_value", nonfinite, "high", "数值字段含NaN或正负无穷")
    add("duplicate_time", frame.time.duplicated().sum(), "high", "time字段重复")
    diffs = frame.time.diff().dropna()
    add("non_increasing_time", (diffs <= 0).sum(), "high", "相邻时间未严格递增")
    add("long_sampling_gap", (diffs > cfg.max_gap_seconds).sum(), "medium",
        f"相邻时间差超过{cfg.max_gap_seconds:g}s")
    irregular = ((diffs - cfg.expected_sample_seconds).abs() > .05).sum()
    add("irregular_sampling", irregular, "low", "采样间隔偏离0.2秒超过0.05秒")
    add("negative_time", (frame.time < 0).sum(), "medium", "飞行相对时间小于0秒")
    for column, mask, detail, severity in (
        ("wind_speed", frame.wind_speed < 0, "风速小于0m/s", "high"),
        ("wind_angle", (frame.wind_angle < 0) | (frame.wind_angle >= 360), "风向角不在[0,360)度", "medium"),
        ("battery_voltage", frame.battery_voltage <= 0, "电池电压非正", "high"),
        ("battery_current", frame.battery_current < 0, "电池电流为负", "medium"),
        ("battery_remain", (frame.battery_remain < 0) | (frame.battery_remain > 1), "电池剩余比例不在[0,1]", "high"),
        ("power", frame.power < 0, "功率为负", "high"),
        ("gps_z", frame.gps_z < 0, "离地高度为负", "medium"),
    ):
        add(f"physical_range_{column}", mask.sum(), severity, detail)
    expected_power = frame.battery_voltage * frame.battery_current
    power_diff = (frame.power - expected_power).abs()
    mismatch = power_diff > np.maximum(2.0, expected_power.abs() * .05)
    add("power_formula_mismatch", mismatch.sum(), "medium", "|power-voltage*current|超过max(2W, 5%)")
    quat = frame[["o_x", "o_y", "o_z", "o_w"]]
    quat_norm = np.sqrt((quat ** 2).sum(axis=1))
    add("quaternion_norm", (quat_norm.sub(1).abs() > .05).sum(), "low", "四元数模长偏离1超过0.05")
    return events


def analyze_dataset(cfg: AnalysisConfig) -> dict:
    """功能: 汇总全部ready数据的规模、维度、描述统计和异常清单。
    参数: cfg为分析配置。
    返回: 各类统计表路径、全局信息和代表性航段数据。
    调用位置: main.py。
    """
    files = ready_files(cfg)
    if not files:
        raise FileNotFoundError(f"未找到ready CSV: {cfg.data_dir}")

    frames, flight_rows, anomaly_rows = [], [], []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing_columns = sorted(set(NUMERIC_METADATA) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{path.name}缺少字段: {missing_columns}")
        frame.insert(0, "flight_name", path.stem)
        frame.insert(1, "scene", path.parent.name)
        frame["payload_g"] = _payload_from_name(path.name)
        frame["speed_mps"] = np.sqrt(frame[["v_x", "v_y", "v_z"]].pow(2).sum(axis=1))
        frame["power_from_vi_w"] = frame.battery_voltage * frame.battery_current
        frames.append(frame)
        finite_time = frame.time[np.isfinite(frame.time)]
        sorted_time = np.sort(finite_time.to_numpy(float))
        diffs = np.diff(sorted_time)
        valid = frame.dropna(subset=["gps_x", "gps_y", "gps_z"])
        flight_rows.append({
            "flight_name": path.stem, "scene": path.parent.name, "source_file": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame), "columns": len(frame.columns) - 3,
            "duration_s": float(finite_time.max() - finite_time.min()) if len(finite_time) else np.nan,
            "time_start_s": float(finite_time.min()) if len(finite_time) else np.nan,
            "time_end_s": float(finite_time.max()) if len(finite_time) else np.nan,
            "max_sorted_gap_s": float(diffs.max()) if len(diffs) else np.nan,
            "duplicate_time_rows": int(frame.time.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
            "position_span_x_m": float(valid.gps_x.max() - valid.gps_x.min()) if len(valid) else np.nan,
            "position_span_y_m": float(valid.gps_y.max() - valid.gps_y.min()) if len(valid) else np.nan,
            "altitude_min_m": float(valid.gps_z.min()) if len(valid) else np.nan,
            "altitude_max_m": float(valid.gps_z.max()) if len(valid) else np.nan,
            "mean_power_w": float(frame.power.mean()),
            "energy_integral_wh": float(np.trapezoid(frame.power.fillna(0), frame.time.fillna(0)) / 3600)
                if frame.time.is_monotonic_increasing else np.nan,
            "payload_g": _payload_from_name(path.name),
            "uav": "R" if path.stem.startswith("UavR_") else "G" if path.stem.startswith("UavG_") else "Y",
        })
        anomaly_rows.extend(_physical_anomalies(frame, path, cfg))

    data = pd.concat(frames, ignore_index=True, sort=False)
    flights = pd.DataFrame(flight_rows)
    dimensions = []
    numeric_stats = []
    numeric_columns = [column for column in data.columns
                       if pd.api.types.is_numeric_dtype(data[column]) and column != "Unnamed: 0"]
    for column in data.columns:
        series = data[column]
        if column in {"flight_name", "scene"}:
            kind, unit, meaning = "类别", "-", "飞行文件及场景索引"
        elif column == "Unnamed: 0":
            kind, unit, meaning = "整数索引", "-", "源CSV行序号，不作传感器变量"
        elif column == "payload_g":
            kind, unit, meaning = "数值派生", "g", "从文件名P参数解析的载荷"
        elif column == "speed_mps":
            kind, unit, meaning = "数值派生", "m/s", "由v_x/v_y/v_z求三维模长"
        elif column == "power_from_vi_w":
            kind, unit, meaning = "数值派生", "W", "battery_voltage*battery_current，用于一致性核验"
        else:
            kind = "数值" if pd.api.types.is_numeric_dtype(series) else "文本"
            unit, meaning = NUMERIC_METADATA.get(column, ("-", "数据集原始字段"))
        dimensions.append({"字段": column, "类型": kind, "单位": unit, "非空数": int(series.notna().sum()),
                           "缺失数": int(series.isna().sum()), "缺失率_percent": float(series.isna().mean() * 100),
                           "唯一值数": int(series.nunique(dropna=True)), "含义": meaning})
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite):
                q1, median, q3 = finite.quantile([.25, .5, .75])
                iqr = q3 - q1
                lower, upper = q1 - cfg.iqr_factor * iqr, q3 + cfg.iqr_factor * iqr
                extreme = (finite < lower) | (finite > upper)
                numeric_stats.append({
                    "字段": column, "单位": unit, "有效数": int(len(finite)),
                    "均值": float(finite.mean()), "标准差": float(finite.std()), "最小值": float(finite.min()),
                    "P01": float(finite.quantile(.01)), "P25": float(q1), "中位数": float(median),
                    "P75": float(q3), "P99": float(finite.quantile(.99)), "最大值": float(finite.max()),
                    "IQR下界": float(lower), "IQR上界": float(upper), "IQR极端候选数": int(extreme.sum()),
                    "IQR极端候选率_percent": float(extreme.mean() * 100),
                })

    stats = pd.DataFrame(numeric_stats)
    scene_summary = flights.groupby("scene", sort=False).agg(
        flight_count=("flight_name", "count"), total_rows=("rows", "sum"),
        total_duration_hours=("duration_s", lambda x: x.sum() / 3600),
        median_duration_s=("duration_s", "median"), mean_duration_s=("duration_s", "mean"),
        mean_flight_energy_wh=("energy_integral_wh", "mean"),
        median_altitude_max_m=("altitude_max_m", "median"),
    ).reset_index()
    scene_summary["duration_share_percent"] = scene_summary.total_duration_hours / scene_summary.total_duration_hours.sum() * 100
    missing = pd.DataFrame([{"字段": column, "缺失数": int(data[column].isna().sum()),
                             "缺失率_percent": float(data[column].isna().mean() * 100)}
                            for column in data.columns if data[column].isna().any()])
    anomalies = pd.DataFrame(anomaly_rows, columns=["flight_name", "scene", "code", "count", "severity", "detail"])
    anomaly_summary = (anomalies.groupby(["code", "severity", "detail"], sort=False)
                       .agg(affected_flights=("flight_name", "nunique"), flagged_values=("count", "sum"))
                       .reset_index().sort_values(["severity", "affected_flights"], ascending=[True, False]))
    info_path = cfg.data_dir / "Flight_info.csv"
    if info_path.exists():
        info = pd.read_csv(info_path, encoding="utf-8-sig")
        if "FlightName" in info and "Data Dir" in info:
            metadata = info.rename(columns={"FlightName": "flight_name", "Data Dir": "scene"})
            flights = flights.merge(metadata, on=["flight_name", "scene"], how="left", suffixes=("", "_info"))

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dimensions": cfg.result_dir / "dimension_inventory.csv",
        "statistics": cfg.result_dir / "numeric_statistics.csv",
        "scene": cfg.result_dir / "scene_summary.csv",
        "flights": cfg.result_dir / "flight_summary.csv",
        "missing": cfg.result_dir / "missingness.csv",
        "anomalies": cfg.result_dir / "anomaly_events.csv",
        "anomaly_summary": cfg.result_dir / "anomaly_summary.csv",
    }
    for key, table in (("dimensions", pd.DataFrame(dimensions)), ("statistics", stats),
                       ("scene", scene_summary), ("flights", flights), ("missing", missing),
                       ("anomalies", anomalies), ("anomaly_summary", anomaly_summary)):
        table.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    representatives = []
    for scene, group in flights.groupby("scene", sort=False):
        representatives.append(group.sort_values(["rows", "duration_s"], ascending=False).iloc[0].to_dict())
    # 图表按需从代表性文件重新读取，避免长期保留完整逐点数据副本。
    record = {
        "source_directory": str(cfg.data_dir), "ready_files": len(files), "total_rows": int(len(data)),
        "raw_columns": int(len(NUMERIC_METADATA) + 1), "derived_columns": 3,
        "scene_counts": {str(k): int(v) for k, v in flights.scene.value_counts().items()},
        "missing_cells": int(data.isna().sum().sum()), "numeric_columns": int(len(numeric_columns)),
        "text_columns": int(data.shape[1] - len(numeric_columns)),
        "duration_hours": float(flights.duration_s.sum() / 3600),
        "mean_flight_duration_s": float(flights.duration_s.mean()),
        "median_flight_duration_s": float(flights.duration_s.median()),
        "flight_info_entries": int(pd.read_csv(info_path, encoding="utf-8-sig").shape[0]) if info_path.exists() else None,
        "unmatched_flight_info": int(flights.flight_name.isin(info.FlightName).eq(False).sum()) if info_path.exists() and "FlightName" in info else None,
        "unmatched_flight_names": flights.loc[~flights.flight_name.isin(info.FlightName), "flight_name"].tolist()
            if info_path.exists() and "FlightName" in info else [],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "representatives": [{"scene": row["scene"], "flight_name": row["flight_name"],
                             "source_file": row["source_file"], "rows": int(row["rows"]),
                             "duration_s": float(row["duration_s"])} for row in representatives],
        "iqr_definition": "per-field global Tukey fences Q1-1.5*IQR and Q3+1.5*IQR; candidates require context, not automatic errors",
        "physical_checks": "nonfinite, duplicate/non-increasing time, >2s gaps, sampling jitter, basic physical ranges, power=V*I tolerance, quaternion norm",
    }
    (cfg.result_dir / "analysis_summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"record": record, "tables": outputs, "frames": {"flights": flights, "scene": scene_summary,
            "stats": stats, "dimensions": pd.DataFrame(dimensions), "missing": missing,
            "anomalies": anomalies, "anomaly_summary": anomaly_summary},
            "representatives": representatives}
