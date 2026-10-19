# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: gecr_adapter.py
# 开发时间: 2026-10-19
# 文件名: gecr_adapter.py
# 功能说明: 下载并转换GECR真实道路电动车行程为4.1字段的轨迹回放验证数据
# 版本号：4.1

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ROOT = "https://raw.githubusercontent.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev/main/data/real_world_data"
REQUIRED_COLUMNS = {
    "PhaseTime", "DAActualSpeed", "DAScheduleSpeed", "REESSCurrent",
    "REESSVoltage", "tempAmb", "elevation", "lat", "long", "gear",
    "tot_disc", "tot_char",
}
METERS_PER_DEGREE_LAT = 110_540.0
METERS_PER_DEGREE_LON = 111_320.0


def build_parser() -> argparse.ArgumentParser:
    """功能: 构造GECR数据下载和轨迹回放转换命令行参数。
    参数: 无。
    返回: 子命令参数解析器。
    调用位置: main。
    """

    parser = argparse.ArgumentParser(description="GECR真实道路BEV行程数据连接器")
    parser.add_argument("action", choices=("download", "convert"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "gecr")
    parser.add_argument("--trip", type=int, nargs="+", help="行程编号1-106；不指定时处理全部行程。")
    return parser


def _trip_ids(values: list[int] | None) -> list[int]:
    """功能: 校验并规范化GECR行程编号。
    参数: values为可选行程编号列表。
    返回: 去重且升序的合法编号。
    调用位置: download_trips、convert_trips。
    """

    ids = sorted(set(values if values else range(1, 107)))
    invalid = [trip for trip in ids if trip < 1 or trip > 106]
    if invalid:
        raise ValueError(f"GECR行程编号必须位于1-106: {invalid}")
    return ids


def download_trips(data_dir: Path, trips: list[int] | None = None) -> list[Path]:
    """功能: 从数据集公开GitHub仓库下载指定真实道路行程Excel文件。
    参数: data_dir为原始文件保存目录，trips为1-106行程编号。
    返回: 成功下载或已存在的文件路径列表。
    调用位置: main的download子命令。
    """

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for trip in _trip_ids(trips):
        path = raw_dir / f"{trip:03d}.xlsx"
        if not path.exists():
            url = f"{DATASET_ROOT}/{trip:03d}.xlsx"
            request = urllib.request.Request(url, headers={"User-Agent": "Energy-prediction-4.1/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as target:
                target.write(response.read())
        outputs.append(path)
    return outputs


def convert_trips(data_dir: Path, trips: list[int] | None = None) -> dict:
    """功能: 将GECR行程转为标明回放代理语义的4.1字段CSV。
    参数: data_dir为原始Excel及转换CSV所在目录，trips为1-106行程编号。
    返回: 转换行程数、记录数、缺失输入和能耗统计。
    调用位置: main的convert子命令。
    """

    raw_dir = data_dir / "raw"
    output_dir = data_dir / "replay_proxy"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for trip in _trip_ids(trips):
        source_path = raw_dir / f"{trip:03d}.xlsx"
        if not source_path.exists():
            raise FileNotFoundError(f"缺少原始行程文件，请先运行download: {source_path}")
        source = pd.read_excel(source_path, sheet_name="Continuous20Hz")
        missing = sorted(REQUIRED_COLUMNS - set(source.columns))
        if missing:
            raise ValueError(f"{source_path.name}缺少字段: {missing}")
        source = source.sort_values("PhaseTime", kind="stable").drop_duplicates("PhaseTime", keep="last").reset_index(drop=True)
        time_s = source["PhaseTime"].to_numpy(float)
        if len(time_s) < 2 or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0):
            raise ValueError(f"{source_path.name}的PhaseTime无效或不递增")

        latitude0 = float(source["lat"].iloc[0])
        longitude0 = float(source["long"].iloc[0])
        east = (source["long"].to_numpy(float) - longitude0) * METERS_PER_DEGREE_LON * np.cos(np.deg2rad(latitude0))
        north = (source["lat"].to_numpy(float) - latitude0) * METERS_PER_DEGREE_LAT
        up = source["elevation"].to_numpy(float) - float(source["elevation"].iloc[0])

        # 样例中放电电流为负，且放电累计量递增；统一以电池放电方向为正功率。
        power_w = np.maximum(-source["REESSVoltage"].to_numpy(float) * source["REESSCurrent"].to_numpy(float), 0.0)
        dt = np.diff(time_s, append=time_s[-1] + float(np.median(np.diff(time_s))))
        energy_wh = power_w * dt / 3600.0
        motor_on = (power_w >= 100.0).astype(np.int8)
        moving = (source["DAActualSpeed"].to_numpy(float) > 1.0).astype(np.int8)
        airborne_proxy = np.minimum(moving, motor_on)
        duration = float(time_s[-1] - time_s[0])
        frame = pd.DataFrame({
            "flight": trip,
            "time_s": time_s - time_s[0],
            "task_duration_s": duration,
            "planned_position_east_m": east,
            "planned_position_north_m": north,
            "planned_position_up_m": up,
            "wind_east_mps": 0.0,
            "wind_north_mps": 0.0,
            "payload_g": 0.0,
            "planned_motor_on": motor_on,
            "planned_airborne": airborne_proxy,
            "power_w": power_w,
            "dt_seconds": dt,
            "energy_interval_wh": energy_wh,
            "vehicle_speed_source": source["DAActualSpeed"],
            "vehicle_schedule_speed_source": source["DAScheduleSpeed"],
            "battery_voltage_v": source["REESSVoltage"],
            "battery_current_a": source["REESSCurrent"],
            "ambient_temperature_c": source["tempAmb"],
            "gear": source["gear"],
            "cumulative_discharge_source": source["tot_disc"],
            "cumulative_charge_source": source["tot_char"],
            "input_semantics": "实测GPS轨迹回放代理；非任务前规划输入",
            "status_semantics": "motor_on为实测电池功率阈值；airborne字段仅为车辆移动状态代理",
        })
        path = output_dir / f"gecr_trip_{trip:03d}_replay_proxy.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        summaries.append({
            "trip": trip,
            "rows": len(frame),
            "duration_s": duration,
            "median_sample_period_s": float(np.median(np.diff(time_s))),
            "energy_from_voltage_current_wh": float(energy_wh.sum()),
            "speed_max_source_units": float(source["DAActualSpeed"].max()),
            "output": str(path),
        })

    result = {
        "dataset": "GECR / Generalized Energy Consumption Rates",
        "trip_count": len(summaries),
        "row_count": int(sum(item["rows"] for item in summaries)),
        "energy_label": "max(-REESSVoltage * REESSCurrent, 0)按采样间隔积分；极性依据仓库样例及累计放电趋势，正式实验前须再核对原文单位定义",
        "domain_warning": "输出中的planned_*和airborne为格式兼容的回放代理，不代表无人机任务前计划量；不能直接用于证明无人机任务前部署性能。",
        "trips": summaries,
    }
    report_path = output_dir / "gecr_conversion_summary.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """功能: 执行GECR数据下载或4.1轨迹回放代理转换。
    参数: 命令行参数。
    返回: None。
    调用位置: 脚本直接运行。
    """

    args = build_parser().parse_args()
    result = download_trips(args.data_dir, args.trip) if args.action == "download" else convert_trips(args.data_dir, args.trip)
    print(json.dumps({"processed": len(result) if isinstance(result, list) else result["trip_count"], "data_dir": str(args.data_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
