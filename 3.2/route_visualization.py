# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: route_visualization.py
# 开发时间: 2026-09-14
# 文件名: route_visualization.py
# 功能说明: 为指定航线生成能量、功率、IMU轨迹静态图和风场标注动图
# 版本号：3.2

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "out"
PREDICTION_FILE = OUT_DIR / "predictions" / "test_predictions_3.2.csv"
RAW_FLIGHTS_FILE = Path("D:/Python-files/Energy-prediction/data/dji_matrice_100/flights.csv")
ROUTE_IDS = (18, 23, 83)
METERS_PER_DEGREE_LAT = 110540.0
METERS_PER_DEGREE_LON = 111320.0


def _metric(value: pd.Series, fallback: float = 0.0) -> float:
    return float(value.iloc[0]) if len(value) else fallback


def _wind_annotation(frame: pd.DataFrame) -> str:
    speed = _metric(frame.get("wind_speed", pd.Series(dtype=float)))
    angle = _metric(frame.get("wind_angle", pd.Series(dtype=float)))
    return f"风速 {speed:.2f} m/s\n风向 {angle:.1f}°"


def _add_local_meter_coordinates(frame: pd.DataFrame, origin_lon: float, origin_lat: float, origin_z: float) -> pd.DataFrame:
    """功能: 将原始经纬度和高度转换为局部米制坐标。
    参数: frame为含position_x、position_y、position_z的数据表；origin为统一参考点。
    返回: 增加position_x_m、position_y_m、position_z_m列的数据表。
    调用位置: generate_route_products。
    """
    result = frame.copy()
    latitude_scale = METERS_PER_DEGREE_LAT
    longitude_scale = METERS_PER_DEGREE_LON * np.cos(np.deg2rad(origin_lat))
    result["position_x_m"] = (result["position_x"] - origin_lon) * longitude_scale
    result["position_y_m"] = (result["position_y"] - origin_lat) * latitude_scale
    result["position_z_m"] = result["position_z"] - origin_z
    return result


def _load_route(route_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = pd.read_csv(PREDICTION_FILE)
    prediction = prediction[prediction["flight"].eq(route_id)].sort_values("time").reset_index(drop=True)
    if prediction.empty:
        raise ValueError(f"预测文件中未找到航线 {route_id}")
    imu = pd.read_csv(RAW_FLIGHTS_FILE, usecols=[
        "flight", "time", "wind_speed", "wind_angle", "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    ])
    imu = imu[imu["flight"].eq(route_id)].sort_values("time").reset_index(drop=True)
    if imu.empty:
        raise ValueError(f"原始IMU文件中未找到航线 {route_id}")
    # 预测数据来自重采样网格，IMU位置按最近时间戳对齐，保留原始轨迹坐标。
    imu = imu.drop_duplicates("time")
    aligned = pd.merge_asof(prediction.sort_values("time"), imu.sort_values("time"), on="time", direction="nearest", suffixes=("", "_imu"))
    return aligned, imu


def _save_static_plots(frame: pd.DataFrame, route_dir: Path, route_id: int) -> list[str]:
    route_dir.mkdir(parents=True, exist_ok=True)
    annotation = _wind_annotation(frame)
    outputs: list[str] = []
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame["time"], frame["power_w"], label="实际功率", lw=1.2)
    axes[0].plot(frame["time"], frame["tcn_predicted_power_w"], label="TCN功率", lw=1.0)
    axes[0].plot(frame["time"], frame["predicted_power_w"], label="RLS校正功率", lw=1.2)
    axes[0].set_ylabel("功率 (W)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"航线 {route_id} 功率与能量")
    axes[1].plot(frame["time"], frame["actual_cumulative_energy_wh"], label="实际累计能量", lw=1.2)
    axes[1].plot(frame["time"], frame["tcn_cumulative_energy_wh"], label="TCN累计能量", lw=1.0)
    axes[1].plot(frame["time"], frame["predicted_cumulative_energy_wh"], label="RLS累计能量", lw=1.2)
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("累计能量 (Wh)")
    axes[1].legend(loc="upper left")
    axes[0].text(0.015, 0.96, annotation, transform=axes[0].transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8})
    fig.tight_layout()
    path = route_dir / f"flight_{route_id}_power_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    return outputs


def _save_animation(frame: pd.DataFrame, route_dir: Path, route_id: int) -> str:
    stride = max(1, len(frame) // 180)
    sampled = frame.iloc[::stride].reset_index(drop=True)
    fig = plt.figure(figsize=(13, 5))
    power_axis = fig.add_subplot(1, 2, 1)
    map_axis = fig.add_subplot(1, 2, 2, projection="3d")
    power_axis.plot(frame["time"], frame["power_w"], color="#d0d0d0", lw=0.8, label="实际功率")
    power_axis.plot(frame["time"], frame["predicted_power_w"], color="#4c78a8", lw=0.9, label="RLS校正功率")
    power_point, = power_axis.plot([], [], "o", color="#e45756", label="当前时刻")
    map_axis.plot(frame["position_x_m"], frame["position_y_m"], frame["position_z_m"], color="#d0d0d0", lw=0.8, label="完整参考轨迹")
    trajectory_line, = map_axis.plot([], [], [], color="#4c78a8", lw=1.6, label="已飞轨迹")
    position_point, = map_axis.plot([], [], [], "o", color="#e45756", label="当前位置")
    annotation = map_axis.text2D(0.02, 0.98, "", transform=map_axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    power_axis.set_xlabel("时间 (s)")
    power_axis.set_ylabel("功率 (W)")
    map_axis.set_xlabel("位置 X (m)")
    map_axis.set_ylabel("位置 Y (m)")
    map_axis.set_zlabel("高度 Z (m)")
    map_axis.set_title("3D IMU轨迹")
    power_axis.legend(loc="upper right", fontsize=8)
    map_axis.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"航线 {route_id} 3D IMU轨迹与功率动态")

    def update(index: int):
        current = sampled.iloc[index]
        power_point.set_data([current["time"]], [current["predicted_power_w"]])
        subset = sampled.iloc[: index + 1]
        trajectory_line.set_data(subset["position_x_m"], subset["position_y_m"])
        trajectory_line.set_3d_properties(subset["position_z_m"])
        position_point.set_data([current["position_x_m"]], [current["position_y_m"]])
        position_point.set_3d_properties([current["position_z_m"]])
        annotation.set_text(f"时间 {current['time']:.1f} s\n风速 {current['wind_speed']:.2f} m/s\n风向 {current['wind_angle']:.1f}°")
        return power_point, trajectory_line, position_point, annotation

    animation = FuncAnimation(fig, update, frames=len(sampled), interval=50, blit=False)
    path = route_dir / f"flight_{route_id}_imu_trajectory.gif"
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    return str(path)


def generate_route_products() -> dict:
    """功能: 为18、23、83航线生成独立图表、动图、IMU数据和摘要。
    参数: 无，读取3.2固定模型测试预测和原始IMU文件。
    返回: 航线到产物路径的摘要字典。
    调用位置: 命令行直接运行或README复现实验。
    """
    summary = {}
    frames: dict[int, pd.DataFrame] = {}
    loaded = {route_id: _load_route(route_id)[0] for route_id in ROUTE_IDS}
    origin_lon = float(np.mean([frame["position_x"].iloc[0] for frame in loaded.values()]))
    origin_lat = float(np.mean([frame["position_y"].iloc[0] for frame in loaded.values()]))
    origin_z = float(np.mean([frame["position_z"].iloc[0] for frame in loaded.values()]))
    for route_id in ROUTE_IDS:
        frame = _add_local_meter_coordinates(loaded[route_id], origin_lon, origin_lat, origin_z)
        frames[route_id] = frame
        route_dir = OUT_DIR / "routes" / f"flight_{route_id}"
        route_dir.mkdir(parents=True, exist_ok=True)
        imu_path = route_dir / f"flight_{route_id}_imu_aligned.csv"
        frame.to_csv(imu_path, index=False, encoding="utf-8")
        outputs = _save_static_plots(frame, route_dir, route_id)
        outputs.append(_save_animation(frame, route_dir, route_id))
        payload = {
            "version": "3.2",
            "flight": route_id,
            "route": str(frame["route"].iloc[0]),
            "rows": int(len(frame)),
            "duration_seconds": float(frame["time"].max() - frame["time"].min()),
            "wind_speed_mps_mean": float(frame["wind_speed"].mean()),
            "wind_angle_deg_mean": float(frame["wind_angle"].mean()),
            "actual_energy_wh": float(frame["actual_cumulative_energy_wh"].iloc[-1]),
            "tcn_energy_wh": float(frame["tcn_cumulative_energy_wh"].iloc[-1]),
            "rls_energy_wh": float(frame["predicted_cumulative_energy_wh"].iloc[-1]),
            "products": outputs + [str(imu_path)],
        }
        summary[str(route_id)] = payload
        (route_dir / f"flight_{route_id}_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_all_routes_comparison(frames)
    (OUT_DIR / "routes" / "route_products_summary_3.2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _save_all_routes_comparison(frames: dict[int, pd.DataFrame]) -> None:
    """功能: 绘制18、23、83三条航线的米制轨迹和累计能量对比图。
    参数: frames为已经统一局部坐标的航线数据字典。
    返回: None。
    调用位置: generate_route_products。
    """
    colors = {18: "#4c78a8", 23: "#f58518", 83: "#54a24b"}
    fig, (trajectory_axis, energy_axis) = plt.subplots(1, 2, figsize=(14, 6))
    for route_id, frame in frames.items():
        color = colors[route_id]
        trajectory_axis.plot(frame["position_x_m"], frame["position_y_m"], color=color, lw=1.5, label=f"航线 {route_id}")
        energy_axis.plot(frame["time"], frame["actual_cumulative_energy_wh"], color=color, lw=1.2, label=f"航线 {route_id} 实际")
        energy_axis.plot(frame["time"], frame["predicted_cumulative_energy_wh"], color=color, lw=1.0, ls="--", label=f"航线 {route_id} RLS")
    trajectory_axis.set_xlabel("局部东向 X (m)")
    trajectory_axis.set_ylabel("局部北向 Y (m)")
    trajectory_axis.set_title("18、23、83 航线局部米制轨迹")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend()
    energy_axis.set_xlabel("时间 (s)")
    energy_axis.set_ylabel("累计能量 (Wh)")
    energy_axis.set_title("不同航线累计能量对比")
    energy_axis.grid(alpha=0.25)
    energy_axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = OUT_DIR / "routes" / "all_routes_trajectory_energy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    result = generate_route_products()
    for route_id, payload in result.items():
        print(route_id, len(payload["products"]), "products")
