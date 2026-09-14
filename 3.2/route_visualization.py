# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: route_visualization.py
# 开发时间: 2026-09-14
# 文件名: route_visualization.py
# 功能说明: 为每种原始航线选择代表飞行并生成统一米制三维轨迹、能量图和风场动图
# 版本号：3.2

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import ScalarFormatter

from config import ExperimentConfig, build_config, ensure_directories
from data_utils import save_json


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METERS_PER_DEGREE_LAT = 110540.0
METERS_PER_DEGREE_LON = 111320.0


def _metric(value: pd.Series, fallback: float = 0.0) -> float:
    """功能: 从序列中读取首个数值并提供空序列默认值。
    参数: value为待读取序列，fallback为空序列默认值。
    返回: 浮点数。
    调用位置: _wind_annotation。
    """

    return float(value.iloc[0]) if len(value) else fallback


def _wind_annotation(frame: pd.DataFrame) -> str:
    """功能: 生成单次飞行的风速和风向角标文本。
    参数: frame为对齐后的单次飞行数据。
    返回: 风场角标字符串。
    调用位置: _save_static_plots。
    """

    speed = _metric(frame.get("wind_speed", pd.Series(dtype=float)))
    angle = _metric(frame.get("wind_angle", pd.Series(dtype=float)))
    aircraft_speed = _metric(frame.get("actual_speed_mps", pd.Series(dtype=float)))
    return f"无人机速度 {aircraft_speed:.2f} m/s\n风速 {speed:.2f} m/s\n风向 {angle:.1f}°"


def _plot_coordinates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """功能: 返回以当前代表航线起点为原点的绘图坐标。
    参数: frame为含全局局部米制坐标的数据表。
    返回: 绘图用东、北、相对高度三轴米制序列。
    调用位置: 静态图、动图和航线对比图。
    """

    return (
        frame["position_x_m"] - frame["position_x_m"].iloc[0],
        frame["position_y_m"] - frame["position_y_m"].iloc[0],
        frame["position_z_m"] - frame["position_z_m"].iloc[0],
    )


def _format_meter_axis(axis) -> None:
    """功能: 将三维坐标轴设置为普通米制刻度并淡化网格。
    参数: axis为matplotlib三维坐标轴。
    返回: 无。
    调用位置: 所有轨迹图绘制函数。
    """

    for setter in (axis.xaxis.set_major_formatter, axis.yaxis.set_major_formatter, axis.zaxis.set_major_formatter):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        setter(formatter)
    axis.xaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.22)
    axis.yaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.22)
    axis.zaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.22)


def _add_local_meter_coordinates(
    frame: pd.DataFrame,
    origin_lon: float,
    origin_lat: float,
    origin_z: float,
) -> pd.DataFrame:
    """功能: 以所有代表飞行的共同参考点将经纬高转换为局部米制坐标。
    参数: frame为含经纬高的数据表，origin_lon、origin_lat、origin_z为共同参考点。
    返回: 增加局部东向X、北向Y和相对高度Z列的数据表。
    调用位置: generate_route_products。
    """

    result = frame.copy()
    valid_position = (
        result["position_x"].abs().gt(1.0)
        & result["position_y"].abs().gt(1.0)
        & result["position_z"].notna()
    )
    longitude_scale = METERS_PER_DEGREE_LON * np.cos(np.deg2rad(origin_lat))
    result["position_x_m"] = (result["position_x"] - origin_lon) * longitude_scale
    result["position_y_m"] = (result["position_y"] - origin_lat) * METERS_PER_DEGREE_LAT
    result["position_z_m"] = result["position_z"] - origin_z
    # A1/A2/A3 的原始位置字段全为0，使用 IMU 速度按采样间隔积分恢复局部轨迹。
    if not valid_position.any():
        dt = pd.to_numeric(result["dt_seconds"], errors="coerce").fillna(0.0).clip(lower=0.0)
        result["position_x_m"] = (pd.to_numeric(result["velocity_x"], errors="coerce").fillna(0.0) * dt).cumsum()
        result["position_y_m"] = (pd.to_numeric(result["velocity_y"], errors="coerce").fillna(0.0) * dt).cumsum()
        result["position_z_m"] = (pd.to_numeric(result["velocity_z"], errors="coerce").fillna(0.0) * dt).cumsum()
    return result


def _select_representative_flights(prediction: pd.DataFrame) -> dict[str, int]:
    """功能: 为每种route选择采样长度最接近该route中位数的代表flight。
    参数: prediction为全部原始航线预测数据。
    返回: route到代表flight编号的映射。
    调用位置: generate_route_products。
    """

    counts = prediction.groupby(["route", "flight"], sort=True).size().rename("rows").reset_index()
    selected: dict[str, int] = {}
    for route, group in counts.groupby("route", sort=True):
        median_rows = float(group["rows"].median())
        ranked = group.assign(distance=(group["rows"] - median_rows).abs()).sort_values(
            ["distance", "flight"], kind="stable"
        )
        selected[str(route)] = int(ranked.iloc[0]["flight"])
    return selected


def _load_aligned_flights(cfg: ExperimentConfig) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """功能: 读取全航线预测和原始IMU，并按时间对齐每种航线的代表flight。
    参数: cfg为实验配置对象。
    返回: route到对齐数据的映射，以及route到代表flight的映射。
    调用位置: generate_route_products。
    """

    if not cfg.all_route_prediction_csv.exists():
        raise FileNotFoundError(f"未找到全航线预测文件: {cfg.all_route_prediction_csv}")
    prediction = pd.read_csv(cfg.all_route_prediction_csv)
    required_prediction = {
        "flight", "route", "time", "power_w", "tcn_predicted_power_w", "predicted_power_w",
        "actual_cumulative_energy_wh", "tcn_cumulative_energy_wh", "predicted_cumulative_energy_wh",
    }
    missing_prediction = sorted(required_prediction - set(prediction.columns))
    if missing_prediction:
        raise ValueError(f"全航线预测缺少字段: {missing_prediction}")
    prediction["route"] = prediction["route"].astype(str).str.strip()
    selected = _select_representative_flights(prediction)

    imu_columns = [
        "flight", "time", "wind_speed", "wind_angle", "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    ]
    imu = pd.read_csv(cfg.raw_flights_csv, usecols=imu_columns)
    imu = imu[imu["flight"].isin(set(selected.values()))].copy()
    aligned: dict[str, pd.DataFrame] = {}
    for route, flight_id in selected.items():
        flight_prediction = prediction[prediction["flight"].eq(flight_id)].sort_values("time").reset_index(drop=True)
        flight_imu = imu[imu["flight"].eq(flight_id)].sort_values("time").drop_duplicates("time").reset_index(drop=True)
        if flight_prediction.empty or flight_imu.empty:
            raise ValueError(f"route={route}、flight={flight_id}缺少预测或IMU记录。")
        aligned[route] = pd.merge_asof(
            flight_prediction,
            flight_imu,
            on="time",
            direction="nearest",
            suffixes=("", "_imu"),
        )
    return aligned, selected


def _save_static_plots(frame: pd.DataFrame, route_dir: Path, route: str, flight_id: int) -> list[str]:
    """功能: 保存代表flight的功率和累计能量静态图。
    参数: frame为对齐数据，route_dir为输出目录，route和flight_id为标识。
    返回: 产物路径列表。
    调用位置: generate_route_products。
    """

    route_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame["time"], frame["power_w"], label="实际功率", lw=1.2)
    axes[0].plot(frame["time"], frame["tcn_predicted_power_w"], label="TCN功率", lw=1.0)
    axes[0].plot(frame["time"], frame["predicted_power_w"], label="RLS校正功率", lw=1.2)
    axes[0].set_ylabel("功率 (W)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"航线 {route} / flight {flight_id} 功率与能量")
    axes[1].plot(frame["time"], frame["actual_cumulative_energy_wh"], label="实际累计能量", lw=1.2)
    axes[1].plot(frame["time"], frame["tcn_cumulative_energy_wh"], label="TCN累计能量", lw=1.0)
    axes[1].plot(frame["time"], frame["predicted_cumulative_energy_wh"], label="RLS累计能量", lw=1.2)
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("累计能量 (Wh)")
    axes[1].legend(loc="upper left")
    axes[0].text(
        0.015, 0.96, _wind_annotation(frame), transform=axes[0].transAxes, va="top", ha="left",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    path = route_dir / f"flight_{flight_id}_power_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    trajectory_fig = plt.figure(figsize=(8, 6))
    trajectory_axis = trajectory_fig.add_subplot(1, 1, 1, projection="3d")
    x_plot, y_plot, z_plot = _plot_coordinates(frame)
    trajectory_axis.plot(
        x_plot, y_plot, z_plot,
        color="#4c78a8", lw=1.4, label="IMU轨迹",
    )
    trajectory_axis.scatter(
        x_plot.iloc[0], y_plot.iloc[0], z_plot.iloc[0],
        color="#e45756", s=24, label="起点",
    )
    if len(frame) > 8:
        idx = min(8, len(frame) - 1)
        trajectory_axis.quiver(x_plot.iloc[0], y_plot.iloc[0], z_plot.iloc[0], x_plot.iloc[idx], y_plot.iloc[idx], z_plot.iloc[idx], color="#e45756", arrow_length_ratio=0.15, linewidth=1.4)
    trajectory_axis.set_xlabel("X (m)")
    trajectory_axis.set_ylabel("Y (m)")
    trajectory_axis.set_zlabel("Z (m)")
    trajectory_axis.set_title(f"航线 {route} / flight {flight_id} 局部米制3D轨迹")
    trajectory_axis.legend(loc="upper right")
    _format_meter_axis(trajectory_axis)
    trajectory_axis.text2D(0.02, 0.98, _wind_annotation(frame), transform=trajectory_axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    trajectory_path = route_dir / f"flight_{flight_id}_imu_trajectory.png"
    trajectory_fig.tight_layout()
    trajectory_fig.savefig(trajectory_path, dpi=180)
    plt.close(trajectory_fig)
    return [str(path), str(trajectory_path)]


def _save_animation(frame: pd.DataFrame, route_dir: Path, route: str, flight_id: int) -> str:
    """功能: 保存带双侧图例和动态风场角标的三维飞行轨迹GIF。
    参数: frame为统一局部坐标数据，route_dir为输出目录，route和flight_id为标识。
    返回: GIF文件路径。
    调用位置: generate_route_products。
    """

    stride = max(1, len(frame) // 180)
    sampled = frame.iloc[::stride].reset_index(drop=True)
    x_plot, y_plot, z_plot = _plot_coordinates(frame)
    sampled_x, sampled_y, sampled_z = _plot_coordinates(sampled)
    fig = plt.figure(figsize=(13, 5))
    power_axis = fig.add_subplot(1, 2, 1)
    map_axis = fig.add_subplot(1, 2, 2, projection="3d")
    power_axis.plot(frame["time"], frame["power_w"], color="#d0d0d0", lw=0.8, label="实际功率")
    power_axis.plot(frame["time"], frame["predicted_power_w"], color="#4c78a8", lw=0.9, label="RLS校正功率")
    power_point, = power_axis.plot([], [], "o", color="#e45756", label="当前时刻")
    map_axis.plot(
        x_plot, y_plot, z_plot,
        color="#d0d0d0", lw=0.8, label="完整参考轨迹",
    )
    trajectory_line, = map_axis.plot([], [], [], color="#4c78a8", lw=1.6, label="已飞轨迹")
    position_point, = map_axis.plot([], [], [], "o", color="#e45756", label="当前位置")
    annotation = map_axis.text2D(
        0.02, 0.98, "", transform=map_axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    power_axis.set_xlabel("时间 (s)")
    power_axis.set_ylabel("功率 (W)")
    map_axis.set_xlabel("X (m)")
    map_axis.set_ylabel("Y (m)")
    map_axis.set_zlabel("Z (m)")
    map_axis.set_title("统一参考点下的3D IMU轨迹")
    power_axis.legend(loc="upper right", fontsize=8)
    map_axis.legend(loc="upper right", fontsize=8)
    _format_meter_axis(map_axis)
    fig.suptitle(f"航线 {route} / flight {flight_id} 三维轨迹与功率动态")

    def update(index: int):
        current = sampled.iloc[index]
        subset = sampled.iloc[: index + 1]
        power_point.set_data([current["time"]], [current["predicted_power_w"]])
        trajectory_line.set_data(sampled_x.iloc[: index + 1], sampled_y.iloc[: index + 1])
        trajectory_line.set_3d_properties(sampled_z.iloc[: index + 1])
        position_point.set_data([sampled_x.iloc[index]], [sampled_y.iloc[index]])
        position_point.set_3d_properties([sampled_z.iloc[index]])
        annotation.set_text(
            f"时间 {current['time']:.1f} s\n无人机速度 {current['actual_speed_mps']:.2f} m/s\n风速 {current['wind_speed']:.2f} m/s\n风向 {current['wind_angle']:.1f}°"
        )
        return power_point, trajectory_line, position_point, annotation

    animation = FuncAnimation(fig, update, frames=len(sampled), interval=120, blit=False)
    path = route_dir / f"flight_{flight_id}_imu_trajectory.gif"
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    return str(path)


def _save_all_routes_comparison(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    """功能: 绘制各类航线代表flight的统一三维米制轨迹和累计能量对比图。
    参数: frames为统一局部坐标数据，output_dir为航线总输出目录。
    返回: 对比图路径。
    调用位置: generate_route_products。
    """

    fig = plt.figure(figsize=(16, 7))
    trajectory_axis = fig.add_subplot(1, 2, 1, projection="3d")
    energy_axis = fig.add_subplot(1, 2, 2)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, max(len(frames), 1)))
    for color, (route, frame) in zip(colors, sorted(frames.items())):
        flight_id = int(frame["flight"].iloc[0])
        x_plot, y_plot, z_plot = _plot_coordinates(frame)
        trajectory_axis.plot(
            x_plot, y_plot, z_plot,
            color=color, lw=1.4, label=f"{route} / flight {flight_id}",
        )
        trajectory_axis.scatter(
            x_plot.iloc[0], y_plot.iloc[0], z_plot.iloc[0],
            color=[color], s=16,
        )
        if len(frame) > 8:
            idx = min(8, len(frame) - 1)
            trajectory_axis.quiver(
                x_plot.iloc[0], y_plot.iloc[0], z_plot.iloc[0],
                x_plot.iloc[idx], y_plot.iloc[idx], z_plot.iloc[idx],
                color=[color], arrow_length_ratio=0.12, linewidth=1.0,
            )
        energy_axis.plot(
            frame["time"], frame["actual_cumulative_energy_wh"], color=color, lw=1.2,
            label=f"{route}实际",
        )
        energy_axis.plot(
            frame["time"], frame["predicted_cumulative_energy_wh"], color=color, lw=1.0, ls="--",
            label=f"{route} RLS",
        )
    trajectory_axis.set_xlabel("X (m)")
    trajectory_axis.set_ylabel("Y (m)")
    trajectory_axis.set_zlabel("Z (m)")
    trajectory_axis.set_title("各类航线代表flight的统一局部三维轨迹")
    trajectory_axis.legend(fontsize=7, loc="upper left")
    _format_meter_axis(trajectory_axis)
    energy_axis.set_xlabel("时间 (s)")
    energy_axis.set_ylabel("累计能量 (Wh)")
    energy_axis.set_title("各类航线代表flight累计能量对比")
    energy_axis.grid(alpha=0.25)
    energy_axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = output_dir / "all_routes_trajectory_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_route_products(cfg: ExperimentConfig | None = None) -> dict:
    """功能: 为每种原始航线的代表flight生成独立图表、三维动图、对齐数据和摘要。
    参数: cfg为实验配置对象，默认由build_config构建。
    返回: route到产物信息的摘要字典。
    调用位置: main.py的route-visualize模式或命令行直接运行。
    """

    cfg = cfg or build_config()
    ensure_directories(cfg)
    output_dir = cfg.out_dir / "routes"
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded, selected = _load_aligned_flights(cfg)
    valid_starts = [
        frame.iloc[0]
        for frame in loaded.values()
        if abs(float(frame["position_x"].iloc[0])) > 1.0 and abs(float(frame["position_y"].iloc[0])) > 1.0
    ]
    if not valid_starts:
        raise ValueError("所有代表航线的 IMU 位置字段均无有效经纬度，无法建立共同参考点。")
    origin_lon = float(np.mean([row["position_x"] for row in valid_starts]))
    origin_lat = float(np.mean([row["position_y"] for row in valid_starts]))
    origin_z = float(np.mean([row["position_z"] for row in valid_starts]))
    origin = {
        "longitude_deg": origin_lon,
        "latitude_deg": origin_lat,
        "altitude_m": origin_z,
        "definition": "各类航线代表flight起点经度、纬度和高度的算术平均值",
    }

    summary: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for route, flight_id in selected.items():
        frame = _add_local_meter_coordinates(loaded[route], origin_lon, origin_lat, origin_z)
        frames[route] = frame
        route_dir = output_dir / f"route_{route}" / f"flight_{flight_id}"
        route_dir.mkdir(parents=True, exist_ok=True)
        aligned_path = route_dir / f"flight_{flight_id}_imu_aligned.csv"
        frame.to_csv(aligned_path, index=False, encoding="utf-8")
        products = _save_static_plots(frame, route_dir, route, flight_id)
        products.append(_save_animation(frame, route_dir, route, flight_id))
        payload = {
            "version": "3.2",
            "route": route,
            "flight": flight_id,
            "rows": int(len(frame)),
            "duration_seconds": float(frame["time"].max() - frame["time"].min()),
            "common_origin": origin,
            "coordinate_definition": {
                "position_x_m": "共同参考点向东为正，单位m",
                "position_y_m": "共同参考点向北为正，单位m",
                "position_z_m": "相对共同参考高度，单位m",
            },
            "wind_speed_mps_mean": float(frame["wind_speed"].mean()),
            "wind_angle_deg_mean": float(frame["wind_angle"].mean()),
            "actual_energy_wh": float(frame["actual_cumulative_energy_wh"].iloc[-1]),
            "tcn_energy_wh": float(frame["tcn_cumulative_energy_wh"].iloc[-1]),
            "rls_energy_wh": float(frame["predicted_cumulative_energy_wh"].iloc[-1]),
            "products": products + [str(aligned_path)],
        }
        summary[route] = payload
        save_json(route_dir / f"flight_{flight_id}_summary.json", payload)

    comparison_path = _save_all_routes_comparison(frames, output_dir)
    save_json(
        output_dir / "route_products_summary_3.2.json",
        {
            "version": "3.2",
            "common_origin": origin,
            "representative_flights": selected,
            "comparison_figure": str(comparison_path),
            "routes": summary,
        },
    )
    return summary


if __name__ == "__main__":
    result = generate_route_products()
    for route_name, payload in result.items():
        print(route_name, payload["flight"], len(payload["products"]), "products")
