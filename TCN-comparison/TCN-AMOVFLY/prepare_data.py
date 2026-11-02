# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: prepare_data.py
# 开发时间: 2026-11-02
# 文件名: prepare_data.py
# 功能说明: 从AMOVFLY原始完整记录生成DJI同标准状态及机型编码加工表
# 版本号：1.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


from config import (RAW_FEATURE_COLUMNS, SCHEMA_VERSION, MAX_GAP_SECONDS,
                    MOTOR_ON_POWER_THRESHOLD_W, AIRBORNE_HORIZONTAL_THRESHOLD_M,
                    AIRBORNE_VERTICAL_THRESHOLD_M, ExperimentConfig, build_config)


def _continuous_stage(evidence: np.ndarray) -> np.ndarray:
    """功能: 按DJI标准填充首末证据间连续状态。
    参数: evidence为原始采样点布尔证据。
    返回: 仅含0和1的状态数组。
    调用位置: _read_ready_file。
    """
    result = np.zeros(len(evidence), dtype=np.int8)
    indices = np.flatnonzero(evidence)
    if len(indices):
        result[indices[0]:indices[-1] + 1] = 1
    return result


def _payload_from_name(name: str) -> float:
    """功能: 从AMOVFLY文件名读取载荷克数。
    参数: name为飞行文件名。
    返回: 载荷质量，单位为克。
    调用位置: _read_ready_file。
    """
    match = re.search(r"_P(\d+)", name)
    return float(match.group(1)) if match else 0.0


def _read_ready_file(path: Path, step_seconds: float = 0.2) -> tuple[pd.DataFrame, dict]:
    """功能: 清洗单个ready飞行文件并重采样到0.2秒网格。
    参数: path为ready CSV路径。
    返回: 加工数据和该文件的审计摘要。
    调用位置: build_dataset。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rows_original = len(frame)
    required = ["time", "wind_speed", "wind_angle", "gps_x", "gps_y", "gps_z",
                "v_x", "v_y", "v_z", "battery_voltage", "battery_current"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}缺少字段: {missing}")
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    rows_finite = len(frame)
    frame = frame.sort_values("time", kind="stable").drop_duplicates("time", keep="first")
    rows_unique = len(frame)
    if len(frame) < 20:
        raise ValueError(f"{path.name}有效样本不足")
    # 保留完整记录，避免裁剪掉停机和原地启动阶段。
    time_values = frame.time.to_numpy(float)
    gaps = np.diff(time_values)
    if len(gaps) and float(gaps.max()) > MAX_GAP_SECONDS:
        raise ValueError(f"{path.name}存在{gaps.max():.3f}s长断点")
    time_s = time_values - time_values[0]
    duration = float(time_s[-1])
    grid = np.arange(0.0, duration, step_seconds)
    if len(grid) < 2:
        raise ValueError(f"{path.name}时间范围不足")

    wind_angle = np.deg2rad(frame.wind_angle.to_numpy(float))
    relative_position = frame[["gps_x", "gps_y", "gps_z"]].to_numpy(float, copy=True)
    relative_position -= relative_position[0]
    payload = _payload_from_name(path.name)
    source_power = np.maximum(frame.battery_voltage.to_numpy(float) * frame.battery_current.to_numpy(float), 0.0)
    motor_source = _continuous_stage(source_power >= MOTOR_ON_POWER_THRESHOLD_W)
    airborne_evidence = (np.hypot(relative_position[:, 0], relative_position[:, 1]) >= AIRBORNE_HORIZONTAL_THRESHOLD_M
                         ) | (np.abs(relative_position[:, 2]) >= AIRBORNE_VERTICAL_THRESHOLD_M)
    airborne_source = _continuous_stage(airborne_evidence) * motor_source
    match = re.match(r"Uav([YRG])_", path.stem)
    if match is None:
        raise ValueError(f"{path.name}未知机型")
    uav_type = {"Y": 1, "R": 2, "G": 3}[match.group(1)]
    raw_energy_wh = float(np.trapezoid(source_power, time_s) / 3600.0)
    values = {
        "time_s": time_s,
        "position_x_m": relative_position[:, 0],
        "position_y_m": relative_position[:, 1],
        "position_z_m": relative_position[:, 2],
        "wind_sensor_sin_mps": frame.wind_speed.to_numpy(float) * np.sin(wind_angle),
        "wind_sensor_cos_mps": frame.wind_speed.to_numpy(float) * np.cos(wind_angle),
        "payload_g": np.full(len(frame), payload, dtype=float),
        "speed_mps": np.sqrt((frame[["v_x", "v_y", "v_z"]].to_numpy(float) ** 2).sum(axis=1)),
        "vertical_speed_mps": frame.v_z.to_numpy(float),
        "uav_type": np.full(len(frame), uav_type, dtype=np.int8),
        "power_w": source_power,
    }
    output = pd.DataFrame({name: np.interp(grid, time_s, data) for name, data in values.items()})
    output["time_s"] = grid
    output["time"] = grid
    output["task_duration_s"] = duration
    # 与DJI相同，分别插值并四舍五入二进制状态，再合并成0/1/2，禁止直接插值三级标签。
    output["planned_motor_on"] = np.rint(np.interp(grid, time_s, motor_source)).astype(np.int8)
    output["planned_airborne"] = np.rint(np.interp(grid, time_s, airborne_source)).astype(np.int8)
    output["planned_airborne"] = np.minimum(output.planned_airborne, output.planned_motor_on)
    output["planned"] = output.planned_motor_on + output.planned_airborne
    output["uav_type"] = uav_type
    output["power_w"] = np.maximum(output.power_w, 0.0)
    output["dt_seconds"] = step_seconds
    output.loc[output.index[-1], "dt_seconds"] = duration - grid[-1]
    # 用实际最后残余区间替代整步后，再次守恒校准。
    resampled_energy_wh = float((output.power_w * output.dt_seconds / 3600.0).sum())
    if raw_energy_wh > 0 and resampled_energy_wh <= 0:
        raise ValueError(f"{path.name}无法进行能量守恒重采样")
    output["power_w"] *= raw_energy_wh / resampled_energy_wh if resampled_energy_wh > 0 else 0.0
    output["energy_interval_wh"] = output.power_w * output.dt_seconds / 3600.0
    return output, {
        "source_file": str(path), "rows_source": int(len(frame)), "rows_resampled": int(len(output)),
        "rows_original": rows_original, "rows_invalid": rows_original - rows_finite,
        "rows_duplicate_time": rows_finite - rows_unique, "rows_trimmed": rows_unique - len(frame),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_s": duration, "raw_energy_wh": raw_energy_wh,
        "resampled_energy_wh": float(output.energy_interval_wh.sum()),
        "max_source_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "uav_type": uav_type,
        "status_quality_flag": "ok" if airborne_source.any() else "no_airborne_evidence" if motor_source.any() else "no_motor_on_evidence",
        **{f"planned_{state}_rows": int(output.planned.eq(state).sum()) for state in (0, 1, 2)},
    }


def _simultaneous_groups(names: list[str], source_dir: Path) -> dict[str, int]:
    """功能: 将Multi-UAV清单中的同时飞行合并为同一拆分组。
    参数: names为ready文件名列表。
    返回: 文件名到分组编号的映射。
    调用位置: build_dataset。
    """
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        if left in parent and right in parent:
            parent[find(right)] = find(left)

    sheet = source_dir / "Multi-Uav_infosheet.csv"
    if sheet.exists():
        info = pd.read_csv(sheet, encoding="utf-8-sig")
        for _, row in info.iterrows():
            members = [str(row[column]) + ".csv" for column in ("Uav1", "Uav2", "Uav3")
                       if column in row and str(row[column]) not in {"/", "nan"}]
            members = [member for member in members if member in parent]
            for member in members[1:]:
                union(members[0], member)
    roots = {root: index for index, root in enumerate(sorted({find(name) for name in names}))}
    return {name: roots[find(name)] for name in names}


def build_dataset(cfg: ExperimentConfig | None = None) -> dict:
    """功能: 生成AMOVFLY的训练、验证、测试加工表和审计元数据。
    参数: 无，路径与参数由本文件常量集中管理。
    返回: 数据规模、拆分和清洗摘要。
    调用位置: 脚本直接运行。
    """
    cfg = cfg or build_config()
    source_dir, output_dir = cfg.source_processed_dir, cfg.data_dir
    ready_files = [path for scene in ("FAFS", "FAVS", "VAFS", "VAVS", "Random")
                   for path in (source_dir / scene / scene).glob("*.csv")]
    processed, audits, rejected = [], [], []
    for path in sorted(ready_files):
        try:
            frame, audit = _read_ready_file(path, cfg.resample_seconds)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            rejected.append({"source_file": str(path), "reason": str(exc)})
            continue
        scene = path.parent.name
        frame.insert(0, "route", scene)
        frame.insert(0, "flight_name", path.stem)
        processed.append(frame)
        audits.append({"flight_name": path.stem, "route": scene, **audit})
    if len(processed) < 30:
        raise RuntimeError(f"有效飞行数量过少: {len(processed)}")

    names = [frame.flight_name.iloc[0] + ".csv" for frame in processed]
    group_map = _simultaneous_groups(names, source_dir)
    split_rng = np.random.default_rng(cfg.random_seed)
    unique_groups = np.array(sorted(set(group_map.values())), dtype=int)
    split_rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_train = max(1, int(round(n_groups * 0.70)))
    n_val = max(1, int(round(n_groups * 0.15)))
    split_by_group = {int(group): ("train" if index < n_train else "val" if index < n_train + n_val else "test")
                      for index, group in enumerate(unique_groups)}
    validation_groups = sorted(group for group, split in split_by_group.items() if split == "val")
    calibration_groups = set(validation_groups[1::2])
    output_frames = {"train": [], "val": [], "test": []}
    flight_id = 0
    audit_by_name = {row["flight_name"]: row for row in audits}
    for frame in processed:
        name = frame.flight_name.iloc[0]
        split = split_by_group[group_map[name + ".csv"]]
        frame = frame.copy()
        frame.insert(0, "flight", flight_id)
        frame.insert(2, "split", split)
        role = "calibration" if group_map[name + ".csv"] in calibration_groups else split
        frame["split_role"] = role
        output_frames[split].append(frame)
        audit_by_name[name].update({"flight": flight_id, "split": split, "split_role": role,
                                   "simultaneous_group": int(group_map[name + ".csv"])})
        flight_id += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frames in output_frames.items():
        table = pd.concat(frames, ignore_index=True)
        columns = ["flight", "flight_name", "route", "split", "split_role", "time", "time_s", "task_duration_s",
                   "position_x_m", "position_y_m", "position_z_m",
                   "wind_sensor_sin_mps", "wind_sensor_cos_mps", "payload_g", "speed_mps",
                   "vertical_speed_mps", "uav_type", "planned", "planned_motor_on", "planned_airborne",
                   "dt_seconds", "power_w", "energy_interval_wh"]
        table[columns].to_csv(output_dir / f"{split}_TCN-comparison.csv", index=False, encoding="utf-8")
    metadata = {
        "version": SCHEMA_VERSION, "source_directory": str(source_dir),
        "output_directory": str(output_dir), "resample_seconds": cfg.resample_seconds,
        "max_gap_seconds": MAX_GAP_SECONDS, "random_seed": cfg.random_seed,
        "feature_columns": list(RAW_FEATURE_COLUMNS),
        "source_files": len(ready_files),
        "uav_type_mapping": {"Y": 1, "R": 2, "G": 3},
        "planned_mapping": {"0": "停机", "1": "电机运行但未飞行", "2": "飞行"},
        "status_labeling": {"motor_on_power_threshold_w": MOTOR_ON_POWER_THRESHOLD_W,
                            "airborne_horizontal_threshold_m": AIRBORNE_HORIZONTAL_THRESHOLD_M,
                            "airborne_vertical_threshold_m": AIRBORNE_VERTICAL_THRESHOLD_M,
                            "method": "原始功率>=100W首末证据填充motor；相对起点水平>=1m或垂直绝对值>=0.5m首末证据填充airborne；airborne*=motor；分别插值取整后planned=motor+airborne"},
        "state_counts": {split: {str(state): int(sum(frame.planned.eq(state).sum() for frame in frames))
                                 for state in (0, 1, 2)} for split, frames in output_frames.items()},
        "target_columns": ["power_w", "energy_interval_wh"],
        "input_dimensions": {"A_B": 10, "C_D_E": 15},
        "cleaning": {"nonfinite_and_duplicate_time": "drop", "long_gap": f"reject > {MAX_GAP_SECONDS}s",
                     "position": "relative to first valid sample; complete source record without altitude trimming", "power": "max(battery_voltage*battery_current, 0)",
                     "energy": "raw trapezoid energy used to scale resampled power"},
        "split_policy": "complete flight and Multi-UAV simultaneous group, seeded 70/15/15",
        "counts": {split: {"rows": int(sum(len(frame) for frame in frames)),
                             "flights": int(len(frames))} for split, frames in output_frames.items()},
        "accepted_flights": int(len(processed)), "rejected_flights": rejected,
        "audit_files": ["amovfly_flight_audit.csv", "dataset_summary_TCN-comparison.json"],
    }
    metadata.update({f"{split}_{key}": value
                     for split, counts in metadata["counts"].items() for key, value in counts.items()})
    metadata["split_flights"] = {split: [int(frame.flight.iloc[0]) for frame in frames]
                                 for split, frames in output_frames.items()}
    metadata["split_overlap"] = False
    metadata["validation_roles"] = {role: [row["flight"] for row in audits if row["split_role"] == role]
                                     for role in ("val", "calibration")}
    pd.DataFrame(audits).to_csv(output_dir / "amovfly_flight_audit.csv", index=False, encoding="utf-8")
    for name in ("train_TCN-comparison.csv", "val_TCN-comparison.csv", "test_TCN-comparison.csv",
                 "amovfly_flight_audit.csv"):
        path = output_dir / name
        if path.exists():
            metadata.setdefault("sha256", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "feature_metadata_TCN-comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata["sha256"]["feature_metadata_TCN-comparison.json"] = hashlib.sha256(
        (output_dir / "feature_metadata_TCN-comparison.json").read_bytes()).hexdigest()
    (output_dir / "dataset_summary_TCN-comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build_dataset(), ensure_ascii=False, indent=2))
