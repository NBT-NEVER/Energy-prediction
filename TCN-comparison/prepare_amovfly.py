# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: prepare_amovfly.py
# 开发时间: 2026-11-01
# 文件名: prepare_amovfly.py
# 功能说明: 清洗AMOVFLY飞行数据并生成TCN对比实验加工表
# 版本号：1.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:/Python-files/Energy-prediction/data/AMOVFLY-Dataset")
SOURCE_DIR = DATA_ROOT / "sourse"
OUTPUT_DIR = DATA_ROOT / "TCN-comparison"
STEP_SECONDS = 0.2
MAX_GAP_SECONDS = 2.0
RANDOM_SEED = 42


def _payload_from_name(name: str) -> float:
    """功能: 从AMOVFLY文件名读取载荷克数。
    参数: name为飞行文件名。
    返回: 载荷质量，单位为克。
    调用位置: _read_ready_file。
    """
    match = re.search(r"_P(\d+)", name)
    return float(match.group(1)) if match else 0.0


def _read_ready_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """功能: 清洗单个ready飞行文件并重采样到0.2秒网格。
    参数: path为ready CSV路径。
    返回: 加工数据和该文件的审计摘要。
    调用位置: build_dataset。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rows_original = len(frame)
    required = ["time", "wind_speed", "wind_angle", "gps_x", "gps_y", "gps_z",
                "v_x", "v_y", "v_z", "power"]
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
    # 仅保留高度超过2米的首末证据之间，裁剪依据不使用功率标签。
    airborne_indices = np.flatnonzero(frame.gps_z.to_numpy(float) > 2.0)
    if len(airborne_indices) < 20:
        raise ValueError(f"{path.name}缺少有效离地航段")
    frame = frame.iloc[airborne_indices[0]:airborne_indices[-1] + 1].copy()
    time_values = frame.time.to_numpy(float)
    gaps = np.diff(time_values)
    if len(gaps) and float(gaps.max()) > MAX_GAP_SECONDS:
        raise ValueError(f"{path.name}存在{gaps.max():.3f}s长断点")
    time_s = time_values - time_values[0]
    duration = float(time_s[-1])
    grid = np.arange(0.0, duration, STEP_SECONDS)
    if len(grid) < 2:
        raise ValueError(f"{path.name}时间范围不足")

    wind_angle = np.deg2rad(frame.wind_angle.to_numpy(float))
    relative_position = frame[["gps_x", "gps_y", "gps_z"]].to_numpy(float, copy=True)
    relative_position -= relative_position[0]
    payload = _payload_from_name(path.name)
    source_power = np.maximum(frame.power.to_numpy(float), 0.0)
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
        "uav_R": np.full(len(frame), float(path.stem.startswith("UavR_"))),
        "uav_G": np.full(len(frame), float(path.stem.startswith("UavG_"))),
        "power_w": source_power,
    }
    output = pd.DataFrame({name: np.interp(grid, time_s, data) for name, data in values.items()})
    output["time_s"] = grid
    output["time"] = grid
    output["task_duration_s"] = duration
    output["power_w"] = np.maximum(output.power_w, 0.0)
    resampled_energy_wh = float(np.sum(output.power_w * STEP_SECONDS / 3600.0))
    if raw_energy_wh > 0 and resampled_energy_wh > 0:
        output["power_w"] *= raw_energy_wh / resampled_energy_wh
    output["dt_seconds"] = STEP_SECONDS
    output.loc[output.index[-1], "dt_seconds"] = duration - grid[-1]
    # 用实际最后残余区间替代整步后，再次守恒校准。
    output["power_w"] *= raw_energy_wh / float((output.power_w * output.dt_seconds / 3600.0).sum())
    output["energy_interval_wh"] = output.power_w * output.dt_seconds / 3600.0
    return output, {
        "source_file": str(path), "rows_source": int(len(frame)), "rows_resampled": int(len(output)),
        "rows_original": rows_original, "rows_invalid": rows_original - rows_finite,
        "rows_duplicate_time": rows_finite - rows_unique, "rows_trimmed": rows_unique - len(frame),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_s": duration, "raw_energy_wh": raw_energy_wh,
        "resampled_energy_wh": float(output.energy_interval_wh.sum()),
        "max_source_gap_s": float(gaps.max()) if len(gaps) else 0.0,
    }


def _simultaneous_groups(names: list[str]) -> dict[str, int]:
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

    sheet = SOURCE_DIR / "Multi-Uav_infosheet.csv"
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


def build_dataset() -> dict:
    """功能: 生成AMOVFLY的训练、验证、测试加工表和审计元数据。
    参数: 无，路径与参数由本文件常量集中管理。
    返回: 数据规模、拆分和清洗摘要。
    调用位置: 脚本直接运行。
    """
    ready_files = [path for scene in ("FAFS", "FAVS", "VAFS", "VAVS", "Random")
                   for path in (SOURCE_DIR / scene / scene).glob("*.csv")]
    processed, audits, rejected = [], [], []
    for path in sorted(ready_files):
        try:
            frame, audit = _read_ready_file(path)
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
    group_map = _simultaneous_groups(names)
    split_rng = np.random.default_rng(RANDOM_SEED)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, frames in output_frames.items():
        table = pd.concat(frames, ignore_index=True)
        columns = ["flight", "flight_name", "route", "split", "split_role", "time", "time_s", "task_duration_s",
                   "position_x_m", "position_y_m", "position_z_m",
                   "wind_sensor_sin_mps", "wind_sensor_cos_mps", "payload_g", "speed_mps",
                   "vertical_speed_mps", "uav_R", "uav_G",
                   "dt_seconds", "power_w", "energy_interval_wh"]
        table[columns].to_csv(OUTPUT_DIR / f"{split}_TCN-comparison.csv", index=False, encoding="utf-8")
    metadata = {
        "version": "AMOVFLY-TCN-comparison-1.0", "source_directory": str(SOURCE_DIR),
        "output_directory": str(OUTPUT_DIR), "resample_seconds": STEP_SECONDS,
        "max_gap_seconds": MAX_GAP_SECONDS, "random_seed": RANDOM_SEED,
        "feature_columns": ["position_x_m", "position_y_m", "position_z_m", "wind_sensor_sin_mps",
                             "wind_sensor_cos_mps", "payload_g", "speed_mps", "vertical_speed_mps", "uav_R", "uav_G"],
        "target_columns": ["power_w", "energy_interval_wh"],
        "input_dimensions": {"A_B": 10, "C_D_E": 15},
        "cleaning": {"nonfinite_and_duplicate_time": "drop", "long_gap": f"reject > {MAX_GAP_SECONDS}s",
                     "position": "relative to first sample", "power": "max(V*I, 0) from ready power column",
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
    pd.DataFrame(audits).to_csv(OUTPUT_DIR / "amovfly_flight_audit.csv", index=False, encoding="utf-8")
    for name in ("train_TCN-comparison.csv", "val_TCN-comparison.csv", "test_TCN-comparison.csv",
                 "amovfly_flight_audit.csv"):
        path = OUTPUT_DIR / name
        if path.exists():
            metadata.setdefault("sha256", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUTPUT_DIR / "feature_metadata_TCN-comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata["sha256"]["feature_metadata_TCN-comparison.json"] = hashlib.sha256(
        (OUTPUT_DIR / "feature_metadata_TCN-comparison.json").read_bytes()).hexdigest()
    (OUTPUT_DIR / "dataset_summary_TCN-comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build_dataset(), ensure_ascii=False, indent=2))
