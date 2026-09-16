# _*_coding:UTF-8_ *_
# 开发者: NBT
# 文件名: visualize.py
# 开发时间: 2026-09-16
# 功能说明: 生成实验4.0训练、结果、预测、RLS和路线追踪SVG/GIF产物
# 版本号：4.0

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from config import ExperimentConfig, ensure_directories
from model import DynamicEnergyRLS
from train import predict_frame


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    return str(path)


def _write_trace(cfg: ExperimentConfig, predictions: pd.DataFrame) -> pd.DataFrame:
    tuning = pd.read_csv(cfg.rls_tuning_csv).sort_values("flight_energy_wape_percent").iloc[0]
    factor, covariance = float(tuning.forgetting_factor), float(tuning.initial_covariance)
    steps = max(1, int(round(cfg.rls_window_seconds / cfg.resample_seconds)))
    rows = []
    for flight, group in predictions.sort_values(["flight", "time"]).groupby("flight", sort=False):
        rls = DynamicEnergyRLS(factor, covariance)
        group = group.reset_index(drop=True)
        for index in range(0, len(group), steps):
            window = group.iloc[index:index + steps]
            baseline = float(window.tcn_predicted_energy_wh.sum())
            actual = float(window.actual_energy_wh.sum())
            before = rls.state()
            rls.update_window(baseline, actual)
            state = rls.state()
            rows.append({"flight": flight, "route": str(window.route.iloc[0]), "window_index": index // steps,
                         "baseline_energy_wh": baseline, "actual_energy_wh": actual,
                         "residual_wh": actual - baseline, "theta_bias_before": before["theta_bias"],
                         "theta_scale_before": before["theta_scale"], "theta_bias": state["theta_bias"],
                         "theta_scale": state["theta_scale"], "residual_ema": state["residual_ema"],
                         "residual_std": state["residual_std"], "covariance_trace": float(np.trace(np.asarray(state["covariance"]))),
                         "update_count": state["update_count"]})
    trace = pd.DataFrame(rows)
    trace.to_csv(cfg.out_rls_dir / "rls_parameter_trace_4.0.csv", index=False, encoding="utf-8")
    stats = trace.groupby("route").agg(windows=("window_index", "size"), bias_mean=("theta_bias", "mean"),
                                        bias_std=("theta_bias", "std"), scale_mean=("theta_scale", "mean"),
                                        scale_std=("theta_scale", "std"), residual_std_mean=("residual_std", "mean"),
                                        covariance_trace_mean=("covariance_trace", "mean")).reset_index()
    stats.to_csv(cfg.out_rls_dir / "rls_parameter_statistics_4.0.csv", index=False, encoding="utf-8")
    summary = {"version": "4.0", "forgetting_factor": factor, "initial_covariance": covariance,
               "updates": int(len(trace)), "bias_range": [float(trace.theta_bias.min()), float(trace.theta_bias.max())],
               "scale_range": [float(trace.theta_scale.min()), float(trace.theta_scale.max())]}
    (cfg.out_rls_dir / "rls_parameter_summary_4.0.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def _route_products(cfg: ExperimentConfig, predictions: pd.DataFrame) -> list[str]:
    output = []
    route_summary = []
    _style()
    for route, group in predictions.groupby("route", sort=True):
        flight = group.flight.iloc[0]
        group = group[group.flight == flight].sort_values("time").reset_index(drop=True)
        directory = cfg.out_route_dir / f"route_{route}" / f"flight_{flight}"
        directory.mkdir(parents=True, exist_ok=True)
        aligned = group[["flight", "route", "time", "planned_speed_mps", "planned_altitude_m", "cumulative_distance_m",
                         "power_w", "tcn_predicted_power_w", "predicted_power_w", "actual_energy_wh", "predicted_energy_wh"]].copy()
        aligned["trajectory_x_m"] = aligned["cumulative_distance_m"]
        aligned["trajectory_y_m"] = 0.0
        aligned["trajectory_z_m"] = aligned["planned_altitude_m"]
        aligned.to_csv(directory / f"flight_{flight}_planning_energy.csv", index=False, encoding="utf-8")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(group.time, group.power_w, label="actual power", linewidth=0.8)
        axes[0].plot(group.time, group.predicted_power_w, label="RLS power", linewidth=0.8)
        axes[0].plot(group.time, group.tcn_predicted_power_w, label="TCN power", linewidth=0.8, alpha=0.7)
        axes[0].set_ylabel("Power (W)"); axes[0].set_title(f"Route {route} | Flight {flight} | Power")
        axes[0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[1].plot(group.time, group.actual_energy_wh.cumsum(), label="actual cumulative energy")
        axes[1].plot(group.time, group.predicted_energy_wh.cumsum(), label="RLS cumulative energy")
        axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Energy (Wh)"); axes[1].legend(fontsize=8)
        output.append(_save(fig, directory / f"flight_{flight}_power_energy.svg"))
        frames = []
        sample = aligned.iloc[np.linspace(0, len(aligned) - 1, min(24, len(aligned))).astype(int)]
        for end in range(1, len(sample) + 1):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sample.trajectory_x_m.iloc[:end], sample.trajectory_z_m.iloc[:end], color="#1565c0", linewidth=2)
            ax.scatter(sample.trajectory_x_m.iloc[end - 1], sample.trajectory_z_m.iloc[end - 1], color="#d84315")
            ax.set_title(f"Route {route} | Flight {flight} | Planned trajectory")
            ax.set_xlabel("Distance along route (m)"); ax.set_ylabel("Altitude (m)")
            ax.set_xlim(sample.trajectory_x_m.min(), sample.trajectory_x_m.max() + 1e-6)
            ax.set_ylim(sample.trajectory_z_m.min() - 1, sample.trajectory_z_m.max() + 1)
            buffer = io.BytesIO(); fig.tight_layout(); fig.savefig(buffer, format="png", dpi=100); plt.close(fig)
            buffer.seek(0); frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
        gif_path = directory / f"flight_{flight}_trajectory.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
        output.append(str(gif_path))
        summary = {"route": str(route), "flight": int(flight), "rows": int(len(group)),
                   "duration_s": float(group.time.max() - group.time.min()),
                   "actual_energy_wh": float(group.actual_energy_wh.sum()),
                   "predicted_energy_wh": float(group.predicted_energy_wh.sum()),
                   "energy_error_wh": float(group.predicted_energy_wh.sum() - group.actual_energy_wh.sum()),
                   "power_energy_svg": str(directory / f"flight_{flight}_power_energy.svg"),
                   "trajectory_gif": str(gif_path)}
        (directory / f"flight_{flight}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        route_summary.append(summary)
    (cfg.out_route_dir / "route_products_summary_4.0.json").write_text(json.dumps(route_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overview = pd.DataFrame(route_summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in overview.iterrows():
        ax.scatter(row.actual_energy_wh, row.predicted_energy_wh, label=str(row.route))
        ax.annotate(str(row.route), (row.actual_energy_wh, row.predicted_energy_wh), fontsize=8)
    low = min(overview.actual_energy_wh.min(), overview.predicted_energy_wh.min()); high = max(overview.actual_energy_wh.max(), overview.predicted_energy_wh.max())
    ax.plot([low, high], [low, high], "k--", linewidth=0.8); ax.set_xlabel("Actual total energy (Wh)"); ax.set_ylabel("Predicted total energy (Wh)"); ax.set_title("All route total energy comparison")
    output.append(_save(fig, cfg.out_route_dir / "all_routes_total_energy.svg"))
    return output


def generate_all_visualizations(cfg: ExperimentConfig) -> dict:
    """参考3.2生成多目录、多角度SVG、GIF、CSV和JSON追踪产物。"""
    ensure_directories(cfg)
    for name in ("training", "results", "prediction", "custom"):
        (cfg.out_figure_dir / name).mkdir(parents=True, exist_ok=True)
    _style()
    files = []
    tuning = pd.read_csv(cfg.tcn_tuning_csv)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(tuning.window_seconds, tuning.val_wape_percent, marker="o"); ax.set_xlabel("TCN window (s)"); ax.set_ylabel("Validation WAPE (%)"); ax.set_title("TCN candidate validation WAPE"); files.append(_save(fig, cfg.out_figure_dir / "training" / "candidate_validation_wape.svg"))
    ranked = tuning.sort_values("val_wape_percent").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(np.arange(len(ranked)), ranked.val_wape_percent); ax.set_xticks(np.arange(len(ranked))); ax.set_xticklabels([f"{x:g}s" for x in ranked.window_seconds]); ax.set_xlabel("Ranked TCN window"); ax.set_ylabel("Validation WAPE (%)"); ax.set_title("TCN hyperparameter ranking"); files.append(_save(fig, cfg.out_figure_dir / "training" / "hyperparameter_ranking.svg"))
    log = pd.read_csv(cfg.training_log_csv)
    fig, ax = plt.subplots(figsize=(8, 4));
    for window, group in log.groupby("window_steps"):
        ax.plot(group.epoch, group.val_loss, label=f"{window} steps")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_title("TCN validation loss curves"); ax.legend(fontsize=8, ncol=2); files.append(_save(fig, cfg.out_figure_dir / "training" / "loss_curve_4.0.svg"))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(log.epoch, log.val_loss, ".-", alpha=.7); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_title("Training progress by recorded epoch"); files.append(_save(fig, cfg.out_figure_dir / "training" / "learning_rate_schedule.svg"))
    evaluation = pd.read_csv(cfg.evaluation_csv).iloc[0]
    metric_names = ["sample_power_w_wape_percent", "window_energy_wh_wape_percent", "flight_energy_wh_wape_percent"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(["power", "window energy", "flight energy"], [evaluation[x] for x in metric_names]); ax.set_ylabel("WAPE (%)"); ax.set_title("Evaluation metrics after RLS"); files.append(_save(fig, cfg.out_figure_dir / "results" / "evaluation_metrics.svg"))
    flight = pd.read_csv(cfg.out_model_dir / "flight_energy_summary_4.0.csv")
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(flight.actual_energy_wh, flight.predicted_energy_wh, alpha=.75); low=min(flight.actual_energy_wh.min(), flight.predicted_energy_wh.min()); high=max(flight.actual_energy_wh.max(), flight.predicted_energy_wh.max()); ax.plot([low, high],[low,high],"k--"); ax.set_xlabel("Actual flight energy (Wh)"); ax.set_ylabel("Predicted flight energy (Wh)"); ax.set_title("Flight energy actual vs predicted"); files.append(_save(fig, cfg.out_figure_dir / "results" / "flight_energy_actual_vs_predicted.svg"))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(flight.flight.astype(str), flight.predicted_energy_wh-flight.actual_energy_wh); ax.set_xlabel("Flight"); ax.set_ylabel("Prediction error (Wh)"); ax.set_title("Flight energy prediction error"); ax.tick_params(axis="x", labelrotation=90); files.append(_save(fig, cfg.out_figure_dir / "results" / "flight_energy_error.svg"))
    bins = pd.read_csv(cfg.out_model_dir / "power_bin_evaluation_4.0.csv"); fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(bins.power_bin, bins.mae_w); ax.set_xlabel("Actual power bin"); ax.set_ylabel("MAE (W)"); ax.set_title("Power-bin error"); files.append(_save(fig, cfg.out_figure_dir / "results" / "power_bin_mae.svg"))
    rls_table = pd.read_csv(cfg.rls_tuning_csv).sort_values(["initial_covariance", "forgetting_factor"])
    rls_grid = rls_table.pivot(index="initial_covariance", columns="forgetting_factor", values="flight_energy_wape_percent")
    fig, ax = plt.subplots(figsize=(8, 5)); image = ax.imshow(rls_grid.to_numpy(), aspect="auto", origin="lower"); fig.colorbar(image, ax=ax, label="Flight WAPE (%)"); ax.set_xticks(range(len(rls_grid.columns))); ax.set_xticklabels([f"{x:.2f}" for x in rls_grid.columns]); ax.set_yticks(range(len(rls_grid.index))); ax.set_yticklabels([f"{x:.2f}" for x in rls_grid.index]); ax.set_xlabel("Forgetting factor"); ax.set_ylabel("Initial covariance"); ax.set_title("RLS parameter grid (7 x 7)"); files.append(_save(fig, cfg.out_figure_dir / "results" / "rls_grid_scores_4.0.svg"))
    if cfg.task_output_csv.exists():
        task_plan = pd.read_csv(cfg.task_output_csv); fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(task_plan.time_s, task_plan.cumulative_energy_wh, label="cumulative baseline"); ax.fill_between(task_plan.time_s, task_plan.cumulative_energy_lower_wh, task_plan.cumulative_energy_upper_wh, alpha=.2, label="95% interval"); ax.set_xlabel("Task time (s)"); ax.set_ylabel("Cumulative energy (Wh)"); ax.set_title("Task-before candidate route energy plan"); ax.legend(fontsize=8); files.append(_save(fig, cfg.out_figure_dir / "custom" / "task_energy_plan_4.0.svg"))
    representative = list(pd.read_csv(cfg.predictions_csv).groupby("flight", sort=True))[:3]
    for flight_id, group in representative:
        group=group.sort_values("time"); fig, ax=plt.subplots(figsize=(10,4)); ax.plot(group.time,group.power_w,label="actual"); ax.plot(group.time,group.tcn_predicted_power_w,label="TCN"); ax.plot(group.time,group.predicted_power_w,label="RLS"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Power (W)"); ax.set_title(f"Flight {flight_id} power time series"); ax.legend(fontsize=8); files.append(_save(fig,cfg.out_figure_dir/"prediction"/f"flight_{flight_id}_power_timeseries.svg"))
    predictions = pd.read_csv(cfg.predictions_csv)
    fig, axes = plt.subplots(len(representative), 1, figsize=(10, 3 * len(representative)), squeeze=False)
    for axis, (flight_id, group) in zip(axes[:, 0], representative):
        group = group.sort_values("time")
        axis.plot(group.time, group.power_w, label="actual", linewidth=.8)
        axis.plot(group.time, group.predicted_power_w, label="RLS", linewidth=.8)
        axis.set_title(f"Flight {flight_id}"); axis.set_ylabel("Power (W)")
    axes[-1, 0].set_xlabel("Time (s)"); axes[0, 0].legend(fontsize=8)
    files.append(_save(fig, cfg.out_figure_dir / "prediction" / "flight_representative_power_timeseries.svg"))
    fig, ax = plt.subplots(figsize=(6,6)); ax.scatter(predictions.power_w,predictions.predicted_power_w,s=2,alpha=.2); ax.set_xlabel("Actual power (W)"); ax.set_ylabel("Predicted power (W)"); ax.set_title("Power prediction scatter"); files.append(_save(fig,cfg.out_figure_dir/"prediction"/"power_prediction_scatter.svg"))
    residual = predictions.predicted_power_w - predictions.power_w; fig, ax=plt.subplots(figsize=(7,4)); ax.hist(residual,bins=60); ax.set_xlabel("Power residual (W)"); ax.set_ylabel("Count"); ax.set_title("Power residual histogram"); files.append(_save(fig,cfg.out_figure_dir/"prediction"/"power_residual_histogram.svg"))
    window_table = pd.read_csv(cfg.out_model_dir / "window_energy_summary_4.0.csv")
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(window_table.actual_energy_wh, window_table.predicted_energy_wh, s=5, alpha=.25); low=min(window_table.actual_energy_wh.min(), window_table.predicted_energy_wh.min()); high=max(window_table.actual_energy_wh.max(), window_table.predicted_energy_wh.max()); ax.plot([low, high], [low, high], "k--"); ax.set_xlabel("Actual window energy (Wh)"); ax.set_ylabel("Predicted window energy (Wh)"); ax.set_title("RLS window energy actual vs predicted"); files.append(_save(fig, cfg.out_figure_dir / "prediction" / "rls_energy_window_prediction_scatter.svg"))
    window_error = window_table.predicted_energy_wh - window_table.actual_energy_wh
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(window_error, bins=60); ax.set_xlabel("Window energy residual (Wh)"); ax.set_ylabel("Count"); ax.set_title("RLS window energy residual histogram"); files.append(_save(fig, cfg.out_figure_dir / "prediction" / "rls_energy_window_residual_histogram.svg"))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(window_table.window_index, window_error, ".", markersize=1, alpha=.25); ax.axhline(0, color="k", linewidth=.8); ax.set_xlabel("Window index"); ax.set_ylabel("Energy error (Wh)"); ax.set_title("RLS window energy error by index"); files.append(_save(fig, cfg.out_figure_dir / "results" / "rls_energy_window_error_comparison.svg"))
    trace = _write_trace(cfg, predictions)
    for flight_id, group in list(trace.groupby("flight", sort=True))[:3]:
        fig, ax=plt.subplots(figsize=(9,4)); ax.plot(group.window_index,group.theta_bias,label="bias"); ax.plot(group.window_index,group.theta_scale,label="scale"); ax.set_xlabel("RLS window index"); ax.set_ylabel("Parameter value"); ax.set_title(f"Flight {flight_id} RLS parameter trace"); ax.legend(); files.append(_save(fig,cfg.out_rls_dir/f"flight_{flight_id}_rls_parameter_trace.svg"))
    route_predictions = predictions
    if cfg.clean_data_csv.exists():
        all_data = pd.read_csv(cfg.clean_data_csv)
        selected = all_data.sort_values(["route", "flight", "time"], kind="stable").groupby("route", sort=True).head(1)[["route", "flight"]]
        missing = selected.loc[~selected.route.isin(route_predictions.route.unique()), "flight"].tolist()
        if missing:
            extra = all_data[all_data.flight.isin(missing)].sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
            extra_power = predict_frame(extra, cfg)
            extra["tcn_predicted_power_w"] = extra_power
            extra["predicted_power_w"] = extra_power
            extra["predicted_power_lower_w"] = np.maximum(extra_power * 0.8, 0.0)
            extra["predicted_power_upper_w"] = extra_power * 1.2
            extra["actual_energy_wh"] = extra.power_w * extra.dt_seconds / 3600.0
            extra["predicted_energy_wh"] = extra.predicted_power_w * extra.dt_seconds / 3600.0
            extra["tcn_predicted_energy_wh"] = extra.tcn_predicted_power_w * extra.dt_seconds / 3600.0
            extra["predicted_energy_lower_wh"] = extra.predicted_power_lower_w * extra.dt_seconds / 3600.0
            extra["predicted_energy_upper_wh"] = extra.predicted_power_upper_w * extra.dt_seconds / 3600.0
            route_predictions = pd.concat([route_predictions, extra], ignore_index=True)
    files.extend(_route_products(cfg, route_predictions))
    summary = {"version":"4.0", "files":files, "svg_count":int(len([x for x in files if x.endswith('.svg')])), "gif_count":int(len([x for x in files if x.endswith('.gif')])), "no_png":True, "chart_font":"DejaVu Sans with ASCII labels to avoid glyph corruption"}
    cfg.visualization_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
