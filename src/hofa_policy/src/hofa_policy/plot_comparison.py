"""Generate publication-style comparisons for MPC, BC, and DAgger runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("mpc", "HOFA-MPC", "#1b9e77", "-"),
    ("bc", "Pure BC", "#d95f02", "--"),
    ("dagger", "BC + DAgger", "#7570b3", "-."),
)
TRACK_ORDER = (
    "test_00_line",
    "test_01_s_curve",
    "test_02_double_s",
    "test_03_lateral_shift",
    "test_04_chicane",
)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def latest_run(root, track_id):
    root = Path(root)
    direct = sorted(root.glob("run_*"))
    if direct:
        return direct[-1]
    candidates = sorted((root / track_id).glob("run_*"))
    return candidates[-1] if candidates else None


def load_run(root, track_id):
    run = latest_run(root, track_id)
    if run is None or not (run / "metrics.json").is_file():
        return None
    odometry = read_csv(run / "odometry.csv")
    reference = read_csv(run / "evaluation_reference.csv")
    path = read_csv(run / "global_reference_path.csv")
    metrics = read_json(run / "metrics.json")
    return {
        "run": str(run),
        "odometry": odometry,
        "reference": reference,
        "path": path,
        "metrics": metrics,
    }


def array(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def project_error(odometry, path):
    points = np.asarray(
        [[float(row["x_ned"]), float(row["y_ned"])] for row in path],
        dtype=np.float64)
    signed, distance, arc = [], [], []
    path_arc = np.zeros(len(points), dtype=np.float64)
    if len(points) > 1:
        path_arc[1:] = np.cumsum(np.linalg.norm(
            np.diff(points, axis=0), axis=1))
    for row in odometry:
        p = np.asarray([float(row["x"]), float(row["y"])])
        best_distance = float("inf")
        best_signed = 0.0
        best_arc = 0.0
        for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
            segment = b - a
            length_sq = float(segment @ segment)
            if length_sq <= 1.0e-12:
                continue
            ratio = float(np.clip(((p - a) @ segment) / length_sq, 0.0, 1.0))
            projection = a + ratio * segment
            error = p - projection
            norm = float(np.linalg.norm(error))
            if norm < best_distance:
                best_distance = norm
                best_arc = float(path_arc[index] +
                                 ratio * (path_arc[index + 1] -
                                          path_arc[index]))
                best_signed = float(
                    (segment[0] * error[1] - segment[1] * error[0]) /
                    max(np.sqrt(length_sq), 1.0e-12))
        signed.append(best_signed)
        distance.append(best_distance)
        arc.append(best_arc)
    return np.asarray(signed), np.asarray(distance), np.asarray(arc)


def time_series(run):
    odom = run["odometry"]
    reference = run["reference"]
    path = run["path"]
    t = array(odom, "t_rel")
    speed = array(odom, "speed_world")
    speed_ref = array(reference, "speed_ref")
    scheduled_arc = array(reference, "scheduled_arc_m")
    actual_arc = array(reference, "actual_arc_m")
    unique_arc, unique_index = np.unique(scheduled_arc, return_index=True)
    unique_time = t[unique_index]
    planned_at_actual = np.interp(
        actual_arc, unique_arc, unique_time,
        left=unique_time[0], right=unique_time[-1])
    lag = t - planned_at_actual
    signed, distance, _ = project_error(odom, path)
    return {
        "t": t,
        "t_norm": t / max(float(t[-1]), 1.0e-9),
        "x": array(odom, "x"),
        "y": array(odom, "y"),
        "speed": speed,
        "speed_ref": speed_ref,
        "cross_signed": signed,
        "cross_distance": distance,
        "schedule_lag": lag,
    }


def metric_summary(runs):
    metrics = {}
    keys = (
        ("cross_track_rmse_m", "cross_track_error_m", "rmse"),
        ("yaw_rmse_deg", "yaw_error_deg", "rmse"),
        ("schedule_lag_rmse_s", "schedule_lag_s", "rmse"),
        ("completion_ratio", "completion_ratio", None),
        ("final_distance_m", "final_distance_to_goal_m", None),
        ("callback_p95_ms", "callback_time_ms", "p95"),
    )
    for method, _, _, _ in METHODS:
        values = []
        for track in TRACK_ORDER:
            run = runs.get((method, track))
            if run is None:
                continue
            source = run["metrics"]
            for output_key, metric_key, nested_key in keys:
                value = source.get(metric_key)
                if nested_key:
                    value = value.get(nested_key) if value else None
                if value is not None:
                    metrics.setdefault((method, output_key), []).append(
                        float(value))
    summary = []
    for (method, key), values in metrics.items():
        summary.append({
            "method": method,
            "metric": key,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        })
    return summary


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_xy(runs, output):
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8),
                             constrained_layout=True)
    for axis, track in zip(axes, TRACK_ORDER):
        plotted = False
        for method, label, color, style in METHODS:
            run = runs.get((method, track))
            if run is None:
                continue
            series = time_series(run)
            axis.plot(series["x"], series["y"], color=color,
                      linestyle=style, linewidth=1.35, label=label)
            plotted = True
        if plotted:
            path = next(
                run["path"] for (method, name), run in runs.items()
                if name == track and run is not None)
            axis.plot(array(path, "x_ned"), array(path, "y_ned"),
                      color="0.35", linewidth=0.8, alpha=0.7,
                      label="reference")
        axis.set_title(track.replace("test_", "").replace("_", " "))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("NED y [m]")
    for axis in axes:
        axis.set_xlabel("NED x [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.08))
    save_figure(fig, output, "comparison_xy")


def plot_errors(runs, output):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0),
                             sharex=True, constrained_layout=True)
    for method, label, color, style in METHODS:
        curves = []
        for track in TRACK_ORDER:
            run = runs.get((method, track))
            if run is not None:
                series = time_series(run)
                curves.append(series)
        if not curves:
            continue
        # Resample each trajectory to a common normalized-time grid.
        grid = np.linspace(0.0, 1.0, 301)
        for axis, key, ylabel in zip(
                axes, ("cross_distance", "schedule_lag", "speed"),
                ("cross-track [m]", "schedule lag [s]", "speed [m/s]")):
            values = np.asarray([
                np.interp(grid, item["t_norm"], item[key])
                for item in curves])
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            axis.plot(grid, mean, color=color, linestyle=style,
                      linewidth=1.5, label=label)
            axis.fill_between(grid, mean - std, mean + std,
                              color=color, alpha=0.10, linewidth=0)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22, linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    # Add the reference speed profile when available.
    ref_run = runs.get(("mpc", TRACK_ORDER[0]))
    if ref_run is not None:
        series = time_series(ref_run)
        grid = np.linspace(0.0, 1.0, 301)
        ref_speed = np.interp(grid, series["t_norm"], series["speed_ref"])
        axes[2].plot(grid, ref_speed, color="0.25", linestyle=":",
                     linewidth=1.1, label="reference speed")
    axes[-1].set_xlabel("normalized trajectory time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    save_figure(fig, output, "comparison_errors")


def plot_summary(summary, output):
    metrics = [
        ("cross_track_rmse_m", "Cross-track RMSE [m]"),
        ("yaw_rmse_deg", "Yaw RMSE [deg]"),
        ("schedule_lag_rmse_s", "Schedule lag RMSE [s]"),
        ("callback_p95_ms", "Callback P95 [ms]"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.0, 2.9),
                             constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        x = np.arange(len(METHODS))
        means, errors, labels, colors = [], [], [], []
        for method, label, color, _ in METHODS:
            row = next((item for item in summary
                        if item["method"] == method and
                        item["metric"] == metric), None)
            means.append(row["mean"] if row else np.nan)
            errors.append(row["std"] if row else 0.0)
            labels.append(label)
            colors.append(color)
        axis.bar(x, means, yerr=errors, color=colors, alpha=0.88,
                 capsize=3, width=0.68, linewidth=0.5,
                 edgecolor="black", error_kw={"linewidth": 0.8})
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(["MPC", "BC", "DAgger"], rotation=35,
                             ha="right")
        axis.grid(axis="y", alpha=0.22, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure(fig, output, "comparison_summary")


def plot_wrench(runs, output):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.5),
                             sharex=True, constrained_layout=True)
    for method, label, color, style in METHODS:
        run = runs.get(("mpc", TRACK_ORDER[2])) if method == "mpc" else \
            runs.get((method, TRACK_ORDER[2]))
        if run is None:
            continue
        filename = Path(run["run"]) / (
            "dagger_teacher_generalized_force.csv"
            if method == "mpc" else "generalized_force.csv")
        if not filename.is_file():
            filename = Path(run["run"]) / "generalized_force.csv"
        rows = read_csv(filename)
        t = array(rows, "stamp")
        t = t - t[0]
        for axis, key, ylabel in zip(
                axes, ("fx", "fy", "nz"), ("Fx [N]", "Fy [N]", "Nz [N m]")):
            axis.plot(t, array(rows, key), color=color, linestyle=style,
                      linewidth=1.15, label=label)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22, linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    save_figure(fig, output, "comparison_wrench")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpc-root", required=True)
    parser.add_argument("--bc-root", required=True)
    parser.add_argument("--dagger-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    roots = {"mpc": args.mpc_root, "bc": args.bc_root,
             "dagger": args.dagger_root}
    runs = {}
    for method, _, _, _ in METHODS:
        for track in TRACK_ORDER:
            run = load_run(roots[method], track)
            if run is not None:
                runs[(method, track)] = run
    if not runs:
        raise RuntimeError("no complete runs found")
    summary = metric_summary(runs)
    with (output / "comparison_metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    with (output / "comparison_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("method", "metric",
                                            "mean", "std", "values"))
        writer.writeheader()
        writer.writerows(summary)
    plot_xy(runs, output)
    plot_errors(runs, output)
    plot_summary(summary, output)
    plot_wrench(runs, output)
    print(json.dumps({
        "output": str(output),
        "runs": len(runs),
        "figures": [
            "comparison_xy", "comparison_errors",
            "comparison_summary", "comparison_wrench",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
