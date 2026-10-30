"""Plot artifacts produced by ``run_tracking_experiment.py``.

The module is deliberately independent of ROS so that an existing run can be
plotted offline.  Matplotlib is imported lazily: a missing plotting optional
dependency must never make a controller experiment fail.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from .arc_profile import project_to_polyline


_PLT = None


def _matplotlib():
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return {} if default is None else default
    with path.open() as stream:
        return json.load(stream)


def _values(rows, key, default=np.nan):
    values = []
    for row in rows:
        value = row.get(key, "")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, dtype=float)


def _finite(values, *others):
    mask = np.isfinite(values)
    for other in others:
        mask &= np.isfinite(other)
    return (values[mask],) + tuple(other[mask] for other in others)


def _unwrap_error(actual, reference):
    delta = np.asarray(actual, dtype=float) - np.asarray(reference, dtype=float)
    return np.arctan2(np.sin(delta), np.cos(delta))


def _time(rows):
    stamps = _values(rows, "stamp")
    finite = np.flatnonzero(np.isfinite(stamps))
    if not len(finite):
        return np.arange(len(rows), dtype=float)
    return stamps - stamps[finite[0]]


def _path_model(rows):
    x = _values(rows, "x")
    y = _values(rows, "y")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return {"x": x, "y": y, "s": np.zeros(len(x))}
    s = np.zeros(len(x), dtype=float)
    s[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    return {"x": x, "y": y, "s": s}


def _project(x, y, model):
    """Project points onto a polyline and return arc, signed error, distance."""
    px = np.asarray(x, dtype=float)
    py = np.asarray(y, dtype=float)
    path_x, path_y, path_s = model["x"], model["y"], model["s"]
    arc = np.zeros(len(px), dtype=float)
    signed = np.zeros(len(px), dtype=float)
    distance = np.full(len(px), np.nan, dtype=float)
    if len(path_x) < 2:
        return arc, signed, distance
    arc_hint = 0.0
    for point_index, (x_value, y_value) in enumerate(zip(px, py)):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        projected_arc, projected_distance, projected_signed = project_to_polyline(
            x_value, y_value, model, arc_hint=arc_hint)
        if not np.isfinite(projected_distance):
            continue
        arc[point_index] = projected_arc
        signed[point_index] = projected_signed
        arc_hint = projected_arc
        distance[point_index] = projected_distance
    return arc, signed, distance


def load_run(run_dir):
    """Load one tracking run and derive plotting series."""
    run_dir = Path(run_dir).expanduser().resolve()
    odom_rows = _read_csv(run_dir / "odometry.csv")
    profile_rows = _read_csv(run_dir / "reference_profile.csv")
    if not odom_rows:
        raise ValueError("no odometry.csv rows in %s" % run_dir)
    if len(profile_rows) < 2:
        raise ValueError("reference_profile.csv needs at least two rows in %s" %
                         run_dir)

    odom = {
        "t": _time(odom_rows),
        "x": _values(odom_rows, "x"),
        "y": _values(odom_rows, "y"),
        "yaw": _values(odom_rows, "yaw"),
        "r": _values(odom_rows, "r"),
        "speed": _values(odom_rows, "speed_world"),
    }
    profile = {
        "x": _values(profile_rows, "x"),
        "y": _values(profile_rows, "y"),
        "s": _values(profile_rows, "s_m"),
        "yaw": _values(profile_rows, "yaw"),
        "speed": _values(profile_rows, "speed"),
        "yaw_rate": _values(profile_rows, "yaw_rate"),
        "time": _values(profile_rows, "time_s"),
    }
    model = {"x": profile["x"], "y": profile["y"], "s": profile["s"]}
    arc, cross, distance = _project(odom["x"], odom["y"], model)
    odom["arc"] = arc
    odom["cross"] = cross
    odom["distance"] = distance
    odom["reference_yaw"] = np.interp(arc, profile["s"], profile["yaw"])
    odom["reference_speed"] = np.interp(arc, profile["s"], profile["speed"])
    odom["reference_yaw_rate"] = np.interp(arc, profile["s"],
                                             profile["yaw_rate"])
    odom["yaw_error"] = _unwrap_error(odom["yaw"], odom["reference_yaw"])
    odom["yaw_rate_error"] = odom["r"] - odom["reference_yaw_rate"]
    return {
        "run_dir": run_dir,
        "odom": odom,
        "profile": profile,
        "force_rows": _read_csv(run_dir / "generalized_force.csv"),
        "pwm_rows": _read_csv(run_dir / "pwm.csv"),
        "status_rows": _read_csv(run_dir / "controller_status.csv"),
        "route": _read_json(run_dir / "route.json", {}),
        "metrics": _read_json(run_dir / "metrics.json", {}),
    }


def _setup_style():
    plt = _matplotlib()
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 160,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / (name + ".png")), dpi=160,
                bbox_inches="tight")
    fig.savefig(str(output_dir / (name + ".pdf")),
                bbox_inches="tight")
    _matplotlib().close(fig)


def _label(run, fallback=None):
    return str(run.get("label") or fallback or run["run_dir"].name)


def plot_run(run, output_dir=None, label=None):
    """Write plots for one run and return the generated file paths."""
    _setup_style()
    plt = _matplotlib()
    output_dir = Path(output_dir or (run["run_dir"] / "plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    odom, profile = run["odom"], run["profile"]
    title = label or run["run_dir"].name

    fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    axis.plot(profile["x"], profile["y"], "k--", linewidth=1.1,
              label="reference profile")
    axis.plot(odom["x"], odom["y"], color="#1976d2", linewidth=1.2,
              label="actual")
    axis.scatter([odom["x"][0], odom["x"][-1]],
                 [odom["y"][0], odom["y"][-1]], c=["#2ca02c", "#d62728"],
                 s=28, zorder=3, label="start / finish")
    axis.set_title("Tracking path: %s" % title)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(fig, output_dir, "tracking_xy")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True,
                             constrained_layout=True)
    axes[0].plot(odom["t"], odom["cross"], color="#1976d2")
    axes[0].axhline(0.0, color="k", linewidth=0.7)
    axes[0].set_ylabel("cross-track [m]")
    axes[0].set_title("Tracking errors: %s" % title)
    axes[1].plot(odom["t"], np.degrees(odom["yaw_error"]), color="#d95f02")
    axes[1].axhline(0.0, color="k", linewidth=0.7)
    axes[1].set_ylabel("yaw error [deg]")
    axes[2].plot(odom["t"], odom["yaw_rate_error"], color="#7570b3")
    axes[2].axhline(0.0, color="k", linewidth=0.7)
    axes[2].set_ylabel("yaw-rate error [rad/s]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    _save(fig, output_dir, "tracking_errors")

    fig, axis = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    axis.plot(odom["t"], odom["speed"], label="actual speed",
              color="#1976d2")
    axis.plot(odom["t"], odom["reference_speed"], "k--",
              label="reference speed")
    axis.set_title("Speed tracking: %s" % title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("speed [m/s]")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save(fig, output_dir, "speed_tracking")

    force_rows = run["force_rows"]
    pwm_rows = run["pwm_rows"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=False,
                             constrained_layout=True)
    if force_rows:
        force_t = _time(force_rows)
        for key, color in (("fx", "#1b9e77"), ("fy", "#d95f02"),
                           ("nz", "#7570b3")):
            axes[0].plot(force_t, _values(force_rows, key), color=color,
                         label=key.upper())
        axes[0].set_ylabel("wrench [N, N, N m]")
        axes[0].legend(frameon=False, ncol=3)
    else:
        axes[0].text(0.5, 0.5, "no generalized_force.csv", ha="center",
                     va="center", transform=axes[0].transAxes)
    if pwm_rows:
        pwm_t = _time(pwm_rows)
        keys = [key for key in pwm_rows[0] if key.startswith("pwm_")]
        for key in keys:
            axes[1].plot(pwm_t, _values(pwm_rows, key), linewidth=0.9,
                         label=key.replace("pwm_", "T"))
        axes[1].set_ylabel("PWM")
        axes[1].legend(frameon=False, ncol=4)
    else:
        axes[1].text(0.5, 0.5, "no pwm.csv", ha="center",
                     va="center", transform=axes[1].transAxes)
    axes[0].set_title("Control output: %s" % title)
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    _save(fig, output_dir, "control_wrench")

    status_rows = run["status_rows"]
    if status_rows:
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True,
                                 constrained_layout=True)
        status_t = _time(status_rows)
        axes[0].plot(status_t, _values(status_rows, "callback_time_ms"),
                     color="#1b9e77")
        axes[0].set_ylabel("callback [ms]")
        axes[1].plot(status_t, _values(status_rows, "position_error_m"),
                     label="position", color="#1976d2")
        axes[1].plot(status_t, np.degrees(_values(status_rows, "yaw_error_rad")),
                     label="yaw [deg]", color="#d95f02")
        axes[1].set_ylabel("controller error")
        axes[1].set_xlabel("time [s]")
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].set_title("Controller diagnostics: %s" % title)
        _save(fig, output_dir, "controller_diagnostics")

    return sorted(str(path) for path in output_dir.iterdir()
                  if path.suffix in (".png", ".pdf"))


def plot_comparison(runs, output_dir):
    """Write overlay and metric plots for ``[(label, run_dir), ...]``."""
    _setup_style()
    plt = _matplotlib()
    loaded = []
    for item in runs:
        if isinstance(item, (tuple, list)):
            label, run_dir = item
        else:
            label, run_dir = item.get("label"), item.get("run_dir")
        run = load_run(run_dir)
        run["label"] = label or run["run_dir"].name
        loaded.append(run)
    if not loaded:
        raise ValueError("comparison needs at least one run")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#1f78b4", "#e7298a"]
    reference = loaded[0]["profile"]

    fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    axis.plot(reference["x"], reference["y"], "k--", linewidth=1.1,
              label="reference")
    for index, run in enumerate(loaded):
        axis.plot(run["odom"]["x"], run["odom"]["y"],
                  color=colors[index % len(colors)], linewidth=1.2,
                  label=run["label"])
    axis.set_title("Tracking path comparison")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(fig, output_dir, "comparison_xy")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True,
                             constrained_layout=True)
    for index, run in enumerate(loaded):
        color = colors[index % len(colors)]
        label = run["label"]
        axes[0].plot(run["odom"]["t"], run["odom"]["cross"], color=color,
                     label=label)
        axes[1].plot(run["odom"]["t"], np.degrees(run["odom"]["yaw_error"]),
                     color=color, label=label)
        axes[2].plot(run["odom"]["t"], run["odom"]["yaw_rate_error"],
                     color=color, label=label)
    axes[0].set_ylabel("cross-track [m]")
    axes[1].set_ylabel("yaw error [deg]")
    axes[2].set_ylabel("yaw-rate error [rad/s]")
    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Tracking error comparison")
    for axis in axes:
        axis.axhline(0.0, color="k", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    _save(fig, output_dir, "comparison_errors")

    metric_specs = (
        ("cross_track_error_m", "cross-track RMSE [m]"),
        ("yaw_error_deg", "yaw RMSE [deg]"),
        ("turn_yaw_error_deg", "turn yaw RMSE [deg]"),
        ("turn_yaw_rate_error_radps", "turn yaw-rate RMSE [rad/s]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5),
                             constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metric_specs):
        values = []
        labels = []
        for run in loaded:
            metric = run["metrics"].get(key, {})
            value = metric.get("rmse") if isinstance(metric, dict) else None
            if value is not None:
                values.append(float(value))
                labels.append(run["label"])
        if values:
            axis.bar(np.arange(len(values)), values,
                     color=[colors[i % len(colors)] for i in range(len(values))])
            axis.set_xticks(np.arange(len(values)))
            axis.set_xticklabels(labels, rotation=20, ha="right")
            axis.set_ylabel(title.split(" [")[1].rstrip("]"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Tracking metric comparison")
    _save(fig, output_dir, "comparison_metrics")
    return sorted(str(path) for path in output_dir.iterdir()
                  if path.suffix in (".png", ".pdf"))


def generate_run_plots(run_dir, output_dir=None, label=None):
    return plot_run(load_run(run_dir), output_dir=output_dir, label=label)


def generate_comparison_plots(runs, output_dir):
    return plot_comparison(runs, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot one or more run_tracking_experiment artifacts")
    parser.add_argument("--run-dir", action="append", required=True,
                        help="run directory; repeat for a comparison")
    parser.add_argument("--label", action="append",
                        help="plot label, repeated in the same order as --run-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    labels = args.label or []
    if len(labels) > len(args.run_dir):
        parser.error("--label cannot be supplied more times than --run-dir")
    labels += [None] * (len(args.run_dir) - len(labels))
    if len(args.run_dir) == 1:
        output = args.output_dir or str(Path(args.run_dir[0]) / "plots")
        generated = generate_run_plots(args.run_dir[0], output, labels[0])
    else:
        output = args.output_dir or str(Path(args.run_dir[0]).parent / "plots")
        generated = generate_comparison_plots(
            list(zip(labels, args.run_dir)), output)
        for run_dir, label in zip(args.run_dir, labels):
            generate_run_plots(run_dir, str(Path(output) / (label or Path(run_dir).name)),
                               label)
    print("Plots: %s" % output)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    main()
