#!/usr/bin/env python3
"""Compare two HOFA policy checkpoints on the same Stonefish route.

Each checkpoint is evaluated in a fresh simulator instance through the
shared sharp-turn runner.  The route name and all reference parameters are
identical; only the policy checkpoint label changes.
"""
from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_BASELINE = (
    "/home/bricsbot/catkin_ws/results/hofa_policy/"
    "train_v1_epoch500/best.pt")
DEFAULT_DAGGER = (
    "/home/bricsbot/catkin_ws/results/hofa_policy/"
    "train_dagger_v1/best.pt")


def load_plotter():
    """Load the shared plotter even when catkin has not been rebuilt yet."""
    try:
        from hofa_mpc_ros.tracking_plots import generate_comparison_plots
        return generate_comparison_plots
    except ImportError:
        source = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../hofa_mpc_ros/src/hofa_mpc_ros/"
            "tracking_plots.py"))
        if not os.path.isfile(source):
            return None
        spec = importlib.util.spec_from_file_location(
            "hofa_mpc_ros_tracking_plots", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_comparison_plots


def latest_run(root, exclude=None):
    exclude = exclude or set()
    candidates = sorted(
        (path for path in Path(root).glob("run_*")
         if path not in exclude and (path / "metrics.json").is_file()),
        key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_one(label, checkpoint, args, output_root):
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise RuntimeError("%s checkpoint does not exist: %s" %
                           (label, checkpoint))
    label_root = os.path.join(output_root, label)
    os.makedirs(label_root, exist_ok=True)
    before = set(Path(label_root).glob("run_*"))
    command = [
        "rosrun", "hofa_mpc_ros", "run_tracking_experiment.py",
        "--controller", "hofa_policy",
        "--route", args.route,
        "--checkpoint", checkpoint,
        "--policy-device", args.policy_device,
        "--scene", args.scene,
        "--goal-depth", str(args.goal_depth),
        "--max-yaw-rate", str(args.max_yaw_rate),
        "--max-yaw-accel", str(args.max_yaw_accel),
        "--timeout", str(args.timeout),
        "--ready-timeout", str(args.ready_timeout),
        "--output-root", label_root,
    ]
    if args.scenario_file:
        command.extend(["--scenario-file", os.path.abspath(args.scenario_file)])
    if args.headless:
        command.append("--headless")
    print("[%s] %s" % (label, " ".join(command)), flush=True)
    completed = subprocess.run(command, check=False)
    run_dir = latest_run(label_root, exclude=before)
    if run_dir is None:
        raise RuntimeError("%s produced no metrics directory (exit=%d)" %
                           (label, completed.returncode))
    if completed.returncode != 0:
        raise RuntimeError("%s runner failed with exit=%d; artifacts=%s" %
                           (label, completed.returncode, run_dir))
    with (run_dir / "metrics.json").open() as stream:
        metrics = json.load(stream)
    metrics["checkpoint"] = checkpoint
    metrics["checkpoint_label"] = label
    metrics["runner_exit_code"] = int(completed.returncode)
    return run_dir, metrics


def metric_value(metrics, *keys):
    value = metrics
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def write_comparison_csv(path, rows):
    fields = [
        "label", "checkpoint", "run_dir", "finish_reason",
        "completion_ratio", "duration_s", "final_distance_to_goal_m",
        "cross_track_rmse_m", "yaw_rmse_deg", "turn_yaw_rmse_deg",
        "turn_yaw_rate_rmse_radps", "mean_speed_mps", "max_speed_mps",
        "solver_success_rate",
    ]
    with open(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare BC and DAgger policy checkpoints on one route")
    parser.add_argument("--route", default="right_angle",
                        choices=("hairpin", "switchback", "right_angle"))
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--dagger-checkpoint", default=DEFAULT_DAGGER)
    parser.add_argument("--output-root",
                        default="results/hofa_policy/policy_comparison")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--scene", default="empty_pool_control")
    parser.add_argument("--scenario-file")
    parser.add_argument("--goal-depth", type=float, default=1.0)
    parser.add_argument("--max-yaw-rate", type=float, default=0.5)
    parser.add_argument("--max-yaw-accel", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    results = []
    for label, checkpoint in (("baseline_bc", args.baseline_checkpoint),
                              ("dagger", args.dagger_checkpoint)):
        run_dir, metrics = run_one(label, checkpoint, args, output_root)
        results.append((label, run_dir, metrics))

    rows = []
    for label, run_dir, metrics in results:
        rows.append({
            "label": label,
            "checkpoint": metrics.get("checkpoint"),
            "run_dir": str(run_dir),
            "finish_reason": metrics.get("finish_reason"),
            "completion_ratio": metrics.get("completion_ratio"),
            "duration_s": metrics.get("duration_s"),
            "final_distance_to_goal_m": metrics.get("final_distance_to_goal_m"),
            "cross_track_rmse_m": metric_value(metrics, "cross_track_error_m", "rmse"),
            "yaw_rmse_deg": metric_value(metrics, "yaw_error_deg", "rmse"),
            "turn_yaw_rmse_deg": metric_value(metrics, "turn_yaw_error_deg", "rmse"),
            "turn_yaw_rate_rmse_radps": metric_value(
                metrics, "turn_yaw_rate_error_radps", "rmse"),
            "mean_speed_mps": metrics.get("mean_speed_mps"),
            "max_speed_mps": metrics.get("max_speed_mps"),
            "solver_success_rate": metrics.get("solver_success_rate"),
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_json = os.path.join(output_root, "comparison_%s.json" % stamp)
    comparison_csv = os.path.join(output_root, "comparison_%s.csv" % stamp)
    payload = {
        "route": args.route,
        "scene": args.scene,
        "reference": {
            "max_yaw_rate_radps": args.max_yaw_rate,
            "max_yaw_accel_radps2": args.max_yaw_accel,
        },
        "runs": rows,
    }
    with open(comparison_json, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    write_comparison_csv(comparison_csv, rows)
    try:
        generate_comparison_plots = load_plotter()
        if generate_comparison_plots is None:
            raise RuntimeError("shared plotter source is unavailable")
        plot_dir = os.path.join(output_root, "plots")
        plots = generate_comparison_plots(
            [(label, str(run_dir)) for label, run_dir, _ in results], plot_dir)
        print("Plots: %s" % plot_dir)
        print("Generated %d comparison plot files" % len(plots))
    except Exception as error:
        print("WARNING: comparison plot generation failed: %s" % error)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Comparison JSON: %s" % comparison_json)
    print("Comparison CSV: %s" % comparison_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
