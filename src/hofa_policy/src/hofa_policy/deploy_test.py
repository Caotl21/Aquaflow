"""Run the BC policy in Stonefish closed loop and aggregate tracking results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import rospkg

from .trajectory_suite import SUITE_SCHEMA, validate_track


def ros_subprocess_environment():
    """Make Conda policy workers able to import the system ROS packages."""
    environment = os.environ.copy()
    ros_python_paths = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/home/bricsbot/catkin_ws/devel/lib/python3/dist-packages",
        "/home/bricsbot/bricsbot/devel/lib/python3/dist-packages",
    ]
    existing = environment.get("PYTHONPATH", "")
    paths = ros_python_paths + ([existing] if existing else [])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PATH"] = os.pathsep.join([
        "/opt/ros/noetic/bin",
        "/home/bricsbot/catkin_ws/devel/bin",
        environment.get("PATH", ""),
    ])
    return environment


def metric_row(track_id, family, run_dir, metrics):
    cross = metrics.get("cross_track_error_m", {})
    yaw = metrics.get("yaw_error_deg", {})
    lag = metrics.get("schedule_lag_s", {})
    callback = metrics.get("callback_time_ms") or {}
    return {
        "id": track_id,
        "family": family,
        "run": str(run_dir),
        "finish_reason": metrics.get("finish_reason"),
        "completion_ratio": metrics.get("completion_ratio"),
        "cross_track_rmse_m": cross.get("rmse"),
        "cross_track_p95_m": cross.get("p95_abs"),
        "yaw_rmse_deg": yaw.get("rmse"),
        "schedule_lag_rmse_s": lag.get("rmse"),
        "final_distance_m": metrics.get("final_distance_to_goal_m"),
        "solver_success_rate": metrics.get("solver_success_rate"),
        "callback_p95_ms": callback.get("p95"),
        "max_speed_mps": metrics.get("max_speed_mps"),
        "pwm_saturation_ratio": metrics.get("pwm_saturation_ratio"),
    }


def display(value):
    return "n/a" if value is None else "%.4f" % value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(output, rows, suite, checkpoint):
    completed = sum(row["finish_reason"] == "goal_reached" for row in rows)
    result = {
        "schema": "hofa_policy_deployment_v1",
        "suite": str(suite),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "tracks": rows,
        "summary": {
            "tracks": len(rows),
            "goal_reached": completed,
            "goal_reached_ratio": completed / max(1, len(rows)),
        },
    }
    (output / "deployment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# HOFA Policy Closed-loop Deployment", "",
        "- Checkpoint: %s" % checkpoint,
        "- Checkpoint SHA-256: %s" % result["checkpoint_sha256"],
        "- Goal reached: %d / %d" % (completed, len(rows)), "",
        "| Track | Family | Finish | Completion | Cross RMSE [m] | "
        "Yaw RMSE [deg] | Lag RMSE [s] | Callback P95 [ms] |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["id"], row["family"], row["finish_reason"],
            display(row["completion_ratio"]),
            display(row["cross_track_rmse_m"]),
            display(row["yaw_rmse_deg"]),
            display(row["schedule_lag_rmse_s"]),
            display(row["callback_p95_ms"])))
    (output / "deployment.md").write_text("\n".join(lines) + "\n")
    return result


def run(args):
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("trajectory suite schema mismatch")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiment = (Path(rospkg.RosPack().get_path("hofa_mpc_ros")) /
                  "scripts/run_mpc_experiment.py")
    selected = [entry for entry in suite["tracks"]
                if args.split is None or entry["split"] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    rows = []
    for number, entry in enumerate(selected, 1):
        track_path = (suite_path.parent / entry["path"]).resolve()
        track = validate_track(json.loads(track_path.read_text()))
        track_root = output / entry["id"]
        track_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(track_root.glob("run_*"))
        if args.resume and existing and (existing[-1] / "metrics.json").is_file():
            run_dir = existing[-1]
        else:
            before = {path.resolve() for path in existing}
            command = [
                sys.executable, str(experiment),
                "--output-root", str(track_root),
                "--controller", "hofa_policy",
                "--policy-checkpoint", str(checkpoint),
                "--policy-device", args.device,
                "--scene", "empty_pool_control",
                "--policy-trajectory", str(track_path),
                "--timeout", str(track["timeout_s"]),
                "--ready-timeout", str(args.ready_timeout),
            ]
            if args.headless:
                command.append("--headless")
            if not args.with_bag:
                command.append("--no-bag")
            attempt = 1
            while (track_root / ("runner_%02d.log" % attempt)).exists():
                attempt += 1
            log_path = track_root / ("runner_%02d.log" % attempt)
            print("[%d/%d] running %s" %
                  (number, len(selected), entry["id"]), flush=True)
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    env=ros_subprocess_environment())
            if completed.returncode != 0:
                raise RuntimeError(
                    "closed-loop run failed for %s; inspect %s" %
                    (entry["id"], log_path))
            created = sorted(
                {path.resolve() for path in track_root.glob("run_*")} - before)
            if len(created) != 1:
                raise RuntimeError("expected one new run for %s" % entry["id"])
            run_dir = created[0]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        row = metric_row(entry["id"], entry["family"], run_dir, metrics)
        rows.append(row)
        print("completed %s: finish=%s cross_rmse=%s callback_p95=%s" % (
            entry["id"], row["finish_reason"], row["cross_track_rmse_m"],
            row["callback_p95_ms"]), flush=True)
    result = write_report(output, rows, suite_path, checkpoint)
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-bag", action="store_true")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--headless", dest="headless", action="store_true")
    backend.add_argument("--rendered", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
