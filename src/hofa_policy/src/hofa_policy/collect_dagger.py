"""Collect Stonefish DAgger runs with BC control and MPC teacher labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import rospkg

from .export_dagger import export_dagger
from .trajectory_suite import SUITE_SCHEMA, validate_track


def ros_subprocess_environment():
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


def collect(args):
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("trajectory suite schema mismatch")
    output = Path(args.output).resolve()
    raw_root = output / "raw"
    dataset_root = output / "datasets"
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    experiment = (Path(rospkg.RosPack().get_path("hofa_mpc_ros")) /
                  "scripts/run_mpc_experiment.py")
    selected = [entry for entry in suite["tracks"]
                if args.split is None or entry["split"] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    accepted = []
    for entry in selected:
        track_path = (suite_path.parent / entry["path"]).resolve()
        track = validate_track(json.loads(track_path.read_text()))
        dataset_dir = dataset_root / entry["id"]
        if dataset_dir.exists():
            if args.resume:
                accepted.append(entry["id"])
                continue
            raise FileExistsError("dataset already exists: %s" % dataset_dir)
        track_raw = raw_root / entry["id"]
        track_raw.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in track_raw.glob("run_*")}
        command = [
            sys.executable, str(experiment),
            "--output-root", str(track_raw),
            "--controller", "hofa_policy_dagger",
            "--policy-checkpoint", str(Path(args.checkpoint).resolve()),
            "--policy-device", args.device,
            "--scene", "empty_pool_control",
            "--policy-trajectory", str(track_path),
            "--timeout", str(track["timeout_s"]),
            "--ready-timeout", str(args.ready_timeout),
            "--no-bag",
        ]
        if args.headless:
            command.append("--headless")
        attempt = 1
        while (track_raw / ("runner_%02d.log" % attempt)).exists():
            attempt += 1
        log_path = track_raw / ("runner_%02d.log" % attempt)
        print("collecting DAgger %s" % entry["id"], flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT,
                env=ros_subprocess_environment())
        if result.returncode != 0:
            raise RuntimeError("DAgger run failed; inspect %s" % log_path)
        created = sorted(
            {path.resolve() for path in track_raw.glob("run_*")} - before)
        if len(created) != 1:
            raise RuntimeError("expected one new DAgger run")
        metadata = export_dagger(created[0], dataset_dir)
        metadata.update({
            "trajectory_id": entry["id"],
            "trajectory_split": entry["split"],
            "trajectory_family": entry["family"],
            "trajectory_file": str(track_path),
        })
        (dataset_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        accepted.append(entry["id"])
        print("accepted %s: run=%s samples=%d" %
              (entry["id"], created[0], metadata["samples"]), flush=True)
    (output / "collection.json").write_text(json.dumps({
        "schema": "hofa_policy_dagger_collection_v1",
        "suite": str(suite_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "accepted": accepted,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--headless", dest="headless", action="store_true")
    backend.add_argument("--rendered", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        parser.error("checkpoint does not exist")
    collect(args)


if __name__ == "__main__":
    main()
