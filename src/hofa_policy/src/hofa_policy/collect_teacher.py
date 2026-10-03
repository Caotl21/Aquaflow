"""Sequentially run Stonefish HOFA-MPC teacher trajectories and export data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import rospkg

from .export_run import export_run
from .trajectory_suite import SUITE_SCHEMA, validate_track


def collect(args):
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("trajectory suite schema mismatch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw"
    dataset_root = output / "datasets"
    raw_root.mkdir(exist_ok=True)
    dataset_root.mkdir(exist_ok=True)
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
        before = {p.resolve() for p in track_raw.glob("run_*")}
        command = [
            sys.executable, str(experiment),
            "--output-root", str(track_raw),
            "--controller", "hofa_mpc",
            "--scene", "empty_pool",
            "--policy-trajectory", str(track_path),
            "--timeout", str(track["timeout_s"]),
            "--ready-timeout", str(args.ready_timeout),
        ]
        if args.headless:
            command.append("--headless")
        if not args.with_bag:
            command.append("--no-bag")
        attempt = 1
        while (track_raw / ("runner_%02d.log" % attempt)).exists():
            attempt += 1
        log_path = track_raw / ("runner_%02d.log" % attempt)
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(
                "teacher run failed for %s; inspect %s" %
                (entry["id"], log_path))
        after = {p.resolve() for p in track_raw.glob("run_*")}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError("expected one new run for %s" % entry["id"])
        metadata = export_run(created[0], dataset_dir)
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
              (entry["id"], created[0], metadata["samples"]))
    (output / "collection.json").write_text(json.dumps({
        "suite": str(suite_path),
        "accepted": accepted,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--with-bag", action="store_true")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--headless", dest="headless",
                         action="store_true",
                         help="use the camera-free Stonefish no-GPU backend")
    backend.add_argument("--rendered", dest="headless",
                         action="store_false",
                         help="debug with the OpenGL rendering backend")
    parser.set_defaults(headless=True)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
