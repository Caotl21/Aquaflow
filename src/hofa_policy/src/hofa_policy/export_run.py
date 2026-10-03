"""Export a recorded HOFA-MPC run into causal policy samples."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import CONTRACT
from .features import horizontal_wrench, reference_features, state_features


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_windows(rows, horizon):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["stamp"]), []).append(row)
    accepted = {}
    for stamp, points in grouped.items():
        points.sort(key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in points[:horizon]]
        if indices == list(range(horizon)):
            accepted[stamp] = points[:horizon]
    return accepted


def nearest_index(stamps, target):
    insertion = bisect_left(stamps, target)
    candidates = []
    if insertion < len(stamps):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    return min(candidates, key=lambda index: abs(stamps[index] - target))


def export_run(run_dir, output_dir, horizon=CONTRACT.horizon,
               max_odom_age=0.05, max_reference_age=0.15,
               max_status_offset=0.06):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output already exists: %s" % output_dir)

    required = {
        "odometry": run_dir / "odometry.csv",
        "reference": run_dir / "mpc_reference_window.csv",
        "wrench": run_dir / "generalized_force.csv",
        "status": run_dir / "controller_status.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing run files: %s" % missing)

    odometry = read_csv(required["odometry"])
    wrench = read_csv(required["wrench"])
    status = read_csv(required["status"])
    windows = group_windows(read_csv(required["reference"]), horizon)
    if not odometry or not wrench or not status or not windows:
        raise ValueError("run does not contain enough controller data")

    odometry.sort(key=lambda row: float(row["stamp"]))
    wrench.sort(key=lambda row: float(row["stamp"]))
    status.sort(key=lambda row: float(row["stamp"]))
    odom_stamps = [float(row["stamp"]) for row in odometry]
    status_stamps = [float(row["stamp"]) for row in status]
    window_stamps = sorted(windows)

    states = []
    references = []
    previous_wrenches = []
    target_wrenches = []
    stamps = []
    rejected = {
        "no_causal_odometry": 0,
        "stale_odometry": 0,
        "no_causal_reference": 0,
        "stale_reference": 0,
        "status_mismatch": 0,
        "solver_not_active": 0,
    }
    previous = np.zeros(3, dtype=np.float32)

    for wrench_row in wrench:
        stamp = float(wrench_row["stamp"])
        odom_index = bisect_right(odom_stamps, stamp) - 1
        if odom_index < 0:
            rejected["no_causal_odometry"] += 1
            continue
        if stamp - odom_stamps[odom_index] > max_odom_age:
            rejected["stale_odometry"] += 1
            continue
        window_index = bisect_right(window_stamps, stamp) - 1
        if window_index < 0:
            rejected["no_causal_reference"] += 1
            continue
        window_stamp = window_stamps[window_index]
        if stamp - window_stamp > max_reference_age:
            rejected["stale_reference"] += 1
            continue
        status_index = nearest_index(status_stamps, stamp)
        if abs(status_stamps[status_index] - stamp) > max_status_offset:
            rejected["status_mismatch"] += 1
            continue
        status_row = status[status_index]
        if (status_row["state"] != "ACTIVE" or
                status_row["solver_success"].lower() != "true"):
            rejected["solver_not_active"] += 1
            continue

        odom = odometry[odom_index]
        target = horizontal_wrench(wrench_row)
        states.append(state_features(odom))
        references.append(reference_features(
            odom, windows[window_stamp], horizon=horizon))
        previous_wrenches.append(previous.copy())
        target_wrenches.append(target)
        stamps.append(stamp)
        previous = target

    if not states:
        raise ValueError("no samples passed causal export checks")

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "samples.npz",
        state=np.asarray(states, dtype=np.float32),
        reference=np.asarray(references, dtype=np.float32),
        previous_wrench=np.asarray(previous_wrenches, dtype=np.float32),
        target_wrench=np.asarray(target_wrenches, dtype=np.float32),
        stamp=np.asarray(stamps, dtype=np.float64),
    )
    metadata = {
        "contract": CONTRACT.to_dict(),
        "source_run": str(run_dir),
        "samples": len(states),
        "candidate_wrenches": len(wrench),
        "rejected": rejected,
        "limits": {
            "max_odom_age_s": max_odom_age,
            "max_reference_age_s": max_reference_age,
            "max_status_offset_s": max_status_offset,
        },
        "source_sha256": {
            name: sha256(path) for name, path in required.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=CONTRACT.horizon)
    parser.add_argument("--max-odom-age", type=float, default=0.05)
    parser.add_argument("--max-reference-age", type=float, default=0.15)
    parser.add_argument("--max-status-offset", type=float, default=0.06)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("horizon must be positive")
    metadata = export_run(
        args.run, args.output, args.horizon, args.max_odom_age,
        args.max_reference_age, args.max_status_offset)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
