"""Export one DAgger run with policy previous actions and MPC labels."""
import argparse
from bisect import bisect_right
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import CONTRACT
from .export_run import group_windows
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


def export_dagger(run_dir, output_dir, horizon=CONTRACT.horizon,
                  max_odom_age=0.05, max_reference_age=0.15):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output already exists: %s" % output_dir)
    required = {
        "odometry": run_dir / "odometry.csv",
        "reference": run_dir / "mpc_reference_window.csv",
        "teacher_wrench": run_dir / "dagger_teacher_generalized_force.csv",
        "policy_wrench": run_dir / "generalized_force.csv",
        "status": run_dir / "controller_status.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing DAgger files: %s" % missing)

    odometry = sorted(read_csv(required["odometry"]),
                      key=lambda row: float(row["stamp"]))
    teacher = sorted(read_csv(required["teacher_wrench"]),
                     key=lambda row: float(row["stamp"]))
    policy = sorted(read_csv(required["policy_wrench"]),
                    key=lambda row: float(row["stamp"]))
    status = sorted(read_csv(required["status"]),
                    key=lambda row: float(row["stamp"]))
    windows = group_windows(read_csv(required["reference"]), horizon)
    window_stamps = sorted(windows)
    odom_stamps = [float(row["stamp"]) for row in odometry]
    policy_stamps = [float(row["stamp"]) for row in policy]
    if not odometry or not teacher or not policy or not windows:
        raise ValueError("DAgger run does not contain enough data")

    accepted = []
    rejected = {"no_causal_odometry": 0, "stale_odometry": 0,
                "no_causal_reference": 0, "stale_reference": 0,
                "no_previous_policy": 0, "inactive_teacher": 0}
    previous_policy = np.zeros(3, dtype=np.float32)
    policy_cursor = 0
    for teacher_row in teacher:
        stamp = float(teacher_row["stamp"])
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
        while (policy_cursor < len(policy) and
               float(policy[policy_cursor]["stamp"]) < stamp - 1.0e-9):
            previous_policy = horizontal_wrench(policy[policy_cursor])
            policy_cursor += 1
        if policy_cursor == 0:
            rejected["no_previous_policy"] += 1
        target = horizontal_wrench(teacher_row)
        odom = odometry[odom_index]
        accepted.append({
            "stamp": stamp,
            "state": state_features(odom),
            "reference": reference_features(
                odom, windows[window_stamp], horizon=horizon),
            "previous_policy": previous_policy.copy(),
            "target_wrench": target,
        })
        # The next causal policy command becomes the next previous action.
        while (policy_cursor < len(policy) and
               float(policy[policy_cursor]["stamp"]) <= stamp + 1.0e-9):
            previous_policy = horizontal_wrench(policy[policy_cursor])
            policy_cursor += 1

    if not accepted:
        raise ValueError("no DAgger samples passed causal checks")
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "samples.npz",
        state=np.asarray([row["state"] for row in accepted], dtype=np.float32),
        reference=np.asarray(
            [row["reference"] for row in accepted], dtype=np.float32),
        previous_wrench=np.asarray(
            [row["previous_policy"] for row in accepted], dtype=np.float32),
        target_wrench=np.asarray(
            [row["target_wrench"] for row in accepted], dtype=np.float32),
        stamp=np.asarray([row["stamp"] for row in accepted], dtype=np.float64),
    )
    metadata = {
        "schema": "hofa_policy_dagger_v1",
        "contract": CONTRACT.to_dict(),
        "source_run": str(run_dir),
        "samples": len(accepted),
        "candidate_teacher_wrenches": len(teacher),
        "rejected": rejected,
        "limits": {
            "max_odom_age_s": max_odom_age,
            "max_reference_age_s": max_reference_age,
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
    args = parser.parse_args()
    print(json.dumps(export_dagger(
        args.run, args.output, args.horizon, args.max_odom_age,
        args.max_reference_age), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
