"""Generate metrics from one recorded hofa_policy dual-Shadow run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .actuation import ThrusterGeometry

AXES = ("fx", "fy", "nz")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values):
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def axis_dict(values):
    return dict(zip(AXES, np.asarray(values, dtype=np.float64).tolist()))


def branch_metrics(run_dir, prefix, active_cycles, geometry):
    stem = prefix + "_" if prefix else ""
    errors = read_csv(run_dir / (stem + "shadow_error.csv"))
    predicted = read_csv(run_dir / (stem + "shadow_generalized_force.csv"))
    timing = read_csv(run_dir / (stem + "shadow_timing.csv"))
    forces = read_csv(run_dir / (stem + "shadow_candidate_forces.csv"))
    if not errors or not predicted or not timing or not forces:
        return None

    error = np.asarray([[float(row[axis]) for axis in AXES]
                        for row in errors], dtype=np.float64)
    inference = np.asarray([float(row["inference_ms"]) for row in timing])
    odom_age = np.asarray([float(row["odom_age_ms"]) for row in timing])
    reference_age = np.asarray(
        [float(row["reference_age_ms"]) for row in timing])
    normalized = np.asarray([
        [float(row["q_%d" % index]) for index in range(4)]
        for row in forces])
    physical = np.asarray([
        [float(row["force_%d" % index]) for index in range(4)]
        for row in forces])

    quarter = max(1, len(error) // 4)
    first = error[:quarter]
    last = error[-quarter:]
    return {
        "predictions": len(error),
        "valid_prediction_ratio": len(error) / max(1, active_cycles),
        "rmse": axis_dict(np.sqrt(np.mean(error ** 2, axis=0))),
        "mae": axis_dict(np.mean(np.abs(error), axis=0)),
        "max_abs": axis_dict(np.max(np.abs(error), axis=0)),
        "first_quarter_rmse": axis_dict(
            np.sqrt(np.mean(first ** 2, axis=0))),
        "last_quarter_rmse": axis_dict(
            np.sqrt(np.mean(last ** 2, axis=0))),
        "inference_ms": describe(inference),
        "odom_age_ms": describe(odom_age),
        "reference_age_ms": describe(reference_age),
        "normalized_action_abs_p95": float(
            np.percentile(np.abs(normalized), 95)),
        "normalized_action_abs_max": float(np.max(np.abs(normalized))),
        "candidate_force_min": np.min(physical, axis=0).tolist(),
        "candidate_force_max": np.max(physical, axis=0).tolist(),
        "candidate_force_violations": int(np.sum(
            (physical < geometry.force_min) |
            (physical > geometry.force_max))),
        "nonfinite_values": int(
            np.sum(~np.isfinite(error)) + np.sum(~np.isfinite(physical))),
    }


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1.0e-12 else None


def evaluate(run_dir, output=None):
    run_dir = Path(run_dir).resolve()
    status = [row for row in read_csv(run_dir / "controller_status.csv")
              if row["state"] == "ACTIVE" and
              row["solver_success"].lower() in ("true", "1")]
    geometry = ThrusterGeometry.bricsbot()
    teacher = branch_metrics(run_dir, "", len(status), geometry)
    if teacher is None:
        raise ValueError("run does not contain complete Shadow artifacts")

    result = {
        "schema": "hofa_policy_shadow_metrics_v2",
        "run": str(run_dir),
        "teacher_active_cycles": len(status),
        "shadow_predictions": teacher["predictions"],
        "valid_prediction_ratio": teacher["valid_prediction_ratio"],
        "rmse": teacher["rmse"],
        "mae": teacher["mae"],
        "max_abs": teacher["max_abs"],
        "inference_ms": teacher["inference_ms"],
        "odom_age_ms": teacher["odom_age_ms"],
        "reference_age_ms": teacher["reference_age_ms"],
        "normalized_action_abs_p95": teacher["normalized_action_abs_p95"],
        "normalized_action_abs_max": teacher["normalized_action_abs_max"],
        "candidate_force_min": teacher["candidate_force_min"],
        "candidate_force_max": teacher["candidate_force_max"],
        "candidate_force_violations": teacher["candidate_force_violations"],
        "nonfinite_values": teacher["nonfinite_values"],
        "teacher_conditioned": teacher,
    }

    free = branch_metrics(run_dir, "free", len(status), geometry)
    if free is not None:
        result["free_running"] = free
        result["free_to_teacher_rmse_ratio"] = {
            axis: safe_ratio(free["rmse"][axis], teacher["rmse"][axis])
            for axis in AXES
        }
        result["free_quarter_drift_ratio"] = {
            axis: safe_ratio(free["last_quarter_rmse"][axis],
                             free["first_quarter_rmse"][axis])
            for axis in AXES
        }

    destination = Path(output).resolve() if output else run_dir
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "shadow_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# HOFA Policy Shadow Report", "",
        "## Teacher-conditioned", "",
        "- Valid predictions: %d / %d (%.3f)" % (
            teacher["predictions"], len(status),
            teacher["valid_prediction_ratio"]),
        "- RMSE: Fx %.4f N, Fy %.4f N, Nz %.5f N*m" % (
            teacher["rmse"]["fx"], teacher["rmse"]["fy"],
            teacher["rmse"]["nz"]),
        "- Inference: P50 %.3f ms, P95 %.3f ms, P99 %.3f ms" % (
            teacher["inference_ms"]["median"],
            teacher["inference_ms"]["p95"],
            teacher["inference_ms"]["p99"]),
        "- Candidate force violations: %d" %
        teacher["candidate_force_violations"],
        "- Nonfinite values: %d" % teacher["nonfinite_values"], "",
    ]
    if free is not None:
        ratios = result["free_to_teacher_rmse_ratio"]
        drift = result["free_quarter_drift_ratio"]
        lines += [
            "## Free-running", "",
            "- Valid predictions: %d / %d (%.3f)" % (
                free["predictions"], len(status),
                free["valid_prediction_ratio"]),
            "- RMSE: Fx %.4f N, Fy %.4f N, Nz %.5f N*m" % (
                free["rmse"]["fx"], free["rmse"]["fy"],
                free["rmse"]["nz"]),
            "- Free/teacher RMSE ratio: Fx %s, Fy %s, Nz %s" % (
                "%.3f" % ratios["fx"] if ratios["fx"] is not None else "n/a",
                "%.3f" % ratios["fy"] if ratios["fy"] is not None else "n/a",
                "%.3f" % ratios["nz"] if ratios["nz"] is not None else "n/a"),
            "- Last/first-quarter drift ratio: Fx %s, Fy %s, Nz %s" % (
                "%.3f" % drift["fx"] if drift["fx"] is not None else "n/a",
                "%.3f" % drift["fy"] if drift["fy"] is not None else "n/a",
                "%.3f" % drift["nz"] if drift["nz"] is not None else "n/a"),
            "- Inference: P50 %.3f ms, P95 %.3f ms, P99 %.3f ms" % (
                free["inference_ms"]["median"],
                free["inference_ms"]["p95"],
                free["inference_ms"]["p99"]),
            "- Candidate force violations: %d" %
            free["candidate_force_violations"],
            "- Nonfinite values: %d" % free["nonfinite_values"], "",
        ]
    (destination / "shadow_report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
