"""Run reproducible Stonefish robustness sweeps for hofa_policy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

import rospkg

from .trajectory_suite import SUITE_SCHEMA, validate_track


SCHEMA = "hofa_policy_robustness_v1"


def enhanced_depth_hold():
    return {
        "kp": 30.0,
        "kd": 10.0,
        "ki": 8.0,
        "integral_limit": 5.0,
        "max_force_n": 35.0,
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_case(case_id="baseline", category="baseline"):
    return {
        "id": case_id,
        "category": category,
        "initial": {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 0.0},
        "physics": {
            "dry_mass_scale": 1.0,
            "water_density_scale": 1.0,
        },
        "depth_hold": {
            "kp": 8.0,
            "kd": 4.0,
            "ki": 0.5,
            "integral_limit": 5.0,
            "max_force_n": 15.0,
        },
        "current_ned_mps": [0.0, 0.0, 0.0],
        "speed_scale": 1.0,
        "relay": {
            "odom": {},
            "reference": {},
            "wrench": {},
        },
    }


def default_cases(profile):
    cases = [base_case()]

    mild = base_case("initial_mild", "initial_offset")
    mild["initial"] = {"dx_m": 0.20, "dy_m": 0.20, "dyaw_deg": 5.0}
    cases.append(mild)
    severe = base_case("initial_severe", "initial_offset")
    severe["initial"] = {"dx_m": -0.30, "dy_m": 0.50, "dyaw_deg": 15.0}
    cases.append(severe)

    light = base_case("mass_light_0p8", "mass")
    light["physics"]["dry_mass_scale"] = 0.8
    light["depth_hold"] = enhanced_depth_hold()
    cases.append(light)
    heavy = base_case("mass_heavy_1p2", "mass")
    heavy["physics"]["dry_mass_scale"] = 1.2
    heavy["depth_hold"] = enhanced_depth_hold()
    cases.append(heavy)

    hydro_low = base_case("hydro_low_0p85", "hydrodynamics")
    hydro_low["physics"]["water_density_scale"] = 0.85
    hydro_low["depth_hold"] = enhanced_depth_hold()
    cases.append(hydro_low)
    hydro_high = base_case("hydro_high_1p15", "hydrodynamics")
    hydro_high["physics"]["water_density_scale"] = 1.15
    hydro_high["depth_hold"] = enhanced_depth_hold()
    cases.append(hydro_high)

    cross = base_case("current_cross_0p10", "current")
    cross["current_ned_mps"] = [0.0, 0.10, 0.0]
    cases.append(cross)
    reverse = base_case("current_reverse_0p15", "current")
    reverse["current_ned_mps"] = [-0.15, 0.0, 0.0]
    cases.append(reverse)

    noise_mild = base_case("sensor_noise_mild", "sensor_noise")
    noise_mild["relay"]["odom"] = {
        "position_std_m": 0.01,
        "yaw_std_deg": 0.5,
        "velocity_std_mps": 0.01,
        "yaw_rate_std_radps": 0.005,
    }
    cases.append(noise_mild)
    noise_severe = base_case("sensor_noise_severe", "sensor_noise")
    noise_severe["relay"]["odom"] = {
        "position_std_m": 0.05,
        "yaw_std_deg": 3.0,
        "velocity_std_mps": 0.05,
        "yaw_rate_std_radps": 0.03,
        "dropout": 0.01,
    }
    cases.append(noise_severe)

    delay_50 = base_case("delay_50ms", "delay")
    delay_50["relay"]["odom"] = {"delay_ms": 50.0, "jitter_ms": 10.0}
    delay_50["relay"]["reference"] = {"delay_ms": 50.0, "jitter_ms": 10.0}
    delay_50["relay"]["wrench"] = {"delay_ms": 30.0, "jitter_ms": 5.0}
    cases.append(delay_50)
    delay_100 = base_case("delay_100ms", "delay")
    delay_100["relay"]["odom"] = {"delay_ms": 100.0, "jitter_ms": 20.0}
    delay_100["relay"]["reference"] = {"delay_ms": 100.0, "jitter_ms": 20.0}
    delay_100["relay"]["wrench"] = {"delay_ms": 50.0, "jitter_ms": 10.0}
    cases.append(delay_100)

    speed_125 = base_case("speed_1p25", "speed")
    speed_125["speed_scale"] = 1.25
    cases.append(speed_125)
    speed_150 = base_case("speed_1p50", "speed")
    speed_150["speed_scale"] = 1.50
    cases.append(speed_150)

    if profile == "smoke":
        selected = {
            "baseline", "initial_mild", "mass_heavy_1p2",
            "current_cross_0p10", "sensor_noise_mild",
            "delay_50ms", "speed_1p25",
        }
        return [case for case in cases if case["id"] in selected]
    return cases


def monte_carlo_cases(count, seed):
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        case = base_case("mc_%03d" % index, "monte_carlo")
        case["initial"] = {
            "dx_m": rng.uniform(-0.5, 0.5),
            "dy_m": rng.uniform(-0.5, 0.5),
            "dyaw_deg": rng.uniform(-15.0, 15.0),
        }
        case["physics"] = {
            "dry_mass_scale": rng.uniform(0.8, 1.2),
            "water_density_scale": rng.uniform(0.9, 1.1),
        }
        # Mass and water-density changes also change vertical buoyancy. Keep
        # the depth channel out of the horizontal policy assessment.
        if (abs(case["physics"]["dry_mass_scale"] - 1.0) > 1.0e-6 or
                abs(case["physics"]["water_density_scale"] - 1.0) > 1.0e-6):
            case["depth_hold"] = enhanced_depth_hold()
        speed = rng.uniform(0.0, 0.15)
        angle = rng.uniform(-math.pi, math.pi)
        case["current_ned_mps"] = [
            speed * math.cos(angle), speed * math.sin(angle), 0.0]
        case["speed_scale"] = rng.uniform(1.0, 1.3)
        case["relay"]["odom"] = {
            "position_std_m": rng.uniform(0.0, 0.03),
            "yaw_std_deg": rng.uniform(0.0, 1.5),
            "velocity_std_mps": rng.uniform(0.0, 0.03),
            "yaw_rate_std_radps": rng.uniform(0.0, 0.02),
            "delay_ms": rng.uniform(0.0, 80.0),
            "jitter_ms": rng.uniform(0.0, 15.0),
        }
        case["relay"]["reference"] = {
            "delay_ms": rng.uniform(0.0, 80.0),
            "jitter_ms": rng.uniform(0.0, 15.0),
        }
        case["relay"]["wrench"] = {
            "delay_ms": rng.uniform(0.0, 50.0),
            "jitter_ms": rng.uniform(0.0, 10.0),
        }
        cases.append(case)
    return cases


def replace_once(text, old, new, description):
    if text.count(old) != 1:
        raise ValueError("expected one %s marker, found %d" %
                         (description, text.count(old)))
    return text.replace(old, new, 1)


def render_scenario(case, output_dir):
    package = Path(rospkg.RosPack().get_path("aquaflow_stonefish"))
    pool_source = package / "scenarios/aquaflow_empty_pool_control.scn"
    robot_source = package / "scenarios/bricsbot_control_reference.scn"
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_path = output_dir / "bricsbot_robustness.scn"
    scenario_path = output_dir / "scenario.scn"

    mass_scale = float(case["physics"]["dry_mass_scale"])
    robot = robot_source.read_text()
    robot = replace_once(
        robot, '<mass value="0.50"/>',
        '<mass value="%.8f"/>' % (0.50 * mass_scale), "hull mass")
    robot = replace_once(
        robot, '<mass value="7.44"/>',
        '<mass value="%.8f"/>' % (7.44 * mass_scale), "ballast mass")
    robot_path.write_text(robot)

    scenario = pool_source.read_text()
    density = 1031.0 * float(case["physics"]["water_density_scale"])
    scenario = replace_once(
        scenario, '<water density="1031.0" jerlov="0.15"/>',
        '<water density="%.8f" jerlov="0.15"/>' % density,
        "water density")
    current = case["current_ned_mps"]
    scenario = replace_once(
        scenario, '<velocity xyz="0.0 0.0 0.0"/>',
        '<velocity xyz="%.8f %.8f %.8f"/>' % tuple(current),
        "uniform current")
    include_marker = (
        'file="$(find aquaflow_stonefish)/scenarios/'
        'bricsbot_control_reference.scn"')
    scenario = replace_once(
        scenario, include_marker, 'file="%s"' % robot_path,
        "robot include")
    initial = case["initial"]
    scenario = replace_once(
        scenario, 'value="-5.0 0.0 1.0"',
        'value="%.8f %.8f 1.0"' %
        (-5.0 + float(initial["dx_m"]), float(initial["dy_m"])),
        "initial position")
    scenario = replace_once(
        scenario, 'value="0.0 0.0 0.0"',
        'value="0.0 0.0 %.8f"' %
        math.radians(float(initial["dyaw_deg"])),
        "initial orientation")
    scenario_path.write_text(scenario)
    return scenario_path


def scaled_track(track, speed_scale, destination):
    result = json.loads(json.dumps(track))
    scale = float(speed_scale)
    for key in ("max_speed_mps", "max_yaw_rate_radps",
                "max_accel_mps2", "max_decel_mps2"):
        result[key] = float(result[key]) * scale
    validate_track(result)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return destination


def relay_config(case, seed, destination):
    data = {"schema": SCHEMA, "seed": int(seed)}
    data.update(case["relay"])
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return destination


def stable_seed(base, case_id, track_id):
    text = "%d:%s:%s" % (base, case_id, track_id)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def read_force_violations(run_dir):
    filename = run_dir / "policy_candidate_forces.csv"
    if not filename.is_file():
        return None
    count = 0
    with filename.open() as stream:
        for row in csv.DictReader(stream):
            for index in range(4):
                value = float(row["force_%d" % index])
                count += int(value < -58.84 or value > 67.67)
    return count


def read_csv_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def geometric_path_distances(run_dir):
    """Return (t_rel, distance-to-reference-polyline) samples."""
    odometry = read_csv_rows(Path(run_dir) / "odometry.csv")
    path_rows = read_csv_rows(Path(run_dir) / "global_reference_path.csv")
    if len(odometry) < 2 or len(path_rows) < 2:
        return []
    points = [
        (float(row["x_ned"]), float(row["y_ned"]))
        for row in path_rows
    ]
    samples = []
    for row in odometry:
        px, py = float(row["x"]), float(row["y"])
        best = float("inf")
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            vx, vy = bx - ax, by - ay
            denominator = vx * vx + vy * vy
            if denominator <= 1.0e-15:
                ratio = 0.0
            else:
                ratio = ((px - ax) * vx + (py - ay) * vy) / denominator
                ratio = max(0.0, min(1.0, ratio))
            qx, qy = ax + ratio * vx, ay + ratio * vy
            best = min(best, math.hypot(px - qx, py - qy))
        samples.append((float(row["t_rel"]), best))
    return samples


def initial_recovery_metrics(run_dir, threshold_m=0.10,
                             hold_s=0.50):
    """Measure return to the path after an initial-state perturbation."""
    samples = geometric_path_distances(run_dir)
    if not samples:
        return {
            "recovery_time_s": None,
            "post_recovery_cross_track_rmse_m": None,
            "post_recovery_cross_track_p95_m": None,
        }
    times = [item[0] for item in samples]
    distances = [item[1] for item in samples]
    recovery_index = None
    for index, (time_s, distance) in enumerate(samples):
        if distance > threshold_m:
            continue
        end = index
        while end + 1 < len(times) and times[end] < time_s + hold_s:
            end += 1
        if max(distances[index:end + 1]) <= threshold_m:
            recovery_index = index
            break
    if recovery_index is None:
        return {
            "recovery_time_s": None,
            "post_recovery_cross_track_rmse_m": None,
            "post_recovery_cross_track_p95_m": None,
        }
    post = distances[recovery_index:]
    return {
        "recovery_time_s": float(times[recovery_index]),
        "post_recovery_cross_track_rmse_m": float(
            math.sqrt(sum(value * value for value in post) / len(post))),
        "post_recovery_cross_track_p95_m": float(
            sorted(post)[int(0.95 * (len(post) - 1))]),
    }


def score_run(case, entry, run_dir, metrics):
    cross = metrics.get("cross_track_error_m", {})
    yaw = metrics.get("yaw_error_deg", {})
    turn_yaw = metrics.get("turn_yaw_error_deg")
    lag = metrics.get("schedule_lag_s", {})
    callback = metrics.get("callback_time_ms") or {}
    yaw_value = ((turn_yaw or {}).get("rmse")
                 if isinstance(turn_yaw, dict) else None)
    if yaw_value is None:
        yaw_value = yaw.get("rmse")
    force_violations = read_force_violations(run_dir)
    recovery = (
        initial_recovery_metrics(run_dir)
        if case["category"] == "initial_offset"
        else {
            "recovery_time_s": None,
            "post_recovery_cross_track_rmse_m": None,
            "post_recovery_cross_track_p95_m": None,
        }
    )
    checks = {
        "goal_reached": metrics.get("finish_reason") == "goal_reached",
        "completion": float(metrics.get("completion_ratio", 0.0)) >= 0.98,
        "final_distance": float(
            metrics.get("final_distance_to_goal_m", 999.0)) <= 0.25,
        "cross_track": float(cross.get("rmse", 999.0)) <= 0.10,
        "yaw": yaw_value is not None and float(yaw_value) <= 5.0,
        # A deliberately perturbed start must recover before schedule lag is
        # judged. The lag remains in the report as a diagnostic.
        "schedule_lag": (
            None if case["category"] == "initial_offset"
            else float(lag.get("rmse", 999.0)) <= 0.25
        ),
        "speed_violation": float(
            metrics.get("speed_limit_violation_ratio", 1.0)) <= 0.05,
        "controller_success": float(
            metrics.get("solver_success_rate", 0.0)) >= 0.999,
        "pwm_saturation": float(
            metrics.get("pwm_saturation_ratio", 1.0)) <= 0.01,
        "callback": float(callback.get("p95", 999.0)) <= 5.0,
        "force_bounds": force_violations == 0,
    }
    if case["category"] == "initial_offset":
        checks.update({
            "recovery": (
                recovery["recovery_time_s"] is not None and
                recovery["recovery_time_s"] <= 10.0
            ),
            "post_recovery_cross_track": (
                recovery["post_recovery_cross_track_rmse_m"] is not None and
                recovery["post_recovery_cross_track_rmse_m"] <= 0.10
            ),
        })
    passed = all(value is not False for value in checks.values())
    return {
        "case_id": case["id"],
        "category": case["category"],
        "track_id": entry["id"],
        "family": entry["family"],
        "run": str(run_dir),
        "evaluation_scope": (
            "horizontal_xy_yaw_with_recovery"
            if case["category"] == "initial_offset"
            else "horizontal_xy_yaw"
        ),
        "passed": passed,
        "checks": checks,
        "metrics": {
            "finish_reason": metrics.get("finish_reason"),
            "completion_ratio": metrics.get("completion_ratio"),
            "final_distance_m": metrics.get("final_distance_to_goal_m"),
            "cross_track_rmse_m": cross.get("rmse"),
            "yaw_rmse_deg": yaw.get("rmse"),
            "turn_yaw_rmse_deg": yaw_value,
            "schedule_lag_rmse_s": lag.get("rmse"),
            "schedule_lag_evaluated": (
                case["category"] != "initial_offset"),
            "recovery_time_s": recovery["recovery_time_s"],
            "post_recovery_cross_track_rmse_m":
                recovery["post_recovery_cross_track_rmse_m"],
            "post_recovery_cross_track_p95_m":
                recovery["post_recovery_cross_track_p95_m"],
            "speed_violation_ratio": metrics.get(
                "speed_limit_violation_ratio"),
            "controller_success_rate": metrics.get("solver_success_rate"),
            "pwm_saturation_ratio": metrics.get("pwm_saturation_ratio"),
            "callback_p95_ms": callback.get("p95"),
            "force_violations": force_violations,
        },
        "perturbation": case,
    }


def write_report(output, manifest, rows):
    passed = sum(bool(row.get("passed")) for row in rows)
    by_category = {}
    for row in rows:
        item = by_category.setdefault(
            row["category"], {"runs": 0, "passed": 0})
        item["runs"] += 1
        item["passed"] += int(bool(row.get("passed")))
    for item in by_category.values():
        item["pass_ratio"] = item["passed"] / max(1, item["runs"])
    result = {
        "schema": SCHEMA,
        "manifest": str(output / "manifest.json"),
        "summary": {
            "runs": len(rows),
            "passed": passed,
            "pass_ratio": passed / max(1, len(rows)),
            "by_category": by_category,
        },
        "runs": rows,
    }
    (output / "robustness.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    failures = [row for row in rows if not row.get("passed")]
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n")
    lines = [
        "# HOFA Policy Robustness Report", "",
        "- Runs: %d" % len(rows),
        "- Passed: %d" % passed,
        "- Pass ratio: %.3f" % result["summary"]["pass_ratio"], "",
        "- Initial-offset cases are judged by recovery time and "
        "post-recovery cross-track error; schedule lag is diagnostic only.",
        "",
        "| Case | Track | Pass | Recovery [s] | Cross RMSE [m] | "
        "Yaw RMSE [deg] | Lag RMSE [s] | Final distance [m] | "
        "Callback P95 [ms] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row.get("metrics", {})
        def shown(key):
            value = metric.get(key)
            return "n/a" if value is None else "%.4f" % value
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["case_id"], row["track_id"],
            "yes" if row.get("passed") else "no",
            shown("recovery_time_s"),
            shown("cross_track_rmse_m"), shown("turn_yaw_rmse_deg"),
            shown("schedule_lag_rmse_s"), shown("final_distance_m"),
            shown("callback_p95_ms")))
    (output / "robustness.md").write_text("\n".join(lines) + "\n")
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

    entries = [entry for entry in suite["tracks"]
               if args.split is None or entry["split"] == args.split]
    if args.limit_tracks is not None:
        entries = entries[:args.limit_tracks]
    cases = default_cases(args.profile)
    cases.extend(monte_carlo_cases(args.monte_carlo, args.seed + 10000))
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["id"] in selected]
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if not cases or not entries:
        raise ValueError("no robustness cases or trajectories selected")

    manifest = {
        "schema": SCHEMA,
        "suite": str(suite_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "profile": args.profile,
        "cases": cases,
        "tracks": [entry["id"] for entry in entries],
        "notes": {
            "water_density_scale": (
                "Stonefish geometry-based hydrodynamics proxy; it also changes buoyancy"),
            "mass_evaluation_scope": (
                "Horizontal XY and yaw only; strengthened vertical hold keeps depth near target"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows = []
    total = len(cases) * len(entries)
    number = 0
    for case in cases:
        assets = output / "case_assets" / case["id"]
        scenario = render_scenario(case, assets)
        for entry in entries:
            number += 1
            source_track = (suite_path.parent / entry["path"]).resolve()
            track = validate_track(json.loads(source_track.read_text()))
            run_assets = assets / entry["id"]
            run_assets.mkdir(parents=True, exist_ok=True)
            trajectory = scaled_track(
                track, case["speed_scale"], run_assets / "trajectory.json")
            seed = stable_seed(args.seed, case["id"], entry["id"])
            config = relay_config(
                case, seed, run_assets / "perturbation.json")
            track_root = output / "runs" / case["id"] / entry["id"]
            track_root.mkdir(parents=True, exist_ok=True)
            existing = sorted(track_root.glob("run_*"))
            if (args.resume and existing and
                    (existing[-1] / "metrics.json").is_file()):
                run_dir = existing[-1]
            else:
                before = {path.resolve() for path in existing}
                command = [
                    sys.executable, str(experiment),
                    "--output-root", str(track_root),
                    "--controller", "hofa_policy",
                    "--policy-checkpoint", str(checkpoint),
                    "--policy-device", args.device,
                    "--policy-depth-kp", str(case["depth_hold"]["kp"]),
                    "--policy-depth-kd", str(case["depth_hold"]["kd"]),
                    "--policy-depth-ki", str(case["depth_hold"]["ki"]),
                    "--policy-depth-integral-limit",
                    str(case["depth_hold"]["integral_limit"]),
                    "--policy-depth-max-force",
                    str(case["depth_hold"]["max_force_n"]),
                    "--scene", "empty_pool_control",
                    "--scenario-file", str(scenario),
                    "--robustness-config", str(config),
                    "--policy-trajectory", str(trajectory),
                    "--timeout", str(
                        float(track["timeout_s"]) * args.timeout_scale),
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
                print("[%d/%d] case=%s track=%s" %
                      (number, total, case["id"], entry["id"]), flush=True)
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode != 0:
                    row = {
                        "case_id": case["id"],
                        "category": case["category"],
                        "track_id": entry["id"],
                        "family": entry["family"],
                        "evaluation_scope": "horizontal_xy_yaw",
                        "passed": False,
                        "run_error": "runner returned %d" % completed.returncode,
                        "log": str(log_path),
                        "perturbation": case,
                    }
                    rows.append(row)
                    write_report(output, manifest, rows)
                    if args.fail_fast:
                        raise RuntimeError(row["run_error"])
                    continue
                created = sorted(
                    {path.resolve() for path in track_root.glob("run_*")} - before)
                if len(created) != 1:
                    raise RuntimeError("expected one new run for %s/%s" %
                                       (case["id"], entry["id"]))
                run_dir = created[0]
            metrics = json.loads((run_dir / "metrics.json").read_text())
            row = score_run(case, entry, run_dir, metrics)
            rows.append(row)
            write_report(output, manifest, rows)
            print("completed case=%s track=%s passed=%s" %
                  (case["id"], entry["id"], row["passed"]), flush=True)
    result = write_report(output, manifest, rows)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--profile", choices=("smoke", "v1"), default="smoke")
    parser.add_argument("--case", action="append",
                        help="run only the named case; may be repeated")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--limit-tracks", type=int)
    parser.add_argument("--monte-carlo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-scale", type=float, default=1.5)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--with-bag", action="store_true")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--headless", dest="headless", action="store_true")
    backend.add_argument("--rendered", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if args.monte_carlo < 0:
        parser.error("--monte-carlo must be nonnegative")
    if args.timeout_scale <= 0.0:
        parser.error("--timeout-scale must be positive")
    run(args)


if __name__ == "__main__":
    main()
