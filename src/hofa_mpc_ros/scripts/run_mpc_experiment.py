#!/usr/bin/env python3
"""Run one reproducible BricsBot tracking experiment.

The script starts the current simulation launch by default, sends a fixed
goal, records the important ROS topics, computes geometry/time/speed/control
metrics, and writes every artifact below one timestamped directory.

Example:
    rosrun hofa_mpc_ros run_mpc_experiment.py --scene empty_pool

Use ``--no-launch`` when the simulation is already running.  ROS and the
workspace must be sourced before invoking the script.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import rospy
import rosgraph
import rospkg
from geometry_msgs.msg import AccelStamped, PoseStamped, WrenchStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow
from hofa_mpc_ros.arc_profile import build_arc_profile


# Fallback profile limits, used only when /reference_processor is unreachable
# and runtime_config.json carries no record of them.  Keep in sync with
# config/reference_processor.yaml.
DEFAULT_PROFILE_PARAMS = {
    "max_speed": 0.2,
    "max_yaw_rate": 0.5,
    "max_yaw_accel_radps2": 0.25,
    "max_accel_mps2": 0.12,
    "max_decel_mps2": 0.18,
    "yaw_smoothing_window": 5,
    "yaw_accel_reserve_ratio": 0.2,
}

GOAL_RADIUS_M = 0.25
GOAL_SPEED_MPS = 0.08
GOAL_MIN_TIME_S = 2.0

# Band above the profile's max_speed that counts as tracking ripple rather
# than a violation.  The profile cruises *at* max_speed, so a zero-tolerance
# test reports ~50% for any well-centred run and cannot distinguish it from
# real overspeed; speed_excess_mps carries the magnitude either way.
SPEED_LIMIT_TOLERANCE = 0.05


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stats(values, degrees=False):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    if degrees:
        a = np.degrees(a)
    return {
        "rmse": float(np.sqrt(np.mean(a * a))),
        "mae": float(np.mean(np.abs(a))),
        "p95_abs": float(np.percentile(np.abs(a), 95)),
        "max_abs": float(np.max(np.abs(a))),
    }


def project_to_path(x, y, model):
    """Project onto the route; return (arc length, signed offset, distance)."""
    xy, arc = model["xy"], model["arc"]
    best_dist = float("inf")
    best_s = 0.0
    best_signed = 0.0
    p = np.asarray([x, y], dtype=float)
    for i in range(len(xy) - 1):
        seg = xy[i + 1] - xy[i]
        length_sq = float(seg @ seg)
        if length_sq < 1e-12:
            continue
        alpha = float(np.clip(((p - xy[i]) @ seg) / length_sq, 0.0, 1.0))
        proj = xy[i] + alpha * seg
        err = p - proj
        dist = float(np.linalg.norm(err))
        if dist < best_dist:
            best_dist = dist
            best_s = float(arc[i] + alpha * (arc[i + 1] - arc[i]))
            best_signed = float((seg[0] * err[1] - seg[1] * err[0]) /
                                 max(math.sqrt(length_sq), 1e-9))
    return best_s, best_signed, best_dist


def build_evaluation_model(path, profile_params, t0):
    """Absolute plan for the route, from the same profiler the controller uses.

    The reference window the controller receives is re-anchored to the
    vehicle's own projection every cycle, so scoring against it measures
    nothing in the time dimension -- planned duration comes out exactly equal
    to the run's duration and the along-track error is identically zero.
    Recomputing the profile here from the route geometry gives an absolute
    schedule that the vehicle can be genuinely early or late against.

    ``t0`` is the wall stamp at which the schedule starts (first odometry
    sample of the scored window), so ``times`` are directly comparable to
    odometry stamps.
    """
    points = path["points"]
    if len(points) < 2:
        return None
    params = dict(DEFAULT_PROFILE_PARAMS)
    params.update({k: v for k, v in (profile_params or {}).items()
                   if k in DEFAULT_PROFILE_PARAMS})
    profile = build_arc_profile(
        [p[1] for p in points],
        [p[2] for p in points],
        [p[3] for p in points],
        max_speed=float(params["max_speed"]),
        max_yaw_rate=float(params["max_yaw_rate"]),
        max_yaw_accel=float(params["max_yaw_accel_radps2"]),
        max_accel=float(params["max_accel_mps2"]),
        max_decel=float(params["max_decel_mps2"]),
        yaw_smoothing_window=int(params["yaw_smoothing_window"]),
        yaw_accel_reserve_ratio=float(params["yaw_accel_reserve_ratio"]),
        # The vehicle is at rest when the route is issued, so the plan it is
        # scored against must also start from rest -- otherwise it is charged
        # for lag it could never have avoided.
        initial_speed=0.0)
    return {
        "xy": np.column_stack([profile["x"], profile["y"]]),
        "yaw": np.unwrap(profile["yaw"]),
        "arc": profile["s"],
        "speed": profile["speed"],
        "dpsi": profile["speed"] * profile["curvature"],
        "times": float(t0) + profile["time"],
        "relative_times": profile["time"],
        "total_length": float(profile["total_length"]),
        "planned_duration": float(profile["time"][-1]),
        "max_speed": float(params["max_speed"]),
        "source": "global_path_speed_profile",
    }


def make_run_dir(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, "run_" + stamp)
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(root, "run_%s_%02d" % (stamp, suffix))
        suffix += 1
    os.makedirs(path)
    return path


def _sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root):
    metadata = {"commit": None, "status": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.STDOUT).decode("utf-8").strip()
        metadata["status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=root,
            stderr=subprocess.STDOUT).decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def capture_runtime_config(args):
    """Capture parameters and source provenance from the running system."""
    try:
        package_root = rospkg.RosPack().get_path("hofa_mpc_ros")
    except rospkg.ResourceNotFound:
        package_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir))
    workspace_root = os.path.abspath(os.path.join(
        package_root, os.pardir, os.pardir))
    tracked_files = [
        "scripts/run_mpc_experiment.py",
        "scripts/hofa_mpc_controller_node.py",
        "scripts/reference_processor.py",
        "src/hofa_mpc_ros/mpc.py",
        "src/hofa_mpc_ros/constraints.py",
        "src/hofa_mpc_ros/hofa.py",
        "src/hofa_mpc_ros/model.py",
        "config/mpc.yaml",
        "config/reference_processor.yaml",
        "config/vehicle_sim.yaml",
        "config/vehicle_real.yaml",
        "config/safety.yaml",
    ]
    source_hashes = {}
    for relative_path in tracked_files:
        filename = os.path.join(package_root, relative_path)
        if os.path.isfile(filename):
            source_hashes[relative_path] = _sha256_file(filename)

    parameter_names = [
        "/hofa_mpc_controller/mpc",
        "/hofa_mpc_controller/vehicle",
        "/hofa_mpc_controller/thrusters",
        "/hofa_mpc_controller/safety",
        "/reference_processor",
        "/thrusters",
        "/vehicle",
    ]
    ros_parameters = {}
    for name in parameter_names:
        value = rospy.get_param(name, None)
        if value is not None:
            ros_parameters[name] = value

    return {
        "captured_at_wall_time": datetime.now().isoformat(),
        "captured_at_wall_epoch": time.time(),
        "captured_at_ros_time": rospy.Time.now().to_sec(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "experiment_args": vars(args),
        "ros_parameters": ros_parameters,
        "source_sha256": source_hashes,
        "git": _git_metadata(workspace_root),
        "workspace_root": workspace_root,
        "package_root": package_root,
    }


def write_runtime_config(output_dir, args):
    filename = os.path.join(output_dir, "runtime_config.json")
    with open(filename, "w") as stream:
        json.dump(capture_runtime_config(args), stream, indent=2,
                  sort_keys=True)
    return filename


def goal_reached_index(odom, goal_xy):
    """First odometry index satisfying the goal test, or None.

    Same predicate the live recorder applies, so a rescored run cannot
    disagree with the run that produced it.
    """
    if not odom:
        return None
    t0 = odom[0]["stamp"]
    for index, row in enumerate(odom):
        if (math.hypot(row["x"] - goal_xy[0], row["y"] - goal_xy[1]) < GOAL_RADIUS_M
                and row["speed_world"] < GOAL_SPEED_MPS
                and (row["stamp"] - t0) > GOAL_MIN_TIME_S):
            return index
    return None


def compute_metrics(odom, model, status, pwm, goal_xy, finish_reason):
    """Score a run against the absolute plan in ``model``.

    Two independent axes, deliberately not mixed:

    * geometry, indexed by position -- cross-track, path distance, and the
      yaw/speed the plan calls for *at the arc position the vehicle actually
      occupies*.  These answer "is it on the route, pointed and moving the
      right way for where it is".
    * timing, indexed by the schedule -- progress error and schedule lag.
      These answer "is it where it should be *by now*".
    """
    if not odom or model is None:
        return {"error": "no odometry or global path recorded"}

    arc = model["arc"]
    profile_speed = model["speed"]
    profile_yaw = model["yaw"]
    relative_times = model["relative_times"]

    cross, geom, yaw_err, speed_err = [], [], [], []
    progress_err, schedule_lag = [], []
    turn_yaw_err, turn_yaw_rate_err = [], []
    actual_arc = []
    t0 = odom[0]["stamp"]

    for row in odom:
        t_rel = row["stamp"] - t0
        actual_s, signed, distance = project_to_path(row["x"], row["y"], model)
        actual_arc.append(actual_s)
        cross.append(signed)
        geom.append(distance)

        # Position-indexed references.
        reference_yaw = float(np.interp(actual_s, arc, profile_yaw))
        reference_speed = float(np.interp(actual_s, arc, profile_speed))
        current_yaw_err = wrap(row["yaw"] - reference_yaw)
        yaw_err.append(current_yaw_err)
        speed_err.append(row["speed_world"] - reference_speed)

        # Schedule-indexed references.
        scheduled_s = float(np.interp(t_rel, relative_times, arc))
        progress_err.append(actual_s - scheduled_s)
        scheduled_t = float(np.interp(actual_s, arc, relative_times))
        schedule_lag.append(t_rel - scheduled_t)

        reference_yaw_rate = float(np.interp(actual_s, arc, model["dpsi"]))
        if abs(reference_yaw_rate) >= 0.05:
            turn_yaw_err.append(current_yaw_err)
            turn_yaw_rate_err.append(row["r"] - reference_yaw_rate)

    duration = max(0.0, odom[-1]["stamp"] - t0)
    planned_duration = model["planned_duration"]
    speeds = np.asarray([row["speed_world"] for row in odom], dtype=float)
    threshold = model["max_speed"] * (1.0 + SPEED_LIMIT_TOLERANCE)
    solver_success = [x["solver_success"] for x in status]
    callback_times = np.asarray([x["callback_time_ms"] for x in status],
                                dtype=float)
    pwm_values = np.asarray([x["pwm"] for x in pwm if x["pwm"]], dtype=float)
    if pwm_values.size:
        saturation_ratio = float(np.mean(
            np.any(np.abs(pwm_values) >= 0.99, axis=1)))
    else:
        saturation_ratio = 0.0

    result = {
        "finish_reason": finish_reason,
        "evaluation_reference": model.get("source", "unknown"),
        "evaluation_reference_samples": int(len(model["times"])),
        "duration_s": float(duration),
        "samples": int(len(odom)),
        "path_length_m": float(model["total_length"]),
        "planned_duration_s": float(planned_duration),
        # Lag where the vehicle actually got to, not where it was supposed to
        # end up.  Defined for incomplete runs too, unlike duration minus
        # planned duration, and identical to it when the route is completed.
        "time_lag_at_finish_s": float(schedule_lag[-1]),
        "completion_ratio": float(
            actual_arc[-1] / max(model["total_length"], 1e-9)),
        "final_position_m": [odom[-1]["x"], odom[-1]["y"]],
        "final_distance_to_goal_m": float(math.hypot(
            odom[-1]["x"] - goal_xy[0], odom[-1]["y"] - goal_xy[1])),
        "final_speed_mps": float(odom[-1]["speed_world"]),
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "speed_limit_violation_ratio": float(np.mean(speeds > threshold)),
        "speed_limit_threshold_mps": float(threshold),
        # Magnitude, not just incidence.  The ratio alone cannot separate
        # "cruising at the limit with normal ripple" from "genuinely too
        # fast", because the profile's cruise speed *is* max_speed: any
        # symmetric tracking ripple puts ~50% of samples above it.  Read
        # these together with mean_speed_mps.
        "speed_excess_mps": stats(np.maximum(speeds - model["max_speed"], 0.0)),
        "cross_track_error_m": stats(cross),
        "geometric_path_error_m": stats(geom),
        "yaw_error_deg": stats(yaw_err, degrees=True),
        "speed_error_mps": stats(speed_err),
        "progress_error_m": stats(progress_err),
        "schedule_lag_s": stats(schedule_lag),
        "solver_success_rate": (float(np.mean(solver_success))
                                 if solver_success else None),
        "status_samples": int(len(status)),
        "pwm_samples": int(len(pwm)),
        "pwm_saturation_ratio": saturation_ratio,
        "turn_yaw_error_deg": stats(turn_yaw_err, degrees=True),
        "turn_yaw_rate_error_radps": stats(turn_yaw_rate_err),
    }
    if callback_times.size:
        result["callback_time_ms"] = {
            "mean": float(np.mean(callback_times)),
            "p95": float(np.percentile(callback_times, 95)),
            "max": float(np.max(callback_times)),
            "deadline_miss_ratio": float(np.mean(callback_times > 80.0)),
        }
    else:
        result["callback_time_ms"] = None
    return result


class ExperimentRecorder(object):
    """ROS topic recorder and offline metric calculator."""

    def __init__(self, goal_xy, timeout_s):
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        self.timeout_s = float(timeout_s)
        self.lock = threading.RLock()
        self.start_wall = time.time()
        self.first_odom_stamp = None
        self.last_odom_stamp = None
        self.done = False
        self.finish_reason = "timeout"

        # Profile limits the reference_processor is actually running with, so
        # the schedule scored against matches the one the controller is given.
        self.profile_params = rospy.get_param("/reference_processor", {})
        self.global_path = None
        self.initial_path = None
        self.odom = []
        self.status = []
        self.wrench = []
        self.virtual_accel = []
        self.pwm = []
        self.window = []

        rospy.Subscriber("/bricsbot/odometry", Odometry,
                         self._odom_cb, queue_size=20)
        rospy.Subscriber("/aquaflow/teacher_global_path", Path,
                         self._path_cb, queue_size=2)
        rospy.Subscriber("/controller/reference_trajectory_window",
                         TrajectoryPointWindow, self._window_cb, queue_size=2)
        rospy.Subscriber("/hofa_mpc_controller/status", ControllerStatus,
                         self._status_cb, queue_size=20)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self._wrench_cb, queue_size=20)
        rospy.Subscriber("/hofa_mpc_controller/virtual_accel_cmd", AccelStamped,
                         self._virtual_accel_cb, queue_size=20)
        rospy.Subscriber("/bricsbot/setpoint/pwm", Float64MultiArray,
                         self._pwm_cb, queue_size=20)

    def reset_reference_model(self):
        """Discard any route published before this run's goal was sent."""
        with self.lock:
            self.initial_path = None

    def reset_measurements(self):
        """Start scoring from the first odometry sample after the goal."""
        with self.lock:
            self.first_odom_stamp = None
            self.last_odom_stamp = None
            self.done = False
            self.finish_reason = "timeout"
            self.odom = []
            self.status = []
            self.wrench = []
            self.virtual_accel = []
            self.pwm = []
            self.window = []

    @staticmethod
    def _stamp(msg):
        stamp = msg.header.stamp.to_sec()
        return stamp if stamp > 0.0 else rospy.Time.now().to_sec()

    @staticmethod
    def _copy_path(msg):
        points = []
        for pose in msg.poses:
            stamp = pose.header.stamp.to_sec()
            points.append((stamp, float(pose.pose.position.x),
                           float(pose.pose.position.y),
                           float(pose.pose.position.z),
                           yaw_from_quaternion(pose.pose.orientation)))
        return {
            "stamp": msg.header.stamp.to_sec(),
            "frame_id": msg.header.frame_id,
            "points": points,
        }

    def _path_cb(self, msg):
        if len(msg.poses) < 2:
            return
        path = self._copy_path(msg)
        with self.lock:
            self.global_path = path
            if self.initial_path is None:
                self.initial_path = path

    def _window_cb(self, msg):
        stamp = self._stamp(msg)
        with self.lock:
            if msg.valid:
                for index, point in enumerate(msg.points):
                    self.window.append({
                        "stamp": stamp,
                        "index": index,
                        "x_ned": float(point.pose.position.x),
                        "y_ned": float(point.pose.position.y),
                        "yaw_ned": yaw_from_quaternion(point.pose.orientation),
                        "dx_ned": float(point.twist.linear.x),
                        "dy_ned": float(point.twist.linear.y),
                        "dpsi_ned": float(point.twist.angular.z),
                        "ddx_ned": float(point.accel.linear.x),
                        "ddy_ned": float(point.accel.linear.y),
                        "ddpsi_ned": float(point.accel.angular.z),
                    })

    def _odom_cb(self, msg):
        stamp = self._stamp(msg)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q)
        # Stonefish publishes linear velocity in the sensor/body frame.
        u = float(msg.twist.twist.linear.x)
        v = float(msg.twist.twist.linear.y)
        c, s = math.cos(yaw), math.sin(yaw)
        vx = c * u - s * v
        vy = s * u + c * v
        row = {
            "stamp": stamp,
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "z": float(msg.pose.pose.position.z),
            "yaw": yaw,
            "u": u, "v": v,
            "vx_world": vx, "vy_world": vy,
            "speed_world": math.hypot(vx, vy),
            "r": float(msg.twist.twist.angular.z),
        }
        with self.lock:
            if self.first_odom_stamp is None:
                self.first_odom_stamp = stamp
            self.last_odom_stamp = stamp
            row["t_rel"] = stamp - self.first_odom_stamp
            self.odom.append(row)
            if (math.hypot(row["x"] - self.goal_xy[0],
                           row["y"] - self.goal_xy[1]) < 0.25 and
                    row["speed_world"] < 0.08 and
                    row["t_rel"] > 2.0):
                self.done = True
                self.finish_reason = "goal_reached"

    def _status_cb(self, msg):
        with self.lock:
            self.status.append({
                "stamp": self._stamp(msg),
                "state": str(msg.state),
                "solver_success": bool(msg.solver_success),
                "solver_iterations": int(msg.solver_iterations),
                "objective_value": float(msg.objective_value),
                "layer1_time_ms": float(msg.layer1_time_ms),
                "layer2_time_ms": float(msg.layer2_time_ms),
                "callback_time_ms": float(msg.callback_time_ms),
                "position_error_m": float(msg.position_error_m),
                "yaw_error_rad": float(msg.yaw_error_rad),
            })

    def _wrench_cb(self, msg):
        with self.lock:
            self.wrench.append({
                "stamp": self._stamp(msg),
                "fx": float(msg.wrench.force.x),
                "fy": float(msg.wrench.force.y),
                "nz": float(msg.wrench.torque.z),
            })

    def _virtual_accel_cb(self, msg):
        with self.lock:
            self.virtual_accel.append({
                "stamp": self._stamp(msg),
                "ddx": float(msg.accel.linear.x),
                "ddy": float(msg.accel.linear.y),
                "ddpsi": float(msg.accel.angular.z),
            })

    def _pwm_cb(self, msg):
        with self.lock:
            self.pwm.append({
                "stamp": rospy.Time.now().to_sec(),
                "pwm": [float(v) for v in msg.data],
            })

    def evaluation_model(self, path, odom):
        """Absolute plan for ``path``, anchored at the first scored sample."""
        if not path or not odom:
            return None
        return build_evaluation_model(path, self.profile_params,
                                      odom[0]["stamp"])

    def snapshot(self):
        with self.lock:
            return {
                "odom": list(self.odom),
                "status": list(self.status),
                "wrench": list(self.wrench),
                "pwm": list(self.pwm),
                "window": list(self.window),
                "virtual_accel": list(self.virtual_accel),
                "path": self.initial_path or self.global_path,
                "finish_reason": self.finish_reason,
            }

    def metrics(self):
        data = self.snapshot()
        model = self.evaluation_model(data["path"], data["odom"])
        if model is None:
            return {"error": "no odometry or global path recorded"}
        return compute_metrics(data["odom"], model, data["status"],
                               data["pwm"], self.goal_xy,
                               data["finish_reason"])

    def save(self, output_dir):
        data = self.snapshot()
        model = self.evaluation_model(data["path"], data["odom"])
        return write_artifacts(output_dir, data, model, self.goal_xy)


def write_csv(filename, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(filename, metrics):
    with open(filename, "w") as f:
        f.write("HOFA-MPC experiment summary\n")
        f.write("===========================\n")
        for key, value in metrics.items():
            f.write("%-30s %s\n" % (key, value))


def write_artifacts(output_dir, data, model, goal_xy):
    """Write every artifact for a run and return its metrics.

    Shared by the live recorder and ``--rescore`` so a rescored run is scored
    by exactly the same code that scored it originally.
    """
    odom = data["odom"]
    if model is None or not odom:
        metrics = {"error": "no odometry or global path recorded"}
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        return metrics

    metrics = compute_metrics(odom, model, data["status"], data["pwm"],
                              goal_xy, data["finish_reason"])
    write_csv(os.path.join(output_dir, "odometry.csv"), odom)
    write_csv(os.path.join(output_dir, "controller_status.csv"), data["status"])
    write_csv(os.path.join(output_dir, "generalized_force.csv"), data["wrench"])
    write_csv(os.path.join(output_dir, "virtual_accel.csv"),
              data["virtual_accel"])
    write_csv(os.path.join(output_dir, "pwm.csv"),
              [{"stamp": x["stamp"],
                **{"pwm_%d" % i: v for i, v in enumerate(x["pwm"])}}
               for x in data["pwm"]])
    write_csv(os.path.join(output_dir, "mpc_reference_window.csv"),
              data["window"])

    t0 = odom[0]["stamp"]
    evaluation_rows = []
    for row in odom:
        t_rel = row["stamp"] - t0
        scheduled_s = float(np.interp(t_rel, model["relative_times"],
                                      model["arc"]))
        evaluation_rows.append({
            "stamp": row["stamp"],
            "t_rel": t_rel,
            "x_ref": float(np.interp(scheduled_s, model["arc"],
                                     model["xy"][:, 0])),
            "y_ref": float(np.interp(scheduled_s, model["arc"],
                                     model["xy"][:, 1])),
            "yaw_ref": float(np.interp(scheduled_s, model["arc"],
                                       model["yaw"])),
            "speed_ref": float(np.interp(scheduled_s, model["arc"],
                                         model["speed"])),
            "scheduled_arc_m": scheduled_s,
            "actual_arc_m": project_to_path(row["x"], row["y"], model)[0],
            "reference_source": model.get("source", "unknown"),
        })
    write_csv(os.path.join(output_dir, "evaluation_reference.csv"),
              evaluation_rows)

    if data["path"]:
        write_csv(os.path.join(output_dir, "global_reference_path.csv"),
                  [{"stamp": p[0], "x_ned": p[1], "y_ned": p[2],
                    "z_ned": p[3], "yaw_ned": p[4]}
                   for p in data["path"]["points"]])
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    write_summary(os.path.join(output_dir, "summary.txt"), metrics)
    render_plots(output_dir, data, model, goal_xy)
    return metrics


def render_plots(output_dir, data, model, goal_xy):
    if plt is None:
        return
    odom = data["odom"]
    if not odom or model is None:
        return
    wrench, virtual_accel = data["wrench"], data["virtual_accel"]
    t0 = odom[0]["stamp"]
    t = np.asarray([row["stamp"] - t0 for row in odom])
    x = np.asarray([row["x"] for row in odom])
    y = np.asarray([row["y"] for row in odom])
    speed = np.asarray([row["speed_world"] for row in odom])
    arc, rel_times = model["arc"], model["relative_times"]

    scheduled_s = np.interp(t, rel_times, arc)
    rx = np.interp(scheduled_s, arc, model["xy"][:, 0])
    ry = np.interp(scheduled_s, arc, model["xy"][:, 1])

    cross, geom, ye, ve, lag = [], [], [], [], []
    for row, t_rel in zip(odom, t):
        actual_s, signed, distance = project_to_path(row["x"], row["y"], model)
        cross.append(signed)
        geom.append(distance)
        ye.append(wrap(row["yaw"] - float(np.interp(actual_s, arc,
                                                    model["yaw"]))))
        ve.append(row["speed_world"] - float(np.interp(actual_s, arc,
                                                       model["speed"])))
        lag.append(t_rel - float(np.interp(actual_s, arc, rel_times)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model["xy"][:, 0], model["xy"][:, 1], "m--", label="planned route")
    # Where the plan says the vehicle should be at each instant.  Under the
    # old window-based reference this curve sat on top of the odometry by
    # construction and so showed nothing.
    ax.plot(rx, ry, "b:", label="scheduled position")
    ax.plot(x, y, "r", label="actual odometry")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="k", marker="x", label="goal")
    ax.set_xlabel("NED x [m]"); ax.set_ylabel("NED y [m]")
    ax.set_title("XY tracking"); ax.axis("equal"); ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tracking_xy.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(t, cross); axes[0].set_ylabel("cross-track [m]")
    axes[0].grid(True)
    axes[1].plot(t, geom); axes[1].set_ylabel("path distance [m]")
    axes[1].grid(True)
    axes[2].plot(t, np.degrees(ye)); axes[2].set_ylabel("yaw error [deg]")
    axes[2].grid(True)
    axes[3].plot(t, lag); axes[3].axhline(0.0, color="k", lw=0.8)
    axes[3].set_ylabel("schedule lag [s]\n(+behind / -ahead)")
    axes[3].set_xlabel("time [s]"); axes[3].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tracking_errors.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, speed, label="actual speed")
    ax.plot(t, np.interp(t, rel_times, model["speed"]), "--",
            label="scheduled speed")
    ax.axhline(model["max_speed"], color="r", lw=0.8, ls=":",
               label="max_speed limit")
    ax.plot(t, ve, ":", label="speed error vs profile at position")
    ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "speed_tracking.png"), dpi=150)
    plt.close(fig)

    if wrench:
        wt = np.asarray([w["stamp"] - t0 for w in wrench])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(wt, [w["fx"] for w in wrench], label="Fx")
        ax.plot(wt, [w["fy"] for w in wrench], label="Fy")
        ax.plot(wt, [w["nz"] for w in wrench], label="Nz")
        ax.set_xlabel("time [s]"); ax.set_ylabel("wrench [N/Nm]")
        ax.grid(True); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "control_wrench.png"), dpi=150)
        plt.close(fig)

    if virtual_accel:
        at = np.asarray([a["stamp"] - t0 for a in virtual_accel])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(at, [a["ddx"] for a in virtual_accel], label="ddx")
        ax.plot(at, [a["ddy"] for a in virtual_accel], label="ddy")
        ax.plot(at, [a["ddpsi"] for a in virtual_accel], label="ddpsi")
        ax.set_xlabel("time [s]"); ax.set_ylabel("virtual acceleration")
        ax.grid(True); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "virtual_accel.png"), dpi=150)
        plt.close(fig)


def _read_csv(filename, casts=None):
    """Read a recorded CSV into dicts, casting the columns we score on."""
    if not os.path.isfile(filename):
        return []
    rows = []
    with open(filename) as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                cast = (casts or {}).get(key, float)
                try:
                    row[key] = cast(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def load_run(run_dir):
    """Reconstruct a recorded run from its CSV artifacts, without ROS."""
    odom = _read_csv(os.path.join(run_dir, "odometry.csv"))
    if not odom:
        raise RuntimeError("no odometry.csv in %s" % run_dir)

    path_rows = _read_csv(os.path.join(run_dir, "global_reference_path.csv"))
    if len(path_rows) < 2:
        raise RuntimeError("no usable global_reference_path.csv in %s" % run_dir)
    path = {
        "stamp": path_rows[0]["stamp"],
        "frame_id": "world_ned",
        "points": [(r["stamp"], r["x_ned"], r["y_ned"], r["z_ned"],
                    r["yaw_ned"]) for r in path_rows],
    }

    pwm_rows = _read_csv(os.path.join(run_dir, "pwm.csv"))
    pwm = [{"stamp": r["stamp"],
            "pwm": [v for k, v in sorted(r.items())
                    if k.startswith("pwm_")]}
           for r in pwm_rows]

    status = _read_csv(os.path.join(run_dir, "controller_status.csv"),
                       casts={"state": str, "solver_success": lambda v: v == "True"})

    run_config = {}
    run_config_file = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(run_config_file):
        with open(run_config_file) as stream:
            run_config = json.load(stream)
    goal_xy = (float(run_config.get("goal_x", 0.0)),
               float(run_config.get("goal_y", 0.0)))

    profile_params = {}
    runtime_config_file = os.path.join(run_dir, "runtime_config.json")
    if os.path.isfile(runtime_config_file):
        with open(runtime_config_file) as stream:
            runtime_config = json.load(stream)
        profile_params = (runtime_config.get("ros_parameters", {})
                          .get("/reference_processor", {}))

    # Re-derive rather than trust the old metrics.json: the point of rescoring
    # is that the previous verdict may have been wrong.
    reached = goal_reached_index(odom, goal_xy)
    data = {
        "odom": odom,
        "status": status,
        "wrench": _read_csv(os.path.join(run_dir, "generalized_force.csv")),
        "pwm": pwm,
        "window": _read_csv(os.path.join(run_dir, "mpc_reference_window.csv")),
        "virtual_accel": _read_csv(os.path.join(run_dir, "virtual_accel.csv")),
        "path": path,
        "finish_reason": "goal_reached" if reached is not None else "timeout",
    }
    return data, goal_xy, profile_params


def rescore(run_dir):
    """Recompute metrics and plots for an existing run directory in place."""
    data, goal_xy, profile_params = load_run(run_dir)
    model = build_evaluation_model(data["path"], profile_params,
                                   data["odom"][0]["stamp"])
    metrics = write_artifacts(run_dir, data, model, goal_xy)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Rescored: %s" % run_dir)
    return metrics


def publish_goal(goal_xy, depth=1.0):
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.header.frame_id = "world_ned"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = float(goal_xy[0])
    msg.pose.position.y = float(goal_xy[1])
    msg.pose.position.z = float(depth)
    msg.pose.orientation.w = 1.0
    # Publish repeatedly so the teacher cannot miss the goal during startup.
    for _ in range(10):
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rospy.sleep(0.2)


def terminate_process(process, name):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
        except Exception:
            print("warning: could not cleanly stop %s" % name,
                  file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default=None,
        help="result root; defaults to results/pid for PID and results/mpc for HOFA-MPC")
    parser.add_argument("--launch-pkg", default="hofa_mpc_ros")
    parser.add_argument("--launch-file", default="simulation.launch")
    parser.add_argument("--controller", choices=("hofa_mpc", "pid"),
                        default="hofa_mpc",
                        help="backend controller launched by simulation.launch")
    parser.add_argument("--scene", default="empty_pool")
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-depth", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-wait", type=float, default=5.0)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-bag", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="use Stonefish parsed_simulator_nogpu")
    parser.add_argument(
        "--rescore", metavar="RUN_DIR", default=None,
        help="recompute metrics and plots for an existing run directory from "
             "its CSVs and exit; needs no simulator and no ROS master")
    args = parser.parse_args(rospy.myargv()[1:])

    if args.rescore:
        rescore(args.rescore)
        return 0

    if args.output_root is None:
        args.output_root = os.path.join(
            "results", "pid" if args.controller == "pid" else "mpc")
    output_dir = make_run_dir(os.path.abspath(args.output_root))
    goal_xy = (args.goal_x, args.goal_y)
    launch_proc = None
    bag_proc = None
    roscore_proc = None
    try:
        if not rosgraph.is_master_online():
            roscore_proc = subprocess.Popen(["roscore"], start_new_session=True)
            deadline = time.time() + 15.0
            while not rosgraph.is_master_online() and time.time() < deadline:
                time.sleep(0.2)
            if not rosgraph.is_master_online():
                raise RuntimeError("roscore did not become available")

        rospy.init_node("mpc_experiment_runner", anonymous=True)
        recorder = ExperimentRecorder(goal_xy, args.timeout)

        if not args.no_launch:
            launch_cmd = ["roslaunch", args.launch_pkg, args.launch_file,
                          "controller:=" + args.controller,
                          "vehicle_name:=bricsbot",
                          "vehicle_model:=bricsbot", "scene:=" + args.scene,
                          "enable:=true"]
            if args.headless:
                launch_cmd.append("headless:=true")
            launch_proc = subprocess.Popen(launch_cmd, start_new_session=True)

        if not args.no_bag:
            bag_file = os.path.join(output_dir, "topics.bag")
            bag_topics = ["/clock", "/bricsbot/odometry",
                          "/aquaflow/teacher_global_path",
                          "/controller/reference_path",
                          "/controller/reference_velocity",
                          "/controller/reference_trajectory_window",
                          "/hofa_mpc_controller/status",
                          "/hofa_mpc_controller/virtual_accel_cmd",
                          "/controller/generalized_force",
                          "/bricsbot/setpoint/pwm"]
            try:
                bag_proc = subprocess.Popen(
                    ["rosbag", "record", "-O", bag_file] + bag_topics,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except OSError:
                print("warning: rosbag not found; continuing without bag",
                      file=sys.stderr)

        rospy.sleep(args.startup_wait)
        write_runtime_config(output_dir, args)
        # The teacher can publish its default/nominal route during startup.
        # Score only the route generated after this run's explicit goal.
        recorder.reset_reference_model()
        publish_goal(goal_xy, args.goal_depth)
        recorder.reset_measurements()
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < args.timeout:
            if recorder.done:
                break
            rospy.sleep(0.2)
        if not recorder.done:
            recorder.finish_reason = "timeout"
        with open(os.path.join(output_dir, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        metrics = recorder.save(output_dir)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print("Artifacts: %s" % output_dir)
    finally:
        terminate_process(bag_proc, "rosbag")
        terminate_process(launch_proc, "roslaunch")
        terminate_process(roscore_proc, "roscore")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        sys.exit(130)
