#!/usr/bin/env python3
"""Run one reproducible BricsBot tracking experiment.

The script starts the current simulation launch by default, sends a fixed
goal, records the important ROS topics, computes geometry/time/speed/control
metrics, and writes every artifact below one timestamped directory.

Example:
    rosrun hofa_mpc_ros run_mpc_experiment.py --scene random_pillars

Use ``--no-launch`` when the simulation is already running.  ROS and the
workspace must be sourced before invoking the script.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import signal
import shutil
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
from rosgraph.roslogging import RospyLogger
from geometry_msgs.msg import (AccelStamped, PoseStamped, Vector3Stamped,
                               WrenchStamped)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Time as StartTime
from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow
from hofa_mpc_ros.arc_profile import build_arc_profile


# ROS Noetic's logging caller lookup can loop indefinitely under Python 3.11.
# The policy collection environment uses Python 3.11, so use the standard
# logging implementation in this experiment-runner process.
if sys.version_info >= (3, 11):
    RospyLogger.findCaller = logging.Logger.findCaller


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

    Rebuild the frozen route's profile with the same limits and start epoch
    used online. Scoring remains independent of measured vehicle progress.

    ``t0`` is the shared ROS experiment start stamp. Legacy runs without
    start metadata retain their historical first-odometry origin.
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
            ["git", "-c", "safe.directory=" + root, "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.STDOUT).decode("utf-8").strip()
        metadata["status"] = subprocess.check_output(
            ["git", "-c", "safe.directory=" + root, "status", "--short"], cwd=root,
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
        "src/hofa_mpc_ros/experiment_gate.py",
        "launch/simulation.launch",
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
        "/hofa_mpc_controller/experiment_start_topic",
        "/hofa_mpc_controller/experiment_depth",
        "/hofa_policy_controller",
        "/hofa_policy_perturbation_relay",
        "/planar_pid_tracker/experiment_start_topic",
        "/reference_processor",
        "/privileged_teacher",
        "/nominal_reference_publisher",
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


def goal_reached_index(odom, goal_xy, start_stamp=None):
    """First odometry index satisfying the goal test, or None.

    Same predicate the live recorder applies, so a rescored run cannot
    disagree with the run that produced it.
    """
    if not odom:
        return None
    t0 = odom[0]["stamp"] if start_stamp is None else start_stamp
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
    t0 = float(model["times"][0])

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

    def __init__(self, goal_xy, timeout_s,
                 global_path_topic="/aquaflow/teacher_global_path",
                 status_topic="/hofa_mpc_controller/status"):
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        self.timeout_s = float(timeout_s)
        self.lock = threading.RLock()
        self.start_wall = time.time()
        self.start_stamp = None
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
        self.shadow_wrench = []
        self.shadow_error = []
        self.shadow_forces = []
        self.shadow_timing = []
        self.free_shadow_wrench = []
        self.free_shadow_error = []
        self.free_shadow_forces = []
        self.free_shadow_timing = []
        self.policy_forces = []
        self.policy_timing = []
        self.dagger_teacher_wrench = []

        rospy.Subscriber("/bricsbot/odometry", Odometry,
                         self._odom_cb, queue_size=20)
        rospy.Subscriber(global_path_topic, Path,
                         self._path_cb, queue_size=2)
        rospy.Subscriber("/controller/reference_trajectory_window",
                         TrajectoryPointWindow, self._window_cb, queue_size=2)
        rospy.Subscriber(status_topic, ControllerStatus,
                         self._status_cb, queue_size=20)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self._wrench_cb, queue_size=20)
        rospy.Subscriber("/hofa_mpc_controller/virtual_accel_cmd", AccelStamped,
                         self._virtual_accel_cb, queue_size=20)
        rospy.Subscriber("/bricsbot/setpoint/pwm", Float64MultiArray,
                         self._pwm_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/shadow_generalized_force",
                         WrenchStamped, self._shadow_wrench_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/shadow_error",
                         WrenchStamped, self._shadow_error_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/shadow_candidate_forces",
                         JointState, self._shadow_forces_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/shadow_timing",
                         Vector3Stamped, self._shadow_timing_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/free_shadow_generalized_force",
                         WrenchStamped, self._free_shadow_wrench_cb,
                         queue_size=20)
        rospy.Subscriber("/hofa_policy/free_shadow_error",
                         WrenchStamped, self._free_shadow_error_cb,
                         queue_size=20)
        rospy.Subscriber("/hofa_policy/free_shadow_candidate_forces",
                         JointState, self._free_shadow_forces_cb,
                         queue_size=20)
        rospy.Subscriber("/hofa_policy/free_shadow_timing",
                         Vector3Stamped, self._free_shadow_timing_cb,
                         queue_size=20)
        rospy.Subscriber("/hofa_policy_controller/candidate_forces",
                         JointState, self._policy_forces_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy_controller/timing",
                         Vector3Stamped, self._policy_timing_cb, queue_size=20)
        rospy.Subscriber("/hofa_policy/dagger_teacher_generalized_force",
                         WrenchStamped, self._dagger_teacher_wrench_cb,
                         queue_size=20)

    def reset_reference_model(self):
        """Discard any route published before this run's goal was sent."""
        with self.lock:
            self.initial_path = None

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
            "u": u, "v": v, "w": float(msg.twist.twist.linear.z),
            "vx_world": vx, "vy_world": vy,
            "speed_world": math.hypot(vx, vy),
            "r": float(msg.twist.twist.angular.z),
        }
        with self.lock:
            if self.first_odom_stamp is None:
                self.first_odom_stamp = stamp
            self.last_odom_stamp = stamp
            row["t_rel"] = stamp - (self.start_stamp if self.start_stamp is not None else self.first_odom_stamp)
            self.odom.append(row)
            if (math.hypot(row["x"] - self.goal_xy[0],
                           row["y"] - self.goal_xy[1]) < 0.25 and
                    row["speed_world"] < 0.08 and
                    self.start_stamp is not None and row["t_rel"] > 2.0):
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
                "fz": float(msg.wrench.force.z),
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

    @staticmethod
    def _wrench_row(msg):
        stamp = msg.header.stamp.to_sec()
        return {
            "stamp": stamp if stamp > 0.0 else rospy.Time.now().to_sec(),
            "fx": float(msg.wrench.force.x),
            "fy": float(msg.wrench.force.y),
            "nz": float(msg.wrench.torque.z),
        }

    def _shadow_wrench_cb(self, msg):
        with self.lock:
            self.shadow_wrench.append(self._wrench_row(msg))

    def _shadow_error_cb(self, msg):
        with self.lock:
            self.shadow_error.append(self._wrench_row(msg))

    def _shadow_forces_cb(self, msg):
        if len(msg.effort) < 4 or len(msg.position) < 4:
            return
        stamp = self._stamp(msg)
        with self.lock:
            self.shadow_forces.append({
                "stamp": stamp,
                **{"force_%d" % i: float(msg.effort[i]) for i in range(4)},
                **{"q_%d" % i: float(msg.position[i]) for i in range(4)},
            })

    def _shadow_timing_cb(self, msg):
        with self.lock:
            self.shadow_timing.append({
                "stamp": self._stamp(msg),
                "inference_ms": float(msg.vector.x),
                "odom_age_ms": 1000.0 * float(msg.vector.y),
                "reference_age_ms": 1000.0 * float(msg.vector.z),
            })

    def _free_shadow_wrench_cb(self, msg):
        with self.lock:
            self.free_shadow_wrench.append(self._wrench_row(msg))

    def _free_shadow_error_cb(self, msg):
        with self.lock:
            self.free_shadow_error.append(self._wrench_row(msg))

    def _free_shadow_forces_cb(self, msg):
        if len(msg.effort) < 4 or len(msg.position) < 4:
            return
        stamp = self._stamp(msg)
        with self.lock:
            self.free_shadow_forces.append({
                "stamp": stamp,
                **{"force_%d" % i: float(msg.effort[i]) for i in range(4)},
                **{"q_%d" % i: float(msg.position[i]) for i in range(4)},
            })

    def _free_shadow_timing_cb(self, msg):
        with self.lock:
            self.free_shadow_timing.append({
                "stamp": self._stamp(msg),
                "inference_ms": float(msg.vector.x),
                "odom_age_ms": 1000.0 * float(msg.vector.y),
                "reference_age_ms": 1000.0 * float(msg.vector.z),
            })

    def _policy_forces_cb(self, msg):
        if len(msg.effort) < 4 or len(msg.position) < 4:
            return
        stamp = self._stamp(msg)
        with self.lock:
            self.policy_forces.append({
                "stamp": stamp,
                **{"force_%d" % i: float(msg.effort[i]) for i in range(4)},
                **{"q_%d" % i: float(msg.position[i]) for i in range(4)},
            })

    def _policy_timing_cb(self, msg):
        with self.lock:
            self.policy_timing.append({
                "stamp": self._stamp(msg),
                "inference_ms": float(msg.vector.x),
                "odom_age_ms": 1000.0 * float(msg.vector.y),
                "reference_age_ms": 1000.0 * float(msg.vector.z),
            })

    def _dagger_teacher_wrench_cb(self, msg):
        with self.lock:
            self.dagger_teacher_wrench.append({
                "stamp": self._stamp(msg),
                "fx": float(msg.wrench.force.x),
                "fy": float(msg.wrench.force.y),
                "nz": float(msg.wrench.torque.z),
            })

    def evaluation_model(self, path, odom):
        """Absolute plan for ``path``, anchored at the first scored sample."""
        if not path or not odom:
            return None
        return build_evaluation_model(path, self.profile_params,
                                      self.start_stamp if self.start_stamp is not None else odom[0]["stamp"])

    def snapshot(self):
        with self.lock:
            data = {key: list(getattr(self, key)) for key in
                    ("odom", "status", "wrench", "pwm", "window",
                     "virtual_accel", "shadow_wrench", "shadow_error",
                     "shadow_forces", "shadow_timing",
                     "free_shadow_wrench", "free_shadow_error",
                     "free_shadow_forces", "free_shadow_timing",
                     "policy_forces", "policy_timing",
                     "dagger_teacher_wrench")}
            if self.start_stamp is not None:
                data = {k: [r for r in rows if r["stamp"] >= self.start_stamp]
                        for k, rows in data.items()}
                for row in data["odom"]:
                    row["t_rel"] = row["stamp"] - self.start_stamp
            data.update(path=self.initial_path or self.global_path,
                        finish_reason=self.finish_reason)
            return data

    def metrics(self):
        data = self.snapshot()
        model = self.evaluation_model(data["path"], data["odom"])
        if model is None:
            return {"error": "no odometry or global path recorded"}
        return compute_metrics(data["odom"], model, data["status"],
                               data["pwm"], self.goal_xy,
                               data["finish_reason"])

    def save(self, output_dir):
        with self.lock:
            if self.start_stamp is not None:
                write_csv(os.path.join(output_dir, "prestart_odometry.csv"),
                          [r for r in self.odom if r["stamp"] < self.start_stamp])
                write_csv(os.path.join(output_dir, "prestart_wrench.csv"),
                          [r for r in self.wrench if r["stamp"] < self.start_stamp])
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
    write_csv(os.path.join(output_dir, "shadow_generalized_force.csv"),
              data.get("shadow_wrench", []))
    write_csv(os.path.join(output_dir, "shadow_error.csv"),
              data.get("shadow_error", []))
    write_csv(os.path.join(output_dir, "shadow_candidate_forces.csv"),
              data.get("shadow_forces", []))
    write_csv(os.path.join(output_dir, "shadow_timing.csv"),
              data.get("shadow_timing", []))
    write_csv(os.path.join(output_dir, "free_shadow_generalized_force.csv"),
              data.get("free_shadow_wrench", []))
    write_csv(os.path.join(output_dir, "free_shadow_error.csv"),
              data.get("free_shadow_error", []))
    write_csv(os.path.join(output_dir, "free_shadow_candidate_forces.csv"),
              data.get("free_shadow_forces", []))
    write_csv(os.path.join(output_dir, "free_shadow_timing.csv"),
              data.get("free_shadow_timing", []))
    write_csv(os.path.join(output_dir, "policy_candidate_forces.csv"),
              data.get("policy_forces", []))
    write_csv(os.path.join(output_dir, "policy_timing.csv"),
              data.get("policy_timing", []))
    write_csv(os.path.join(output_dir,
                           "dagger_teacher_generalized_force.csv"),
              data.get("dagger_teacher_wrench", []))

    t0 = float(model["times"][0])
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
    t0 = float(model["times"][0])
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
    start_stamp = None
    start_file = os.path.join(run_dir, "experiment_start.json")
    if os.path.isfile(start_file):
        with open(start_file) as stream:
            start_stamp = json.load(stream)["start_stamp"]
    reached = goal_reached_index(odom, goal_xy, start_stamp)
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
    start_file = os.path.join(run_dir, "experiment_start.json")
    start_stamp = data["odom"][0]["stamp"]
    if os.path.isfile(start_file):
        with open(start_file) as stream:
            start_stamp = json.load(stream)["start_stamp"]
    model = build_evaluation_model(data["path"], profile_params, start_stamp)
    metrics = write_artifacts(run_dir, data, model, goal_xy)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Rescored: %s" % run_dir)
    return metrics


class ExperimentBarrier:
    """Wait on fresh data and a calm vehicle, then negotiate a shared epoch."""
    def __init__(self, recorder, topic, controller, depth, timeout):
        self.recorder, self.topic = recorder, topic
        self.controller = {
            "hofa_mpc": "/hofa_mpc_controller",
            "hofa_policy": "/hofa_policy_controller",
            "hofa_policy_dagger": "/hofa_policy_controller",
            "pid": "/planar_pid_tracker",
        }[controller]
        self.roles = ("reference", "controller")
        self.required_nodes = (self.controller, "/reference_processor")
        if controller == "hofa_policy_dagger":
            self.roles += ("dagger_teacher",)
            self.required_nodes += ("/hofa_policy_dagger_teacher",)
        self.depth, self.timeout = depth, timeout
        self.acks = {}
        self.pub = rospy.Publisher(topic, StartTime, queue_size=1, latch=True)
        self.subs = [rospy.Subscriber(topic + '/' + role + '_ack', StartTime,
                     lambda msg, role=role: self.acks.update({role: msg.data.to_sec()}),
                     queue_size=1) for role in self.roles]

    def wait_ready(self, require_path, goal_stamp=0.):
        deadline = time.monotonic() + self.timeout
        stable_since = None
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            with self.recorder.lock:
                row = self.recorder.odom[-1] if self.recorder.odom else None
                path = self.recorder.global_path
                window = self.recorder.window[-1] if self.recorder.window else None
                wrench = self.recorder.wrench[-1] if self.recorder.wrench else None
                pwm = self.recorder.pwm[-1] if self.recorder.pwm else None
            configured = all(
                rospy.get_param(node + '/experiment_start_topic', '') ==
                self.topic for node in self.required_nodes)
            ready = (configured and self.pub.get_num_connections() >= 2 and
                     row is not None and 0 <= now-row['stamp'] < .25 and
                     row['speed_world'] < .01 and abs(row['r']) < .02 and abs(row['w']) < .02 and
                     abs(row['z']-self.depth) < .05 and
                     wrench is not None and 0 <= now-wrench['stamp'] < .3 and
                     all(abs(wrench[k]) < 1e-9 for k in ('fx', 'fy', 'nz')) and
                     pwm is not None and 0 <= now-pwm['stamp'] < .3 and
                     len(pwm['pwm']) >= 6 and all(abs(v) < 1e-9 for v in pwm['pwm'][:4]))
            if require_path:
                ready = (ready and path is not None and path['stamp'] >= goal_stamp and
                         math.hypot(path['points'][-1][1]-self.recorder.goal_xy[0],
                                    path['points'][-1][2]-self.recorder.goal_xy[1]) < .35 and
                         window is not None and window['stamp'] >= goal_stamp and
                         0 <= now-window['stamp'] < .3 and
                         rospy.get_param('/reference_processor/progress_mode', 'schedule') == 'schedule')
            if ready:
                if stable_since is None:
                    stable_since = now
                if now-stable_since >= 1.:
                    if require_path:
                        with self.recorder.lock:
                            self.recorder.initial_path = path
                    return
            else:
                stable_since = None
            if time.monotonic() > deadline:
                raise RuntimeError("Experiment not ready: require fresh odometry/reference, "
                                   "armed controller, allocator, depth within 5 cm and horizontal rest")
            time.sleep(.05)
        raise RuntimeError("ROS shut down before experiment readiness")

    def start(self, output_dir):
        start_time = rospy.Time.now() + rospy.Duration(1.0)
        start_stamp = start_time.to_sec()
        self.acks.clear()
        # Arm recorder before any node can move. Never clear captured samples.
        self.recorder.start_stamp = start_stamp
        self.recorder.done = False
        self.pub.publish(StartTime(data=start_time))
        deadline = time.monotonic() + 2.0
        while not all(self.acks.get(role) == start_stamp
                      for role in self.roles):
            if rospy.is_shutdown() or rospy.Time.now().to_sec() >= start_stamp-.2 or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("Start acknowledgement missing; experiment cancelled")
            time.sleep(.01)
        with open(os.path.join(output_dir, 'experiment_start.json'), 'w') as stream:
            json.dump(dict(start_stamp=start_stamp, topic=self.topic, acks=self.acks,
                           depth_target_m=self.depth, protocol='acknowledged_future_ros_time'), stream, indent=2)
        rospy.loginfo("Experiment armed: shared start %.9f", start_stamp)

    def stop(self):
        try:
            self.pub.publish(StartTime(data=rospy.Time()))
            time.sleep(.1)
        except rospy.ROSException:
            pass  # process cleanup must still run after ROS shutdown



def publish_goal(goal_xy, depth=1.0):
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.header.frame_id = "world_ned"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = float(goal_xy[0])
    msg.pose.position.y = float(goal_xy[1])
    msg.pose.position.z = float(depth)
    msg.pose.orientation.w = 1.0
    deadline = time.monotonic() + 10.
    while pub.get_num_connections() == 0:
        if rospy.is_shutdown() or time.monotonic() > deadline:
            raise RuntimeError("Goal subscriber not ready")
        time.sleep(.02)
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    return pub, msg.header.stamp.to_sec()


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


BC_TOPICS = ["/bricsbot/camera_front/image_color", "/bricsbot/camera_front/camera_info",
             "/bricsbot/imu", "/bricsbot/pressure", "/tf", "/tf_static",
             "/aquaflow/nominal_path", "/aquaflow/teacher_reference",
             "/aquaflow/teacher_labels", "/move_base_simple/goal"]


def prepare_bc_course(args, output_dir):
    """Keep generated scene, teacher geometry and task provenance together."""
    if not args.bc_course:
        return None
    course_path = os.path.abspath(args.bc_course)
    with open(course_path) as f:
        course = json.load(f)
    if course.get("vehicle") != "bricsbot" or course.get("schema") not in (
            "bricsbot_course_v1", "bricsbot_course_v2"):
        raise ValueError("BC collection requires a generated BricsBot course")
    if course.get("obstacle_look") != "pillar_black":
        raise ValueError("BC collection requires black pillars; use the v2 course suite")
    for name, key in (("scenario.scn", "scenario_sha256"), ("map.yaml", "map_sha256")):
        source = os.path.join(os.path.dirname(course_path), name)
        if _sha256_file(source) != course[key]:
            raise ValueError("BC course hash mismatch: " + name)
        shutil.copy2(source, os.path.join(output_dir, name))
    shutil.copy2(course_path, os.path.join(output_dir, "course.json"))
    # The generated scenario is authoritative.  Keep the scene argument
    # compatible with the launch defaults for auxiliary nodes while allowing
    # mixed 0/1/2/3-pillar courses to carry their own geometry.
    if course.get("schema") == "bricsbot_course_v2":
        args.scene = "random_pillars" if course["map"]["obstacles"] else "empty_pool"
    else:
        args.scene = "single_obstacle" if course["map"]["obstacles"] else "empty_pool"
    args.goal_x, args.goal_y, args.goal_depth = course["goal"]
    return course


def prepare_policy_trajectory(args, output_dir):
    """Validate and snapshot an obstacle-free hofa_policy trajectory."""
    if not args.policy_trajectory:
        return None
    source = os.path.abspath(args.policy_trajectory)
    with open(source) as stream:
        track = json.load(stream)
    if track.get("schema") != "hofa_policy_trajectory_v1":
        raise ValueError("policy trajectory schema mismatch")
    points = track.get("points", [])
    if len(points) < 20 or any(len(point) != 4 for point in points):
        raise ValueError("policy trajectory requires at least 20 [x,y,z,yaw] points")
    if not all(all(math.isfinite(float(value)) for value in point)
               for point in points):
        raise ValueError("policy trajectory contains nonfinite values")
    snapshot = os.path.join(output_dir, "policy_trajectory.json")
    shutil.copy2(source, snapshot)
    args.scene = "empty_pool_control"
    args.goal_x = float(points[-1][0])
    args.goal_y = float(points[-1][1])
    args.goal_depth = float(points[-1][2])
    return track, snapshot


class BCCaptureGuard:
    def __init__(self):
        from sensor_msgs.msg import Image, CameraInfo, Imu
        self.latest = {}
        self.subs = []
        types = [(BC_TOPICS[0], Image), (BC_TOPICS[1], CameraInfo),
                 ("/bricsbot/imu", Imu), ("/aquaflow/nominal_path", Path)]
        for topic, cls in types:
            self.subs.append(rospy.Subscriber(topic, cls, self.receive,
                             callback_args=topic, queue_size=1, buff_size=4000000))

    def receive(self, msg, topic):
        self.latest[topic] = msg

    def wait(self, process, goal, timeout):
        deadline = time.monotonic()+timeout
        required = set(BC_TOPICS[:3] + ["/aquaflow/nominal_path"])
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if process.poll() is not None:
                raise RuntimeError("BC rosbag recorder exited; inspect rosbag.log")
            now = rospy.Time.now().to_sec()
            fresh = all(t in self.latest and 0 <= now-self.latest[t].header.stamp.to_sec() < .7
                        for t in required)
            nominal = self.latest.get("/aquaflow/nominal_path")
            same_goal = nominal is not None and nominal.poses and math.hypot(
                nominal.poses[-1].pose.position.x-goal[0],
                nominal.poses[-1].pose.position.y-goal[1]) < .01
            # Require rosbag subscriptions before releasing the controller.
            subscribers = rosgraph.Master(rospy.get_name()).getSystemState()[1]
            recording = {t for t, nodes in subscribers if "/aquaflow_bc_recorder" in nodes}
            if fresh and same_goal and required <= recording:
                return
            time.sleep(.05)
        raise RuntimeError("BC camera/state/nominal/recorder readiness timeout")


def snapshot_bc_sources(output_dir):
    root = os.path.abspath(os.path.join(rospkg.RosPack().get_path("hofa_mpc_ros"), "../.."))
    files = []
    for package in ("aquaflow_bc", "aquaflow_ref", "aquaflow_stonefish", "hofa_mpc_ros"):
        for directory, _, names in os.walk(os.path.join(root, "src", package)):
            for name in names:
                if name.endswith((".py", ".yaml", ".launch", ".scn", ".msg")):
                    files.append(os.path.join(directory, name))
    for name in ("bricsbot_stonefish_physical_proxy_5000_no_top_bump_pre_refined_no_normals.obj",
                 "bricsbot_stonefish_baked.obj", "bricsbot.png"):
        files.append(os.path.join(root, "src/aquaflow_stonefish/data/bricsbot", name))
    # Record the actual RGB bridge source and built ROS library used by BC.
    for relative in ("src/stonefish_ros/src/ROSSimulationManager.cpp",
                     "src/stonefish_ros/include/stonefish_ros/ROSSimulationManager.h",
                     "src/stonefish_ros/include/stonefish_ros/ImageStampGate.h",
                     "devel/lib/libstonefish_ros.so"):
        filename = os.path.join(root, relative)
        if not os.path.isfile(filename):
            raise RuntimeError("Missing RGB publisher provenance: " + filename)
        files.append(filename)
    hashes = {}
    for filename in files:
        rel = os.path.relpath(filename, root)
        hashes[rel] = _sha256_file(filename)
        # Copy code/configuration, hash large robot assets without duplicating them.
        if not filename.endswith((".obj", ".png", ".so")):
            dest = os.path.join(output_dir, "source_snapshot", rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(filename, dest)
    with open(os.path.join(output_dir, "source_manifest.json"), "w") as f:
        json.dump({"git": _git_metadata(root), "sha256": hashes}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default=None,
        help="result root; defaults depend on the selected controller")
    parser.add_argument("--launch-pkg", default="hofa_mpc_ros")
    parser.add_argument("--launch-file", default="simulation.launch")
    parser.add_argument("--controller",
                        choices=("hofa_mpc", "hofa_policy",
                                 "hofa_policy_dagger", "pid"),
                        default="hofa_mpc",
                        help="backend controller launched by simulation.launch")
    parser.add_argument("--scene", default="random_pillars")
    parser.add_argument("--scenario-file",
                        help="explicit Stonefish scenario file override")
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-depth", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-wait", type=float, default=5.0,
                        help="deprecated; startup is now readiness-driven")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--experiment-start-topic", default=None,
                        help="for --no-launch: topic configured on BOTH gated nodes")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-bag", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="use Stonefish parsed_simulator_nogpu")
    parser.add_argument(
        "--rescore", metavar="RUN_DIR", default=None,
        help="recompute metrics and plots for an existing run directory from "
             "its CSVs and exit; needs no simulator and no ROS master")
    parser.add_argument("--bc-course", help="generated course.json; enables complete BC recording")
    parser.add_argument(
        "--policy-trajectory",
        help="hofa_policy trajectory JSON; disables privileged path planning")
    parser.add_argument("--shadow-checkpoint",
                        help="enable hofa_policy Shadow with this checkpoint")
    parser.add_argument("--shadow-device", default="cpu")
    parser.add_argument("--policy-checkpoint",
                        help="BC checkpoint used by hofa_policy closed loop")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--policy-depth-kp", type=float, default=8.0)
    parser.add_argument("--policy-depth-kd", type=float, default=4.0)
    parser.add_argument("--policy-depth-ki", type=float, default=0.5)
    parser.add_argument("--policy-depth-integral-limit", type=float,
                        default=5.0)
    parser.add_argument("--policy-depth-max-force", type=float, default=15.0)
    parser.add_argument("--robustness-config",
                        help="JSON perturbation config for hofa_policy relay")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.bc_course and (args.headless or args.no_bag or args.no_launch or args.controller != "hofa_mpc"):
        parser.error("--bc-course requires rendered simulation, rosbag, launch and HOFA-MPC")
    if args.bc_course and args.policy_trajectory:
        parser.error("--bc-course and --policy-trajectory are mutually exclusive")
    if args.bc_course and args.scenario_file:
        parser.error("--bc-course and --scenario-file are mutually exclusive")
    if args.policy_trajectory and (
            args.no_launch or
            args.controller not in ("hofa_mpc", "hofa_policy",
                                    "hofa_policy_dagger")):
        parser.error(
            "--policy-trajectory requires launch and a supported controller")
    if args.shadow_checkpoint and (args.no_launch or args.controller != "hofa_mpc"):
        parser.error("--shadow-checkpoint requires launch and HOFA-MPC")
    if args.controller in ("hofa_policy", "hofa_policy_dagger") and not args.policy_checkpoint:
        parser.error("%s requires --policy-checkpoint" % args.controller)
    if args.policy_checkpoint and args.controller not in (
            "hofa_policy", "hofa_policy_dagger"):
        parser.error("--policy-checkpoint requires a policy controller")
    if (args.policy_checkpoint and
            not os.path.isfile(os.path.abspath(args.policy_checkpoint))):
        parser.error("policy checkpoint does not exist: %s" %
                     args.policy_checkpoint)
    if args.scenario_file and not os.path.isfile(
            os.path.abspath(args.scenario_file)):
        parser.error("scenario file does not exist: %s" % args.scenario_file)
    if args.robustness_config and args.controller != "hofa_policy":
        parser.error("--robustness-config requires --controller hofa_policy")
    if args.robustness_config and not os.path.isfile(
            os.path.abspath(args.robustness_config)):
        parser.error("robustness config does not exist: %s" %
                     args.robustness_config)

    if args.rescore:
        rescore(args.rescore)
        return 0

    if args.output_root is None:
        default_roots = {
            "hofa_mpc": "mpc",
            "hofa_policy": os.path.join("hofa_policy", "deployment"),
            "hofa_policy_dagger": os.path.join(
                "hofa_policy", "dagger_raw"),
            "pid": "pid",
        }
        args.output_root = os.path.join("results", default_roots[args.controller])
    output_dir = make_run_dir(os.path.abspath(args.output_root))
    course = prepare_bc_course(args, output_dir)
    policy_trajectory = prepare_policy_trajectory(args, output_dir)
    global_path_topic = ("/hofa_policy/teacher_global_path"
                         if policy_trajectory else
                         "/aquaflow/teacher_global_path")
    goal_xy = (args.goal_x, args.goal_y)
    launch_proc = None
    bag_proc = None
    roscore_proc = None
    barrier = None
    try:
        if not rosgraph.is_master_online():
            roscore_proc = subprocess.Popen(["roscore"], start_new_session=True)
            deadline = time.time() + 15.0
            while not rosgraph.is_master_online() and time.time() < deadline:
                time.sleep(0.2)
            if not rosgraph.is_master_online():
                raise RuntimeError("roscore did not become available")

        rospy.init_node("mpc_experiment_runner", anonymous=True)
        controller_namespace = {
            "hofa_mpc": "/hofa_mpc_controller",
            "hofa_policy": "/hofa_policy_controller",
            "hofa_policy_dagger": "/hofa_policy_controller",
            "pid": "/planar_pid_tracker",
        }[args.controller]
        controller_status_topic = controller_namespace + "/status"
        recorder = ExperimentRecorder(
            goal_xy, args.timeout, global_path_topic=global_path_topic,
            status_topic=controller_status_topic)
        bc_guard = BCCaptureGuard() if course else None
        if args.no_launch and not args.experiment_start_topic:
            raise RuntimeError("--no-launch requires --experiment-start-topic and pre-armed nodes")
        start_topic = args.experiment_start_topic or ("/experiment/" + os.path.basename(output_dir) + "/start")
        barrier = ExperimentBarrier(recorder, start_topic, args.controller,
                                    args.goal_depth, args.ready_timeout)

        if not args.no_launch:
            launch_cmd = ["roslaunch", args.launch_pkg, args.launch_file,
                          "controller:=" + args.controller,
                          "vehicle_name:=bricsbot",
                          "vehicle_model:=bricsbot", "scene:=" + args.scene,
                          "enable:=true", "progress_mode:=schedule",
                          "experiment_start_topic:=" + start_topic,
                          "experiment_depth:=" + str(args.goal_depth)]
            if course:
                launch_cmd += ["scenario_file:=" + os.path.join(output_dir, "scenario.scn"),
                               "map_file:=" + os.path.join(output_dir, "map.yaml"),
                               "nominal_start_x:=" + str(course["start"][0]),
                               "nominal_start_y:=" + str(course["start"][1])]
            elif args.scenario_file:
                launch_cmd.append(
                    "scenario_file:=" + os.path.abspath(args.scenario_file))
            if policy_trajectory:
                track, snapshot = policy_trajectory
                launch_cmd += [
                    "reference_source:=policy_trajectory",
                    "reference_topic:=/hofa_policy/teacher_global_path",
                    "policy_trajectory_file:=" + snapshot,
                    "reference_max_speed:=" + str(track["max_speed_mps"]),
                    "reference_max_yaw_rate:=" + str(track["max_yaw_rate_radps"]),
                    "reference_max_accel:=" + str(track["max_accel_mps2"]),
                    "reference_max_decel:=" + str(track["max_decel_mps2"]),
                ]
            if args.headless:
                launch_cmd.append("headless:=true")
            if args.shadow_checkpoint:
                launch_cmd += [
                    "shadow_enable:=true",
                    "shadow_checkpoint:=" + os.path.abspath(args.shadow_checkpoint),
                    "shadow_device:=" + args.shadow_device,
                ]
            if args.controller == "hofa_policy":
                launch_cmd += [
                    "policy_checkpoint:=" +
                    os.path.abspath(args.policy_checkpoint),
                    "policy_device:=" + args.policy_device,
                    "policy_depth_kp:=" + str(args.policy_depth_kp),
                    "policy_depth_kd:=" + str(args.policy_depth_kd),
                    "policy_depth_ki:=" + str(args.policy_depth_ki),
                    "policy_depth_integral_limit:=" +
                    str(args.policy_depth_integral_limit),
                    "policy_depth_max_force_n:=" +
                    str(args.policy_depth_max_force),
                ]
            if args.controller == "hofa_policy_dagger":
                launch_cmd += [
                    "policy_checkpoint:=" +
                    os.path.abspath(args.policy_checkpoint),
                    "policy_device:=" + args.policy_device,
                    "policy_depth_kp:=" + str(args.policy_depth_kp),
                    "policy_depth_kd:=" + str(args.policy_depth_kd),
                    "policy_depth_ki:=" + str(args.policy_depth_ki),
                    "policy_depth_integral_limit:=" +
                    str(args.policy_depth_integral_limit),
                    "policy_depth_max_force_n:=" +
                    str(args.policy_depth_max_force),
                ]
            if args.robustness_config:
                launch_cmd += [
                    "robustness_enable:=true",
                    "robustness_config:=" +
                    os.path.abspath(args.robustness_config),
                ]
            launch_proc = subprocess.Popen(launch_cmd, start_new_session=True)

        if not args.no_bag:
            bag_file = os.path.join(output_dir, "topics.bag")
            bag_topics = [start_topic, start_topic+"/reference_ack", start_topic+"/controller_ack",
                          "/clock", "/bricsbot/odometry",
                          global_path_topic,
                          "/controller/reference_path",
                          "/controller/reference_velocity",
                          "/controller/reference_trajectory_window",
                          controller_status_topic,
                          "/hofa_policy_dagger_teacher/status",
                          "/hofa_mpc_controller/virtual_accel_cmd",
                          "/controller/generalized_force",
                          "/hofa_policy_controller/candidate_forces",
                          "/hofa_policy_controller/timing",
                          "/hofa_policy/dagger_teacher_generalized_force",
                          "/hofa_policy/robust/odometry",
                          "/hofa_policy/robust/reference_trajectory_window",
                          "/hofa_policy/robust/raw_wrench",
                          "/hofa_policy/shadow_generalized_force",
                          "/hofa_policy/shadow_error",
                          "/hofa_policy/shadow_candidate_forces",
                          "/hofa_policy/shadow_timing",
                          "/hofa_policy/free_shadow_generalized_force",
                          "/hofa_policy/free_shadow_error",
                          "/hofa_policy/free_shadow_candidate_forces",
                          "/hofa_policy/free_shadow_timing",
                          "/bricsbot/setpoint/pwm"]
            if course:
                bag_topics += BC_TOPICS
            try:
                bag_command = ["rosbag", "record", "-O", bag_file]
                if course:
                    bag_command += ["--lz4", "__name:=aquaflow_bc_recorder"]
                bag_log = open(os.path.join(output_dir, "rosbag.log"), "w")
                bag_proc = subprocess.Popen(
                    bag_command + bag_topics,
                    start_new_session=True,
                    stdout=bag_log, stderr=subprocess.STDOUT)
                bag_log.close()
            except OSError:
                if course:
                    raise
                print("warning: rosbag not found; continuing without bag",
                      file=sys.stderr)

        barrier.wait_ready(require_path=False)
        recorder.profile_params = rospy.get_param("/reference_processor", {})
        recorder.reset_reference_model()
        goal_pub, goal_stamp = publish_goal(goal_xy, args.goal_depth)
        barrier.wait_ready(require_path=True, goal_stamp=goal_stamp)
        if bc_guard:
            bc_guard.wait(bag_proc, goal_xy, args.ready_timeout)
            snapshot_bc_sources(output_dir)
        write_runtime_config(output_dir, args)
        barrier.start(output_dir)
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < args.timeout:
            if bc_guard and bag_proc.poll() is not None:
                raise RuntimeError("BC rosbag recorder failed during collection")
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
        if barrier is not None:
            barrier.stop()
        terminate_process(bag_proc, "rosbag")
        terminate_process(launch_proc, "roslaunch")
        terminate_process(roscore_proc, "roscore")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        sys.exit(130)
