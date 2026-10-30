#!/usr/bin/env python3
"""Run a deterministic sharp-turn tracking experiment.

The experiment publishes a synthetic world_ned route with explicit heading
wraps and large tangent changes, then launches the selected controller
(``pid``, ``hofa_mpc`` or ``hofa_policy``) through ``simulation.launch``.
Artifacts use the same CSV/metrics layout as ``run_mpc_experiment.py`` while
also recording the planned yaw, yaw-rate and curvature so reference-induced
turning can be separated from controller-induced turning.

Run inside the sourced ROS/Stonefish environment, for example:

    rosrun hofa_mpc_ros run_tracking_experiment.py --controller hofa_mpc \
      --turns hairpin,switchback --headless
"""
from __future__ import print_function

import argparse
import csv
import copy
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped, WrenchStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray, String
from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow
from hofa_mpc_ros.arc_profile import build_arc_profile, project_to_polyline
from hofa_mpc_ros.tracking_routes import build_route, route_sha256, sharp_route


def _load_plotter():
    """Load plotting code from the built package or the source checkout."""
    try:
        from hofa_mpc_ros.tracking_plots import generate_run_plots
        return generate_run_plots
    except ImportError:
        source = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../src/hofa_mpc_ros/tracking_plots.py"))
        if not os.path.isfile(source):
            return None
        spec = importlib.util.spec_from_file_location(
            "hofa_mpc_ros.tracking_plots", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_run_plots


# Plotting is optional; the experiment and its CSV artifacts must remain
# usable on a ROS installation without matplotlib.
generate_run_plots = _load_plotter()


CONTROLLERS = ("pid", "hofa_mpc", "hofa_policy")
GOAL_RADIUS = 0.25
GOAL_SPEED = 0.08


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def make_dir(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, "run_" + stamp)
    index = 1
    while os.path.exists(path):
        path = os.path.join(root, "run_%s_%02d" % (stamp, index))
        index += 1
    os.makedirs(path)
    return path


def publish_route(route, publisher):
    msg = Path()
    msg.header.frame_id = route["frame_id"]
    msg.header.stamp = rospy.Time.now()
    for x, y, z, yaw in route["points"]:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        q = quaternion(yaw)
        pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
        pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
        msg.poses.append(pose)
    publisher.publish(msg)
    return msg.header.stamp.to_sec()


class Recorder:
    def __init__(self, controller, goal):
        self.controller = controller
        self.goal = goal
        self.odom = []
        self.status = []
        self.wrench = []
        self.pwm = []
        self.window = []
        self.residual_diagnostics = []
        self.path = None
        self.start = None
        self.done = False
        self.finish_reason = "timeout"
        status_topic = {
            "pid": "/planar_pid_tracker/status",
            "hofa_mpc": "/hofa_mpc_controller/status",
            "hofa_policy": "/hofa_policy_controller/status",
        }[controller]
        rospy.Subscriber("/bricsbot/odometry", Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self.wrench_cb, queue_size=50)
        rospy.Subscriber("/bricsbot/setpoint/pwm", Float64MultiArray,
                         self.pwm_cb, queue_size=50)
        rospy.Subscriber(status_topic, ControllerStatus,
                         self.status_cb, queue_size=50)
        rospy.Subscriber("/controller/reference_trajectory_window",
                         TrajectoryPointWindow, self.window_cb, queue_size=10)
        rospy.Subscriber("/hofa_policy_controller/residual_diagnostics",
                         String, self.residual_diagnostics_cb, queue_size=100)
        rospy.Subscriber("/tracking_test/reference", Path,
                         self.path_cb, queue_size=2)

    @staticmethod
    def stamp(msg):
        value = msg.header.stamp.to_sec()
        return value if value > 0.0 else rospy.Time.now().to_sec()

    def path_cb(self, msg):
        if len(msg.poses) >= 2:
            self.path = msg

    def odom_cb(self, msg):
        stamp = self.stamp(msg)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q)
        u = float(msg.twist.twist.linear.x)
        v = float(msg.twist.twist.linear.y)
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = c * u - s * v, s * u + c * v
        row = {
            "stamp": stamp, "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y), "z": float(msg.pose.pose.position.z),
            "yaw": yaw, "u": u, "v": v, "r": float(msg.twist.twist.angular.z),
            "vx_world": vx, "vy_world": vy,
            "speed_world": math.hypot(vx, vy),
        }
        self.odom.append(row)
        if self.start is not None:
            row["t_rel"] = stamp - self.start
            if (math.hypot(row["x"] - self.goal[0], row["y"] - self.goal[1]) < GOAL_RADIUS
                    and row["speed_world"] < GOAL_SPEED and row["t_rel"] > 2.0):
                self.done = True
                self.finish_reason = "goal_reached"

    def status_cb(self, msg):
        self.status.append({
            "stamp": self.stamp(msg), "state": str(msg.state),
            "solver_success": bool(msg.solver_success),
            "callback_time_ms": float(msg.callback_time_ms),
            "position_error_m": float(msg.position_error_m),
            "yaw_error_rad": float(msg.yaw_error_rad),
        })

    def wrench_cb(self, msg):
        self.wrench.append({
            "stamp": self.stamp(msg), "fx": float(msg.wrench.force.x),
            "fy": float(msg.wrench.force.y), "fz": float(msg.wrench.force.z),
            "nz": float(msg.wrench.torque.z),
        })

    def pwm_cb(self, msg):
        self.pwm.append({"stamp": rospy.Time.now().to_sec(),
                         "pwm": [float(v) for v in msg.data]})

    def residual_diagnostics_cb(self, msg):
        self.residual_diagnostics.append(json.loads(msg.data))

    def window_cb(self, msg):
        if not msg.valid:
            return
        stamp = self.stamp(msg)
        for index, point in enumerate(msg.points):
            self.window.append({
                "stamp": stamp, "index": index,
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


def route_diagnostics(route, profile_params):
    points = route["points"]
    profile = build_arc_profile(
        [p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
        max_speed=float(profile_params.get("max_speed", 0.2)),
        max_yaw_rate=float(profile_params.get("max_yaw_rate", 0.5)),
        max_yaw_accel=float(profile_params.get("max_yaw_accel_radps2", 0.25)),
        max_accel=float(profile_params.get("max_accel_mps2", 0.12)),
        max_decel=float(profile_params.get("max_decel_mps2", 0.18)),
        yaw_smoothing_window=int(profile_params.get("yaw_smoothing_window", 5)),
        yaw_accel_reserve_ratio=float(profile_params.get("yaw_accel_reserve_ratio", 0.2)),
        initial_speed=0.0)
    return profile


def project(x, y, profile, arc_hint=None):
    return project_to_polyline(x, y, profile, arc_hint=arc_hint)


def stats(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return {"rmse": float(np.sqrt(np.mean(array * array))),
            "mean": float(np.mean(array)), "p95_abs": float(np.percentile(np.abs(array), 95)),
            "max_abs": float(np.max(np.abs(array)))}


def evaluate(recorder, route, profile, goal):
    odom = recorder.odom
    if not odom:
        return {"error": "no odometry"}
    t0 = recorder.start if recorder.start is not None else odom[0]["stamp"]
    cross, geom, yaw_error, turn_yaw_error, turn_rate_error = [], [], [], [], []
    actual_s = []
    arc_hint = 0.0
    for row in odom:
        s, distance, signed = project(row["x"], row["y"], profile, arc_hint)
        arc_hint = s
        actual_s.append(s)
        reference_yaw = float(np.interp(s, profile["s"], profile["yaw"]))
        reference_rate = float(np.interp(s, profile["s"], profile["speed"] * profile["curvature"]))
        yaw_err = wrap(row["yaw"] - reference_yaw)
        cross.append(signed)
        geom.append(distance)
        yaw_error.append(yaw_err)
        if abs(reference_rate) >= 0.05:
            turn_yaw_error.append(yaw_err)
            turn_rate_error.append(row["r"] - reference_rate)
    duration = max(0.0, odom[-1]["stamp"] - t0)
    speeds = [row["speed_world"] for row in odom]
    pwm_values = np.asarray([row["pwm"] for row in recorder.pwm if row["pwm"]],
                            dtype=float)
    saturation_ratio = (float(np.mean(np.any(np.abs(pwm_values) >= 0.99, axis=1)))
                        if pwm_values.size else None)
    return {
        "controller": recorder.controller,
        "route": route["name"], "finish_reason": recorder.finish_reason,
        "route_randomized": bool(route.get("randomized", False)),
        "route_seed": route.get("route_seed"),
        "route_sha256": route.get("route_sha256"),
        "duration_s": duration, "samples": len(odom),
        "path_length_m": float(profile["total_length"]),
        "completion_ratio": float(actual_s[-1] / max(profile["total_length"], 1e-9)),
        "final_position_m": [odom[-1]["x"], odom[-1]["y"]],
        "final_distance_to_goal_m": math.hypot(odom[-1]["x"] - goal[0], odom[-1]["y"] - goal[1]),
        "final_yaw_error_deg": math.degrees(wrap(odom[-1]["yaw"] - profile["yaw"][-1])),
        "final_speed_mps": float(speeds[-1]),
        "mean_speed_mps": float(np.mean(speeds)), "max_speed_mps": float(np.max(speeds)),
        "reference_max_yaw_rate_radps": float(np.max(np.abs(profile["speed"] * profile["curvature"]))),
        "reference_max_curvature_radpm": float(np.max(np.abs(profile["curvature"]))),
        "reference_min_speed_mps": float(np.min(profile["speed"][1:-1])),
        "cross_track_error_m": stats(cross), "geometric_path_error_m": stats(geom),
        "yaw_error_deg": stats(np.degrees(yaw_error)),
        "turn_yaw_error_deg": stats(np.degrees(turn_yaw_error)),
        "turn_yaw_rate_error_radps": stats(turn_rate_error),
        "status_samples": len(recorder.status),
        "solver_success_rate": (float(np.mean([s["solver_success"] for s in recorder.status]))
                                 if recorder.status else None),
        "pwm_samples": len(recorder.pwm),
        "pwm_saturation_ratio": saturation_ratio,
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_route_json(path, expected_name, expected_depth):
    with open(path) as stream:
        route = json.load(stream)
    points = np.asarray(route.get("points"), dtype=float)
    goal = np.asarray(route.get("goal"), dtype=float)
    if route.get("name") != expected_name:
        raise ValueError("route JSON name differs from --route")
    if points.ndim != 2 or points.shape[1] != 4 or len(points) < 2:
        raise ValueError("route JSON points must be a finite N x 4 array")
    if not np.isfinite(points).all() or goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("route JSON points and goal must be finite")
    if not np.allclose(goal[:2], points[-1, :2], rtol=0.0, atol=1.0e-9):
        raise ValueError("route JSON goal must match its final point")
    if not math.isclose(float(route.get("depth", float("nan"))),
                        float(expected_depth), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("route JSON depth differs from --goal-depth")
    if route.get("frame_id") != "world_ned":
        raise ValueError("route JSON frame_id must be world_ned")
    route["points"] = points.tolist()
    route["goal"] = goal.tolist()
    digest = route_sha256(route)
    if route.get("route_sha256") not in (None, digest):
        raise ValueError("route JSON SHA256 does not match its content")
    route["route_sha256"] = digest
    return route


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
        return
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLERS, default="hofa_mpc")
    parser.add_argument("--route", choices=("hairpin", "switchback", "right_angle"),
                        default="hairpin")
    parser.add_argument("--turns", help="comma-separated routes; overrides --route")
    parser.add_argument("--route-seed", type=int,
                        help="seeded route-family perturbation matching residual native evaluation")
    parser.add_argument("--route-json",
                        help="exact route JSON; mutually exclusive with --route-seed and --turns")
    parser.add_argument("--output-root", default="results/tracking_sharp_turns")
    parser.add_argument("--checkpoint", help="hofa_policy checkpoint")
    parser.add_argument("--residual-model", help="optional frozen-DAgger TD3/SAC residual .zip")
    parser.add_argument("--residual-algo", choices=("td3", "sac"), default="td3")
    parser.add_argument("--policy-python", help="Python with Torch and SB3 for a residual policy")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--scene", default="empty_pool_control")
    parser.add_argument("--scenario-file")
    parser.add_argument("--goal-depth", type=float, default=1.0)
    parser.add_argument("--max-yaw-rate", type=float, default=0.5,
                        help="reference yaw-rate limit in rad/s")
    parser.add_argument("--max-yaw-accel", type=float, default=0.25,
                        help="reference yaw-acceleration limit in rad/s^2")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-bag", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.controller == "hofa_policy" and not args.checkpoint:
        parser.error("--checkpoint is required for hofa_policy")
    if args.residual_model and args.controller != "hofa_policy":
        parser.error("--residual-model requires --controller hofa_policy")
    if args.residual_model and not os.path.isfile(args.residual_model):
        parser.error("residual model does not exist: %s" % args.residual_model)
    if args.checkpoint and not os.path.isfile(os.path.abspath(args.checkpoint)):
        parser.error("checkpoint does not exist: %s" % args.checkpoint)
    if args.route_json and args.turns:
        parser.error("--route-json cannot be combined with --turns")
    if args.route_seed is not None and args.route_seed < 0:
        parser.error("--route-seed must be non-negative")
    routes = [x.strip() for x in (args.turns.split(",") if args.turns else [args.route]) if x.strip()]
    for name in routes:
        sharp_route(name)
    route_override = None
    if args.route_json:
        if len(routes) != 1:
            parser.error("--route-json requires exactly one route")
        try:
            route_override = read_route_json(args.route_json, routes[0], args.goal_depth)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            parser.error("invalid --route-json: %s" % error)
    root = os.path.abspath(args.output_root)
    os.makedirs(root, exist_ok=True)
    if not rosgraph.is_master_online():
        subprocess.Popen(["roscore"], start_new_session=True)
        deadline = time.time() + 15.0
        while not rosgraph.is_master_online() and time.time() < deadline:
            time.sleep(0.2)
    rospy.init_node("tracking_experiment_runner", anonymous=True)
    route_pub = rospy.Publisher("/tracking_test/reference", Path, queue_size=1, latch=True)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    for _ in range(20):
        if rospy.is_shutdown():
            return 130
        rospy.sleep(0.05)
    for route_index, route_name in enumerate(routes):
        route_seed = args.route_seed + route_index if args.route_seed is not None else None
        route = (copy.deepcopy(route_override) if route_override is not None else
                 build_route(route_name, args.goal_depth, route_seed=route_seed))
        output = make_dir(root)
        recorder = Recorder(args.controller, route["goal"])
        launch = None
        profile_params = {}
        try:
            launch_cmd = ["roslaunch", "hofa_mpc_ros", "simulation.launch",
                          "controller:=" + args.controller,
                          "vehicle_name:=bricsbot", "vehicle_model:=bricsbot",
                          "scene:=" + args.scene, "enable:=true",
                          "reference_source:=external",
                          "reference_topic:=/tracking_test/reference",
                          "reference_max_speed:=0.2",
                          "reference_max_yaw_rate:=" + str(args.max_yaw_rate),
                          "reference_max_yaw_accel:=" + str(args.max_yaw_accel),
                          "reference_max_accel:=0.12",
                          "reference_max_decel:=0.18",
                          "experiment_start_topic:=",
                          "experiment_depth:=" + str(args.goal_depth)]
            # The generated route is supplied on an external Path topic; the
            # launch starts only the shared simulator/reference/controller
            # chain and does not start the privileged teacher.
            if args.controller == "hofa_policy":
                launch_cmd += ["policy_checkpoint:=" + os.path.abspath(args.checkpoint),
                               "policy_device:=" + args.policy_device]
                if args.residual_model:
                    launch_cmd += ["policy_residual_model:=" + os.path.abspath(args.residual_model),
                                   "policy_residual_algo:=" + args.residual_algo]
                policy_python = args.policy_python
                if not policy_python and args.residual_model:
                    policy_python = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "../../../.venv-hofa-rl/bin/python"))
                if policy_python:
                    launch_cmd.append("policy_python:=" + policy_python)
            if args.scenario_file:
                launch_cmd.append("scenario_file:=" + os.path.abspath(args.scenario_file))
            if args.headless:
                launch_cmd.append("headless:=true")
            launch = subprocess.Popen(launch_cmd, start_new_session=True)
            for _ in range(10):
                publish_route(route, route_pub)
                rospy.sleep(0.1)
            deadline = time.time() + args.ready_timeout
            while not recorder.window and time.time() < deadline and not rospy.is_shutdown():
                publish_route(route, route_pub)
                rospy.sleep(0.1)
            if not recorder.window:
                raise RuntimeError("reference processor did not publish a trajectory window")
            # A published topic does not imply roslaunch has loaded this run's
            # parameters.  Wait for an actual window before reading them, or
            # a persistent ROS master can return the previous run's limits.
            profile_params = rospy.get_param("/reference_processor")
            profile = route_diagnostics(route, profile_params)
            goal = PoseStamped()
            goal.header.frame_id = "world_ned"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = route["goal"]
            q = quaternion(route["points"][-1][3])
            goal.pose.orientation.x, goal.pose.orientation.y = q[0], q[1]
            goal.pose.orientation.z, goal.pose.orientation.w = q[2], q[3]
            goal_pub.publish(goal)
            recorder.start = rospy.Time.now().to_sec()
            start = time.time()
            while time.time() - start < args.timeout and not rospy.is_shutdown() and not recorder.done:
                publish_route(route, route_pub)
                rospy.sleep(0.2)
            if not recorder.done:
                recorder.finish_reason = "timeout"
        finally:
            terminate(launch)
        try:
            metrics = evaluate(recorder, route, profile, route["goal"])
        except Exception as error:
            metrics = {
                "error": "evaluation_%s: %s" %
                (type(error).__name__, error),
                "controller": args.controller,
                "route": route["name"],
                "finish_reason": recorder.finish_reason,
                "odometry_samples": len(recorder.odom),
            }
        with open(os.path.join(output, "route.json"), "w") as stream:
            json.dump(route, stream, indent=2)
        with open(os.path.join(output, "reference_parameters.json"), "w") as stream:
            json.dump(profile_params, stream, indent=2, sort_keys=True)
        with open(os.path.join(output, "metrics.json"), "w") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)
        write_csv(os.path.join(output, "odometry.csv"), recorder.odom)
        write_csv(os.path.join(output, "controller_status.csv"), recorder.status)
        write_csv(os.path.join(output, "generalized_force.csv"), recorder.wrench)
        write_csv(os.path.join(output, "residual_components.csv"), recorder.residual_diagnostics)
        write_csv(os.path.join(output, "pwm.csv"),
                  [{"stamp": r["stamp"], **{"pwm_%d" % i: v for i, v in enumerate(r["pwm"])} }
                   for r in recorder.pwm])
        write_csv(os.path.join(output, "reference_window.csv"), recorder.window)
        write_csv(os.path.join(output, "reference_profile.csv"),
                  [{"s_m": float(s), "x": float(x), "y": float(y),
                    "yaw": float(yaw), "curvature": float(k),
                    "speed": float(v), "time_s": float(t),
                    "yaw_rate": float(v * k)}
                   for s, x, y, yaw, k, v, t in zip(
                       profile["s"], profile["x"], profile["y"], profile["yaw"],
                       profile["curvature"], profile["speed"], profile["time"])])
        if generate_run_plots is not None:
            try:
                plots = generate_run_plots(output)
                print("Plots: %s" % os.path.join(output, "plots"))
                print("Generated %d plot files" % len(plots))
            except Exception as error:
                print("WARNING: plot generation failed: %s" % error)
        else:
            print("Plots skipped: hofa_mpc_ros.tracking_plots unavailable")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print("Artifacts: %s" % output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
