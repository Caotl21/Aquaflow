#!/usr/bin/env python3
"""Run one reproducible BricsBot HOFA-MPC tracking experiment.

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
import json
import math
import os
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
from geometry_msgs.msg import PoseStamped, WrenchStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def make_run_dir(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, "run_" + stamp)
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(root, "run_%s_%02d" % (stamp, suffix))
        suffix += 1
    os.makedirs(path)
    return path


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

        self.global_path = None
        self.initial_path = None
        self.initial_path_model = None
        self.odom = []
        self.status = []
        self.wrench = []
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
        rospy.Subscriber("/bricsbot/setpoint/pwm", Float64MultiArray,
                         self._pwm_cb, queue_size=20)

    def reset_reference_model(self):
        """Discard any route published before this run's goal was sent."""
        with self.lock:
            self.initial_path = None
            self.initial_path_model = None

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
                self.initial_path_model = self._build_path_model(path)

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

    def _pwm_cb(self, msg):
        with self.lock:
            self.pwm.append({
                "stamp": rospy.Time.now().to_sec(),
                "pwm": [float(v) for v in msg.data],
            })

    @staticmethod
    def _build_path_model(path):
        points = path["points"]
        xy = np.asarray([[p[1], p[2]] for p in points], dtype=float)
        yaw = np.unwrap(np.asarray([p[4] for p in points], dtype=float))
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(ds)))
        times = np.asarray([p[0] for p in points], dtype=float)
        valid_times = np.all(np.isfinite(times)) and np.all(np.diff(times) > 1e-9)
        if not valid_times:
            # Fallback for publishers that do not provide per-pose timestamps.
            times = np.arange(len(points), dtype=float)
            times *= 0.1
        speed = np.zeros(len(points), dtype=float)
        for i in range(1, len(points)):
            dt = max(times[i] - times[i - 1], 1e-3)
            speed[i] = ds[i - 1] / dt
        speed[0] = speed[1] if len(speed) > 1 else 0.0
        return {"xy": xy, "yaw": yaw, "arc": arc,
                "times": times, "speed": speed}

    @staticmethod
    def _project(x, y, model):
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

    def _timed_reference(self, stamp, model):
        times = model["times"]
        rel = float(stamp - times[0])
        x = float(np.interp(stamp, times, model["xy"][:, 0]))
        y = float(np.interp(stamp, times, model["xy"][:, 1]))
        yaw = float(np.interp(stamp, times, model["yaw"]))
        speed = float(np.interp(stamp, times, model["speed"]))
        expected_s = float(np.interp(stamp, times, model["arc"]))
        return x, y, yaw, speed, expected_s, rel

    def metrics(self):
        with self.lock:
            odom = list(self.odom)
            model = self.initial_path_model
            path = self.initial_path or self.global_path
            status = list(self.status)
            wrench = list(self.wrench)
            pwm = list(self.pwm)
        if not odom or model is None:
            return {"error": "no odometry or global path recorded"}

        cross, geom, yaw_err, speed_err, progress_err = [], [], [], [], []
        for row in odom:
            sx, sy, syaw, rspeed, expected_s, rel = self._timed_reference(
                row["stamp"], model)
            actual_s, signed, distance = self._project(row["x"], row["y"], model)
            cross.append(signed)
            geom.append(distance)
            yaw_err.append(wrap(row["yaw"] - syaw))
            speed_err.append(row["speed_world"] - rspeed)
            progress_err.append(actual_s - expected_s)

        def stats(values, degrees=False):
            a = np.asarray(values, dtype=float)
            if degrees:
                a = np.degrees(a)
            return {
                "rmse": float(np.sqrt(np.mean(a * a))),
                "mae": float(np.mean(np.abs(a))),
                "p95_abs": float(np.percentile(np.abs(a), 95)),
                "max_abs": float(np.max(np.abs(a))),
            }

        duration = max(0.0, odom[-1]["t_rel"] - odom[0]["t_rel"])
        solver_success = [x["solver_success"] for x in status]
        callback_times = np.asarray([x["callback_time_ms"] for x in status], dtype=float)
        pwm_values = np.asarray([x["pwm"] for x in pwm if x["pwm"]], dtype=float)
        if pwm_values.size:
            sat = np.any(np.abs(pwm_values) >= 0.99, axis=1)
            saturation_ratio = float(np.mean(sat))
        else:
            saturation_ratio = 0.0

        result = {
            "finish_reason": self.finish_reason,
            "duration_s": float(duration),
            "samples": int(len(odom)),
            "path_length_m": float(model["arc"][-1]),
            "final_position_m": [odom[-1]["x"], odom[-1]["y"]],
            "final_speed_mps": float(odom[-1]["speed_world"]),
            "cross_track_error_m": stats(cross),
            "geometric_path_error_m": stats(geom),
            "yaw_error_deg": stats(yaw_err, degrees=True),
            "speed_error_mps": stats(speed_err),
            "progress_error_m": stats(progress_err),
            "solver_success_rate": (float(np.mean(solver_success))
                                     if solver_success else None),
            "status_samples": int(len(status)),
            "pwm_samples": int(len(pwm)),
            "pwm_saturation_ratio": saturation_ratio,
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

    def save(self, output_dir):
        with self.lock:
            odom, status, wrench, pwm, window = (
                list(self.odom), list(self.status), list(self.wrench),
                list(self.pwm), list(self.window))
            path = self.initial_path or self.global_path
        metrics = self.metrics()
        self._write_csv(os.path.join(output_dir, "odometry.csv"), odom)
        self._write_csv(os.path.join(output_dir, "controller_status.csv"), status)
        self._write_csv(os.path.join(output_dir, "generalized_force.csv"), wrench)
        pwm_rows = [{"stamp": x["stamp"],
                     **{"pwm_%d" % i: v for i, v in enumerate(x["pwm"])}}
                    for x in pwm]
        self._write_csv(os.path.join(output_dir, "pwm.csv"), pwm_rows)
        self._write_csv(os.path.join(output_dir, "mpc_reference_window.csv"),
                        window)
        if path:
            self._write_csv(
                os.path.join(output_dir, "global_reference_path.csv"),
                [{"stamp": p[0], "x_ned": p[1], "y_ned": p[2],
                  "z_ned": p[3], "yaw_ned": p[4]}
                 for p in path["points"]])
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        self._write_summary(os.path.join(output_dir, "summary.txt"), metrics)
        self._plot(output_dir, metrics)
        return metrics

    @staticmethod
    def _write_csv(filename, rows):
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

    @staticmethod
    def _write_summary(filename, metrics):
        with open(filename, "w") as f:
            f.write("HOFA-MPC experiment summary\n")
            f.write("===========================\n")
            for key, value in metrics.items():
                f.write("%-28s %s\n" % (key, value))

    def _plot(self, output_dir, metrics):
        if plt is None:
            return
        with self.lock:
            odom = list(self.odom)
            model = self.initial_path_model
            status = list(self.status)
            wrench = list(self.wrench)
        if not odom or model is None:
            return
        t = np.asarray([x["t_rel"] for x in odom])
        x = np.asarray([x["x"] for x in odom])
        y = np.asarray([x["y"] for x in odom])
        speed = np.asarray([x["speed_world"] for x in odom])
        refs = [self._timed_reference(row["stamp"], model) for row in odom]
        rx = np.asarray([r[0] for r in refs])
        ry = np.asarray([r[1] for r in refs])
        rs = np.asarray([r[3] for r in refs])
        cross, geom, ye, ve = [], [], [], []
        for row, ref in zip(odom, refs):
            _, signed, distance = self._project(row["x"], row["y"], model)
            cross.append(signed)
            geom.append(distance)
            ye.append(wrap(row["yaw"] - ref[2]))
            ve.append(row["speed_world"] - ref[3])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(model["xy"][:, 0], model["xy"][:, 1], "m--", label="global reference")
        ax.plot(rx, ry, "b:", label="time reference")
        ax.plot(x, y, "r", label="actual odometry")
        ax.scatter([self.goal_xy[0]], [self.goal_xy[1]], c="k", marker="x", label="goal")
        ax.set_xlabel("NED x [m]"); ax.set_ylabel("NED y [m]")
        ax.set_title("XY tracking"); ax.axis("equal"); ax.grid(True); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "tracking_xy.png"), dpi=150); plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(t, cross); axes[0].set_ylabel("cross-track [m]"); axes[0].grid(True)
        axes[1].plot(t, geom); axes[1].set_ylabel("path distance [m]"); axes[1].grid(True)
        axes[2].plot(t, np.degrees(ye)); axes[2].set_ylabel("yaw error [deg]"); axes[2].set_xlabel("time [s]"); axes[2].grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "tracking_errors.png"), dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, speed, label="actual speed")
        ax.plot(t, rs, "--", label="reference speed")
        ax.plot(t, ve, ":", label="speed error")
        ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]"); ax.grid(True); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "speed_tracking.png"), dpi=150); plt.close(fig)

        if wrench:
            wt = np.asarray([w["stamp"] - odom[0]["stamp"] for w in wrench])
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(wt, [w["fx"] for w in wrench], label="Fx")
            ax.plot(wt, [w["fy"] for w in wrench], label="Fy")
            ax.plot(wt, [w["nz"] for w in wrench], label="Nz")
            ax.set_xlabel("time [s]"); ax.set_ylabel("wrench [N/Nm]"); ax.grid(True); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(output_dir, "control_wrench.png"), dpi=150); plt.close(fig)


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
    parser.add_argument("--output-root", default="results/mpc")
    parser.add_argument("--launch-pkg", default="hofa_mpc_ros")
    parser.add_argument("--launch-file", default="simulation.launch")
    parser.add_argument("--scene", default="empty_pool")
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-depth", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--startup-wait", type=float, default=5.0)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-bag", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="use Stonefish parsed_simulator_nogpu")
    args = parser.parse_args(rospy.myargv()[1:])

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
                          "controller:=hofa_mpc", "vehicle_name:=bricsbot",
                          "vehicle_model:=bricsbot", "scene:=" + args.scene,
                          "enable:=true"]
            if args.headless:
                launch_cmd.append("headless:=true")
            launch_proc = subprocess.Popen(launch_cmd, start_new_session=True)

        if not args.no_bag:
            bag_file = os.path.join(output_dir, "topics.bag")
            bag_topics = ["/clock", "/bricsbot/odometry",
                          "/aquaflow/teacher_global_path",
                          "/controller/reference_trajectory_window",
                          "/hofa_mpc_controller/status",
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
