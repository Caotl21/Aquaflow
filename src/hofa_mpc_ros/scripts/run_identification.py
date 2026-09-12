#!/usr/bin/env python3
"""Run a constant-force identification sweep and record one CSV per level.

This produces the trial files that ``fit_drag_coefficients.py`` consumes.
Launch the simulator with no controller first, so nothing competes for the
wrench topic::

    roslaunch hofa_mpc_ros simulation.launch controller:=none enable:=true

then, for each axis::

    rosrun hofa_mpc_ros run_identification.py --axis surge --forces 4 6 8 12

Two details make the recorded data usable for a mass-independent drag fit:

* Depth is held by a small PID on Fz while the axis under test gets a pure
  constant command.  The hull is slightly positively buoyant, and over a
  20 s trial that is enough drift to surface the vehicle, which would
  contaminate every horizontal measurement.
* Force levels alternate sign by default, so the vehicle returns towards
  where it started instead of running into a pool wall after two trials.
  The drag model is odd-symmetric, so negative levels fit exactly the same.

The script waits for the vehicle to come to rest between trials and reports
whether each one actually reached terminal velocity; a trial that did not
settle cannot give a mass-independent drag estimate.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import rospy
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry

AXES = {
    "surge": {"component": "fx", "velocity": "u", "unit": "N"},
    "sway": {"component": "fy", "velocity": "v", "unit": "N"},
    "yaw": {"component": "nz", "velocity": "r", "unit": "N.m"},
}


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class IdentificationRunner(object):
    def __init__(self, args):
        self.args = args
        self.axis = AXES[args.axis]
        self.odom = None
        self.int_z = 0.0
        self.last_depth_time = None
        self.pub = rospy.Publisher(args.wrench_topic, WrenchStamped,
                                   queue_size=1)
        rospy.Subscriber("/%s/odometry" % args.vehicle_name, Odometry,
                         self._odom_cb, queue_size=1)

    def _odom_cb(self, msg):
        self.odom = msg

    def wait_for_odometry(self, timeout=10.0):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and self.odom is None:
            if rospy.Time.now() > deadline:
                raise RuntimeError(
                    "no odometry on /%s/odometry; is the simulator running?"
                    % self.args.vehicle_name)
            rospy.sleep(0.1)

    def _depth_force(self):
        """PID on depth only, so the test axis stays a pure constant force."""
        if self.odom is None:
            return 0.0
        now = rospy.Time.now()
        dt = 0.05
        if self.last_depth_time is not None:
            dt = max(1e-3, min(0.2, (now - self.last_depth_time).to_sec()))
        self.last_depth_time = now
        error = self.args.depth - self.odom.pose.pose.position.z
        self.int_z = max(-5.0, min(5.0, self.int_z + error * dt))
        velocity_z = self.odom.twist.twist.linear.z
        return (self.args.kp_z * error - self.args.kd_z * velocity_z
                + self.args.ki_z * self.int_z)

    def publish(self, value):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        if self.args.axis == "surge":
            msg.wrench.force.x = float(value)
        elif self.args.axis == "sway":
            msg.wrench.force.y = float(value)
        else:
            msg.wrench.torque.z = float(value)
        msg.wrench.force.z = self._depth_force()
        self.pub.publish(msg)

    def _axis_velocity(self):
        twist = self.odom.twist.twist
        if self.args.axis == "surge":
            return twist.linear.x
        if self.args.axis == "sway":
            return twist.linear.y
        return twist.angular.z

    def settle(self, rate):
        """Hold zero test force until the vehicle is at rest and on depth."""
        deadline = rospy.Time.now() + rospy.Duration(self.args.settle)
        stable_since = None
        while not rospy.is_shutdown():
            self.publish(0.0)
            at_rest = abs(self._axis_velocity()) < self.args.rest_speed
            on_depth = abs(self.args.depth
                           - self.odom.pose.pose.position.z) < 0.05
            if at_rest and on_depth:
                if stable_since is None:
                    stable_since = rospy.Time.now()
                elif (rospy.Time.now() - stable_since).to_sec() > 1.0:
                    return True
            else:
                stable_since = None
            if rospy.Time.now() > deadline:
                rospy.logwarn("settle timed out: |v|=%.4f depth_err=%.3f",
                              abs(self._axis_velocity()),
                              self.args.depth - self.odom.pose.pose.position.z)
                return False
            rate.sleep()
        return False

    def run_trial(self, force, rate):
        rospy.loginfo("--- trial: %s = %+.3f %s for %.1f s ---",
                      self.axis["component"], force, self.axis["unit"],
                      self.args.duration)
        self.settle(rate)
        rows = []
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start).to_sec()
            if elapsed > self.args.duration:
                break
            self.publish(force)
            pose = self.odom.pose.pose
            twist = self.odom.twist.twist
            rows.append({
                "stamp": self.odom.header.stamp.to_sec(),
                "t_rel": elapsed,
                "x": pose.position.x, "y": pose.position.y,
                "z": pose.position.z,
                "yaw": yaw_from_quaternion(pose.orientation),
                "u": twist.linear.x, "v": twist.linear.y,
                "r": twist.angular.z,
            })
            rate.sleep()
        self.publish(0.0)
        return rows

    @staticmethod
    def terminal_check(rows, velocity_key, tail_fraction=0.25):
        """Did the trial actually settle?  Reports the tail acceleration."""
        if len(rows) < 10:
            return None
        tail = rows[-max(5, int(len(rows) * tail_fraction)):]
        span = tail[-1]["stamp"] - tail[0]["stamp"]
        if span < 1e-6:
            return None
        drift = (tail[-1][velocity_key] - tail[0][velocity_key]) / span
        values = [row[velocity_key] for row in tail]
        return {
            "terminal_velocity": sum(values) / len(values),
            "tail_acceleration": drift,
            "settled": abs(drift) < 0.01,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=sorted(AXES), required=True)
    parser.add_argument("--forces", nargs="+", type=float, required=True,
                        help="force magnitudes (N, or N.m for yaw); signs are "
                             "alternated unless --no-alternate")
    parser.add_argument("--no-alternate", action="store_true",
                        help="apply every level in the sign given, instead of "
                             "flipping each one to keep the vehicle centred")
    parser.add_argument("--duration", type=float, default=25.0,
                        help="seconds to hold each level; must be long enough "
                             "to reach terminal velocity")
    parser.add_argument("--settle", type=float, default=20.0,
                        help="max seconds to wait for rest between trials")
    parser.add_argument("--rest-speed", type=float, default=0.01,
                        help="axis speed counted as 'at rest'")
    parser.add_argument("--depth", type=float, default=1.0,
                        help="depth to hold during the sweep (NED, metres)")
    parser.add_argument("--kp-z", type=float, default=8.0)
    parser.add_argument("--kd-z", type=float, default=4.0)
    parser.add_argument("--ki-z", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--vehicle-name", default="bricsbot")
    parser.add_argument("--wrench-topic",
                        default="/controller/generalized_force")
    parser.add_argument("--output-dir", default=None,
                        help="default results/identification/<axis>_<stamp>")
    args = parser.parse_args(rospy.myargv()[1:] if argv is None else argv)

    if any(f == 0.0 for f in args.forces):
        parser.error("a zero force level carries no information")
    if len(args.forces) < 3:
        print("warning: %d level(s) given; three or more distinct magnitudes "
              "are needed to separate the linear and quadratic drag terms"
              % len(args.forces), file=sys.stderr)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results", "identification",
            "%s_%s" % (args.axis, datetime.now().strftime("%Y%m%d_%H%M%S")))
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    rospy.init_node("run_identification", anonymous=True)
    runner = IdentificationRunner(args)
    runner.wait_for_odometry()
    rate = rospy.Rate(args.rate)

    velocity_key = AXES[args.axis]["velocity"]
    trials = []
    for index, magnitude in enumerate(args.forces):
        sign = 1.0 if (args.no_alternate or index % 2 == 0) else -1.0
        force = sign * magnitude
        rows = runner.run_trial(force, rate)
        if rospy.is_shutdown():
            break
        filename = os.path.join(
            output_dir, "%s_%+.2f.csv" % (args.axis, force))
        with open(filename, "w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        check = runner.terminal_check(rows, velocity_key) or {}
        trials.append({"force": force, "file": filename,
                       "samples": len(rows), **check})
        rospy.loginfo("    v_terminal=%.4f  tail_accel=%+.5f  settled=%s",
                      check.get("terminal_velocity", float("nan")),
                      check.get("tail_acceleration", float("nan")),
                      check.get("settled"))
    runner.publish(0.0)

    unsettled = [t for t in trials if not t.get("settled")]
    manifest = {
        "axis": args.axis,
        "depth_m": args.depth,
        "duration_s": args.duration,
        "trials": trials,
        "all_settled": not unsettled,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

    print()
    if unsettled:
        print("WARNING: %d trial(s) had not reached terminal velocity; their "
              "drag estimate still depends on the assumed mass. Re-run those "
              "levels with a longer --duration." % len(unsettled))
        for trial in unsettled:
            print("  %+.2f -> tail acceleration %+.5f"
                  % (trial["force"], trial.get("tail_acceleration", 0.0)))
        print()
    print("Artifacts: %s" % output_dir)
    print("Next, fit the drag for this axis:")
    print("  rosrun hofa_mpc_ros fit_drag_coefficients.py --mode drag "
          "--axis %s \\\n      --odom %s \\\n      --force %s"
          % (args.axis,
             " ".join(t["file"] for t in trials),
             " ".join("%g" % t["force"] for t in trials)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        sys.exit(130)
    except (RuntimeError, OSError) as exc:
        print("run_identification.py: %s" % exc, file=sys.stderr)
        sys.exit(2)
