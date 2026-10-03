#!/home/bricsbot/miniconda3/envs/oceangym/bin/python
"""Run hofa_policy beside HOFA-MPC without publishing control commands."""
from collections import deque
import math
import sys
import time

print("hofa_policy Shadow: importing runtime", file=sys.stderr, flush=True)

import numpy as np
import rospy
import logging
from rosgraph.roslogging import RospyLogger
from geometry_msgs.msg import Vector3Stamped, WrenchStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow
from hofa_policy.contract import CONTRACT
from hofa_policy.features import reference_features, state_features


# ROS Noetic's RospyLogger.findCaller walks backward until it finds the exact
# frame returned by logging.Logger.findCaller.  Python 3.11 changed that frame
# selection and the Noetic loop can remain on the root frame forever.  Use the
# standard implementation in this Python 3.11-only inference process.  Do not
# patch the system ROS installation or other nodes.
if sys.version_info >= (3, 11):
    RospyLogger.findCaller = logging.Logger.findCaller


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HofaPolicyShadowNode:
    def __init__(self):
        print("hofa_policy Shadow: initializing rospy", file=sys.stderr,
              flush=True)
        rospy.init_node("hofa_policy_shadow")
        print("hofa_policy Shadow: rospy initialized", file=sys.stderr,
              flush=True)
        rospy.loginfo("HOFA policy Shadow initializing")
        checkpoint = rospy.get_param("~checkpoint")
        device = rospy.get_param("~device", "cpu")
        threads = int(rospy.get_param("~torch_threads", 1))
        self.max_odom_age = float(rospy.get_param("~max_odom_age_s", 0.05))
        self.max_reference_age = float(
            rospy.get_param("~max_reference_age_s", 0.15))
        self.max_status_age = float(
            rospy.get_param("~max_teacher_status_offset_s", 0.08))
        self.max_pair_wait = float(
            rospy.get_param("~max_pair_wait_s", 0.08))
        self.max_free_gap = float(
            rospy.get_param("~max_free_gap_s", 0.15))
        self.odom = deque(maxlen=100)
        self.windows = deque(maxlen=20)
        self.teacher = deque(maxlen=20)
        self.pending_teacher = deque(maxlen=100)
        self.status = deque(maxlen=20)
        self.last_processed_stamp = None
        self.last_free_stamp = None
        self.previous_free_wrench = np.zeros(3, dtype=np.float32)
        self.rejected = {}

        self.shadow_pub = rospy.Publisher(
            "/hofa_policy/shadow_generalized_force", WrenchStamped,
            queue_size=20)
        self.error_pub = rospy.Publisher(
            "/hofa_policy/shadow_error", WrenchStamped, queue_size=20)
        self.force_pub = rospy.Publisher(
            "/hofa_policy/shadow_candidate_forces", JointState,
            queue_size=20)
        self.timing_pub = rospy.Publisher(
            "/hofa_policy/shadow_timing", Vector3Stamped, queue_size=20)
        self.free_shadow_pub = rospy.Publisher(
            "/hofa_policy/free_shadow_generalized_force", WrenchStamped,
            queue_size=20)
        self.free_error_pub = rospy.Publisher(
            "/hofa_policy/free_shadow_error", WrenchStamped, queue_size=20)
        self.free_force_pub = rospy.Publisher(
            "/hofa_policy/free_shadow_candidate_forces", JointState,
            queue_size=20)
        self.free_timing_pub = rospy.Publisher(
            "/hofa_policy/free_shadow_timing", Vector3Stamped, queue_size=20)

        # Register the ROS interface before importing PyTorch.  This makes
        # startup failures visible to roslaunch and avoids presenting a live
        # process with no ROS node while native libraries are initialized.
        rospy.loginfo("HOFA policy Shadow loading checkpoint: %s", checkpoint)
        from hofa_policy.shadow_core import ShadowPolicy
        self.policy = ShadowPolicy(checkpoint, device, threads)

        rospy.Subscriber("/bricsbot/odometry", Odometry,
                         self.odom_cb, queue_size=100)
        rospy.Subscriber("/controller/reference_trajectory_window",
                         TrajectoryPointWindow, self.window_cb, queue_size=20)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self.teacher_cb, queue_size=20)
        rospy.Subscriber("/hofa_mpc_controller/status", ControllerStatus,
                         self.status_cb, queue_size=20)
        rospy.loginfo(
            "HOFA policy Shadow ready: checkpoint=%s epoch=%d device=%s",
            checkpoint, self.policy.epoch, device)

    @staticmethod
    def stamp(msg):
        value = msg.header.stamp.to_sec()
        return value if value > 0.0 else rospy.Time.now().to_sec()

    def reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        rospy.logwarn_throttle(
            5.0, "hofa_policy Shadow skipped %s; counters=%s",
            reason, self.rejected)

    def odom_cb(self, msg):
        self.odom.append((self.stamp(msg), msg))
        self.process_pending_teacher()

    def window_cb(self, msg):
        if msg.valid and len(msg.points) >= CONTRACT.horizon:
            self.windows.append((self.stamp(msg), msg))
            self.process_pending_teacher()

    def teacher_cb(self, msg):
        stamp = self.stamp(msg)
        wrench = np.asarray([
            msg.wrench.force.x, msg.wrench.force.y,
            msg.wrench.torque.z], dtype=np.float32)
        self.teacher.append((stamp, msg.header, wrench))
        self.pending_teacher.append((stamp, msg.header, wrench))
        self.process_pending_teacher()

    @staticmethod
    def causal_item(items, stamp):
        for item in reversed(items):
            if item[0] <= stamp + 1.0e-9:
                return item
        return None

    @staticmethod
    def odom_row(msg):
        return {
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": yaw_from_quaternion(msg.pose.pose.orientation),
            "u": float(msg.twist.twist.linear.x),
            "v": float(msg.twist.twist.linear.y),
            "r": float(msg.twist.twist.angular.z),
        }

    @staticmethod
    def reference_rows(msg):
        rows = []
        for point in msg.points[:CONTRACT.horizon]:
            rows.append({
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
        return rows

    def status_cb(self, msg):
        if msg.state != "ACTIVE" or not bool(msg.solver_success):
            self.reset_free_running()
        self.status.append((self.stamp(msg), msg.state,
                            bool(msg.solver_success)))
        self.process_pending_teacher()

    def reset_free_running(self):
        self.previous_free_wrench.fill(0.0)
        self.last_free_stamp = None

    def process_pending_teacher(self):
        while self.pending_teacher:
            teacher_stamp, header, teacher = self.pending_teacher[0]
            nearby = [item for item in self.status
                      if abs(item[0] - teacher_stamp) <= self.max_status_age]
            if not nearby:
                if (self.status and
                        self.status[-1][0] > teacher_stamp + self.max_pair_wait):
                    self.pending_teacher.popleft()
                    self.reject("teacher_status_mismatch")
                    continue
                break

            status_item = min(
                nearby, key=lambda item: abs(item[0] - teacher_stamp))
            if status_item[1] != "ACTIVE" or not status_item[2]:
                self.pending_teacher.popleft()
                self.reject("teacher_not_active")
                continue

            odom_item = self.causal_item(self.odom, teacher_stamp)
            odom_fresh = (odom_item is not None and
                          teacher_stamp - odom_item[0] <= self.max_odom_age)
            if not odom_fresh:
                if (self.odom and
                        self.odom[-1][0] > teacher_stamp + self.max_pair_wait):
                    self.pending_teacher.popleft()
                    self.reject("stale_odometry" if odom_item is not None
                                else "missing_causal_odometry")
                    continue
                break

            window_item = self.causal_item(self.windows, teacher_stamp)
            reference_fresh = (
                window_item is not None and
                teacher_stamp - window_item[0] <= self.max_reference_age)
            if not reference_fresh:
                if (self.windows and
                        self.windows[-1][0] > teacher_stamp + self.max_pair_wait):
                    self.pending_teacher.popleft()
                    self.reject("stale_reference" if window_item is not None
                                else "missing_causal_reference")
                    continue
                break

            self.pending_teacher.popleft()
            self.process_valid_teacher(teacher_stamp, header, teacher,
                                       odom_item, window_item)

    def process_valid_teacher(self, teacher_stamp, header, teacher,
                              odom_item, window_item):
        if self.last_processed_stamp == teacher_stamp:
            return
        previous = np.zeros(3, dtype=np.float32)
        for index in range(len(self.teacher) - 1, -1, -1):
            if self.teacher[index][0] < teacher_stamp - 1.0e-9:
                previous = self.teacher[index][2]
                break
        odom_age = teacher_stamp - odom_item[0]
        reference_age = teacher_stamp - window_item[0]
        odom = self.odom_row(odom_item[1])
        if (self.last_free_stamp is not None and
                (teacher_stamp <= self.last_free_stamp or
                 teacher_stamp - self.last_free_stamp > self.max_free_gap)):
            self.reset_free_running()
        free_previous = self.previous_free_wrench.copy()
        try:
            state = state_features(odom)
            reference = reference_features(
                odom, self.reference_rows(window_item[1]))
            start_ns = time.perf_counter_ns()
            predicted, force, normalized = self.policy.predict(
                state, reference, previous)
            inference_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
            free_start_ns = time.perf_counter_ns()
            free_predicted, free_force, free_normalized = self.policy.predict(
                state, reference, free_previous)
            free_inference_ms = (
                time.perf_counter_ns() - free_start_ns) / 1.0e6
        except Exception as error:
            self.reject("inference_%s" % type(error).__name__)
            rospy.logerr_throttle(5.0, "Shadow inference failed: %s", error)
            return
        if not all(np.all(np.isfinite(item))
                   for item in (predicted, force, normalized)):
            self.reject("nonfinite_output")
            return
        if not all(np.all(np.isfinite(item)) for item in
                   (free_predicted, free_force, free_normalized)):
            self.reject("nonfinite_free_output")
            self.reset_free_running()
            return
        error = predicted - teacher
        free_error = free_predicted - teacher

        shadow = WrenchStamped()
        shadow.header = header
        shadow.wrench.force.x = float(predicted[0])
        shadow.wrench.force.y = float(predicted[1])
        shadow.wrench.torque.z = float(predicted[2])
        self.shadow_pub.publish(shadow)

        error_msg = WrenchStamped()
        error_msg.header = header
        error_msg.wrench.force.x = float(error[0])
        error_msg.wrench.force.y = float(error[1])
        error_msg.wrench.torque.z = float(error[2])
        self.error_pub.publish(error_msg)

        force_msg = JointState()
        force_msg.header = header
        force_msg.name = ["T1", "T2", "T3", "T4"]
        force_msg.effort = [float(value) for value in force]
        force_msg.position = [float(value) for value in normalized]
        self.force_pub.publish(force_msg)

        timing = Vector3Stamped()
        timing.header = header
        timing.vector.x = inference_ms
        timing.vector.y = odom_age
        timing.vector.z = reference_age
        self.timing_pub.publish(timing)

        free_shadow = WrenchStamped()
        free_shadow.header = header
        free_shadow.wrench.force.x = float(free_predicted[0])
        free_shadow.wrench.force.y = float(free_predicted[1])
        free_shadow.wrench.torque.z = float(free_predicted[2])
        self.free_shadow_pub.publish(free_shadow)

        free_error_msg = WrenchStamped()
        free_error_msg.header = header
        free_error_msg.wrench.force.x = float(free_error[0])
        free_error_msg.wrench.force.y = float(free_error[1])
        free_error_msg.wrench.torque.z = float(free_error[2])
        self.free_error_pub.publish(free_error_msg)

        free_force_msg = JointState()
        free_force_msg.header = header
        free_force_msg.name = ["T1", "T2", "T3", "T4"]
        free_force_msg.effort = [float(value) for value in free_force]
        free_force_msg.position = [float(value) for value in free_normalized]
        self.free_force_pub.publish(free_force_msg)

        free_timing = Vector3Stamped()
        free_timing.header = header
        free_timing.vector.x = free_inference_ms
        free_timing.vector.y = odom_age
        free_timing.vector.z = reference_age
        self.free_timing_pub.publish(free_timing)

        self.previous_free_wrench = free_predicted.copy()
        self.last_free_stamp = teacher_stamp
        self.last_processed_stamp = teacher_stamp


if __name__ == "__main__":
    try:
        HofaPolicyShadowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
