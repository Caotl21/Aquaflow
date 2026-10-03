#!/home/bricsbot/miniconda3/envs/oceangym/bin/python
"""Run the distilled HOFA policy as the Stonefish closed-loop controller."""
import logging
import math
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import Vector3Stamped, WrenchStamped
from nav_msgs.msg import Odometry
from rosgraph.roslogging import RospyLogger
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from hofa_mpc_ros.experiment_gate import ExperimentGate
from hofa_mpc_ros.msg import ControllerStatus, TrajectoryPointWindow
from hofa_policy.contract import CONTRACT, bricsbot_horizontal_actuation
from hofa_policy.features import reference_features, state_features


if sys.version_info >= (3, 11):
    RospyLogger.findCaller = logging.Logger.findCaller


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HofaPolicyControllerNode:
    def __init__(self):
        rospy.init_node("hofa_policy_controller")
        checkpoint = rospy.get_param("~checkpoint")
        device = rospy.get_param("~device", "cpu")
        threads = int(rospy.get_param("~torch_threads", 1))
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.rate_hz = float(rospy.get_param("~control_rate_hz", 10.0))
        self.max_odom_age = float(rospy.get_param("~max_odom_age_s", 0.20))
        self.max_reference_age = float(
            rospy.get_param("~max_reference_age_s", 0.30))
        self.max_position_abs = np.asarray(
            rospy.get_param("~max_position_abs_m", [14.0, 8.0]),
            dtype=np.float64)
        self.max_world_speed = float(
            rospy.get_param("~max_world_speed_mps", 1.0))
        self.max_yaw_rate = float(
            rospy.get_param("~max_yaw_rate_radps", 1.2))

        from hofa_policy.shadow_core import ShadowPolicy
        self.policy = ShadowPolicy(checkpoint, device, threads)
        actuation = bricsbot_horizontal_actuation()
        self.allocation = actuation["allocation_matrix"]
        self.force_min = actuation["force_min"]
        self.force_max = actuation["force_max"]

        self.odom = None
        self.odom_stamp = None
        self.window = None
        self.reference_stamp = None
        self.previous_wrench = np.zeros(3, dtype=np.float32)
        self.failure_count = 0
        self.state_name = "DISABLED"

        odom_topic = rospy.get_param("~odom_topic", "/bricsbot/odometry")
        reference_topic = rospy.get_param(
            "~reference_topic", "/controller/reference_trajectory_window")
        wrench_topic = rospy.get_param(
            "~wrench_topic", "/controller/generalized_force")
        self.wrench_pub = rospy.Publisher(
            wrench_topic, WrenchStamped, queue_size=5)
        self.status_pub = rospy.Publisher(
            "~status", ControllerStatus, queue_size=5)
        self.force_pub = rospy.Publisher(
            "~candidate_forces", JointState, queue_size=5)
        self.timing_pub = rospy.Publisher(
            "~timing", Vector3Stamped, queue_size=5)
        self.error_x_pub = rospy.Publisher(
            "/aquaflow/tracking_error/x_body_m", Float64, queue_size=10)
        self.error_y_pub = rospy.Publisher(
            "/aquaflow/tracking_error/y_body_m", Float64, queue_size=10)
        self.error_yaw_pub = rospy.Publisher(
            "/aquaflow/tracking_error/yaw_rad", Float64, queue_size=10)
        self.error_norm_pub = rospy.Publisher(
            "/aquaflow/tracking_error/xy_norm_m", Float64, queue_size=10)

        rospy.Subscriber(odom_topic, Odometry,
                         self.odom_cb, queue_size=20)
        rospy.Subscriber(reference_topic,
                         TrajectoryPointWindow, self.window_cb, queue_size=5)
        self.experiment_gate = ExperimentGate("controller", depth_hold=True)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self.control_cb)
        rospy.loginfo(
            "HOFA policy controller ready: checkpoint=%s epoch=%d device=%s "
            "rate=%.1f Hz odom=%s reference=%s wrench=%s",
            checkpoint, self.policy.epoch, device, self.rate_hz,
            odom_topic, reference_topic, wrench_topic)

    @staticmethod
    def stamp(msg):
        value = msg.header.stamp.to_sec()
        return value if value > 0.0 else rospy.Time.now().to_sec()

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

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_stamp = self.stamp(msg)

    def window_cb(self, msg):
        if msg.valid and len(msg.points) >= CONTRACT.horizon:
            self.window = msg
            self.reference_stamp = self.stamp(msg)

    def reset_history(self):
        self.previous_wrench.fill(0.0)

    def publish_wrench(self, wrench):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = float(wrench[0])
        msg.wrench.force.y = float(wrench[1])
        msg.wrench.torque.z = float(wrench[2])
        msg.wrench.force.z = (
            self.experiment_gate.depth_force() if self.enabled else 0.0)
        self.wrench_pub.publish(msg)

    def publish_zero(self):
        self.publish_wrench(np.zeros(3, dtype=np.float32))

    def publish_status(self, success, callback_ms, inference_ms=0.0,
                       position_error=0.0, yaw_error=0.0):
        msg = ControllerStatus()
        msg.header.stamp = rospy.Time.now()
        msg.state = self.state_name
        msg.solver_success = bool(success)
        msg.solver_iterations = 1 if success else 0
        msg.objective_value = 0.0
        msg.layer1_time_ms = 0.0
        msg.layer2_time_ms = float(inference_ms)
        msg.callback_time_ms = float(callback_ms)
        msg.position_error_m = float(position_error)
        msg.yaw_error_rad = float(yaw_error)
        self.status_pub.publish(msg)

    def reject_cycle(self, state_name, started, reason):
        self.state_name = state_name
        self.failure_count += 1
        self.reset_history()
        self.publish_zero()
        rospy.logwarn_throttle(
            2.0, "HOFA policy control rejected %s; failures=%d",
            reason, self.failure_count)
        self.publish_status(False, (time.perf_counter() - started) * 1000.0)

    def control_cb(self, _event):
        started = time.perf_counter()
        now = rospy.Time.now().to_sec()
        if (not self.enabled or not self.experiment_gate.allows(now) or
                (self.experiment_gate.topic and
                 self.experiment_gate.start_at is not None and
                 (self.reference_stamp is None or
                  self.reference_stamp < self.experiment_gate.start_at))):
            self.state_name = "DISABLED"
            self.reset_history()
            self.publish_zero()
            self.publish_status(True, (time.perf_counter() - started) * 1000.0)
            return

        if self.odom is None or self.odom_stamp is None:
            self.reject_cycle("WAITING_FOR_STATE", started, "missing_odometry")
            return
        if self.window is None or self.reference_stamp is None:
            self.reject_cycle("WAITING_FOR_REFERENCE", started,
                              "missing_reference")
            return
        odom_age = now - self.odom_stamp
        reference_age = now - self.reference_stamp
        if odom_age < 0.0 or odom_age > self.max_odom_age:
            self.reject_cycle("SAFETY_STOP", started, "stale_odometry")
            return
        if reference_age < 0.0 or reference_age > self.max_reference_age:
            self.reject_cycle("SAFETY_STOP", started, "stale_reference")
            return

        odom = self.odom_row(self.odom)
        if (abs(odom["x"]) > self.max_position_abs[0] or
                abs(odom["y"]) > self.max_position_abs[1] or
                math.hypot(odom["u"], odom["v"]) > self.max_world_speed or
                abs(odom["r"]) > self.max_yaw_rate):
            self.reject_cycle("SAFETY_STOP", started, "state_envelope")
            return

        try:
            state = state_features(odom)
            reference = reference_features(
                odom, self.reference_rows(self.window))
            inference_start = time.perf_counter_ns()
            wrench, force, normalized = self.policy.predict(
                state, reference, self.previous_wrench)
            inference_ms = (
                time.perf_counter_ns() - inference_start) / 1.0e6
        except Exception as error:
            self.reject_cycle(
                "SAFETY_STOP", started,
                "inference_%s" % type(error).__name__)
            rospy.logerr_throttle(2.0, "Policy inference failed: %s", error)
            return

        if not all(np.all(np.isfinite(value))
                   for value in (wrench, force, normalized)):
            self.reject_cycle("SAFETY_STOP", started, "nonfinite_output")
            return
        if np.any(force < self.force_min - 1.0e-4) or np.any(
                force > self.force_max + 1.0e-4):
            self.reject_cycle("SAFETY_STOP", started,
                              "candidate_force_violation")
            return
        reconstructed = self.allocation.dot(force)
        if not np.allclose(reconstructed, wrench, atol=2.0e-3, rtol=2.0e-4):
            self.reject_cycle("SAFETY_STOP", started,
                              "wrench_force_inconsistency")
            return

        self.publish_wrench(wrench)
        force_msg = JointState()
        force_msg.header.stamp = rospy.Time.now()
        force_msg.name = ["T1", "T2", "T3", "T4"]
        force_msg.effort = [float(value) for value in force]
        force_msg.position = [float(value) for value in normalized]
        self.force_pub.publish(force_msg)

        timing = Vector3Stamped()
        timing.header = force_msg.header
        timing.vector.x = inference_ms
        timing.vector.y = odom_age
        timing.vector.z = reference_age
        self.timing_pub.publish(timing)

        position_error = math.hypot(reference[0, 0], reference[0, 1])
        signed_yaw_error = math.atan2(reference[0, 2], reference[0, 3])
        self.error_x_pub.publish(Float64(data=float(reference[0, 0])))
        self.error_y_pub.publish(Float64(data=float(reference[0, 1])))
        self.error_yaw_pub.publish(Float64(data=float(signed_yaw_error)))
        self.error_norm_pub.publish(Float64(data=position_error))

        self.previous_wrench = np.asarray(wrench, dtype=np.float32).copy()
        self.failure_count = 0
        self.state_name = "ACTIVE"
        self.publish_status(
            True, (time.perf_counter() - started) * 1000.0,
            inference_ms, position_error, abs(signed_yaw_error))

    def shutdown(self):
        self.enabled = False
        self.reset_history()
        self.publish_zero()
        rospy.loginfo("HOFA policy controller shut down")


if __name__ == "__main__":
    try:
        node = HofaPolicyControllerNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
