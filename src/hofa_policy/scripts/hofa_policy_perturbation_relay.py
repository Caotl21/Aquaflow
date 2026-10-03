#!/usr/bin/env python3
"""Inject deterministic sensor, reference, and actuator perturbations."""
import copy
import heapq
import json
import math
import random
import threading

import rospy
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry

from hofa_mpc_ros.msg import TrajectoryPointWindow


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def set_yaw(q, yaw):
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)


class PerturbationRelay:
    def __init__(self):
        rospy.init_node("hofa_policy_perturbation_relay")
        config_path = rospy.get_param("~config_file")
        with open(config_path) as stream:
            self.config = json.load(stream)
        self.random = random.Random(int(self.config.get("seed", 0)))
        self.lock = threading.Lock()
        self.queue = []
        self.sequence = 0
        self.stats = {"queued": 0, "published": 0, "dropped": 0}

        raw_odom = rospy.get_param("~raw_odom_topic", "/bricsbot/odometry")
        perturbed_odom = rospy.get_param(
            "~perturbed_odom_topic", "/hofa_policy/robust/odometry")
        raw_reference = rospy.get_param(
            "~raw_reference_topic", "/controller/reference_trajectory_window")
        perturbed_reference = rospy.get_param(
            "~perturbed_reference_topic",
            "/hofa_policy/robust/reference_trajectory_window")
        raw_wrench = rospy.get_param(
            "~raw_wrench_topic", "/hofa_policy/robust/raw_wrench")
        delayed_wrench = rospy.get_param(
            "~delayed_wrench_topic", "/controller/generalized_force")

        self.odom_pub = rospy.Publisher(
            perturbed_odom, Odometry, queue_size=100)
        self.reference_pub = rospy.Publisher(
            perturbed_reference, TrajectoryPointWindow, queue_size=20)
        self.wrench_pub = rospy.Publisher(
            delayed_wrench, WrenchStamped, queue_size=20)
        rospy.Subscriber(raw_odom, Odometry, self.odom_cb, queue_size=100)
        rospy.Subscriber(raw_reference, TrajectoryPointWindow,
                         self.reference_cb, queue_size=20)
        rospy.Subscriber(raw_wrench, WrenchStamped,
                         self.wrench_cb, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(0.002), self.flush)
        rospy.loginfo(
            "HOFA robustness relay ready: config=%s seed=%d",
            config_path, int(self.config.get("seed", 0)))

    @staticmethod
    def validate_probability(value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("dropout probability must be in [0, 1]")
        return value

    def enqueue(self, publisher, message, settings):
        dropout = self.validate_probability(settings.get("dropout", 0.0))
        if self.random.random() < dropout:
            self.stats["dropped"] += 1
            return
        delay_ms = max(0.0, float(settings.get("delay_ms", 0.0)))
        jitter_ms = max(0.0, float(settings.get("jitter_ms", 0.0)))
        if jitter_ms:
            delay_ms = max(
                0.0, delay_ms + self.random.gauss(0.0, jitter_ms))
        release = rospy.Time.now().to_sec() + delay_ms / 1000.0
        with self.lock:
            self.sequence += 1
            heapq.heappush(
                self.queue, (release, self.sequence, publisher, message))
            self.stats["queued"] += 1

    def perturb_odom(self, source):
        msg = copy.deepcopy(source)
        settings = self.config.get("odom", {})
        position_std = max(
            0.0, float(settings.get("position_std_m", 0.0)))
        yaw_std = math.radians(max(
            0.0, float(settings.get("yaw_std_deg", 0.0))))
        velocity_std = max(
            0.0, float(settings.get("velocity_std_mps", 0.0)))
        yaw_rate_std = max(
            0.0, float(settings.get("yaw_rate_std_radps", 0.0)))
        position_bias = settings.get("position_bias_xy_m", [0.0, 0.0])
        velocity_bias = settings.get("velocity_bias_uv_mps", [0.0, 0.0])
        yaw_bias = math.radians(float(settings.get("yaw_bias_deg", 0.0)))
        yaw_rate_bias = float(settings.get("yaw_rate_bias_radps", 0.0))

        msg.pose.pose.position.x += (
            float(position_bias[0]) + self.random.gauss(0.0, position_std))
        msg.pose.pose.position.y += (
            float(position_bias[1]) + self.random.gauss(0.0, position_std))
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        set_yaw(msg.pose.pose.orientation,
                yaw + yaw_bias + self.random.gauss(0.0, yaw_std))
        msg.twist.twist.linear.x += (
            float(velocity_bias[0]) +
            self.random.gauss(0.0, velocity_std))
        msg.twist.twist.linear.y += (
            float(velocity_bias[1]) +
            self.random.gauss(0.0, velocity_std))
        msg.twist.twist.angular.z += (
            yaw_rate_bias + self.random.gauss(0.0, yaw_rate_std))
        return msg

    def odom_cb(self, msg):
        settings = self.config.get("odom", {})
        self.enqueue(self.odom_pub, self.perturb_odom(msg), settings)

    def reference_cb(self, msg):
        settings = self.config.get("reference", {})
        self.enqueue(self.reference_pub, copy.deepcopy(msg), settings)

    def wrench_cb(self, msg):
        settings = self.config.get("wrench", {})
        self.enqueue(self.wrench_pub, copy.deepcopy(msg), settings)

    def flush(self, _event):
        ready = []
        now = rospy.Time.now().to_sec()
        with self.lock:
            while self.queue and self.queue[0][0] <= now:
                ready.append(heapq.heappop(self.queue))
        for _, _, publisher, message in ready:
            publisher.publish(message)
            self.stats["published"] += 1
        rospy.loginfo_throttle(
            10.0,
            "Robustness relay stats: queued=%d published=%d dropped=%d pending=%d",
            self.stats["queued"], self.stats["published"],
            self.stats["dropped"], len(self.queue))


if __name__ == "__main__":
    try:
        PerturbationRelay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
