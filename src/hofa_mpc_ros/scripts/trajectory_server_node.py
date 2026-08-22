#!/usr/bin/env python3
"""Trajectory server node for HOFA-MPC controller.

Publishes reference trajectories as TrajectoryPoint messages and
nav_msgs/Path for RViz visualization.  Supports parameter-based
trajectory selection and reset.
"""
import math
import sys
import os
import rospy
import tf.transformations as tft
from std_msgs.msg import Header, String
from geometry_msgs.msg import Pose, Twist, Accel, PoseStamped, PoseArray, Quaternion
from nav_msgs.msg import Path
from hofa_mpc_ros.msg import TrajectoryPoint
from hofa_mpc_ros.trajectory import create_trajectory


def yaw_to_quat(yaw):
    q = tft.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class TrajectoryServerNode:
    def __init__(self):
        rospy.init_node("trajectory_server")

        self.traj_name = rospy.get_param("~trajectory",
                                          rospy.get_param("~default_trajectory", "hover"))
        self.startup_time = float(rospy.get_param("~startup_time_s", 3.0))
        self.yaw_mode = rospy.get_param("~yaw_mode", "tangent")
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.path_frame = rospy.get_param("~path_frame", "world_ned")

        # Load trajectory config from YAML
        traj_params = self._load_trajectory_params()

        # Create trajectory generator
        self.trajectory = create_trajectory(self.traj_name, **traj_params)

        # Publishers
        self.ref_pub = rospy.Publisher(
            "~reference", TrajectoryPoint, queue_size=10)
        self.path_pub = rospy.Publisher(
            "~reference_path", Path, queue_size=1, latch=True)
        self.path_array_pub = rospy.Publisher(
            "~reference_poses", PoseArray, queue_size=1, latch=True)

        # Subscriber for trajectory switching
        rospy.Subscriber("~set_trajectory", String, self._set_trajectory_cb)

        # Publish initial path
        self._publish_path()

        # Timer for continuous reference publishing
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate),
                                  self._timer_cb)

        rospy.loginfo("Trajectory server ready: trajectory=%s, startup=%.1fs",
                      self.traj_name, self.startup_time)

    def _load_trajectory_params(self):
        """Load trajectory-specific parameters."""
        params = {}
        if self.traj_name == "hover":
            params["x"] = float(rospy.get_param("~x", 0.0))
            params["y"] = float(rospy.get_param("~y", 0.0))
            params["psi"] = float(rospy.get_param("~psi", 0.0))
            params["yaw_mode"] = rospy.get_param("~yaw_mode", "fixed")
            if params["yaw_mode"] == "scan":
                params["yaw_scan_amplitude"] = float(
                    rospy.get_param("~yaw_scan_amplitude", 0.5))
                params["yaw_scan_freq"] = float(
                    rospy.get_param("~yaw_scan_freq", 0.1))
        elif self.traj_name == "line":
            params["x0"] = float(rospy.get_param("~x0", 0.0))
            params["y0"] = float(rospy.get_param("~y0", 0.0))
            params["x1"] = float(rospy.get_param("~x1", 1.5))
            params["y1"] = float(rospy.get_param("~y1", 0.0))
            params["speed"] = float(rospy.get_param("~speed", 0.3))
        elif self.traj_name == "circle":
            params["cx"] = float(rospy.get_param("~cx", 0.0))
            params["cy"] = float(rospy.get_param("~cy", 0.0))
            params["radius"] = float(rospy.get_param("~radius", 0.8))
            params["speed"] = float(rospy.get_param("~speed", 0.3))
        elif self.traj_name == "figure_eight":
            params["cx"] = float(rospy.get_param("~cx", 0.0))
            params["cy"] = float(rospy.get_param("~cy", 0.0))
            params["scale_x"] = float(rospy.get_param("~scale_x", 0.8))
            params["scale_y"] = float(rospy.get_param("~scale_y", 0.4))
            params["speed"] = float(rospy.get_param("~speed", 0.3))
        params["yaw_mode"] = rospy.get_param("~yaw_mode", self.yaw_mode)
        params["startup_time"] = self.startup_time
        return params

    def _set_trajectory_cb(self, msg):
        """Switch trajectory on-the-fly."""
        new_name = msg.data.strip()
        if new_name == self.traj_name:
            return
        params = {}
        # Reload params for new trajectory
        old_name = self.traj_name
        self.traj_name = new_name
        params = self._load_trajectory_params()
        self.trajectory = create_trajectory(new_name, **params)
        self._publish_path()
        rospy.loginfo("Switched trajectory: %s -> %s", old_name, new_name)

    def _publish_path(self):
        """Pre-compute and publish reference path for RViz."""
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.path_frame

        poses = PoseArray()
        poses.header = path.header

        dt = 0.05
        t_end = 30.0
        t = self.startup_time
        while t < t_end:
            ref = self.trajectory.get_reference(t)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = ref.x
            ps.pose.position.y = ref.y
            ps.pose.position.z = 0.0
            ps.pose.orientation = yaw_to_quat(ref.psi)
            path.poses.append(ps)
            poses.poses.append(ps.pose)
            t += dt

        self.path_pub.publish(path)
        self.path_array_pub.publish(poses)

    def _timer_cb(self, _event):
        """Publish current reference point at fixed rate."""
        t = rospy.Time.now().to_sec()

        ref = self.trajectory.get_reference(t)

        msg = TrajectoryPoint()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.path_frame

        msg.pose.position.x = ref.x
        msg.pose.position.y = ref.y
        msg.pose.position.z = 0.0
        msg.pose.orientation = yaw_to_quat(ref.psi)

        msg.twist.linear.x = ref.dx
        msg.twist.linear.y = ref.dy
        msg.twist.linear.z = 0.0
        msg.twist.angular.z = ref.dpsi

        msg.accel.linear.x = ref.ddx
        msg.accel.linear.y = ref.ddy
        msg.accel.linear.z = 0.0
        msg.accel.angular.z = ref.ddpsi

        msg.valid = True
        msg.trajectory_id = self.traj_name

        self.ref_pub.publish(msg)


if __name__ == "__main__":
    try:
        TrajectoryServerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
