#!/usr/bin/env python3
"""Continuously publish one generated teacher trajectory as nav_msgs/Path."""
import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from hofa_policy.trajectory_suite import validate_track


def quaternion_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class TrajectoryPublisher:
    def __init__(self):
        filename = rospy.get_param("~trajectory_file")
        with open(filename) as handle:
            self.track = validate_track(json.load(handle))
        self.publisher = rospy.Publisher(
            "/hofa_policy/teacher_global_path", Path, queue_size=1, latch=True)
        # run_mpc_experiment publishes the terminal pose through the standard
        # navigation-goal topic and waits for a subscriber before arming the
        # shared experiment epoch.  Generated trajectories already contain
        # their terminal pose, so this subscriber is deliberately a no-op; it
        # preserves the experiment barrier protocol without replanning.
        self.goal_subscriber = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped,
            lambda _message: None, queue_size=1)
        self.frame = self.track.get("frame_id", "world_ned")
        self.timer = rospy.Timer(rospy.Duration(0.2), self.publish)
        rospy.loginfo("HOFA policy trajectory ready: %s family=%s points=%d",
                      self.track["id"], self.track["family"],
                      len(self.track["points"]))

    def publish(self, _event):
        now = rospy.Time.now()
        msg = Path()
        msg.header.frame_id = self.frame
        msg.header.stamp = now
        for x, y, z, yaw in self.track["points"]:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            q = quaternion_from_yaw(yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            msg.poses.append(pose)
        self.publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node("hofa_policy_trajectory_publisher")
    TrajectoryPublisher()
    rospy.spin()
