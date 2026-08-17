#!/usr/bin/env python3
"""Publish a deterministic straight nominal path in world_ned."""
import math
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def main():
    rospy.init_node("nominal_reference_publisher")
    topic = rospy.get_param("~topic", "/aquaflow/nominal_path")
    frame = rospy.get_param("~frame_id", "world_ned")
    trajectory_type = rospy.get_param("~trajectory_type", "line")
    x0 = float(rospy.get_param("~x_start", -5.0))
    x1 = float(rospy.get_param("~x_end", 5.0))
    y = float(rospy.get_param("~y", 0.0))
    z = float(rospy.get_param("~z", 0.0))
    spacing = max(0.05, float(rospy.get_param("~spacing", 0.25)))
    yaw = float(rospy.get_param("~yaw", 0.0))
    if trajectory_type == "figure_eight":
        center_x = float(rospy.get_param("~figure_eight_center_x", 0.0))
        center_y = float(rospy.get_param("~figure_eight_center_y", 0.0))
        amplitude_x = float(rospy.get_param("~figure_eight_amplitude_x", 5.0))
        amplitude_y = float(rospy.get_param("~figure_eight_amplitude_y", 4.0))
        count = max(80, int(rospy.get_param("~figure_eight_points", 240)))
        theta0 = -0.5 * math.pi
        theta1 = theta0 + 2.0 * math.pi
    elif trajectory_type == "line":
        count = max(2, int(math.ceil(abs(x1 - x0) / spacing)) + 1)
    else:
        raise rospy.ROSInitException("unsupported trajectory_type: %s" % trajectory_type)

    pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)
    msg = Path()
    msg.header.frame_id = frame
    q = quat_from_yaw(yaw)
    for i in range(count):
        u = float(i) / float(count - 1)
        pose = PoseStamped()
        pose.header.frame_id = frame
        if trajectory_type == "figure_eight":
            theta = theta0 + u * (theta1 - theta0)
            st, ct = math.sin(theta), math.cos(theta)
            pose.pose.position.x = center_x + amplitude_x * st
            pose.pose.position.y = center_y + amplitude_y * st * ct
            dx = amplitude_x * ct
            dy = amplitude_y * (ct * ct - st * st)
            pose_yaw = math.atan2(dy, dx)
            q = quat_from_yaw(pose_yaw)
        else:
            pose.pose.position.x = x0 + u * (x1 - x0)
            pose.pose.position.y = y
            q = quat_from_yaw(yaw)
        pose.pose.position.z = z
        pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
        pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
        msg.poses.append(pose)
    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        for pose in msg.poses:
            pose.header.stamp = msg.header.stamp
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
