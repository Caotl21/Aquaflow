#!/usr/bin/env python3
"""Accumulate odometry into a bounded nav_msgs/Path for RViz inspection."""
from collections import deque
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class ActualPathRecorder:
    def __init__(self):
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.frame_id = rospy.get_param("~frame_id", "world_ned")
        self.output_topic = rospy.get_param("~output_topic", "/aquaflow/actual_path")
        self.min_distance = max(0.0, float(rospy.get_param("~min_distance", 0.02)))
        self.min_interval = max(0.0, float(rospy.get_param("~min_interval", 0.05)))
        self.max_points = max(10, int(rospy.get_param("~max_points", 5000)))
        self.points = deque(maxlen=self.max_points)
        self.last_position = None
        self.last_stamp = None
        self.pub = rospy.Publisher(self.output_topic, Path, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/aquaflow/actual_path_points", MarkerArray,
                                          queue_size=1, latch=True)
        rospy.Subscriber("/%s/odometry" % self.vehicle_name, Odometry,
                         self.odom_cb, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_path)
        rospy.on_shutdown(self.shutdown)

    def odom_cb(self, msg):
        source_frame = msg.header.frame_id or self.frame_id
        if source_frame != self.frame_id:
            rospy.logwarn_throttle(5.0, "odometry frame is %s, expected %s", source_frame, self.frame_id)
            return
        stamp = msg.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()
        position = msg.pose.pose.position
        current = (position.x, position.y, position.z)
        if self.last_stamp is not None and stamp < self.last_stamp:
            rospy.logwarn("odometry timestamp moved backwards; resetting actual path")
            self.points.clear()
            self.last_position = None
        if self.last_position is not None:
            distance = math.sqrt(sum((current[i] - self.last_position[i]) ** 2 for i in range(3)))
            interval = (stamp - self.last_stamp).to_sec()
            if distance < self.min_distance and interval < self.min_interval:
                return
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose = msg.pose.pose
        self.points.append(pose)
        self.last_position = current
        self.last_stamp = stamp

    def publish_path(self, _event):
        # Odometry callbacks can append to the deque while this timer thread
        # is publishing.  Iterate over a stable snapshot so the added RViz
        # point layer cannot terminate the recorder with "deque mutated".
        snapshot = list(self.points)
        path = Path()
        path.header.stamp = self.last_stamp or rospy.Time.now()
        path.header.frame_id = self.frame_id
        path.poses = snapshot
        self.pub.publish(path)

        points = MarkerArray()
        trail = Marker()
        trail.header.frame_id = self.frame_id
        trail.header.stamp = path.header.stamp
        trail.ns = "aquaflow_actual_path"
        trail.id = 0
        trail.type = Marker.SPHERE_LIST
        trail.action = Marker.ADD
        trail.pose.orientation.w = 1.0
        trail.scale.x = trail.scale.y = trail.scale.z = 0.10
        trail.color.r, trail.color.g, trail.color.b, trail.color.a = 1.0, 0.05, 0.05, 0.90
        trail.points = [Point(p.pose.position.x, p.pose.position.y, p.pose.position.z)
                        for p in snapshot]
        points.markers.append(trail)
        self.marker_pub.publish(points)

    def shutdown(self):
        self.publish_path(None)


if __name__ == "__main__":
    rospy.init_node("actual_path_recorder")
    ActualPathRecorder()
    rospy.spin()
