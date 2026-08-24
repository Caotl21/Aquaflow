#!/usr/bin/env python3
"""ROS wrapper for the ESDF-costmap global A* privileged teacher."""
import os
import sys

# catkin's devel-space launcher executes this file through a wrapper located
# in ``devel/lib/<package>``.  That wrapper directory is not the source
# ``scripts`` directory, so explicitly add the directory containing this
# file before importing the ROS-independent planning core.  The same code
# also works in an installed catkin space where teacher_core.py is installed
# beside the node.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from teacher_core import (ESDFCostmap, GlobalAStarPlanner, LocalReferenceSampler,
                          point_to_polyline_distance, round_corners, shortcut_smooth,
                          path_is_free, time_parameterize)


def yaw_from_quaternion(q):
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion_from_yaw(yaw):
    import math
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PrivilegedTeacherNode:
    def __init__(self):
        self.odom, self.nominal, self.global_plan = None, None, None
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.obstacles = rospy.get_param("~obstacles", [])
        self.goal = rospy.get_param("~global_goal", None)
        self.goal_depth = float(rospy.get_param("~goal_depth", 1.0))
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.goal_yaw = (float(self.goal["yaw"]) if isinstance(self.goal, dict)
                         and "yaw" in self.goal else None)
        self.replan_deviation = float(rospy.get_param("~replan_deviation", 1.0))
        self.max_speed = float(rospy.get_param("~max_speed", 0.35))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.5))
        self.max_accel = float(rospy.get_param("~max_accel_mps2", 0.12))
        self.max_decel = float(rospy.get_param("~max_decel_mps2", 0.18))
        self.output_spacing = float(rospy.get_param("~output_spacing", 0.20))
        self.corner_radius = float(rospy.get_param("~corner_radius", 0.35))
        self.cost_scaling_factor = float(rospy.get_param("~cost_scaling_factor", 3.0))
        self.costmap = ESDFCostmap(
            float(rospy.get_param("~pool_x_min", -7.0)), float(rospy.get_param("~pool_x_max", 7.0)),
            float(rospy.get_param("~pool_y_min", -3.8)), float(rospy.get_param("~pool_y_max", 3.8)),
            max(0.05, float(rospy.get_param("~grid_resolution", 0.10))), self.obstacles,
            float(rospy.get_param("~robot_radius", 0.365)), float(rospy.get_param("~safety_margin", 0.25)),
            cost_scaling_factor=self.cost_scaling_factor)
        self.planner = GlobalAStarPlanner(self.costmap)
        self.sampler = LocalReferenceSampler(
            max(2, int(rospy.get_param("~horizon_points", 10))),
            float(rospy.get_param("~local_spacing", 0.20)))
        # The global topic is the hand-off to the runtime local planner.  The
        # teacher_reference topic is retained for visualization/backward
        # compatibility, but consumers should not re-project this already
        # local window as if it were the global route.
        self.reference_pub = rospy.Publisher("/aquaflow/teacher_reference", Path, queue_size=1, latch=True)
        self.global_pub = rospy.Publisher("/aquaflow/teacher_global_path", Path, queue_size=1, latch=True)
        self.local_points_pub = rospy.Publisher("/aquaflow/teacher_local_points", MarkerArray,
                                                queue_size=1, latch=True)
        self.global_points_pub = rospy.Publisher("/aquaflow/teacher_global_points", MarkerArray,
                                                 queue_size=1, latch=True)
        self.label_pub = rospy.Publisher("/aquaflow/teacher_labels", Float32MultiArray, queue_size=1)
        # Clear any latched global point marker from an earlier teacher. The
        # global Path remains published; only its non-uniform point overlay is
        # intentionally removed from RViz.
        clear = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        clear.markers.append(delete)
        self.global_points_pub.publish(clear)
        rospy.Subscriber("/%s/odometry" % self.vehicle_name, Odometry,
                         self.odom_cb, queue_size=1)
        rospy.Subscriber("/aquaflow/nominal_path", Path, self.nominal_cb, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.update)
        rospy.loginfo("ESDF A* teacher ready: vehicle=%s  goal=%s  %d obstacles, %dx%d grid",
                      self.vehicle_name, self.goal, len(self.obstacles),
                      self.costmap.width, self.costmap.height)

    def odom_cb(self, msg):
        self.odom = msg

    def nominal_cb(self, msg):
        self.nominal = msg

    def goal_cb(self, msg):
        """Accept RViz 2D Nav Goal in the shared world_ned frame."""
        frame = msg.header.frame_id or "world_ned"
        if frame != "world_ned":
            rospy.logwarn_throttle(5.0,
                                   "2D Nav Goal frame is '%s'; expected world_ned", frame)
        pose = msg.pose
        self.goal = {"x": float(pose.position.x),
                     "y": float(pose.position.y),
                     "z": self.goal_depth,
                     "yaw": yaw_from_quaternion(pose.orientation)}
        self.goal_yaw = self.goal["yaw"]
        self.global_plan = None
        self.sampler.reset()
        rospy.loginfo("New RViz goal: x=%.2f y=%.2f z=%.2f yaw=%.2f",
                      self.goal["x"], self.goal["y"], self.goal["z"],
                      self.goal["yaw"])

    def goal_pose(self):
        if self.goal is not None:
            return (float(self.goal["x"]), float(self.goal["y"]), self.goal_depth)
        pose = self.nominal.poses[-1].pose.position
        return (pose.x, pose.y, self.goal_depth)

    def make_path(self, samples, frame, now):
        path = Path()
        path.header.frame_id, path.header.stamp = frame, now
        for x, y, z, yaw, _speed, relative_time in samples:
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = now + rospy.Duration(relative_time)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            q = quaternion_from_yaw(yaw)
            pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
            pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
            path.poses.append(pose)
        return path

    def make_point_markers(self, samples, frame, now, local=False):
        """Create readable point markers without changing the Path topics."""
        output = MarkerArray()
        points = Marker()
        points.header.frame_id, points.header.stamp = frame, now
        points.ns = "aquaflow_teacher_local" if local else "aquaflow_teacher_global"
        points.id = 0
        points.type = Marker.SPHERE_LIST
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0
        points.scale.x = points.scale.y = points.scale.z = 0.07 if local else 0.05
        if local:
            points.color.r, points.color.g, points.color.b, points.color.a = 0.0, 0.55, 1.0, 0.95
        else:
            points.color.r, points.color.g, points.color.b, points.color.a = 0.70, 0.0, 1.0, 0.80
        points.points = [Point(x=item[0], y=item[1], z=item[2]) for item in samples]
        output.markers.append(points)
        if local and samples:
            target = Marker()
            target.header.frame_id, target.header.stamp = frame, now
            target.ns = points.ns
            target.id = 1
            target.type = Marker.SPHERE
            target.action = Marker.ADD
            target.pose.position.x = samples[min(2, len(samples) - 1)][0]
            target.pose.position.y = samples[min(2, len(samples) - 1)][1]
            target.pose.position.z = samples[min(2, len(samples) - 1)][2]
            target.pose.orientation.w = 1.0
            target.scale.x = target.scale.y = target.scale.z = 0.14
            target.color.r, target.color.g, target.color.b, target.color.a = 1.0, 0.85, 0.0, 1.0
            output.markers.append(target)
        return output

    def replan(self, current, goal):
        cells = self.planner.plan(current[:2], goal[:2])
        if not cells:
            self.global_plan = None
            return False
        smooth = shortcut_smooth(self.costmap, cells)
        smooth = round_corners(self.costmap, smooth, self.corner_radius)
        if not path_is_free(self.costmap, smooth):
            rospy.logwarn("teacher: smoothed path failed final collision check")
            self.global_plan = None
            return False
        self.global_plan = time_parameterize(smooth, goal[2], self.max_speed,
                                             self.max_yaw_rate, self.output_spacing,
                                             self.max_accel, self.max_decel)
        if self.global_plan and self.goal_yaw is not None:
            last = self.global_plan[-1]
            self.global_plan[-1] = (last[0], last[1], last[2], self.goal_yaw,
                                    0.0, last[5])
        self.sampler.reset()
        return bool(self.global_plan)

    def update(self, _event):
        if self.odom is None or self.nominal is None or not self.nominal.poses:
            rospy.logwarn_throttle(5.0, "teacher: waiting — odom=%s nominal=%s",
                                   "ok" if self.odom else "None",
                                   "ok(%d)" % len(self.nominal.poses) if self.nominal and self.nominal.poses else "None")
            return
        now = rospy.Time.now()
        p = self.odom.pose.pose.position
        current = (p.x, p.y, p.z)
        goal = self.goal_pose()
        if self.global_plan is None or point_to_polyline_distance(current[:2], self.global_plan) > self.replan_deviation:
            ok = self.replan(current, goal)
            rospy.loginfo("teacher: replan from=(%.2f,%.2f) goal=(%.2f,%.2f) ok=%s plan_len=%s",
                          current[0], current[1], goal[0], goal[1], ok,
                          len(self.global_plan) if self.global_plan else 0)
            if not ok:
                self.label_pub.publish(Float32MultiArray(data=[1.0, 0.0, 1.0]))
                return
        frame = self.nominal.header.frame_id or "world_ned"
        self.global_pub.publish(self.make_path(self.global_plan, frame, now))
        local = self.sampler.sample(self.global_plan, current[:2])
        self.reference_pub.publish(self.make_path(local, frame, now))
        self.local_points_pub.publish(self.make_point_markers(local, frame, now, local=True))
        rospy.loginfo_throttle(2.0, "teacher: published %d local points near (%.2f,%.2f)",
                               len(local), current[0], current[1])
        clearance = min(self.costmap.signed_distance(item[0], item[1]) - self.costmap.inflation
                        for item in self.global_plan)
        risk = 1.0 if clearance < 0.15 else 0.0
        self.label_pub.publish(Float32MultiArray(data=[risk, clearance, 0.0]))


if __name__ == "__main__":
    rospy.init_node("privileged_teacher")
    PrivilegedTeacherNode()
    rospy.spin()
