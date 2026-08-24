#!/usr/bin/env python3
"""Deterministic planar PID tracker; publishes generalized force only."""
import math
import rospy
from geometry_msgs.msg import Point, WrenchStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PlanarPIDTracker:
    def __init__(self):
        self.odom = None
        self.path = None
        self.last_time = None
        self.int_x = self.int_y = self.int_yaw = 0.0
        self.int_z = 0.0
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.25))
        self.path_timeout = float(rospy.get_param("~path_timeout", 0.50))
        self.lookahead = max(0, int(rospy.get_param("~lookahead_index", 2)))
        self.kp_xy = float(rospy.get_param("~kp_xy", 15.0))
        self.kd_xy = float(rospy.get_param("~kd_xy", 3.0))
        self.ki_xy = float(rospy.get_param("~ki_xy", 0.0))
        self.kp_yaw = float(rospy.get_param("~kp_yaw", 4.0))
        self.kd_yaw = float(rospy.get_param("~kd_yaw", 1.0))
        self.ki_yaw = float(rospy.get_param("~ki_yaw", 0.3))
        self.yaw_deadband = math.radians(max(0.0, float(rospy.get_param("~yaw_deadband_deg", 1.0))))
        self.kp_z = float(rospy.get_param("~kp_z", 3.0))
        self.kd_z = float(rospy.get_param("~kd_z", 1.0))
        self.ki_z = float(rospy.get_param("~ki_z", 0.5))
        self.z_integral_limit = max(0.0, float(rospy.get_param("~z_integral_limit", 5.0)))
        self.max_fx = float(rospy.get_param("~max_fx", 4.0))
        self.max_fy = float(rospy.get_param("~max_fy", 4.0))
        self.max_nz = float(rospy.get_param("~max_nz", 1.4))
        self.max_fz = float(rospy.get_param("~max_fz", 20.0))
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.odom_topic = "/%s/odometry" % self.vehicle_name
        self.reference_topic = rospy.get_param("~reference_topic", "/aquaflow/nominal_path")
        self.pub = rospy.Publisher("/controller/generalized_force", WrenchStamped, queue_size=1)
        # Scalar topics are intentionally separate so rqt_plot can subscribe
        # without decoding an array: x/y are body-frame metres, yaw is radians.
        self.error_x_pub = rospy.Publisher("/aquaflow/tracking_error/x_body_m", Float64, queue_size=10)
        self.error_y_pub = rospy.Publisher("/aquaflow/tracking_error/y_body_m", Float64, queue_size=10)
        self.error_yaw_pub = rospy.Publisher("/aquaflow/tracking_error/yaw_rad", Float64, queue_size=10)
        self.error_norm_pub = rospy.Publisher("/aquaflow/tracking_error/xy_norm_m", Float64, queue_size=10)
        self.error_marker_pub = rospy.Publisher("/aquaflow/tracking_error_markers", MarkerArray,
                                                queue_size=1)
        self.target_marker_pub = rospy.Publisher("/aquaflow/tracking_target_markers", MarkerArray,
                                                 queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.reference_topic, Path, self.path_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~rate", 20.0))), self.update)
        rospy.on_shutdown(self.shutdown)

    def odom_cb(self, msg):
        self.odom = msg

    def path_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != "world_ned":
            rospy.logwarn_throttle(5.0, "reference path frame must be world_ned, got %s", msg.header.frame_id)
        self.path = msg

    def publish_zero(self, stamp=None):
        msg = WrenchStamped()
        msg.header.stamp = stamp or rospy.Time.now()
        msg.header.frame_id = "base_link"
        self.pub.publish(msg)

    def shutdown(self):
        self.publish_zero(rospy.Time.now())

    @staticmethod
    def arrow(marker_id, frame, stamp, start, end, color, scale=0.035):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = frame, stamp
        marker.ns, marker.id = "aquaflow_tracking_error", marker_id
        marker.type, marker.action = Marker.ARROW, Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale, 2.0 * scale, 3.0 * scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = [Point(*start), Point(*end)]
        return marker

    def publish_errors(self, now, position, target, yaw, target_yaw, ex, ey, eyaw):
        """Publish plot-friendly tracking errors and a compact RViz overlay."""
        self.error_x_pub.publish(Float64(data=ex))
        self.error_y_pub.publish(Float64(data=ey))
        self.error_yaw_pub.publish(Float64(data=eyaw))
        self.error_norm_pub.publish(Float64(data=math.hypot(ex, ey)))

        z = position.z + 0.12
        current = (position.x, position.y, z)
        desired = (target.position.x, target.position.y, z)
        arrows = MarkerArray()
        # Yellow: world-frame position error from current pose to the selected
        # local target. Blue/orange: actual and desired headings at the robot.
        arrows.markers.append(self.arrow(0, "world_ned", now, current, desired,
                                         (1.0, 0.85, 0.0, 0.95), 0.028))
        heading_length = 0.42
        arrows.markers.append(self.arrow(
            1, "world_ned", now, current,
            (position.x + heading_length * math.cos(yaw),
             position.y + heading_length * math.sin(yaw), z),
            (0.10, 0.55, 1.0, 0.95)))
        arrows.markers.append(self.arrow(
            2, "world_ned", now, current,
            (position.x + heading_length * math.cos(target_yaw),
             position.y + heading_length * math.sin(target_yaw), z),
            (1.0, 0.35, 0.05, 0.95)))
        text = Marker()
        text.header.frame_id, text.header.stamp = "world_ned", now
        text.ns, text.id = "aquaflow_tracking_error", 3
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y = position.x, position.y
        text.pose.position.z, text.pose.orientation.w = z + 0.22, 1.0
        text.scale.z = 0.16
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 0.95
        text.text = "ex=%+.2f m  ey=%+.2f m  yaw=%+.1f deg" % (ex, ey, math.degrees(eyaw))
        arrows.markers.append(text)
        self.error_marker_pub.publish(arrows)

    def publish_target_markers(self, now, position, yaw, nearest_idx, target):
        """Publish the nearest path point and lookahead target as RViz markers."""
        z = position.z + 0.12
        markers = MarkerArray()

        # Nearest point on path (cyan sphere)
        nearest_pose = self.path.poses[nearest_idx].pose
        nearest_marker = Marker()
        nearest_marker.header.frame_id = "world_ned"
        nearest_marker.header.stamp = now
        nearest_marker.ns = "aquaflow_tracking_target"
        nearest_marker.id = 0
        nearest_marker.type = Marker.SPHERE
        nearest_marker.action = Marker.ADD
        nearest_marker.pose.position.x = nearest_pose.position.x
        nearest_marker.pose.position.y = nearest_pose.position.y
        nearest_marker.pose.position.z = z
        nearest_marker.pose.orientation.w = 1.0
        nearest_marker.scale.x = nearest_marker.scale.y = nearest_marker.scale.z = 0.18
        nearest_marker.color.r, nearest_marker.color.g = 0.0, 0.85
        nearest_marker.color.b, nearest_marker.color.a = 1.0, 0.95
        markers.markers.append(nearest_marker)

        # Lookahead target point (magenta diamond)
        target_marker = Marker()
        target_marker.header.frame_id = "world_ned"
        target_marker.header.stamp = now
        target_marker.ns = "aquaflow_tracking_target"
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = target.position.x
        target_marker.pose.position.y = target.position.y
        target_marker.pose.position.z = z
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.22
        target_marker.color.r, target_marker.color.g = 1.0, 0.0
        target_marker.color.b, target_marker.color.a = 0.8, 0.95
        markers.markers.append(target_marker)

        # Line from robot to nearest point (cyan dashed)
        line1 = Marker()
        line1.header.frame_id = "world_ned"
        line1.header.stamp = now
        line1.ns = "aquaflow_tracking_target"
        line1.id = 2
        line1.type = Marker.LINE_STRIP
        line1.action = Marker.ADD
        line1.scale.x = 0.03
        line1.color.r, line1.color.g = 0.0, 0.85
        line1.color.b, line1.color.a = 1.0, 0.6
        line1.points = [Point(position.x, position.y, z),
                        Point(nearest_pose.position.x, nearest_pose.position.y, z)]
        markers.markers.append(line1)

        # Line from nearest point to lookahead target (magenta dashed)
        line2 = Marker()
        line2.header.frame_id = "world_ned"
        line2.header.stamp = now
        line2.ns = "aquaflow_tracking_target"
        line2.id = 3
        line2.type = Marker.LINE_STRIP
        line2.action = Marker.ADD
        line2.scale.x = 0.03
        line2.color.r, line2.color.g = 1.0, 0.0
        line2.color.b, line2.color.a = 0.8, 0.6
        line2.points = [Point(nearest_pose.position.x, nearest_pose.position.y, z),
                        Point(target.position.x, target.position.y, z)]
        markers.markers.append(line2)

        self.target_marker_pub.publish(markers)

    def update(self, _event):
        now = rospy.Time.now()
        if self.odom is None or self.path is None or not self.path.poses:
            self.publish_zero(now)
            return
        if (now - self.odom.header.stamp).to_sec() > self.odom_timeout:
            self.publish_zero(now)
            return
        if (now - self.path.header.stamp).to_sec() > self.path_timeout:
            self.publish_zero(now)
            return

        dt = 1.0 / 20.0 if self.last_time is None else max(1e-3, min(0.2, (now - self.last_time).to_sec()))
        self.last_time = now
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = yaw_from_quat(q)
        nearest = 0  # reference_processor already projects to nearest
        target = self.path.poses[min(len(self.path.poses) - 1, nearest + self.lookahead)].pose
        dx, dy = target.position.x - p.x, target.position.y - p.y
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        target_yaw = yaw_from_quat(target.orientation)
        eyaw = wrap(target_yaw - yaw)
        if abs(eyaw) < self.yaw_deadband:
            eyaw = 0.0
            self.int_yaw = 0.0
        self.publish_errors(now, p, target, yaw, target_yaw, ex, ey, eyaw)
        self.publish_target_markers(now, p, yaw, nearest, target)
        vx = self.odom.twist.twist.linear.x
        vy = self.odom.twist.twist.linear.y
        wz = self.odom.twist.twist.angular.z
        vz = self.odom.twist.twist.linear.z
        self.int_x = max(-1.0, min(1.0, self.int_x + ex * dt))
        self.int_y = max(-1.0, min(1.0, self.int_y + ey * dt))
        self.int_yaw = max(-1.0, min(1.0, self.int_yaw + eyaw * dt))
        ez = target.position.z - p.z
        self.int_z = max(-self.z_integral_limit,
                         min(self.z_integral_limit, self.int_z + ez * dt))
        fx = self.kp_xy * ex - self.kd_xy * vx + self.ki_xy * self.int_x
        fy = self.kp_xy * ey - self.kd_xy * vy + self.ki_xy * self.int_y
        nz = self.kp_yaw * eyaw - self.kd_yaw * wz + self.ki_yaw * self.int_yaw
        fz = self.kp_z * ez - self.kd_z * vz + self.ki_z * self.int_z
        msg = WrenchStamped()
        msg.header.stamp = now
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = max(-self.max_fx, min(self.max_fx, fx))
        msg.wrench.force.y = max(-self.max_fy, min(self.max_fy, fy))
        msg.wrench.torque.z = max(-self.max_nz, min(self.max_nz, nz))
        msg.wrench.force.z = max(-self.max_fz, min(self.max_fz, fz))
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("planar_pid_tracker")
    PlanarPIDTracker()
    rospy.spin()
