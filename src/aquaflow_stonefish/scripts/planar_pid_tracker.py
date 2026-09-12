#!/usr/bin/env python3
"""Decoupled planar position/velocity PID trajectory tracker.

The preferred reference is the same time-parameterized window the MPC
consumes (``/controller/reference_trajectory_window``): point i of that
window is the pose/twist/accel the vehicle should hold at t + i*dt along the
speed-profiled route, so a lookahead is a genuine time offset instead of the
arc-length offset the plain Path interface could express.  The Path plus
TwistStamped inputs are kept as a fallback for the pure path-tracking
baseline in tracking.launch.

The outer loop maps pose error to a desired body velocity; the inner loop
maps velocity error to generalized force, around a model feedforward that
cancels the nominal hydrodynamic drag at the commanded velocity.
"""
import math
import rospy
from geometry_msgs.msg import Point, TwistStamped, WrenchStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from hofa_mpc_ros.msg import TrajectoryPointWindow
from hofa_mpc_ros.vehicle_params import resolve_inertia, describe_inertia


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PlanarPIDTracker:
    def __init__(self):
        self.odom = None
        self.path = None
        self.window = None
        self.window_stamp = None
        self.velocity_reference = None
        self.velocity_reference_stamp = None
        self.last_time = None
        self.int_pos_x = self.int_pos_y = self.int_pos_yaw = 0.0
        self.int_vel_x = self.int_vel_y = self.int_vel_yaw = 0.0
        self.int_z = 0.0
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.25))
        self.path_timeout = float(rospy.get_param("~path_timeout", 0.50))
        self.lookahead = max(0, int(rospy.get_param("~lookahead_index", 0)))
        self.control_mode = rospy.get_param("~control_mode", "dual_loop")
        # Trajectory-window interface.  The window carries no dt field, so the
        # sample interval is taken from the publisher's own parameter rather
        # than duplicated as a literal that could silently drift out of sync.
        self.trajectory_dt = max(1e-3, float(rospy.get_param(
            "~trajectory_dt_s",
            rospy.get_param("/reference_processor/mpc_dt_s", 0.05))))
        # 0.0 by default: with reference_processor in "schedule" mode, window
        # point 0 is already where the vehicle is supposed to be *now*, so any
        # lookahead adds a standing along-track error of v*lookahead that the
        # outer loop reads as "go faster".  At 0.4 s and 0.2 m/s that bias was
        # enough to run the vehicle 16% over the speed limit for a whole run.
        # Non-zero values trade that steady bias for phase lead.
        self.lookahead_time = max(0.0, float(
            rospy.get_param("~lookahead_time_s", 0.0)))
        self.trajectory_timeout = float(
            rospy.get_param("~trajectory_timeout", 0.30))
        self.kp_pos_xy = float(rospy.get_param("~kp_pos_xy", 2.0))
        self.ki_pos_xy = float(rospy.get_param("~ki_pos_xy", 0.0))
        self.kp_pos_yaw = float(rospy.get_param("~kp_pos_yaw", 1.2))
        self.ki_pos_yaw = float(rospy.get_param("~ki_pos_yaw", 0.0))
        # Surge and sway see very different drag (drag_linear is [54, 10, 1.4]),
        # so a single shared inner gain cannot suit both.  The legacy combined
        # parameters stay as the per-axis defaults for older configs.
        kp_vel_xy = float(rospy.get_param("~kp_vel_xy", 25.0))
        ki_vel_xy = float(rospy.get_param("~ki_vel_xy", 15.0))
        self.kp_vel_x = float(rospy.get_param("~kp_vel_x", kp_vel_xy))
        self.kp_vel_y = float(rospy.get_param("~kp_vel_y", 20.0))
        self.ki_vel_x = float(rospy.get_param("~ki_vel_x", ki_vel_xy))
        self.ki_vel_y = float(rospy.get_param("~ki_vel_y", ki_vel_xy))
        self.kp_vel_yaw = float(rospy.get_param("~kp_vel_yaw", 2.5))
        self.ki_vel_yaw = float(rospy.get_param("~ki_vel_yaw", 1.0))
        # Fraction of each axis' force limit the integrator alone may command.
        # Bounding the integrator by its force contribution instead of by
        # accumulated error-seconds is what keeps it able to cancel a steady
        # drag offset; the old +/-1.0 error-second clamp capped its authority
        # at ki newtons, which saturated within seconds and then did nothing.
        self.integral_authority = self.clamp(
            float(rospy.get_param("~integral_authority", 0.6)), 0.0, 1.0)
        self.max_cmd_vx = float(rospy.get_param("~max_cmd_vx", 0.45))
        self.max_cmd_vy = float(rospy.get_param("~max_cmd_vy", 0.45))
        self.max_cmd_wz = float(rospy.get_param("~max_cmd_wz", 0.6))
        self.max_reference_speed = float(rospy.get_param("~max_reference_speed", 0.25))
        self.velocity_timeout = float(rospy.get_param("~velocity_timeout", 0.5))
        self.require_velocity_reference = bool(
            rospy.get_param("~require_velocity_reference", False))
        self.kp_xy = float(rospy.get_param("~kp_xy", 15.0))
        self.kd_xy = float(rospy.get_param("~kd_xy", 3.0))
        self.ki_xy = float(rospy.get_param("~ki_xy", 0.0))
        self.kp_yaw = float(rospy.get_param("~kp_yaw", 4.0))
        self.kd_yaw = float(rospy.get_param("~kd_yaw", 1.0))
        self.ki_yaw = float(rospy.get_param("~ki_yaw", 0.3))
        self.yaw_deadband = math.radians(max(0.0, float(rospy.get_param("~yaw_deadband_deg", 1.0))))
        self.kp_z = float(rospy.get_param("~kp_z", 8.0))
        self.kd_z = float(rospy.get_param("~kd_z", 4.0))
        self.ki_z = float(rospy.get_param("~ki_z", 0.5))
        self.z_integral_limit = max(0.0, float(rospy.get_param("~z_integral_limit", 5.0)))
        # The four 45-degree thrusters deliver roughly 179 N of surge and
        # 46 N.m of yaw torque at the rated limits, and holding the 0.2 m/s
        # reference speed already costs ~11 N against drag_linear[0]=54.  The
        # previous 4 N / 1.4 N.m ceilings capped the vehicle near 0.085 m/s,
        # so the speed reference was physically unreachable no matter the gains.
        self.max_fx = float(rospy.get_param("~max_fx", 30.0))
        self.max_fy = float(rospy.get_param("~max_fy", 30.0))
        self.max_nz = float(rospy.get_param("~max_nz", 5.0))
        self.max_fz = float(rospy.get_param("~max_fz", 20.0))
        # Nominal plant used for the inner-loop feedforward.  Keyed under
        # ~vehicle so config/vehicle.yaml can be rosparam-loaded verbatim.
        vehicle = rospy.get_param("~vehicle", {})
        self.drag_linear = [float(v) for v in
                            vehicle.get("drag_linear", [54.0, 10.0, 1.4])]
        self.drag_quadratic = [float(v) for v in
                               vehicle.get("drag_quadratic", [2.0, 15.0, 0.35])]
        # Effective inertia (rigid body + added mass) -- the feedforward has to
        # supply the force that actually accelerates the vehicle in water.
        mass_matrix, inertia_info = resolve_inertia(vehicle)
        self.mass = [float(mass_matrix[0, 0]), float(mass_matrix[1, 1]),
                     float(mass_matrix[2, 2])]
        rospy.loginfo(describe_inertia(inertia_info))
        self.feedforward_enabled = bool(
            rospy.get_param("~feedforward_enabled", True))
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.odom_topic = "/%s/odometry" % self.vehicle_name
        self.reference_topic = rospy.get_param("~reference_topic", "/aquaflow/nominal_path")
        self.velocity_reference_topic = rospy.get_param(
            "~velocity_reference_topic", "/controller/reference_velocity")
        self.trajectory_window_topic = rospy.get_param(
            "~trajectory_window_topic", "/controller/reference_trajectory_window")
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
        rospy.Subscriber(self.velocity_reference_topic, TwistStamped,
                         self.velocity_cb, queue_size=1)
        rospy.Subscriber(self.trajectory_window_topic, TrajectoryPointWindow,
                         self.window_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~rate", 20.0))), self.update)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "planar_pid_tracker ready: mode=%s, window=%s (dt=%.3f s, "
            "lookahead=%.2f s -> index %d), limits fx=%.1f N nz=%.1f N.m, "
            "feedforward=%s",
            self.control_mode, self.trajectory_window_topic,
            self.trajectory_dt, self.lookahead_time, self.lookahead_index(),
            self.max_fx, self.max_nz, self.feedforward_enabled)

    def odom_cb(self, msg):
        self.odom = msg

    def path_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != "world_ned":
            rospy.logwarn_throttle(5.0, "reference path frame must be world_ned, got %s", msg.header.frame_id)
        self.path = msg

    def velocity_cb(self, msg):
        self.velocity_reference = msg.twist
        self.velocity_reference_stamp = msg.header.stamp

    def window_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != "world_ned":
            rospy.logwarn_throttle(
                5.0, "reference trajectory window frame must be world_ned, got %s",
                msg.header.frame_id)
        self.window = msg
        self.window_stamp = msg.header.stamp

    def lookahead_index(self):
        """Window index matching the configured lookahead time."""
        return int(round(self.lookahead_time / self.trajectory_dt))

    def publish_zero(self, stamp=None):
        msg = WrenchStamped()
        msg.header.stamp = stamp or rospy.Time.now()
        msg.header.frame_id = "base_link"
        self.pub.publish(msg)

    def shutdown(self):
        self.publish_zero(rospy.Time.now())

    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def reset_integrators(self):
        self.int_pos_x = self.int_pos_y = self.int_pos_yaw = 0.0
        self.int_vel_x = self.int_vel_y = self.int_vel_yaw = 0.0
        self.int_z = 0.0

    def fallback_velocity_world(self, target_index):
        if self.path is None or len(self.path.poses) < 2:
            return 0.0, 0.0, 0.0
        i0 = max(0, min(len(self.path.poses) - 2, target_index))
        p0 = self.path.poses[i0].pose.position
        p1 = self.path.poses[i0 + 1].pose.position
        dx, dy = p1.x - p0.x, p1.y - p0.y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 0.0, 0.0, 0.0
        speed = min(self.max_reference_speed, length * 20.0)
        return speed * dx / length, speed * dy / length, 0.0

    def reference_velocity_world(self, target_index, now):
        if self.velocity_reference is not None and self.velocity_reference_stamp is not None:
            age = (now - self.velocity_reference_stamp).to_sec()
            if 0.0 <= age <= self.velocity_timeout:
                return (self.velocity_reference.linear.x,
                        self.velocity_reference.linear.y,
                        self.velocity_reference.angular.z)
        if self.require_velocity_reference:
            return None
        return self.fallback_velocity_world(target_index)

    def target_from_window(self, now):
        """Pick the time-parameterized lookahead point from the MPC window.

        Point i of the window is the reference state at t + i*trajectory_dt
        along the speed-profiled route, so the lookahead is a time offset and
        stays consistent between straight and curved sections instead of
        varying with the point spacing the way an index into a Path does.
        """
        window, stamp = self.window, self.window_stamp
        if window is None or stamp is None or not window.valid or not window.points:
            return None
        age = (now - stamp).to_sec()
        if age < 0.0 or age > self.trajectory_timeout:
            return None
        index = self.clamp(self.lookahead_index(), 0, len(window.points) - 1)
        point = window.points[index]
        if not point.valid:
            return None
        return (point.pose,
                (point.twist.linear.x, point.twist.linear.y,
                 point.twist.angular.z),
                (point.accel.linear.x, point.accel.linear.y,
                 point.accel.angular.z))

    def target_from_path(self, now):
        """Fallback to the spatially uniform Path plus TwistStamped inputs."""
        if self.path is None or not self.path.poses:
            return None
        if (now - self.path.header.stamp).to_sec() > self.path_timeout:
            return None
        nearest = 0  # reference_processor already projects to nearest
        index = min(len(self.path.poses) - 1, nearest + self.lookahead)
        reference_velocity = self.reference_velocity_world(index, now)
        if reference_velocity is None:
            return None
        # The Path interface carries no acceleration, so the feedforward loses
        # its inertial term and keeps only the drag part.
        return self.path.poses[index].pose, reference_velocity, (0.0, 0.0, 0.0)

    def pi_axis(self, key, error, feedforward, kp, ki, limit, dt):
        """One PI axis with feedforward and conditional-integration anti-windup.

        The integrator is bounded by the force it may contribute rather than by
        accumulated error-seconds, and it stops accumulating once the output is
        railed and the error would only push it further out of range.
        """
        integral = getattr(self, key)
        if ki <= 1e-9:
            # No integral action configured; accumulating would grow a state
            # that is never read.  ki_pos_xy defaults to 0, so this is the
            # normal path for the outer loop.
            setattr(self, key, 0.0)
            return self.clamp(feedforward + kp * error, -limit, limit)
        bound = self.integral_authority * limit / ki
        candidate = self.clamp(integral + error * dt, -bound, bound)
        command = feedforward + kp * error + ki * candidate
        if abs(command) > limit and command * error > 0.0:
            command = feedforward + kp * error + ki * integral
        else:
            integral = candidate
        setattr(self, key, integral)
        return self.clamp(command, -limit, limit)

    def drag_feedforward(self, axis, velocity):
        """Nominal force needed to hold ``velocity`` against modelled drag."""
        if not self.feedforward_enabled:
            return 0.0
        return (self.drag_linear[axis] * velocity
                + self.drag_quadratic[axis] * velocity * abs(velocity))

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

        # Nearest point on path (cyan sphere).  When tracking the trajectory
        # window there may be no Path at all, so fall back to the target.
        if self.path is not None and self.path.poses:
            nearest_pose = self.path.poses[
                min(nearest_idx, len(self.path.poses) - 1)].pose
        else:
            nearest_pose = target
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
        if self.odom is None:
            self.reset_integrators()
            self.publish_zero(now)
            return
        if (now - self.odom.header.stamp).to_sec() > self.odom_timeout:
            self.reset_integrators()
            self.publish_zero(now)
            return

        # Prefer the time-parameterized window shared with the MPC; fall back
        # to the Path interface so the pure path-tracking baseline still runs.
        reference = self.target_from_window(now)
        if reference is None:
            reference = self.target_from_path(now)
            # Only a window that was live and went stale is worth warning
            # about.  tracking.launch never publishes one, and there the Path
            # interface is the intended configuration, not a degraded mode.
            if reference is not None and self.window is not None:
                rospy.logwarn_throttle(
                    5.0, "trajectory window stale, falling back to Path %s",
                    self.reference_topic)
        if reference is None:
            self.reset_integrators()
            self.publish_zero(now)
            return
        target, reference_velocity, reference_accel = reference

        dt = 1.0 / 20.0 if self.last_time is None else max(1e-3, min(0.2, (now - self.last_time).to_sec()))
        self.last_time = now
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = yaw_from_quat(q)
        dx, dy = target.position.x - p.x, target.position.y - p.y
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        target_yaw = yaw_from_quat(target.orientation)
        eyaw = wrap(target_yaw - yaw)
        if abs(eyaw) < self.yaw_deadband:
            eyaw = 0.0
            self.int_pos_yaw = 0.0
        self.publish_errors(now, p, target, yaw, target_yaw, ex, ey, eyaw)
        self.publish_target_markers(now, p, yaw, 0, target)
        vx = self.odom.twist.twist.linear.x
        vy = self.odom.twist.twist.linear.y
        wz = self.odom.twist.twist.angular.z
        vz = self.odom.twist.twist.linear.z
        ez = target.position.z - p.z
        self.int_z = max(-self.z_integral_limit,
                         min(self.z_integral_limit, self.int_z + ez * dt))

        if self.control_mode == "legacy":
            self.int_pos_x = max(-1.0, min(1.0, self.int_pos_x + ex * dt))
            self.int_pos_y = max(-1.0, min(1.0, self.int_pos_y + ey * dt))
            self.int_pos_yaw = max(-1.0, min(1.0, self.int_pos_yaw + eyaw * dt))
            fx = self.kp_xy * ex - self.kd_xy * vx + self.ki_xy * self.int_pos_x
            fy = self.kp_xy * ey - self.kd_xy * vy + self.ki_xy * self.int_pos_y
            nz = self.kp_yaw * eyaw - self.kd_yaw * wz + self.ki_yaw * self.int_pos_yaw
        else:
            ref_vx_world, ref_vy_world, ref_wz = reference_velocity
            ref_ax_world, ref_ay_world, ref_awz = reference_accel
            # Convert the world-frame reference derivatives to the same body
            # frame used by Stonefish odometry and by the generalized-force
            # output.  The centripetal correction to the body acceleration is
            # dropped: at these speeds and yaw rates it is well under 0.05 N.
            ref_vx = c * ref_vx_world + s * ref_vy_world
            ref_vy = -s * ref_vx_world + c * ref_vy_world
            ref_ax = c * ref_ax_world + s * ref_ay_world
            ref_ay = -s * ref_ax_world + c * ref_ay_world

            # Outer loop: pose error -> commanded body velocity, about the
            # reference velocity carried by the trajectory point.
            cmd_vx = self.pi_axis("int_pos_x", ex, ref_vx, self.kp_pos_xy,
                                  self.ki_pos_xy, self.max_cmd_vx, dt)
            cmd_vy = self.pi_axis("int_pos_y", ey, ref_vy, self.kp_pos_xy,
                                  self.ki_pos_xy, self.max_cmd_vy, dt)
            cmd_wz = self.pi_axis("int_pos_yaw", eyaw, ref_wz, self.kp_pos_yaw,
                                  self.ki_pos_yaw, self.max_cmd_wz, dt)

            # Inner loop: velocity error -> force, about a model feedforward
            # that already cancels the drag at the commanded velocity and
            # supplies the inertial term for the reference acceleration.  Pure
            # feedback cannot do this job here: with drag_linear[0]=54 the
            # closed-loop DC gain kp/(kp+drag) leaves a large steady offset.
            vel_ex, vel_ey, vel_ewz = cmd_vx - vx, cmd_vy - vy, cmd_wz - wz
            ff_x = self.drag_feedforward(0, cmd_vx)
            ff_y = self.drag_feedforward(1, cmd_vy)
            ff_wz = self.drag_feedforward(2, cmd_wz)
            if self.feedforward_enabled:
                ff_x += self.mass[0] * ref_ax
                ff_y += self.mass[1] * ref_ay
                ff_wz += self.mass[2] * ref_awz
            fx = self.pi_axis("int_vel_x", vel_ex, ff_x, self.kp_vel_x,
                              self.ki_vel_x, self.max_fx, dt)
            fy = self.pi_axis("int_vel_y", vel_ey, ff_y, self.kp_vel_y,
                              self.ki_vel_y, self.max_fy, dt)
            nz = self.pi_axis("int_vel_yaw", vel_ewz, ff_wz, self.kp_vel_yaw,
                              self.ki_vel_yaw, self.max_nz, dt)
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
