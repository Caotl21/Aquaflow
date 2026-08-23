#!/usr/bin/env python3
"""Arc-length parameterized reference processor for HOFA-MPC.

Subscribes to a planned Path from privileged_teacher, builds an internal
arc-length parameterized representation, projects the robot position onto
the curve with progress tracking, extracts a lookahead window, resamples
uniformly, and publishes:
  - /controller/reference_path (Path)           → PID
  - /controller/reference_trajectory (TrajectoryPoint) → MPC

All coordinates are NED (world_ned frame).
"""
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from hofa_mpc_ros.msg import TrajectoryPoint


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _lerp(a, b, t):
    return a + t * (b - a)


def _lerp_angle(a, b, t):
    return _wrap(a + t * _wrap(b - a))


class ReferenceProcessor:
    def __init__(self):
        rospy.init_node("reference_processor")

        # --- Parameters ---
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        ref_topic = rospy.get_param("~reference_topic",
                                    "/aquaflow/teacher_reference")
        self.lookahead_distance = float(
            rospy.get_param("~lookahead_distance_m", 1.5))
        self.n_resample = int(rospy.get_param("~n_resample_points", 20))
        self.max_speed = float(rospy.get_param("~max_speed", 0.35))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.5))
        self.s_backtrack_tol = float(
            rospy.get_param("~s_backtrack_tolerance_m", 0.5))
        rate = float(rospy.get_param("~rate", 20.0))
        odom_timeout = float(rospy.get_param("~odom_timeout_s", 0.25))
        path_timeout = float(rospy.get_param("~path_timeout_s", 2.0))

        # --- State ---
        self.robot_xy = None
        self.robot_stamp = None
        self.arc_path = None  # dict with x, y, z, yaw, speed, s arrays
        self.s_progress = 0.0
        self.prev_speed = 0.0
        self.path_stamp = None

        # --- Arc-length path from incoming Path ---
        self._raw_x = None
        self._raw_y = None
        self._raw_z = None
        self._raw_yaw = None

        # --- Subscribers ---
        rospy.Subscriber(ref_topic, Path, self._path_cb, queue_size=1)
        odom_topic = "/%s/odometry" % self.vehicle_name
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=1)

        # --- Publishers ---
        self.path_pub = rospy.Publisher(
            "/controller/reference_path", Path, queue_size=1)
        self.traj_pub = rospy.Publisher(
            "/controller/reference_trajectory", TrajectoryPoint, queue_size=1)

        # --- Timer ---
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._update)
        self.odom_timeout = odom_timeout
        self.path_timeout = path_timeout

        rospy.loginfo("Reference processor ready: ref_topic=%s, L_d=%.2f, "
                      "n_resample=%d, max_speed=%.2f",
                      ref_topic, self.lookahead_distance,
                      self.n_resample, self.max_speed)

    # --- Callbacks ---

    def _odom_cb(self, msg):
        self.robot_xy = (msg.pose.pose.position.x,
                         msg.pose.pose.position.y)
        self.robot_stamp = msg.header.stamp

    def _path_cb(self, msg):
        """Build arc-length parameterized representation from Path."""
        if len(msg.poses) < 2:
            return

        n = len(msg.poses)
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        for i, ps in enumerate(msg.poses):
            x[i] = ps.pose.position.x
            y[i] = ps.pose.position.y
            z[i] = ps.pose.position.z

        # Compute yaw from consecutive points (tangent direction)
        yaw = np.zeros(n)
        for i in range(n - 1):
            yaw[i] = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i])
        yaw[-1] = yaw[-2] if n >= 2 else 0.0

        # Smooth yaw with a simple moving average to reduce noise
        if n >= 3:
            yaw_smooth = yaw.copy()
            for i in range(1, n - 1):
                yaw_smooth[i] = _lerp_angle(
                    _lerp_angle(yaw[i - 1], yaw[i], 0.5),
                    _lerp_angle(yaw[i], yaw[i + 1], 0.5), 0.5)
            yaw_smooth[0] = yaw[0]
            yaw_smooth[-1] = yaw[-1]
            yaw = yaw_smooth

        # Cumulative arc length
        s = np.zeros(n)
        for i in range(1, n):
            ds = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            s[i] = s[i - 1] + ds
        total_length = s[-1]

        # Speed from curvature (kinematic limit, same as time_parameterize)
        speed = np.full(n, self.max_speed)
        for i in range(1, n - 1):
            ds = max(s[i + 1] - s[i], 1e-6)
            curvature = abs(_wrap(yaw[i + 1] - yaw[i])) / ds
            spd = min(self.max_speed,
                      self.max_yaw_rate / max(curvature, 1e-6))
            speed[i] = max(0.08, spd)
        speed[0] = speed[1] if n >= 2 else self.max_speed
        speed[-1] = 0.0  # stop at goal

        self.arc_path = {
            'x': x, 'y': y, 'z': z,
            'yaw': yaw, 'speed': speed, 's': s,
            'total_length': total_length,
        }

        # Reset progress if path changed significantly
        if self.s_progress > total_length:
            self.s_progress = 0.0

        self.path_stamp = msg.header.stamp
        rospy.loginfo("Path received: %d points, length=%.2f m",
                      n, total_length)

    def _update(self, _event):
        now = rospy.Time.now()

        # Check inputs
        if self.robot_xy is None:
            return
        if (now - self.robot_stamp).to_sec() > self.odom_timeout:
            return
        if self.arc_path is None:
            return
        if self.path_stamp is not None:
            if (now - self.path_stamp).to_sec() > self.path_timeout:
                rospy.logwarn_throttle(5.0, "Reference path stale, holding last")
                return

        # ① Project robot position onto arc-length curve
        s_proj = self._project_to_curve(self.robot_xy[0], self.robot_xy[1])

        # ② Clamp: allow small backtrack, enforce monotonic advance
        s_proj = max(s_proj, self.s_progress - self.s_backtrack_tol)
        s_proj = max(s_proj, self.s_progress)  # monotonic in normal case
        s_proj = min(s_proj, self.arc_path['total_length'])
        self.s_progress = s_proj

        # ③ Lookahead window [s_proj, s_proj + L_d]
        s_start = s_proj
        s_end = min(s_proj + self.lookahead_distance,
                    self.arc_path['total_length'])

        # If too close to end, shift window back slightly
        if s_end - s_start < 0.1 and self.arc_path['total_length'] > 0.1:
            s_start = max(0.0, s_end - self.lookahead_distance)

        # ④ Resample window into N uniform points
        local = self._resample_window(s_start, s_end, self.n_resample)
        if not local:
            return

        # ⑤ Publish
        self._publish_path(local, now)
        self._publish_trajectory(local, now)

    # --- Arc-length projection ---

    def _project_to_curve(self, rx, ry):
        """Project robot position onto the arc-length curve, return s value."""
        x = self.arc_path['x']
        y = self.arc_path['y']
        s = self.arc_path['s']
        n = len(x)

        min_dist = float('inf')
        best_s = self.s_progress

        for i in range(n - 1):
            # Project point onto line segment [P_i, P_{i+1}]
            dx_seg = x[i + 1] - x[i]
            dy_seg = y[i + 1] - y[i]
            seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
            if seg_len_sq < 1e-12:
                continue

            t = ((rx - x[i]) * dx_seg + (ry - y[i]) * dy_seg) / seg_len_sq
            t = max(0.0, min(1.0, t))

            proj_x = x[i] + t * dx_seg
            proj_y = y[i] + t * dy_seg
            dist = math.hypot(rx - proj_x, ry - proj_y)

            if dist < min_dist:
                min_dist = dist
                best_s = s[i] + t * (s[i + 1] - s[i])

        return best_s

    # --- Window resampling ---

    def _resample_window(self, s_start, s_end, n_points):
        """Resample n_points uniformly in [s_start, s_end] arc-length window."""
        s_arr = self.arc_path['s']
        x = self.arc_path['x']
        y = self.arc_path['y']
        z = self.arc_path['z']
        yaw = self.arc_path['yaw']
        speed = self.arc_path['speed']

        s_values = np.linspace(s_start, s_end, n_points)
        points = []

        for sv in s_values:
            # Find segment index
            idx = np.searchsorted(s_arr, sv) - 1
            idx = max(0, min(idx, len(s_arr) - 2))

            seg_len = s_arr[idx + 1] - s_arr[idx]
            if seg_len < 1e-12:
                t = 0.0
            else:
                t = np.clip((sv - s_arr[idx]) / seg_len, 0.0, 1.0)

            px = _lerp(x[idx], x[idx + 1], t)
            py = _lerp(y[idx], y[idx + 1], t)
            pz = _lerp(z[idx], z[idx + 1], t)
            pyaw = _lerp_angle(yaw[idx], yaw[idx + 1], t)
            pspeed = _lerp(speed[idx], speed[idx + 1], t)

            # World-frame velocity from speed and heading
            dx = pspeed * math.cos(pyaw)
            dy = pspeed * math.sin(pyaw)

            points.append({
                'x': px, 'y': py, 'z': pz,
                'yaw': pyaw, 'speed': pspeed,
                'dx': dx, 'dy': dy,
            })

        return points

    # --- Publishers ---

    def _yaw_to_quat(self, yaw):
        q = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def _publish_path(self, local, now):
        """Publish reference path for PID (NED)."""
        msg = Path()
        msg.header.stamp = now
        msg.header.frame_id = "world_ned"

        for pt in local:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = pt['x']
            ps.pose.position.y = pt['y']
            ps.pose.position.z = pt['z']
            ps.pose.orientation = self._yaw_to_quat(pt['yaw'])
            msg.poses.append(ps)

        self.path_pub.publish(msg)

    def _publish_trajectory(self, local, now):
        """Publish reference trajectory for MPC (NED).

        The first point is the current tracking target with its velocity.
        We also compute acceleration from the velocity difference between
        the first two points.
        """
        if len(local) < 2:
            return

        ref = local[0]
        ref_next = local[1]

        # Estimate dt from speed and distance to next point
        ds = math.hypot(ref_next['x'] - ref['x'],
                        ref_next['y'] - ref['y'])
        avg_speed = max(0.08, 0.5 * (ref['speed'] + ref_next['speed']))
        dt = ds / avg_speed if avg_speed > 1e-3 else 0.1

        # Acceleration: (v_next - v_current) / dt
        ddx = (ref_next['dx'] - ref['dx']) / dt if dt > 1e-3 else 0.0
        ddy = (ref_next['dy'] - ref['dy']) / dt if dt > 1e-3 else 0.0
        # Yaw acceleration from yaw rate difference
        dpsi_cur = ref['speed'] * 0.0  # placeholder: no explicit yaw rate
        dpsi_nxt = ref_next['speed'] * 0.0
        ddpsi = 0.0

        msg = TrajectoryPoint()
        msg.header.stamp = now
        msg.header.frame_id = "world_ned"
        msg.pose.position.x = ref['x']
        msg.pose.position.y = ref['y']
        msg.pose.position.z = ref['z']
        msg.pose.orientation = self._yaw_to_quat(ref['yaw'])
        msg.twist.linear.x = ref['dx']
        msg.twist.linear.y = ref['dy']
        msg.twist.linear.z = 0.0
        msg.twist.angular.z = ref['speed'] * math.tan(0.0)  # placeholder
        msg.accel.linear.x = ddx
        msg.accel.linear.y = ddy
        msg.accel.linear.z = 0.0
        msg.accel.angular.z = ddpsi
        msg.valid = True
        msg.trajectory_id = "arc_length"

        self.traj_pub.publish(msg)


if __name__ == "__main__":
    try:
        ReferenceProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
