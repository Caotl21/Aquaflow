#!/usr/bin/env python3
"""Arc-length parameterized reference processor for HOFA-MPC.

Subscribes to a planned Path from privileged_teacher, builds an internal
arc-length parameterized representation, advances an absolute schedule along
it, extracts a lookahead window, resamples uniformly, and publishes:
  - /controller/reference_path (Path)           → PID
  - /controller/reference_trajectory (TrajectoryPoint) → MPC

The window anchor is the scheduled arc position s_desired(t), obtained by
integrating the speed profile in real time.  Anchoring on the robot's own
projection instead (``progress_mode: projection``) makes the reference
re-derive itself from the vehicle every cycle, so no controller downstream
can observe that it is ahead of or behind plan -- the along-track error is
identically zero by construction.

All coordinates are NED (world_ned frame).
"""
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header, Float64
from hofa_mpc_ros.msg import TrajectoryPoint, TrajectoryPointWindow
from hofa_mpc_ros.arc_profile import build_arc_profile


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
        self.mpc_horizon_points = max(
            2, int(rospy.get_param("~mpc_horizon_points", 20)))
        self.mpc_dt = max(1e-3, float(rospy.get_param("~mpc_dt_s", 0.1)))
        self.max_speed = float(rospy.get_param("~max_speed", 0.35))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.5))
        self.max_yaw_accel = max(
            1e-3, float(rospy.get_param("~max_yaw_accel_radps2", 0.25)))
        self.yaw_accel_reserve_ratio = max(0.0, float(
            rospy.get_param("~yaw_accel_reserve_ratio", 0.2)))
        self.yaw_smoothing_window = max(
            3, int(rospy.get_param("~yaw_smoothing_window", 5)))
        if self.yaw_smoothing_window % 2 == 0:
            self.yaw_smoothing_window += 1
        self.max_accel = max(1e-3, float(rospy.get_param("~max_accel_mps2", 0.12)))
        self.max_decel = max(1e-3, float(rospy.get_param("~max_decel_mps2", 0.18)))
        self.s_backtrack_tol = float(
            rospy.get_param("~s_backtrack_tolerance_m", 0.5))
        self.progress_mode = str(
            rospy.get_param("~progress_mode", "schedule")).lower()
        if self.progress_mode not in ("schedule", "projection"):
            rospy.logwarn("unknown progress_mode '%s', using 'schedule'",
                          self.progress_mode)
            self.progress_mode = "schedule"
        # How far the schedule may run ahead of where the vehicle actually is.
        # Without this a stalled or blocked vehicle lets the reference escape
        # to the end of the route, which the controllers would chase as an
        # unbounded error.
        self.max_schedule_lead = max(0.0, float(
            rospy.get_param("~max_schedule_lead_m", 1.0)))
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
        self.path_signature = None
        # Absolute schedule state.  ``schedule_t0`` is the ROS time at which
        # the current route's profile starts; it is re-initialized whenever a
        # new route arrives and rolled back whenever the lead clamp bites.
        self.schedule_t0 = None
        self.s_schedule = 0.0

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
        self.velocity_pub = rospy.Publisher(
            "/controller/reference_velocity", TwistStamped, queue_size=1)
        self.traj_pub = rospy.Publisher(
            "/controller/reference_trajectory", TrajectoryPoint, queue_size=1)
        self.traj_window_pub = rospy.Publisher(
            "/controller/reference_trajectory_window", TrajectoryPointWindow, queue_size=1)
        self.schedule_lag_pub = rospy.Publisher(
            "/controller/schedule_lag_s", Float64, queue_size=1)

        # --- Timer ---
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._update)
        self.odom_timeout = odom_timeout
        self.path_timeout = path_timeout

        rospy.loginfo("Reference processor ready: ref_topic=%s, L_d=%.2f, "
                      "n_resample=%d, max_speed=%.2f, progress_mode=%s, "
                      "max_lead=%.2f m",
                      ref_topic, self.lookahead_distance,
                      self.n_resample, self.max_speed, self.progress_mode,
                      self.max_schedule_lead)

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

        # The profile is shared with the offline scorer so that evaluation
        # measures the same plan the controller is handed; see arc_profile.py.
        profile = build_arc_profile(
            x, y, z,
            max_speed=self.max_speed,
            max_yaw_rate=self.max_yaw_rate,
            max_yaw_accel=self.max_yaw_accel,
            max_accel=self.max_accel,
            max_decel=self.max_decel,
            yaw_smoothing_window=self.yaw_smoothing_window,
            yaw_accel_reserve_ratio=self.yaw_accel_reserve_ratio,
            # None, not 0.0: with no speed carried over from a previous plan
            # the start is unconstrained, matching the pre-refactor behaviour.
            initial_speed=self.prev_speed if self.prev_speed > 0.0 else None)
        self.prev_speed = float(profile['speed'][0])

        signature = (float(x[0]), float(y[0]), float(x[-1]), float(y[-1]), int(n))
        if self.path_signature is not None and signature != self.path_signature:
            self.s_progress = 0.0
            self.prev_speed = 0.0
            # A replanned route starts where the vehicle is now, so its
            # schedule starts now too; any lag against the old route is not
            # carried across.
            self.schedule_t0 = None
            self.s_schedule = 0.0
        self.path_signature = signature
        self.arc_path = profile

        if self.s_progress > profile['total_length']:
            self.s_progress = 0.0

        # This assignment used to sit after a ``return`` in another method, so
        # path_stamp stayed None and the staleness guard in _update never
        # fired.  privileged_teacher republishes at 5 Hz, well inside
        # path_timeout_s, so the guard now only trips if the teacher dies.
        self.path_stamp = msg.header.stamp
        rospy.loginfo_throttle(
            5.0, "Path received: %d points, length=%.2f m, planned %.1f s",
            n, profile['total_length'], profile['time'][-1])

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
        # Permit bounded backtracking after a disturbance, but prevent a
        # projection jump to an earlier route branch.
        s_proj = max(s_proj, self.s_progress - self.s_backtrack_tol)
        s_proj = min(s_proj, self.arc_path['total_length'])
        self.s_progress = s_proj

        # ③ Anchor the window on the absolute schedule, not on the vehicle.
        s_start = self._schedule_anchor(now, s_proj)
        s_end = min(s_start + self.lookahead_distance,
                    self.arc_path['total_length'])

        # If too close to end, shift window back slightly
        if s_end - s_start < 0.1 and self.arc_path['total_length'] > 0.1:
            s_start = max(0.0, s_end - self.lookahead_distance)

        # ④ Resample window into N uniform points
        # Keep the spatially uniform path for PID and visualization.  MPC gets
        # a separate time-uniform window so its i-th point matches its fixed
        # prediction interval dt.
        local_spatial = self._resample_window(s_start, s_end, self.n_resample)
        local_temporal = self._resample_time_window(
            s_start, self.mpc_horizon_points, self.mpc_dt)
        if not local_spatial or not local_temporal:
            return

        # ⑤ Publish
        self._publish_path(local_spatial, now)
        self._publish_trajectory(local_temporal, now, sample_dt=self.mpc_dt)

    # --- Absolute schedule ---

    def _schedule_anchor(self, now, s_proj):
        """Arc position the vehicle is supposed to occupy at ``now``.

        In ``projection`` mode this degenerates to the vehicle's own
        projection, which is the legacy behaviour: the reference follows the
        vehicle, so the along-track error it reports is always zero.
        """
        if self.progress_mode == "projection":
            self.s_schedule = s_proj
            self._publish_schedule_lag(0.0)
            return s_proj

        s_arr = self.arc_path['s']
        t_arr = self.arc_path['time']
        if self.schedule_t0 is None:
            self.schedule_t0 = now
        elapsed = (now - self.schedule_t0).to_sec()
        s_sched = float(np.interp(elapsed, t_arr, s_arr))

        # Clamp the lead, and roll the schedule clock back to match.  Without
        # the rollback the clock would keep accruing an invisible debt and the
        # reference would jump forward the moment the vehicle caught up.
        if s_sched > s_proj + self.max_schedule_lead:
            s_sched = min(s_proj + self.max_schedule_lead, s_arr[-1])
            self.schedule_t0 = now - rospy.Duration(
                float(np.interp(s_sched, s_arr, t_arr)))
            elapsed = (now - self.schedule_t0).to_sec()

        self.s_schedule = s_sched
        # Negative means the vehicle is behind where the plan wants it.
        self._publish_schedule_lag(
            float(np.interp(s_proj, s_arr, t_arr)) - elapsed)
        return s_sched

    def _publish_schedule_lag(self, lag_s):
        self.schedule_lag_pub.publish(Float64(data=float(lag_s)))

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

    def _sample_at_s(self, sv):
        """Interpolate one path sample at arc length ``sv``."""
        s_arr = self.arc_path['s']
        x = self.arc_path['x']
        y = self.arc_path['y']
        z = self.arc_path['z']
        yaw = self.arc_path['yaw']
        speed = self.arc_path['speed']
        curvature = self.arc_path['curvature']

        sv = float(np.clip(sv, s_arr[0], s_arr[-1]))
        idx = np.searchsorted(s_arr, sv) - 1
        idx = max(0, min(idx, len(s_arr) - 2))
        seg_len = s_arr[idx + 1] - s_arr[idx]
        t = 0.0 if seg_len < 1e-12 else np.clip(
            (sv - s_arr[idx]) / seg_len, 0.0, 1.0)
        pyaw = _lerp_angle(yaw[idx], yaw[idx + 1], t)
        pspeed = _lerp(speed[idx], speed[idx + 1], t)
        return {
            'x': _lerp(x[idx], x[idx + 1], t),
            'y': _lerp(y[idx], y[idx + 1], t),
            'z': _lerp(z[idx], z[idx + 1], t),
            'yaw': pyaw,
            'speed': pspeed,
            'dx': pspeed * math.cos(pyaw),
            'dy': pspeed * math.sin(pyaw),
            'curvature': _lerp(curvature[idx], curvature[idx + 1], t),
        }

    def _resample_window(self, s_start, s_end, n_points):
        """Resample n_points uniformly in [s_start, s_end] arc-length window."""
        s_arr = self.arc_path['s']
        x = self.arc_path['x']
        y = self.arc_path['y']
        z = self.arc_path['z']
        yaw = self.arc_path['yaw']
        speed = self.arc_path['speed']
        curvature = self.arc_path['curvature']

        s_values = np.linspace(s_start, s_end, n_points)
        return [self._sample_at_s(sv) for sv in s_values]

    def _resample_time_window(self, s_start, n_points, dt):
        """Sample the path at fixed time intervals for the MPC horizon."""
        if self.arc_path is None or len(self.arc_path['s']) < 2:
            return []
        s_arr = self.arc_path['s']
        t_arr = self.arc_path['time']
        t_start = float(np.interp(s_start, s_arr, t_arr))
        t_values = t_start + np.arange(n_points, dtype=float) * float(dt)
        s_values = np.interp(t_values, t_arr, s_arr)
        return [self._sample_at_s(sv) for sv in s_values]

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

    def _publish_trajectory(self, local, now, sample_dt=None):
        """Publish reference trajectory for MPC (NED).

        The first point is the current tracking target with its velocity.
        We also compute acceleration from the velocity difference between
        the first two points.
        """
        if len(local) < 2:
            return
        points = []
        for i, ref in enumerate(local):
            prev = local[max(0, i - 1)]
            nxt = local[min(len(local) - 1, i + 1)]
            is_endpoint = i == 0 or i == len(local) - 1
            if sample_dt is None:
                ds = math.hypot(nxt['x'] - prev['x'], nxt['y'] - prev['y'])
                avg_speed = max(0.08, 0.5 * (prev['speed'] + nxt['speed']))
                dt = max(1e-3, ds / avg_speed)
            else:
                dt = max(1e-3, float(sample_dt))
            difference_dt = dt if is_endpoint else 2.0 * dt
            ddx = (nxt['dx'] - prev['dx']) / difference_dt
            ddy = (nxt['dy'] - prev['dy']) / difference_dt
            dpsi = ref['speed'] * ref.get('curvature', 0.0)
            dpsi_prev = prev['speed'] * prev.get('curvature', 0.0)
            dpsi_next = nxt['speed'] * nxt.get('curvature', 0.0)
            ddpsi = (dpsi_next - dpsi_prev) / difference_dt
            point = TrajectoryPoint()
            point.header.stamp = now
            point.header.frame_id = "world_ned"
            point.pose.position.x, point.pose.position.y, point.pose.position.z = ref['x'], ref['y'], ref['z']
            point.pose.orientation = self._yaw_to_quat(ref['yaw'])
            point.twist.linear.x, point.twist.linear.y = ref['dx'], ref['dy']
            point.twist.angular.z = dpsi
            point.accel.linear.x, point.accel.linear.y = ddx, ddy
            point.accel.angular.z = ddpsi
            point.valid, point.trajectory_id = True, "arc_length"
            points.append(point)
        window = TrajectoryPointWindow(header=Header(stamp=now, frame_id="world_ned"),
                                       points=points, valid=True, trajectory_id="arc_length")
        self.traj_window_pub.publish(window)
        velocity = TwistStamped()
        velocity.header.stamp = now
        velocity.header.frame_id = "world_ned"
        velocity.twist = points[0].twist
        self.velocity_pub.publish(velocity)
        # Keep publishing the first point for legacy consumers.
        self.traj_pub.publish(points[0])


if __name__ == "__main__":
    try:
        ReferenceProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
