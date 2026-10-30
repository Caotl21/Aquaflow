#!/usr/bin/env python3
"""P1 Stonefish node: angular observations -> PF local path.

Normal modes are map-free and consume only the segmentation/RGB image, camera
calibration, odometry and a goal.  ``truth_angle`` is a privileged debug mode
that supplies exact bearing/angular-width observations from the fixed P1
geometry, while still withholding distance from the PF and planner.  A separate
PID/MPC node executes the published local path.
"""
import csv
import ast
import importlib.util
import math
import os
import sys
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np


def _load_demo_module():
    root = Path(__file__).resolve().parents[2] / "bearing_sim" / "bearing_angle_avoidance_demo" / "bearing_avoidance"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from avoidance_demo import (AngularObservation, Config, Tracker, candidate_risks, plan)
    from stonefish_observation import (CameraModel, ColorConfig,
                                       SegmentationConfig, extract_color_components,
                                       extract_observations, image_to_rgb,
                                       image_to_uint16, pad_edge_sectors, truth_angle_sensor)
    from rgb_association import RgbAssociator
    from bearing_angle_pf import visible_sector
    return (AngularObservation, Config, Tracker, candidate_risks, plan, CameraModel, ColorConfig,
            SegmentationConfig, extract_color_components, extract_observations,
            image_to_rgb, image_to_uint16, pad_edge_sectors, truth_angle_sensor, RgbAssociator,
            visible_sector)


(AngularObservation, DemoConfig, Tracker, candidate_risks, plan, CameraModel, ColorConfig,
 SegmentationConfig, extract_color_components, extract_observations,
 image_to_rgb, image_to_uint16, pad_edge_sectors, truth_angle_sensor, RgbAssociator,
 visible_sector) = _load_demo_module()

import rospy
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def parse_label_list(value):
    """Accept ROS YAML lists as either lists or launch-substitution strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [item for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_float_list(value, default):
    """Accept ROS YAML arrays, launch strings, or scalar fallback values."""
    if value is None or value == "":
        return [float(item) for item in default]
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [item for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [float(item) for item in value]


def parse_point_list(value, default):
    """Accept ROS YAML nested lists for privileged truth-angle geometry."""
    if value is None or value == "":
        value = default
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            raise ValueError("truth pillar positions must be a nested list")
    points = [[float(component) for component in point] for point in value]
    if any(len(point) != 2 for point in points):
        raise ValueError("truth pillar positions must contain [x, y] pairs")
    return points


def parse_bool(value, default=False):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
    if value is None:
        return bool(default)
    return bool(value)


def camera_pose_from_odom(odom, camera_offset):
    p = odom.pose.pose.position
    yaw = yaw_from_quat(odom.pose.pose.orientation)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([p.x + c * camera_offset[0] - s * camera_offset[1],
                     p.y + s * camera_offset[0] + c * camera_offset[1], yaw], dtype=float)


class BearingPFStonefishNode:
    def __init__(self):
        rospy.init_node("bearing_pf_stonefish")
        self.lock = threading.RLock()
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.sensor_mode = str(rospy.get_param("~sensor_mode", "segmentation")).lower()
        if self.sensor_mode not in ("segmentation", "rgb", "truth_angle"):
            raise ValueError("~sensor_mode must be segmentation, rgb, or truth_angle")
        self.motion_enabled = parse_bool(rospy.get_param("~motion_enabled", True), True)
        self.image_topic = rospy.get_param("~image_topic", "/bricsbot/bearing_segmentation/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/bricsbot/bearing_segmentation/camera_info")
        self.odom_topic = rospy.get_param("~odom_topic", "/bricsbot/odometry")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.path_topic = rospy.get_param("~path_topic", "/bearing_pf/local_path")
        self.velocity_topic = rospy.get_param("~velocity_topic", "/bearing_pf/local_velocity")
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.rate_hz = max(1.0, float(rospy.get_param("~rate_hz", 10.0)))
        self.image_timeout = max(0.05, float(rospy.get_param("~image_timeout_s", 0.5)))
        self.odom_timeout = max(0.05, float(rospy.get_param("~odom_timeout_s", 0.3)))
        self.max_sensor_skew = max(0.0, float(rospy.get_param("~max_sensor_skew_s", 0.15)))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance_m", 0.3))
        self.target_depth = float(rospy.get_param("~target_depth_m", 1.0))
        self.camera_offset = np.array(parse_float_list(
            rospy.get_param("~camera_offset_xy_m", [0.20, 0.0]), [0.20, 0.0]), dtype=float)
        if self.camera_offset.size != 2:
            raise ValueError("~camera_offset_xy_m must contain two values")
        self.publish_points = max(2, int(rospy.get_param("~publish_points", 20)))
        self.reference_dt = max(0.02, float(rospy.get_param("~reference_dt_s", 0.10)))
        self.debug_ray_range = max(0.5, float(rospy.get_param("~debug_ray_range_m", 5.0)))
        # Display-only ground truth (truth_pillars_* params); never used by
        # association, the PF, or the planner in rgb/segmentation mode.
        self.debug_show_truth = bool(rospy.get_param("~debug_show_truth", False))
        self.debug_particle_count = max(0, int(rospy.get_param("~debug_particle_count", 120)))
        self.debug_z_offset = float(rospy.get_param("~debug_z_offset_m", 0.08))
        self.debug_rng = np.random.default_rng(int(rospy.get_param("~seed", 7)) + 50000)
        self.rgb_frame_index = 0
        # RGB tracks are matched against their own PF-predicted bearing and
        # width from the current camera pose (ego-motion compensated).
        self.rgb_associator = RgbAssociator(
            gate_min_deg=float(rospy.get_param("~rgb_match_gate_min_deg", 3.0)),
            gate_max_deg=float(rospy.get_param("~rgb_match_gate_max_deg", 10.0)),
            width_tolerance=float(rospy.get_param("~rgb_width_tolerance", 0.4)),
            min_new_width_deg=float(rospy.get_param("~rgb_min_new_width_deg", 0.6)),
            min_new_edge_width_deg=float(rospy.get_param("~rgb_min_new_edge_width_deg", 2.0)),
            max_new_edge_width_deg=float(rospy.get_param("~rgb_max_new_edge_width_deg", 60.0)),
            confirm_hits=int(rospy.get_param("~rgb_confirm_hits", 3)),
            tentative_timeout_s=float(rospy.get_param("~rgb_tentative_timeout_s", 2.0)))
        # Recovery after the planner has mostly held for stuck_timeout_s
        # without moving: turn the camera toward the best heading outside the
        # current view, escape if the hold position itself is risky, or back off.
        self.stuck_timeout = max(0.5, float(rospy.get_param("~stuck_timeout_s", 3.0)))
        self.stuck_hold_fraction = float(rospy.get_param("~stuck_hold_fraction", 0.7))
        self.stuck_max_progress = float(rospy.get_param("~stuck_max_progress_m", 0.2))
        self.recovery_backoff_speed = max(0.0, float(rospy.get_param("~recovery_backoff_speed_mps", 0.10)))
        self.stuck_window = deque()
        self.recovering = False
        self.allowed_labels = parse_label_list(rospy.get_param("~allowed_labels", []))
        self.ignore_labels = parse_label_list(rospy.get_param("~ignore_labels", [0]))
        self.truth_pillars_xy = np.asarray(parse_point_list(
            rospy.get_param("~truth_pillars_xy_m", [
                [-2.4, 0.0], [0.0, 0.0], [2.4, 0.0], [-1.0, 1.45],
                [1.0, -1.45], [3.5, 1.55], [3.7, -1.55], [-3.7, -1.55],
            ]), [
                [-2.4, 0.0], [0.0, 0.0], [2.4, 0.0], [-1.0, 1.45],
                [1.0, -1.45], [3.5, 1.55], [3.7, -1.55], [-3.7, -1.55],
            ]), dtype=float)
        self.truth_diameters = np.asarray(parse_float_list(
            rospy.get_param("~truth_pillar_diameter_m", [
                0.60, 0.56, 0.60, 0.48, 0.44, 0.40, 0.36, 0.32,
            ]), [0.60, 0.56, 0.60, 0.48, 0.44, 0.40, 0.36, 0.32]), dtype=float)
        truth_labels = parse_label_list(rospy.get_param(
            "~truth_labels", [12, 14, 16, 18, 20, 22, 24, 26]))
        if (self.truth_pillars_xy.shape[0] != self.truth_diameters.size or
                self.truth_pillars_xy.shape[0] != len(truth_labels)):
            raise ValueError("truth pillar positions, diameters, and labels must have equal length")
        self.truth_labels = tuple(truth_labels)
        self.truth_sensor_range = max(0.0, float(rospy.get_param(
            "~truth_sensor_range_m", 13.0)))
        # Optional realism for the privileged baseline: occlusion turns hidden
        # pillars into partial sectors, noise matches the PF likelihood sigma.
        self.truth_occlusion = parse_bool(rospy.get_param("~truth_occlusion", False), False)
        self.truth_noise_deg = (
            max(0.0, float(rospy.get_param("~truth_noise_beta_deg", 0.0))),
            max(0.0, float(rospy.get_param("~truth_noise_theta_deg", 0.0))))
        self.truth_rng = np.random.default_rng(int(rospy.get_param("~seed", 7)) + 70000)
        raw_log_dir = str(rospy.get_param("~log_dir", "")).strip()
        self.log_dir = Path(raw_log_dir) if raw_log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_stream = None
        self.log_writer = None
        self.planner_log_stream = None
        self.planner_log_writer = None
        self.estimate_log_stream = None
        self.estimate_log_writer = None
        self.odom_log_stream = None
        self.odom_log_writer = None

        goal_xy = parse_float_list(rospy.get_param("~goal_xy", [5.0, 0.0]), [5.0, 0.0])
        if len(goal_xy) != 2:
            raise ValueError("~goal_xy must contain two values")
        distance_prior = parse_float_list(
            rospy.get_param("~distance_prior_m", [0.55, 13.0]), [0.55, 13.0])
        diameter_prior = parse_float_list(
            rospy.get_param("~diameter_prior_m", [0.05, 0.75]), [0.05, 0.75])
        if len(distance_prior) != 2 or len(diameter_prior) != 2:
            raise ValueError("distance and diameter priors must contain two values")

        self.demo_cfg = DemoConfig(
            seed=int(rospy.get_param("~seed", 7)),
            particles=int(rospy.get_param("~particles", 1000)),
            map_samples=int(rospy.get_param("~map_samples", 180)),
            dt=float(rospy.get_param("~planner_dt_s", 0.35)),
            horizon=float(rospy.get_param("~planner_horizon_s", 2.8)),
            fov_deg=float(rospy.get_param("~planner_fov_deg", 140.0)),
            robot_radius=float(rospy.get_param("~robot_radius_m", 0.20)),
            safety_margin=float(rospy.get_param("~safety_margin_m", 0.10)),
            risk_threshold=float(rospy.get_param("~risk_threshold", 0.08)),
            distance_prior=tuple(distance_prior),
            diameter_prior=tuple(diameter_prior),
        )
        self.obs_cfg = SegmentationConfig(
            min_pixels=int(rospy.get_param("~min_pixels", 12)),
            min_height=int(rospy.get_param("~min_height_px", 8)),
            max_width_fraction=float(rospy.get_param("~max_width_fraction", 0.30)),
            max_labels=int(rospy.get_param("~max_labels", 32)),
            partial_margin_px=int(rospy.get_param("~partial_margin_px", 2)),
        )
        self.color_cfg = ColorConfig(
            dark_max_value=int(rospy.get_param("~rgb_dark_max_value", 80)),
            dark_channel_spread=int(rospy.get_param("~rgb_dark_channel_spread", 60)),
            min_pixels=int(rospy.get_param("~rgb_min_pixels", 40)),
            min_height=int(rospy.get_param("~rgb_min_height_px", 12)),
            min_width_px=int(rospy.get_param("~rgb_min_width_px", 2)),
            band_half_height_px=int(rospy.get_param("~rgb_band_half_height_px", 24)),
            band_fill=float(rospy.get_param("~rgb_band_fill", 0.6)),
            partial_margin_px=int(rospy.get_param("~partial_margin_px", 2)),
            max_components=int(rospy.get_param("~rgb_max_components", 24)),
            morph_kernel=int(rospy.get_param("~rgb_morph_kernel", 3)),
            split_min_piece_px=int(rospy.get_param("~rgb_split_min_piece_px", 5)),
        )
        # Pillars truncated by the image side get a range-aware heading pad
        # (see pad_edge_sectors); 0 keeps the planner's fixed 8 deg.
        self.partial_edge_min_radius = max(0.0, float(rospy.get_param("~partial_edge_min_radius_m", 0.15)))
        self.camera = None
        self.info = None
        self.odom = None
        self.goal = np.array(goal_xy, dtype=float)
        self.latest_image = None
        self.latest_image_stamp = None
        self.processed_image_stamp = None
        self.latest_mask = None
        self.tracker = Tracker(self.demo_cfg)
        self.estimate_history_limit = max(2, int(rospy.get_param(
            "~estimate_history_points", 80)))
        self.estimate_history = {}
        self.previous_velocity = np.zeros(2, dtype=float)
        self.last_observation = AngularObservation({}, [])
        self.last_status = "waiting_for_camera_info"
        self.last_plan_time = None

        self.path_pub = rospy.Publisher(self.path_topic, RosPath, queue_size=1, latch=True)
        self.velocity_pub = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("/bearing_pf/status", Bool, queue_size=1, latch=True)
        self.debug_marker_pub = rospy.Publisher("/bearing_pf/debug_markers", MarkerArray,
                                                queue_size=1)
        self.rgb_mask_pub = rospy.Publisher("/bearing_pf/rgb_mask", Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)
        rospy.on_shutdown(self.shutdown)

        if self.log_dir is not None:
            self.log_stream = (self.log_dir / "observations.csv").open("w", newline="")
            self.log_writer = csv.writer(self.log_stream)
            self.log_writer.writerow(["stamp", "odom_x", "odom_y", "odom_yaw", "label", "bearing_rad", "width_rad", "partial_count", "planned_vx", "planned_vy", "risk", "feasible", "stopped", "truncated_side"])
            self.planner_log_stream = (self.log_dir / "planner_history.csv").open("w", newline="")
            self.planner_log_writer = csv.writer(self.planner_log_stream)
            self.planner_log_writer.writerow(["stamp", "odom_x", "odom_y", "odom_yaw", "goal_x", "goal_y", "vx", "vy", "risk", "clearance_5p_m", "feasible", "stopped", "reason", "partial_count", "discovered_count", "robot_radius_m", "safety_margin_m", "partial_sectors_deg"])
            self.estimate_log_stream = (self.log_dir / "particle_estimates.csv").open("w", newline="")
            self.estimate_log_writer = csv.writer(self.estimate_log_stream)
            self.estimate_log_writer.writerow(["stamp", "label", "center_x", "center_y", "diameter", "radius_m", "inflated_radius_m", "std_x_m", "std_y_m", "std_diameter_m", "ess_before_resample", "resamples"])
            self.odom_log_stream = (self.log_dir / "odometry.csv").open("w", newline="")
            self.odom_log_writer = csv.writer(self.odom_log_stream)
            self.odom_log_writer.writerow(["stamp", "x", "y", "z", "yaw"])
        rospy.loginfo("bearing PF Stonefish node ready: mode=%s image=%s info=%s odom=%s",
                      self.sensor_mode, self.image_topic, self.info_topic, self.odom_topic)
        if self.sensor_mode == "truth_angle":
            rospy.logwarn("truth_angle mode is privileged debug only: it publishes bearing/angular width, never distance "
                          "(occlusion=%s noise_deg=%.2f/%.2f)", self.truth_occlusion,
                          self.truth_noise_deg[0], self.truth_noise_deg[1])

    def info_cb(self, msg):
        with self.lock:
            self.info = msg
            fx = msg.K[0]
            fy = msg.K[4] if msg.K[4] else fx
            self.camera = CameraModel.from_values(msg.width, msg.height, fx, fy, msg.K[2], msg.K[5])

    def image_cb(self, msg):
        with self.lock:
            if self.sensor_mode == "truth_angle":
                return
            if self.sensor_mode == "segmentation" and msg.encoding not in ("16UC1", "mono16"):
                rospy.logwarn_throttle(5.0, "expected segmentation encoding 16UC1, got %s", msg.encoding)
                return
            if self.sensor_mode == "rgb" and msg.encoding.lower() not in ("rgb8", "bgr8", "rgba8", "bgra8"):
                rospy.logwarn_throttle(5.0, "expected RGB encoding, got %s", msg.encoding)
                return
            self.latest_image = msg
            self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def odom_cb(self, msg):
        with self.lock:
            self.odom = msg

    def goal_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != "world_ned":
            rospy.logwarn("ignoring goal in frame %s; expected world_ned", msg.header.frame_id)
            return
        with self.lock:
            self.goal[:] = [msg.pose.position.x, msg.pose.position.y]

    def publish_hold(self, stamp, pose):
        path = RosPath()
        path.header.stamp = stamp
        path.header.frame_id = "world_ned"
        point = PoseStamped()
        point.header = path.header
        point.pose.position.x, point.pose.position.y, point.pose.position.z = pose[0], pose[1], self.target_depth
        point.pose.orientation.z = math.sin(float(pose[2]) / 2.0)
        point.pose.orientation.w = math.cos(float(pose[2]) / 2.0)
        path.poses = [point] * self.publish_points
        self.path_pub.publish(path)
        velocity = TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = "world_ned"
        self.velocity_pub.publish(velocity)

    def publish_plan(self, stamp, pose, velocity, yaw=None):
        path = RosPath()
        path.header.stamp = stamp
        path.header.frame_id = "world_ned"
        if yaw is None:
            yaw = math.atan2(float(velocity[1]), float(velocity[0])) if np.linalg.norm(velocity) > 1e-9 else float(pose[2])
        for i in range(self.publish_points):
            t = i * self.reference_dt
            point = PoseStamped()
            point.header = path.header
            point.pose.position.x = float(pose[0] + velocity[0] * t)
            point.pose.position.y = float(pose[1] + velocity[1] * t)
            point.pose.position.z = self.target_depth
            point.pose.orientation.z = math.sin(yaw / 2.0)
            point.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(point)
        self.path_pub.publish(path)
        velocity_msg = TwistStamped()
        velocity_msg.header = path.header
        velocity_msg.twist.linear.x = float(velocity[0])
        velocity_msg.twist.linear.y = float(velocity[1])
        velocity_msg.twist.angular.z = 0.0
        self.velocity_pub.publish(velocity_msg)

    def _rgb_observation(self, image, camera, camera_pose, now):
        rgb = image_to_rgb(image.data, image.height, image.width, image.step,
                           image.encoding, image.is_bigendian)
        components, mask = extract_color_components(rgb, camera, self.color_cfg)
        detections, partial = self.rgb_associator.associate(
            components, camera_pose, self.tracker.filters, camera, now)
        for label in self.rgb_associator.prune(self.tracker.filters, now, self.tracker.first_seen):
            self.estimate_history.pop(label, None)
            rospy.loginfo("dropped tentative RGB track %s", label)
        self.latest_mask = mask
        return AngularObservation(detections, partial)

    def _truth_angle_observation(self, camera_pose):
        detections, partial = truth_angle_sensor(
            camera_pose, self.truth_pillars_xy, self.truth_diameters,
            self.truth_labels, self.demo_cfg.fov_deg, self.truth_sensor_range,
            occlusion=self.truth_occlusion, noise_deg=self.truth_noise_deg,
            rng=self.truth_rng)
        return AngularObservation(detections, partial)

    def publish_rgb_mask(self, stamp, mask):
        if mask is None:
            return
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "bricsbot/camera_front"
        msg.height, msg.width = mask.shape
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = int(mask.shape[1])
        msg.data = np.asarray(mask, dtype=np.uint8).tobytes()
        self.rgb_mask_pub.publish(msg)

    @staticmethod
    def _point(x, y, z):
        point = Point()
        point.x, point.y, point.z = float(x), float(y), float(z)
        return point

    @staticmethod
    def _set_color(marker, color):
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    def _line_marker(self, stamp, namespace, marker_id, points, color,
                     width=0.025, marker_type=Marker.LINE_STRIP):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "world_ned"
        marker.ns, marker.id = namespace, int(marker_id)
        marker.type, marker.action = marker_type, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        if marker_type == Marker.POINTS:
            marker.scale.y = float(width)
        self._set_color(marker, color)
        marker.points = list(points)
        marker.lifetime = rospy.Duration(0.5)
        return marker

    def _text_marker(self, stamp, namespace, marker_id, text, position, color):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "world_ned"
        marker.ns, marker.id = namespace, int(marker_id)
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.13
        self._set_color(marker, color)
        marker.text = str(text)
        marker.lifetime = rospy.Duration(0.5)
        return marker

    @staticmethod
    def _circle_points(center_x, center_y, radius, z, count=48):
        angles = np.linspace(0.0, 2.0 * math.pi, count + 1)
        return [Point(x=float(center_x + radius * math.cos(a)),
                      y=float(center_y + radius * math.sin(a)), z=float(z))
                for a in angles]

    def _particle_points(self, pf, z):
        if self.debug_particle_count <= 0 or pf.particles is None:
            return []
        count = min(self.debug_particle_count, pf.n)
        weights = np.asarray(pf.weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if weights.sum() <= 0.0:
            weights = np.full(pf.n, 1.0 / pf.n)
        else:
            weights = weights / weights.sum()
        indices = self.debug_rng.choice(pf.n, count, replace=False, p=weights)
        return [self._point(pf.particles[i, 0], pf.particles[i, 1], z)
                for i in indices]

    def _covariance_ellipse(self, pf, z):
        if pf.particles is None or pf.n < 2:
            return []
        weights = np.asarray(pf.weights, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if weights.sum() <= 0.0:
            weights = np.full(pf.n, 1.0 / pf.n)
        else:
            weights /= weights.sum()
        mean = np.average(pf.particles, axis=0, weights=weights)
        delta = pf.particles[:, :2] - mean[:2]
        cov = (delta * weights[:, None]).T @ delta
        if not np.isfinite(cov).all():
            return []
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 1e-8)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # Two standard deviations make the posterior ambiguity visible without
        # presenting it as a hard confidence boundary.
        axes = 2.0 * np.sqrt(values)
        angles = np.linspace(0.0, 2.0 * math.pi, 49)
        points = []
        for angle in angles:
            local = np.array([axes[0] * math.cos(angle), axes[1] * math.sin(angle)])
            point = mean[:2] + vectors @ local
            points.append(self._point(point[0], point[1], z))
        return points

    def publish_debug_markers(self, stamp, pose, camera_pose, obs, info, depth):
        """Publish only online quantities: image sectors, PF posterior, and plan."""
        markers = MarkerArray()
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = "world_ned"
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        viz_z = float(depth + self.debug_z_offset)
        cam_z = float(depth + self.debug_z_offset + 0.02)
        cam_x, cam_y, cam_yaw = camera_pose

        if self.debug_show_truth:
            inflate = self.demo_cfg.robot_radius + self.demo_cfg.safety_margin
            for index, ((x, y), diameter) in enumerate(zip(self.truth_pillars_xy, self.truth_diameters)):
                radius = float(diameter) / 2.0
                markers.markers.append(self._line_marker(
                    stamp, "truth_obstacle", 2000 + index,
                    self._circle_points(x, y, radius, viz_z - 0.02),
                    (0.85, 0.85, 0.85, 0.9), 0.03))
                markers.markers.append(self._line_marker(
                    stamp, "truth_inflated", 2000 + index,
                    self._circle_points(x, y, radius + inflate, viz_z - 0.02),
                    (0.6, 0.6, 0.6, 0.5), 0.015))
                markers.markers.append(self._text_marker(
                    stamp, "truth_labels", 2000 + index, "T%02d" % (index + 1),
                    self._point(x, y, viz_z - 0.15), (0.85, 0.85, 0.85, 0.8)))

        # Camera FOV and current detected angular intervals.
        fov_left = cam_yaw - math.radians(self.demo_cfg.fov_deg / 2.0)
        fov_right = cam_yaw + math.radians(self.demo_cfg.fov_deg / 2.0)
        fov_points = [self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(fov_left),
                                  cam_y + self.debug_ray_range * math.sin(fov_left), cam_z),
                      self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(fov_right),
                                  cam_y + self.debug_ray_range * math.sin(fov_right), cam_z)]
        markers.markers.append(self._line_marker(
            stamp, "camera_fov", 1, fov_points, (0.2, 0.5, 1.0, 0.45), 0.018,
            Marker.LINE_LIST))

        for index, (label, measurement) in enumerate(sorted(obs.detections.items())):
            # Truncated [edge, bound, side] observations show their visible part.
            beta, theta = visible_sector(measurement)
            left, right = cam_yaw + beta - theta / 2.0, cam_yaw + beta + theta / 2.0
            center = cam_yaw + beta
            points = [self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(left),
                                  cam_y + self.debug_ray_range * math.sin(left), cam_z),
                      self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(right),
                                  cam_y + self.debug_ray_range * math.sin(right), cam_z),
                      self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(center),
                                  cam_y + self.debug_ray_range * math.sin(center), cam_z)]
            markers.markers.append(self._line_marker(
                stamp, "camera_detections", 100 + index, points,
                (0.1, 1.0, 0.2, 0.9), 0.035, Marker.LINE_LIST))
            text_position = self._point(
                cam_x + 0.7 * self.debug_ray_range * math.cos(center),
                cam_y + 0.7 * self.debug_ray_range * math.sin(center),
                cam_z + 0.08)
            markers.markers.append(self._text_marker(
                stamp, "camera_labels", 100 + index,
                "id=%s b=%.1fdeg w=%.1fdeg" %
                (label, math.degrees(beta), math.degrees(theta)),
                text_position, (0.2, 1.0, 0.2, 0.95)))

        for index, (left, right) in enumerate(obs.partial_sectors):
            points = [self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(cam_yaw + left),
                                  cam_y + self.debug_ray_range * math.sin(cam_yaw + left), cam_z),
                      self._point(cam_x, cam_y, cam_z),
                      self._point(cam_x + self.debug_ray_range * math.cos(cam_yaw + right),
                                  cam_y + self.debug_ray_range * math.sin(cam_yaw + right), cam_z)]
            markers.markers.append(self._line_marker(
                stamp, "camera_partial_sectors", 200 + index, points,
                (1.0, 0.65, 0.05, 0.9), 0.04, Marker.LINE_LIST))

        # Posterior means, particle spread, and the exact inflated radius used
        # by candidate_risks(). These are the key estimation/planning split.
        for index, (label, pf) in enumerate(sorted(self.tracker.filters.items())):
            if pf.particles is None:
                continue
            weights = np.asarray(pf.weights, dtype=float)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            if weights.sum() <= 0.0:
                weights = np.full(pf.n, 1.0 / pf.n)
            else:
                weights /= weights.sum()
            mean = np.average(pf.particles, axis=0, weights=weights)
            inflated = float(mean[2] / 2.0 + self.demo_cfg.robot_radius + self.demo_cfg.safety_margin)
            history = self.estimate_history.setdefault(
                str(label), deque(maxlen=self.estimate_history_limit))
            if not history or np.linalg.norm(np.asarray(history[-1]) - mean[:2]) > 1e-6:
                history.append((float(mean[0]), float(mean[1])))
            history_points = [self._point(x, y, viz_z + 0.01) for x, y in history]
            if len(history_points) >= 2:
                markers.markers.append(self._line_marker(
                    stamp, "pf_estimate_trail", 1000 + index, history_points,
                    (1.0, 0.55, 0.05, 0.8), 0.035))

            # Filled disks make the current posterior visible when the RViz
            # camera is zoomed out; the outline markers below retain the exact
            # geometry and uncertainty layers.
            estimate_marker = Marker()
            estimate_marker.header.stamp = stamp
            estimate_marker.header.frame_id = "world_ned"
            estimate_marker.ns, estimate_marker.id = "pf_estimated_obstacle", 1000 + index
            estimate_marker.type, estimate_marker.action = Marker.CYLINDER, Marker.ADD
            estimate_marker.pose.position.x = float(mean[0])
            estimate_marker.pose.position.y = float(mean[1])
            estimate_marker.pose.position.z = viz_z
            estimate_marker.pose.orientation.w = 1.0
            estimate_marker.scale.x = estimate_marker.scale.y = max(0.01, float(mean[2]))
            estimate_marker.scale.z = 0.035
            self._set_color(estimate_marker, (1.0, 0.55, 0.05, 0.24))
            estimate_marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(estimate_marker)

            inflated_marker = Marker()
            inflated_marker.header.stamp = stamp
            inflated_marker.header.frame_id = "world_ned"
            inflated_marker.ns, inflated_marker.id = "pf_estimated_inflated", 1000 + index
            inflated_marker.type, inflated_marker.action = Marker.CYLINDER, Marker.ADD
            inflated_marker.pose.position.x = float(mean[0])
            inflated_marker.pose.position.y = float(mean[1])
            inflated_marker.pose.position.z = viz_z - 0.01
            inflated_marker.pose.orientation.w = 1.0
            inflated_marker.scale.x = inflated_marker.scale.y = max(0.01, 2.0 * inflated)
            inflated_marker.scale.z = 0.025
            self._set_color(inflated_marker, (1.0, 0.05, 0.05, 0.10))
            inflated_marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(inflated_marker)
            markers.markers.append(self._line_marker(
                stamp, "pf_mean_circle", 300 + index,
                self._circle_points(mean[0], mean[1], max(0.01, mean[2] / 2.0), viz_z),
                (1.0, 0.65, 0.05, 0.9), 0.035))
            markers.markers.append(self._line_marker(
                stamp, "pf_inflated_circle", 300 + index,
                self._circle_points(mean[0], mean[1], inflated, viz_z + 0.02),
                (1.0, 0.05, 0.05, 0.75), 0.045))
            markers.markers.append(self._line_marker(
                stamp, "pf_uncertainty_2sigma", 300 + index,
                self._covariance_ellipse(pf, viz_z + 0.04),
                (0.1, 0.9, 1.0, 0.75), 0.025))
            markers.markers.append(self._line_marker(
                stamp, "pf_particles", 300 + index,
                self._particle_points(pf, viz_z + 0.06),
                (0.1, 0.9, 1.0, 0.65), 0.025, Marker.POINTS))
            delta = pf.particles[:, :2] - mean[:2]
            std_xy = np.sqrt(np.maximum(np.average(delta * delta, axis=0, weights=weights), 0.0))
            markers.markers.append(self._text_marker(
                stamp, "pf_labels", 300 + index,
                "id=%s d=%.2fm Rinf=%.2fm std=(%.2f,%.2f)" %
                (label, mean[2], inflated, std_xy[0], std_xy[1]),
                self._point(mean[0], mean[1], viz_z + 0.18),
                (1.0, 0.8, 0.1, 0.95)))

        # Current body footprint and the local segment selected by the planner.
        robot = Marker()
        robot.header.stamp = stamp
        robot.header.frame_id = "world_ned"
        robot.ns, robot.id = "robot_footprint", 1
        robot.type, robot.action = Marker.CYLINDER, Marker.ADD
        robot.pose.position.x, robot.pose.position.y, robot.pose.position.z = pose[0], pose[1], viz_z
        robot.pose.orientation.w = 1.0
        robot.scale.x = robot.scale.y = 2.0 * self.demo_cfg.robot_radius
        robot.scale.z = 0.04
        self._set_color(robot, (0.1, 0.4, 1.0, 0.55))
        robot.lifetime = rospy.Duration(0.5)
        markers.markers.append(robot)
        endpoint = np.asarray(info.get("planned_endpoint", pose[:2]), dtype=float)
        plan_color = (0.1, 1.0, 0.2, 0.95) if not info.get("stopped") else (1.0, 0.1, 0.1, 0.95)
        markers.markers.append(self._line_marker(
            stamp, "selected_plan", 1,
            [self._point(pose[0], pose[1], viz_z + 0.08),
             self._point(endpoint[0], endpoint[1], viz_z + 0.08)],
            plan_color, 0.06))
        markers.markers.append(self._text_marker(
            stamp, "planner_diagnostics", 1,
            "risk=%.3f clear5=%.2fm feasible=%d %s R=%.2f M=%.2f" %
            (info.get("risk", float("nan")), info.get("clearance_5p_m", float("nan")),
             info.get("feasible", 0), info.get("reason", ""),
             self.demo_cfg.robot_radius, self.demo_cfg.safety_margin),
            self._point(pose[0], pose[1], viz_z + 0.35),
            plan_color))
        self.debug_marker_pub.publish(markers)

    def update(self, _event):
        now = rospy.Time.now()
        with self.lock:
            camera, image, image_stamp, odom, goal = self.camera, self.latest_image, self.latest_image_stamp, self.odom, self.goal.copy()
        requires_image = self.sensor_mode != "truth_angle"
        if (not self.enabled or odom is None or
                (requires_image and (camera is None or image is None))):
            self.status_pub.publish(Bool(data=False))
            self.last_status = "waiting_for_inputs"
            if odom is not None:
                self.publish_hold(now, [odom.pose.pose.position.x,
                                        odom.pose.pose.position.y,
                                        yaw_from_quat(odom.pose.pose.orientation),
                                        self.target_depth])
            return
        image_age = ((now - image_stamp).to_sec()
                     if image_stamp is not None else 0.0)
        odom_age = (now - odom.header.stamp).to_sec()
        sensor_skew = (abs((image_stamp - odom.header.stamp).to_sec())
                       if image_stamp is not None else 0.0)
        if (self.sensor_mode != "truth_angle" and
                (image_age > self.image_timeout or sensor_skew > self.max_sensor_skew)) or (
                odom_age > self.odom_timeout or
                image_age < 0.0 or odom_age < 0.0 or sensor_skew > self.max_sensor_skew):
            self.status_pub.publish(Bool(data=False))
            self.last_status = "stale_inputs"
            self.publish_hold(now, [odom.pose.pose.position.x, odom.pose.pose.position.y, yaw_from_quat(odom.pose.pose.orientation), self.target_depth])
            return
        if (self.sensor_mode != "truth_angle" and
                self.processed_image_stamp is not None and image_stamp <= self.processed_image_stamp):
            return
        if image_stamp is not None:
            self.processed_image_stamp = image_stamp
        self.rgb_frame_index += 1
        camera_pose = camera_pose_from_odom(odom, self.camera_offset)
        if self.sensor_mode == "truth_angle":
            obs = self._truth_angle_observation(camera_pose)
            detected_labels = list(obs.detections.items())
        elif self.sensor_mode == "rgb":
            obs = self._rgb_observation(image, camera, camera_pose, now.to_sec())
            self.publish_rgb_mask(image_stamp, self.latest_mask)
            detected_labels = list(obs.detections.items())
        else:
            labels = image_to_uint16(image.data, image.height, image.width, image.step, image.is_bigendian)
            result = extract_observations(labels, camera, self.obs_cfg, self.allowed_labels, self.ignore_labels)
            obs = AngularObservation(result.detections, result.partial_sectors)
            detected_labels = list(result.detections.items())
        self.tracker.update(camera_pose, obs, int(now.to_sec() * self.rate_hz))
        maps = self.tracker.sample_maps()
        pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw_from_quat(odom.pose.pose.orientation)], dtype=float)
        if not self.motion_enabled:
            velocity = np.zeros(2, dtype=float)
            info = {"risk": 0.0, "clearance_5p_m": float("inf"), "feasible": 0,
                    "stopped": True, "reason": "depth_only"}
        elif np.linalg.norm(pose[:2] - goal) <= self.goal_tolerance:
            velocity = np.zeros(2, dtype=float)
            info = {"risk": 0.0, "feasible": 0, "stopped": True, "reason": "goal_reached"}
        else:
            if self.sensor_mode == "truth_angle":
                half_fov = math.radians(self.demo_cfg.fov_deg / 2.0)
            else:
                half_fov = min(math.atan2(camera.cx, camera.fx),
                               math.atan2(camera.width - camera.cx, camera.fx))
            planner_obs = AngularObservation(obs.detections, pad_edge_sectors(
                obs.partial_sectors, half_fov, self.partial_edge_min_radius,
                self.demo_cfg.robot_radius + self.demo_cfg.safety_margin))
            velocity, info = plan(pose, goal, self.previous_velocity, maps, planner_obs, self.demo_cfg)
        recovery_yaw = None
        if self._stuck(now.to_sec(), pose, info.get("reason") in ("no_feasible", "cost_prefers_hold")):
            velocity, recovery_yaw, info = self._recover(pose, goal, maps, info)
        self.publish_debug_markers(
            now, pose, camera_pose, obs, info, self.target_depth)
        if info.get("stopped"):
            hold_yaw = pose[2] if recovery_yaw is None else recovery_yaw
            self.publish_hold(now, [pose[0], pose[1], hold_yaw, self.target_depth])
        else:
            self.publish_plan(now, pose, velocity, recovery_yaw)
        self.previous_velocity = velocity
        self.last_observation = obs
        self.status_pub.publish(Bool(data=True))
        self.last_status = "active"
        partial_count = len(obs.partial_sectors)
        if self.odom_log_writer:
            self.odom_log_writer.writerow([now.to_sec(), pose[0], pose[1], odom.pose.pose.position.z, pose[2]])
            self.odom_log_stream.flush()
        if self.planner_log_writer:
            self.planner_log_writer.writerow([now.to_sec(), pose[0], pose[1], pose[2], goal[0], goal[1], velocity[0], velocity[1], info.get("risk", 0.0), info.get("clearance_5p_m", float("nan")), info.get("feasible", 0), int(info.get("stopped", False)), info.get("reason", ""), partial_count, len(self.tracker.filters), self.demo_cfg.robot_radius, self.demo_cfg.safety_margin,
                                             ";".join("%.1f:%.1f" % (math.degrees(lo), math.degrees(hi)) for lo, hi in obs.partial_sectors)])
            self.planner_log_stream.flush()
        if self.estimate_log_writer:
            for label, pf in self.tracker.filters.items():
                mean = np.average(pf.particles, axis=0, weights=pf.weights)
                delta = pf.particles[:, :2] - mean[:2]
                weights = np.asarray(pf.weights, dtype=float)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                if weights.sum() <= 0.0:
                    weights = np.full(pf.n, 1.0 / pf.n)
                else:
                    weights /= weights.sum()
                std_xy = np.sqrt(np.maximum(np.average(delta * delta, axis=0, weights=weights), 0.0))
                std_d = math.sqrt(max(0.0, float(np.average((pf.particles[:, 2] - mean[2]) ** 2, weights=weights))))
                inflated = mean[2] / 2.0 + self.demo_cfg.robot_radius + self.demo_cfg.safety_margin
                self.estimate_log_writer.writerow([now.to_sec(), label, mean[0], mean[1], mean[2], mean[2] / 2.0, inflated, std_xy[0], std_xy[1], std_d, pf.last_ess, pf.resamples])
            self.estimate_log_stream.flush()
        if self.log_writer:
            for label, z in detected_labels:
                # Truncated observations log their visible sector; side is
                # +1/-1 for the image side that cuts them, 0 if complete.
                beta, theta = visible_sector(z)
                side = int(z[2]) if len(z) == 3 else 0
                self.log_writer.writerow([now.to_sec(), pose[0], pose[1], pose[2], label, beta, theta, partial_count, velocity[0], velocity[1], info.get("risk", 0.0), info.get("feasible", 0), int(info.get("stopped", False)), side])
            self.log_stream.flush()

    def _stuck(self, now, pose, held):
        """True if the planner is holding now and held for at least
        ``stuck_hold_fraction`` of the last ``stuck_timeout_s`` while moving
        less than ``stuck_max_progress_m``.  Single slow "moving" ticks
        between holds (rgb_dense_s103) no longer restart the timer."""
        window = self.stuck_window
        window.append((now, pose[0], pose[1], held))
        while len(window) > 1 and window[1][0] <= now - self.stuck_timeout:
            window.popleft()
        if not held:
            self.recovering = False
        elif not self.recovering and now - window[0][0] >= self.stuck_timeout:
            # Once triggered, recovery continues until the planner moves on
            # its own, even after the recovery motion itself makes progress.
            samples = np.array([entry[1:] for entry in window], dtype=float)
            progress = np.max(np.hypot(samples[:, 0] - pose[0], samples[:, 1] - pose[1]))
            self.recovering = bool(np.mean(samples[:, 2]) >= self.stuck_hold_fraction and
                                   progress <= self.stuck_max_progress)
        return self.recovering

    def _escape_risks(self, position, endpoints, maps):
        """Collision fraction that ignores what the robot is already in.

        A map whose inflated pillars already overlap the hold position makes
        every segment (including holding) count as a collision, so a
        single-view track whose ray tail reaches the robot (rgb_dense_s101,
        hold risk 0.11-0.17) blocks all motion.  Here a map only counts if
        the segment comes closer than the current clearance (or collides
        from a free start)."""
        if maps.shape[1] == 0:
            return np.zeros(len(endpoints))
        cfg = self.demo_cfg
        inflation = maps[:, :, 2] / 2.0 + cfg.robot_radius + cfg.safety_margin
        start = np.min(np.linalg.norm(maps[:, :, :2] - position, axis=-1) - inflation, axis=1)
        ab = endpoints - position
        q = maps[None, :, :, :2] - position
        fraction = np.clip(np.sum(q * ab[:, None, None, :], axis=-1) /
                           np.maximum(np.sum(ab * ab, axis=1)[:, None, None], 1e-15), 0.0, 1.0)
        nearest = position + fraction[..., None] * ab[:, None, None, :]
        segment = np.min(np.linalg.norm(maps[None, :, :, :2] - nearest, axis=-1) - inflation[None], axis=2)
        worse = segment < np.minimum(start, 0.0)[None, :] - 0.02
        return np.mean(worse, axis=1)

    def _escape(self, pose, goal, maps, info):
        """Slow straight escape from a risky hold position: the lowest-cost
        heading (planner progress term) whose escape risk is acceptable.
        Headings outside the planner view are first turned to in place."""
        cfg = self.demo_cfg
        speed = max(self.recovery_backoff_speed, 0.05)
        headings = np.radians(np.arange(-180.0, 180.0, 10.0))
        velocity = speed * np.column_stack((np.cos(headings), np.sin(headings)))
        endpoints = pose[:2] + cfg.horizon * velocity
        risk = self._escape_risks(pose[:2], endpoints, maps)
        lookahead = np.array([min(goal[0], pose[0] + 1.6), 0.0])
        cost = np.linalg.norm(endpoints - lookahead, axis=1) + 0.10 * endpoints[:, 1] ** 2
        cost[(risk > cfg.risk_threshold) |
             (np.abs(endpoints[:, 1]) >= cfg.corridor_half_width - cfg.robot_radius - cfg.safety_margin)] = np.inf
        best = int(np.argmin(cost))
        if not np.isfinite(cost[best]):
            return None
        heading = float(headings[best])
        if abs(wrap(heading - pose[2])) > math.radians(cfg.fov_deg / 2.0 - 12.0):
            return np.zeros(2), heading, dict(info, reason="recover_escape_turn")
        return velocity[best], heading, dict(info, reason="recover_escape", stopped=False,
                                             risk=float(risk[best]), planned_endpoint=endpoints[best])

    def _recover(self, pose, goal, maps, info):
        """Planner has held too long.  If the hold position itself exceeds the
        risk threshold, escape without making any map worse.  Otherwise its
        headings are limited to the camera view and partial sectors, so look
        for any low-risk heading on the current posterior and turn the camera
        there in place; if none, back off along the current heading while
        keeping the camera forward.  Returns ``(velocity, yaw or None, info)``."""
        if info.get("risk", 0.0) > self.demo_cfg.risk_threshold:
            escape = self._escape(pose, goal, maps, info)
            if escape is not None:
                return escape
        wide = replace(self.demo_cfg, fov_deg=400.0)
        velocity, alt = plan(pose, goal, np.zeros(2), maps, AngularObservation({}, []), wide)
        if not alt.get("stopped"):
            info = dict(info, reason="recover_turn")
            return np.zeros(2), math.atan2(velocity[1], velocity[0]), info
        backward = -self.recovery_backoff_speed * np.array([math.cos(pose[2]), math.sin(pose[2])])
        if self.recovery_backoff_speed > 0.0:
            end = pose[:2] + self.demo_cfg.horizon * backward
            risk, clearance = candidate_risks(pose[:2], end[None, :], maps, self.demo_cfg)
            if risk[0] <= self.demo_cfg.risk_threshold:
                info = dict(info, reason="recover_backoff", stopped=False, risk=float(risk[0]),
                            clearance_5p_m=float(clearance[0]), planned_endpoint=end)
                return backward, float(pose[2]), info
        return np.zeros(2), None, info

    def shutdown(self):
        if self.log_stream:
            self.log_stream.close()
        if self.planner_log_stream:
            self.planner_log_stream.close()
        if self.estimate_log_stream:
            self.estimate_log_stream.close()
        if self.odom_log_stream:
            self.odom_log_stream.close()


if __name__ == "__main__":
    BearingPFStonefishNode()
    rospy.spin()

