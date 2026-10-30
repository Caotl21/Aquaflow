#!/usr/bin/env python3
"""Read-only truth-angle vs segmentation comparison, in a separate process.

No Tracker, planner, path, velocity, or actuator publishers are used here.
Image headers are kept unchanged; pairing cannot correct rendering latency.
"""
import ast
import csv
import json
import math
import os
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

adapter_root = Path(__file__).resolve().parents[2] / "bearing_sim" / "bearing_angle_avoidance_demo" / "bearing_avoidance"
sys.path.insert(0, str(adapter_root))
from stonefish_observation import CameraModel, SegmentationConfig, image_to_uint16
from bearing_shadow import ComparisonSummary, camera_pose, compare_truth_and_segmentation, match_odometry


FRAME_FIELDS = ["image_stamp", "image_frame", "camera_info_stamp", "odom_frame",
                "status", "pair_mode", "nearest_skew_s", "bracket_span_s",
                "before_stamp", "after_stamp", "image_age_s", "queue_wait_s",
                "body_x", "body_y", "body_yaw_rad", "roll_rad", "pitch_rad",
                "camera_x", "camera_y", "camera_yaw_rad", "fx", "cx",
                "observed_labels", "unexpected_labels", "detail"]
OBS_FIELDS = ["label", "truth_available", "truth_bearing_rad", "truth_width_rad",
              "truth_visibility", "predicted_visible_fraction", "predicted_occlusion",
              "segmentation_pixels", "segmentation_status", "rejection_reason",
              "segmentation_bearing_rad", "segmentation_width_rad",
              "bearing_error_rad", "width_error_rad", "bbox_bearing_error_rad",
              "bbox_width_error_rad", "left_px", "right_px", "top_px", "bottom_px"]


def array_parameter(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def odometry_sample(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid odometry quaternion")
    x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
    roll = math.atan2(2*(w*x + y*z), 1-2*(x*x + y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))
    sample = (msg.header.stamp.to_sec(), p.x, p.y, yaw, roll, pitch)
    if not all(math.isfinite(v) for v in sample) or sample[0] <= 0:
        raise ValueError("invalid odometry sample")
    return sample


class BearingPFShadowNode:
    def __init__(self):
        rospy.init_node("bearing_pf_shadow")
        self.lock, self.work_lock = threading.Lock(), threading.Lock()
        self.closed = False
        self.pending = deque()
        self.odom = deque(maxlen=2000)
        self.camera_infos = deque(maxlen=120)
        self.last_image_ns = None
        self.transport = Counter()
        self.summary = ComparisonSummary()
        self.last_frame = {}
        source = str(rospy.get_param("~source_namespace", "/bearing_pf_stonefish")).rstrip("/")
        params = rospy.get_param(source)
        if params.get("sensor_mode") != "truth_angle":
            raise ValueError("shadow comparison requires truth_angle driving mode")
        if rospy.get_param("~scene", "bearing_pf_pillars") != "bearing_pf_pillars":
            raise ValueError("this launch's oracle geometry is for bearing_pf_pillars only")
        self.centers = np.asarray(array_parameter(params["truth_pillars_xy_m"]), float)
        self.diameters = np.asarray(array_parameter(params["truth_pillar_diameter_m"]), float)
        self.labels = [int(v) for v in array_parameter(params["truth_labels"])]
        self.offset = np.asarray(array_parameter(params["camera_offset_xy_m"]), float)
        if (self.centers.shape != (len(self.labels), 2) or
                self.diameters.shape != (len(self.labels),) or not self.labels or
                len(set(self.labels)) != len(self.labels) or min(self.labels) <= 0 or
                self.offset.shape != (2,) or not np.isfinite(self.offset).all() or
                not np.isfinite(self.centers).all() or not np.isfinite(self.diameters).all()
                or np.any(self.diameters <= 0)):
            raise ValueError("invalid truth geometry/labels/camera offset")
        self.fov = float(params["planner_fov_deg"])
        self.sensor_range = float(params["truth_sensor_range_m"])
        self.allowed = array_parameter(params.get("allowed_labels", []))
        self.ignored = array_parameter(params.get("ignore_labels", [0]))
        self.config = SegmentationConfig(
            min_pixels=int(params.get("min_pixels", 12)),
            min_height=int(params.get("min_height_px", 8)),
            max_width_fraction=float(params.get("max_width_fraction", .30)),
            max_labels=int(params.get("max_labels", 32)),
            partial_margin_px=int(params.get("partial_margin_px", 2)))
        self.max_skew = float(rospy.get_param("~max_pair_skew_s", .05))
        self.max_gap = float(rospy.get_param("~max_interpolation_gap_s", .10))
        self.wait_s = float(rospy.get_param("~pair_wait_s", .20))
        self.queue_limit = max(1, int(rospy.get_param("~queue_size", 60)))
        self.depth = float(params.get("target_depth_m", 1.))
        self.image_topic = rospy.get_param("~image_topic", "/bricsbot/bearing_segmentation/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/bricsbot/bearing_segmentation/camera_info")
        self.odom_topic = params.get("odom_topic", "/bricsbot/odometry")
        if (not all(math.isfinite(v) for v in (self.max_skew, self.max_gap, self.wait_s,
                self.fov, self.sensor_range, self.depth)) or
                min(self.max_skew, self.max_gap, self.wait_s, self.sensor_range) <= 0 or
                not 0 < self.fov <= 360):
            raise ValueError("pairing tolerances must be positive")
        root = Path(str(rospy.get_param("~log_dir", "")).strip() or "/tmp/bearing_pf_shadow")
        self.run_dir = root / ("shadow_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.frames_file = (self.run_dir / "frames.csv").open("x", newline="")
        self.rows_file = (self.run_dir / "angle_comparison.csv").open("x", newline="")
        self.frame_writer = csv.DictWriter(self.frames_file, FRAME_FIELDS)
        self.row_writer = csv.DictWriter(self.rows_file, FRAME_FIELDS + OBS_FIELDS)
        self.frame_writer.writeheader()
        self.row_writer.writeheader()
        metadata = dict(source_namespace=source, driving_sensor_mode="truth_angle",
            scene="bearing_pf_pillars", truth_pillars_xy_m=self.centers.tolist(),
            truth_pillar_diameter_m=self.diameters.tolist(), truth_labels=self.labels,
            camera_offset_xy_m=self.offset.tolist(), oracle_fov_deg=self.fov,
            oracle_range_m=self.sensor_range, segmentation_config=asdict(self.config),
            allowed_labels=self.allowed, ignore_labels=self.ignored,
            image_topic=self.image_topic, camera_info_topic=self.info_topic,
            odom_topic=self.odom_topic, max_pair_skew_s=self.max_skew,
            max_interpolation_gap_s=self.max_gap, pair_wait_s=self.wait_s,
            error_sign="segmentation minus truth; bearing wrapped to [-pi,pi)",
            timestamp_policy="unmodified image header; interpolate odometry, bounded nearest fallback",
            attitude_model="yaw-only horizontal angles; roll/pitch logged, not compensated")
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.status_pub = rospy.Publisher("/bearing_pf/shadow_status", String, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/bearing_pf/shadow_markers", MarkerArray, queue_size=1)
        self.last_flush = time.monotonic()
        self.save_summary()
        self.subs = [rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=200),
            rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb, queue_size=60),
            rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=30,
                             buff_size=8 * 1024 * 1024)]
        self.timer = rospy.Timer(rospy.Duration(.05), self.update)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Truth/segmentation shadow logging to %s", self.run_dir)

    def odom_cb(self, msg):
        with self.lock:
            if msg.header.frame_id != "world_ned":
                self.transport["wrong_odom_frame"] += 1
                return
            try:
                sample = odometry_sample(msg)
            except ValueError:
                self.transport["invalid_odometry"] += 1
                return
            if self.odom and sample[0] <= self.odom[-1][0]:
                self.transport["nonmonotonic_odometry"] += 1
                return
            self.odom.append(sample)

    def info_cb(self, msg):
        with self.lock:
            if (not msg.header.frame_id or not all(math.isfinite(v) for v in msg.K) or
                    msg.K[0] <= 0 or msg.K[4] <= 0):
                self.transport["invalid_camera_info"] += 1
                return
            self.camera_infos.append(msg)

    def image_cb(self, msg):
        with self.lock:
            if self.closed:
                return
            self.transport["received_images"] += 1
            ns = msg.header.stamp.to_nsec()
            if ns <= 0:
                self.transport["invalid_image_stamp"] += 1
                return
            if self.last_image_ns is not None and ns <= self.last_image_ns:
                self.transport["duplicate_or_backward_image_stamp"] += 1
                return
            self.last_image_ns = ns
            if len(self.pending) >= self.queue_limit:
                self.pending.popleft()
                self.transport["queue_dropped_images"] += 1
            self.pending.append((msg, time.monotonic()))

    def update(self, _event):
        with self.work_lock:
            if self.closed:
                return
            # Bounded work per timer callback; all skipped frames are counted.
            for _ in range(4):
                with self.lock:
                    if not self.pending:
                        break
                    image, received = self.pending[0]
                    age = time.monotonic() - received
                    infos = [m for m in self.camera_infos if
                             m.header.frame_id == image.header.frame_id and
                             m.width == image.width and m.height == image.height]
                    calibration = min(infos, key=lambda m: abs(
                        (m.header.stamp - image.header.stamp).to_sec())) if infos else None
                    info_ok = calibration is not None and abs(
                        (calibration.header.stamp - image.header.stamp).to_sec()) <= self.max_skew
                    ready = self.odom and self.odom[-1][0] >= image.header.stamp.to_sec() and info_ok
                    if not ready and age < self.wait_s:
                        break
                    self.pending.popleft()
                    odometry = list(self.odom)
                frame = dict(image_stamp=image.header.stamp.to_sec(), image_frame=image.header.frame_id,
                    camera_info_stamp=calibration.header.stamp.to_sec() if calibration else None,
                    odom_frame="world_ned", image_age_s=(rospy.Time.now()-image.header.stamp).to_sec(),
                    queue_wait_s=age)
                rows = []
                try:
                    if not info_ok:
                        frame["status"] = "no_matching_camera_info"
                    elif image.encoding not in ("16UC1", "mono16"):
                        frame["status"] = "wrong_image_encoding"
                    else:
                        match = match_odometry(frame["image_stamp"], odometry, self.max_skew, self.max_gap)
                        frame.update({k: v for k, v in asdict(match).items() if k not in ("pose", "mode")})
                        frame["pair_mode"] = match.mode
                        frame["status"] = "compared" if match.pose is not None else match.mode
                        if match.pose is not None:
                            camera = CameraModel.from_values(image.width, image.height, calibration.K[0],
                                calibration.K[4], calibration.K[2], calibration.K[5])
                            pose = camera_pose(match.pose, self.offset)
                            frame.update(body_x=match.pose[0], body_y=match.pose[1], body_yaw_rad=match.pose[2],
                                roll_rad=match.pose[3], pitch_rad=match.pose[4], camera_x=pose[0],
                                camera_y=pose[1], camera_yaw_rad=pose[2], fx=camera.fx, cx=camera.cx)
                            mask = image_to_uint16(image.data, image.height, image.width,
                                                   image.step, image.is_bigendian)
                            ids = set(int(v) for v in np.unique(mask)) - {0}
                            frame["observed_labels"] = json.dumps(sorted(ids))
                            frame["unexpected_labels"] = json.dumps(sorted(ids-set(self.labels)-set(self.ignored)))
                            rows = compare_truth_and_segmentation(mask, camera, pose, self.centers,
                                self.diameters, self.labels, self.fov, self.sensor_range, self.config,
                                self.allowed, self.ignored)
                            self.publish_markers(image.header.stamp, pose, rows)
                except (ValueError, TypeError, IndexError) as error:
                    frame.update(status="invalid_frame", detail=str(error))
                    rows = []
                self.frame_writer.writerow(frame)
                self.row_writer.writerows(dict(frame, **row) for row in rows)
                self.summary.add_frame(frame["status"], rows)
                self.last_frame = frame
            if time.monotonic() - self.last_flush >= 2.:
                self.save_summary()
                self.last_flush = time.monotonic()
                if not self.summary.frame_status.get("compared"):
                    rospy.logwarn_throttle(5., "Shadow comparison has no paired frames yet; see %s/summary.json", self.run_dir)

    def publish_markers(self, stamp, pose, rows):
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp = "world_ned", stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for index, row in enumerate(rows):
            for prefix, color in (("truth", (1., .65, .05)), ("segmentation", (1., .1, 1.))):
                beta, width = row[prefix + "_bearing_rad"], row[prefix + "_width_rad"]
                if beta is None or (prefix == "truth" and not row["truth_available"]):
                    continue
                marker = Marker()
                marker.header.frame_id, marker.header.stamp = "world_ned", stamp
                marker.ns, marker.id = "shadow_" + prefix, index
                marker.type, marker.action = Marker.LINE_LIST, Marker.ADD
                marker.pose.orientation.w = 1.
                marker.scale.x = .012
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = .8
                marker.lifetime = rospy.Duration(1.)
                for angle in (beta - width/2, beta, beta + width/2):
                    marker.points += [Point(pose[0], pose[1], self.depth-.08),
                        Point(pose[0]+1.5*math.cos(pose[2]+angle),
                              pose[1]+1.5*math.sin(pose[2]+angle), self.depth-.08)]
                markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def save_summary(self):
        self.frames_file.flush()
        self.rows_file.flush()
        result = self.summary.result()
        with self.lock:
            result["transport"] = dict(self.transport)
            result["pending_images"] = len(self.pending)
        result.update(run_dir=str(self.run_dir), last_frame=self.last_frame)
        temporary = self.run_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(str(temporary), str(self.run_dir / "summary.json"))
        self.status_pub.publish(String(data=json.dumps(result, allow_nan=False)))

    def shutdown(self):
        self.timer.shutdown()
        with self.work_lock:
            if self.closed:
                return
            self.closed = True
            with self.lock:
                self.transport["unprocessed_at_shutdown"] += len(self.pending)
            self.save_summary()
            self.frames_file.close()
            self.rows_file.close()


if __name__ == "__main__":
    BearingPFShadowNode()
    rospy.spin()
