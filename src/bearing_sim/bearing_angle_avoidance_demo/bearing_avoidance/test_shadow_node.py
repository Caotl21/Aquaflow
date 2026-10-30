"""Node wiring/CSV tests using ROS message doubles, not a live ROS validation."""
import ast
import csv
import importlib.util
import json
import math
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
# Load native dependencies before patch.dict restores sys.modules between tests.
import bearing_shadow

WORKSPACE = Path(__file__).resolve().parents[4]
NODE = WORKSPACE / "src/aquaflow_stonefish/scripts/bearing_pf_shadow_node.py"


class Stamp:
    def __init__(self, seconds=0.):
        self.seconds = seconds
    def to_sec(self):
        return self.seconds
    def to_nsec(self):
        return round(self.seconds * 1e9)
    def __sub__(self, other):
        return Stamp(self.seconds - other.seconds)
    @staticmethod
    def now():
        return Stamp(10.02)


class Point:
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = x, y, z


class Marker:
    DELETEALL, LINE_LIST, ADD = 3, 5, 0
    def __init__(self):
        self.header = NS(frame_id="", stamp=Stamp())
        self.pose = NS(orientation=NS(w=0.), position=Point())
        self.scale = Point()
        self.color = NS(r=0., g=0., b=0., a=0.)
        self.points = []


class MarkerArray:
    def __init__(self):
        self.markers = []


class Publisher:
    def __init__(self, topic, *args, **kwargs):
        self.topic, self.messages = topic, []
    def publish(self, msg):
        self.messages.append(msg)


def module(name, **members):
    result = types.ModuleType(name)
    result.__dict__.update(members)
    return result


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        params = dict(sensor_mode="truth_angle", truth_pillars_xy_m=[[2., 0.]],
            truth_pillar_diameter_m=[4. * math.sin(math.atan2(20, 320))],
            truth_labels=[12], camera_offset_xy_m=[0., 0.],
            planner_fov_deg=140., truth_sensor_range_m=13., allowed_labels="[12]")
        def get_param(key, *default):
            if key == "/bearing_pf_stonefish":
                return params
            if key == "~log_dir":
                return self.temp.name
            if default:
                return default[0]
            raise KeyError(key)
        noop = lambda *args, **kwargs: None
        ros = module("rospy", init_node=noop, get_param=get_param,
            Publisher=Publisher, Subscriber=noop, Timer=lambda *a: NS(shutdown=noop),
            Duration=Stamp, Time=Stamp, on_shutdown=noop, loginfo=noop, logwarn_throttle=noop)
        modules = {"rospy": ros,
            "geometry_msgs": module("geometry_msgs"),
            "geometry_msgs.msg": module("geometry_msgs.msg", Point=Point),
            "nav_msgs": module("nav_msgs"), "nav_msgs.msg": module("nav_msgs.msg", Odometry=NS),
            "sensor_msgs": module("sensor_msgs"),
            "sensor_msgs.msg": module("sensor_msgs.msg", CameraInfo=NS, Image=NS),
            "std_msgs": module("std_msgs"), "std_msgs.msg": module("std_msgs.msg", String=NS),
            "visualization_msgs": module("visualization_msgs"),
            "visualization_msgs.msg": module("visualization_msgs.msg", Marker=Marker, MarkerArray=MarkerArray)}
        patches = patch.dict(sys.modules, modules)
        patches.start()
        self.addCleanup(patches.stop)
        spec = importlib.util.spec_from_file_location("shadow_node_under_test", NODE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.node = self.module.BearingPFShadowNode()
        self.addCleanup(self.node.shutdown)

    def odom(self, stamp, x=0., frame="world_ned"):
        msg = NS(header=NS(stamp=Stamp(stamp), frame_id=frame),
                 pose=NS(pose=NS(position=Point(x, 0., 1.),
                     orientation=NS(x=0., y=0., z=0., w=1.))))
        self.node.odom_cb(msg)

    def image(self, stamp=10.01, with_info=True, encoding="16UC1"):
        mask = np.zeros((480, 640), np.uint16)
        mask[100:400, 300:340] = 12
        header = NS(stamp=Stamp(stamp), frame_id="bricsbot/bearing_segmentation")
        if with_info:
            self.node.info_cb(NS(header=header, width=640, height=480,
                K=[320., 0., 320., 0., 320., 240., 0., 0., 1.]))
        msg = NS(header=header, width=640, height=480, step=1280,
                 encoding=encoding, data=mask.astype("<u2").tobytes(), is_bigendian=False)
        self.node.image_cb(msg)
        return msg

    def expire_pending(self):
        self.node.pending = type(self.node.pending)((image, time.monotonic()-1.)
                                                   for image, _ in self.node.pending)

    def read_csv(self, filename):
        self.node.save_summary()
        with (self.node.run_dir / filename).open() as stream:
            return list(csv.DictReader(stream))

    def test_paired_frame_markers_csv_and_summary(self):
        self.odom(10.)
        self.odom(10.02)
        self.image()
        self.node.update(None)
        rows = self.read_csv("angle_comparison.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair_mode"], "interpolated")
        self.assertEqual(rows[0]["segmentation_status"], "accepted_complete")
        self.assertAlmostEqual(float(rows[0]["bearing_error_rad"]), 0.)
        self.assertAlmostEqual(float(rows[0]["width_error_rad"]), 0.)
        report = json.loads((self.node.run_dir / "summary.json").read_text())
        self.assertEqual(report["frame_status"], {"compared": 1})
        namespaces = {m.ns for m in self.node.marker_pub.messages[-1].markers[1:]}
        self.assertEqual(namespaces, {"shadow_truth", "shadow_segmentation"})

    def test_waits_for_future_odometry_then_interpolates(self):
        self.odom(10.)
        self.image()
        self.node.update(None)
        self.assertEqual(len(self.node.pending), 1)
        self.odom(10.02)
        self.node.update(None)
        self.assertEqual(self.read_csv("frames.csv")[0]["pair_mode"], "interpolated")

    def test_unpaired_frames_are_logged_without_errors(self):
        self.odom(9.)
        self.image()
        self.expire_pending()
        self.node.update(None)
        self.assertEqual(self.read_csv("frames.csv")[0]["status"], "unpaired")
        self.assertEqual(self.read_csv("angle_comparison.csv"), [])

    def test_no_info_wrong_encoding_does_not_touch_control(self):
        self.odom(10.)
        self.odom(10.02)
        self.image(with_info=False)
        self.expire_pending()
        self.node.update(None)
        self.image(10.02, encoding="rgb8")
        self.node.update(None)
        self.assertEqual([r["status"] for r in self.read_csv("frames.csv")],
                         ["no_matching_camera_info", "wrong_image_encoding"])
        self.assertEqual(self.read_csv("angle_comparison.csv"), [])

    def test_duplicates_backwards_overflow_wrong_frame_counted(self):
        self.node.queue_limit = 1
        self.odom(10., frame="world_enu")
        self.image()
        self.image()
        self.image(10.)
        self.image(10.02)
        self.assertEqual(self.node.transport["duplicate_or_backward_image_stamp"], 2)
        self.assertEqual(self.node.transport["queue_dropped_images"], 1)
        self.assertEqual(self.node.transport["wrong_odom_frame"], 1)

    def test_no_control_publishers_and_no_tracker_imports(self):
        tree = ast.parse(NODE.read_text())
        topics = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "Publisher"}
        self.assertEqual(topics, {"/bearing_pf/shadow_status", "/bearing_pf/shadow_markers"})
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("avoidance_demo", imports)
        self.assertNotIn("bearing_angle_pf", imports)


if __name__ == "__main__":
    unittest.main()
