#!/usr/bin/env python3
"""Stuck detection and escape recovery of bearing_pf_stonefish_node.py,
loaded with stub ROS modules (no live ROS)."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

from avoidance_demo import Config
from stonefish_observation import pad_edge_sectors

NODE = Path(__file__).resolve().parents[3] / "aquaflow_stonefish/scripts/bearing_pf_stonefish_node.py"


def load_node():
    for name, attrs in {
            "rospy": [], "geometry_msgs": [], "nav_msgs": [], "sensor_msgs": [],
            "std_msgs": [], "visualization_msgs": [],
            "geometry_msgs.msg": ["Point", "PoseStamped", "TwistStamped"],
            "nav_msgs.msg": ["Odometry", "Path"], "sensor_msgs.msg": ["CameraInfo", "Image"],
            "std_msgs.msg": ["Bool", "ColorRGBA"], "visualization_msgs.msg": ["Marker", "MarkerArray"]}.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for attr in attrs:
            # Marker constants are used as default arguments at import time.
            setattr(module, attr, type(attr, (), {"LINE_STRIP": 4, "LINE_LIST": 5, "SPHERE_LIST": 7,
                                                  "POINTS": 8, "TEXT_VIEW_FACING": 9, "ADD": 0,
                                                  "DELETEALL": 3}))
    spec = importlib.util.spec_from_file_location("bearing_pf_stonefish_node", NODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_node(module):
    from collections import deque
    node = object.__new__(module.BearingPFStonefishNode)
    node.demo_cfg = Config(robot_radius=0.20, safety_margin=0.10, fov_deg=80.0)
    node.stuck_timeout, node.stuck_hold_fraction, node.stuck_max_progress = 3.0, 0.7, 0.2
    node.recovery_backoff_speed = 0.10
    node.stuck_window, node.recovering = deque(), False
    return node


def main():
    module = load_node()
    node = make_node(module)

    # rgb_dense_s103: holds interrupted by single slow "moving" ticks.
    pose = np.array([-2.58, -0.91, 0.0])
    fired = [node._stuck(0.1 * k, pose, k % 4 != 0) or False for k in range(40)]
    assert not any(fired[:30]) and any(fired[30:]), fired
    # Recovery continues while the planner holds, and ends when it moves.
    moved = pose + np.array([0.5, 0.0, 0.0])
    assert node._stuck(4.1, moved, True) and not node._stuck(4.2, moved, False)
    # Steady slow progress is not "stuck".
    node = make_node(module)
    assert not any(node._stuck(0.1 * k, pose + [0.012 * k, 0, 0], k % 4 != 0) for k in range(60))

    # rgb_dense_s101: a single-view track's ray tail overlaps the hold
    # position (behind-right), so every planner candidate and the hold
    # itself exceed the risk threshold.  Escape must still move toward the
    # goal without coming closer to the tail.
    node = make_node(module)
    pose = np.array([4.58, 0.41, math.radians(-37.0)])
    goal = np.array([5.0, 0.0])
    tail = np.array([[4.30, -0.05, 0.5]])
    maps = np.repeat(tail[None], 100, axis=0)
    maps[:85, 0, :2] += [-3.0, 0.0]                  # 85 % of maps: far away
    hold = {"risk": 0.15, "stopped": True, "reason": "no_feasible"}
    velocity, yaw, info = node._recover(pose, goal, maps, hold)
    assert info["reason"] in ("recover_escape", "recover_escape_turn"), info
    heading = yaw
    endpoint = pose[:2] + node.demo_cfg.horizon * 0.1 * np.array([math.cos(heading), math.sin(heading)])
    assert np.linalg.norm(endpoint - tail[0, :2]) > np.linalg.norm(pose[:2] - tail[0, :2])
    assert node._escape_risks(pose[:2], endpoint[None], maps)[0] == 0.0
    # Moving straight through the tail is still rejected.
    through = pose[:2] + 2.0 * (tail[0, :2] - pose[:2])
    assert node._escape_risks(pose[:2], through[None], maps)[0] > 0.08

    # rgb_dense_s105: a pillar truncated at the image side (-40:-18.8 deg)
    # needs more than the fixed 8 deg pad; narrow edge slivers keep 8 deg.
    half = math.atan2(320.0, 381.4)
    (lo, hi), = pad_edge_sectors([(-half, math.radians(-18.8))], half, 0.15, 0.30)
    assert lo == -half and math.degrees(hi) + 8.0 > 0.0, math.degrees(hi)
    assert pad_edge_sectors([(-half, -half + math.radians(2.0))], half, 0.15, 0.30) == \
        [(-half, -half + math.radians(2.0))]
    inner = [(math.radians(5.0), math.radians(15.0))]
    assert pad_edge_sectors(inner, half, 0.15, 0.30) == inner
    print("PASS: stuck window, escape from a risky hold position, edge-sector padding")


if __name__ == "__main__":
    main()
