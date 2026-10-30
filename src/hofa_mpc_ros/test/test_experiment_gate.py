"""Depth-hold regression tests with ROS transport replaced by mocks."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def gate_factory(monkeypatch):
    rospy = ModuleType("rospy")
    rospy.Publisher = Mock()
    rospy.Subscriber = Mock()
    rospy.logerr = Mock()
    rospy.Time = SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 10.0))
    std_msgs = ModuleType("std_msgs.msg")
    std_msgs.Time = object
    nav_msgs = ModuleType("nav_msgs.msg")
    nav_msgs.Odometry = object
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msgs)
    path = Path(__file__).resolve().parents[1] / "src/hofa_mpc_ros/experiment_gate.py"
    spec = importlib.util.spec_from_file_location("depth_gate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def make(params=None, **kwargs):
        params = params or {}
        rospy.get_param = lambda name, default=None: params.get(name, default)
        return module.ExperimentGate("controller", **kwargs)

    return make, rospy


def odometry(stamp=10.0, depth=0.5, vertical_speed=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp)),
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(z=depth))),
        twist=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(z=vertical_speed))),
    )


def test_manual_policy_depth_hold_does_not_require_start_topic(gate_factory):
    make, rospy = gate_factory
    gate = make({"~vehicle_name": "custom_vehicle"},
                depth_hold=True, continuous_depth_hold=True)
    assert gate.allows(10.0)
    assert not rospy.Publisher.called
    assert rospy.Subscriber.call_args.args[0] == "/custom_vehicle/odometry"
    assert gate.depth_force() == 0.0
    gate._odom(odometry(depth=0.5, vertical_speed=0.1))
    assert gate.depth_force() == pytest.approx(3.6)
    gate._odom(odometry(stamp=10.1, depth=0.5, vertical_speed=0.1))
    rospy.Time.now = lambda: SimpleNamespace(to_sec=lambda: 10.1)
    assert gate.depth_force() == pytest.approx(3.625)


def test_existing_manual_controllers_keep_opt_in_depth_hold(gate_factory):
    make, rospy = gate_factory
    gate = make(depth_hold=True)
    assert gate.allows(10.0)
    assert not rospy.Subscriber.called
    gate._odom(odometry())
    assert gate.depth_force() == 0.0


def test_synchronized_depth_hold_still_waits_for_start(gate_factory):
    make, _ = gate_factory
    gate = make({"~experiment_start_topic": "/experiment/start"},
                depth_hold=True, continuous_depth_hold=True)
    gate._odom(odometry())
    assert gate.depth_force() == pytest.approx(4.0)
    assert not gate.allows(10.0)
    gate._start(SimpleNamespace(data=SimpleNamespace(to_sec=lambda: 11.0)))
    assert not gate.allows(10.99)
    assert gate.allows(11.0)


@pytest.mark.parametrize("stamp", [9.7, 10.1])
def test_manual_depth_hold_rejects_stale_or_future_odometry(gate_factory, stamp):
    make, _ = gate_factory
    gate = make(depth_hold=True, continuous_depth_hold=True)
    gate._odom(odometry(stamp=stamp))
    assert gate.depth_force() == 0.0


@pytest.mark.parametrize("depth, expected", [(-10.0, 15.0), (10.0, -15.0)])
def test_manual_depth_hold_limits_vertical_force(gate_factory, depth, expected):
    make, _ = gate_factory
    gate = make(depth_hold=True, continuous_depth_hold=True)
    gate._odom(odometry(depth=depth))
    assert gate.depth_force() == expected
