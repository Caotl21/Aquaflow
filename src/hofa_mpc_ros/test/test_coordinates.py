"""Unit tests for coordinate transformations."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.coordinates import (
    ned_to_enu_position, enu_to_ned_position,
    ned_to_enu_yaw, enu_to_ned_yaw,
    ned_to_enu_velocity, enu_to_ned_velocity,
    body_to_world, world_to_body,
    shortest_angle_error
)


class TestNedEnuPosition:
    def test_roundtrip(self):
        for pos in [np.array([1.0, 2.0, 3.0]),
                     np.array([-1.0, 0.5, -3.0]),
                     np.array([0.0, 0.0, 0.0])]:
            result = enu_to_ned_position(ned_to_enu_position(pos))
            np.testing.assert_allclose(result, pos, atol=1e-10)

    def test_north_becomes_east(self):
        pos_ned = np.array([1.0, 0.0, 0.0])
        pos_enu = ned_to_enu_position(pos_ned)
        np.testing.assert_allclose(pos_enu, np.array([0.0, 1.0, 0.0]))

    def test_east_becomes_north(self):
        pos_ned = np.array([0.0, 1.0, 0.0])
        pos_enu = ned_to_enu_position(pos_ned)
        np.testing.assert_allclose(pos_enu, np.array([1.0, 0.0, 0.0]))

    def test_down_becomes_up_negated(self):
        pos_ned = np.array([0.0, 0.0, 1.0])
        pos_enu = ned_to_enu_position(pos_ned)
        np.testing.assert_allclose(pos_enu, np.array([0.0, 0.0, -1.0]))


class TestNedEnuYaw:
    def test_roundtrip(self):
        for yaw in [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]:
            result = enu_to_ned_yaw(ned_to_enu_yaw(yaw))
            np.testing.assert_allclose(result, yaw, atol=1e-10)

    def test_north_yaw_maps_to_east(self):
        """NED north (0) maps to ENU east (pi/2) via yaw_enu = pi/2 - yaw_ned."""
        yaw_enu = ned_to_enu_yaw(0.0)
        np.testing.assert_allclose(yaw_enu, np.pi / 2, atol=1e-10)


class TestNedEnuVelocity:
    def test_roundtrip(self):
        vel_ned = np.array([1.0, 0.5, 0.2])
        result = enu_to_ned_velocity(ned_to_enu_velocity(vel_ned, 0.0))
        np.testing.assert_allclose(result, vel_ned, atol=1e-10)


class TestBodyWorldRotation:
    def test_identity_at_zero(self):
        v = np.array([1.0, 0.5])
        result = world_to_body(0.0, body_to_world(0.0, v))
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_ninety_degrees(self):
        v_body = np.array([1.0, 0.0])
        v_world = body_to_world(np.pi / 2, v_body)
        np.testing.assert_allclose(v_world, np.array([0.0, 1.0]), atol=1e-10)

    def test_roundtrip_various_angles(self):
        v = np.array([1.0, -0.5])
        for psi in np.linspace(0, 2 * np.pi, 20):
            result = world_to_body(psi, body_to_world(psi, v))
            np.testing.assert_allclose(result, v, atol=1e-10)


class TestShortestAngleError:
    def test_zero(self):
        assert abs(shortest_angle_error(0.5, 0.5)) < 1e-10

    def test_small_positive(self):
        err = shortest_angle_error(0.1, 0.0)
        assert abs(err - 0.1) < 1e-10

    def test_wrap_around(self):
        """From pi-0.1 to -pi+0.1: shortest path is +0.2 (through +pi)."""
        err = shortest_angle_error(-np.pi + 0.1, np.pi - 0.1)
        assert abs(err - 0.2) < 1e-10

    def test_symmetry(self):
        err1 = shortest_angle_error(0.5, 0.0)
        err2 = shortest_angle_error(-0.5, 0.0)
        assert abs(err1 + err2) < 1e-10
