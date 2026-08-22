"""Unit tests for thrust allocation."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.allocator import ThrusterAllocator
from hofa_mpc_ros.types import ThrusterConfig


def make_bricsbot_allocator():
    """Create allocator with BricsBot-like thruster config."""
    thrusters = [
        ThrusterConfig(
            position=np.array([0.1278, -0.1278]),
            direction=np.array([0.7071, 0.7071]),
            thrust_max=67.67, thrust_min=-58.84),
        ThrusterConfig(
            position=np.array([0.1273, 0.1273]),
            direction=np.array([-0.7071, 0.7071]),
            thrust_max=67.67, thrust_min=-58.84),
        ThrusterConfig(
            position=np.array([-0.1278, -0.1278]),
            direction=np.array([-0.7071, 0.7071]),
            thrust_max=67.67, thrust_min=-58.84),
        ThrusterConfig(
            position=np.array([-0.1273, 0.1273]),
            direction=np.array([0.7071, 0.7071]),
            thrust_max=67.67, thrust_min=-58.84),
    ]
    return ThrusterAllocator(thrusters)


@pytest.fixture
def allocator():
    return make_bricsbot_allocator()


class TestAllocationMatrix:
    def test_shape(self, allocator):
        B = allocator.allocation_matrix()
        assert B.shape == (3, 4)

    def test_rank(self, allocator):
        assert allocator.rank() == 3

    def test_symmetry(self, allocator):
        B = allocator.allocation_matrix()
        assert abs(B[0, 0] + B[0, 2]) < 0.1
        assert abs(B[0, 1] + B[0, 3]) < 0.1


class TestAllocation:
    def test_zero_force(self, allocator):
        f_h, residual = allocator.allocate(np.zeros(3))
        np.testing.assert_allclose(f_h, np.zeros(4), atol=1e-6)
        assert residual < 1e-6

    def test_forward_produces_force(self, allocator):
        """Pure X force should produce non-zero thrust that approximates input."""
        tau = np.array([10.0, 0.0, 0.0])
        f_h, residual = allocator.allocate(tau)
        tau_approx = allocator.Bh @ f_h
        assert abs(tau_approx[0] - 10.0) < 1.0  # within 1N

    def test_residual_small(self, allocator):
        tau = np.array([5.0, -3.0, 1.0])
        f_h, residual = allocator.allocate(tau)
        tau_actual = allocator.Bh @ f_h
        np.testing.assert_allclose(tau_actual, tau, atol=1.0)

    def test_within_limits(self, allocator):
        tau = np.array([20.0, -10.0, 5.0])
        f_h, _ = allocator.allocate(tau)
        for i in range(4):
            assert f_h[i] >= -58.84 - 0.1
            assert f_h[i] <= 67.67 + 0.1

    def test_yaw_moment(self, allocator):
        tau = np.array([0.0, 0.0, 5.0])
        f_h, _ = allocator.allocate(tau)
        assert np.any(np.abs(f_h) > 0.1)


class TestReachability:
    def test_zero_reachable(self, allocator):
        assert allocator.is_reachable(np.zeros(3))

    def test_small_force_reachable(self, allocator):
        assert allocator.is_reachable(np.array([5.0, -3.0, 1.0]))

    def test_large_force_not_reachable(self, allocator):
        assert not allocator.is_reachable(np.array([200.0, 200.0, 100.0]))
