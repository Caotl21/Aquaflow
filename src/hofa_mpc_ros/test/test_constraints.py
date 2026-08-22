"""Unit tests for virtual input constraints."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.constraints import SafeInnerBoxStrategy, CurrentStateBoxStrategy
from hofa_mpc_ros.types import VehicleParams, VehicleState, ThrusterConfig
from hofa_mpc_ros.model import ThreeDOFModel
from hofa_mpc_ros.allocator import ThrusterAllocator


def make_simple_allocator():
    thrusters = [
        ThrusterConfig(
            position=np.array([0.2, -0.2]),
            direction=np.array([1.0, 0.0]),
            thrust_max=10.0, thrust_min=-10.0),
        ThrusterConfig(
            position=np.array([0.2, 0.2]),
            direction=np.array([0.0, 1.0]),
            thrust_max=10.0, thrust_min=-10.0),
        ThrusterConfig(
            position=np.array([-0.2, -0.2]),
            direction=np.array([-1.0, 0.0]),
            thrust_max=10.0, thrust_min=-10.0),
        ThrusterConfig(
            position=np.array([-0.2, 0.2]),
            direction=np.array([0.0, -1.0]),
            thrust_max=10.0, thrust_min=-10.0),
    ]
    return ThrusterAllocator(thrusters)


@pytest.fixture
def model():
    return ThreeDOFModel(VehicleParams())


@pytest.fixture
def allocator():
    return make_simple_allocator()


class TestCurrentStateBoxStrategy:
    def test_bounds_valid(self, model, allocator):
        strategy = CurrentStateBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        bounds = strategy.compute(state, model, allocator)
        assert np.all(bounds.lower < bounds.upper)

    def test_bounds_symmetric_at_rest(self, model, allocator):
        strategy = CurrentStateBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        bounds = strategy.compute(state, model, allocator)
        center = 0.5 * (bounds.lower + bounds.upper)
        np.testing.assert_allclose(center, np.zeros(3), atol=1.0)


class TestSafeInnerBoxStrategy:
    def test_bounds_valid(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        bounds = strategy.compute(state, model, allocator)
        assert np.all(bounds.lower < bounds.upper)

    def test_inner_box_smaller_than_outer(self, model, allocator):
        strategy_outer = CurrentStateBoxStrategy()
        strategy_inner = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)

        outer = strategy_outer.compute(state, model, allocator)
        inner = strategy_inner.compute(state, model, allocator)

        outer_width = outer.upper - outer.lower
        inner_width = inner.upper - inner.lower
        assert np.all(inner_width <= outer_width + 1e-6)

    def test_bounds_with_velocity(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0.5, v=0.2, r=0.1)
        bounds = strategy.compute(state, model, allocator)
        assert np.all(bounds.lower < bounds.upper)

    def test_bounds_scale(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)

        bounds_full = strategy.compute(state, model, allocator, scale=1.0)
        bounds_half = strategy.compute(state, model, allocator, scale=0.5)

        width_full = bounds_full.upper - bounds_full.lower
        width_half = bounds_half.upper - bounds_half.lower
        np.testing.assert_allclose(width_half, width_full * 0.5, atol=1e-6)

    def test_compute_for_step(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state_pred = np.array([0, 0, 0, 0, 0, 0])
        bounds = strategy.compute_for_step(state_pred, model, allocator)
        assert np.all(bounds.lower < bounds.upper)
