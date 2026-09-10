"""Unit tests for virtual input constraints."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from itertools import product
import numpy as np
from scipy.optimize import linprog
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
        assert np.all(width_half <= width_full + 1e-6)
        assert np.all(width_half > 0.0)

    def test_compute_for_step(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state_pred = np.array([0, 0, 0, 0, 0, 0])
        bounds = strategy.compute_for_step(state_pred, model, allocator)
        assert np.all(bounds.lower < bounds.upper)

    def test_inner_box_corners_are_actuator_feasible(self, model, allocator):
        strategy = SafeInnerBoxStrategy()
        state_pred = np.array([0.0, 0.0, 0.3, 0.4, -0.2, 0.1])
        bounds = strategy.compute_for_step(state_pred, model, allocator)
        nu = state_pred[3:]
        psi = state_pred[2]
        from hofa_mpc_ros.hofa import kinematic_matrix, kinematic_matrix_dot
        J = kinematic_matrix(psi)
        Jdot = kinematic_matrix_dot(psi, nu[2])
        drift = Jdot @ nu + J @ model.M_inv @ (
            -model.coriolis(nu) @ nu - model.drag(nu) @ nu)
        gain = J @ model.M_inv @ allocator.Bh
        f_min = np.array([t.thrust_min for t in allocator.thrusters])
        f_max = np.array([t.thrust_max for t in allocator.thrusters])
        for corner in product(*[[bounds.lower[i], bounds.upper[i]]
                                for i in range(3)]):
            result = linprog(
                np.zeros(allocator.n_thrusters),
                A_eq=gain,
                b_eq=np.asarray(corner) - drift,
                bounds=list(zip(f_min, f_max)),
                method="highs",
            )
            assert result.success
