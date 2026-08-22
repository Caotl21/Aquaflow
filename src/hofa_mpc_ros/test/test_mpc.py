"""Unit tests for MPC solver."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.mpc import HofaMPC
from hofa_mpc_ros.types import (MPCParams, VehicleParams, VehicleState,
                                 ReferencePoint, VirtualInputBounds)
from hofa_mpc_ros.model import ThreeDOFModel
from hofa_mpc_ros.constraints import SafeInnerBoxStrategy


@pytest.fixture
def mpc():
    return HofaMPC(MPCParams(), VehicleParams())


@pytest.fixture
def allocator():
    from test_allocator import make_bricsbot_allocator
    return make_bricsbot_allocator()


def _default_bounds():
    """Reasonable default bounds for testing."""
    return VirtualInputBounds(lower=np.array([-5.0, -5.0, -5.0]),
                               upper=np.array([5.0, 5.0, 5.0]))


class TestMPCDimensions:
    def test_ad_shape(self, mpc):
        assert mpc.Ad.shape == (6, 6)

    def test_bd_shape(self, mpc):
        assert mpc.Bd.shape == (6, 3)

    def test_q_shape(self, mpc):
        assert mpc.Q.shape == (6, 6)

    def test_r_shape(self, mpc):
        assert mpc.R.shape == (3, 3)


class TestErrorState:
    def test_zero_error(self, mpc):
        """Zero error requires world-frame velocities in reference."""
        psi = 0.5
        u, v, r = 0.1, 0.05, 0.02
        c, s = np.cos(psi), np.sin(psi)
        dx_world = c * u - s * v
        dy_world = s * u + c * v

        state = VehicleState(x=1.0, y=2.0, psi=psi, u=u, v=v, r=r)
        refs = [ReferencePoint(x=1.0, y=2.0, psi=psi,
                               dx=dx_world, dy=dy_world, dpsi=r)]
        z = mpc.compute_error_state(state, refs)
        np.testing.assert_allclose(z, np.zeros(6), atol=1e-10)

    def test_position_error(self, mpc):
        state = VehicleState(x=1.0, y=2.0, psi=0.0)
        refs = [ReferencePoint(x=1.5, y=2.5, psi=0.0)]
        z = mpc.compute_error_state(state, refs)
        assert abs(z[0] - (-0.5)) < 1e-10
        assert abs(z[1] - (-0.5)) < 1e-10

    def test_yaw_error_wrapped(self, mpc):
        """wrap_to_pi((pi-0.1) - (-pi+0.1)) = wrap_to_pi(2pi-0.2) = -0.2."""
        state = VehicleState(psi=np.pi - 0.1)
        refs = [ReferencePoint(psi=-np.pi + 0.1)]
        z = mpc.compute_error_state(state, refs)
        assert abs(z[2] - (-0.2)) < 1e-10


class TestMPCSolver:
    def test_zero_error_output_near_zero(self, mpc):
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        refs = [ReferencePoint() for _ in range(mpc.Np)]
        bounds = _default_bounds()
        sol = mpc.solve(state, refs, bounds=bounds)
        assert sol.success
        np.testing.assert_allclose(sol.virtual_accel, np.zeros(3), atol=0.5)

    def test_position_error_produces_correction(self, mpc):
        state = VehicleState(x=0, y=0, psi=0)
        refs = [ReferencePoint(x=1.0, y=0, psi=0) for _ in range(mpc.Np)]
        bounds = _default_bounds()
        sol = mpc.solve(state, refs, bounds=bounds)
        assert sol.success
        assert sol.virtual_accel[0] > 0

    def test_warm_start(self, mpc):
        state = VehicleState(x=0, y=0, psi=0)
        refs = [ReferencePoint(x=0.5, y=0, psi=0) for _ in range(mpc.Np)]
        bounds = _default_bounds()
        sol1 = mpc.solve(state, refs, bounds=bounds)
        sol2 = mpc.solve(state, refs, bounds=bounds)
        assert sol1.success and sol2.success

    def test_reset_clears_warm_start(self, mpc):
        state = VehicleState(x=0, y=0, psi=0)
        refs = [ReferencePoint(x=0.5, y=0, psi=0) for _ in range(mpc.Np)]
        bounds = _default_bounds()
        mpc.solve(state, refs, bounds=bounds)
        mpc.reset()
        np.testing.assert_allclose(mpc._prev_w, np.zeros((14, 3)))


class TestConstraints:
    def test_safe_inner_box(self, allocator):
        model = ThreeDOFModel(VehicleParams())
        strategy = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        bounds = strategy.compute(state, model, allocator)
        assert np.all(bounds.lower < bounds.upper)

    def test_bounds_nonzero(self, allocator):
        model = ThreeDOFModel(VehicleParams())
        strategy = SafeInnerBoxStrategy()
        state = VehicleState(x=0, y=0, psi=0, u=0, v=0, r=0)
        bounds = strategy.compute(state, model, allocator)
        width = bounds.upper - bounds.lower
        assert np.all(width > 0)
