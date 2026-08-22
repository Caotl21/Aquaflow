"""Unit tests for the 3-DOF dynamics model."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.model import ThreeDOFModel
from hofa_mpc_ros.types import VehicleParams


@pytest.fixture
def model():
    return ThreeDOFModel(VehicleParams())


class TestThreeDOFModel:
    def test_mass_matrix_shape(self, model):
        M = model.mass_matrix()
        assert M.shape == (3, 3)

    def test_mass_matrix_symmetric(self, model):
        M = model.mass_matrix()
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_mass_matrix_positive_definite(self, model):
        M = model.mass_matrix()
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_mass_matrix_inverse(self, model):
        M = model.mass_matrix()
        product = M @ model.M_inv
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_coriolis_zero_by_default(self, model):
        nu = np.array([1.0, 0.5, 0.2])
        C = model.coriolis(nu)
        np.testing.assert_allclose(C, np.zeros((3, 3)), atol=1e-10)

    def test_drag_diagonal(self, model):
        nu = np.array([1.0, 0.5, 0.2])
        D = model.drag(nu)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(
            D - np.diag(np.diag(D)), np.zeros((3, 3)), atol=1e-10)

    def test_drag_positive(self, model):
        nu = np.array([1.0, 0.5, 0.2])
        D = model.drag(nu)
        assert np.all(np.diag(D) > 0)

    def test_kinematics_shape(self, model):
        J = model.kinematics(0.0)
        assert J.shape == (3, 3)

    def test_kinematics_dot_at_zero_rate(self, model):
        Jdot = model.kinematics_dot(0.0, 0.0)
        np.testing.assert_allclose(Jdot, np.zeros((3, 3)), atol=1e-10)

    def test_state_derivative_shape(self, model):
        nu = np.array([1.0, 0.5, 0.2])
        tau = np.array([10.0, -5.0, 2.0])
        dnu = model.state_derivative(nu, tau)
        assert dnu.shape == (3,)

    def test_full_state_derivative_shape(self, model):
        state = np.array([1.0, 0.5, 0.3, 0.2, -0.1, 0.05])
        tau = np.array([10.0, -5.0, 2.0])
        ds = model.full_state_derivative(state, tau)
        assert ds.shape == (6,)

    def test_rk4_step_shape(self, model):
        state = np.array([1.0, 0.5, 0.3, 0.2, -0.1, 0.05])
        tau = np.array([10.0, -5.0, 2.0])
        new_state = model.rk4_step(state, tau, 0.1)
        assert new_state.shape == (6,)

    def test_rk4_step_angle_wrapped(self, model):
        state = np.array([0, 0, np.pi + 0.1, 0, 0, 0])
        tau = np.zeros(3)
        new_state = model.rk4_step(state, tau, 0.1)
        assert -np.pi <= new_state[2] <= np.pi

    def test_zero_input_stationary(self, model):
        state = np.zeros(6)
        tau = np.zeros(3)
        new_state = model.rk4_step(state, tau, 0.1)
        np.testing.assert_allclose(new_state, np.zeros(6), atol=1e-10)
