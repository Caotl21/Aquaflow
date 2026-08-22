"""Unit tests for HOFA transforms and related functions."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.hofa import (
    wrap_to_pi, kinematic_matrix, kinematic_matrix_dot,
    drag_force, hofa_forward, hofa_inverse, hofa_forward_inverse_identity
)
from hofa_mpc_ros.model import ThreeDOFModel
from hofa_mpc_ros.types import VehicleParams


@pytest.fixture
def model():
    return ThreeDOFModel(VehicleParams())


class TestWrapToPi:
    def test_zero(self):
        assert abs(wrap_to_pi(0.0)) < 1e-10

    def test_pi(self):
        assert abs(wrap_to_pi(np.pi) - np.pi) < 1e-10 or \
               abs(wrap_to_pi(np.pi) + np.pi) < 1e-10

    def test_negative_pi(self):
        result = wrap_to_pi(-np.pi)
        assert abs(result - np.pi) < 1e-10 or abs(result + np.pi) < 1e-10

    def test_two_pi(self):
        assert abs(wrap_to_pi(2 * np.pi)) < 1e-10

    def test_three_pi_over_two(self):
        result = wrap_to_pi(3 * np.pi / 2)
        assert abs(result - (-np.pi / 2)) < 1e-10


class TestKinematicMatrix:
    def test_identity_at_zero(self):
        J = kinematic_matrix(0.0)
        np.testing.assert_allclose(J, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for psi in np.linspace(0, 2 * np.pi, 20):
            J = kinematic_matrix(psi)
            product = J.T @ J
            np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        for psi in np.linspace(0, 2 * np.pi, 20):
            J = kinematic_matrix(psi)
            assert abs(np.linalg.det(J) - 1.0) < 1e-10

    def test_ninety_degrees(self):
        J = kinematic_matrix(np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(J, expected, atol=1e-10)


class TestKinematicMatrixDot:
    def test_zero_at_zero_rate(self):
        Jdot = kinematic_matrix_dot(0.0, 0.0)
        np.testing.assert_allclose(Jdot, np.zeros((3, 3)), atol=1e-10)

    def test_numerical_consistency(self):
        psi, r = 0.5, 0.3
        dt = 1e-6
        J1 = kinematic_matrix(psi)
        J2 = kinematic_matrix(psi + r * dt)
        Jdot_num = (J2 - J1) / dt
        Jdot_analytic = kinematic_matrix_dot(psi, r)
        np.testing.assert_allclose(Jdot_analytic, Jdot_num, atol=1e-4)


class TestDragForce:
    def test_zero_velocity(self):
        dl = np.array([8.0, 10.0, 1.4])
        dq = np.array([12.0, 15.0, 0.35])
        f = drag_force(np.zeros(3), dl, dq)
        np.testing.assert_allclose(f, np.zeros(3), atol=1e-10)

    def test_drag_force_direction(self):
        """drag_force returns D(nu)*nu which is same-direction as velocity.
        It gets subtracted in the dynamics equation: tau - D*nu."""
        dl = np.array([8.0, 10.0, 1.4])
        dq = np.array([12.0, 15.0, 0.35])
        nu = np.array([1.0, -0.5, 0.2])
        f = drag_force(nu, dl, dq)
        # D(nu)*nu has same sign as nu (gets subtracted in dynamics)
        for i in range(3):
            assert f[i] * nu[i] >= 0

    def test_linear_dominant_at_low_speed(self):
        dl = np.array([8.0, 10.0, 1.4])
        dq = np.array([12.0, 15.0, 0.35])
        nu = np.array([0.01, 0.01, 0.01])
        f = drag_force(nu, dl, dq)
        f_linear = dl * nu
        np.testing.assert_allclose(f, f_linear, rtol=0.1)


class TestHofaForwardInverse:
    def test_identity_at_rest(self, model):
        state = np.zeros(6)
        a_c = np.array([0.1, -0.2, 0.05])
        error = hofa_forward_inverse_identity(state, a_c, model)
        assert error < 1e-8

    def test_identity_at_various_states(self, model):
        states = [
            np.array([1.0, 0.5, 0.3, 0.2, -0.1, 0.05]),
            np.array([0.0, 0.0, np.pi / 4, 0.5, 0.3, 0.1]),
            np.array([2.0, -1.0, -np.pi / 3, -0.3, 0.2, -0.1]),
        ]
        a_cs = [
            np.array([0.1, -0.2, 0.05]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.3, 0.4, -0.1]),
        ]
        for state, a_c in zip(states, a_cs):
            error = hofa_forward_inverse_identity(state, a_c, model)
            assert error < 1e-8, f"Failed for state={state}, a_c={a_c}, error={error}"

    def test_hofa_forward_at_rest(self, model):
        state = np.zeros(6)
        tau = np.zeros(3)
        a = hofa_forward(state, tau, model)
        np.testing.assert_allclose(a, np.zeros(3), atol=1e-10)

    def test_hofa_inverse_compensates_dynamics(self, model):
        state = np.array([0, 0, 0, 1.0, 0, 0])
        a_c = np.zeros(3)
        tau = hofa_inverse(state, a_c, model)
        assert tau[0] > 0  # positive X force to counter drag
