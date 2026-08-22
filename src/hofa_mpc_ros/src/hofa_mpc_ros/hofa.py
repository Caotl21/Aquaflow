"""High-Order Forward Attachment (HOFA) transforms.

HOFA maps between world-frame virtual acceleration and body-frame
generalized forces, accounting for the vehicle's kinematics and dynamics.
"""
import numpy as np
from .model import ThreeDOFModel


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_to_pi_array(angles: np.ndarray) -> np.ndarray:
    """Wrap array of angles to [-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def kinematic_matrix(psi: float) -> np.ndarray:
    """Rotation matrix J(psi) from body to world frame.

    J(psi) = [[cos(psi), -sin(psi), 0],
              [sin(psi),  cos(psi), 0],
              [0,         0,        1]]
    """
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def kinematic_matrix_dot(psi: float, r: float) -> np.ndarray:
    """Time derivative of J(psi)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [-r * s, -r * c, 0],
        [r * c, -r * s, 0],
        [0, 0, 0]
    ])


def drag_force(nu: np.ndarray, dl: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Linear + quadratic drag force: D(nu)*nu."""
    d = np.array([dl[i] + dq[i] * abs(nu[i]) for i in range(3)])
    return d * nu


def hofa_forward(state: np.ndarray, tau: np.ndarray,
                 model: ThreeDOFModel) -> np.ndarray:
    """HOFA forward: compute world-frame virtual acceleration from state and force.

    dd_eta = f(x) + G(x) * tau
    where f(x) = Jdot * nu + J * M_inv * (-C*nu - D*nu)
          G(x) = J * M_inv

    Args:
        state: [x, y, psi, u, v, r]
        tau: body-frame generalized force [X, Y, N]
        model: vehicle dynamics model

    Returns:
        World-frame acceleration [ddx, ddy, ddpsi]
    """
    psi = state[2]
    nu = state[3:]

    J = kinematic_matrix(psi)
    Jdot = kinematic_matrix_dot(psi, nu[2])
    M_inv = model.M_inv
    C = model.coriolis(nu)
    D = model.drag(nu)

    f_drift = Jdot @ nu + J @ M_inv @ (-C @ nu - D @ nu)
    G = J @ M_inv

    return f_drift + G @ tau


def hofa_inverse(state: np.ndarray, a_c: np.ndarray,
                 model: ThreeDOFModel) -> np.ndarray:
    """HOFA inverse: compute body-frame generalized force from virtual acceleration.

    tau_d = M * J^T * (a_c - Jdot * nu) + C(nu)*nu + D(nu)*nu

    Args:
        state: [x, y, psi, u, v, r]
        a_c: desired world-frame acceleration [ddx_c, ddy_c, ddpsi_c]
        model: vehicle dynamics model

    Returns:
        Body-frame generalized force [X, Y, N]
    """
    psi = state[2]
    nu = state[3:]

    J = kinematic_matrix(psi)
    Jdot = kinematic_matrix_dot(psi, nu[2])
    M = model.mass_matrix()
    C = model.coriolis(nu)
    D = model.drag(nu)

    tau_d = M @ J.T @ (a_c - Jdot @ nu) + C @ nu + D @ nu
    return tau_d


def hofa_forward_inverse_identity(state: np.ndarray, a_c: np.ndarray,
                                  model: ThreeDOFModel) -> float:
    """Verify HOFA forward/inverse identity: forward(inverse(a_c)) should ≈ a_c.

    Returns the max absolute error across all 3 axes.
    """
    tau_d = hofa_inverse(state, a_c, model)
    a_recovered = hofa_forward(state, tau_d, model)
    return float(np.max(np.abs(a_recovered - a_c)))
