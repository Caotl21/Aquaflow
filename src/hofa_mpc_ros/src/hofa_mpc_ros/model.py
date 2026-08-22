"""Three-DOF planar underwater vehicle dynamics model."""
import numpy as np
from .types import VehicleParams


class ThreeDOFModel:
    """Planar 3-DOF model: position (x, y, psi) and body velocity (u, v, r).

    Dynamics:
        M * dnu/dt + C(nu)*nu + D(nu)*nu = tau + d
        d_eta/dt = J(psi) * nu
    """

    def __init__(self, params: VehicleParams):
        self.M = params.mass_matrix.copy()
        self.M_inv = np.linalg.inv(self.M)
        self.dl = params.drag_linear.copy()
        self.dq = params.drag_quadratic.copy()
        self.coriolis_enabled = params.coriolis_enabled

    def mass_matrix(self) -> np.ndarray:
        return self.M

    def coriolis(self, nu: np.ndarray) -> np.ndarray:
        """Coriolis matrix C(nu). Defaults to zero; override for full model."""
        if not self.coriolis_enabled:
            return np.zeros((3, 3))
        # Standard skew-symmetric form for 3-DOF
        u, v, r = nu
        return np.array([
            [0, 0, -self.M[1, 1] * v - self.M[1, 2] * r],
            [0, 0, self.M[0, 0] * u + self.M[0, 2] * r],
            [self.M[1, 1] * v + self.M[1, 2] * r,
             -self.M[0, 0] * u - self.M[0, 2] * r, 0]
        ])

    def drag(self, nu: np.ndarray) -> np.ndarray:
        """Diagonal drag matrix D(nu) with linear + quadratic terms."""
        d = np.zeros(3)
        for i in range(3):
            d[i] = self.dl[i] + self.dq[i] * abs(nu[i])
        return np.diag(d)

    def drag_force(self, nu: np.ndarray) -> np.ndarray:
        """Drag force vector D(nu)*nu."""
        D = self.drag(nu)
        return D @ nu

    def kinematics(self, psi: float) -> np.ndarray:
        """Rotation matrix J(psi) from body to world frame."""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def kinematics_dot(self, psi: float, r: float) -> np.ndarray:
        """Time derivative of J(psi)."""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [-r * s, -r * c, 0],
            [r * c, -r * s, 0],
            [0, 0, 0]
        ])

    def body_velocity(self, psi: float, world_vel: np.ndarray) -> np.ndarray:
        """Convert world-frame velocity to body-frame: nu = J^T * d_eta."""
        J = self.kinematics(psi)
        return J.T @ world_vel

    def world_velocity(self, psi: float, nu: np.ndarray) -> np.ndarray:
        """Convert body-frame velocity to world-frame: d_eta = J * nu."""
        J = self.kinematics(psi)
        return J @ nu

    def state_derivative(self, nu: np.ndarray, tau: np.ndarray,
                         d: np.ndarray = None) -> np.ndarray:
        """Compute d(nu)/dt = M_inv * (tau - C(nu)*nu - D(nu)*nu + d)."""
        if d is None:
            d = np.zeros(3)
        C = self.coriolis(nu)
        D = self.drag(nu)
        return self.M_inv @ (tau - C @ nu - D @ nu + d)

    def full_state_derivative(self, state: np.ndarray, tau: np.ndarray,
                              d: np.ndarray = None) -> np.ndarray:
        """Full 6-state derivative: d/dt [eta; nu] = [J*nu; M_inv*(tau-C*nu-D*nu)]."""
        eta = state[:3]
        nu = state[3:]
        psi = eta[2]
        J = self.kinematics(psi)
        deta = J @ nu
        dnu = self.state_derivative(nu, tau, d)
        return np.concatenate([deta, dnu])

    def rk4_step(self, state: np.ndarray, tau: np.ndarray,
                 dt: float, d: np.ndarray = None) -> np.ndarray:
        """4th-order Runge-Kutta integration step."""
        k1 = self.full_state_derivative(state, tau, d)
        k2 = self.full_state_derivative(state + 0.5 * dt * k1, tau, d)
        k3 = self.full_state_derivative(state + 0.5 * dt * k2, tau, d)
        k4 = self.full_state_derivative(state + dt * k3, tau, d)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Wrap angle
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        return new_state
