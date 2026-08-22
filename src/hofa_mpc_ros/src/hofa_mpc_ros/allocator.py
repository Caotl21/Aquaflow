"""Thrust allocation from generalized force to individual thrusters.

Given desired body-frame generalized force [X, Y, N], solve for
individual thruster forces f_h that minimize:
    ||W(B_h * f_h - tau)||^2 + lambda * ||f_h||^2
subject to f_min <= f_h <= f_max.
"""
import numpy as np
from typing import List, Tuple, Optional
from .types import ThrusterConfig


class ThrusterAllocator:
    """Allocates generalized force to N horizontal thrusters."""

    def __init__(self, thrusters: List[ThrusterConfig],
                 yaw_weight: float = 2.0, regularization: float = 1e-4):
        self.n_thrusters = len(thrusters)
        self.thrusters = thrusters
        self.yaw_weight = yaw_weight
        self.reg = regularization

        # Build allocation matrix B_h (3 x n_thrusters)
        self.Bh = self._build_allocation_matrix(thrusters)
        self.Bh_pinv = self._weighted_pseudoinverse(self.Bh, yaw_weight,
                                                     regularization)

    @staticmethod
    def _build_allocation_matrix(thrusters: List[ThrusterConfig]) -> np.ndarray:
        """Build B_h matrix: tau = B_h * f_h.

        Each column i: [dx_i, dy_i, r_i]
        where (dx_i, dy_i) is the thruster direction and
        r_i = rx_i * dy_i - ry_i * dx_i is the yaw moment arm.
        """
        n = len(thrusters)
        B = np.zeros((3, n))
        for i, t in enumerate(thrusters):
            dx, dy = t.direction[0], t.direction[1]
            rx, ry = t.position[0], t.position[1]
            B[0, i] = dx          # X force
            B[1, i] = dy          # Y force
            B[2, i] = rx * dy - ry * dx  # N moment
        return B

    @staticmethod
    def _weighted_pseudoinverse(B: np.ndarray, yaw_weight: float,
                                reg: float) -> np.ndarray:
        """Weighted least-squares pseudoinverse: B^T(BW^TB + reg*I)^{-1}W^T."""
        n = B.shape[1]
        W = np.diag([1.0, 1.0, yaw_weight])
        BtWt = B.T @ W.T
        gram = BtWt @ B + reg * np.eye(n)
        return np.linalg.solve(gram, BtWt)

    def allocate(self, tau: np.ndarray,
                 f_min: Optional[np.ndarray] = None,
                 f_max: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Allocate generalized force to thruster forces.

        Args:
            tau: desired generalized force [X, Y, N]
            f_min: per-thruster minimum force (default from config)
            f_max: per-thruster maximum force (default from config)

        Returns:
            (f_h, residual) where f_h is the allocated thrust vector
            and residual is ||B_h*f_h - tau||.
        """
        if f_min is None:
            f_min = np.array([t.thrust_min for t in self.thrusters])
        if f_max is None:
            f_max = np.array([t.thrust_max for t in self.thrusters])

        f_h = self.Bh_pinv @ tau

        # Clip to limits
        f_h_clipped = np.clip(f_h, f_min, f_max)

        residual = float(np.linalg.norm(self.Bh @ f_h_clipped - tau))
        return f_h_clipped, residual

    def is_reachable(self, tau: np.ndarray,
                     f_min: Optional[np.ndarray] = None,
                     f_max: Optional[np.ndarray] = None) -> bool:
        """Check if a generalized force is within the reachable set."""
        if f_min is None:
            f_min = np.array([t.thrust_min for t in self.thrusters])
        if f_max is None:
            f_max = np.array([t.thrust_max for t in self.thrusters])

        f_h, residual = self.allocate(tau, f_min, f_max)
        # If residual is large relative to tau magnitude, it's not reachable
        tau_norm = float(np.linalg.norm(tau))
        if tau_norm < 1e-10:
            return True
        return residual / tau_norm < 0.1  # 10% tolerance

    def allocation_matrix(self) -> np.ndarray:
        """Return the B_h allocation matrix."""
        return self.Bh.copy()

    def rank(self) -> int:
        """Rank of the allocation matrix."""
        return int(np.linalg.matrix_rank(self.Bh))
