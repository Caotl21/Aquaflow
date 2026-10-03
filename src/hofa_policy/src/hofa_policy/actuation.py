"""Bounded latent-thruster parameterization for policy outputs."""
from dataclasses import dataclass

import numpy as np

from .contract import bricsbot_horizontal_actuation


@dataclass
class ThrusterGeometry:
    allocation_matrix: np.ndarray
    force_min: np.ndarray
    force_max: np.ndarray

    def __post_init__(self):
        self.allocation_matrix = np.asarray(
            self.allocation_matrix, dtype=np.float64)
        self.force_min = np.asarray(self.force_min, dtype=np.float64)
        self.force_max = np.asarray(self.force_max, dtype=np.float64)
        if self.allocation_matrix.ndim != 2:
            raise ValueError("allocation_matrix must be two-dimensional")
        if self.allocation_matrix.shape[0] != 3:
            raise ValueError("allocation_matrix must map to [Fx,Fy,Nz]")
        n = self.allocation_matrix.shape[1]
        if self.force_min.shape != (n,) or self.force_max.shape != (n,):
            raise ValueError("force limits must match thruster count")
        if np.any(self.force_min >= self.force_max):
            raise ValueError("each force_min must be below force_max")
        if np.linalg.matrix_rank(self.allocation_matrix) < 3:
            raise ValueError("allocation matrix must span planar 3-DOF")

    @classmethod
    def bricsbot(cls):
        data = bricsbot_horizontal_actuation()
        return cls(data["allocation_matrix"],
                   data["force_min"], data["force_max"])

    @property
    def force_mid(self):
        return 0.5 * (self.force_min + self.force_max)

    @property
    def force_half(self):
        return 0.5 * (self.force_max - self.force_min)

    @property
    def wrench_center(self):
        return self.allocation_matrix @ self.force_mid

    @property
    def wrench_scale(self):
        half = np.sum(
            np.abs(self.allocation_matrix) * self.force_half, axis=1)
        return np.maximum(np.abs(self.wrench_center) + half, 1.0e-6)

    def force_from_normalized(self, normalized):
        normalized = np.asarray(normalized, dtype=np.float64)
        if normalized.shape[-1] != self.force_min.size:
            raise ValueError("normalized action has wrong thruster dimension")
        bounded = np.clip(normalized, -1.0, 1.0)
        return self.force_mid + self.force_half * bounded

    def normalized_from_force(self, force):
        force = np.asarray(force, dtype=np.float64)
        return np.clip((force - self.force_mid) / self.force_half, -1.0, 1.0)

    def wrench_from_force(self, force):
        force = np.asarray(force, dtype=np.float64)
        return np.einsum("ij,...j->...i", self.allocation_matrix, force)

    def wrench_from_normalized(self, normalized):
        return self.wrench_from_force(self.force_from_normalized(normalized))
