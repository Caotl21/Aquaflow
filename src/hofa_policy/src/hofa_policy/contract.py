"""Stable input, output, and actuator contract for hofa_policy."""
from dataclasses import asdict, dataclass

import numpy as np


SCHEMA = "hofa_policy_v1"
REFERENCE_FIELDS = (
    "x_body", "y_body", "sin_delta_yaw", "cos_delta_yaw",
    "vx_body", "vy_body", "dpsi",
    "ax_body", "ay_body", "ddpsi",
)
STATE_FIELDS = ("u", "v", "r")
WRENCH_FIELDS = ("fx", "fy", "nz")


@dataclass(frozen=True)
class PolicyContract:
    schema: str = SCHEMA
    horizon: int = 20
    reference_features: int = len(REFERENCE_FIELDS)
    state_features: int = len(STATE_FIELDS)
    previous_wrench_features: int = len(WRENCH_FIELDS)
    action_features: int = 4
    reference_dt_s: float = 0.1
    frame: str = "NED_FRD"

    def to_dict(self):
        result = asdict(self)
        result["reference_fields"] = list(REFERENCE_FIELDS)
        result["state_fields"] = list(STATE_FIELDS)
        result["wrench_fields"] = list(WRENCH_FIELDS)
        return result


CONTRACT = PolicyContract()


def bricsbot_horizontal_actuation():
    """Return the BricsBot T1-T4 geometry in body FRD coordinates."""
    positions = np.asarray([
        [0.12755, -0.12755],
        [0.12755, 0.12755],
        [-0.12755, -0.12755],
        [-0.12755, 0.12755],
    ], dtype=np.float64)
    directions = np.asarray([
        [0.7071, 0.7071],
        [0.7071, -0.7071],
        [-0.7071, 0.7071],
        [-0.7071, -0.7071],
    ], dtype=np.float64)
    allocation = np.zeros((3, 4), dtype=np.float64)
    allocation[0] = directions[:, 0]
    allocation[1] = directions[:, 1]
    allocation[2] = (
        positions[:, 0] * directions[:, 1]
        - positions[:, 1] * directions[:, 0])
    return {
        "positions_xy": positions,
        "directions_xy": directions,
        "allocation_matrix": allocation,
        "force_min": np.full(4, -58.84, dtype=np.float64),
        "force_max": np.full(4, 67.67, dtype=np.float64),
    }
