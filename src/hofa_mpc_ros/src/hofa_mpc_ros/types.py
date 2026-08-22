"""Core data structures for HOFA-MPC."""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ControllerState(Enum):
    DISABLED = "DISABLED"
    WAITING_FOR_STATE = "WAITING_FOR_STATE"
    WAITING_FOR_REFERENCE = "WAITING_FOR_REFERENCE"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"


@dataclass
class VehicleState:
    """Planar 3-DOF vehicle state in ENU/FLU convention."""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0
    timestamp: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r])

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0) -> "VehicleState":
        return cls(x=arr[0], y=arr[1], psi=arr[2],
                   u=arr[3], v=arr[4], r=arr[5], timestamp=timestamp)


@dataclass
class ReferencePoint:
    """Reference trajectory point with pose, velocity, and acceleration."""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dpsi: float = 0.0
    ddx: float = 0.0
    ddy: float = 0.0
    ddpsi: float = 0.0

    def pose_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi])

    def velocity_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dpsi])

    def acceleration_array(self) -> np.ndarray:
        return np.array([self.ddx, self.ddy, self.ddpsi])


@dataclass
class MPCInput:
    """Virtual acceleration command output from MPC."""
    ddx_c: float = 0.0
    ddy_c: float = 0.0
    ddpsi_c: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.ddx_c, self.ddy_c, self.ddpsi_c])


@dataclass
class GeneralizedForce:
    """Body-frame generalized force (X, Y, N)."""
    X: float = 0.0
    Y: float = 0.0
    N: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.X, self.Y, self.N])


@dataclass
class VirtualInputBounds:
    """Per-axis bounds on virtual acceleration."""
    lower: np.ndarray = field(default_factory=lambda: np.zeros(3))
    upper: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class MPCSolution:
    """Result from one MPC solve cycle."""
    success: bool = False
    virtual_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(3))
    predicted_path: np.ndarray = field(default_factory=lambda: np.zeros((14, 3)))
    objective: float = 0.0
    iterations: int = 0
    layer1_time_ms: float = 0.0
    layer2_time_ms: float = 0.0
    bounds: VirtualInputBounds = field(default_factory=VirtualInputBounds)


@dataclass
class ThrusterConfig:
    """Configuration for a single thruster."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    thrust_max: float = 1.0
    thrust_min: float = -1.0
    time_constant: float = 0.1
    deadzone: float = 0.0
    rate_limit: float = float("inf")


@dataclass
class VehicleParams:
    """Vehicle dynamics parameters."""
    mass_matrix: np.ndarray = field(
        default_factory=lambda: np.diag([7.94, 7.94, 0.15]))
    drag_linear: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 10.0, 1.4]))
    drag_quadratic: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 15.0, 0.35]))
    coriolis_enabled: bool = False


@dataclass
class MPCParams:
    """MPC controller parameters."""
    control_rate_hz: float = 10.0
    horizon: int = 14
    weight_pose: np.ndarray = field(
        default_factory=lambda: np.array([55.0, 55.0, 32.0]))
    weight_pose_rate: np.ndarray = field(
        default_factory=lambda: np.array([7.0, 7.0, 5.5]))
    weight_virtual_input: np.ndarray = field(
        default_factory=lambda: np.array([0.20, 0.20, 0.14]))
    weight_input_increment: np.ndarray = field(
        default_factory=lambda: np.array([1.20, 1.20, 0.75]))
    terminal_multiplier: float = 4.0
    safe_box_scale: float = 1.0 / 3.0
    max_iterations: int = 90
    ftol: float = 1e-10
    gtol: float = 1e-7
    deadline_ms: float = 80.0
    warm_start: bool = True
    hold_last_cycles: int = 1
    max_consecutive_failures: int = 3
