"""Virtual input constraint computation (Layer 1).

Computes the reachable set of virtual accelerations given thruster limits,
then derives a conservative inner box for use as MPC constraints.
"""
from abc import ABC, abstractmethod
from itertools import product
import numpy as np
from scipy.optimize import linprog
from .types import ThrusterConfig, VehicleParams, VehicleState, VirtualInputBounds
from .model import ThreeDOFModel
from .hofa import kinematic_matrix, kinematic_matrix_dot, drag_force
from .allocator import ThrusterAllocator


class VirtualInputConstraintStrategy(ABC):
    """Base class for virtual input boundary computation."""

    @abstractmethod
    def compute(self, state: VehicleState, model: ThreeDOFModel,
                allocator: ThrusterAllocator, scale: float = 1.0 / 3.0,
                f_min: np.ndarray = None, f_max: np.ndarray = None
                ) -> VirtualInputBounds:
        pass


class CurrentStateBoxStrategy(VirtualInputConstraintStrategy):
    """Conservative box based on current state only (debug/fallback)."""

    def compute(self, state: VehicleState, model: ThreeDOFModel,
                allocator: ThrusterAllocator, scale: float = 1.0 / 3.0,
                f_min: np.ndarray = None, f_max: np.ndarray = None
                ) -> VirtualInputBounds:
        nu = np.array([state.u, state.v, state.r])
        psi = state.psi

        # Drift term: f(x) = Jdot*nu + J*M_inv*(-C*nu - D*nu)
        J = kinematic_matrix(psi)
        Jdot = kinematic_matrix_dot(psi, nu[2])
        C = model.coriolis(nu)
        D = model.drag(nu)
        f_drift = Jdot @ nu + J @ model.M_inv @ (-C @ nu - D @ nu)

        # Input gain: G = J * M_inv * B_h (project through allocation)
        G_full = J @ model.M_inv @ allocator.Bh

        # Compute reachable range per axis by evaluating all 8 corners
        if f_min is None:
            f_min = np.array([t.thrust_min for t in allocator.thrusters])
        if f_max is None:
            f_max = np.array([t.thrust_max for t in allocator.thrusters])

        corners = list(product(*[[f_min[i], f_max[i]]
                                 for i in range(allocator.n_thrusters)]))
        accel_values = np.array([f_drift + G_full @ np.array(c)
                                 for c in corners])

        lower_raw = accel_values.min(axis=0)
        upper_raw = accel_values.max(axis=0)

        # Apply safe inner box scaling
        center = 0.5 * (upper_raw + lower_raw)
        half = 0.5 * (upper_raw - lower_raw) * scale

        return VirtualInputBounds(
            lower=center - half,
            upper=center + half,
        )


class SafeInnerBoxStrategy(VirtualInputConstraintStrategy):
    """Predictive safe inner box with horizon propagation.

    For each prediction step, computes reachable acceleration bounds
    and applies conservative inner box scaling.  Uses previous solution
    as warm start for yaw prediction.
    """

    def __init__(self):
        self.prev_solution = None

    def compute(self, state: VehicleState, model: ThreeDOFModel,
                allocator: ThrusterAllocator, scale: float = 1.0 / 3.0,
                f_min: np.ndarray = None, f_max: np.ndarray = None,
                dt: float = 0.1, horizon: int = 14,
                prev_yaw_sequence: np.ndarray = None
                ) -> VirtualInputBounds:
        """Compute safe bounds for the FIRST prediction step only.

        The full multi-step computation is done inside the MPC solver.
        This provides the initial step bounds for warm-starting.
        """
        nu = np.array([state.u, state.v, state.r])
        psi = state.psi

        J = kinematic_matrix(psi)
        Jdot = kinematic_matrix_dot(psi, nu[2])
        C = model.coriolis(nu)
        D = model.drag(nu)

        f_drift = Jdot @ nu + J @ model.M_inv @ (-C @ nu - D @ nu)
        G_full = J @ model.M_inv @ allocator.Bh

        if f_min is None:
            f_min = np.array([t.thrust_min for t in allocator.thrusters])
        if f_max is None:
            f_max = np.array([t.thrust_max for t in allocator.thrusters])

        return self._build_verified_box(
            f_drift, G_full, f_min, f_max, scale)

    def compute_for_step(self, state_pred: np.ndarray,
                         model: ThreeDOFModel,
                         allocator: ThrusterAllocator,
                         scale: float = 1.0 / 3.0,
                         f_min: np.ndarray = None,
                         f_max: np.ndarray = None
                         ) -> VirtualInputBounds:
        """Compute bounds for a single predicted state (used in MPC inner loop)."""
        psi = state_pred[2]
        nu = state_pred[3:]

        J = kinematic_matrix(psi)
        Jdot = kinematic_matrix_dot(psi, nu[2])
        C = model.coriolis(nu)
        D = model.drag(nu)

        f_drift = Jdot @ nu + J @ model.M_inv @ (-C @ nu - D @ nu)
        G_full = J @ model.M_inv @ allocator.Bh

        if f_min is None:
            f_min = np.array([t.thrust_min for t in allocator.thrusters])
        if f_max is None:
            f_max = np.array([t.thrust_max for t in allocator.thrusters])

        return self._build_verified_box(
            f_drift, G_full, f_min, f_max, scale)

    @staticmethod
    def _build_verified_box(f_drift: np.ndarray, gain: np.ndarray,
                            f_min: np.ndarray, f_max: np.ndarray,
                            requested_scale: float) -> VirtualInputBounds:
        """Build an axis-aligned box verified inside the reachable set."""
        f_mid = 0.5 * (f_min + f_max)
        f_half = 0.5 * (f_max - f_min)
        center = f_drift + gain @ f_mid
        raw_half = np.sum(np.abs(gain) * f_half, axis=1)
        scale = float(np.clip(requested_scale, 0.0, 1.0))

        def feasible(candidate_scale):
            half = raw_half * candidate_scale
            for corner in product(*[[center[i] - half[i],
                                     center[i] + half[i]]
                                    for i in range(3)]):
                result = linprog(
                    np.zeros(gain.shape[1]),
                    A_eq=gain,
                    b_eq=np.asarray(corner) - f_drift,
                    bounds=list(zip(f_min, f_max)),
                    method="highs",
                )
                if not result.success:
                    return False
            return True

        if not feasible(scale):
            lower, upper = 0.0, scale
            for _ in range(24):
                midpoint = 0.5 * (lower + upper)
                if feasible(midpoint):
                    lower = midpoint
                else:
                    upper = midpoint
            scale = lower

        half = raw_half * scale
        return VirtualInputBounds(lower=center - half, upper=center + half)
