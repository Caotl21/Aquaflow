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
        """Solve one LP for the largest centered axis-aligned inner box.

        The reachable acceleration set is the affine image of the thruster
        force box.  For a candidate acceleration box, convexity means that it
        is sufficient to provide one feasible thruster vector for each of its
        eight corners.  Solving for all corner witnesses and all three box
        half-widths in one LP replaces the previous sequence of independent
        corner-feasibility LPs and scale bisection.

        ``requested_scale`` caps each optimized half-width by that fraction of
        the reachable set's axis-aligned outer projection.  Keeping the
        existing one-third setting therefore preserves the previous nominal
        authority while allowing the three axes to shrink independently when
        coupling makes the full candidate box infeasible.  Setting it to one
        asks for the largest normalized inner box under this fixed center.
        """
        f_mid = 0.5 * (f_min + f_max)
        f_half = 0.5 * (f_max - f_min)
        center = f_drift + gain @ f_mid
        raw_half = np.sum(np.abs(gain) * f_half, axis=1)
        scale = float(np.clip(requested_scale, 0.0, 1.0))
        max_half = raw_half * scale

        n_accel = gain.shape[0]
        n_thrusters = gain.shape[1]
        corner_signs = np.asarray(
            list(product([-1.0, 1.0], repeat=n_accel)), dtype=float)
        n_corners = corner_signs.shape[0]

        # Variables are [half_widths, balanced_scale,
        # force_for_corner_0, ..., force_for_corner_7].  The balanced scale is
        # the smallest normalized half-width across active acceleration axes.
        # Maximizing it prevents the LP from sacrificing one axis completely
        # to enlarge the other two.  A small secondary weight expands any axes
        # that still have unused feasible authority.
        balance_index = n_accel
        n_variables = n_accel + 1 + n_corners * n_thrusters
        objective = np.zeros(n_variables)
        active = raw_half > 1.0e-12
        objective[balance_index] = -1.0
        objective[np.flatnonzero(active)] = -1.0e-4 / raw_half[active]

        A_eq = np.zeros((n_corners * n_accel, n_variables))
        b_eq = np.tile(gain @ f_mid, n_corners)
        for corner_index, signs in enumerate(corner_signs):
            row = slice(corner_index * n_accel,
                        (corner_index + 1) * n_accel)
            force_start = n_accel + 1 + corner_index * n_thrusters
            force_columns = slice(force_start, force_start + n_thrusters)
            # gain @ force = gain @ f_mid + signs * half
            A_eq[row, :n_accel] = -np.diag(signs)
            A_eq[row, force_columns] = gain

        active_indices = np.flatnonzero(active)
        A_ub = np.zeros((len(active_indices), n_variables))
        for row_index, accel_index in enumerate(active_indices):
            # raw_half[i] * balanced_scale <= half_width[i]
            A_ub[row_index, accel_index] = -1.0
            A_ub[row_index, balance_index] = raw_half[accel_index]
        b_ub = np.zeros(len(active_indices))

        variable_bounds = [(0.0, float(limit)) for limit in max_half]
        variable_bounds.append((0.0, scale))
        variable_bounds.extend(
            (float(lower), float(upper))
            for _ in range(n_corners)
            for lower, upper in zip(f_min, f_max))

        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=variable_bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                "maximum inner-box LP failed: %s" % result.message)

        # Clipping only removes solver tolerance overshoot.  Shrinking a
        # centered feasible box preserves containment in the convex reachable
        # set.
        half = np.clip(result.x[:n_accel], 0.0, max_half)
        return VirtualInputBounds(lower=center - half, upper=center + half)
