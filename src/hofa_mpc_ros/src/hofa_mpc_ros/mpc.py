"""Three-DOF HOFA-MPC solver (Layer 2).

Minimizes tracking error over a prediction horizon subject to
virtual input bounds from Layer 1.
"""
import numpy as np
from scipy.optimize import minimize
from .types import (MPCParams, VehicleParams, VehicleState,
                    ReferencePoint, MPCSolution, VirtualInputBounds)
from .model import ThreeDOFModel
from .hofa import wrap_to_pi
from .constraints import SafeInnerBoxStrategy


class HofaMPC:
    """3-DOF error MPC with HOFA virtual input constraints."""

    def __init__(self, mpc_params: MPCParams, vehicle_params: VehicleParams):
        self.params = mpc_params
        self.model = ThreeDOFModel(vehicle_params)
        self.Np = mpc_params.horizon
        self.dt = 1.0 / mpc_params.control_rate_hz

        # Prediction matrices (linear error model)
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        self.Ad = np.block([
            [I3, self.dt * I3],
            [Z3, I3]
        ])
        self.Bd = np.block([
            [0.5 * self.dt**2 * I3],
            [self.dt * I3]
        ])

        # Cost weights
        self.Q = np.diag(np.concatenate([
            mpc_params.weight_pose,
            mpc_params.weight_pose_rate
        ]))
        self.R = np.diag(mpc_params.weight_virtual_input)
        self.S = np.diag(mpc_params.weight_input_increment)

        # Terminal cost multiplier
        self.F = self.Q * mpc_params.terminal_multiplier

        # Constraint strategy
        self.constraint_strategy = SafeInnerBoxStrategy()

        # Warm start
        self._prev_w = np.zeros((self.Np, 3))

    def reset(self):
        """Clear warm start and internal state."""
        self._prev_w = np.zeros((self.Np, 3))

    def compute_error_state(self, state: VehicleState,
                            refs: list) -> np.ndarray:
        """Compute MPC error state z = [e_eta; e_dot_eta].

        Args:
            state: current vehicle state
            refs: list of ReferencePoint, length >= Np

        Returns:
            z: 6-element error state vector
        """
        # Position error
        ex = state.x - refs[0].x
        ey = state.y - refs[0].y
        epsi = wrap_to_pi(state.psi - refs[0].psi)

        # World velocity of current state
        c, s = np.cos(state.psi), np.sin(state.psi)
        dx_world = c * state.u - s * state.v
        dy_world = s * state.u + c * state.v

        # Rate error
        edx = dx_world - refs[0].dx
        edy = dy_world - refs[0].dy
        edpsi = state.r - refs[0].dpsi

        return np.array([ex, ey, epsi, edx, edy, edpsi])

    def solve(self, state: VehicleState, refs: list,
              bounds: VirtualInputBounds = None,
              f_min: np.ndarray = None,
              f_max: np.ndarray = None,
              prev_forces: np.ndarray = None
              ) -> MPCSolution:
        """Solve one MPC cycle.

        Args:
            state: current vehicle state
            refs: list of Np ReferencePoint objects
            bounds: virtual input bounds from Layer 1 (per-step)
            f_min, f_max: thruster force limits
            prev_forces: previous thruster allocation for rate limits

        Returns:
            MPCSolution with the optimal first virtual acceleration
        """
        Np = self.Np
        n_var = Np * 3  # decision variables: w_{0..Np-1}

        # Initial error state
        z0 = self.compute_error_state(state, refs)

        # Precompute reference accelerations
        ref_accels = np.array([refs[i].acceleration_array()
                               for i in range(Np)])

        # Warm start from previous solution
        w0 = self._prev_w.flatten()

        # ``bounds`` are bounds on total virtual acceleration.  The decision
        # variable is the tracking correction w, so shift each stage by the
        # corresponding reference acceleration.
        if bounds is not None:
            lb_bounds = np.concatenate([
                bounds.lower - refs[i].acceleration_array() for i in range(Np)])
            ub_bounds = np.concatenate([
                bounds.upper - refs[i].acceleration_array() for i in range(Np)])
        else:
            lb_bounds = np.full(n_var, -10.0)
            ub_bounds = np.full(n_var, 10.0)

        var_bounds = list(zip(lb_bounds, ub_bounds))

        # Cost function
        def cost_fn(w_flat):
            w_seq = w_flat.reshape(Np, 3)
            z = z0.copy()
            J_total = 0.0

            for i in range(Np):
                # State cost
                if i < Np - 1:
                    J_total += float(z @ self.Q @ z)
                else:
                    J_total += float(z @ self.F @ z)

                # Input cost
                J_total += float(w_seq[i] @ self.R @ w_seq[i])

                # Input increment cost
                if i == 0:
                    dw = w_seq[i]
                else:
                    dw = w_seq[i] - w_seq[i - 1]
                J_total += float(dw @ self.S @ dw)

                # Error dynamics for a moving reference.  Omitting this term
                # turns the controller into a point regulator even when the
                # local planner supplies a full trajectory window.
                if i < Np - 1:
                    ref_delta = np.concatenate([
                        refs[i].pose_array() - refs[i + 1].pose_array(),
                        refs[i].velocity_array() - refs[i + 1].velocity_array()])
                else:
                    ref_delta = np.zeros(6)
                z = self.Ad @ z + self.Bd @ w_seq[i]
                z += ref_delta

            return J_total

        # Gradient via finite differences (reliable for scipy)
        def cost_grad(w_flat):
            grad = np.zeros_like(w_flat)
            eps = 1e-6
            f0 = cost_fn(w_flat)
            for j in range(n_var):
                w_pert = w_flat.copy()
                w_pert[j] += eps
                grad[j] = (cost_fn(w_pert) - f0) / eps
            return grad

        # Solve
        try:
            result = minimize(
                cost_fn, w0, jac=cost_grad, method='L-BFGS-B',
                bounds=var_bounds,
                options={
                    'maxiter': self.params.max_iterations,
                    'ftol': self.params.ftol,
                    'gtol': self.params.gtol,
                }
            )
            # Treat as success if scipy converged or objective is negligible
            success = result.success or result.fun < 1e-20
            w_opt = result.x.reshape(Np, 3)
            objective = float(result.fun)
            iterations = result.nit
        except Exception:
            success = False
            w_opt = self._prev_w.copy()
            objective = float('inf')
            iterations = 0

        # Update warm start
        if success:
            self._prev_w = w_opt.copy()

        # Extract first control action
        dd_eta_r = refs[0].acceleration_array()
        a_c = dd_eta_r + w_opt[0]

        # Predicted path for visualization
        predicted_path = np.zeros((Np, 3))
        z_pred = z0.copy()
        for i in range(Np):
            z_pred = self.Ad @ z_pred + self.Bd @ w_opt[i]
            if i < Np - 1:
                z_pred += np.concatenate([
                    refs[i].pose_array() - refs[i + 1].pose_array(),
                    refs[i].velocity_array() - refs[i + 1].velocity_array()])
            predicted_path[i] = z_pred[:3] + refs[min(i + 1, Np - 1)].pose_array()

        return MPCSolution(
            success=success,
            virtual_accel=a_c,
            predicted_path=predicted_path,
            objective=objective,
            iterations=iterations,
            bounds=bounds or VirtualInputBounds(),
        )

    def predict_states(self, z0: np.ndarray,
                       w_seq: np.ndarray) -> np.ndarray:
        """Predict error states over the horizon.

        Returns:
            (Np+1, 6) array of predicted error states
        """
        states = np.zeros((self.Np + 1, 6))
        states[0] = z0
        for i in range(self.Np):
            states[i + 1] = self.Ad @ states[i] + self.Bd @ w_seq[i]
        return states
