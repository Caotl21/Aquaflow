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

    def _reference_affine_terms(self, refs):
        """Build the affine terms for the time-indexed error dynamics.

        With ``z = state - reference`` and ``z[i+1] = Ad*z[i] + Bd*w[i] + d[i]``,
        ``d[i]`` is ``Ad*reference[i] - reference[i+1]``.  The yaw component is
        wrapped only after including the expected ``dt*dpsi`` increment.
        """
        affine = np.zeros((self.Np - 1, 6))
        for i in range(self.Np - 1):
            ref_i = np.concatenate([
                refs[i].pose_array(), refs[i].velocity_array()])
            ref_next = np.concatenate([
                refs[i + 1].pose_array(), refs[i + 1].velocity_array()])
            affine[i] = self.Ad @ ref_i - ref_next
            affine[i, 2] = wrap_to_pi(
                self.dt * refs[i].dpsi + refs[i].psi - refs[i + 1].psi)
        return affine

    def _cost_and_grad(self, w_flat, z0, ref_affine):
        """Return the exact cost and gradient for one MPC decision vector.

        The gradient uses one forward rollout and one reverse adjoint pass,
        replacing the previous O(n_var) finite-difference cost evaluations.
        The cost ordering intentionally matches the existing SciPy objective.
        """
        w_seq = np.asarray(w_flat, dtype=float).reshape(self.Np, 3)
        z_seq = np.zeros((self.Np + 1, 6))
        z_seq[0] = z0
        for i in range(self.Np):
            z_seq[i + 1] = self.Ad @ z_seq[i] + self.Bd @ w_seq[i]
            if i < self.Np - 1:
                z_seq[i + 1] += ref_affine[i]

        total = 0.0
        for i in range(self.Np):
            Q_i = self.F if i == self.Np - 1 else self.Q
            total += float(z_seq[i] @ Q_i @ z_seq[i])
            total += float(w_seq[i] @ self.R @ w_seq[i])
            dw = w_seq[i] if i == 0 else w_seq[i] - w_seq[i - 1]
            total += float(dw @ self.S @ dw)

        # Reverse-mode derivative through z[i+1] = Ad*z[i] + Bd*w[i] + d[i].
        # The existing objective charges z[i], not z[Np], hence lambda[Np]=0.
        lam = np.zeros((self.Np + 1, 6))
        grad = np.zeros((self.Np, 3))
        for i in range(self.Np - 1, -1, -1):
            Q_i = self.F if i == self.Np - 1 else self.Q
            grad[i] = 2.0 * self.R @ w_seq[i] + self.Bd.T @ lam[i + 1]
            lam[i] = 2.0 * Q_i @ z_seq[i] + self.Ad.T @ lam[i + 1]

        # Derivative of the input-increment regularizer.
        for i in range(self.Np):
            dw = w_seq[i] if i == 0 else w_seq[i] - w_seq[i - 1]
            grad[i] += 2.0 * self.S @ dw
            if i > 0:
                grad[i - 1] -= 2.0 * self.S @ dw

        return total, grad.reshape(-1)

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

        # Precompute the affine terms induced by the moving, time-indexed
        # reference.  They are independent of the optimization variables.
        ref_affine = self._reference_affine_terms(refs)

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
            return self._cost_and_grad(w_flat, z0, ref_affine)[0]

        def cost_grad(w_flat):
            return self._cost_and_grad(w_flat, z0, ref_affine)[1]

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
                z_pred += ref_affine[i]
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
