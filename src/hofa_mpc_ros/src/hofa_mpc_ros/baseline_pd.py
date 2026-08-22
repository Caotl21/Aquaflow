"""Computed-force PD baseline controller for fair comparison with MPC.

Implements:
    a_c = dd_eta_r - Kp * e_eta - Kd * e_dot_eta

Then applies HOFA inverse to get body-frame generalized force.
"""
import numpy as np
from .types import VehicleState, ReferencePoint, MPCSolution, VirtualInputBounds
from .model import ThreeDOFModel
from .hofa import hofa_inverse, wrap_to_pi
from .mpc import HofaMPC


class ComputedForcePD:
    """Three-DOF computed-force PD controller.

    Uses the same HOFA model, reference feedforward, and thruster
    constraints as the MPC controller for a fair comparison.
    """

    def __init__(self, kp: np.ndarray = None, kd: np.ndarray = None,
                 model: ThreeDOFModel = None):
        self.kp = kp if kp is not None else np.diag([8.0, 8.0, 4.0])
        self.kd = kd if kd is not None else np.diag([5.0, 5.0, 2.0])
        self.model = model

    def solve(self, state: VehicleState, refs: list,
              bounds: VirtualInputBounds = None,
              **kwargs) -> MPCSolution:
        """Compute control action using computed-force PD.

        Args:
            state: current vehicle state
            refs: list of ReferencePoint, uses only refs[0]
            bounds: virtual input bounds (for monitoring, not used in control)

        Returns:
            MPCSolution with the PD-computed control
        """
        ref = refs[0]

        # Position error (shortest angle for yaw)
        ex = state.x - ref.x
        ey = state.y - ref.y
        epsi = wrap_to_pi(state.psi - ref.psi)

        # World-frame velocity
        c, s = np.cos(state.psi), np.sin(state.psi)
        dx_world = c * state.u - s * state.v
        dy_world = s * state.u + c * state.v

        # Rate error
        edx = dx_world - ref.dx
        edy = dy_world - ref.dy
        edpsi = state.r - ref.dpsi

        e_eta = np.array([ex, ey, epsi])
        e_dot_eta = np.array([edx, edy, edpsi])

        # Computed-force control law
        dd_eta_r = ref.acceleration_array()
        a_c = dd_eta_r - self.kp @ e_eta - self.kd @ e_dot_eta

        # HOFA inverse to get body-frame force
        state_arr = state.to_array()
        if self.model is not None:
            wrench = hofa_inverse(state_arr, a_c, self.model)
        else:
            wrench = np.zeros(3)

        return MPCSolution(
            success=True,
            virtual_accel=a_c,
            wrench=wrench,
            predicted_path=np.zeros((14, 3)),
            objective=0.0,
            iterations=0,
            bounds=bounds or VirtualInputBounds(),
        )
