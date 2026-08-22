"""Safety supervisor logic for HOFA-MPC controller.

Monitors state timeouts, solver failures, NaN, boundary violations,
and produces override commands when unsafe.
"""
import math
import numpy as np
from .types import ControllerState, VehicleState


class SafetyParams:
    """Safety threshold parameters."""

    def __init__(self):
        self.state_timeout_s = 0.20
        self.reference_timeout_s = 0.30
        self.max_position_abs = np.array([4.2, 2.7])
        self.max_world_speed = 1.0
        self.max_yaw_rate = 1.2
        self.max_consecutive_solver_failures = 3
        self.max_saturation_duration_s = 2.0
        self.zero_on_time_jump = True


class SafetySupervisor:
    """Safety monitoring and override logic."""

    def __init__(self, params: SafetyParams = None):
        self.params = params or SafetyParams()
        self.state = ControllerState.DISABLED
        self.consecutive_failures = 0
        self.last_state_time = 0.0
        self.last_ref_time = 0.0
        self.last_valid_cmd = np.zeros(3)
        self.hold_cycles_remaining = 0

    def reset(self):
        """Reset supervisor state."""
        self.state = ControllerState.DISABLED
        self.consecutive_failures = 0
        self.hold_cycles_remaining = 0
        self.last_valid_cmd = np.zeros(3)

    def check_state(self, vehicle_state: VehicleState,
                    current_time: float,
                    ref_time: float = None) -> ControllerState:
        """Evaluate safety conditions and return desired controller state.

        Args:
            vehicle_state: latest vehicle state
            current_time: current timestamp
            ref_time: timestamp of latest reference

        Returns:
            ControllerState indicating safe operation mode
        """
        # Check for NaN
        state_arr = vehicle_state.to_array()
        if np.any(np.isnan(state_arr)) or np.any(np.isinf(state_arr)):
            self.consecutive_failures = 0
            return ControllerState.FAULT

        # Check state timeout
        if current_time - vehicle_state.timestamp > self.params.state_timeout_s:
            self.consecutive_failures = 0
            return ControllerState.WAITING_FOR_STATE

        # Check reference timeout
        if ref_time is not None:
            if current_time - ref_time > self.params.reference_timeout_s:
                return ControllerState.WAITING_FOR_REFERENCE

        # Check position bounds
        pos = np.array([vehicle_state.x, vehicle_state.y])
        if np.any(np.abs(pos) > self.params.max_position_abs):
            return ControllerState.FAULT

        # Check speed bounds
        speed = math.hypot(vehicle_state.u, vehicle_state.v)
        if speed > self.params.max_world_speed:
            return ControllerState.FAULT

        # Check yaw rate
        if abs(vehicle_state.r) > self.params.max_yaw_rate:
            return ControllerState.FAULT

        # All checks passed — report ACTIVE
        return ControllerState.ACTIVE

    def on_solver_result(self, success: bool, wrench: np.ndarray):
        """Update state based on solver result.

        Args:
            success: whether the solver converged
            wrench: the computed wrench command
        """
        if success:
            self.consecutive_failures = 0
            self.last_valid_cmd = wrench.copy()
            self.hold_cycles_remaining = 0
        else:
            self.consecutive_failures += 1

    def get_override_command(self, current_state: ControllerState,
                             ) -> tuple:
        """Determine the command to send based on safety state.

        Returns:
            (command, state) where command is the wrench to send
            and state is the effective controller state.
        """
        if current_state == ControllerState.FAULT:
            return np.zeros(3), ControllerState.FAULT

        if current_state == ControllerState.WAITING_FOR_STATE:
            return np.zeros(3), ControllerState.WAITING_FOR_STATE

        if current_state == ControllerState.WAITING_FOR_REFERENCE:
            return np.zeros(3), ControllerState.WAITING_FOR_REFERENCE

        if current_state == ControllerState.DISABLED:
            return np.zeros(3), ControllerState.DISABLED

        # Check consecutive failures
        if self.consecutive_failures >= self.params.max_consecutive_solver_failures:
            return np.zeros(3), ControllerState.FAULT

        # Degraded mode: hold last valid command
        if self.consecutive_failures > 0:
            self.hold_cycles_remaining = self.params.max_consecutive_solver_failures \
                - self.consecutive_failures
            return self.last_valid_cmd, ControllerState.DEGRADED

        return None, ControllerState.ACTIVE

    def validate_wrench(self, wrench: np.ndarray,
                        max_force: float = 100.0,
                        max_torque: float = 50.0) -> np.ndarray:
        """Numerical validity and hard限幅 check on wrench.

        Args:
            wrench: [X, Y, N] wrench command
            max_force: maximum force magnitude
            max_torque: maximum torque magnitude

        Returns:
            Validated and clipped wrench
        """
        if np.any(np.isnan(wrench)) or np.any(np.isinf(wrench)):
            return self.last_valid_cmd.copy()

        w = wrench.copy()
        w[0] = np.clip(w[0], -max_force, max_force)
        w[1] = np.clip(w[1], -max_force, max_force)
        w[2] = np.clip(w[2], -max_torque, max_torque)
        return w
