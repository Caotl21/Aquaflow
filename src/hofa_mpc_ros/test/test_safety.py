"""Unit tests for the safety supervisor state machine.

The regression these guard against: a single solver failure used to make
DEGRADED an absorbing state.  ``get_override_command`` returned a non-None
command, the controller published it and returned before ever reaching the
solver, and only a completed solve calls ``on_solver_result``.  So
``consecutive_failures`` stayed pinned at 1 — never cleared, never escalated
to FAULT — and the vehicle held a zero ``last_valid_cmd`` indefinitely.
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.safety import SafetySupervisor, SafetyParams
from hofa_mpc_ros.types import ControllerState, VehicleState


@pytest.fixture
def supervisor():
    return SafetySupervisor(SafetyParams())


def healthy_state(timestamp=100.0):
    """A state that passes every check in :meth:`check_state`."""
    return VehicleState(x=0.0, y=0.0, psi=0.0, u=0.1, v=0.0, r=0.0,
                        timestamp=timestamp)


class TestDegradedDoesNotBlockSolver:
    def test_pre_solve_yields_after_one_failure(self, supervisor):
        supervisor.on_solver_result(False, np.zeros(3))
        assert supervisor.consecutive_failures == 1

        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)
        # None means "caller proceeds to the solve".  Anything else and the
        # failure counter can never be cleared.
        assert command is None
        assert state == ControllerState.DEGRADED

    def test_post_failure_still_returns_fallback(self, supervisor):
        supervisor.on_solver_result(False, np.zeros(3))

        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE)
        assert command is not None
        assert state == ControllerState.DEGRADED

    def test_recovers_after_a_successful_solve(self, supervisor):
        supervisor.on_solver_result(False, np.zeros(3))
        assert supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)[0] is None

        wrench = np.array([1.5, -0.5, 0.25])
        supervisor.on_solver_result(True, wrench)

        assert supervisor.consecutive_failures == 0
        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)
        assert command is None
        assert state == ControllerState.ACTIVE
        np.testing.assert_allclose(supervisor.last_valid_cmd, wrench)

    def test_escalates_to_fault_after_the_configured_failures(self, supervisor):
        limit = supervisor.params.max_consecutive_solver_failures
        for _ in range(limit):
            supervisor.on_solver_result(False, np.zeros(3))

        # Once the limit is reached the solve must be pre-empted, even though
        # the same call yielded to it while merely degraded.
        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)
        assert state == ControllerState.FAULT
        np.testing.assert_allclose(command, np.zeros(3))

    def test_degraded_cycles_alone_never_escalate(self, supervisor):
        """Repeated degraded cycles without a solve must not drift.

        This is the shape of the original bug: the controller looped here
        forever, so neither recovery nor escalation was reachable.
        """
        supervisor.on_solver_result(False, np.zeros(3))
        for _ in range(50):
            command, state = supervisor.get_override_command(
                ControllerState.ACTIVE, pre_solve=True)
            assert command is None
            assert state == ControllerState.DEGRADED
        assert supervisor.consecutive_failures == 1


class TestHardStatesAlwaysPreempt:
    @pytest.mark.parametrize("hard_state", [
        ControllerState.FAULT,
        ControllerState.WAITING_FOR_STATE,
        ControllerState.WAITING_FOR_REFERENCE,
        ControllerState.DISABLED,
    ])
    def test_hard_state_returns_zero_regardless_of_pre_solve(
            self, supervisor, hard_state):
        for pre_solve in (False, True):
            command, state = supervisor.get_override_command(
                hard_state, pre_solve=pre_solve)
            assert state == hard_state
            np.testing.assert_allclose(command, np.zeros(3))

    def test_healthy_active_yields_to_solver(self, supervisor):
        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)
        assert command is None
        assert state == ControllerState.ACTIVE


class TestCheckState:
    def test_healthy_state_is_active(self, supervisor):
        state = supervisor.check_state(healthy_state(100.0), 100.0, 100.0)
        assert state == ControllerState.ACTIVE

    def test_stale_state_waits(self, supervisor):
        now = 100.0 + supervisor.params.state_timeout_s + 0.01
        assert supervisor.check_state(healthy_state(100.0), now, now) == \
            ControllerState.WAITING_FOR_STATE

    def test_stale_reference_waits(self, supervisor):
        now = 100.0
        ref_time = now - supervisor.params.reference_timeout_s - 0.01
        assert supervisor.check_state(healthy_state(now), now, ref_time) == \
            ControllerState.WAITING_FOR_REFERENCE

    def test_nan_state_faults(self, supervisor):
        bad = healthy_state(100.0)
        bad.x = float("nan")
        assert supervisor.check_state(bad, 100.0, 100.0) == ControllerState.FAULT

    def test_spawn_position_is_within_configured_bounds(self, supervisor):
        """BricsBot spawns at NED (-5, 0), i.e. ENU (0, -5).

        The controller loads max_position_abs_m from safety.yaml, but a
        SafetyParams built with its own defaults must not fault there either,
        or a default-constructed supervisor disagrees with the deployed one.
        """
        supervisor.params.max_position_abs = np.array([14.0, 8.0])
        state = healthy_state(100.0)
        state.x, state.y = 0.0, -5.0
        assert supervisor.check_state(state, 100.0, 100.0) == \
            ControllerState.ACTIVE


class TestReset:
    def test_reset_clears_failures(self, supervisor):
        supervisor.on_solver_result(False, np.zeros(3))
        supervisor.reset()
        assert supervisor.consecutive_failures == 0
        command, state = supervisor.get_override_command(
            ControllerState.ACTIVE, pre_solve=True)
        assert command is None
        assert state == ControllerState.ACTIVE
