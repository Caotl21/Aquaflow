"""Explicit tracking scales and fixed physical residual regularization scales."""
import math
import numpy as np


def _smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def potential_values(config, reference_yaw_rate, yaw_rate_error,
                    terminal_yaw_error, remaining_arc_m, comparison=False):
    """Return smooth negative error potentials used by difference shaping."""
    objective = config.get("comparison_objective", {}) if comparison else config
    potential = objective.get("potential_reward", {})
    if potential.get("mode", "disabled") != "difference":
        return {"terminal_yaw": 0.0, "turn_yaw_rate": 0.0, "total": 0.0}
    arc_window = float(potential.get("terminal_arc_window_m", 1.0))
    yaw_scale = float(potential.get("terminal_yaw_scale_deg", 5.0))
    rate_scale = float(potential.get("turn_yaw_rate_scale_radps", 0.5))
    turn_start = float(potential.get("turn_gate_start_radps", 0.02))
    turn_full = float(potential.get("turn_gate_full_radps", 0.20))
    if (arc_window <= 0 or rate_scale <= 0 or not 0 < yaw_scale <= 180
            or turn_start < 0 or turn_full <= turn_start):
        raise ValueError("invalid potential reward scales or gates")
    end_gate = _smoothstep(1.0 - float(remaining_arc_m) / arc_window)
    turn_gate = _smoothstep(
        (abs(float(reference_yaw_rate)) - turn_start) / (turn_full - turn_start))
    yaw_denominator = 1.0 - math.cos(math.radians(yaw_scale))
    terminal_metric = (1.0 - math.cos(float(terminal_yaw_error))) / yaw_denominator
    rate_metric = min((float(yaw_rate_error) / rate_scale) ** 2, 4.0)
    terminal = -float(potential.get("terminal_yaw_weight", 0.0)) * end_gate * terminal_metric
    turn = -float(potential.get("turn_yaw_rate_weight", 0.0)) * turn_gate * rate_metric
    return {"terminal_yaw": terminal, "turn_yaw_rate": turn,
            "total": terminal + turn}


def reward_terms(config, dt, progress, cross, yaw, yaw_rate, lag, residual,
                 previous_residual, limits, slew, success, failure,
                 comparison=False, reference_yaw_rate=0.0,
                 remaining_distance_m=float("inf"), remaining_arc_m=float("inf"),
                 terminal_yaw_error=0.0, potential_previous=None,
                 potential_current=None, desired_residual=None,
                 previous_desired_residual=None):
    residual = np.asarray(residual, dtype=float)
    previous_residual = np.asarray(previous_residual, dtype=float)
    if residual.shape != (3,) or previous_residual.shape != (3,):
        raise ValueError("residual and previous_residual must have shape (3,)")
    objective = config.get("comparison_objective", {}) if comparison else config
    weights = objective.get("reward", config["reward"])
    tracking = objective.get("tracking_reward", {})
    mode = tracking.get("mode", "legacy")
    if mode == "legacy":
        cross_scale, yaw_denominator, rate_scale, lag_scale = 0.1, 1.0, 0.5, 1.0
    elif mode == "tolerance":
        cross_scale = float(tracking["cross_track_scale_m"])
        angle = float(tracking["yaw_scale_deg"])
        rate_scale = float(tracking["yaw_rate_scale_radps"])
        lag_scale = float(tracking["lag_scale_m"])
        if not 0 < angle <= 180 or min(cross_scale, rate_scale, lag_scale) <= 0:
            raise ValueError("positive tracking scales and yaw tolerance in (0, 180] required")
        yaw_denominator = 1.0 - math.cos(math.radians(angle))
    else:
        raise ValueError("unknown tracking reward mode: " + str(mode))
    regularization = objective.get("regularization", config.get("regularization", {}))
    magnitude_scale = np.asarray(regularization.get("magnitude_scale", limits), dtype=float)
    change_scale = np.asarray(regularization.get("change_per_step_scale", slew), dtype=float)
    if magnitude_scale.shape != (3,) or change_scale.shape != (3,) or np.any(magnitude_scale <= 0) or np.any(change_scale <= 0):
        raise ValueError("three positive physical regularization scales required")
    desired_smooth = 0.0
    desired_weight = float(weights.get("desired_smooth", 0.0))
    if not np.isfinite(desired_weight) or desired_weight < 0:
        raise ValueError("desired smooth weight must be finite and nonnegative")
    if desired_weight:
        desired = np.asarray(desired_residual, dtype=float)
        previous_desired = np.asarray(previous_desired_residual, dtype=float)
        desired_scale = np.asarray(regularization.get("desired_change_per_step_scale", change_scale), dtype=float)
        axis_weights = np.asarray(regularization.get("desired_axis_weights", [0., 1., 1.]), dtype=float)
        if (any(a.shape != (3,) or not np.isfinite(a).all()
                for a in (desired, previous_desired, desired_scale, axis_weights))
                or np.any(desired_scale <= 0) or np.any(axis_weights < 0) or axis_weights.sum() <= 0):
            raise ValueError("desired residual history and valid three-axis scales/weights required")
        desired_smooth = -desired_weight * dt * float(np.sum(
            axis_weights * ((desired - previous_desired) / desired_scale)**2) / axis_weights.sum())
    turn_threshold = float(tracking.get("turn_yaw_rate_threshold_radps", 0.05))
    turn_weight = float(weights.get("turn_yaw_rate", 0.0))
    turn_active = abs(float(reference_yaw_rate)) >= turn_threshold
    endpoint_weight = float(weights.get("terminal_yaw", 0.0))
    endpoint_window = float(tracking.get("terminal_yaw_window_m", 0.5))
    endpoint_arc_window = float(tracking.get("terminal_arc_window_m", endpoint_window))
    endpoint_scale_deg = float(tracking.get("terminal_yaw_scale_deg", 5.0))
    if (turn_threshold < 0 or endpoint_window < 0 or endpoint_arc_window < 0
            or not 0 < endpoint_scale_deg <= 180):
        raise ValueError("invalid terminal/turn reward gate parameters")
    endpoint_active = (float(remaining_distance_m) <= endpoint_window
                       and float(remaining_arc_m) <= endpoint_arc_window)
    endpoint_denominator = 1.0 - math.cos(math.radians(endpoint_scale_deg))
    turn_yaw_rate = (-turn_weight * (yaw_rate / rate_scale) ** 2 * dt
                     if turn_active else 0.0)
    terminal_yaw = (-endpoint_weight * (1.0 - np.cos(terminal_yaw_error))
                    / endpoint_denominator * dt if endpoint_active else 0.0)
    potential_enabled = objective.get("potential_reward", {}).get("mode", "disabled") == "difference"
    if (not potential_enabled or potential_previous is None
            or potential_current is None):
        potential_terminal_yaw = 0.0
        potential_turn_yaw_rate = 0.0
    else:
        shaping_discount = float(objective.get("potential_reward", {}).get(
            "difference_discount", 1.0))
        if not 0 < shaping_discount <= 1.0:
            raise ValueError("potential difference discount must be in (0, 1]")
        potential_terminal_yaw = (shaping_discount * potential_current["terminal_yaw"]
                                  - potential_previous["terminal_yaw"])
        potential_turn_yaw_rate = (shaping_discount * potential_current["turn_yaw_rate"]
                                   - potential_previous["turn_yaw_rate"])
    return dict(
        progress=weights["progress"] * progress,
        cross=-weights["cross_track"] * (cross / cross_scale) ** 2 * dt,
        yaw=-weights["yaw"] * (1 - np.cos(yaw)) / yaw_denominator * dt,
        yaw_rate=-weights["yaw_rate"] * (yaw_rate / rate_scale) ** 2 * dt,
        turn_yaw_rate=turn_yaw_rate,
        terminal_yaw=terminal_yaw,
        potential_terminal_yaw=potential_terminal_yaw,
        potential_turn_yaw_rate=potential_turn_yaw_rate,
        potential=potential_terminal_yaw + potential_turn_yaw_rate,
        lag=-weights["lag"] * (lag / lag_scale) ** 2 * dt,
        effort=-weights["effort"] * np.mean((residual / magnitude_scale) ** 2) * dt,
        smooth=-weights["smooth"] * np.mean(((residual - previous_residual) / change_scale) ** 2) * dt,
        desired_smooth=desired_smooth,
        time=-weights["time"] * dt,
        terminal=weights["success"] * float(success) - weights["failure"] * float(failure),
    )
