"""Performance metrics computation for trajectory tracking experiments."""
import math
import numpy as np
from typing import Dict, List


def rmse(errors: np.ndarray) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean(errors**2)))


def mae(errors: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(errors)))


def percentile_95(errors: np.ndarray) -> float:
    """95th percentile of absolute errors."""
    return float(np.percentile(np.abs(errors), 95))


def max_error(errors: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(errors)))


def tracking_metrics(position_errors: np.ndarray,
                     yaw_errors: np.ndarray) -> Dict[str, float]:
    """Compute tracking metrics for position and yaw.

    Args:
        position_errors: (N, 2) array of [x, y] errors, or (N,) for norm
        yaw_errors: (N,) array of yaw errors in radians

    Returns:
        Dictionary of metric name to value
    """
    if position_errors.ndim == 2:
        pos_norm = np.linalg.norm(position_errors, axis=1)
    else:
        pos_norm = np.abs(position_errors)

    yaw_deg = np.degrees(yaw_errors)

    return {
        "position_rmse_m": rmse(pos_norm),
        "position_mae_m": mae(pos_norm),
        "position_95_m": percentile_95(pos_norm),
        "position_max_m": max_error(pos_norm),
        "yaw_rmse_deg": rmse(yaw_deg),
        "yaw_mae_deg": mae(yaw_deg),
        "yaw_95_deg": percentile_95(yaw_deg),
        "yaw_max_deg": max_error(yaw_deg),
    }


def thruster_metrics(thruster_commands: np.ndarray,
                     dt: float) -> Dict[str, float]:
    """Compute thruster utilization metrics.

    Args:
        thruster_commands: (N, n_thrusters) array of thrust values
        dt: time step

    Returns:
        Dictionary of metrics
    """
    n = thruster_commands.shape[0]
    if n < 2:
        return {}

    # Energy: sum of f_i^2 * dt
    energy = float(np.sum(thruster_commands**2) * dt)

    # Thrust rate: sum of |f_i(k) - f_i(k-1)|
    thrust_rate = float(np.sum(np.abs(np.diff(thruster_commands, axis=0))))

    # Saturation ratio (assuming max ~67 N)
    max_thrust = 67.67
    saturated = np.any(np.abs(thruster_commands) >= max_thrust * 0.99, axis=1)
    sat_ratio = float(np.mean(saturated))

    # Max consecutive saturation
    max_consec = 0
    consec = 0
    for s in saturated:
        if s:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "energy_J": energy,
        "thrust_rate_N": thrust_rate,
        "saturation_ratio": sat_ratio,
        "max_consecutive_saturation_s": max_consec * dt,
    }


def timing_metrics(callback_times_ms: np.ndarray,
                   deadline_ms: float = 80.0) -> Dict[str, float]:
    """Compute timing performance metrics.

    Args:
        callback_times_ms: array of callback durations in milliseconds
        deadline_ms: deadline in milliseconds

    Returns:
        Dictionary of metrics
    """
    return {
        "avg_time_ms": float(np.mean(callback_times_ms)),
        "p95_time_ms": float(np.percentile(callback_times_ms, 95)),
        "max_time_ms": float(np.max(callback_times_ms)),
        "deadline_miss_ratio": float(np.mean(callback_times_ms > deadline_ms)),
    }


def disturbance_metrics(position_errors: np.ndarray,
                        disturbance_start_idx: int,
                        dt: float,
                        threshold: float = 0.05) -> Dict[str, float]:
    """Compute disturbance rejection metrics.

    Args:
        position_errors: (N, 2) array of position errors
        disturbance_start_idx: index where disturbance was applied
        dt: time step
        threshold: error threshold for recovery

    Returns:
        Dictionary of metrics
    """
    if position_errors.ndim == 2:
        pos_norm = np.linalg.norm(position_errors, axis=1)
    else:
        pos_norm = np.abs(position_errors)

    post_disturbance = pos_norm[disturbance_start_idx:]
    peak_error = float(np.max(post_disturbance))

    # Recovery time: first time after peak where error stays below threshold
    peak_idx = np.argmax(post_disturbance)
    recovery_idx = len(post_disturbance)
    for i in range(peak_idx, len(post_disturbance)):
        if post_disturbance[i] < threshold:
            # Check it stays below for 0.5s
            window = int(0.5 / dt)
            end_idx = min(i + window, len(post_disturbance))
            if np.all(post_disturbance[i:end_idx] < threshold):
                recovery_idx = i
                break

    recovery_time = (recovery_idx - peak_idx) * dt

    return {
        "peak_error_m": peak_error,
        "recovery_time_s": recovery_time,
    }
