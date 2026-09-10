"""Arc-length reparameterization and speed profiling for a planned route.

This is the single definition of "what schedule is the controller being asked
to follow".  ``reference_processor`` uses it online to build the reference
window, and ``run_mpc_experiment`` uses it offline to score a run against an
absolute plan.  Keeping one implementation is what makes the evaluation
independent of the controller while still measuring the same intent.

Everything here is a pure function of the route geometry and the profile
limits: no ROS, no node state.
"""
import math

import numpy as np


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def smooth_by_arc_length(s, values, window):
    """Smooth an unwrapped path signal on a uniform arc grid."""
    if len(values) < 3 or window <= 1 or s[-1] - s[0] < 1e-9:
        return values
    window = min(int(window), len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values
    uniform_s = np.linspace(s[0], s[-1], len(values))
    uniform_values = np.interp(uniform_s, s, values)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(uniform_values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = uniform_values[0]
    smoothed[-1] = uniform_values[-1]
    return np.interp(s, uniform_s, smoothed)


def build_arc_profile(x, y, z=None, max_speed=0.2, max_yaw_rate=0.5,
                      max_yaw_accel=0.25, max_accel=0.12, max_decel=0.18,
                      yaw_smoothing_window=5, initial_speed=None,
                      min_profile_speed=0.08, yaw_accel_reserve_ratio=0.2):
    """Reparameterize a route by arc length and attach a speed/time profile.

    Args:
        x, y: route coordinates.  ``z`` is carried through untouched.
        max_speed, max_yaw_rate, max_yaw_accel, max_accel, max_decel:
            profile limits, same meaning as the ``/reference_processor``
            parameters of those names.
        yaw_smoothing_window: odd window for the arc-length tangent filter.
        initial_speed: speed the route must start at.  ``None`` leaves the
            start unconstrained, so the profile may begin at full speed -- the
            online processor uses this when it has no previous speed to carry
            over.  ``0.0`` forces a start from rest, which is what offline
            scoring wants: a plan the vehicle can actually meet from a
            standstill.
        min_profile_speed: floor on the average speed of a segment when
            integrating arrival times.  Without it the stationary goal point
            gives an infinite final interval.
        yaw_accel_reserve_ratio: fraction of ``max_yaw_accel`` that stays
            available to the yaw-accel speed limit no matter how tight the
            curvature.  0.0 restores the pre-fix behaviour, where a tight
            corner could force the profile speed to exactly zero.

    Returns:
        dict with ``x``, ``y``, ``z``, ``yaw``, ``curvature``, ``speed``,
        ``s``, ``time`` and ``total_length``.  ``time`` is the arrival time at
        each point measured from the start of the route.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least two route points, got %d" % n)
    z = np.zeros(n) if z is None else np.asarray(z, dtype=float)

    yaw_smoothing_window = max(3, int(yaw_smoothing_window))
    if yaw_smoothing_window % 2 == 0:
        yaw_smoothing_window += 1
    max_yaw_accel = max(1e-3, float(max_yaw_accel))
    max_accel = max(1e-3, float(max_accel))
    max_decel = max(1e-3, float(max_decel))

    # Compute yaw from the geometric tangent.  Do not trust any incoming pose
    # yaw: the global route's tangent is the canonical heading.
    yaw = np.zeros(n)
    for i in range(n - 1):
        yaw[i] = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i])
    yaw[-1] = yaw[-2]

    # Cumulative arc length
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
    total_length = s[-1]

    # Smooth the unwrapped tangent in arc-length coordinates.  Filtering yaw
    # as a function of s keeps the reference geometrically tied to the route
    # and avoids noisy curvature spikes from point spacing.
    yaw = smooth_by_arc_length(s, yaw, yaw_smoothing_window)

    # Signed curvature from the arc-length tangent.  Using |dyaw|/ds instead
    # would overestimate curvature at wrapped angles and would not distinguish
    # left from right turns.
    curvature = np.zeros(n)
    for i in range(1, n - 1):
        ds = max(s[i + 1] - s[i - 1], 1e-6)
        curvature[i] = wrap_to_pi(yaw[i + 1] - yaw[i - 1]) / ds
    if n > 1:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

    # Curvature-limited speed followed by forward/backward acceleration
    # passes.  This creates a physically continuous speed profile instead of
    # independently changing speed at every point.
    speed_limit = np.minimum(
        max_speed, max_yaw_rate / np.maximum(np.abs(curvature), 1e-6))
    # A rapidly changing curvature creates a rapidly changing reference yaw
    # rate even when |dpsi| itself is below max_yaw_rate.  Reduce the spatial
    # speed limit in those sections so dpsi = v*k stays smooth.
    if n >= 3 and max_yaw_accel > 0.0:
        curvature_gradient = np.gradient(curvature, s, edge_order=1)
        longitudinal_accel_bound = max(max_accel, max_decel)
        # The subtracted term reserves yaw-acceleration budget for changing
        # speed mid-turn, but it assumes worst-case longitudinal acceleration
        # everywhere -- including at steady cruise, where dv/dt is ~0.  On a
        # tight corner that assumption drives the budget negative and pins the
        # speed limit at exactly zero, i.e. the plan orders a dead stop in the
        # middle of the route.  Floor the reserve so the constraint degrades
        # instead of collapsing.
        reserve_floor = max(0.0, yaw_accel_reserve_ratio) * max_yaw_accel
        remaining_yaw_accel = np.maximum(
            max_yaw_accel - np.abs(curvature) * longitudinal_accel_bound,
            reserve_floor)
        yaw_accel_speed_limit = np.sqrt(
            remaining_yaw_accel / np.maximum(np.abs(curvature_gradient), 1e-6))
        speed_limit = np.minimum(speed_limit, yaw_accel_speed_limit)
    speed_limit = np.maximum(speed_limit, 0.0)

    speed = speed_limit.copy()
    if initial_speed is not None:
        speed[0] = min(speed[0], max(0.0, float(initial_speed)))
    for i in range(1, n):
        ds = max(s[i] - s[i - 1], 1e-6)
        speed[i] = min(speed[i], math.sqrt(
            max(0.0, speed[i - 1] ** 2 + 2.0 * max_accel * ds)))
    speed[-1] = 0.0
    for i in range(n - 2, -1, -1):
        ds = max(s[i + 1] - s[i], 1e-6)
        speed[i] = min(speed[i], math.sqrt(
            max(0.0, speed[i + 1] ** 2 + 2.0 * max_decel * ds)))

    # Monotonic time-of-arrival table from the spatial speed profile.
    time_from_start = np.zeros(n)
    for i in range(1, n):
        ds = max(s[i] - s[i - 1], 1e-9)
        avg_speed = max(min_profile_speed, 0.5 * (speed[i - 1] + speed[i]))
        time_from_start[i] = time_from_start[i - 1] + ds / avg_speed

    return {
        'x': x, 'y': y, 'z': z,
        'yaw': yaw, 'curvature': curvature, 'speed': speed, 's': s,
        'time': time_from_start,
        'total_length': total_length,
    }
