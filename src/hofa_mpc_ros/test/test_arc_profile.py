"""Unit tests for the shared arc-length speed profile."""
import sys
import os
import math
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.arc_profile import build_arc_profile, smooth_by_arc_length


def straight_line(length=10.0, n=101):
    x = np.linspace(-length / 2.0, length / 2.0, n)
    return x, np.zeros(n)


def quarter_turn(radius=2.0, n=101):
    theta = np.linspace(0.0, math.pi / 2.0, n)
    return radius * np.cos(theta), radius * np.sin(theta)


LIMITS = dict(max_speed=0.2, max_yaw_rate=0.5, max_yaw_accel=0.25,
              max_accel=0.12, max_decel=0.18, yaw_smoothing_window=5)


class TestGeometry:
    def test_arc_length_matches_straight_line(self):
        x, y = straight_line(length=10.0)
        profile = build_arc_profile(x, y, **LIMITS)
        assert profile['total_length'] == pytest.approx(10.0, abs=1e-9)
        assert profile['s'][0] == 0.0
        assert np.all(np.diff(profile['s']) > 0.0)

    def test_straight_line_has_zero_curvature(self):
        x, y = straight_line()
        profile = build_arc_profile(x, y, **LIMITS)
        assert np.max(np.abs(profile['curvature'])) < 1e-6

    def test_curvature_sign_distinguishes_turn_direction(self):
        x, y = quarter_turn()
        left = build_arc_profile(x, y, **LIMITS)
        right = build_arc_profile(x, -y, **LIMITS)
        interior = slice(5, -5)
        assert np.all(left['curvature'][interior] *
                      right['curvature'][interior] < 0.0)

    def test_z_is_carried_through(self):
        x, y = straight_line(n=11)
        z = np.linspace(1.0, 2.0, 11)
        profile = build_arc_profile(x, y, z, **LIMITS)
        assert np.allclose(profile['z'], z)

    def test_too_few_points_rejected(self):
        with pytest.raises(ValueError):
            build_arc_profile([0.0], [0.0], **LIMITS)


class TestSpeedProfile:
    def test_route_ends_stationary(self):
        x, y = straight_line()
        profile = build_arc_profile(x, y, **LIMITS)
        assert profile['speed'][-1] == pytest.approx(0.0, abs=1e-9)

    def test_unconstrained_start_may_begin_at_speed(self):
        """initial_speed=None is the online default and does NOT force rest."""
        x, y = straight_line()
        profile = build_arc_profile(x, y, initial_speed=None, **LIMITS)
        assert profile['speed'][0] == pytest.approx(0.2, abs=1e-6)

    def test_zero_initial_speed_starts_from_rest(self):
        x, y = straight_line()
        profile = build_arc_profile(x, y, initial_speed=0.0, **LIMITS)
        assert profile['speed'][0] == pytest.approx(0.0, abs=1e-9)

    def test_cruise_reaches_max_speed(self):
        x, y = straight_line(length=20.0, n=401)
        profile = build_arc_profile(x, y, **LIMITS)
        assert np.max(profile['speed']) == pytest.approx(0.2, abs=1e-6)

    def test_never_exceeds_max_speed(self):
        for route in (straight_line(), quarter_turn()):
            profile = build_arc_profile(*route, **LIMITS)
            assert np.max(profile['speed']) <= 0.2 + 1e-9

    def test_acceleration_respects_limits(self):
        """d(v^2)/2ds is the discrete longitudinal acceleration.

        Not ``v * dv/ds``: on the final interval, where v drops to zero, the
        forward difference of that form reports twice the true acceleration
        and would fail against a correct profile.
        """
        x, y = straight_line(length=20.0, n=401)
        profile = build_arc_profile(x, y, initial_speed=0.0, **LIMITS)
        v, s = profile['speed'], profile['s']
        accel = np.diff(v ** 2) / (2.0 * np.maximum(np.diff(s), 1e-9))
        assert np.max(accel) <= 0.12 + 1e-6
        assert -np.min(accel) <= 0.18 + 1e-6

    def test_curvature_limits_speed_on_tight_turn(self):
        # max_yaw_rate / curvature = 0.5 / (1/0.5) = 0.25 -> above max_speed,
        # so tighten further: r=0.2 gives 0.5/5 = 0.1 m/s.
        x, y = quarter_turn(radius=0.2, n=201)
        profile = build_arc_profile(x, y, **LIMITS)
        assert np.max(profile['speed']) < 0.2

    def test_initial_speed_caps_the_start(self):
        x, y = straight_line()
        profile = build_arc_profile(x, y, initial_speed=0.05, **LIMITS)
        assert profile['speed'][0] == pytest.approx(0.05, abs=1e-9)


class TestYawAccelReserve:
    """A tight corner must not be planned as a dead stop.

    Reproduces the corner at s=8.29 of the recorded route, where
    max_yaw_accel - |k|*max(acc,dec) = 0.25 - 1.5*0.18 goes negative.
    """

    def tight_corner(self):
        straight = np.linspace(-2.0, 0.0, 60)
        # Skip theta=0: it would repeat the straight's last point, and a
        # zero-length segment makes np.gradient divide by zero.
        theta = np.linspace(0.0, math.pi / 2.0, 60)[1:]
        radius = 0.66
        x = np.concatenate([straight, radius * np.sin(theta)])
        y = np.concatenate([np.zeros(60), radius * (1.0 - np.cos(theta))])
        return x, y

    def test_zero_reserve_reproduces_the_dead_stop(self):
        x, y = self.tight_corner()
        profile = build_arc_profile(x, y, yaw_accel_reserve_ratio=0.0, **LIMITS)
        assert np.min(profile['speed'][1:-1]) == pytest.approx(0.0, abs=1e-9)

    def test_default_reserve_keeps_the_vehicle_moving(self):
        x, y = self.tight_corner()
        profile = build_arc_profile(x, y, **LIMITS)
        assert np.min(profile['speed'][1:-1]) > 0.05

    def test_reserve_never_raises_speed_above_curvature_limit(self):
        x, y = self.tight_corner()
        profile = build_arc_profile(x, y, yaw_accel_reserve_ratio=10.0, **LIMITS)
        curvature_limit = np.minimum(
            LIMITS['max_speed'],
            LIMITS['max_yaw_rate'] / np.maximum(np.abs(profile['curvature']), 1e-6))
        assert np.all(profile['speed'] <= curvature_limit + 1e-9)


class TestTimeProfile:
    def test_time_is_monotonic_from_zero(self):
        x, y = straight_line()
        profile = build_arc_profile(x, y, **LIMITS)
        assert profile['time'][0] == 0.0
        assert np.all(np.diff(profile['time']) > 0.0)

    def test_duration_matches_hand_computed_trapezoid(self):
        """Long straight run: ramp up, cruise, ramp down."""
        length, vmax, acc, dec = 40.0, 0.2, 0.12, 0.18
        x, y = straight_line(length=length, n=2001)
        profile = build_arc_profile(x, y, **LIMITS)
        s_acc = vmax ** 2 / (2.0 * acc)
        s_dec = vmax ** 2 / (2.0 * dec)
        expected = (vmax / acc + vmax / dec +
                    (length - s_acc - s_dec) / vmax)
        assert profile['time'][-1] == pytest.approx(expected, rel=0.02)

    def test_min_profile_speed_bounds_the_final_interval(self):
        """Coarse spacing near the stationary goal must not blow up."""
        x, y = straight_line(length=10.0, n=11)
        fast = build_arc_profile(x, y, min_profile_speed=0.5, **LIMITS)
        slow = build_arc_profile(x, y, min_profile_speed=1e-9, **LIMITS)
        assert fast['time'][-1] <= slow['time'][-1]
        assert np.isfinite(slow['time'][-1])


class TestSmoothing:
    def test_short_input_passes_through(self):
        values = np.array([1.0, 2.0])
        assert np.allclose(smooth_by_arc_length(np.array([0.0, 1.0]),
                                                values, 5), values)

    def test_endpoints_are_preserved(self):
        s = np.linspace(0.0, 1.0, 21)
        values = np.sin(s * 6.0)
        smoothed = smooth_by_arc_length(s, values, 5)
        assert smoothed[0] == pytest.approx(values[0])
        assert smoothed[-1] == pytest.approx(values[-1])


class TestRecordedRouteRegression:
    """Pin the profile against the route from results/pid/run_20260910_122124.

    That run finished in 49.61 s while this profile schedules ~58 s.  The gap
    is the overspeed the old self-referential metrics reported as +0.31 s.
    """

    ROUTE = os.path.join(
        os.path.dirname(__file__), 'data', 'run_20260910_122124_global_path.csv')

    @pytest.mark.skipif(not os.path.isfile(ROUTE), reason="fixture not present")
    def test_matches_offline_recomputation(self):
        import csv
        with open(self.ROUTE) as stream:
            rows = list(csv.DictReader(stream))
        x = [float(r['x_ned']) for r in rows]
        y = [float(r['y_ned']) for r in rows]
        profile = build_arc_profile(x, y, initial_speed=0.0, **LIMITS)
        assert profile['total_length'] == pytest.approx(11.078, abs=0.01)
        assert profile['time'][-1] == pytest.approx(58.2, abs=0.5)
        # Never planned to stop: the corner at s=8.3 used to zero the speed.
        assert np.min(profile['speed'][1:-1]) > 0.05
