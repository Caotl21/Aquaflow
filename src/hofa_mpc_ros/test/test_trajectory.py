"""Unit tests for trajectory generators."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hofa_mpc_ros.trajectory import (
    create_trajectory, HoverTrajectory, LineTrajectory,
    CircleTrajectory, FigureEightTrajectory, EllipseTrajectory,
    STrajectory
)


class TestHoverTrajectory:
    def test_fixed_position(self):
        gen = HoverTrajectory(x=1.0, y=2.0, psi=0.5)
        ref = gen.get_reference(5.0)
        assert abs(ref.x - 1.0) < 1e-10
        assert abs(ref.y - 2.0) < 1e-10
        assert abs(ref.psi - 0.5) < 1e-10

    def test_zero_velocity(self):
        gen = HoverTrajectory(x=1.0, y=2.0)
        ref = gen.get_reference(5.0)
        assert abs(ref.dx) < 1e-10
        assert abs(ref.dy) < 1e-10

    def test_yaw_scan(self):
        gen = HoverTrajectory(psi=0.0, yaw_mode="scan",
                              yaw_scan_amplitude=0.5, yaw_scan_freq=1.0)
        ref1 = gen.get_reference(5.0)
        ref2 = gen.get_reference(5.25)
        assert abs(ref1.psi - ref2.psi) > 0.01


class TestLineTrajectory:
    def test_start_and_end(self):
        gen = LineTrajectory(x0=0, y0=0, x1=1, y1=0, speed=1.0)
        ref_start = gen.get_reference(gen.startup_time)
        ref_end = gen.get_reference(gen.startup_time + 2.0)
        assert abs(ref_start.x) < 0.1
        assert ref_end.x > ref_start.x

    def test_constant_yaw(self):
        gen = LineTrajectory(x0=0, y0=0, x1=1, y1=0, speed=1.0)
        ref = gen.get_reference(gen.startup_time + 1.0)
        assert abs(ref.psi) < 0.1

    def test_velocity_along_line(self):
        gen = LineTrajectory(x0=0, y0=0, x1=1, y1=1, speed=1.0)
        ref = gen.get_reference(gen.startup_time + 1.0)
        speed = np.hypot(ref.dx, ref.dy)
        assert abs(speed - 1.0) < 0.1


class TestCircleTrajectory:
    def test_returns_to_start(self):
        gen = CircleTrajectory(cx=0, cy=0, radius=0.5, speed=0.3)
        T = 2 * np.pi * 0.5 / 0.3
        t0 = gen.startup_time
        ref1 = gen.get_reference(t0)
        ref2 = gen.get_reference(t0 + T)
        np.testing.assert_allclose(
            [ref1.x, ref1.y], [ref2.x, ref2.y], atol=0.1)

    def test_radius_consistent(self):
        gen = CircleTrajectory(cx=1.0, cy=2.0, radius=0.5, speed=0.3)
        t = gen.startup_time + 1.0
        ref = gen.get_reference(t)
        dist = np.hypot(ref.x - 1.0, ref.y - 2.0)
        assert abs(dist - 0.5) < 0.1

    def test_speed_consistent(self):
        gen = CircleTrajectory(cx=0, cy=0, radius=0.5, speed=0.3)
        t = gen.startup_time + 1.0
        ref = gen.get_reference(t)
        speed = np.hypot(ref.dx, ref.dy)
        assert abs(speed - 0.3) < 0.05


class TestFigureEightTrajectory:
    def test_symmetry(self):
        gen = FigureEightTrajectory(cx=0, cy=0, scale_x=0.6, scale_y=0.3)
        t = gen.startup_time + 1.0
        ref = gen.get_reference(t)
        assert np.hypot(ref.x, ref.y) < 1.0


class TestEllipseTrajectory:
    def test_bounded(self):
        gen = EllipseTrajectory(cx=0, cy=0, a=0.8, b=0.4)
        for t in np.linspace(gen.startup_time, gen.startup_time + 10, 50):
            ref = gen.get_reference(t)
            assert abs(ref.x) <= 1.0
            assert abs(ref.y) <= 0.5


class TestCreateTrajectory:
    def test_all_types(self):
        for name in ["hover", "line", "circle", "figure_eight",
                      "ellipse", "s_shape"]:
            gen = create_trajectory(name)
            ref = gen.get_reference(gen.startup_time + 1.0)
            assert ref is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_trajectory("unknown_type")


class TestGetReferences:
    def test_horizon_length(self):
        gen = HoverTrajectory()
        refs = gen.get_references(5.0, horizon=14, dt=0.1)
        assert len(refs) == 14

    def test_time_progression(self):
        gen = CircleTrajectory(radius=0.5, speed=0.3)
        t = gen.startup_time + 1.0
        refs = gen.get_references(t, horizon=5, dt=0.1)
        for i in range(1, len(refs)):
            assert refs[i].x != refs[i-1].x or refs[i].y != refs[i-1].y
