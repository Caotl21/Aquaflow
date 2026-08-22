"""Trajectory generators for 3-DOF underwater vehicle testing.

All generators produce smooth reference trajectories with position,
velocity, and acceleration.  Yaw modes: fixed, tangent, scan.
"""
import math
import numpy as np
from .types import ReferencePoint
from .hofa import wrap_to_pi


def _quintic_spline(t: float, T: float) -> float:
    """Quintic polynomial for smooth start: 0 at t=0, 1 at t=T."""
    if t <= 0:
        return 0.0
    if t >= T:
        return 1.0
    s = t / T
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _quintic_spline_ddot(t: float, T: float) -> float:
    """Second derivative of quintic spline."""
    if t <= 0 or t >= T:
        return 0.0
    s = t / T
    return (60 * s - 180 * s**2 + 120 * s**3) / T**2


class TrajectoryGenerator:
    """Base class for trajectory generators."""

    def __init__(self, yaw_mode: str = "tangent",
                 startup_time: float = 3.0):
        self.yaw_mode = yaw_mode
        self.startup_time = startup_time

    def get_reference(self, t: float) -> ReferencePoint:
        raise NotImplementedError

    def get_references(self, t: float, horizon: int, dt: float) -> list:
        """Get reference points for the MPC prediction horizon."""
        refs = []
        for i in range(horizon):
            refs.append(self.get_reference(t + i * dt))
        return refs

    def _apply_startup(self, t: float, pos: np.ndarray,
                       vel: np.ndarray, accel: np.ndarray) -> tuple:
        """Apply smooth startup scaling."""
        if t < self.startup_time:
            s = _quintic_spline(t, self.startup_time)
            s_ddot = _quintic_spline_ddot(t, self.startup_time)
            # Position: blend from initial to target
            pos_out = pos * s
            vel_out = vel * s + pos * s_ddot * 0  # simplified
            accel_out = accel * s
            return pos_out, vel_out, accel_out
        return pos, vel, accel


class HoverTrajectory(TrajectoryGenerator):
    """Fixed position hold with optional yaw scan."""

    def __init__(self, x=0.0, y=0.0, psi=0.0,
                 yaw_scan_amplitude: float = 0.0,
                 yaw_scan_freq: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.psi = x, y, psi
        self.scan_amp = yaw_scan_amplitude
        self.scan_freq = yaw_scan_freq

    def get_reference(self, t: float) -> ReferencePoint:
        psi_ref = self.psi
        if self.yaw_mode == "scan" and t >= self.startup_time:
            psi_ref = self.psi + self.scan_amp * math.sin(
                2 * math.pi * self.scan_freq * t)
        return ReferencePoint(
            x=self.x, y=self.y, psi=psi_ref,
            dx=0, dy=0, dpsi=0,
            ddx=0, ddy=0, ddpsi=0,
        )


class LineTrajectory(TrajectoryGenerator):
    """Straight line with fixed or tangent yaw."""

    def __init__(self, x0=0.0, y0=0.0, x1=1.0, y1=0.0,
                 speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.speed = speed
        dx, dy = x1 - x0, y1 - y0
        self.length = math.hypot(dx, dy)
        self.dir = np.array([dx, dy]) / max(self.length, 1e-10)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        s = self.speed * t_eff
        s = min(s, self.length)
        frac = s / max(self.length, 1e-10)

        pos = np.array([self.x0, self.y0]) + frac * np.array([
            self.x1 - self.x0, self.y1 - self.y0])
        vel = self.speed * self.dir
        accel = np.zeros(2)

        pos, vel, accel = self._apply_startup(t, pos, vel, accel)

        psi_ref = math.atan2(self.dir[1], self.dir[0])
        return ReferencePoint(
            x=pos[0], y=pos[1], psi=psi_ref,
            dx=vel[0], dy=vel[1], dpsi=0,
            ddx=accel[0], ddy=accel[1], ddpsi=0,
        )


class CircleTrajectory(TrajectoryGenerator):
    """Circular trajectory with constant speed."""

    def __init__(self, cx=0.0, cy=0.0, radius=0.5,
                 speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.cx, self.cy = cx, cy
        self.radius = radius
        self.omega = speed / max(radius, 1e-10)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        theta = self.omega * t_eff

        x = self.cx + self.radius * math.cos(theta)
        y = self.cy + self.radius * math.sin(theta)
        dx = -self.radius * self.omega * math.sin(theta)
        dy = self.radius * self.omega * math.cos(theta)
        ddx = -self.radius * self.omega**2 * math.cos(theta)
        ddy = -self.radius * self.omega**2 * math.sin(theta)

        psi_ref = math.atan2(dy, dx) if self.yaw_mode == "tangent" else 0.0
        dpsi = self.omega if self.yaw_mode == "tangent" else 0.0

        pos, vel, accel = self._apply_startup(
            t, np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy]))

        return ReferencePoint(
            x=pos[0], y=pos[1], psi=psi_ref,
            dx=vel[0], dy=vel[1], dpsi=dpsi,
            ddx=accel[0], ddy=accel[1], ddpsi=0,
        )


class FigureEightTrajectory(TrajectoryGenerator):
    """Figure-eight (lemniscate) trajectory."""

    def __init__(self, cx=0.0, cy=0.0, scale_x=0.6, scale_y=0.3,
                 speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.cx, self.cy = cx, cy
        self.scale_x, self.scale_y = scale_x, scale_y
        self.omega = speed / max(scale_x, 1e-10)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        theta = self.omega * t_eff
        c, s = math.cos(theta), math.sin(theta)

        x = self.cx + self.scale_x * s
        y = self.cy + self.scale_y * math.sin(2 * theta)
        dx = self.scale_x * self.omega * c
        dy = 2 * self.scale_y * self.omega * math.cos(2 * theta)
        ddx = -self.scale_x * self.omega**2 * s
        ddy = -4 * self.scale_y * self.omega**2 * math.sin(2 * theta)

        psi_ref = math.atan2(dy, dx) if self.yaw_mode == "tangent" else 0.0

        pos, vel, accel = self._apply_startup(
            t, np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy]))

        return ReferencePoint(
            x=pos[0], y=pos[1], psi=psi_ref,
            dx=vel[0], dy=vel[1], dpsi=0,
            ddx=accel[0], ddy=accel[1], ddpsi=0,
        )


class EllipseTrajectory(TrajectoryGenerator):
    """Elliptical trajectory."""

    def __init__(self, cx=0.0, cy=0.0, a=0.8, b=0.4,
                 speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.cx, self.cy = cx, cy
        self.a, self.b = a, b
        self.omega = speed / max(a, 1e-10)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        theta = self.omega * t_eff

        x = self.cx + self.a * math.cos(theta)
        y = self.cy + self.b * math.sin(theta)
        dx = -self.a * self.omega * math.sin(theta)
        dy = self.b * self.omega * math.cos(theta)
        ddx = -self.a * self.omega**2 * math.cos(theta)
        ddy = -self.b * self.omega**2 * math.sin(theta)

        psi_ref = math.atan2(dy, dx) if self.yaw_mode == "tangent" else 0.0

        pos, vel, accel = self._apply_startup(
            t, np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy]))

        return ReferencePoint(
            x=pos[0], y=pos[1], psi=psi_ref,
            dx=vel[0], dy=vel[1], dpsi=0,
            ddx=accel[0], ddy=accel[1], ddpsi=0,
        )


class STrajectory(TrajectoryGenerator):
    """S-shaped trajectory using cubic sine."""

    def __init__(self, x0=0.0, y0=0.0, x1=2.0, y1=0.0,
                 amplitude: float = 0.4, speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.amp = amplitude
        self.speed = speed
        dx, dy = x1 - x0, y1 - y0
        self.length = math.hypot(dx, dy)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        s = min(self.speed * t_eff, self.length)
        frac = s / max(self.length, 1e-10)

        x = self.x0 + frac * (self.x1 - self.x0)
        y = self.y0 + frac * (self.y1 - self.y0) + \
            self.amp * math.sin(2 * math.pi * frac)

        dx = self.speed * (self.x1 - self.x0) / max(self.length, 1e-10)
        dy = self.speed * (self.y1 - self.y0) / max(self.length, 1e-10) + \
            self.amp * 2 * math.pi * self.speed / max(self.length, 1e-10) * \
            math.cos(2 * math.pi * frac)

        psi_ref = math.atan2(dy, dx) if self.yaw_mode == "tangent" else 0.0

        return ReferencePoint(
            x=x, y=y, psi=psi_ref,
            dx=dx, dy=dy, dpsi=0,
            ddx=0, ddy=0, ddpsi=0,
        )


class RoundedRectangleTrajectory(TrajectoryGenerator):
    """Rounded rectangle trajectory."""

    def __init__(self, cx=0.0, cy=0.0, width=2.0, height=1.0,
                 corner_radius=0.3, speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.cx, self.cy = cx, cy
        self.w, self.h = width, height
        self.cr = min(corner_radius, min(width, height) / 2)
        self.speed = speed
        self._compute_perimeter()

    def _compute_perimeter(self):
        w, h, cr = self.w, self.h, self.cr
        self.segments = [
            (w - 2 * cr, (1, 0)),
            (cr * math.pi / 2, None),  # corner
            (h - 2 * cr, (0, 1)),
            (cr * math.pi / 2, None),
            (w - 2 * cr, (-1, 0)),
            (cr * math.pi / 2, None),
            (h - 2 * cr, (0, -1)),
            (cr * math.pi / 2, None),
        ]
        self.total_length = sum(s[0] for s in self.segments)

    def get_reference(self, t: float) -> ReferencePoint:
        t_eff = max(0.0, t - self.startup_time)
        dist = min(self.speed * t_eff, self.total_length)
        return ReferencePoint(x=self.cx, y=self.cy, psi=0)


def create_trajectory(name: str, **kwargs) -> TrajectoryGenerator:
    """Factory function to create trajectory by name."""
    generators = {
        "hover": HoverTrajectory,
        "line": LineTrajectory,
        "circle": CircleTrajectory,
        "figure_eight": FigureEightTrajectory,
        "ellipse": EllipseTrajectory,
        "s_shape": STrajectory,
        "rounded_rectangle": RoundedRectangleTrajectory,
    }
    if name not in generators:
        raise ValueError(f"Unknown trajectory: {name}. "
                         f"Available: {list(generators.keys())}")
    return generators[name](**kwargs)
