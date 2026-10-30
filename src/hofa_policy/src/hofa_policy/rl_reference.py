"""ROS-free arc-profile reference with the same temporal sampling as ROS."""
import math
import numpy as np
from hofa_mpc_ros.arc_profile import build_arc_profile
from .contract import CONTRACT

ROUTES = {
    "right_angle": [(-5, 0), (-3, 0), (-1, 0), (-1, 1), (-1, 2.5), (0, 2.5), (2.5, 2.5)],
    "hairpin": [(-5, 0), (-3, 0), (-1, 0), (-1, 1), (-1, 2), (-2, 2),
                (-3, 2), (-3, 1), (-3, 0), (-1, 0), (1, 0), (3, 0)],
    "switchback": [(-5, 0), (-3, 0), (-1, 0), (-0.2, 0.8), (0.6, 1.6),
                   (0.6, 0), (0.6, -1.6), (1.4, -2.4), (3.5, -2.4)],
}


def make_route(name, depth=1.0, rng=None, randomize=False):
    raw = np.asarray(ROUTES[name], dtype=float).copy()
    if randomize:
        raw[:, 0] = -5 + (raw[:, 0] + 5) * rng.uniform(0.85, 1.05)
        raw[:, 1] *= rng.uniform(0.8, 1.05) * rng.choice([-1, 1])
    xy = []
    for a, b in zip(raw[:-1], raw[1:]):
        count = max(2, int(math.ceil(np.linalg.norm(b-a) / 0.08)))
        xy.extend(a + (b-a) * i/count for i in range(count))
    xy.append(raw[-1])
    xy = np.asarray(xy)
    yaw = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    return dict(name=name, frame_id="world_ned", depth=float(depth),
                points=np.column_stack((xy, np.full(len(xy), depth), yaw)).tolist(),
                goal=[float(xy[-1, 0]), float(xy[-1, 1]), float(depth)])


class Reference:
    def __init__(self, route, params):
        points = np.asarray(route["points"])
        self.route = route
        self.profile = build_arc_profile(
            points[:, 0], points[:, 1], points[:, 2], initial_speed=0.0,
            max_speed=params["max_speed"], max_yaw_rate=params["max_yaw_rate"],
            max_yaw_accel=params["max_yaw_accel_radps2"],
            max_accel=params["max_accel_mps2"], max_decel=params["max_decel_mps2"],
            yaw_smoothing_window=params["yaw_smoothing_window"],
            yaw_accel_reserve_ratio=params["yaw_accel_reserve_ratio"])
        p = self.profile
        self.xy = np.column_stack((p["x"], p["y"]))
        self.segments = np.diff(self.xy, axis=0)
        self.length_sq = np.sum(self.segments**2, axis=1)
        self.arc, self.schedule_t, self.last_t = 0.0, 0.0, 0.0

    def project(self, x, y, update=False):
        # Only adjacent arc segments are eligible. A hairpin crossing must not
        # teleport progress onto a later part of the route.
        p = self.profile
        indexes = np.flatnonzero((p["s"][:-1] <= self.arc+0.6) &
                                 (p["s"][1:] >= self.arc-0.3))
        error = np.asarray([x, y]) - self.xy[indexes]
        den = self.length_sq[indexes]
        ratio = np.clip(np.sum(error*self.segments[indexes], axis=1) /
                        np.maximum(den, 1e-12), 0, 1)
        segment_length = p["s"][indexes+1]-p["s"][indexes]
        lower = np.clip((self.arc-0.3-p["s"][indexes])/segment_length, 0, 1)
        upper = np.clip((self.arc+0.6-p["s"][indexes])/segment_length, 0, 1)
        ratio = np.clip(ratio, lower, upper)
        residual = error-ratio[:, None]*self.segments[indexes]
        best = int(np.argmin(np.sum(residual**2, axis=1)))
        index = indexes[best]
        arc = float(p["s"][index]+ratio[best]*(p["s"][index+1]-p["s"][index]))
        signed = float((self.segments[index, 0]*residual[best, 1] -
                        self.segments[index, 1]*residual[best, 0]) /
                       max(np.sqrt(den[best]), 1e-12))
        if update:
            self.arc = arc
        return arc, signed, float(np.linalg.norm(residual[best]))

    def sample(self, arcs):
        p = self.profile
        values = {key: np.interp(arcs, p["s"], p[key])
                  for key in ("x", "y", "z", "yaw", "speed", "curvature")}
        values["dx"] = values["speed"]*np.cos(values["yaw"])
        values["dy"] = values["speed"]*np.sin(values["yaw"])
        values["dpsi"] = values["speed"]*values["curvature"]
        return values

    def window(self, elapsed):
        p = self.profile
        self.schedule_t += max(0.0, elapsed-self.last_t)
        self.last_t = elapsed
        anchor = float(np.interp(self.schedule_t, p["time"], p["s"]))
        if anchor > self.arc+1.0:
            anchor = min(self.arc+1.0, p["total_length"])
            self.schedule_t = float(np.interp(anchor, p["s"], p["time"]))
        times = self.schedule_t+np.arange(CONTRACT.horizon)*CONTRACT.reference_dt_s
        v = self.sample(np.interp(times, p["time"], p["s"]))
        acceleration = {key: np.gradient(v[key], CONTRACT.reference_dt_s)
                        for key in ("dx", "dy", "dpsi")}
        return [dict(x_ned=float(v["x"][i]), y_ned=float(v["y"][i]),
                     yaw_ned=float(v["yaw"][i]), dx_ned=float(v["dx"][i]),
                     dy_ned=float(v["dy"][i]), dpsi_ned=float(v["dpsi"][i]),
                     ddx_ned=float(acceleration["dx"][i]),
                     ddy_ned=float(acceleration["dy"][i]),
                     ddpsi_ned=float(acceleration["dpsi"][i]))
                for i in range(CONTRACT.horizon)]
