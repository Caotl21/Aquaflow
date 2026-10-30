"""Deterministic sharp-turn route geometry shared by ROS and offline tests."""
import math
import hashlib
import json

import numpy as np


ROUTES = {
    "hairpin": [
        (-5.0, 0.0), (-3.0, 0.0), (-1.0, 0.0),
        (-1.0, 1.0), (-1.0, 2.0), (-2.0, 2.0),
        (-3.0, 2.0), (-3.0, 1.0), (-3.0, 0.0),
        (-1.0, 0.0), (1.0, 0.0), (3.0, 0.0),
    ],
    "switchback": [
        (-5.0, 0.0), (-3.0, 0.0), (-1.0, 0.0),
        (-0.2, 0.8), (0.6, 1.6), (0.6, 0.0),
        (0.6, -1.6), (1.4, -2.4), (3.5, -2.4),
    ],
    "right_angle": [
        (-5.0, 0.0), (-3.0, 0.0), (-1.0, 0.0),
        (-1.0, 1.0), (-1.0, 2.5), (0.0, 2.5),
        (2.5, 2.5),
    ],
}


def route_sha256(route):
    """Hash canonical route content, excluding the digest metadata itself."""
    content = dict(route)
    content.pop("route_sha256", None)
    encoded = (json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def sharp_route(name, route_seed=None):
    """Return canonical or reproducibly perturbed sharp-turn route points."""
    if name not in ROUTES:
        raise ValueError("unknown route %s; choose hairpin, switchback, right_angle" % name)
    raw = np.asarray(ROUTES[name], dtype=float)
    if route_seed is not None:
        rng = np.random.default_rng(int(route_seed))
        raw[:, 0] = -5.0 + (raw[:, 0] + 5.0) * rng.uniform(0.85, 1.05)
        raw[:, 1] *= rng.uniform(0.8, 1.05) * rng.choice([-1, 1])
    points = []
    for a, b in zip(raw[:-1], raw[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        count = max(2, int(math.ceil(length / 0.08)))
        for index in range(count):
            ratio = float(index) / count
            points.append((a[0] + ratio * (b[0] - a[0]),
                           a[1] + ratio * (b[1] - a[1])))
    points.append(raw[-1])
    return points


def build_route(name, depth=1.0, route_seed=None):
    xy = sharp_route(name, route_seed=route_seed)
    x = np.asarray([point[0] for point in xy], dtype=float)
    y = np.asarray([point[1] for point in xy], dtype=float)
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    return {
        "name": name,
        "randomized": route_seed is not None,
        "route_seed": int(route_seed) if route_seed is not None else None,
        "frame_id": "world_ned",
        "depth": float(depth),
        "points": [(float(px), float(py), float(depth), float(heading))
                   for px, py, heading in zip(x, y, yaw)],
        "goal": (float(x[-1]), float(y[-1]), float(depth)),
    }
