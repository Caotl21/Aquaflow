"""Deterministic obstacle-free trajectory suites for HOFA teacher data."""
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "hofa_policy_trajectory_v1"
SUITE_SCHEMA = "hofa_policy_trajectory_suite_v1"
FAMILIES = ("line", "s_curve", "double_s",
            "lateral_shift", "chicane", "long_bend")


def _smoothstep(t):
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _path_length(x, y):
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def generate_track(track_id, split, family, seed, depth=1.0):
    rng = np.random.default_rng(seed)
    count = 181
    t = np.linspace(0.0, 1.0, count)
    x0 = -5.0
    x1 = float(rng.uniform(3.6, 5.0))
    y0 = float(rng.uniform(-0.28, 0.28))
    x = x0 + (x1 - x0) * t
    amplitude = float(rng.uniform(0.45, 1.35))

    if family == "line":
        y1 = float(rng.uniform(-1.4, 1.4))
        y = y0 + (y1 - y0) * t
    elif family == "s_curve":
        y = y0 + amplitude * np.sin(2.0 * math.pi * t)
    elif family == "double_s":
        y = y0 + 0.72 * amplitude * np.sin(4.0 * math.pi * t)
    elif family == "lateral_shift":
        shift = float(rng.choice([-1.0, 1.0]) * amplitude)
        y = y0 + shift * _smoothstep(t)
    elif family == "chicane":
        sign = float(rng.choice([-1.0, 1.0]))
        y = y0 + sign * amplitude * (
            np.sin(2.0 * math.pi * t)
            + 0.32 * np.sin(4.0 * math.pi * t))
    elif family == "long_bend":
        shift = float(rng.choice([-1.0, 1.0]) * amplitude)
        y = y0 + shift * 0.5 * (1.0 - np.cos(math.pi * t))
    else:
        raise ValueError("unknown trajectory family: %s" % family)

    if np.max(np.abs(y)) > 3.0:
        y *= 2.9 / np.max(np.abs(y))
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.unwrap(np.arctan2(dy, dx))
    points = [[float(px), float(py), float(depth), float(psi)]
              for px, py, psi in zip(x, y, yaw)]
    speed = float(rng.choice([0.18, 0.22, 0.26, 0.30, 0.34]))
    length = _path_length(x, y)
    return {
        "schema": SCHEMA,
        "id": track_id,
        "split": split,
        "family": family,
        "seed": int(seed),
        "frame_id": "world_ned",
        "depth_m": float(depth),
        "points": points,
        "start": points[0][:3],
        "goal": points[-1][:3],
        "length_m": length,
        "max_speed_mps": speed,
        "max_yaw_rate_radps": 0.5,
        "max_accel_mps2": 0.12,
        "max_decel_mps2": 0.18,
        "timeout_s": max(75.0, length / max(speed, 0.05) + 45.0),
    }


def validate_track(track):
    if track.get("schema") != SCHEMA:
        raise ValueError("trajectory schema mismatch")
    points = np.asarray(track.get("points"), dtype=float)
    if points.ndim != 2 or points.shape[0] < 20 or points.shape[1] != 4:
        raise ValueError("trajectory points must have shape [N,4]")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory contains nonfinite points")
    if np.any(points[:, 0] < -6.5) or np.any(points[:, 0] > 6.5):
        raise ValueError("trajectory leaves pool x bounds")
    if np.any(points[:, 1] < -3.2) or np.any(points[:, 1] > 3.2):
        raise ValueError("trajectory leaves pool y bounds")
    spacing = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    if np.any(spacing <= 0.0) or np.max(spacing) > 0.15:
        raise ValueError("trajectory point spacing is invalid")
    return track


def generate_suite(output, seed=27, train=20, val=5, test=5):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("suite output already exists: %s" % output)
    tracks_dir = output / "tracks"
    tracks_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    entries = []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            track_seed = int(rng.integers(0, 2**31 - 1))
            track_id = "%s_%02d_%s" % (split, index, family)
            track = validate_track(generate_track(
                track_id, split, family, track_seed))
            path = tracks_dir / (track_id + ".json")
            path.write_text(json.dumps(track, indent=2) + "\n")
            entries.append({
                "id": track_id,
                "split": split,
                "family": family,
                "path": str(path.relative_to(output)),
                "seed": track_seed,
            })
    suite = {
        "schema": SUITE_SCHEMA,
        "seed": int(seed),
        "tracks": entries,
    }
    (output / "suite.json").write_text(
        json.dumps(suite, indent=2) + "\n")
    return suite

