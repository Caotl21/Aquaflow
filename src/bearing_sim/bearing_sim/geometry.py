"""Metres, seconds, radians; +x forward, +y left, positive yaw counterclockwise."""
from dataclasses import dataclass
import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    def __post_init__(self):
        if not np.isfinite([self.x, self.y, self.radius]).all() or self.radius <= 0:
            raise ValueError("circle must be finite with positive radius")


def clearance(p, circles, radius, bounds):
    x, y = p
    xmin, xmax, ymin, ymax = bounds
    values = [x-xmin-radius, xmax-x-radius, y-ymin-radius, ymax-y-radius]
    values.extend(math.hypot(x-o.x, y-o.y)-o.radius-radius for o in circles)
    return min(values)


def swept_clearance(a, b, circles, radius, bounds):
    """Exact circular footprint clearance for the piecewise-linear executed path."""
    a, b = np.asarray(a), np.asarray(b)
    d = b-a
    dd = float(d @ d)
    value = min(clearance(a, (), radius, bounds), clearance(b, (), radius, bounds))
    for o in circles:
        center = np.array([o.x, o.y])
        t = np.clip((center-a) @ d / dd, 0, 1) if dd else 0.0
        value = min(value, np.linalg.norm(a+t*d-center)-o.radius-radius)
    return float(value)


def first_contact(a, b, circles, radius, bounds):
    """Fraction of segment before first physical contact, including pool walls."""
    a, b = np.asarray(a), np.asarray(b)
    if clearance(a, circles, radius, bounds) <= 0:
        return 0.0
    d = b-a
    dd = float(d @ d)
    hit = 1.0
    if dd:
        for o in circles:
            rel = a-np.array([o.x, o.y])
            bb = 2*float(rel @ d)
            cc = float(rel @ rel)-(o.radius+radius)**2
            disc = bb*bb-4*dd*cc
            if disc >= 0:
                t = (-bb-math.sqrt(disc))/(2*dd)
                if 0 <= t <= hit:
                    hit = t
    xmin, xmax, ymin, ymax = bounds
    for axis, lo, hi in [(0, xmin+radius, xmax-radius), (1, ymin+radius, ymax-radius)]:
        if b[axis] < lo and d[axis] < 0:
            hit = min(hit, (lo-a[axis])/d[axis])
        if b[axis] > hi and d[axis] > 0:
            hit = min(hit, (hi-a[axis])/d[axis])
    return float(hit)


def camera_tokens(pose, circles, bounds, cfg, rng):
    """Ray visibility with near-object occlusion; walls are visible angular objects.

    Tokens: sin(center), cos(center), left, right, half_width, confidence,
    FOV_truncated. Bearings are from the VISIBLE silhouette, never hidden
    tangent points. Track IDs are ideal simulator associations, returned
    separately; no distance or geometry is returned to the policy.
    """
    cam = pose[:2] + rotation(pose[2]) @ np.asarray(cfg.camera_offset)
    angles = np.linspace(-cfg.fov/2, cfg.fov/2, cfg.rays)
    directions = np.stack([np.cos(angles+pose[2]), np.sin(angles+pose[2])], axis=1)
    distances = []
    for o in circles:
        rel = np.array([o.x, o.y])-cam
        projection = directions @ rel
        disc = projection**2-(rel @ rel-o.radius**2)
        root = np.sqrt(np.maximum(disc, 0))
        near, far = projection-root, projection+root
        hit = np.where(near >= 0, near, far)
        distances.append(np.where((disc >= 0) & (hit >= 0), hit, np.inf))
    xmin, xmax, ymin, ymax = bounds
    for axis, wall, other_lo, other_hi in [(0,xmin,ymin,ymax),(0,xmax,ymin,ymax),
                                          (1,ymin,xmin,xmax),(1,ymax,xmin,xmax)]:
        denom = directions[:, axis]
        t = np.full(cfg.rays, np.inf)
        np.divide(wall-cam[axis], denom, out=t, where=np.abs(denom)>1e-12)
        cross = cam[1-axis] + np.where(np.isfinite(t), t, 0)*directions[:, 1-axis]
        distances.append(np.where((t>=0)&(cross>=other_lo)&(cross<=other_hi), t, np.inf))
    distances = np.asarray(distances)
    winner = np.argmin(distances, axis=0)
    nearest = np.min(distances, axis=0)
    winner[nearest > cfg.sensor_range] = -1
    detected = []
    for identifier in np.unique(winner):
        if identifier < 0 or rng.random() < cfg.dropout:
            continue
        pixels = np.flatnonzero(winner == identifier)
        left, right = angles[pixels[0]], angles[pixels[-1]]
        truncated = pixels[0] == 0 or pixels[-1] == cfg.rays-1
        noisy = np.sort(np.clip([left, right]+rng.normal(0,cfg.bearing_noise,2),
                                -cfg.fov/2,cfg.fov/2))
        left, right = noisy
        center = (left+right)/2
        detected.append((abs(center), int(identifier), [math.sin(center),math.cos(center),
                         left,right,(right-left)/2,1.0,float(truncated)]))
    detected.sort(key=lambda item: (item[0], item[1]))
    tokens = np.zeros((cfg.max_tracks,7), dtype=np.float32)
    mask = np.zeros(cfg.max_tracks,dtype=bool)
    ids = np.full(cfg.max_tracks,-1,dtype=np.int64)
    for k, (_, identifier, token) in enumerate(detected[:cfg.max_tracks]):
        tokens[k], mask[k], ids[k] = token, True, identifier
    return tokens, mask, ids
