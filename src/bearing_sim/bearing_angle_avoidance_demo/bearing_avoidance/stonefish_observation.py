#!/usr/bin/env python3
"""Stonefish segmentation-image to bearing-angle observation adapter.

The adapter deliberately consumes only a label image and camera intrinsics.
It never uses depth or a scene map.  The segmentation mode provides an
association oracle for P1; the same AngularObservation interface can later be
fed by an RGB detector.
"""
from dataclasses import dataclass
import math

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - ROS machines normally provide OpenCV.
    cv2 = None


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_values(cls, width, height, fx, fy=None, cx=None, cy=None):
        width = int(width)
        height = int(height)
        fx = float(fx)
        fy = fx if fy is None else float(fy)
        cx = width / 2.0 if cx is None else float(cx)
        cy = height / 2.0 if cy is None else float(cy)
        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
            raise ValueError("camera dimensions and focal lengths must be positive")
        return cls(width, height, fx, fy, cx, cy)


@dataclass(frozen=True)
class SegmentationConfig:
    min_pixels: int = 12
    min_height: int = 8
    max_width_fraction: float = 0.30
    max_labels: int = 32
    partial_margin_px: int = 2


@dataclass(frozen=True)
class SegmentationResult:
    detections: dict
    partial_sectors: list
    accepted_labels: tuple
    rejected_labels: tuple


@dataclass(frozen=True)
class ColorConfig:
    # Distant pillars are tinted blue by water attenuation (channel spread
    # ~35-57 at 4-9 m in the P1 pool) while the lit background is ~130.
    dark_max_value: int = 80
    dark_channel_spread: int = 60
    min_pixels: int = 40
    min_height: int = 12
    min_width_px: int = 2
    # Pillars span the water column, so at the camera depth they always cross
    # the horizon row; the water surface and pool floor never do.
    band_half_height_px: int = 24
    band_fill: float = 0.6
    # Overlapping pillars are split where the per-column brightness (water
    # attenuation grows with distance) or floor-contact row steps; both are
    # flat across a single pillar.
    split_brightness: float = 6.0
    split_bottom_px: int = 12
    split_window_px: int = 3
    split_min_piece_px: int = 5
    partial_margin_px: int = 2
    max_components: int = 24
    morph_kernel: int = 3


@dataclass(frozen=True)
class ColorComponent:
    left: int
    right: int
    top: int
    bottom: int
    area: int
    beta: float
    theta: float
    complete: bool
    # Cut by a nearer pillar at a split (not by the image side).
    occluded: bool = False


def image_to_rgb(data, height, width, step=None, encoding="rgb8", is_bigendian=False):
    """Decode common ROS color encodings without cv_bridge."""
    height, width = int(height), int(width)
    encoding = str(encoding).lower()
    if encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
        raise ValueError("unsupported RGB encoding: %s" % encoding)
    channels = 4 if len(encoding) == 5 else 3
    step = int(step or width * channels)
    if height <= 0 or width <= 0 or step < width * channels:
        raise ValueError("invalid RGB image shape or row step")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError("RGB image data is shorter than height*step")
    rows = raw[:expected].reshape(height, step)
    image = rows[:, :width * channels].reshape(height, width, channels)
    if encoding in ("bgr8", "bgra8"):
        image = image[..., ::-1]
    return image[..., :3]


def extract_color_components(rgb, camera, config=None, rejected=None):
    """Detect dark vertical pillars that cross the horizon band of an image.

    Pillar extents are the runs of dark columns in a band of rows around
    ``camera.cy``.  There is no width or aspect gate: a pillar close enough to
    fill most of the image is still reported, and one touching the image side
    is marked ``complete=False`` so the caller can treat it as a partial
    sector.  A run covering overlapping pillars is split where the brightness
    or floor-contact row steps; the farther piece is occluded at the split and
    is also marked incomplete.  Overlapping pillars at similar distance and
    brightness stay merged.

    This intentionally uses only color and image shape. It does not use depth,
    Stonefish labels, or a scene map, so it can be reused with a real camera.
    If ``rejected`` is a list, every filtered-out run is appended as
    ``(left, top, width, height, area, reason)`` for threshold tuning.
    """
    config = config or ColorConfig()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[:2] != (camera.height, camera.width) or rgb.shape[2] != 3:
        raise ValueError("RGB image shape does not match camera model")
    values = rgb.astype(np.uint16)
    dark = ((values.max(axis=2) <= int(config.dark_max_value)) &
            ((values.max(axis=2) - values.min(axis=2)) <= int(config.dark_channel_spread)))
    mask = (dark.astype(np.uint8) * 255)
    if cv2 is None:
        raise RuntimeError("RGB mode requires OpenCV (python3-opencv)")
    kernel_size = max(1, int(config.morph_kernel))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    dark = mask > 0
    centre = int(round(camera.cy))
    half = max(0, int(config.band_half_height_px))
    band_top = min(max(0, centre - half), camera.height - 1)
    band_bottom = min(camera.height, centre + half + 1)
    band = dark[band_top:band_bottom]
    columns = band.mean(axis=0) >= float(config.band_fill)
    edges = np.diff(np.r_[False, columns, False].astype(np.int8))
    brightness = np.where(band, values[band_top:band_bottom].max(axis=2), 0).sum(axis=0)
    brightness = brightness / np.maximum(band.sum(axis=0), 1)
    below = ~dark[centre:]
    floor_row = np.where(below.any(axis=0), centre + below.argmax(axis=0) - 1, camera.height - 1)
    segments = []
    for left, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segments.extend(_split_run(int(left), int(end) - 1, brightness, floor_row, config))
    margin = int(config.partial_margin_px)
    candidates = []
    for left, right, occluded in segments:
        width = right - left + 1
        # Vertical extent: the contiguous run of mostly-dark rows through the
        # band.  It may extend into a dark surface region; only the columns
        # are used for the angle, so that does not bias the observation.
        rows = dark[:, left:right + 1].mean(axis=1) >= 0.5
        band_rows = np.flatnonzero(rows[band_top:band_bottom])
        if band_rows.size == 0:
            top = bottom = centre
        else:
            seed = band_top + int(band_rows[band_rows.size // 2])
            top = seed
            while top > 0 and rows[top - 1]:
                top -= 1
            bottom = seed
            while bottom < camera.height - 1 and rows[bottom + 1]:
                bottom += 1
        height = bottom - top + 1
        area = int(dark[top:bottom + 1, left:right + 1].sum())
        reason = None
        if width < int(config.min_width_px):
            reason = "min_width"
        elif height < int(config.min_height):
            reason = "min_height"
        elif area < int(config.min_pixels):
            reason = "min_pixels"
        if reason is not None:
            if rejected is not None:
                rejected.append((left, top, width, height, area, reason))
            continue
        beta_left, beta_right = _sector_from_columns(left, right, camera)
        complete = not (occluded or left <= margin or right >= camera.width - 1 - margin)
        candidates.append(ColorComponent(
            left, right, top, bottom, area,
            0.5 * (beta_left + beta_right), beta_right - beta_left,
            complete, bool(occluded)))
    candidates.sort(key=lambda item: item.area, reverse=True)
    return candidates[:max(1, int(config.max_components))], mask


def _split_run(left, right, brightness, floor_row, config):
    """Split one dark-column run into ``(left, right, occluded)`` pieces."""
    w = max(1, int(config.split_window_px))
    scores = {}
    for i in range(left + w, right - w + 2):
        step_b = np.median(brightness[i:i + w]) - np.median(brightness[i - w:i])
        step_f = np.median(floor_row[i:i + w]) - np.median(floor_row[i - w:i])
        score = max(abs(step_b) / float(config.split_brightness),
                    abs(step_f) / float(config.split_bottom_px))
        if score >= 1.0:
            # The nearer pillar ends lower on the floor or, when both run off
            # the image, is darker; the right piece is farther if it is not.
            right_farther = step_f < 0 if abs(step_f) >= config.split_bottom_px else step_b > 0
            scores[i] = (score, right_farther)
    # Windowed medians give a plateau of equal scores around a step; cut at
    # the largest single-column change inside it.
    jump = np.maximum(np.abs(np.diff(brightness)) / float(config.split_brightness),
                      np.abs(np.diff(floor_row)) / float(config.split_bottom_px))
    cuts = []
    for i in sorted(scores, key=lambda k: scores[k][0], reverse=True):
        if all(abs(i - c) >= w for c, _ in cuts):
            near = [k for k in range(i - w + 1, i + w) if k in scores]
            best = max(near, key=lambda k: jump[k - 1])
            cuts.append((best, scores[best][1]))
    cuts.sort()
    # Shading and anti-aliasing at a pillar's own silhouette also step the
    # brightness; a cut there leaves a 1-3 px "occluded" sliver whose partial
    # sector blocks +-8 deg of headings.  Keep only cuts leaving wide pieces.
    min_piece = max(1, int(config.split_min_piece_px))
    kept, start = [], left
    for i, right_farther in cuts:
        if i - start >= min_piece and right + 1 - i >= min_piece:
            kept.append((i, right_farther))
            start = i
    cuts = kept
    pieces = []
    start, occluded_left = left, False
    for i, right_farther in cuts:
        pieces.append((start, i - 1, occluded_left or not right_farther))
        start, occluded_left = i, right_farther
    pieces.append((start, right, occluded_left))
    return pieces


def image_to_uint16(data, height, width, step=None, is_bigendian=False):
    """Decode a ROS ``16UC1`` image without requiring cv_bridge."""
    height = int(height)
    width = int(width)
    step = int(step or width * 2)
    if height <= 0 or width <= 0 or step < width * 2:
        raise ValueError("invalid 16UC1 image shape or row step")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError("16UC1 image data is shorter than height*step")
    rows = raw[:expected].reshape(height, step)
    byte_order = ">u2" if is_bigendian else "<u2"
    return rows[:, :width * 2].view(byte_order).reshape(height, width).astype(np.uint16, copy=False)


def _label_bbox(mask, label):
    rows, cols = np.nonzero(mask == label)
    if rows.size == 0:
        return None
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max()), int(rows.size)


def _sector_from_columns(left, right, camera):
    # Pixel edges are used for a conservative angular interval.
    left_edge = float(left)
    right_edge = float(right + 1)
    beta_left = math.atan2(left_edge - camera.cx, camera.fx)
    beta_right = math.atan2(right_edge - camera.cx, camera.fx)
    return beta_left, beta_right


def pad_edge_sectors(sectors, half_fov, min_radius, clearance, base_pad=math.radians(8.0),
                     edge_tol=math.radians(0.5)):
    """Widen the inner side of partial sectors cut by the image side.

    Such a pillar may never be seen complete (rgb_dense_s105: a pillar
    passed at 0.6-1 m stayed truncated at the image side and never got a
    track).  Its inner edge is a tangent ray; for a pillar of radius ``r``
    and angular width ``theta``, a straight path at ``delta`` from that ray
    misses it by ``clearance`` (robot radius + margin) when
    ``sin(theta/2 + delta) >= (r + clearance) sin(theta/2) / r``.  The
    visible width is used for ``theta`` and ``min_radius`` for ``r``
    (a smaller pillar at the same width is nearer), so the pad grows as the
    pillar fills the view.  The planner adds ``base_pad`` itself; only the
    excess is added here.  ``min_radius <= 0`` disables padding.
    """
    if min_radius <= 0.0:
        return list(sectors)
    padded = []
    for lo, hi in sectors:
        at_left, at_right = lo <= -half_fov + edge_tol, hi >= half_fov - edge_tol
        if at_left == at_right:
            padded.append((lo, hi))
            continue
        half = 0.5 * (hi - lo)
        ratio = (min_radius + clearance) * math.sin(half) / min_radius
        delta = (math.asin(ratio) if ratio < 1.0 else 0.5 * math.pi) - half
        extra = max(0.0, delta - base_pad)
        padded.append((lo, hi + extra) if at_left else (lo - extra, hi))
    return padded


def truth_angle_observations(pose, pillars_xy, diameters, labels,
                             fov_deg=140.0, max_range=np.inf):
    """Return privileged exact ``[bearing, angular_width]`` observations.

    This debug-only helper uses pillar geometry to remove image projection and
    association errors.  It deliberately returns only the two angular
    quantities; distance is used internally for visibility and the bearing
    model, but is never included in the observation.
    """
    pose = np.asarray(pose, dtype=float)
    pillars_xy = np.asarray(pillars_xy, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    labels = list(labels)
    if pose.shape != (3,):
        raise ValueError("truth pose must have shape (3,)")
    if pillars_xy.ndim != 2 or pillars_xy.shape[1] != 2:
        raise ValueError("truth pillar positions must have shape (N, 2)")
    if diameters.shape != (len(pillars_xy),) or len(labels) != len(pillars_xy):
        raise ValueError("truth pillar positions, diameters, and labels must have equal length")
    if np.any(diameters <= 0.0):
        raise ValueError("truth pillar diameters must be positive")

    delta = pillars_xy - pose[:2]
    distance = np.linalg.norm(delta, axis=1)
    bearing = np.arctan2(delta[:, 1], delta[:, 0]) - pose[2]
    bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi
    angular_width = 2.0 * np.arcsin(
        np.clip(diameters / (2.0 * np.maximum(distance, 1e-12)), 0.0, 1.0))
    half_fov = math.radians(float(fov_deg)) / 2.0
    visible = ((distance > diameters / 2.0) &
               (distance <= float(max_range)) &
               (np.abs(bearing) + angular_width / 2.0 <= half_fov))
    return {
        str(label): np.array([bearing[index], angular_width[index]], dtype=float)
        for index, label in enumerate(labels) if visible[index]
    }


def truth_angle_sensor(pose, pillars_xy, diameters, labels, fov_deg=140.0,
                       max_range=np.inf, occlusion=False, noise_deg=(0.0, 0.0),
                       rng=None, rays=1001):
    """Privileged angle sensor with optional occlusion and Gaussian noise.

    Returns ``(detections, partial_sectors)`` in the same form as the
    segmentation adapter.  With ``occlusion=False`` and zero noise the
    detections equal :func:`truth_angle_observations`.  With occlusion, a
    pillar is a complete detection only if every ray inside its angular
    footprint hits it first; otherwise its visible rays become partial
    sectors, matching what an image adapter could actually see.  Distance is
    used only for ray casting and is never returned.
    """
    exact = truth_angle_observations(pose, pillars_xy, diameters, labels,
                                     360.0, np.inf)
    pose = np.asarray(pose, dtype=float)
    centers = np.asarray(pillars_xy, dtype=float)
    radii = np.asarray(diameters, dtype=float) / 2.0
    labels = [str(label) for label in labels]
    half_fov = math.radians(float(fov_deg)) / 2.0
    grid = np.linspace(-half_fov, half_fov, int(rays))
    step = grid[1] - grid[0]

    # Range of the first hit of every ray against every circle, (N, rays).
    q = centers - pose[:2]
    u = np.column_stack((np.cos(grid + pose[2]), np.sin(grid + pose[2])))
    b = q @ u.T
    disc = b * b - (np.sum(q * q, axis=1) - radii * radii)[:, None]
    near = b - np.sqrt(np.maximum(disc, 0.0))
    ranges = np.where((disc >= 0.0) & (near >= 0.0), near, np.inf)
    ranges[np.sum(q * q, axis=1) <= radii * radii] = 0.0
    nearest = np.argmin(ranges, axis=0)
    nearest[np.min(ranges, axis=0) > float(max_range)] = -1

    noise = np.radians(np.asarray(noise_deg, dtype=float))
    if rng is None and np.any(noise > 0.0):
        rng = np.random.default_rng()
    detections, partial = {}, []
    for index, label in enumerate(labels):
        if label not in exact:  # robot inside the pillar footprint
            continue
        beta, theta = exact[label]
        distance = float(np.linalg.norm(q[index]))
        footprint = (grid >= beta - theta / 2.0) & (grid <= beta + theta / 2.0)
        complete = (abs(beta) + theta / 2.0 <= half_fov and
                    distance <= float(max_range) and footprint.any() and
                    (not occlusion or np.all(nearest[footprint] == index)))
        if complete:
            z = np.array([beta, theta], dtype=float)
            if np.any(noise > 0.0):
                z = z + rng.normal(0.0, noise)
                z[0] = math.atan2(math.sin(z[0]), math.cos(z[0]))
                z[1] = max(z[1], 1e-4)
            detections[label] = z
            continue
        if occlusion:
            visible = nearest == index
        else:
            visible = np.isfinite(ranges[index]) & (ranges[index] <= float(max_range))
        edges = np.diff(np.r_[False, visible, False].astype(int))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            partial.append((float(grid[lo] - 0.5 * step), float(grid[hi - 1] + 0.5 * step)))
    return detections, partial


def extract_observations(labels, camera, config=None, allowed_labels=None,
                         ignore_labels=None):
    """Extract bearing-angle detections from a 16-bit segmentation label image.

    A label is accepted as a complete observation only when its mask is large
    enough and is not a scene-sized region such as the floor or a pool wall.
    Rejected labels that are explicitly allowed are returned as partial angle
    sectors.  With no allow-list, broad scene labels are ignored rather than
    turning into false obstacles.
    """
    config = config or SegmentationConfig()
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape != (camera.height, camera.width):
        raise ValueError("label image shape does not match camera model")
    if labels.dtype.kind not in "ui":
        raise ValueError("label image must contain integer object IDs")

    # An empty ROS parameter list means "all non-background labels".
    allowed = None if not allowed_labels else {int(v) for v in allowed_labels}
    ignored = set() if ignore_labels is None else {int(v) for v in ignore_labels}
    values, counts = np.unique(labels, return_counts=True)
    candidates = []
    for value, count in zip(values.tolist(), counts.tolist()):
        value = int(value)
        if value == 0 or value in ignored:
            continue
        if allowed is not None and value not in allowed:
            continue
        candidates.append((int(count), value))
    candidates.sort(reverse=True)
    candidates = candidates[:max(1, int(config.max_labels))]

    detections = {}
    partial = []
    accepted = []
    rejected = []
    for count, label in candidates:
        bbox = _label_bbox(labels, label)
        if bbox is None:
            continue
        left, right, top, bottom, area = bbox
        width = right - left + 1
        height = bottom - top + 1
        beta_left, beta_right = _sector_from_columns(left, right, camera)
        if (area < int(config.min_pixels) or
                height < int(config.min_height) or
                width > config.max_width_fraction * camera.width):
            rejected.append(label)
            if allowed is not None and area >= int(config.min_pixels):
                partial.append((beta_left, beta_right))
            continue

        beta = 0.5 * (beta_left + beta_right)
        theta = beta_right - beta_left
        if not (math.isfinite(beta) and math.isfinite(theta) and theta > 0.0):
            rejected.append(label)
            continue
        touches_side = left <= config.partial_margin_px or right >= camera.width - 1 - config.partial_margin_px
        if touches_side:
            rejected.append(label)
            if allowed is not None:
                partial.append((beta_left, beta_right))
            continue
        detections[str(label)] = np.array([beta, theta], dtype=float)
        accepted.append(label)

    return SegmentationResult(detections, partial, tuple(accepted), tuple(rejected))

