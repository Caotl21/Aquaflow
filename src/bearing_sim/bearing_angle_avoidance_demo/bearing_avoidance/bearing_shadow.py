"""ROS-free diagnostics; never imported by the PF or planner.

Angles use the same horizontal/yaw-only convention as truth_angle. Predicted
occlusion is a sampled 2-D pillar-only model, not a claim about rendered pixels.
"""
import bisect
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from stonefish_observation import (
    SegmentationConfig, _label_bbox, _sector_from_columns,
    extract_observations, truth_angle_observations,
)


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PoseMatch:
    pose: object = None
    mode: str = "no_odometry"
    nearest_skew_s: object = None
    bracket_span_s: object = None
    before_stamp: object = None
    after_stamp: object = None


def match_odometry(stamp, samples, max_skew_s=0.05, max_gap_s=0.10):
    """Match sorted (stamp,x,y,yaw[,roll,pitch]) samples; never bridge big gaps."""
    if not math.isfinite(stamp) or stamp <= 0:
        return PoseMatch(mode="invalid_stamp")
    if not samples:
        return PoseMatch()
    times = [row[0] for row in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("odometry samples must be strictly increasing")
    index = bisect.bisect_left(times, stamp)
    if index < len(times) and times[index] == stamp:
        return PoseMatch(np.array(samples[index][1:], float), "exact", 0., 0., stamp, stamp)
    if 0 < index < len(times):
        before, after = samples[index - 1], samples[index]
        gap = after[0] - before[0]
        nearest = min(stamp - before[0], after[0] - stamp)
        if gap > max_gap_s:
            return PoseMatch(None, "odometry_gap", nearest, gap, before[0], after[0])
        a, b = np.array(before[1:], float), np.array(after[1:], float)
        delta = b - a
        delta[2:] = wrap(delta[2:])
        pose = a + (stamp - before[0]) / gap * delta
        pose[2:] = wrap(pose[2:])
        return PoseMatch(pose, "interpolated", nearest, gap, before[0], after[0])
    nearest = samples[0 if index == 0 else -1]
    skew = abs(nearest[0] - stamp)
    mode = "nearest" if skew <= max_skew_s else "unpaired"
    return PoseMatch(np.array(nearest[1:], float) if mode == "nearest" else None,
                     mode, skew, None, nearest[0], nearest[0])


def camera_pose(body_pose, offset):
    x, y, yaw = body_pose[:3]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * offset[0] - s * offset[1],
                     y + s * offset[0] + c * offset[1], yaw])


def _visible_fraction(index, centers, radii, pose, lo, hi, rays=129):
    # Midpoint samples include narrow targets without depending on image resolution.
    angles = lo + (np.arange(rays) + .5) / rays * (hi - lo) + pose[2]
    delta = centers - pose[:2]
    projection = delta @ np.array([np.cos(angles), np.sin(angles)])
    disc = projection ** 2 - (np.sum(delta ** 2, axis=1) - radii ** 2)[:, None]
    near = projection - np.sqrt(np.maximum(disc, 0.))
    ranges = np.where((disc >= 0) & (near >= 0), near, np.inf)
    ranges[np.sum(delta ** 2, axis=1) <= radii ** 2] = 0.
    return float(np.mean(np.argmin(ranges, axis=0) == index))


def compare_truth_and_segmentation(mask, camera, pose, centers, diameters, labels,
                                   fov_deg=140., max_range=13., config=None,
                                   allowed_labels=None, ignore_labels=None):
    """One diagnostic row per configured pillar; 'accepted' is adapter judgement.

    Image extraction uses the unchanged adapter and its configured allow-list.
    Missing/partial masks are not silently turned into zero-error observations.
    """
    config = config or SegmentationConfig()
    labels = [int(label) for label in labels]
    centers, diameters = np.asarray(centers, float), np.asarray(diameters, float)
    if len(set(labels)) != len(labels) or any(label <= 0 for label in labels):
        raise ValueError("truth labels must be unique nonzero IDs")
    # All geometric angles for diagnosis, including those outside the oracle FOV.
    all_angles = truth_angle_observations(pose, centers, diameters, labels, 360., np.inf)
    oracle = truth_angle_observations(pose, centers, diameters, labels, fov_deg, max_range)
    extracted = extract_observations(mask, camera, config, allowed_labels, ignore_labels)
    allowed = None if not allowed_labels else set(allowed_labels)
    ignored = set(ignore_labels or [0])
    left_fov, right_fov = _sector_from_columns(0, camera.width - 1, camera)
    distances = np.linalg.norm(centers - pose[:2], axis=1)
    rows = []
    for index, label in enumerate(labels):
        key = str(label)
        truth = all_angles.get(key)
        row = dict(label=label, truth_available=int(key in oracle),
                   truth_bearing_rad=None, truth_width_rad=None,
                   truth_visibility="inside_pillar", predicted_visible_fraction=None,
                   predicted_occlusion="not_evaluated", segmentation_pixels=0,
                   segmentation_status="missing", rejection_reason="no_pixels",
                   segmentation_bearing_rad=None, segmentation_width_rad=None,
                   bearing_error_rad=None, width_error_rad=None,
                   bbox_bearing_error_rad=None, bbox_width_error_rad=None,
                   left_px=None, right_px=None, top_px=None, bottom_px=None)
        if truth is not None:
            beta, theta = map(float, truth)
            row.update(truth_bearing_rad=beta, truth_width_rad=theta)
            lo, hi = beta - theta / 2, beta + theta / 2
            if distances[index] > max_range:
                visibility = "out_of_range"
            elif hi <= left_fov or lo >= right_fov:
                visibility = "outside_fov"
            elif lo < left_fov or hi > right_fov:
                visibility = "partial_fov"
            else:
                visibility = "full_fov"
            row["truth_visibility"] = visibility
            if visibility in ("full_fov", "partial_fov"):
                fraction = _visible_fraction(index, centers, diameters / 2, pose,
                                             max(lo, left_fov), min(hi, right_fov))
                row["predicted_visible_fraction"] = fraction
                row["predicted_occlusion"] = ("occluded" if fraction == 0. else
                    "unoccluded" if fraction == 1. else "partial_occlusion")
        bbox = _label_bbox(mask, label)
        if bbox is not None:
            left, right, top, bottom, area = bbox
            lo, hi = _sector_from_columns(left, right, camera)
            beta, theta = (lo + hi) / 2, hi - lo
            row.update(segmentation_pixels=area, left_px=left, right_px=right,
                       top_px=top, bottom_px=bottom,
                       segmentation_bearing_rad=beta, segmentation_width_rad=theta)
            reasons = []
            if label in ignored or (allowed is not None and label not in allowed):
                reasons.append("label_filtered")
            if area < config.min_pixels:
                reasons.append("low_pixels")
            if bottom - top + 1 < config.min_height:
                reasons.append("low_height")
            if right - left + 1 > config.max_width_fraction * camera.width:
                reasons.append("too_wide")
            if left <= config.partial_margin_px or right >= camera.width - 1 - config.partial_margin_px:
                reasons.append("image_edge")
            if key in extracted.detections:
                row.update(segmentation_status="accepted_complete", rejection_reason="")
            else:
                # Mirrors exactly whether extract_observations emits a partial sector.
                partial = (label in extracted.rejected_labels and allowed is not None
                           and area >= config.min_pixels)
                row.update(segmentation_status="partial" if partial else "rejected",
                           rejection_reason=";".join(reasons) or "candidate_limit")
            if truth is not None:
                row["bbox_bearing_error_rad"] = float(wrap(beta - truth[0]))
                row["bbox_width_error_rad"] = float(theta - truth[1])
                if key in extracted.detections:
                    row["bearing_error_rad"] = row["bbox_bearing_error_rad"]
                    row["width_error_rad"] = row["bbox_width_error_rad"]
        rows.append(row)
    return rows


@dataclass
class ErrorStats:
    count: int = 0
    total: float = 0.
    absolute: float = 0.
    squared: float = 0.
    maximum: float = 0.

    def add(self, radians):
        degrees = math.degrees(radians)
        self.count += 1
        self.total += degrees
        self.absolute += abs(degrees)
        self.squared += degrees ** 2
        self.maximum = max(self.maximum, abs(degrees))

    def result(self):
        return dict(count=self.count,
                    bias_deg=self.total / self.count if self.count else None,
                    mae_deg=self.absolute / self.count if self.count else None,
                    rmse_deg=math.sqrt(self.squared / self.count) if self.count else None,
                    max_abs_deg=self.maximum if self.count else None)


@dataclass
class ComparisonSummary:
    frame_status: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    errors: dict = field(default_factory=dict)
    per_label: dict = field(default_factory=dict)

    def add_frame(self, status, rows=()):
        self.frame_status[status] += 1
        for row in rows:
            key = str(row["label"])
            self.counts[row["segmentation_status"]] += 1
            if row["truth_available"]:
                self.counts["oracle_available"] += 1
                self.counts["oracle_available_" + row["segmentation_status"]] += 1
                self.counts["oracle_available_" + row["predicted_occlusion"] +
                            "_" + row["segmentation_status"]] += 1
            self.counts["predicted_" + row["predicted_occlusion"]] += 1
            metrics = self.per_label.setdefault(key, dict(counts=Counter(), errors={}))
            metrics["counts"][row["segmentation_status"]] += 1
            metrics["counts"]["predicted_" + row["predicted_occlusion"]] += 1
            for axis in ("bearing", "width"):
                error = row[axis + "_error_rad"]
                if error is None:
                    continue
                for group in (["accepted"] + (["unoccluded_full_fov"] if
                        row["predicted_occlusion"] == "unoccluded" and
                        row["truth_visibility"] == "full_fov" else [])):
                    name = group + "_" + axis
                    self.errors.setdefault(name, ErrorStats()).add(error)
                    metrics["errors"].setdefault(name, ErrorStats()).add(error)

    def result(self):
        return dict(frame_status=dict(self.frame_status), counts=dict(self.counts),
                    errors={k: v.result() for k, v in self.errors.items()},
                    per_label={k: dict(counts=dict(v["counts"]),
                        errors={n: s.result() for n, s in v["errors"].items()})
                        for k, v in self.per_label.items()},
                    caveats=["truth_available ignores occlusion, matching the driving oracle",
                             "accepted_complete is the adapter decision, not proof of full contour",
                             "occlusion uses 129 horizontal rays per pillar; no walls or 3-D attitude",
                             "image header time is not independently verified exposure time"])
