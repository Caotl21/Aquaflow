#!/usr/bin/env python3
"""Associate RGB pillar components with existing bearing-angle PF tracks.

Each track's expected ``[bearing, angular_width]`` is predicted by projecting
its own posterior particles from the current camera pose, so ego-motion and
yaw are compensated and a pillar that has left the image cannot claim a new
pillar that appears at a similar pixel position.  Only the node's own
posterior is used: no depth, labels, or scene map.

A component cut by exactly one image side still measures its inner tangent
ray; it updates (or starts) a track with the truncated observation
``[edge, bound, side]`` of :func:`bearing_angle_pf.log_likelihood`, so a near
pillar that never fits in the image is still mapped and remembered.
"""
import math

import numpy as np

from bearing_angle_pf import observe, wrap


def predict_track(pf, camera_pose):
    """Return ``(beta, beta_std, theta, theta_std)`` or ``None``."""
    if getattr(pf, "particles", None) is None:
        return None
    z, valid = observe(pf.particles, camera_pose)
    weights = np.nan_to_num(np.asarray(pf.weights, dtype=float)) * valid
    if weights.sum() <= 0.0:
        return None
    weights = weights / weights.sum()
    beta = math.atan2(float(weights @ np.sin(z[:, 0])), float(weights @ np.cos(z[:, 0])))
    beta_std = math.sqrt(float(weights @ wrap(z[:, 0] - beta) ** 2))
    theta = float(weights @ z[:, 1])
    theta_std = math.sqrt(float(weights @ (z[:, 1] - theta) ** 2))
    return beta, beta_std, theta, theta_std


def predict_edge(pf, camera_pose, side):
    """Return ``(edge, edge_std, far)`` for a silhouette continuing toward
    ``side``: the inner tangent bearing and the mean far tangent."""
    if getattr(pf, "particles", None) is None:
        return None
    z, valid = observe(pf.particles, camera_pose)
    weights = np.nan_to_num(np.asarray(pf.weights, dtype=float)) * valid
    if weights.sum() <= 0.0:
        return None
    weights = weights / weights.sum()
    near = z[:, 0] - side * z[:, 1] / 2.0
    far = z[:, 0] + side * z[:, 1] / 2.0
    edge = math.atan2(float(weights @ np.sin(near)), float(weights @ np.cos(near)))
    edge_std = math.sqrt(float(weights @ wrap(near - edge) ** 2))
    far_mean = math.atan2(float(weights @ np.sin(far)), float(weights @ np.cos(far)))
    return edge, edge_std, far_mean


class RgbAssociator:
    def __init__(self, gate_min_deg=3.0, gate_max_deg=10.0, gate_sigmas=3.0,
                 width_tolerance=0.4, width_slack_deg=1.0, confident_deg=2.0,
                 min_new_width_deg=0.6, new_edge_margin_deg=1.0,
                 min_new_edge_width_deg=2.0, max_new_edge_width_deg=60.0,
                 confirm_hits=3, tentative_timeout_s=2.0, prefix="rgb_"):
        self.gate_min = math.radians(gate_min_deg)
        self.gate_max = math.radians(gate_max_deg)
        self.gate_sigmas = float(gate_sigmas)
        self.width_tolerance = float(width_tolerance)
        self.width_slack = math.radians(width_slack_deg)
        # Only a track this sure of its bearing may turn a too-wide blob into
        # a "merged" partial sector; a vague track just doesn't match it.
        self.confident = math.radians(confident_deg)
        # New tracks need a resolvable width and must not graze the image side.
        self.min_new_width = math.radians(min_new_width_deg)
        self.new_edge_margin = math.radians(new_edge_margin_deg)
        # Edge-truncated components: a sliver hardly constrains the pillar,
        # and a very wide one is incompatible with the distance prior.
        self.min_new_edge_width = math.radians(min_new_edge_width_deg)
        self.max_new_edge_width = math.radians(max_new_edge_width_deg)
        self.confirm_hits = int(confirm_hits)
        self.tentative_timeout = float(tentative_timeout_s)
        self.prefix = prefix
        self.next_id = 1
        self.hits = {}
        self.last_seen = {}

    def prune(self, filters, now, first_seen=None):
        """Drop tentative tracks (< ``confirm_hits`` detections) not seen for
        ``tentative_timeout_s``; one-off false detections otherwise stay in
        the planner map forever with an unconverged, inflated posterior."""
        removed = []
        for label in list(filters):
            if label not in self.hits:
                continue
            if (self.hits[label] < self.confirm_hits and
                    now - self.last_seen[label] > self.tentative_timeout):
                del filters[label]
                if first_seen is not None:
                    first_seen.pop(label, None)
                del self.hits[label], self.last_seen[label]
                removed.append(label)
        return removed

    def associate(self, components, camera_pose, filters, camera, now=0.0):
        """Return ``(detections, partial_sectors)`` for the PF tracker.

        ``filters`` maps track label to a PF with ``particles``/``weights``;
        ``camera`` is the :class:`CameraModel` that produced ``components``.
        Complete components whose width contradicts a confident track
        (merged with or occluded by another pillar), components cut at a
        split, and components cut by both image sides become partial sectors
        and do not update any track.  A component cut by one image side is
        matched on its inner tangent and emitted as ``[edge, bound, side]``;
        its sector is also kept as a partial sector for the planner.
        Unmatched components only start a track if they are at least
        ``min_new_width_deg`` wide and clear of the image side (complete), or
        ``min_new_edge_width_deg``..``max_new_edge_width_deg`` wide (cut by
        one side).  ``now`` (s) feeds :meth:`prune`.
        """
        detections, partial = {}, []
        left_bound = -math.atan2(camera.cx, camera.fx)
        right_bound = math.atan2(camera.width - camera.cx, camera.fx)
        half_fov = min(-left_bound, right_bound)
        predictions, tracks = {}, {}
        for label, pf in filters.items():
            if not str(label).startswith(self.prefix):
                continue
            tracks[label] = pf
            pred = predict_track(pf, camera_pose)
            # A track predicted outside the image cannot produce a complete
            # detection in this frame.
            if pred is not None and abs(pred[0]) + pred[2] / 2.0 < half_fov:
                predictions[label] = pred
            elif label in self.last_seen:
                # Out of view is not a miss: the tentative timeout only runs
                # while the track could have been detected.  A track cut by
                # the image side refreshes here too; a missing sliver at the
                # edge is too unreliable to count as a miss.
                self.last_seen[label] = now
        complete, truncated = [], []
        for component in components:
            sector = (component.beta - component.theta / 2.0, component.beta + component.theta / 2.0)
            if component.complete:
                complete.append((component, sector))
                continue
            partial.append(sector)
            at_left = sector[0] <= left_bound + self.new_edge_margin
            at_right = sector[1] >= right_bound - self.new_edge_margin
            if component.occluded or at_left == at_right:
                continue
            # The pillar continues past the image side it touches.
            side = 1.0 if at_right else -1.0
            edge, bound = (sector[0], right_bound) if at_right else (sector[1], left_bound)
            truncated.append((component, edge, bound, side))

        pairs = []
        for index, (component, _sector) in enumerate(complete):
            for label, (beta, beta_std, theta, theta_std) in predictions.items():
                error = abs(wrap(component.beta - beta))
                gate = min(self.gate_max, self.gate_min + self.gate_sigmas * beta_std)
                if error > gate:
                    continue
                slack = (max(self.width_slack, self.width_tolerance * theta) +
                         self.gate_sigmas * theta_std)
                if abs(component.theta - theta) <= slack:
                    kind = "match"
                elif component.theta > theta and beta_std <= self.confident:
                    kind = "merged"
                else:
                    # Bearing fits but the width contradicts this track
                    # (e.g. a vague one-off track at 10 m "claiming" a near
                    # pillar).  Leave the detection free for other tracks.
                    continue
                pairs.append((error / gate, index, label, kind))
        pairs.sort()
        matched, used = {}, set()
        for _score, index, label, kind in pairs:
            if index in matched or label in used:
                continue
            matched[index] = (label, kind)
            used.add(label)

        for index, (component, sector) in enumerate(complete):
            label, kind = matched.get(index, (None, None))
            if kind == "merged":
                partial.append(sector)
                continue
            if label is None:
                if component.theta < self.min_new_width:
                    continue
                if abs(component.beta) + component.theta / 2.0 > half_fov - self.new_edge_margin:
                    partial.append(sector)
                    continue
                label = "%s%d" % (self.prefix, self.next_id)
                while label in filters or label in detections:
                    self.next_id += 1
                    label = "%s%d" % (self.prefix, self.next_id)
                self.next_id += 1
            detections[label] = np.array([component.beta, component.theta], dtype=float)
            self.hits[label] = self.hits.get(label, 0) + 1
            self.last_seen[label] = now

        pairs = []
        for index, (component, edge, bound, side) in enumerate(truncated):
            for label, pf in tracks.items():
                if label in used:
                    continue
                pred = predict_edge(pf, camera_pose, side)
                if pred is None:
                    continue
                pred_edge, edge_std, far = pred
                error = abs(wrap(edge - pred_edge))
                gate = min(self.gate_max, self.gate_min + self.gate_sigmas * edge_std)
                # The track must also reach the image side.
                if error > gate or side * wrap(far - bound) < -gate:
                    continue
                pairs.append((error / gate, index, label))
        pairs.sort()
        matched = {}
        for _score, index, label in pairs:
            if index in matched or label in used:
                continue
            matched[index] = label
            used.add(label)
        for index, (component, edge, bound, side) in enumerate(truncated):
            label = matched.get(index)
            if label is None:
                if not self.min_new_edge_width <= component.theta <= self.max_new_edge_width:
                    continue
                label = "%s%d" % (self.prefix, self.next_id)
                while label in filters or label in detections:
                    self.next_id += 1
                    label = "%s%d" % (self.prefix, self.next_id)
                self.next_id += 1
            detections[label] = np.array([edge, bound, side], dtype=float)
            self.hits[label] = self.hits.get(label, 0) + 1
            self.last_seen[label] = now
        return detections, partial

