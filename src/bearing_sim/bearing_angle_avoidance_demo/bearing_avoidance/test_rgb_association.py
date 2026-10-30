#!/usr/bin/env python3
"""Checks for PF-prediction based RGB track association."""
import math

import numpy as np

from bearing_angle_pf import BearingAnglePF, observe
from rgb_association import RgbAssociator
from stonefish_observation import CameraModel, ColorComponent


class FakePF:
    """Posterior concentrated around one pillar ``[cx, cy, L]``."""

    def __init__(self, cx, cy, diameter, spread=0.02, n=400, seed=0):
        rng = np.random.default_rng(seed)
        self.particles = np.column_stack((rng.normal(cx, spread, n), rng.normal(cy, spread, n),
                                          np.abs(rng.normal(diameter, 0.01, n))))
        self.weights = np.full(n, 1.0 / n)


def component(beta_deg, theta_deg, complete=True):
    return ColorComponent(0, 0, 0, 0, 0, math.radians(beta_deg), math.radians(theta_deg), complete)


def main():
    camera = CameraModel.from_values(640, 480, 381.4, cx=320.0, cy=240.0)
    pillar_a = FakePF(-2.4, 0.0, 0.6)

    # Logged failure: A has left the 80 deg image on the right, pillar B
    # appears at +32 deg.  The old 25 deg bearing gate gave B A's ID.
    assoc = RgbAssociator()
    pose = np.array([-2.91, -0.46, math.radians(-24.0)])
    detections, partial = assoc.associate([component(32.0, 12.8)], pose, {"rgb_1": pillar_a}, camera)
    assert len(detections) == 1 and "rgb_1" not in detections and not partial

    # While A is in view, its detection keeps the ID despite a yaw change
    # that moves it 10 deg in the image since the last frame.
    pose = np.array([-4.4, 0.05, math.radians(-10.0)])
    beta = math.degrees(math.atan2(-0.05, 2.0)) + 10.0
    theta = math.degrees(2.0 * math.asin(0.3 / math.hypot(2.0, 0.05)))
    detections, partial = assoc.associate([component(beta, theta)], pose, {"rgb_1": pillar_a}, camera)
    assert list(detections) == ["rgb_1"] and not partial

    # A blob much wider than A's predicted width is A merged with another
    # pillar: it becomes a partial sector and does not update A's PF.
    detections, partial = assoc.associate([component(beta + 3.0, theta * 2.0)], pose,
                                          {"rgb_1": pillar_a}, camera)
    assert detections == {} and len(partial) == 1

    # Edge-touching components never create or update tracks.
    detections, partial = assoc.associate([component(beta, theta, complete=False)], pose,
                                          {"rgb_1": pillar_a}, camera)
    assert detections == {} and len(partial) == 1

    # rgb_fov80_1029_1316: a one-off, vague track (1-px blob at 37 deg,
    # posterior at ~(7.6, 1.6) +-2 m) sat inside the 10 deg gate of pillar C.
    # C's wide detection used to match it, fail the width check and become
    # a partial sector every frame, closing the planner's free headings.
    pose = np.array([0.38, -1.0, math.radians(10.0)])
    vague = FakePF(7.56, 1.63, 0.09, spread=2.0)
    beta_c = math.degrees(math.atan2(1.0, 2.02)) - 10.0
    theta_c = math.degrees(2.0 * math.asin(0.3 / math.hypot(2.02, 1.0)))
    assoc = RgbAssociator()
    detections, partial = assoc.associate([component(beta_c, theta_c)], pose, {"rgb_6": vague},
                                          camera, now=10.0)
    assert list(detections) == ["rgb_1"] and not partial, (detections, partial)

    # One-pixel blobs don't start tracks; blobs grazing the image side are
    # only sectors (rgb_8 was E seen at -27.3 deg with 24.9 deg width).
    detections, partial = assoc.associate([component(20.0, 0.2), component(-27.3, 24.9)],
                                          pose, {}, camera, now=10.0)
    assert detections == {} and len(partial) == 1

    # Tentative tracks expire; confirmed ones are kept.
    filters = {"rgb_1": FakePF(2.4, 0.0, 0.6), "rgb_9": vague}
    assoc.hits.update({"rgb_1": 3, "rgb_9": 1})
    assoc.last_seen.update({"rgb_1": 10.0, "rgb_9": 10.0})
    assert assoc.prune(filters, 11.0) == []
    assert assoc.prune(filters, 12.5) == ["rgb_9"] and list(filters) == ["rgb_1"]

    # rgb_fov80_1029_1528: pillar B got 2 hits at the image edge, left the
    # view and was pruned, so the robot passed it with B missing from the
    # map.  A tentative track outside the view must not expire.
    behind = {"rgb_5": FakePF(0.0, 0.0, 0.56)}
    assoc.hits["rgb_5"], assoc.last_seen["rgb_5"] = 2, 10.0
    side_pose = np.array([-2.3, -0.8, math.radians(-24.0)])   # B at ~43 deg
    for t in (11.0, 12.0, 13.0, 14.0):
        assoc.associate([], side_pose, behind, camera, now=t)
        assert assoc.prune(behind, t) == []
    assoc.associate([], np.array([-3.0, 0.0, 0.0]), behind, camera, now=14.5)  # back in view, missed
    assert assoc.prune(behind, 16.6) == ["rgb_5"]

    edge_track_s101(camera)
    print("PASS: RGB association uses PF-predicted bearing/width, ignores out-of-view tracks, "
          "tracks edge-truncated pillars")


def edge_track_s101(camera):
    """rgb_dense_s101_1029_1646: BearingPillar01 (3.476, -0.112, d 0.535) was
    cut by the right image side for 8 s, never got a track and was grazed
    after it left the view.  Its inner tangent alone must now build a
    confirmed track that maps the pillar at the collision pose."""
    truth = np.array([3.476, -0.112, 0.535])
    knots = np.array([[29.1, 1.48, -0.64, 38], [30.8, 1.79, -0.31, 43], [32.2, 2.09, -0.09, 30],
                      [34.0, 2.52, 0.10, 23], [35.6, 2.97, 0.20, 12], [36.7, 3.17, 0.27, -7],
                      [37.8, 3.31, 0.35, -18]])
    left_bound = -math.atan2(camera.cx, camera.fx)
    right_bound = math.atan2(camera.width - camera.cx, camera.fx)
    rng = np.random.default_rng(3)
    assoc, filters, truncated = RgbAssociator(), {}, 0
    for t in np.arange(29.1, 37.8, 0.1):
        x, y, yaw = (np.interp(t, knots[:, 0], knots[:, k]) for k in (1, 2, 3))
        yaw = math.radians(yaw)
        pose = np.array([x + 0.2 * math.cos(yaw), y + 0.2 * math.sin(yaw), yaw])
        (beta, theta), = observe(truth, pose)[0]
        # 0.25 deg per image edge: width noise ~ the PF theta sigma (0.35 deg).
        lo, hi = beta - theta / 2 + rng.normal(0, 0.0044), beta + theta / 2 + rng.normal(0, 0.0044)
        if hi <= left_bound or lo >= right_bound:
            continue
        cut = lo < left_bound or hi > right_bound
        truncated += cut
        lo, hi = max(lo, left_bound), min(hi, right_bound)
        blob = ColorComponent(0, 0, 0, 0, 0, (lo + hi) / 2, hi - lo, not cut)
        detections, partial = assoc.associate([blob], pose, filters, camera, now=t)
        for label, z in detections.items():
            if label not in filters:
                filters[label] = BearingAnglePF(n=1000, seed=len(filters), distance_prior=(0.55, 13.0),
                                                diameter_prior=(0.05, 0.75), mh_steps=3)
            filters[label].update(pose, z)
    assert truncated > 60, truncated
    assert list(filters) == ["rgb_1"] and assoc.hits["rgb_1"] >= truncated, (list(filters), assoc.hits)
    pf = filters["rgb_1"]
    mean = np.average(pf.particles, axis=0, weights=pf.weights)
    assert np.linalg.norm(mean[:2] - truth[:2]) < 0.15 and abs(mean[2] - truth[2]) < 0.15, mean
    # Robot centre at the logged minimum clearance (3.57, 0.32): the mapped
    # pillar plus the 0.20 m robot radius must cover it.
    gap = np.linalg.norm(pf.particles[:, :2] - [3.57, 0.32], axis=1) - pf.particles[:, 2] / 2 - 0.20
    assert pf.weights @ (gap < 0.0) > 0.8, pf.weights @ (gap < 0.0)


if __name__ == "__main__":
    main()
